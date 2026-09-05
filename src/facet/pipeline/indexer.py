"""Indexing orchestration: directory -> detect -> quality -> align -> encode -> store.

Deliberately stops at the embedding. Attribute prediction is a separate pass
(`scripts/predict_attributes.py`) because docs/RESEARCH.md 15.1 splits the pipeline into an
expensive cached half and a cheap replaceable half. Keeping them in one command would make it
natural to re-detect and re-encode every time a head changes, which is exactly the cost the
architecture exists to avoid.

Everything the lazy passes need is persisted: bbox, keypoints, quality signals and the
embedding row. MiVOLO re-crops from the original image at predict time rather than the
indexer storing a second crop, because E4 made age/gender lazy - storing a wide crop for every
detected face would cost disk for work that mostly never runs.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..models.insightface_backend import ArcFaceEmbedder, InsightFaceDetector, align_to_template
from ..models.quality import composite_v2, per_crop_signals
from ..utils.hashing import hash_obj
from .db import Index
from .discovery import content_hash, load_image, plan_scan
from .store import FeatureStore


@dataclass
class IndexConfig:
    """Every value here is part of the cache key, so changing one invalidates the right work."""

    pack: str = "buffalo_l"
    det_size: str | int = "auto"        # E8: adaptive, never upscales small images
    pad_frac: str | float = "auto"      # E8: unpadded first, padded retry
    crop_size: int = 112
    crop_margin: float = 0.25           # E5: selected on transfer, not in-benchmark accuracy
    align: str = "template"
    max_faces_per_image: int = 20
    min_face_px: float = 24.0
    batch_size: int = 64
    use_gpu: bool = True
    clip: bool = True                   # exp001/E5: ArcFace+CLIP fusion is the production rep

    @property
    def crop_version(self) -> str:
        return f"crop:{self.align}:m{self.crop_margin:g}:s{self.crop_size}"

    def hash(self) -> str:
        return hash_obj(self.__dict__)


@dataclass
class IndexStats:
    seen: int = 0
    indexed: int = 0
    skipped: int = 0
    failed: int = 0
    no_faces: int = 0
    faces: int = 0
    seconds: float = 0.0
    errors: list = field(default_factory=list)


class Indexer:
    def __init__(self, index_path, features_dir, config: IndexConfig | None = None):
        self.cfg = config or IndexConfig()
        self.index = Index(index_path)
        self.detector = InsightFaceDetector(
            pack=self.cfg.pack, det_size=self.cfg.det_size,
            pad_frac=self.cfg.pad_frac, use_gpu=self.cfg.use_gpu,
        )
        self.embedder = ArcFaceEmbedder(pack=self.cfg.pack, use_gpu=self.cfg.use_gpu)
        self.clip = None
        if self.cfg.clip:
            from ..models.clip_backend import ClipEmbedder
            self.clip = ClipEmbedder(use_gpu=self.cfg.use_gpu)

        self.encoder_version = self.embedder.version + (
            f"+{self.clip.version}" if self.clip else ""
        )
        self.dim = self.embedder.dim + (self.clip.dim if self.clip else 0)
        self.store = FeatureStore(features_dir, self.encoder_version,
                                  self.cfg.crop_version, dim=self.dim)

    # ------------------------------------------------------------------ helpers

    def _detect_and_crop(self, img):
        """Returns (list of face dicts, list of aligned crops)."""
        dets = self.detector.detect(img)[: self.cfg.max_faces_per_image]
        faces, crops = [], []
        for i, d in enumerate(dets):
            if d.keypoints is None or not np.isfinite(d.keypoints).all():
                continue
            x1, y1, x2, y2 = d.bbox
            px = max(x2 - x1, y2 - y1)
            if px < self.cfg.min_face_px:
                continue                      # too small to be a useful result (E8)
            crop = align_to_template(img, d.keypoints, size=self.cfg.crop_size,
                                     margin=self.cfg.crop_margin)
            sig = per_crop_signals(crop)
            sig["det_score"] = float(d.score)
            sig["face_pixels"] = float(px)
            faces.append({
                "face_idx": i, "x1": float(x1), "y1": float(y1),
                "x2": float(x2), "y2": float(y2),
                "det_score": float(d.score), "face_px": float(px),
                "kps": d.keypoints.astype(np.float32).tobytes(),
                "_signals": sig,
            })
            crops.append(crop)
        return faces, crops

    def _encode(self, crops: np.ndarray):
        emb, norms = self.embedder.encode_with_norm(crops, batch_size=self.cfg.batch_size)
        if self.clip is not None:
            ce = self.clip.encode(crops, batch_size=self.cfg.batch_size)
            emb = np.hstack([emb, ce])
        return emb.astype(np.float32), norms

    # -------------------------------------------------------------------- main

    def index_directories(self, roots, force: bool = False, limit: int = 0,
                          progress_every: int = 200) -> IndexStats:
        cfg = self.cfg
        plan = plan_scan(roots, self.index.image_fingerprints(), force=force)
        todo = plan.to_process
        if limit:
            todo = todo[:limit]
        print(f"scan: {plan.summary()}  -> processing {len(todo)}")

        run_id = self.index.start_run(str(roots), cfg.hash(), cfg.__dict__)
        st = IndexStats(seen=len(plan.to_process) + len(plan.unchanged),
                        skipped=len(plan.unchanged))
        t0 = time.time()

        buf_faces, buf_crops, buf_owner = [], [], []

        def flush():
            """Encode a batch of crops and write image+face rows in one transaction."""
            if not buf_crops:
                return
            emb, norms = self._encode(np.stack(buf_crops))
            with self.index.tx():
                rows = self.store.append(emb)
                for k, (image_kw, face) in enumerate(zip(buf_owner, buf_faces)):
                    image_id = image_kw["_image_id"]
                    sig = dict(face.pop("_signals"))
                    sig["embedding_norm"] = float(norms[k])
                    q = float(composite_v2({kk: np.array([vv]) for kk, vv in sig.items()})[0])
                    self.index.insert_face(
                        image_id=image_id, **face, quality=q,
                        quality_json=_json(sig), feature_row=int(rows[k]),
                        encoder_version=self.encoder_version,
                        crop_version=cfg.crop_version,
                    )
            buf_faces.clear(); buf_crops.clear(); buf_owner.clear()

        for n, cand in enumerate(todo, 1):
            img, err = load_image(cand.path)
            if img is None:
                self.index.upsert_image(
                    path=cand.path, size_bytes=cand.size_bytes, mtime=cand.mtime,
                    status="corrupt", error=err, n_faces=0,
                    detector_version=self.detector.version,
                )
                st.failed += 1
                if len(st.errors) < 50:
                    st.errors.append({"path": cand.path, "error": err})
            else:
                faces, crops = self._detect_and_crop(img)
                image_id = self.index.upsert_image(
                    path=cand.path, content_hash=content_hash(cand.path),
                    size_bytes=cand.size_bytes, mtime=cand.mtime,
                    width=img.shape[1], height=img.shape[0],
                    status="ok" if faces else "no_faces", error=None,
                    n_faces=len(faces), detector_version=self.detector.version,
                )
                self.index.clear_faces(image_id)      # re-index replaces, never duplicates
                st.indexed += 1
                if not faces:
                    st.no_faces += 1
                for f, c in zip(faces, crops):
                    buf_faces.append(f); buf_crops.append(c)
                    buf_owner.append({"_image_id": image_id})
                    st.faces += 1
                if len(buf_crops) >= cfg.batch_size:
                    flush()

            if n % progress_every == 0:
                el = time.time() - t0
                print(f"  {n}/{len(todo)}  {n/el:.1f} img/s  faces={st.faces} "
                      f"failed={st.failed}", flush=True)

        flush()
        st.seconds = time.time() - t0
        self.index.finish_run(run_id, n_seen=st.seen, n_indexed=st.indexed,
                              n_skipped=st.skipped, n_failed=st.failed, n_faces=st.faces)
        return st

    def close(self):
        self.index.close()


def _json(d):
    import json
    return json.dumps({k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                       for k, v in d.items()})
