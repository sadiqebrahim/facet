"""InsightFace adapters (SCRFD detection, ArcFace embedding, 106-pt landmarks).

LICENSING: InsightFace *code* is MIT, but the pretrained *weights* used here
(buffalo_l, antelopev2) are NON-COMMERCIAL RESEARCH ONLY. See docs/LICENSING.md.
Every class here sets `commercial_use = False` so a --commercial-safe run can refuse it.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..utils.gpu import assert_gpu_session, ensure_cuda_libs
from .base import Detection

DEFAULT_ROOT = Path.home() / ".insightface" / "models"

# Canonical 5-point template for 112x112 ArcFace alignment (ArcFace / InsightFace standard).
ARCFACE_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def _session(path: Path, use_gpu: bool = True, strict_gpu: bool = True):
    if use_gpu:
        ensure_cuda_libs()
    import onnxruntime as ort

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
    )
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    sess = ort.InferenceSession(str(path), sess_options=opts, providers=providers)
    if use_gpu:
        assert_gpu_session(sess, strict=strict_gpu)
    return sess


def crop_bbox(
    image: np.ndarray, bbox: tuple[float, float, float, float], size: int = 112,
    margin: float = 0.0
) -> np.ndarray:
    """Plain square bbox crop with margin - NO similarity transform.

    The control condition for experiment E5: it isolates how much the 5-point alignment
    is worth, separately from how much the crop framing is worth. Without alignment the
    face keeps its in-plane rotation and its position within the frame varies.
    """
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    half = max(x2 - x1, y2 - y1) * (1.0 + margin) / 2.0
    h, w = image.shape[:2]
    xa, xb = int(round(cx - half)), int(round(cx + half))
    ya, yb = int(round(cy - half)), int(round(cy + half))
    pad = max(0, -xa, -ya, xb - w, yb - h)
    if pad:
        image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        xa, xb, ya, yb = xa + pad, xb + pad, ya + pad, yb + pad
    patch = image[ya:yb, xa:xb]
    if patch.size == 0:
        return np.zeros((size, size, 3), dtype=image.dtype)
    return cv2.resize(patch, (size, size))


def align_to_template(
    image: np.ndarray, keypoints: np.ndarray, size: int = 112, margin: float = 0.0
) -> np.ndarray:
    """Similarity-transform a face to the canonical ArcFace 112x112 frame.

    `margin` (0.0 = standard ArcFace crop) zooms out around the template centre, which is
    the knob experiment E5 sweeps. Crop protocol is part of the feature cache key, so
    changing this invalidates cached features by design.
    """
    dst = ARCFACE_TEMPLATE.copy()
    if margin:
        centre = dst.mean(axis=0)
        dst = centre + (dst - centre) / (1.0 + margin)
    dst = dst * (size / 112.0)
    M, _ = cv2.estimateAffinePartial2D(keypoints.astype(np.float32), dst, method=cv2.LMEDS)
    if M is None:  # degenerate keypoints - fall back to a centre crop
        h, w = image.shape[:2]
        s = min(h, w)
        y0, x0 = (h - s) // 2, (w - s) // 2
        return cv2.resize(image[y0 : y0 + s, x0 : x0 + s], (size, size))
    return cv2.warpAffine(image, M, (size, size), borderValue=0.0)


class InsightFaceDetector:
    """SCRFD detector via the insightface package (handles NMS/anchor decoding for us)."""

    name = "scrfd_det_10g"
    commercial_use = False
    license = "research-only (weights); MIT (code)"

    def __init__(
        self,
        pack: str = "buffalo_l",
        det_size: int | str = "auto",
        use_gpu: bool = True,
        pad_frac: float | str = "auto",
    ):
        """SCRFD detector with scale- and padding-adaptive behaviour.

        Two independent adaptations, both forced by measurement rather than taste:

        **det_size.** InsightFace resizes every image so its longest side equals det_size,
        which means a small image gets UPSCALED. E8 found this silently destroys the
        portrait case: a 350x350 portrait at det_size 1024 is blown up ~3x, the face
        exceeds the detector's scale range, and recall on SCUT-FBP5500 crops falls from
        1.00 (det 640) to 0.36 (det 1024) - to 0.00 unpadded. Meanwhile scenes clearly
        prefer 1024 (recall_all 0.641 vs 0.511 at 640) because their faces are tiny.
        `det_size="auto"` keeps a detector at each scale and routes per image so small
        images are never upscaled hard.

        **pad_frac.** E5 found SCRFD's recall on frame-filling portraits collapses to 46%
        without padding. E8 found padding is not free on scenes - at det 1024 it helps
        only the >=128px bucket (+0.037) and costs -0.156 on faces under 16px.
        `pad_frac="auto"` runs unpadded first and retries padded only when nothing was
        found or the top face fills the frame, which recovers portraits at near-zero cost
        on scenes (measured: recall_all 0.6588 auto vs 0.6580 unpadded vs 0.5725 padded).
        """
        if use_gpu:
            ensure_cuda_libs()
        from insightface.app import FaceAnalysis

        self.pad_frac = pad_frac
        self.auto_pad = pad_frac == "auto"
        self.retry_pad = 0.25
        self.auto_scale = det_size == "auto"
        #: image longest-side threshold -> det_size to use for it
        self.scales = (640, 1024) if self.auto_scale else (int(det_size),)
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu
            else ["CPUExecutionProvider"]
        )
        self.apps = {}
        for s in self.scales:
            app = FaceAnalysis(name=pack, allowed_modules=["detection"], providers=providers)
            app.prepare(ctx_id=0 if use_gpu else -1, det_size=(s, s))
            self.apps[s] = app
        self.app = self.apps[self.scales[-1]]
        tag_p = "auto" if self.auto_pad else f"{pad_frac:g}"
        tag_s = "auto" if self.auto_scale else str(det_size)
        self.version = f"insightface:{pack}:pad{tag_p}:det{tag_s}"

    def _app_for(self, image: np.ndarray):
        """Pick a det_size that does not upscale a small image beyond the detector's range."""
        if not self.auto_scale:
            return self.apps[self.scales[0]]
        longest = max(image.shape[:2])
        return self.apps[640] if longest <= 700 else self.apps[1024]

    def _detect_at(self, image: np.ndarray, pad_frac: float) -> list[Detection]:
        pad = int(round(min(image.shape[:2]) * pad_frac)) if pad_frac > 0 else 0
        padded = (
            cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
            if pad else image
        )
        out = []
        for f in self._app_for(image).get(padded):
            x1, y1, x2, y2 = [float(v) for v in f.bbox]
            kps = np.asarray(f.kps, dtype=np.float32).copy()
            if pad:  # map back to original image coordinates
                x1, y1, x2, y2 = x1 - pad, y1 - pad, x2 - pad, y2 - pad
                kps -= pad
            out.append(
                Detection(bbox=(x1, y1, x2, y2), score=float(f.det_score), keypoints=kps)
            )
        return sorted(out, key=lambda d: -d.score)

    def _fills_frame(self, dets: list[Detection], image: np.ndarray, frac: float = 0.6) -> bool:
        if not dets:
            return False
        h, w = image.shape[:2]
        x1, y1, x2, y2 = dets[0].bbox
        return max(x2 - x1, y2 - y1) >= frac * min(h, w)

    def detect(self, image: np.ndarray) -> list[Detection]:
        """`image` is BGR uint8, as OpenCV reads it. Coordinates are in ORIGINAL space."""
        if not self.auto_pad:
            return self._detect_at(image, float(self.pad_frac))

        # Unpadded first: best for scenes, which is the common case when scanning a
        # directory of photographs.
        dets = self._detect_at(image, 0.0)
        # Retry padded only in the two situations where padding is known to help:
        # nothing was found at all, or the top face fills the frame (the portrait case
        # where SCRFD's recall collapses - E5 measured 46% unpadded on such crops).
        if not dets or self._fills_frame(dets, image):
            retry = self._detect_at(image, self.retry_pad)
            if len(retry) > len(dets) or (retry and not dets):
                return retry
        return dets


class ArcFaceEmbedder:
    """ArcFace embedding from a raw ONNX session.

    Deliberately does NOT go through insightface's wrapper, so the preprocessing is
    explicit and versioned - the feature store depends on it being reproducible.
    """

    commercial_use = False
    license = "research-only (weights); MIT (code)"

    #: pack -> (onnx filename, embedding dim)
    MODELS = {
        "buffalo_l": ("w600k_r50.onnx", 512),
        "antelopev2": ("glintr100.onnx", 512),
    }

    def __init__(self, pack: str = "buffalo_l", root: Path = DEFAULT_ROOT, use_gpu: bool = True):
        if pack not in self.MODELS:
            raise ValueError(f"unknown pack {pack!r}; expected one of {list(self.MODELS)}")
        fname, dim = self.MODELS[pack]
        path = Path(root) / pack / fname
        if not path.exists():
            raise FileNotFoundError(f"missing ArcFace weights: {path}")
        self.name = f"arcface_{pack}"
        self.version = f"insightface:{pack}:{fname}"
        self.dim = dim
        self.sess = _session(path, use_gpu)
        self.input_name = self.sess.get_inputs()[0].name
        self.input_size = int(self.sess.get_inputs()[0].shape[-1])

    def encode(self, crops: np.ndarray, batch_size: int = 128) -> np.ndarray:
        """`crops`: (N, H, W, 3) BGR uint8, already aligned. Returns (N, dim) float32.

        Embeddings are L2-normalised: for a linear head the norm is a nuisance scale, and
        normalising makes ridge regularisation behave consistently across backbones.
        (The un-normalised norm is itself a useful quality signal - see RESEARCH.md 2.4 -
        so it is returned separately by `encode_with_norm`.)
        """
        emb, _ = self.encode_with_norm(crops, batch_size)
        return emb

    def encode_with_norm(
        self, crops: np.ndarray, batch_size: int = 128
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns (L2-normalised embeddings, pre-normalisation L2 norms).

        The norm is the MagFace-style free quality proxy from RESEARCH.md section 2.4.
        """
        outs, norms = [], []
        for i in range(0, len(crops), batch_size):
            batch = crops[i : i + batch_size]
            # ArcFace preprocessing: BGR -> RGB, (x - 127.5) / 127.5, NCHW
            x = batch[..., ::-1].astype(np.float32)
            x = (x - 127.5) / 127.5
            x = np.transpose(x, (0, 3, 1, 2)).copy()
            e = self.sess.run(None, {self.input_name: x})[0]
            n = np.linalg.norm(e, axis=1, keepdims=True)
            outs.append(e / np.clip(n, 1e-9, None))
            norms.append(n.ravel())
        return (
            np.concatenate(outs).astype(np.float32),
            np.concatenate(norms).astype(np.float32),
        )


class GenderAgePredictor:
    """InsightFace `genderage` model - the cheap off-the-shelf attribute baseline.

    Included so E11 can audit demographic performance end to end without first acquiring
    MiVOLO. Its accuracy is well below MiVOLO's (docs/RESEARCH.md 2.5); the point here is
    to measure how error VARIES ACROSS GROUPS, which is a property of the model family
    rather than of any one checkpoint.

    LICENSING: research-only weights, like the rest of the InsightFace pack.
    """

    name = "insightface_genderage"
    commercial_use = False
    license = "research-only (weights); MIT (code)"

    def __init__(self, pack: str = "buffalo_l", root: Path = DEFAULT_ROOT, use_gpu: bool = True):
        path = Path(root) / pack / "genderage.onnx"
        if not path.exists():
            raise FileNotFoundError(f"missing genderage weights: {path}")
        self.version = f"insightface:{pack}:genderage.onnx"
        self.sess = _session(path, use_gpu)
        self.input_name = self.sess.get_inputs()[0].name
        self.input_size = int(self.sess.get_inputs()[0].shape[-1])

    def predict(self, crops: np.ndarray, batch_size: int = 128):
        """`crops`: (N, H, W, 3) BGR uint8, aligned. Returns (gender_prob_female, age).

        The model emits 3 values: two gender logits followed by age/100.
        """
        genders, ages = [], []
        for i in range(0, len(crops), batch_size):
            batch = crops[i : i + batch_size]
            if batch.shape[1] != self.input_size:
                batch = np.stack(
                    [cv2.resize(c, (self.input_size, self.input_size)) for c in batch]
                )
            x = batch[..., ::-1].astype(np.float32)
            x = np.transpose(x, (0, 3, 1, 2)).copy()
            out = self.sess.run(None, {self.input_name: x})[0]
            logits = out[:, :2]
            e = np.exp(logits - logits.max(axis=1, keepdims=True))
            prob = e / e.sum(axis=1, keepdims=True)
            genders.append(prob[:, 1])          # index 1 = male in InsightFace's ordering
            ages.append(out[:, 2] * 100.0)
        return np.concatenate(genders), np.concatenate(ages)
