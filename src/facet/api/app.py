"""FastAPI application: search, media, indexing progress, feedback, export.

Thin by design. The API translates HTTP into `QuerySpec` and back, and adds nothing to the
ranking - `docs/QUERY.md` explains why the scoring function stays transparent and explainable
rather than moving into a service layer.

Two responsibilities beyond plumbing:

* **Every payload carries provenance.** Attractiveness values leave this process with the
  SCUT-FBP5500 source string and the measured demographic skew attached, so a client cannot
  render a bare number without also having the caveats (RESEARCH.md 11.4, 13.5).
* **Nothing leaves the machine.** Images are served from local disk to a local UI; there is no
  outbound call anywhere in this file. That is LICENSING.md section 4's local-only commitment,
  which is easy to honour now and expensive to retrofit.
"""
from __future__ import annotations

import io
import json
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, Response
from pydantic import BaseModel

from ..pipeline.db import Index
from ..query.engine import BEAUTY_SOURCE_NOTE, SearchEngine
from ..query.spec import QuerySpec

WEB_DIR = Path(__file__).resolve().parents[3] / "web"

DISCLAIMER = {
    "source": BEAUTY_SOURCE_NOTE,
    "estimates_not_measurements": (
        "All attributes here are model estimates. Attractiveness in particular is a "
        "prediction of how one narrow group of raters would have scored a face; it is not a "
        "property of the person."
    ),
    "measured_skew": (
        "On a demographically balanced test set this model selects White faces into its "
        "top-100 at 2.2x their share and Southeast Asian faces at 0.23x. See "
        "docs/RESEARCH.md section 13.5."
    ),
}


class IndexRequest(BaseModel):
    roots: list[str]
    limit: int = 0
    force: bool = False


class FeedbackRequest(BaseModel):
    face_id: int
    kind: str          # like | dislike | hide | wrong
    note: str | None = None
    user: str = "default"
    remove: bool = False


class SaveSearchRequest(BaseModel):
    name: str
    spec: dict[str, Any]


class IndexJob:
    """Tracks a background indexing run so the UI can show progress and resume."""

    def __init__(self):
        self.lock = threading.Lock()
        self.state: dict[str, Any] = {"status": "idle"}
        self.thread: threading.Thread | None = None

    def running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def start(self, index_path, features_dir, roots, limit, force):
        if self.running():
            raise HTTPException(409, "an indexing run is already in progress")

        def work():
            from ..pipeline.indexer import IndexConfig, Indexer
            try:
                with self.lock:
                    self.state = {"status": "loading models", "started_at": time.time()}
                ix = Indexer(index_path, features_dir, IndexConfig())
                with self.lock:
                    self.state.update(status="scanning", encoder=ix.encoder_version)
                st = ix.index_directories(roots, force=force, limit=limit)
                with self.lock:
                    self.state = {"status": "done", "finished_at": time.time(),
                                  **{k: v for k, v in st.__dict__.items() if k != "errors"},
                                  "errors": st.errors[:20]}
                ix.close()
            except Exception as e:  # noqa: BLE001 - surface failures to the UI
                with self.lock:
                    self.state = {"status": "failed", "error": f"{type(e).__name__}: {e}"}

        self.thread = threading.Thread(target=work, daemon=True)
        self.thread.start()
        return {"status": "started"}


def create_app(index_path: str, features_dir: str) -> FastAPI:
    app = FastAPI(title="Facet", version="0.1.0",
                  description="Face analysis, filtering and ranking. All estimates, no "
                              "measurements - see /api/about.")
    job = IndexJob()
    local = threading.local()

    def engine() -> SearchEngine:
        if not Path(index_path).exists():
            raise HTTPException(404, f"no index at {index_path} - run an index first")
        if getattr(local, "eng", None) is None:
            local.eng = SearchEngine(index_path)      # sqlite conns are per-thread
        return local.eng

    def db() -> Index:
        if getattr(local, "db", None) is None:
            local.db = Index(index_path)
        return local.db

    # ------------------------------------------------------------------ meta

    @app.get("/api/about")
    def about():
        return {"name": "Facet", "disclaimer": DISCLAIMER}

    @app.get("/api/stats")
    def stats():
        s = db().stats()
        s["index_path"] = str(index_path)
        s["saved_searches"] = len(db().list_saved_searches())
        s["feedback"] = db().conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
        s["runs"] = [dict(r) for r in db().conn.execute(
            "SELECT id, started_at, finished_at, n_indexed, n_faces, n_failed, status "
            "FROM runs ORDER BY id DESC LIMIT 5")]
        return s

    # ---------------------------------------------------------------- search

    @app.post("/api/search")
    def search(spec: dict[str, Any]):
        try:
            q = QuerySpec.from_dict(spec)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(400, f"bad query: {e}") from e
        resp = engine().search(q)
        out = resp.as_dict()
        fb = db().feedback_for(spec.get("user", "default"))
        for r in out["results"]:
            r["feedback"] = fb.get(r["face_id"], [])
        out["disclaimer"] = DISCLAIMER
        return out

    @app.get("/api/face/{face_id}")
    def face(face_id: int):
        row = db().conn.execute(
            "SELECT f.*, i.path, i.width, i.height FROM faces f "
            "JOIN images i ON i.id=f.image_id WHERE f.id=?", (face_id,)).fetchone()
        if row is None:
            raise HTTPException(404, "no such face")
        preds = {}
        for p in db().conn.execute("SELECT * FROM predictions WHERE face_id=?", (face_id,)):
            d = dict(p)
            for k in ("distribution", "extra"):
                if d.get(k):
                    d[k] = json.loads(d[k])
            preds[d.pop("model")] = d
        out = dict(row)
        out.pop("kps", None)
        out["quality_json"] = json.loads(out["quality_json"]) if out["quality_json"] else {}
        out["predictions"] = preds
        out["duplicates"] = [dict(r) for r in db().conn.execute(
            "SELECT kind, group_id FROM duplicates WHERE face_id=?", (face_id,))]
        out["disclaimer"] = DISCLAIMER
        return out

    # ----------------------------------------------------------------- media

    @app.get("/api/image/{image_id}")
    def image(image_id: int, max_px: int = Query(1600, ge=64, le=4096)):
        r = db().conn.execute("SELECT path FROM images WHERE id=?", (image_id,)).fetchone()
        if r is None:
            raise HTTPException(404, "no such image")
        return _encode(_load_scaled(r["path"], max_px))

    @app.get("/api/crop/{face_id}")
    def crop(face_id: int, size: int = Query(256, ge=32, le=1024), margin: float = 0.4):
        import cv2
        r = db().conn.execute(
            "SELECT f.x1,f.y1,f.x2,f.y2,i.path FROM faces f JOIN images i ON i.id=f.image_id "
            "WHERE f.id=?", (face_id,)).fetchone()
        if r is None:
            raise HTTPException(404, "no such face")
        from ..models.insightface_backend import crop_bbox
        img = cv2.imread(r["path"])
        if img is None:
            raise HTTPException(410, "source image is no longer readable")
        return _encode(crop_bbox(img, (r["x1"], r["y1"], r["x2"], r["y2"]),
                                 size=size, margin=margin))

    # --------------------------------------------------------------- indexing

    @app.post("/api/index")
    def start_index(req: IndexRequest):
        for r in req.roots:
            if not Path(r).expanduser().is_dir():
                raise HTTPException(400, f"not a directory: {r}")
        return job.start(index_path, features_dir,
                         [str(Path(r).expanduser()) for r in req.roots], req.limit, req.force)

    @app.get("/api/index/status")
    def index_status():
        with job.lock:
            s = dict(job.state)
        s["running"] = job.running()
        return s

    # --------------------------------------------------------- feedback / saved

    @app.post("/api/feedback")
    def feedback(req: FeedbackRequest):
        if req.kind not in {"like", "dislike", "hide", "wrong"}:
            raise HTTPException(400, "kind must be like, dislike, hide or wrong")
        if req.remove:
            db().remove_feedback(req.face_id, req.kind, req.user)
        else:
            db().add_feedback(req.face_id, req.kind, req.user, req.note)
        return {"ok": True, "feedback": db().feedback_for(req.user).get(req.face_id, [])}

    @app.get("/api/searches")
    def list_searches():
        return db().list_saved_searches()

    @app.post("/api/searches")
    def save_search(req: SaveSearchRequest):
        db().save_search(req.name, req.spec)
        return {"ok": True}

    @app.delete("/api/searches/{name}")
    def delete_search(name: str):
        db().delete_saved_search(name)
        return {"ok": True}

    # ----------------------------------------------------------------- export

    @app.post("/api/export")
    def export(spec: dict[str, Any], fmt: str = Query("csv", pattern="^(csv|json)$")):
        resp = engine().search(QuerySpec.from_dict(spec))
        if fmt == "json":
            return JSONResponse({"disclaimer": DISCLAIMER, **resp.as_dict()})
        import csv
        buf = io.StringIO()
        # The provenance travels with the export - a CSV of bare numbers would strip exactly
        # the context that makes these figures honest.
        buf.write(f"# {DISCLAIMER['estimates_not_measurements']}\n")
        buf.write(f"# {DISCLAIMER['measured_skew']}\n")
        w = csv.writer(buf)
        w.writerow(["face_id", "path", "relevance", "attractiveness",
                    "attractiveness_percentile", "p_ge4", "confidence", "interval_lo",
                    "interval_hi", "age", "gender", "gender_confidence", "quality",
                    "out_of_distribution", "x1", "y1", "x2", "y2"])
        for r in resp.results:
            iv = r.interval or (None, None)
            w.writerow([r.face_id, r.path, round(r.relevance, 4), r.attractiveness,
                        r.attractiveness_percentile, r.p_ge4, r.confidence, iv[0], iv[1],
                        r.age, r.gender, r.gender_confidence, round(r.quality, 4),
                        r.ood, *[round(v, 1) for v in r.bbox]])
        return PlainTextResponse(buf.getvalue(), media_type="text/csv",
                                 headers={"Content-Disposition":
                                          "attachment; filename=facet_results.csv"})

    # --------------------------------------------------------------------- ui

    @app.get("/", response_class=HTMLResponse)
    def ui():
        f = WEB_DIR / "index.html"
        if not f.exists():
            return HTMLResponse("<h1>Facet API</h1><p>UI not found. See /docs</p>")
        return HTMLResponse(f.read_text())

    return app


def _load_scaled(path: str, max_px: int):
    import cv2
    img = cv2.imread(path)
    if img is None:
        raise HTTPException(410, "source image is no longer readable")
    h, w = img.shape[:2]
    s = min(1.0, max_px / max(h, w))
    if s < 1.0:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


def _encode(img):
    import cv2
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise HTTPException(500, "encode failed")
    return Response(buf.tobytes(), media_type="image/jpeg",
                    headers={"Cache-Control": "public, max-age=3600"})
