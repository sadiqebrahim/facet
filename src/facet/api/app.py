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

import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, Response
from pydantic import BaseModel

from ..pipeline.db import Index
from .auth import Accounts, valid_username
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
    #: Age/gender is ~190x the cost of the rest (E4), so it is opt-out and quality-gated.
    predict_age: bool = True
    min_quality: float = 0.35
    age_limit: int = 0


class FeedbackRequest(BaseModel):
    face_id: int
    kind: str          # like | dislike | hide | wrong
    note: str | None = None
    user: str = "default"
    remove: bool = False
    #: Seconds during which the judgement can be undone. The face leaves the results
    #: immediately, but the preference model does not learn from it until this elapses -
    #: so an undo inside the window leaves nothing to unlearn.
    undo_seconds: float = 10.0


class UndoRequest(BaseModel):
    face_id: int
    kind: str | None = None
    user: str = "default"


class Credentials(BaseModel):
    username: str
    password: str
    display_name: str | None = None


class PasswordChange(BaseModel):
    current_password: str | None = None
    new_password: str
    username: str | None = None      # admin only: reset somebody else's


class AdminUserAction(BaseModel):
    username: str
    make_admin: bool | None = None


class ReferenceRequest(BaseModel):
    """Teach the preference model from example images the user picks."""

    paths: list[str]
    kind: str = "like"          # like | dislike
    user: str = "default"


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

    def start(self, index_path, features_dir, roots, limit, force,
              predict_age=True, min_quality=0.35, age_limit=0):
        if self.running():
            raise HTTPException(409, "an indexing run is already in progress")

        def work():
            """Run the WHOLE pipeline, not just the index pass.

            The index/predict split is right architecturally (15.1) but it was wrong as a
            product: indexing alone leaves every face with no attractiveness, age or gender,
            so the UI showed a grid of dashes and 0% matches and looked broken. One user
            action should produce usable results.

            Stages are reported separately so progress stays legible, and age/gender is
            still gated - E4 measured MiVOLO at ~190x the cost of everything else, so it
            runs on the best faces rather than all of them.
            """
            from ..pipeline.indexer import IndexConfig, Indexer

            def set(**kw):
                with self.lock:
                    self.state.update(**kw)

            try:
                with self.lock:
                    self.state = {"status": "running", "stage": "loading models",
                                  "started_at": time.time(), "stages": {}}
                ix = Indexer(index_path, features_dir, IndexConfig())
                set(encoder=ix.encoder_version)

                set(stage="scanning and encoding")
                st = ix.index_directories(roots, force=force, limit=limit)
                with self.lock:
                    self.state["stages"]["index"] = {
                        k: v for k, v in st.__dict__.items() if k != "errors"}
                    self.state.update(indexed=st.indexed, skipped=st.skipped,
                                      faces=st.faces, failed=st.failed,
                                      errors=st.errors[:20])
                ix.close()

                # --- predictions over the cached features ------------------------
                import sys as _sys
                _sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
                from predict_attributes import open_store, run_age, run_beauty, run_dupes

                idx = Index(index_path)
                try:
                    store, _, _ = open_store(Path(features_dir), idx)
                except SystemExit:
                    store = None

                if store is not None:
                    head = Path(__file__).resolve().parents[3] / "artifacts/models/beauty_head.npz"
                    if head.exists():
                        set(stage="scoring attractiveness")
                        with self.lock:
                            self.state["stages"]["beauty"] = run_beauty(idx, store, head)
                    else:
                        with self.lock:
                            self.state["stages"]["beauty"] = {
                                "skipped": "no trained head at artifacts/models/beauty_head.npz "
                                           "- run scripts/train_beauty_head.py"}
                    if predict_age:
                        set(stage="estimating age and gender (slow)")
                        with self.lock:
                            self.state["stages"]["age"] = run_age(idx, min_quality, age_limit)
                    set(stage="finding duplicates")
                    with self.lock:
                        self.state["stages"]["dupes"] = run_dupes(idx, store, 0.92)
                idx.close()

                set(status="done", stage="done", finished_at=time.time())
            except Exception as e:  # noqa: BLE001 - surface failures to the UI
                import traceback
                with self.lock:
                    self.state.update(status="failed", stage="failed",
                                      error=f"{type(e).__name__}: {e}",
                                      traceback=traceback.format_exc()[-1200:])

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
            # features_dir is needed for personalisation: the preference model is fitted
            # over the same cached embeddings the ranking already uses.
            local.eng = SearchEngine(index_path, features_dir)   # sqlite conns are per-thread
        return local.eng

    def db() -> Index:
        if getattr(local, "db", None) is None:
            local.db = Index(index_path)
        return local.db

    def accounts() -> Accounts:
        if getattr(local, "acc", None) is None:
            local.acc = Accounts(db().conn)
        return local.acc

    def require_admin(user: str) -> str:
        if not accounts().is_admin(user):
            raise HTTPException(403, "administrator access required")
        return user

    def current_user(authorization: str | None = Header(default=None)) -> str:
        """Resolve the caller.

        Until the first account is created the app runs open, under the shared 'default'
        user - so a fresh local install works without a login step. The moment anyone
        registers, authentication becomes mandatory, because from then on there are
        separate taste models to keep apart.
        """
        acc = accounts()
        if acc.count() == 0:
            return "default"
        token = None
        if authorization and authorization.lower().startswith("bearer "):
            token = authorization[7:].strip()
        user = acc.resolve(token)
        if user is None:
            raise HTTPException(401, "sign in to continue")
        return user

    # ------------------------------------------------------------------ auth

    @app.get("/api/auth/status")
    def auth_status(authorization: str | None = Header(default=None)):
        acc = accounts()
        token = authorization[7:].strip() if (
            authorization and authorization.lower().startswith("bearer ")) else None
        user = acc.resolve(token)
        return {"accounts_exist": acc.count() > 0, "authenticated": user is not None,
                "user": user, "open_mode": acc.count() == 0,
                "is_admin": bool(user and acc.is_admin(user)),
                "must_change_password": bool(user and acc.must_change(user)),
                # Named so "forgot password" has somewhere to point: there is no mail server
                # here, so a reset means asking a person.
                "admins": acc.admins(),
                "note": ("No accounts yet - running in open mode under a shared profile. "
                         "Create an account to get a taste model of your own."
                         if acc.count() == 0 else "")}

    @app.post("/api/auth/register")
    def register(c: Credentials):
        acc = accounts()
        if not valid_username(c.username):
            raise HTTPException(400, "username must be 2-32 chars: letters, digits, . _ -")
        if len(c.password) < 6:
            raise HTTPException(400, "password must be at least 6 characters")
        if acc.exists(c.username):
            raise HTTPException(409, "that username is taken")
        first = acc.count() == 0
        acc.create(c.username, c.password, c.display_name)
        if first:
            # Carry the open-mode profile over so a first-run user does not lose the
            # preferences they taught before creating an account.
            for t, col in (("feedback", "user"), ("reference_faces", "user")):
                db().conn.execute(f"UPDATE {t} SET {col}=? WHERE {col}='default'",
                                  (c.username,))
            db().conn.commit()
        return {"ok": True, "token": acc.start_session(c.username), "user": c.username,
                "migrated_default_profile": first}

    @app.post("/api/auth/login")
    def login(c: Credentials):
        acc = accounts()
        if not acc.check(c.username, c.password):
            raise HTTPException(401, "wrong username or password")
        return {"ok": True, "token": acc.start_session(c.username), "user": c.username}

    @app.post("/api/auth/logout")
    def logout(authorization: str | None = Header(default=None)):
        if authorization and authorization.lower().startswith("bearer "):
            accounts().end_session(authorization[7:].strip())
        return {"ok": True}

    @app.get("/api/auth/me")
    def me(user: str = Depends(current_user)):
        acc = accounts()
        return {"user": user, "is_admin": acc.is_admin(user),
                "must_change_password": acc.must_change(user), **acc.info(user)}

    @app.post("/api/auth/password")
    def change_password(req: PasswordChange, user: str = Depends(current_user)):
        """Change your own password, or - as an admin - reset somebody else's."""
        acc = accounts()
        target = req.username or user
        if len(req.new_password) < 6:
            raise HTTPException(400, "password must be at least 6 characters")
        if target != user:
            require_admin(user)
        elif not (req.current_password and acc.check(user, req.current_password)):
            raise HTTPException(401, "current password is wrong")
        if not acc.exists(target):
            raise HTTPException(404, "no such user")
        # A self-service change should not force another change; an admin reset should.
        acc.set_password(target, req.new_password, must_change=(target != user))
        if target == user:
            acc.clear_must_change(user)
            return {"ok": True, "token": acc.start_session(user),
                    "note": "password changed; other sessions were signed out"}
        return {"ok": True, "note": f"{target} must set a new password at next sign-in"}

    # ----------------------------------------------------------------- admin

    @app.get("/api/admin/users")
    def admin_users(user: str = Depends(current_user)):
        require_admin(user)
        acc, ix = accounts(), db()
        rows = acc.details()
        for r in rows:
            r["likes"] = ix.conn.execute(
                "SELECT COUNT(*) FROM feedback WHERE user=? AND kind='like'",
                (r["username"],)).fetchone()[0]
            r["dislikes"] = ix.conn.execute(
                "SELECT COUNT(*) FROM feedback WHERE user=? AND kind='dislike'",
                (r["username"],)).fetchone()[0]
            r["references"] = ix.conn.execute(
                "SELECT COUNT(*) FROM reference_faces WHERE user=?",
                (r["username"],)).fetchone()[0]
            r["sessions"] = ix.conn.execute(
                "SELECT COUNT(*) FROM sessions WHERE username=? AND expires_at > ?",
                (r["username"], __import__("time").time())).fetchone()[0]
        return {"users": rows, "you": user}

    @app.post("/api/admin/users")
    def admin_create(c: Credentials, user: str = Depends(current_user)):
        require_admin(user)
        acc = accounts()
        if not valid_username(c.username):
            raise HTTPException(400, "username must be 2-32 chars: letters, digits, . _ -")
        if len(c.password) < 6:
            raise HTTPException(400, "password must be at least 6 characters")
        if acc.exists(c.username):
            raise HTTPException(409, "that username is taken")
        acc.create(c.username, c.password, c.display_name, is_admin=False)
        acc.set_password(c.username, c.password, must_change=True)
        return {"ok": True, "note": f"{c.username} created; they must set a new password"}

    @app.post("/api/admin/role")
    def admin_role(req: AdminUserAction, user: str = Depends(current_user)):
        require_admin(user)
        acc = accounts()
        if not acc.exists(req.username):
            raise HTTPException(404, "no such user")
        if req.make_admin is False and acc.admins() == [req.username]:
            raise HTTPException(400, "that is the only administrator")
        acc.set_admin(req.username, bool(req.make_admin))
        return {"ok": True}

    @app.delete("/api/admin/users/{username}")
    def admin_delete(username: str, user: str = Depends(current_user)):
        require_admin(user)
        acc = accounts()
        if username == user:
            raise HTTPException(400, "you cannot delete your own account")
        if not acc.exists(username):
            raise HTTPException(404, "no such user")
        if acc.is_admin(username) and len(acc.admins()) == 1:
            raise HTTPException(400, "that is the only administrator")
        ix = db()
        # Deleting an account removes the biometric-derived data it accumulated, per
        # docs/LICENSING.md section 4.2: deletion must actually delete.
        for t in ("feedback", "reference_faces"):
            ix.conn.execute(f"DELETE FROM {t} WHERE user=?", (username,))
        ix.conn.commit()
        acc.delete(username)
        return {"ok": True, "note": "account and all learned preferences removed"}

    @app.get("/api/admin/overview")
    def admin_overview(user: str = Depends(current_user)):
        require_admin(user)
        ix, acc = db(), accounts()
        st = ix.stats()
        st["users"] = acc.count()
        st["admins"] = acc.admins()
        st["errors"] = [dict(r) for r in ix.conn.execute(
            "SELECT path, error FROM images WHERE status IN ('corrupt','unreadable') LIMIT 25")]
        st["runs"] = [dict(r) for r in ix.conn.execute(
            "SELECT id, started_at, finished_at, n_indexed, n_faces, n_failed, status "
            "FROM runs ORDER BY id DESC LIMIT 10")]
        st["feature_shards"] = sorted(
            p.name for p in Path(features_dir).glob("*.json")) if Path(features_dir).is_dir() else []
        return st

    # ------------------------------------------------------------------ meta

    @app.get("/api/about")
    def about():
        return {"name": "Facet", "disclaimer": DISCLAIMER}

    @app.get("/api/stats")
    def stats(user: str = Depends(current_user)):
        s = db().stats()
        s["ready"] = s["faces"] > 0
        s["has_predictions"] = s["predictions"] > 0
        s["beauty_head_trained"] = (
            Path(__file__).resolve().parents[3] / "artifacts/models/beauty_head.npz").exists()
        s["index_path"] = str(index_path)
        s["saved_searches"] = len(db().list_saved_searches())
        s["feedback"] = db().conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
        s["runs"] = [dict(r) for r in db().conn.execute(
            "SELECT id, started_at, finished_at, n_indexed, n_faces, n_failed, status "
            "FROM runs ORDER BY id DESC LIMIT 5")]
        return s

    # ---------------------------------------------------------------- search

    @app.post("/api/search")
    def search(spec: dict[str, Any], user: str = Depends(current_user)):
        # The caller does not get to choose whose profile is used: personalisation and the
        # dislike filter both read per-user rows, so a client-supplied name would be a way
        # to read someone else's taste.
        spec = dict(spec)
        spec["personalisation"] = {**(spec.get("personalisation") or {}), "user": user}
        try:
            q = QuerySpec.from_dict(spec)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(400, f"bad query: {e}") from e
        resp = engine().search(q)
        out = resp.as_dict()
        fb = db().feedback_for(user)
        for r in out["results"]:
            r["feedback"] = fb.get(r["face_id"], [])
        out["disclaimer"] = DISCLAIMER
        return out

    @app.get("/api/face/{face_id}")
    def face(face_id: int, user: str = Depends(current_user)):
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
    #
    # A browser <img src="..."> cannot attach an Authorization header, so media accepts the
    # session token as a query parameter as well. It is the same token with the same
    # lifetime - the point is that media stops being anonymously readable, which it was.

    def media_user(authorization: str | None = Header(default=None),
                   t: str | None = Query(default=None)) -> str:
        acc = accounts()
        if acc.count() == 0:
            return "default"
        token = t
        if not token and authorization and authorization.lower().startswith("bearer "):
            token = authorization[7:].strip()
        u = acc.resolve(token)
        if u is None:
            raise HTTPException(401, "sign in to view images")
        return u

    @app.get("/api/image/{image_id}")
    def image(image_id: int, max_px: int = Query(1600, ge=64, le=4096),
              user: str = Depends(media_user)):
        r = db().conn.execute("SELECT path FROM images WHERE id=?", (image_id,)).fetchone()
        if r is None:
            raise HTTPException(404, "no such image")
        return _encode(_load_scaled(r["path"], max_px))

    @app.get("/api/crop/{face_id}")
    def crop(face_id: int, size: int = Query(256, ge=32, le=1024), margin: float = 0.4,
             user: str = Depends(media_user)):
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
    def start_index(req: IndexRequest, user: str = Depends(current_user)):
        for r in req.roots:
            if not Path(r).expanduser().is_dir():
                raise HTTPException(400, f"not a directory: {r}")
        return job.start(index_path, features_dir,
                         [str(Path(r).expanduser()) for r in req.roots], req.limit, req.force,
                         req.predict_age, req.min_quality, req.age_limit)

    @app.get("/api/index/status")
    def index_status(user: str = Depends(current_user)):
        with job.lock:
            s = dict(job.state)
        s["running"] = job.running()
        return s

    # --------------------------------------------------------- feedback / saved

    @app.post("/api/feedback")
    def feedback(req: FeedbackRequest, user: str = Depends(current_user)):
        req.user = user
        if req.kind not in {"like", "dislike", "hide", "wrong"}:
            raise HTTPException(400, "kind must be like, dislike, hide or wrong")
        if req.remove:
            db().remove_feedback(req.face_id, req.kind, req.user)
            return {"ok": True, "feedback": db().feedback_for(user).get(req.face_id, []),
                    "commit_at": None, "undo_seconds": 0}
        commit_at = db().add_feedback(req.face_id, req.kind, req.user, req.note,
                                      undo_seconds=req.undo_seconds)
        return {"ok": True, "feedback": db().feedback_for(user).get(req.face_id, []),
                "commit_at": commit_at, "undo_seconds": req.undo_seconds,
                "note": ("hidden from results now; it starts shaping your taste model in "
                         f"{req.undo_seconds:g}s unless you undo")}

    @app.post("/api/feedback/undo")
    def undo(req: UndoRequest, user: str = Depends(current_user)):
        """Reverse a judgement: the face returns to results and, if still inside the undo
        window, the preference model never saw it."""
        n = db().undo_feedback(req.face_id, user, req.kind)
        return {"ok": True, "removed": n,
                "feedback": db().feedback_for(user).get(req.face_id, [])}

    @app.get("/api/feedback/pending")
    def pending(user: str = Depends(current_user)):
        return {"pending": db().pending_feedback(user)}

    @app.get("/api/searches")
    def list_searches(user: str = Depends(current_user)):
        return db().list_saved_searches()

    @app.post("/api/searches")
    def save_search(req: SaveSearchRequest, user: str = Depends(current_user)):
        db().save_search(req.name, req.spec)
        return {"ok": True}

    @app.delete("/api/searches/{name}")
    def delete_search(name: str, user: str = Depends(current_user)):
        db().delete_saved_search(name)
        return {"ok": True}

    # -------------------------------------------------------- personalisation

    _enc: dict = {}

    def encoder():
        """Detector + embedder for reference images, loaded once and reused.

        Reference faces must be encoded exactly as indexed faces were, or the preference
        model would be comparing vectors from different preprocessing - the same class of
        error the versioned feature shards exist to prevent.
        """
        if not _enc:
            from ..models.insightface_backend import (
                ArcFaceEmbedder, InsightFaceDetector, align_to_template)
            from ..pipeline.indexer import IndexConfig
            cfg = IndexConfig()
            _enc["cfg"] = cfg
            _enc["det"] = InsightFaceDetector(pack=cfg.pack, det_size=cfg.det_size,
                                              pad_frac=cfg.pad_frac)
            _enc["arc"] = ArcFaceEmbedder(pack=cfg.pack)
            _enc["align"] = align_to_template
            if cfg.clip:
                from ..models.clip_backend import ClipEmbedder
                _enc["clip"] = ClipEmbedder()
        return _enc

    @app.get("/api/preference")
    def preference(user: str = Depends(current_user)):
        eng = engine()
        pm = eng.preference_model(user)
        refs = [{"id": r["id"], "path": r["path"], "kind": r["kind"]}
                for r in db().references(user)]
        fb = db().conn.execute(
            "SELECT kind, COUNT(*) n FROM feedback WHERE user=? GROUP BY kind",
            (user,)).fetchall()
        out = {"references": refs, "feedback": {r["kind"]: r["n"] for r in fb},
               "trained": pm is not None}
        if pm is not None:
            st = pm.status()
            out.update(alpha=st.alpha, method=st.method, note=st.note,
                       n_likes=st.n_likes, n_dislikes=st.n_dislikes)
        else:
            out["note"] = ("Not taught yet. Add reference faces you find attractive, or "
                           "rate results with ★ / Not for me.")
            out["alpha"] = 0.0
        return out

    @app.post("/api/preference/references")
    def add_references(req: ReferenceRequest, user: str = Depends(current_user)):
        req.user = user
        import cv2
        if req.kind not in {"like", "dislike"}:
            raise HTTPException(400, "kind must be like or dislike")
        e = encoder()
        added, skipped = [], []
        for raw in req.paths:
            path = Path(raw).expanduser()
            files = ([p for p in sorted(path.iterdir())
                      if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}]
                     if path.is_dir() else [path])
            if not files:
                skipped.append({"path": str(path), "reason": "no images found"})
            for f in files:
                img = cv2.imread(str(f))
                if img is None:
                    skipped.append({"path": str(f), "reason": "could not read"})
                    continue
                dets = e["det"].detect(img)
                if not dets:
                    skipped.append({"path": str(f), "reason": "no face detected"})
                    continue
                cfg = e["cfg"]
                crop = e["align"](img, dets[0].keypoints, size=cfg.crop_size,
                                  margin=cfg.crop_margin)
                vec = e["arc"].encode(np.stack([crop]))[0]
                if "clip" in e:
                    vec = np.concatenate([vec, e["clip"].encode(np.stack([crop]))[0]])
                db().add_reference(str(f), 0, vec, req.kind, req.user)
                added.append(str(f))
        return {"added": len(added), "skipped": skipped,
                "total_references": len(db().references(req.user))}

    @app.delete("/api/preference/references/{ref_id}")
    def delete_reference(ref_id: int, user: str = Depends(current_user)):
        db().delete_reference(ref_id, user)
        return {"ok": True}

    @app.post("/api/preference/reset")
    def reset_preference(user: str = Depends(current_user)):
        n = db().clear_references(user)
        db().conn.execute("DELETE FROM feedback WHERE user=? AND kind IN ('like','dislike')",
                          (user,))
        db().conn.commit()
        return {"ok": True, "cleared_references": n}

    # ----------------------------------------------------------------- export

    @app.post("/api/export")
    def export(spec: dict[str, Any], fmt: str = Query("csv", pattern="^(csv|json)$"),
               user: str = Depends(current_user)):
        spec = {**spec, "personalisation": {**(spec.get("personalisation") or {}),
                                            "user": user}}
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
