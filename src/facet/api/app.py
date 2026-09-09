"""FastAPI application: search, media, indexing progress, feedback, export.

Thin by design. The API translates HTTP into `QuerySpec` and back, and adds nothing to the
ranking - `docs/QUERY.md` explains why the scoring function stays transparent and explainable
rather than moving into a service layer.

Three responsibilities beyond plumbing:

* **Every payload carries provenance.** Attractiveness values leave this process with the
  SCUT-FBP5500 source string and the measured demographic skew attached, so a client cannot
  render a bare number without also having the caveats (RESEARCH.md 11.4, 13.5).
* **Tenancy is enforced here, once.** `current_user` resolves the session and every data
  path is scoped to it. A client never names the account it wants to read; the two places
  that used to accept one (`spec.personalisation.user`, `FeedbackRequest.user`) now
  overwrite it. See docs/HOSTING.md.
* **Outbound requests are enumerable.** There are exactly two: Google's token endpoint
  during sign-in, and fetching an image the user pasted a URL for. Both are opt-in, both
  are in this file's imports, and a deployment with neither configured still makes none -
  which keeps LICENSING.md section 4's local-only promise available to self-hosters.
"""
from __future__ import annotations

import io
import json
import os
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import (
    Depends, FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile,
)
from fastapi.responses import (
    HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse, Response,
)
from pydantic import BaseModel

from ..pipeline.db import Index
from . import oauth, uploads
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


class RateLimiter:
    """Fixed-window attempt counter, keyed by client address and target account.

    A password endpoint on the public internet gets found. Without this, the only cost of
    guessing is the ~200ms PBKDF2 does, which is not a deterrent at scale. Keyed on both
    the caller and the account so one noisy network cannot lock out everybody, and one
    attacker cannot spread a single account's guesses across many source ports.
    """

    def __init__(self, limit: int = 10, window: float = 300.0):
        self.limit, self.window = limit, window
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, *key_parts: str) -> None:
        key = "\x00".join(str(k) for k in key_parts)
        now = time.time()
        with self._lock:
            q = self._hits[key]
            while q and q[0] < now - self.window:
                q.popleft()
            if len(q) >= self.limit:
                wait = int(self.window - (now - q[0])) + 1
                raise HTTPException(429, f"too many attempts - try again in {wait}s")
            q.append(now)

    def clear(self, *key_parts: str) -> None:
        self._hits.pop("\x00".join(str(k) for k in key_parts), None)


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
    remove: bool = False
    #: Seconds during which THIS judgement can be undone. The face leaves the results
    #: immediately, but the preference model does not learn from it until this elapses -
    #: so an undo inside the window leaves nothing to unlearn. Marking anything else
    #: closes the window early: only one judgement is ever reversible at a time.
    undo_seconds: float = 10.0


class UndoRequest(BaseModel):
    face_id: int
    kind: str | None = None


class UrlImportRequest(BaseModel):
    urls: list[str]
    #: library = something to search through. reference = an example of your taste.
    target: str = "library"
    kind: str = "like"          # reference imports only


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
    """Teach the preference model from example images on the server's own filesystem.

    Only reachable when FACET_ALLOW_SERVER_PATHS is set - see uploads.py on why a hosted
    deployment must not offer this.
    """

    paths: list[str]
    kind: str = "like"          # like | dislike


class DeleteAccountRequest(BaseModel):
    #: Typed confirmation. Deleting an account destroys the uploaded photographs, the
    #: derived embeddings and the taste model, none of which can be reconstructed.
    confirm: str
    password: str | None = None


class SaveSearchRequest(BaseModel):
    name: str
    spec: dict[str, Any]


class IndexJob:
    """Tracks background indexing runs so the UI can show progress and resume.

    One slot per account. A shared slot would mean one user's import blocked everyone
    else's and - worse - that everyone watched the same progress bar, which leaks how much
    other people are uploading and when.
    """

    def __init__(self):
        # Re-entrant on purpose. The state helpers are nested closures called from all over
        # the worker, and a plain Lock turned one accidental re-entry into a deadlock that
        # froze every request on the server, not just the import. Re-entrancy costs nothing
        # here - every critical section is a dictionary update.
        self.lock = threading.RLock()
        self.states: dict[str, dict[str, Any]] = {}
        self.threads: dict[str, threading.Thread] = {}
        #: Roots that arrived while a run was in progress. Uploading a second batch while
        #: the first is still indexing is the normal case, not an error - the files are
        #: already saved by then, so refusing the request would leave them unindexed
        #: forever.
        self.queued: dict[str, set[str]] = {}

    def state_for(self, owner: str) -> dict[str, Any]:
        with self.lock:
            return dict(self.states.get(owner) or {"status": "idle"})

    def running(self, owner: str) -> bool:
        t = self.threads.get(owner)
        return t is not None and t.is_alive()

    def any_running(self) -> list[str]:
        return [u for u, t in self.threads.items() if t.is_alive()]

    def start(self, index_path, features_dir, roots, limit, force,
              predict_age=True, min_quality=0.35, age_limit=0, owner: str = "",
              source: str = "local", note: str = ""):
        if self.running(owner):
            with self.lock:
                self.queued.setdefault(owner, set()).update(roots)
            return {"status": "queued",
                    "note": "an import is already running - these will be picked up when "
                            "it finishes"}

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

            def report(**kw):
                """Publish progress. NOT called `set`: shadowing the builtin once turned
                `self.queued.pop(owner, set())` - written inside `with self.lock:` - into a
                recursive call that deadlocked the whole server on its own lock."""
                with self.lock:
                    self.states.setdefault(owner, {}).update(**kw)

            def stage(name, value):
                with self.lock:
                    self.states.setdefault(owner, {}).setdefault("stages", {})[name] = value

            def try_stage(name, label, fn):
                """Run one stage, and record a failure instead of losing the whole run.

                Detection and encoding are the expensive part; attractiveness, age and
                duplicates each run over what they produced. Letting a failure in one of
                those abort the job threw away work that had already succeeded - which is
                exactly what happened when the age pass hit a full GPU.
                """
                report(stage=label)
                try:
                    stage(name, fn())
                    return True
                except Exception as e:  # noqa: BLE001 - report and carry on
                    stage(name, {"failed": f"{type(e).__name__}: {e}"[:300]})
                    with self.lock:
                        self.states.setdefault(owner, {}).setdefault(
                            "warnings", []).append(f"{label} failed: {type(e).__name__}")
                    return False

            total = {"indexed": 0, "faces": 0, "failed": 0, "skipped": 0}

            def one_pass(scan_roots):
                ix = Indexer(index_path, features_dir, IndexConfig())
                report(encoder=ix.encoder_version, device=ix.device)
                report(stage="scanning and encoding")
                st = ix.index_directories(scan_roots, force=force, limit=limit,
                                          owner=owner, source=source)
                for k in total:
                    total[k] += getattr(st, k)
                stage("index", dict(total))
                report(errors=st.errors[:20], **total)
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
                    head = (Path(__file__).resolve().parents[3]
                            / "artifacts/models/beauty_head.npz")
                    if head.exists():
                        try_stage("beauty", "scoring attractiveness",
                                  lambda: run_beauty(idx, store, head, owner=owner))
                    else:
                        stage("beauty", {
                            "skipped": "no trained head at artifacts/models/beauty_head.npz "
                                       "- run scripts/train_beauty_head.py"})
                    if predict_age:
                        try_stage("age", "estimating age and gender (slow)",
                                  lambda: run_age(idx, min_quality, age_limit, owner=owner))
                    try_stage("dupes", "finding duplicates",
                              lambda: run_dupes(idx, store, 0.92, owner=owner))
                idx.close()

                idx2 = Index(index_path)
                idx2.bump_generation()
                idx2.close()

            try:
                with self.lock:
                    self.states[owner] = {"status": "running", "stage": "loading models",
                                          "started_at": time.time(), "stages": {},
                                          "note": note}
                pending = list(roots)
                while pending:
                    # Anything uploaded mid-run joins the next lap rather than ending up
                    # indexed but unscored - or, worse, refused with a 409 after the bytes
                    # were already accepted.
                    with self.lock:
                        pending = list(dict.fromkeys([*pending,
                                                      *self.queued.pop(owner, set())]))
                    one_pass(pending)
                    with self.lock:
                        pending = list(self.queued.pop(owner, set()))
                    if pending:
                        report(stage="picking up newly added images")
                report(status="done", stage="done", finished_at=time.time())
            except Exception as e:  # noqa: BLE001 - surface failures to the UI
                import traceback
                with self.lock:
                    self.states.setdefault(owner, {}).update(
                        status="failed", stage="failed",
                        error=f"{type(e).__name__}: {e}",
                        traceback=traceback.format_exc()[-1200:])

        t = threading.Thread(target=work, daemon=True)
        self.threads[owner] = t
        t.start()
        return {"status": "started"}


def create_app(index_path: str, features_dir: str,
               upload_root: str | Path | None = None) -> FastAPI:
    app = FastAPI(title="Facet", version="0.2.0",
                  description="Face analysis, filtering and ranking. All estimates, no "
                              "measurements - see /api/about.")
    job = IndexJob()
    local = threading.local()
    google = oauth.GoogleConfig.from_env()
    states = oauth.StateStore()
    login_limit = RateLimiter(limit=10, window=300.0)
    signup_limit = RateLimiter(
        limit=int(os.environ.get("FACET_SIGNUP_LIMIT", "10")), window=3600.0)
    upload_dir = Path(upload_root) if upload_root else uploads.default_upload_root()

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        """Headers a hosted deployment needs and a localhost tool never did.

        The framing rules matter most: an attacker who can put this app in an invisible
        iframe can trick a signed-in user into rating faces, and every click here is a
        judgement recorded against a real person's photograph.
        """
        resp = await call_next(request)
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("X-Frame-Options", "DENY")
        resp.headers.setdefault("Referrer-Policy", "no-referrer")
        resp.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' data: blob:; "
            "style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; "
            "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; "
            "form-action 'self'")
        resp.headers.setdefault("Cross-Origin-Opener-Policy", "same-origin")
        return resp

    def storage(user: str) -> uploads.UserStorage:
        return uploads.UserStorage(upload_dir, user)

    def client_ip(request: Request) -> str:
        """Best-effort caller address.

        X-Forwarded-For is only consulted when the operator says a proxy is in front,
        because a client can otherwise set it themselves and make the rate limiter
        keyless.
        """
        if os.environ.get("FACET_TRUST_PROXY", "").lower() in ("1", "true", "yes", "on"):
            fwd = request.headers.get("x-forwarded-for", "")
            if fwd:
                return fwd.split(",")[0].strip()
        return request.client.host if request.client else "?"

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

    def open_mode_allowed() -> bool:
        """Whether a browser with no session may act as the shared 'default' profile.

        Convenient on a laptop, indefensible on a host: it would hand anyone who finds the
        URL a signed-in session over other people's photographs. So it switches itself off
        as soon as the deployment looks hosted - a configured public URL - and an operator
        can force it either way.
        """
        forced = os.environ.get("FACET_REQUIRE_AUTH", "").strip().lower()
        if forced in ("1", "true", "yes", "on"):
            return False
        if forced in ("0", "false", "no", "off"):
            return True
        return not bool(os.environ.get("FACET_PUBLIC_URL", "").strip())

    def token_from(authorization: str | None) -> str | None:
        if authorization and authorization.lower().startswith("bearer "):
            return authorization[7:].strip()
        return None

    def current_user(authorization: str | None = Header(default=None)) -> str:
        """Resolve the caller.

        Until the first account is created a *local* install runs open, under the shared
        'default' user, so a fresh checkout works without a login step. The moment anyone
        registers - or the moment the deployment is hosted - authentication becomes
        mandatory, because from then on there are separate libraries and taste models to
        keep apart.
        """
        acc = accounts()
        if acc.count() == 0 and open_mode_allowed():
            return "default"
        user = acc.resolve(token_from(authorization))
        if user is None:
            raise HTTPException(401, "sign in to continue")
        return user

    # ------------------------------------------------------------------ auth

    @app.get("/api/auth/status")
    def auth_status(authorization: str | None = Header(default=None)):
        acc = accounts()
        user = acc.resolve(token_from(authorization))
        open_mode = acc.count() == 0 and open_mode_allowed()
        return {"accounts_exist": acc.count() > 0, "authenticated": user is not None,
                "user": user, "open_mode": open_mode,
                "is_admin": bool(user and acc.is_admin(user)),
                "must_change_password": bool(user and acc.must_change(user)),
                "google": {"enabled": google.enabled(), "reason": google.why_disabled()},
                # Named so "forgot password" has somewhere to point: there is no mail server
                # here, so a reset means asking a person.
                "admins": acc.admins(),
                "note": ("No accounts yet - running in open mode under a shared profile. "
                         "Create an account to get a library and taste model of your own."
                         if open_mode else "")}

    @app.post("/api/auth/register")
    def register(c: Credentials, request: Request):
        acc = accounts()
        signup_limit.check(client_ip(request))
        if not valid_username(c.username):
            raise HTTPException(400, "username must be 2-32 chars: letters, digits, . _ -")
        if len(c.password) < 8:
            raise HTTPException(400, "password must be at least 8 characters")
        if acc.exists(c.username):
            raise HTTPException(409, "that username is taken")
        first = acc.count() == 0
        acc.create(c.username, c.password, c.display_name)
        if first and open_mode_allowed():
            # Carry the open-mode profile over so a first-run user does not lose the
            # preferences and library they built before creating an account.
            for t, col in (("feedback", "user"), ("reference_faces", "user"),
                           ("saved_searches", "user")):
                db().conn.execute(f"UPDATE {t} SET {col}=? WHERE {col} IN ('default','')",
                                  (c.username,))
            db().conn.execute("UPDATE images SET owner=? WHERE owner IN ('default','')",
                              (c.username,))
            db().conn.commit()
        return {"ok": True, "token": acc.start_session(c.username), "user": c.username,
                "migrated_default_profile": first}

    @app.post("/api/auth/login")
    def login(c: Credentials, request: Request):
        acc = accounts()
        ip = client_ip(request)
        login_limit.check(ip, c.username)
        if not acc.check(c.username, c.password):
            raise HTTPException(401, "wrong username or password")
        login_limit.clear(ip, c.username)
        acc.touch(c.username)
        return {"ok": True, "token": acc.start_session(c.username), "user": c.username}

    # ------------------------------------------------------- google sign-in

    @app.get("/api/auth/google/start")
    def google_start(next: str = Query("/")):
        if not google.enabled():
            raise HTTPException(503, google.why_disabled() or "Google sign-in is not set up")
        # Only same-origin destinations: an open redirect on a sign-in route is how a
        # phishing page borrows a real domain's credibility.
        dest = next if next.startswith("/") and not next.startswith("//") else "/"
        state, nonce = states.issue(dest)
        return RedirectResponse(oauth.authorize_url(google, state, nonce), status_code=302)

    @app.get("/api/auth/google/callback")
    def google_callback(code: str | None = Query(None), state: str | None = Query(None),
                        error: str | None = Query(None)):
        if not google.enabled():
            raise HTTPException(503, "Google sign-in is not set up")
        if error:
            return _signin_redirect(error=f"Google sign-in was cancelled ({error})")
        got = states.take(state or "")
        if got is None:
            return _signin_redirect(
                error="that sign-in link has expired - please try again")
        nonce, dest = got
        if not code:
            return _signin_redirect(error="Google did not return an authorization code")
        try:
            tok = oauth.exchange_code(google, code)
            claims = oauth.decode_id_token(tok.get("id_token", ""))
            ident = oauth.verify_claims(claims, google, nonce)
        except oauth.OAuthError as e:
            return _signin_redirect(error=str(e))
        except Exception as e:  # noqa: BLE001 - network and parse failures both land here
            return _signin_redirect(error=f"could not complete sign-in: {type(e).__name__}")

        acc = accounts()
        username, created = acc.upsert_federated(
            "google", ident["sub"], ident["email"], ident["name"], ident["picture"])
        token = acc.start_session(username)
        # The token rides in the fragment, which browsers do not send to servers and do not
        # write to the Referer header - so it does not end up in access logs the way a
        # query parameter would.
        return RedirectResponse(
            f"{dest}#token={token}&user={username}&new={'1' if created else '0'}",
            status_code=302)

    def _signin_redirect(error: str) -> RedirectResponse:
        from urllib.parse import quote
        return RedirectResponse(f"/#error={quote(error)}", status_code=302)

    @app.post("/api/auth/logout")
    def logout(authorization: str | None = Header(default=None)):
        if authorization and authorization.lower().startswith("bearer "):
            accounts().end_session(authorization[7:].strip())
        return {"ok": True}

    @app.get("/api/auth/me")
    def me(user: str = Depends(current_user)):
        acc = accounts()
        st = storage(user) if user else None
        used = st.used_bytes() if st else 0
        return {"user": user, "is_admin": acc.is_admin(user),
                "must_change_password": acc.must_change(user),
                "storage": {"used_bytes": used, "quota_bytes": uploads.DEFAULT_QUOTA_BYTES,
                            "used_pct": round(100 * used / max(1, uploads.DEFAULT_QUOTA_BYTES), 1)},
                "server_paths_allowed": uploads.server_paths_allowed(),
                **acc.info(user)}

    @app.post("/api/auth/password")
    def change_password(req: PasswordChange, user: str = Depends(current_user)):
        """Change your own password, or - as an admin - reset somebody else's."""
        acc = accounts()
        target = req.username or user
        if acc.info(target).get("provider", "password") != "password":
            raise HTTPException(
                400, f"{target} signs in with Google - there is no password to change")
        if len(req.new_password) < 8:
            raise HTTPException(400, "password must be at least 8 characters")
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

    # ---------------------------------------------------- your data, your call

    @app.get("/api/account/export")
    def export_account(user: str = Depends(current_user)):
        """Everything this service holds about you, in one JSON document.

        Not a nicety. Embeddings derived from photographs are biometric data
        (docs/LICENSING.md 4), and a hosted service that cannot answer "what do you have on
        me" has no defensible answer to the question either.
        """
        ix = db()
        return {
            "account": accounts().info(user),
            "generated_at": time.time(),
            "library": [dict(r) for r in ix.conn.execute(
                "SELECT id, path, source, origin_url, width, height, status, n_faces, "
                "indexed_at FROM images WHERE owner=?", (user,))],
            "judgements": [dict(r) for r in ix.conn.execute(
                "SELECT face_id, kind, created_at, commit_at FROM feedback WHERE user=?",
                (user,))],
            "reference_faces": [{"id": r["id"], "path": r["path"], "kind": r["kind"]}
                                for r in ix.references(user)],
            "saved_searches": ix.list_saved_searches(user),
            "storage": {"used_bytes": storage(user).used_bytes()},
            "note": ("Face embeddings are not included: they are large binary vectors and "
                     "are deleted with the face rows above. Delete your account to erase "
                     "everything listed here."),
        }

    @app.post("/api/account/delete")
    def delete_account(req: DeleteAccountRequest, user: str = Depends(current_user)):
        """Erase the account, the uploads, the derived embeddings and the taste model."""
        acc = accounts()
        if req.confirm.strip().lower() != "delete":
            raise HTTPException(400, "type 'delete' to confirm")
        if acc.info(user).get("provider", "password") == "password":
            if not (req.password and acc.check(user, req.password)):
                raise HTTPException(401, "password is wrong")
        if acc.is_admin(user) and len(acc.admins()) == 1:
            raise HTTPException(
                400, "you are the only administrator - promote somebody else first")
        counts = db().delete_owner_data(user)
        db().bump_generation()
        counts["files"] = storage(user).purge()
        acc.delete(user)
        return {"ok": True, "deleted": counts,
                "note": "account, library, embeddings and taste model are gone"}

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
                (r["username"], time.time())).fetchone()[0]
            r["images"] = ix.conn.execute(
                "SELECT COUNT(*) FROM images WHERE owner=?", (r["username"],)).fetchone()[0]
            r["storage_bytes"] = storage(r["username"]).used_bytes()
        return {"users": rows, "you": user,
                "note": "Counts only. An administrator cannot open another account's "
                        "library or see their photographs - the search path is scoped to "
                        "the session, not to a role."}

    @app.post("/api/admin/users")
    def admin_create(c: Credentials, user: str = Depends(current_user)):
        require_admin(user)
        acc = accounts()
        if not valid_username(c.username):
            raise HTTPException(400, "username must be 2-32 chars: letters, digits, . _ -")
        if len(c.password) < 8:
            raise HTTPException(400, "password must be at least 8 characters")
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
        # Deleting an account removes the biometric-derived data it accumulated, per
        # docs/LICENSING.md section 4.2: deletion must actually delete. That now includes
        # the uploaded originals and every image row they produced.
        counts = db().delete_owner_data(username)
        db().bump_generation()
        counts["files"] = storage(username).purge()
        acc.delete(username)
        return {"ok": True, "deleted": counts,
                "note": "account, library, embeddings and taste model removed"}

    @app.get("/api/admin/overview")
    def admin_overview(user: str = Depends(current_user)):
        require_admin(user)
        ix, acc = db(), accounts()
        st = ix.stats()                      # whole index: this is the operator's view
        st["users"] = acc.count()
        st["by_owner"] = [dict(r) for r in ix.conn.execute(
            "SELECT owner, COUNT(*) images, SUM(n_faces) faces FROM images "
            "GROUP BY owner ORDER BY images DESC")]
        st["google_signin"] = {"enabled": google.enabled(), "reason": google.why_disabled()}
        st["open_mode_allowed"] = open_mode_allowed()
        st["server_paths_allowed"] = uploads.server_paths_allowed()
        st["upload_dir"] = str(upload_dir)
        st["running_imports"] = job.any_running()
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
        s = db().stats(owner=user)
        s["ready"] = s["faces"] > 0
        s["has_predictions"] = s["predictions"] > 0
        s["beauty_head_trained"] = (
            Path(__file__).resolve().parents[3] / "artifacts/models/beauty_head.npz").exists()
        s["saved_searches"] = len(db().list_saved_searches(user))
        s["feedback"] = db().conn.execute(
            "SELECT COUNT(*) FROM feedback WHERE user=?", (user,)).fetchone()[0]
        s["runs"] = [dict(r) for r in db().conn.execute(
            "SELECT id, started_at, finished_at, n_indexed, n_faces, n_failed, status "
            "FROM runs WHERE owner=? ORDER BY id DESC LIMIT 5", (user,))]
        s["storage"] = {"used_bytes": storage(user).used_bytes(),
                        "quota_bytes": uploads.DEFAULT_QUOTA_BYTES}
        return s

    @app.get("/api/models")
    def models(user: str = Depends(current_user)):
        """What is actually loaded, and what each piece was decided by.

        A ranking that cannot say which models produced it is not auditable, and every
        number here is an estimate whose provenance matters (RESEARCH.md 11.4).
        """
        from ..models.beauty_head import BeautyHead
        from ..pipeline.indexer import IndexConfig

        cfg = IndexConfig()
        ix = db()
        row = ix.conn.execute(
            "SELECT f.encoder_version, f.crop_version, i.detector_version, COUNT(*) n "
            "FROM faces f JOIN images i ON i.id=f.image_id WHERE i.owner=? "
            "GROUP BY 1,2,3 ORDER BY n DESC LIMIT 1", (user,)).fetchone()
        head_path = Path(__file__).resolve().parents[3] / "artifacts/models/beauty_head.npz"
        head = None
        if head_path.exists():
            try:
                h = BeautyHead.load(head_path)
                head = {"version": h.version, "metrics": h.metrics,
                        "members": h.metrics.get("n_members"),
                        "source": h.__class__.__name__}
            except Exception as e:  # noqa: BLE001
                head = {"error": str(e)}
        pm = engine().preference_model(user)
        return {
            "detector": {"name": "SCRFD (buffalo_l)",
                         "version": row["detector_version"] if row else None,
                         "det_size": cfg.det_size, "pad": cfg.pad_frac,
                         "decided_by": "E8 — both axes adaptive; the best scene config has "
                                       "zero recall on cropped portraits"},
            "encoder": {"name": "ArcFace R50 + CLIP ViT-B/32",
                        "version": row["encoder_version"] if row else None,
                        "crop": row["crop_version"] if row else cfg.crop_version,
                        "decided_by": "exp001/E5 — frozen features beat fine-tuned CNNs; "
                                      "crop margin 0.25 chosen on transfer, not in-benchmark"},
            "attractiveness": {"name": "LDL ensemble + split-conformal", "head": head,
                               "trained_on": "SCUT-FBP5500 (60 raters, aged 18-27, 2017)",
                               "decided_by": "E6 — objectives tie on accuracy; LDL wins on "
                                             "what it reports"},
            "age_gender": {"name": "MiVOLO v2 (Apache-2.0)", "policy": "lazy, quality-gated",
                           "decided_by": "E4 — ~3x fairer than the retired baseline, "
                                         "~190x slower"},
            "quality": {"name": "composite_v2",
                        "decided_by": "E9 — validated by Error-vs-Reject on LFW"},
            "your_model": ({"trained": True, "method": pm.status().method,
                            "alpha": pm.status().alpha, "likes": pm.status().n_likes,
                            "dislikes": pm.status().n_dislikes,
                            "references": pm.status().n_references,
                            "note": pm.status().note}
                           if pm else {"trained": False,
                                       "note": "Not taught yet — rate results or add "
                                               "reference faces."}),
            "faces_indexed": row["n"] if row else 0,
        }

    # ---------------------------------------------------------------- search

    @app.post("/api/search")
    def search(spec: dict[str, Any], user: str = Depends(current_user)):
        # The caller does not get to choose whose profile is used: personalisation and the
        # dislike filter both read per-user rows, so a client-supplied name would be a way
        # to read someone else's taste.
        spec = dict(spec)
        spec["personalisation"] = {**(spec.get("personalisation") or {}), "user": user}
        spec["owner"] = user
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
            "SELECT f.*, i.path, i.width, i.height, i.owner FROM faces f "
            "JOIN images i ON i.id=f.image_id WHERE f.id=? AND i.owner=?",
            (face_id, user)).fetchone()
        # A face in somebody else's library is reported as absent rather than forbidden:
        # a 403 would confirm the id exists, which is itself a fact about their library.
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
        if acc.count() == 0 and open_mode_allowed():
            return "default"
        u = acc.resolve(t or token_from(authorization))
        if u is None:
            raise HTTPException(401, "sign in to view images")
        return u

    @app.get("/api/image/{image_id}")
    def image(image_id: int, max_px: int = Query(1600, ge=64, le=4096),
              user: str = Depends(media_user)):
        r = db().conn.execute("SELECT path FROM images WHERE id=? AND owner=?",
                              (image_id, user)).fetchone()
        if r is None:
            raise HTTPException(404, "no such image")
        return _encode(_load_scaled(r["path"], max_px))

    @app.get("/api/crop/{face_id}")
    def crop(face_id: int, size: int = Query(256, ge=32, le=1024), margin: float = 0.4,
             user: str = Depends(media_user)):
        import cv2
        r = db().conn.execute(
            "SELECT f.x1,f.y1,f.x2,f.y2,i.path FROM faces f JOIN images i ON i.id=f.image_id "
            "WHERE f.id=? AND i.owner=?", (face_id, user)).fetchone()
        if r is None:
            raise HTTPException(404, "no such face")
        from ..models.insightface_backend import crop_bbox
        img = cv2.imread(r["path"])
        if img is None:
            raise HTTPException(410, "source image is no longer readable")
        return _encode(crop_bbox(img, (r["x1"], r["y1"], r["x2"], r["y2"]),
                                 size=size, margin=margin))

    # --------------------------------------------------------------- indexing

    def _start_index(user: str, roots: list[str], req: IndexRequest | None = None,
                     source: str = "local", note: str = ""):
        r = req or IndexRequest(roots=roots)
        return job.start(index_path, features_dir, roots, r.limit, r.force,
                         r.predict_age, r.min_quality, r.age_limit,
                         owner=user, source=source, note=note)

    @app.post("/api/index")
    def start_index(req: IndexRequest, user: str = Depends(current_user)):
        """Index a directory that already exists on the server.

        Off by default. On a shared host this is an arbitrary filesystem read for anyone
        with an account, and no amount of path checking makes "index /home" safe when
        /home is not yours. Self-hosters set FACET_ALLOW_SERVER_PATHS=1 and get the
        original behaviour back.
        """
        if not uploads.server_paths_allowed():
            raise HTTPException(
                403, "indexing server directories is disabled on this deployment - "
                     "upload your images instead (POST /api/library/upload)")
        for r in req.roots:
            if not Path(r).expanduser().is_dir():
                raise HTTPException(400, f"not a directory: {r}")
        return _start_index(user, [str(Path(r).expanduser()) for r in req.roots], req,
                            source="local", note=f"{len(req.roots)} server folder(s)")

    @app.get("/api/index/status")
    def index_status(user: str = Depends(current_user)):
        s = job.state_for(user)
        s["running"] = job.running(user)
        return s

    # ---------------------------------------------------------------- library

    @app.post("/api/library/upload")
    async def upload_library(files: list[UploadFile] = File(...),
                             user: str = Depends(current_user)):
        """Take images from the browser into this account's library, then index them.

        The whole batch lands in one directory named after the account, and indexing is
        pointed at that directory rather than at whatever the client asked for. There is
        no request field that can widen the scope.
        """
        stored, errors = await _store_uploads(files, user, "library")
        if not stored:
            raise HTTPException(400, {"message": "nothing could be imported",
                                      "errors": errors[:20]})
        _start_index(user, [str(storage(user).library)], source="upload",
                     note=f"{len(stored)} uploaded image(s)")
        return {"ok": True, "accepted": len(stored), "errors": errors[:20],
                "note": "indexing started - watch /api/index/status"}

    @app.post("/api/library/url")
    def import_urls(req: UrlImportRequest, user: str = Depends(current_user)):
        """Import images by URL.

        Each URL is validated before the fetch and again after every redirect
        (uploads.check_url), because the server can reach networks the caller cannot.
        """
        if not req.urls:
            raise HTTPException(400, "no URLs given")
        if len(req.urls) > 50:
            raise HTTPException(400, "50 URLs at a time, please")
        target = "references" if req.target == "reference" else "library"
        st = storage(user)
        got, errors = [], []
        for raw in req.urls:
            url = (raw or "").strip()
            if not url:
                continue
            try:
                data, final = uploads.fetch_url(url)
                got.append(st.save(data, Path(final).name or "image", target, final))
            except uploads.UploadError as e:
                errors.append({"url": url[:200], "error": str(e)})
            except Exception as e:  # noqa: BLE001 - timeouts, DNS, TLS
                errors.append({"url": url[:200], "error": f"{type(e).__name__}"})
        if not got:
            raise HTTPException(400, {"message": "nothing could be fetched",
                                      "errors": errors[:20]})
        if target == "library":
            _start_index(user, [str(st.library)], source="url",
                         note=f"{len(got)} image(s) from URLs")
            return {"ok": True, "accepted": len(got), "errors": errors,
                    "note": "indexing started - watch /api/index/status"}
        added = _add_reference_files([g.path for g in got], req.kind, user)
        return {"ok": True, "accepted": len(got), "errors": errors, **added}

    @app.get("/api/library")
    def library(user: str = Depends(current_user), limit: int = Query(60, ge=1, le=500)):
        ix, st = db(), storage(user)
        rows = [dict(r) for r in ix.conn.execute(
            "SELECT id, path, source, origin_url, status, n_faces, width, height, "
            "size_bytes, indexed_at FROM images WHERE owner=? ORDER BY indexed_at DESC "
            "LIMIT ?", (user, limit))]
        for r in rows:
            r["name"] = Path(r["path"]).name
        used = st.used_bytes()
        return {"images": rows,
                "total": ix.conn.execute("SELECT COUNT(*) FROM images WHERE owner=?",
                                         (user,)).fetchone()[0],
                "storage": {"used_bytes": used,
                            "quota_bytes": uploads.DEFAULT_QUOTA_BYTES,
                            "used_pct": round(100 * used / max(1, uploads.DEFAULT_QUOTA_BYTES), 1)}}

    @app.delete("/api/library/{image_id}")
    def delete_image(image_id: int, user: str = Depends(current_user)):
        ix = db()
        r = ix.conn.execute("SELECT path, source FROM images WHERE id=? AND owner=?",
                            (image_id, user)).fetchone()
        if r is None:
            raise HTTPException(404, "no such image")
        ix.conn.execute("DELETE FROM faces WHERE image_id=?", (image_id,))
        ix.conn.execute("DELETE FROM images WHERE id=?", (image_id,))
        ix.conn.commit()
        ix.bump_generation()
        removed = storage(user).remove(r["path"]) if r["source"] in ("upload", "url") else False
        return {"ok": True, "file_deleted": removed}

    @app.delete("/api/library")
    def clear_library(user: str = Depends(current_user)):
        ix = db()
        n = ix.conn.execute("SELECT COUNT(*) FROM images WHERE owner=?",
                            (user,)).fetchone()[0]
        ix.conn.execute("DELETE FROM faces WHERE image_id IN "
                        "(SELECT id FROM images WHERE owner=?)", (user,))
        ix.conn.execute("DELETE FROM images WHERE owner=?", (user,))
        ix.conn.commit()
        ix.bump_generation()
        files = storage(user).purge()
        engine().invalidate_percentiles(user)
        return {"ok": True, "images_removed": n, "files_removed": files}

    async def _store_uploads(files, user: str, kind: str):
        st = storage(user)
        if len(files) > uploads.MAX_FILES_PER_REQUEST:
            raise HTTPException(
                400, f"{uploads.MAX_FILES_PER_REQUEST} files at a time, please")
        stored, errors = [], []
        for f in files:
            try:
                data = await f.read()
                stored.append(st.save(data, f.filename or "image", kind))
            except uploads.UploadError as e:
                errors.append({"file": uploads.safe_name(f.filename or ""), "error": str(e)})
            finally:
                await f.close()
        return stored, errors

    # --------------------------------------------------------- feedback / saved

    @app.post("/api/feedback")
    def feedback(req: FeedbackRequest, user: str = Depends(current_user)):
        """Record a judgement, and close the previous one's undo window.

        Exactly one judgement is reversible at a time - the one you just made. Marking
        anything else commits the previous one on the spot, so the model is at most one
        judgement behind what the user has said, and "undo" always means the obvious thing
        rather than the head of a queue.
        """
        if req.kind not in {"like", "dislike", "hide", "wrong"}:
            raise HTTPException(400, "kind must be like, dislike, hide or wrong")
        ix = db()
        if ix.owner_of_face(req.face_id) != user:
            raise HTTPException(404, "no such face")
        if req.remove:
            ix.remove_feedback(req.face_id, req.kind, user)
            return {"ok": True, "feedback": ix.feedback_for(user).get(req.face_id, []),
                    "undo": None}
        prev = ix.active_undo(user)
        secs = float(min(max(req.undo_seconds, 0.0), 120.0))
        commit_at = ix.add_feedback(req.face_id, req.kind, user, req.note,
                                    undo_seconds=secs)
        committed = prev["face_id"] if prev and prev["face_id"] != req.face_id else None
        return {"ok": True, "feedback": ix.feedback_for(user).get(req.face_id, []),
                "undo": {"face_id": req.face_id, "kind": req.kind,
                         "commit_at": commit_at, "seconds": secs} if secs > 0 else None,
                "committed_face_id": committed,
                "note": ("hidden from results now; it starts shaping your taste model in "
                         f"{secs:g}s unless you undo" if secs > 0
                         else "applied to your taste model immediately")}

    @app.post("/api/feedback/undo")
    def undo(req: UndoRequest, user: str = Depends(current_user)):
        """Reverse a judgement: the face returns to results and, if still inside the undo
        window, the preference model never saw it."""
        ix = db()
        if ix.owner_of_face(req.face_id) != user:
            raise HTTPException(404, "no such face")
        n = ix.undo_feedback(req.face_id, user, req.kind)
        return {"ok": True, "removed": n,
                "feedback": ix.feedback_for(user).get(req.face_id, [])}

    @app.post("/api/feedback/commit")
    def commit_feedback(user: str = Depends(current_user)):
        """Close the open undo window now. The client calls this when the card scrolls
        away or the tab is closing, so a judgement is never left half-made."""
        return {"ok": True, "committed": db().commit_pending(user)}

    @app.get("/api/feedback/undo")
    def active_undo(user: str = Depends(current_user)):
        return {"undo": db().active_undo(user)}

    @app.get("/api/feedback/pending")
    def pending(user: str = Depends(current_user)):
        """Deprecated alias for /api/feedback/undo. There is only ever one now."""
        u = db().active_undo(user)
        return {"pending": [u] if u else []}

    @app.get("/api/searches")
    def list_searches(user: str = Depends(current_user)):
        return db().list_saved_searches(user)

    @app.post("/api/searches")
    def save_search(req: SaveSearchRequest, user: str = Depends(current_user)):
        if not req.name.strip():
            raise HTTPException(400, "give the search a name")
        db().save_search(req.name.strip()[:64], req.spec, user)
        return {"ok": True}

    @app.delete("/api/searches/{name}")
    def delete_search(name: str, user: str = Depends(current_user)):
        db().delete_saved_search(name, user)
        return {"ok": True}

    # -------------------------------------------------------- personalisation

    _enc: dict = {}
    _enc_lock = threading.Lock()

    def encoder():
        """Detector + embedder for reference images, loaded once and reused.

        Reference faces must be encoded exactly as indexed faces were, or the preference
        model would be comparing vectors from different preprocessing - the same class of
        error the versioned feature shards exist to prevent.

        Two things this got wrong before, both of which surfaced as a bare 500 on the
        "add taste samples" button:

        * **No lock.** FastAPI runs sync endpoints in a threadpool, so two clicks loaded
          two full copies of SCRFD, ArcFace and CLIP at once.
        * **GPU-only.** These models were loaded onto whatever the indexer had left, and a
          busy card meant a CUDA allocation failure with no message a user could act on.
          A handful of reference images does not need a GPU, so a failed CUDA load now
          retries on CPU instead of failing the request.
        """
        with _enc_lock:
            if _enc:
                return _enc
            from ..models.insightface_backend import (
                ArcFaceEmbedder, InsightFaceDetector, align_to_template)
            from ..pipeline.indexer import IndexConfig
            cfg = IndexConfig()
            errors = []
            for use_gpu in (cfg.use_gpu, False) if cfg.use_gpu else (False,):
                try:
                    e = {"cfg": cfg, "align": align_to_template,
                         "det": InsightFaceDetector(pack=cfg.pack, det_size=cfg.det_size,
                                                    pad_frac=cfg.pad_frac, use_gpu=use_gpu),
                         "arc": ArcFaceEmbedder(pack=cfg.pack, use_gpu=use_gpu)}
                    if cfg.clip:
                        from ..models.clip_backend import ClipEmbedder
                        e["clip"] = ClipEmbedder(use_gpu=use_gpu)
                    e["device"] = "gpu" if use_gpu else "cpu"
                    _enc.update(e)
                    return _enc
                except Exception as ex:  # noqa: BLE001 - retry on CPU, then report
                    errors.append(f"{'gpu' if use_gpu else 'cpu'}: {type(ex).__name__}: {ex}")
            raise HTTPException(
                503, "could not load the face encoder needed to read reference images - "
                     + " | ".join(errors)[:400])

    def _add_reference_files(files, kind: str, user: str) -> dict:
        """Encode reference images and store them against this account.

        Failures are per-file and reported, never fatal: one unreadable photo in a batch of
        twenty should not lose the other nineteen.
        """
        import cv2

        if kind not in {"like", "dislike"}:
            raise HTTPException(400, "kind must be like or dislike")
        e = encoder()
        cfg = e["cfg"]
        added, skipped = 0, []
        for f in files:
            f = Path(f)
            try:
                img = cv2.imread(str(f))
                if img is None:
                    skipped.append({"path": f.name, "reason": "could not read"})
                    continue
                dets = e["det"].detect(img)
                if not dets:
                    skipped.append({"path": f.name, "reason": "no face detected"})
                    continue
                crop = e["align"](img, dets[0].keypoints, size=cfg.crop_size,
                                  margin=cfg.crop_margin)
                vec = e["arc"].encode(np.stack([crop]))[0]
                if "clip" in e:
                    vec = np.concatenate([vec, e["clip"].encode(np.stack([crop]))[0]])
                db().add_reference(str(f), 0, vec, kind, user)
                added += 1
            except Exception as ex:  # noqa: BLE001 - one bad file must not lose the batch
                skipped.append({"path": f.name, "reason": f"{type(ex).__name__}"})
        return {"added": added, "skipped": skipped,
                "total_references": len(db().references(user)),
                "device": e.get("device")}

    @app.get("/api/preference")
    def preference(user: str = Depends(current_user)):
        eng = engine()
        pm = eng.preference_model(user)
        refs = [{"id": r["id"], "kind": r["kind"], "name": _display_name(r["path"])}
                for r in db().references(user)]
        # The stored path is deliberately not returned. It is a server filesystem location,
        # and for a self-hosted install it can name a directory outside the app entirely.
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
                           "rate results with \u2605 / Not for me.")
            out["alpha"] = 0.0
        return out

    @app.post("/api/preference/upload")
    async def upload_references(files: list[UploadFile] = File(...),
                                kind: str = Form("like"),
                                user: str = Depends(current_user)):
        """Upload example faces that define this user's taste.

        Kept separate from the library on purpose: these are idealised examples, they are
        never returned as search results, and the preference model weights them differently
        from judgements made on real candidates (models/preference.py).
        """
        stored, errors = await _store_uploads(files, user, "references")
        if not stored:
            raise HTTPException(400, {"message": "nothing could be read",
                                      "errors": errors[:20]})
        out = _add_reference_files([g.path for g in stored], kind, user)
        out["errors"] = errors[:20]
        return out

    @app.post("/api/preference/references")
    def add_references(req: ReferenceRequest, user: str = Depends(current_user)):
        if not uploads.server_paths_allowed():
            raise HTTPException(
                403, "reading images from server paths is disabled on this deployment - "
                     "upload them instead (POST /api/preference/upload)")
        if req.kind not in {"like", "dislike"}:
            raise HTTPException(400, "kind must be like or dislike")
        files = []
        for raw in req.paths:
            path = Path(raw).expanduser()
            if path.is_dir():
                files += [p for p in sorted(path.iterdir())
                          if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}]
            else:
                files.append(path)
        if not files:
            raise HTTPException(400, "no images found at those paths")
        return _add_reference_files(files, req.kind, user)

    @app.get("/api/preference/thumb/{ref_id}")
    def reference_thumb(ref_id: int, size: int = Query(128, ge=32, le=512),
                        user: str = Depends(media_user)):
        """The reference image itself, so the taste panel can show faces rather than
        filenames. Scoped to the owner like every other media route."""
        r = db().conn.execute(
            "SELECT path FROM reference_faces WHERE id=? AND user=?",
            (ref_id, user)).fetchone()
        if r is None:
            raise HTTPException(404, "no such reference")
        return _encode(_load_scaled(r["path"], size))

    @app.delete("/api/preference/references/{ref_id}")
    def delete_reference(ref_id: int, user: str = Depends(current_user)):
        row = db().conn.execute(
            "SELECT path FROM reference_faces WHERE id=? AND user=?",
            (ref_id, user)).fetchone()
        db().delete_reference(ref_id, user)
        if row:
            storage(user).remove(row["path"])
        return {"ok": True}

    @app.post("/api/preference/reset")
    def reset_preference(user: str = Depends(current_user)):
        st = storage(user)
        for r in db().references(user):
            st.remove(r["path"])
        n = db().clear_references(user)
        db().conn.execute("DELETE FROM feedback WHERE user=? AND kind IN ('like','dislike')",
                          (user,))
        db().conn.commit()
        return {"ok": True, "cleared_references": n,
                "note": "your taste model is empty; ranking falls back to the population "
                        "model until you teach it again"}

    # ----------------------------------------------------------------- export

    @app.post("/api/export")
    def export(spec: dict[str, Any], fmt: str = Query("csv", pattern="^(csv|json)$"),
               user: str = Depends(current_user)):
        spec = {**spec, "owner": user,
                "personalisation": {**(spec.get("personalisation") or {}), "user": user}}
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


def _display_name(path: str) -> str:
    """A label a person can read.

    Uploaded files are stored under their content hash - which is what makes re-uploading
    the same photo a no-op and keeps client filenames off the filesystem - but "sample" beats
    64 hex characters in a tooltip.
    """
    name = Path(path).name
    stem = Path(name).stem
    if len(stem) == 64 and all(c in "0123456789abcdef" for c in stem):
        return "uploaded sample"
    return name


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
