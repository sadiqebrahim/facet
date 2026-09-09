"""SQLite index for images, faces, predictions and runs.

Chosen over a server database for the reasons in docs/RESEARCH.md 15.3: single file, zero
ops, transactional, trivially backed up, and more than adequate for millions of faces. The
schema encodes several decisions the research phase forced:

* **Versioning on every row.** `detector_version`, `encoder_version`, `crop_version` and
  `config_hash` are stored per record, not globally. E8 found a detection cache keyed only on
  dataset+pack would silently reuse stale boxes after a detector change; the same hazard
  applies to every stage. Versioned rows are what make "reprocess only what changed" safe.
* **Faces are separate from images.** One image may hold many faces, and a face is the unit
  everything downstream ranks.
* **Predictions are separate from faces.** The encode/predict split (15.1) means heads get
  swapped and retrained often while embeddings stay valid, so predictions must be replaceable
  without touching the face row or the feature store.
* **Failures are recorded, not dropped.** A corrupt file or an image with no detectable face
  is a row with a status, because 13.2's "a missed face is an invisible failure" applies just
  as much to a file we could not open.
"""
from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

SCHEMA_VERSION = 2

SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS images (
    id            INTEGER PRIMARY KEY,
    -- Whose library this image belongs to. There is no shared pool: a hosted deployment
    -- has strangers on it, and an image one person uploaded must never appear in another
    -- person's results. Every read path joins through here (docs/HOSTING.md 2).
    owner         TEXT NOT NULL DEFAULT '',
    path          TEXT NOT NULL,
    content_hash  TEXT,
    size_bytes    INTEGER,
    mtime         REAL,
    width         INTEGER,
    height        INTEGER,
    -- ok | no_faces | corrupt | unreadable
    status        TEXT NOT NULL DEFAULT 'pending',
    error         TEXT,
    n_faces       INTEGER NOT NULL DEFAULT 0,
    detector_version TEXT,
    indexed_at    REAL,
    -- upload | url | local. Uploaded bytes live under the owner's own directory and are
    -- deleted with the account; `local` only exists for a self-hosted install pointed at
    -- a directory the operator already had.
    source        TEXT NOT NULL DEFAULT 'local',
    origin_url    TEXT,
    -- Two people can upload the same photo; the same person cannot hold it twice.
    UNIQUE(owner, path)
);
CREATE INDEX IF NOT EXISTS idx_images_status ON images(status);
CREATE INDEX IF NOT EXISTS idx_images_hash   ON images(content_hash);
-- idx_images_owner is created in _migrate(), not here: this script runs before the
-- migration, and on a pre-multi-tenancy index there is no `owner` column yet for it to
-- index. Creating it there covers the fresh case and the upgraded one alike.

CREATE TABLE IF NOT EXISTS faces (
    id            INTEGER PRIMARY KEY,
    image_id      INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    face_idx      INTEGER NOT NULL,
    x1 REAL, y1 REAL, x2 REAL, y2 REAL,
    det_score     REAL,
    face_px       REAL,
    kps           BLOB,
    quality       REAL,
    quality_json  TEXT,
    -- row offset into the feature store shard identified by encoder/crop version
    feature_row      INTEGER,
    encoder_version  TEXT,
    crop_version     TEXT,
    UNIQUE(image_id, face_idx)
);
CREATE INDEX IF NOT EXISTS idx_faces_image   ON faces(image_id);
CREATE INDEX IF NOT EXISTS idx_faces_quality ON faces(quality);

CREATE TABLE IF NOT EXISTS predictions (
    face_id       INTEGER NOT NULL REFERENCES faces(id) ON DELETE CASCADE,
    model         TEXT NOT NULL,
    model_version TEXT NOT NULL,
    config_hash   TEXT,
    value         REAL,
    confidence    REAL,
    std           REAL,
    interval_lo   REAL,
    interval_hi   REAL,
    distribution  TEXT,
    extra         TEXT,
    created_at    REAL,
    PRIMARY KEY (face_id, model)
);
CREATE INDEX IF NOT EXISTS idx_pred_model ON predictions(model, value);

CREATE TABLE IF NOT EXISTS duplicates (
    face_id   INTEGER NOT NULL REFERENCES faces(id) ON DELETE CASCADE,
    group_id  INTEGER NOT NULL,
    kind      TEXT NOT NULL,          -- exact | near
    PRIMARY KEY (face_id, kind)
);
CREATE INDEX IF NOT EXISTS idx_dup_group ON duplicates(group_id);

-- Phase 10 groundwork. E14 found personalisation only pays off when the rater pool is
-- diverse, and then only via a residual model gated on how poorly the population model
-- fits that user - so feedback is stored separately from predictions and never mixed into
-- the general model (RESEARCH.md 15.5).
CREATE TABLE IF NOT EXISTS feedback (
    face_id    INTEGER NOT NULL REFERENCES faces(id) ON DELETE CASCADE,
    user       TEXT NOT NULL DEFAULT 'default',
    kind       TEXT NOT NULL,          -- like | dislike | hide | wrong
    note       TEXT,
    created_at REAL,
    -- A judgement takes effect on the MODEL only once this passes. Until then it is
    -- reversible: the face disappears from results immediately (which is what the user
    -- asked for) but the preference model has not learned from it yet, so an undo leaves
    -- no trace rather than requiring the model to unlearn something.
    commit_at  REAL DEFAULT 0,
    PRIMARY KEY (face_id, user, kind)
);
CREATE INDEX IF NOT EXISTS idx_feedback_user ON feedback(user, kind);

-- Reference faces: examples the user supplies to teach the preference model. Kept apart
-- from `feedback` because they are idealised examples rather than judgements on real
-- candidates, and the model weights them differently.
CREATE TABLE IF NOT EXISTS reference_faces (
    id        INTEGER PRIMARY KEY,
    user      TEXT NOT NULL DEFAULT 'default',
    path      TEXT NOT NULL,
    face_idx  INTEGER NOT NULL DEFAULT 0,
    feature   BLOB NOT NULL,
    dim       INTEGER NOT NULL,
    kind      TEXT NOT NULL DEFAULT 'like',   -- like | dislike
    added_at  REAL,
    UNIQUE(user, path, face_idx)
);
CREATE INDEX IF NOT EXISTS idx_ref_user ON reference_faces(user, kind);

CREATE TABLE IF NOT EXISTS saved_searches (
    id         INTEGER PRIMARY KEY,
    user       TEXT NOT NULL DEFAULT '',
    name       TEXT NOT NULL,
    spec       TEXT NOT NULL,
    created_at REAL,
    UNIQUE(user, name)
);

CREATE TABLE IF NOT EXISTS runs (
    id           INTEGER PRIMARY KEY,
    started_at   REAL,
    finished_at  REAL,
    root         TEXT,
    config_hash  TEXT,
    config       TEXT,
    n_seen       INTEGER DEFAULT 0,
    n_indexed    INTEGER DEFAULT 0,
    n_skipped    INTEGER DEFAULT 0,
    n_failed     INTEGER DEFAULT 0,
    n_faces      INTEGER DEFAULT 0,
    status       TEXT DEFAULT 'running',
    owner        TEXT NOT NULL DEFAULT ''
);
"""


class Index:
    """Thin typed wrapper over the SQLite index."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), timeout=60.0)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self._migrate()
        self.set_meta("schema_version", str(SCHEMA_VERSION))
        self.conn.commit()

    def _migrate(self) -> None:
        """Bring an index created by an earlier version up to the current schema.

        Additive where possible. The one rebuild is `images`, whose old UNIQUE(path)
        constraint made multi-tenancy impossible: two accounts uploading the same photo
        would collide, and worse, a shared path meant a shared row meant a shared library.
        A rebuild is cheap here (one row per image) and the ids are preserved, so every
        `faces.image_id` still resolves.
        """
        self._add_column("feedback", "commit_at", "REAL DEFAULT 0")
        self._add_column("images", "source", "TEXT NOT NULL DEFAULT 'local'")
        self._add_column("images", "origin_url", "TEXT")
        self._add_column("runs", "owner", "TEXT NOT NULL DEFAULT ''")
        self._migrate_images_owner()
        self._migrate_saved_searches()
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_images_owner ON images(owner)")
        self.conn.commit()

    def _add_column(self, table: str, col: str, ddl: str) -> None:
        cols = {r[1] for r in self.conn.execute(f"PRAGMA table_info({table})")}
        if cols and col not in cols:
            self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {ddl}")

    def _legacy_owner(self) -> str:
        """Who the images predating multi-tenancy belong to.

        The earliest administrator, because they are the account that ran the indexing.
        Assigning them to nobody would be safer in the abstract but would silently empty
        the library of the person who built it; assigning them to everybody is the exact
        leak this migration exists to close.
        """
        try:
            r = self.conn.execute(
                "SELECT username FROM users WHERE is_admin=1 ORDER BY created_at LIMIT 1"
            ).fetchone()
            if r:
                return r["username"]
            r = self.conn.execute(
                "SELECT username FROM users ORDER BY created_at LIMIT 1").fetchone()
            if r:
                return r["username"]
        except Exception:      # noqa: BLE001 - no users table yet on a fresh index
            pass
        return ""

    def _has_unique_owner_path(self) -> bool:
        for idx in self.conn.execute("PRAGMA index_list(images)"):
            if not idx["unique"]:
                continue
            cols = [r["name"] for r in
                    self.conn.execute(f"PRAGMA index_info('{idx['name']}')")]
            if cols == ["owner", "path"]:
                return True
        return False

    def _migrate_images_owner(self) -> None:
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(images)")}
        if not cols or ("owner" in cols and self._has_unique_owner_path()):
            return
        owner = self._legacy_owner()
        self._backup("pre-owner-migration")
        carried = ("path", "content_hash", "size_bytes", "mtime", "width", "height",
                   "status", "error", "n_faces", "detector_version", "indexed_at",
                   "source", "origin_url")
        keep = [c for c in carried if c in cols]
        self.conn.execute("PRAGMA foreign_keys=OFF")
        self.conn.executescript("""
            CREATE TABLE images_mig (
                id INTEGER PRIMARY KEY, owner TEXT NOT NULL DEFAULT '',
                path TEXT NOT NULL, content_hash TEXT, size_bytes INTEGER, mtime REAL,
                width INTEGER, height INTEGER, status TEXT NOT NULL DEFAULT 'pending',
                error TEXT, n_faces INTEGER NOT NULL DEFAULT 0, detector_version TEXT,
                indexed_at REAL, source TEXT NOT NULL DEFAULT 'local', origin_url TEXT,
                UNIQUE(owner, path));
        """)
        self.conn.execute(
            f"INSERT INTO images_mig(id, owner, {', '.join(keep)}) "
            f"SELECT id, ?, {', '.join(keep)} FROM images", (owner,))
        self.conn.executescript("""
            DROP TABLE images;
            ALTER TABLE images_mig RENAME TO images;
            CREATE INDEX IF NOT EXISTS idx_images_status ON images(status);
            CREATE INDEX IF NOT EXISTS idx_images_hash   ON images(content_hash);
            CREATE INDEX IF NOT EXISTS idx_images_owner  ON images(owner);
        """)
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.commit()

    def _migrate_saved_searches(self) -> None:
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(saved_searches)")}
        if not cols or "user" in cols:
            return
        owner = self._legacy_owner()
        self.conn.executescript(f"""
            CREATE TABLE saved_mig (
                id INTEGER PRIMARY KEY, user TEXT NOT NULL DEFAULT '', name TEXT NOT NULL,
                spec TEXT NOT NULL, created_at REAL, UNIQUE(user, name));
            INSERT INTO saved_mig(user, name, spec, created_at)
                SELECT '{owner}', name, spec, created_at FROM saved_searches;
            DROP TABLE saved_searches;
            ALTER TABLE saved_mig RENAME TO saved_searches;
        """)
        self.conn.commit()

    def _backup(self, tag: str) -> None:
        """Copy the index aside before a destructive rebuild.

        A schema rebuild that loses somebody's library is not recoverable by re-running
        anything, so it costs one file copy to make it recoverable by hand.
        """
        import shutil
        try:
            if self.path.exists() and self.path.stat().st_size > 0:
                dst = self.path.with_suffix(self.path.suffix + f".{tag}.bak")
                if not dst.exists():
                    self.conn.commit()
                    shutil.copy2(self.path, dst)
        except Exception:      # noqa: BLE001 - a failed backup must not block the migration
            pass

    # ------------------------------------------------------------------ basics

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        try:
            yield self.conn
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO meta(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value)
        )

    def bump_generation(self) -> int:
        """Mark the index as changed.

        Read caches - the collection CDF, the memory-mapped feature store's row count -
        are built per connection and would otherwise survive an import that invalidated
        them. A counter in the database is the only signal every thread can see.
        """
        n = int(self.get_meta("generation") or 0) + 1
        self.set_meta("generation", str(n))
        self.conn.commit()
        return n

    def generation(self) -> int:
        return int(self.get_meta("generation") or 0)

    def get_meta(self, key: str) -> str | None:
        r = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return r["value"] if r else None

    # ------------------------------------------------------------------ images

    def image_fingerprints(self, owner: str = "") -> dict[str, tuple[float, int, str]]:
        """path -> (mtime, size, status) for ONE owner. Drives incremental skip decisions.

        Scoped, because two accounts may hold the same path (their own copy of the same
        photo) and each has to decide independently whether their copy needs re-indexing.
        """
        return {
            r["path"]: (r["mtime"], r["size_bytes"], r["status"])
            for r in self.conn.execute(
                "SELECT path, mtime, size_bytes, status FROM images WHERE owner=?", (owner,)
            )
        }

    def upsert_image(self, **kw: Any) -> int:
        kw.setdefault("indexed_at", time.time())
        kw.setdefault("owner", "")
        cols = ",".join(kw)
        ph = ",".join("?" * len(kw))
        upd = ",".join(f"{c}=excluded.{c}" for c in kw if c not in ("path", "owner"))
        cur = self.conn.execute(
            f"INSERT INTO images({cols}) VALUES({ph}) "
            f"ON CONFLICT(owner,path) DO UPDATE SET {upd} RETURNING id",
            tuple(kw.values()),
        )
        return int(cur.fetchone()[0])

    def clear_faces(self, image_id: int) -> None:
        """Remove prior faces for an image before re-indexing it.

        Deletion cascades to predictions and duplicate memberships, which is what
        docs/LICENSING.md section 4.2 requires of a delete: no orphaned biometric data.
        """
        self.conn.execute("DELETE FROM faces WHERE image_id=?", (image_id,))

    # ------------------------------------------------------------------- faces

    def insert_face(self, **kw: Any) -> int:
        cols = ",".join(kw)
        ph = ",".join("?" * len(kw))
        cur = self.conn.execute(
            f"INSERT INTO faces({cols}) VALUES({ph}) RETURNING id", tuple(kw.values())
        )
        return int(cur.fetchone()[0])

    def faces_missing_prediction(self, model: str, model_version: str,
                                 owner: str | None = None) -> list[sqlite3.Row]:
        """Faces with no current-version prediction from `model` - the lazy-work queue.

        E4 measured MiVOLO at ~190x the cost of the retired baseline, so age/gender is not
        run eagerly on every detected face. This is how a later pass finds the work, and
        also how a model-version bump re-queues exactly the affected faces.
        """
        sql = ("SELECT f.* FROM faces f "
               "JOIN images i ON i.id=f.image_id "
               "LEFT JOIN predictions p ON p.face_id=f.id AND p.model=? "
               "WHERE (p.face_id IS NULL OR p.model_version != ?)")
        args: list[Any] = [model, model_version]
        if owner is not None:
            sql += " AND i.owner=?"
            args.append(owner)
        return list(self.conn.execute(sql + " ORDER BY f.id", args))

    def faces_needing_features(self, encoder_version: str, crop_version: str,
                               owner: str | None = None):
        sql = ("SELECT f.* FROM faces f JOIN images i ON i.id=f.image_id "
               "WHERE (f.feature_row IS NULL OR f.encoder_version IS NOT ? "
               "OR f.crop_version IS NOT ?)")
        args: list[Any] = [encoder_version, crop_version]
        if owner is not None:
            sql += " AND i.owner=?"
            args.append(owner)
        return list(self.conn.execute(sql, args))

    # ------------------------------------------------------------- predictions

    def upsert_predictions(self, rows: list[dict]) -> None:
        if not rows:
            return
        now = time.time()
        self.conn.executemany(
            "INSERT INTO predictions(face_id,model,model_version,config_hash,value,"
            "confidence,std,interval_lo,interval_hi,distribution,extra,created_at) "
            "VALUES(:face_id,:model,:model_version,:config_hash,:value,:confidence,:std,"
            ":interval_lo,:interval_hi,:distribution,:extra,:created_at) "
            "ON CONFLICT(face_id,model) DO UPDATE SET "
            "model_version=excluded.model_version, config_hash=excluded.config_hash,"
            "value=excluded.value, confidence=excluded.confidence, std=excluded.std,"
            "interval_lo=excluded.interval_lo, interval_hi=excluded.interval_hi,"
            "distribution=excluded.distribution, extra=excluded.extra,"
            "created_at=excluded.created_at",
            [{**r, "created_at": now,
              "distribution": json.dumps(r["distribution"]) if r.get("distribution") else None,
              "extra": json.dumps(r["extra"]) if r.get("extra") else None,
              **{k: r.get(k) for k in
                 ("config_hash", "confidence", "std", "interval_lo", "interval_hi")}}
             for r in rows]
        )

    # -------------------------------------------------------------------- runs

    def start_run(self, root: str, config_hash: str, config: dict, owner: str = "") -> int:
        cur = self.conn.execute(
            "INSERT INTO runs(started_at,root,config_hash,config,owner) "
            "VALUES(?,?,?,?,?) RETURNING id",
            (time.time(), str(root), config_hash, json.dumps(config, default=str), owner),
        )
        rid = int(cur.fetchone()[0])
        self.conn.commit()
        return rid

    def finish_run(self, run_id: int, **counts: Any) -> None:
        sets = ",".join(f"{k}=?" for k in counts)
        self.conn.execute(
            f"UPDATE runs SET finished_at=?, status='done', {sets} WHERE id=?",
            (time.time(), *counts.values(), run_id),
        )
        self.conn.commit()

    # ------------------------------------------------------------------- stats

    # ---------------------------------------------------------------- feedback

    def add_feedback(self, face_id: int, kind: str, user: str = "default",
                     note: str | None = None, undo_seconds: float = 10.0) -> float:
        """Record a judgement. Returns the timestamp at which it starts affecting ranking.

        **At most one judgement is undoable at a time.** Marking a second face commits the
        first one immediately. A queue of parallel countdowns was the wrong model: it made
        undo a serial thing you had to chase, and it meant the ranking lagged several
        judgements behind what the user had actually said. One live undo, attached to the
        card you just marked, is both simpler to reason about and faster to learn from.
        """
        now = time.time()
        commit_at = now + max(0.0, undo_seconds)
        self.commit_pending(user, except_face_id=face_id)
        self.conn.execute(
            "INSERT INTO feedback(face_id,user,kind,note,created_at,commit_at) "
            "VALUES(?,?,?,?,?,?) "
            "ON CONFLICT(face_id,user,kind) DO UPDATE SET note=excluded.note,"
            "created_at=excluded.created_at, commit_at=excluded.commit_at",
            (face_id, user, kind, note, now, commit_at))
        self.conn.commit()
        return commit_at

    def commit_pending(self, user: str = "default", except_face_id: int | None = None) -> int:
        """Close the undo window on everything still open, optionally sparing one face."""
        now = time.time()
        sql = "UPDATE feedback SET commit_at=? WHERE user=? AND COALESCE(commit_at,0) > ?"
        args: list[Any] = [now, user, now]
        if except_face_id is not None:
            sql += " AND face_id != ?"
            args.append(except_face_id)
        cur = self.conn.execute(sql, args)
        self.conn.commit()
        return cur.rowcount

    def active_undo(self, user: str = "default") -> dict | None:
        """The single judgement still inside its window, if there is one."""
        now = time.time()
        r = self.conn.execute(
            "SELECT face_id, kind, created_at, commit_at FROM feedback "
            "WHERE user=? AND COALESCE(commit_at,0) > ? ORDER BY created_at DESC LIMIT 1",
            (user, now)).fetchone()
        return dict(r) if r else None

    def undo_feedback(self, face_id: int, user: str = "default",
                      kind: str | None = None) -> int:
        """Reverse a judgement. The face returns to results and the model never saw it,
        provided the undo happened inside the window."""
        if kind:
            cur = self.conn.execute(
                "DELETE FROM feedback WHERE face_id=? AND user=? AND kind=?",
                (face_id, user, kind))
        else:
            cur = self.conn.execute(
                "DELETE FROM feedback WHERE face_id=? AND user=?", (face_id, user))
        self.conn.commit()
        return cur.rowcount

    def pending_feedback(self, user: str = "default") -> list[dict]:
        """Judgements still inside their undo window."""
        now = time.time()
        return [dict(r) for r in self.conn.execute(
            "SELECT face_id, kind, created_at, commit_at FROM feedback "
            "WHERE user=? AND commit_at > ? ORDER BY created_at DESC", (user, now))]

    def remove_feedback(self, face_id: int, kind: str, user: str = "default") -> None:
        self.conn.execute("DELETE FROM feedback WHERE face_id=? AND user=? AND kind=?",
                          (face_id, user, kind))
        self.conn.commit()

    def feedback_for(self, user: str = "default") -> dict[int, list[str]]:
        out: dict[int, list[str]] = {}
        for r in self.conn.execute(
                "SELECT face_id, kind FROM feedback WHERE user=?", (user,)):
            out.setdefault(r["face_id"], []).append(r["kind"])
        return out

    # ------------------------------------------------------- reference faces

    def add_reference(self, path: str, face_idx: int, feature, kind: str = "like",
                      user: str = "default") -> None:
        import numpy as np
        f = np.asarray(feature, dtype=np.float32)
        self.conn.execute(
            "INSERT INTO reference_faces(user,path,face_idx,feature,dim,kind,added_at) "
            "VALUES(?,?,?,?,?,?,?) ON CONFLICT(user,path,face_idx) DO UPDATE SET "
            "feature=excluded.feature, kind=excluded.kind, added_at=excluded.added_at",
            (user, path, face_idx, f.tobytes(), int(f.size), kind, time.time()))
        self.conn.commit()

    def references(self, user: str = "default") -> list[dict]:
        import numpy as np
        out = []
        for r in self.conn.execute(
                "SELECT id,path,face_idx,feature,dim,kind FROM reference_faces WHERE user=?",
                (user,)):
            out.append({"id": r["id"], "path": r["path"], "face_idx": r["face_idx"],
                        "kind": r["kind"],
                        "feature": np.frombuffer(r["feature"], dtype=np.float32)})
        return out

    def clear_references(self, user: str = "default") -> int:
        n = self.conn.execute("SELECT COUNT(*) FROM reference_faces WHERE user=?",
                              (user,)).fetchone()[0]
        self.conn.execute("DELETE FROM reference_faces WHERE user=?", (user,))
        self.conn.commit()
        return n

    def delete_reference(self, ref_id: int, user: str = "default") -> None:
        self.conn.execute("DELETE FROM reference_faces WHERE id=? AND user=?", (ref_id, user))
        self.conn.commit()

    def save_search(self, name: str, spec: dict, user: str = "") -> None:
        self.conn.execute(
            "INSERT INTO saved_searches(user,name,spec,created_at) VALUES(?,?,?,?) "
            "ON CONFLICT(user,name) DO UPDATE SET spec=excluded.spec, "
            "created_at=excluded.created_at",
            (user, name, json.dumps(spec, default=str), time.time()))
        self.conn.commit()

    def list_saved_searches(self, user: str = "") -> list[dict]:
        return [{"name": r["name"], "spec": json.loads(r["spec"]), "created_at": r["created_at"]}
                for r in self.conn.execute(
                    "SELECT name, spec, created_at FROM saved_searches WHERE user=? "
                    "ORDER BY created_at DESC", (user,))]

    def delete_saved_search(self, name: str, user: str = "") -> None:
        self.conn.execute("DELETE FROM saved_searches WHERE name=? AND user=?", (name, user))
        self.conn.commit()

    def stats(self, owner: str | None = None) -> dict:
        """Collection counts. `owner=None` is the whole index and is for administrators
        only - every user-facing caller passes their own name."""
        c = self.conn
        w, a = ("", []) if owner is None else (" WHERE owner=?", [owner])
        jw, ja = ("", []) if owner is None else (" WHERE i.owner=?", [owner])
        out = {
            "owner": owner,
            "images": c.execute(f"SELECT COUNT(*) FROM images{w}", a).fetchone()[0],
            "faces": c.execute(
                f"SELECT COUNT(*) FROM faces f JOIN images i ON i.id=f.image_id{jw}",
                ja).fetchone()[0],
            "predictions": c.execute(
                "SELECT COUNT(*) FROM predictions p JOIN faces f ON f.id=p.face_id "
                f"JOIN images i ON i.id=f.image_id{jw}", ja).fetchone()[0],
            "by_status": {r["status"]: r["n"] for r in c.execute(
                f"SELECT status, COUNT(*) n FROM images{w} GROUP BY status", a)},
            "by_model": {r["model"]: r["n"] for r in c.execute(
                "SELECT p.model, COUNT(*) n FROM predictions p JOIN faces f ON f.id=p.face_id "
                f"JOIN images i ON i.id=f.image_id{jw} GROUP BY p.model", ja)},
        }
        r = c.execute("SELECT AVG(f.quality) q FROM faces f "
                      f"JOIN images i ON i.id=f.image_id{jw}", ja).fetchone()
        out["mean_quality"] = r["q"]
        out["bytes"] = c.execute(
            f"SELECT COALESCE(SUM(size_bytes),0) FROM images{w}", a).fetchone()[0]
        return out

    # ------------------------------------------------------------- ownership

    def owner_of_image(self, image_id: int) -> str | None:
        r = self.conn.execute("SELECT owner FROM images WHERE id=?", (image_id,)).fetchone()
        return r["owner"] if r else None

    def owner_of_face(self, face_id: int) -> str | None:
        r = self.conn.execute(
            "SELECT i.owner FROM faces f JOIN images i ON i.id=f.image_id WHERE f.id=?",
            (face_id,)).fetchone()
        return r["owner"] if r else None

    def owned_image_paths(self, owner: str) -> list[str]:
        return [r["path"] for r in self.conn.execute(
            "SELECT path FROM images WHERE owner=? AND source IN ('upload','url')", (owner,))]

    def delete_owner_data(self, owner: str) -> dict:
        """Erase everything an account accumulated.

        Face embeddings are biometric data (docs/LICENSING.md 4.2): deleting an account has
        to actually delete, not just detach. Images cascade to faces, which cascade to
        predictions, feedback, duplicate memberships. Feature-store rows are left in place -
        they are unaddressable once no face row points at them - and the caller is
        responsible for unlinking the uploaded files themselves.
        """
        c = self.conn
        counts = {
            "images": c.execute("SELECT COUNT(*) FROM images WHERE owner=?",
                                (owner,)).fetchone()[0],
            "feedback": c.execute("SELECT COUNT(*) FROM feedback WHERE user=?",
                                  (owner,)).fetchone()[0],
            "references": c.execute("SELECT COUNT(*) FROM reference_faces WHERE user=?",
                                    (owner,)).fetchone()[0],
        }
        c.execute("PRAGMA foreign_keys=ON")
        c.execute("DELETE FROM feedback WHERE user=?", (owner,))
        c.execute("DELETE FROM reference_faces WHERE user=?", (owner,))
        c.execute("DELETE FROM saved_searches WHERE user=?", (owner,))
        c.execute("DELETE FROM faces WHERE image_id IN "
                  "(SELECT id FROM images WHERE owner=?)", (owner,))
        c.execute("DELETE FROM images WHERE owner=?", (owner,))
        c.commit()
        return counts
