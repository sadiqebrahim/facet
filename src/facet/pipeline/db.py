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

SCHEMA_VERSION = 1

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
    path          TEXT NOT NULL UNIQUE,
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
    indexed_at    REAL
);
CREATE INDEX IF NOT EXISTS idx_images_status ON images(status);
CREATE INDEX IF NOT EXISTS idx_images_hash   ON images(content_hash);

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
    status       TEXT DEFAULT 'running'
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
        self.set_meta("schema_version", str(SCHEMA_VERSION))
        self.conn.commit()

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

    def get_meta(self, key: str) -> str | None:
        r = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return r["value"] if r else None

    # ------------------------------------------------------------------ images

    def image_fingerprints(self) -> dict[str, tuple[float, int, str]]:
        """path -> (mtime, size, status). Drives incremental skip decisions."""
        return {
            r["path"]: (r["mtime"], r["size_bytes"], r["status"])
            for r in self.conn.execute(
                "SELECT path, mtime, size_bytes, status FROM images"
            )
        }

    def upsert_image(self, **kw: Any) -> int:
        kw.setdefault("indexed_at", time.time())
        cols = ",".join(kw)
        ph = ",".join("?" * len(kw))
        upd = ",".join(f"{c}=excluded.{c}" for c in kw if c != "path")
        cur = self.conn.execute(
            f"INSERT INTO images({cols}) VALUES({ph}) "
            f"ON CONFLICT(path) DO UPDATE SET {upd} RETURNING id",
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

    def faces_missing_prediction(self, model: str, model_version: str) -> list[sqlite3.Row]:
        """Faces with no current-version prediction from `model` - the lazy-work queue.

        E4 measured MiVOLO at ~190x the cost of the retired baseline, so age/gender is not
        run eagerly on every detected face. This is how a later pass finds the work, and
        also how a model-version bump re-queues exactly the affected faces.
        """
        return list(self.conn.execute(
            "SELECT f.* FROM faces f "
            "LEFT JOIN predictions p ON p.face_id=f.id AND p.model=? "
            "WHERE p.face_id IS NULL OR p.model_version != ? "
            "ORDER BY f.id", (model, model_version)
        ))

    def faces_needing_features(self, encoder_version: str, crop_version: str):
        return list(self.conn.execute(
            "SELECT * FROM faces WHERE feature_row IS NULL "
            "OR encoder_version IS NOT ? OR crop_version IS NOT ?",
            (encoder_version, crop_version)
        ))

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

    def start_run(self, root: str, config_hash: str, config: dict) -> int:
        cur = self.conn.execute(
            "INSERT INTO runs(started_at,root,config_hash,config) VALUES(?,?,?,?) RETURNING id",
            (time.time(), str(root), config_hash, json.dumps(config, default=str)),
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

    def stats(self) -> dict:
        c = self.conn
        out = {
            "images": c.execute("SELECT COUNT(*) FROM images").fetchone()[0],
            "faces": c.execute("SELECT COUNT(*) FROM faces").fetchone()[0],
            "predictions": c.execute("SELECT COUNT(*) FROM predictions").fetchone()[0],
            "by_status": {r["status"]: r["n"] for r in c.execute(
                "SELECT status, COUNT(*) n FROM images GROUP BY status")},
            "by_model": {r["model"]: r["n"] for r in c.execute(
                "SELECT model, COUNT(*) n FROM predictions GROUP BY model")},
        }
        r = c.execute("SELECT AVG(quality) q, AVG(n_faces) f FROM faces "
                      "JOIN images ON images.id=faces.image_id").fetchone()
        out["mean_quality"] = r["q"]
        return out
