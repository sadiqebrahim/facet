"""Upgrading an index built before multi-tenancy existed.

The `images` rebuild is the only destructive migration in this project. It has to preserve
row ids (every `faces.image_id` points at one), it has to attribute the existing library to
somebody real, and it must not run twice.
"""
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from facet.api.auth import Accounts  # noqa: E402
from facet.pipeline.db import Index  # noqa: E402

OLD_SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE images (
    id INTEGER PRIMARY KEY, path TEXT NOT NULL UNIQUE, content_hash TEXT,
    size_bytes INTEGER, mtime REAL, width INTEGER, height INTEGER,
    status TEXT NOT NULL DEFAULT 'pending', error TEXT,
    n_faces INTEGER NOT NULL DEFAULT 0, detector_version TEXT, indexed_at REAL);
CREATE TABLE faces (
    id INTEGER PRIMARY KEY,
    image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    face_idx INTEGER NOT NULL, x1 REAL, y1 REAL, x2 REAL, y2 REAL,
    det_score REAL, face_px REAL, kps BLOB, quality REAL, quality_json TEXT,
    feature_row INTEGER, encoder_version TEXT, crop_version TEXT,
    UNIQUE(image_id, face_idx));
CREATE TABLE saved_searches (
    id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE, spec TEXT NOT NULL,
    created_at REAL);
CREATE TABLE users (
    username TEXT PRIMARY KEY, password_hash TEXT NOT NULL, salt TEXT NOT NULL,
    display_name TEXT, created_at REAL, is_admin INTEGER NOT NULL DEFAULT 0,
    must_change INTEGER NOT NULL DEFAULT 0);
CREATE TABLE feedback (
    face_id INTEGER NOT NULL, user TEXT NOT NULL DEFAULT 'default', kind TEXT NOT NULL,
    note TEXT, created_at REAL, PRIMARY KEY (face_id, user, kind));
"""


def build_old(path):
    c = sqlite3.connect(path)
    c.executescript(OLD_SCHEMA)
    c.execute("INSERT INTO users(username,password_hash,salt,created_at,is_admin) "
              "VALUES('boss','h','s',1.0,1)")
    c.execute("INSERT INTO users(username,password_hash,salt,created_at,is_admin) "
              "VALUES('later','h','s',2.0,0)")
    for i in range(3):
        c.execute("INSERT INTO images(id,path,status,n_faces) VALUES(?,?,'ok',1)",
                  (100 + i, f"/photos/{i}.jpg"))
        c.execute("INSERT INTO faces(id,image_id,face_idx,feature_row) VALUES(?,?,0,?)",
                  (200 + i, 100 + i, i))
    c.execute("INSERT INTO saved_searches(name,spec,created_at) VALUES('old','{}',1.0)")
    c.commit()
    c.close()


def test_existing_library_is_attributed_to_the_first_admin(tmp_path):
    p = tmp_path / "old.db"
    build_old(p)
    ix = Index(p)
    try:
        owners = {r["owner"] for r in ix.conn.execute("SELECT DISTINCT owner FROM images")}
        assert owners == {"boss"}, "the account that built the library keeps it"
        assert ix.stats(owner="boss")["images"] == 3
        assert ix.stats(owner="later")["images"] == 0
    finally:
        ix.close()


def test_face_rows_still_resolve_after_the_rebuild(tmp_path):
    p = tmp_path / "old.db"
    build_old(p)
    ix = Index(p)
    try:
        rows = list(ix.conn.execute(
            "SELECT f.id, i.path FROM faces f JOIN images i ON i.id=f.image_id ORDER BY f.id"))
        assert [r["id"] for r in rows] == [200, 201, 202]
        assert [r["path"] for r in rows] == [f"/photos/{i}.jpg" for i in range(3)]
    finally:
        ix.close()


def test_two_accounts_may_hold_the_same_path(tmp_path):
    """The old UNIQUE(path) made a shared server one shared library. Two people uploading
    the same photo must get two rows."""
    ix = Index(tmp_path / "new.db")
    try:
        a = ix.upsert_image(owner="alice", path="/same.jpg", status="ok")
        b = ix.upsert_image(owner="bob", path="/same.jpg", status="ok")
        assert a != b
        # ...and the same owner re-indexing it still gets one
        assert ix.upsert_image(owner="alice", path="/same.jpg", status="ok") == a
    finally:
        ix.close()


def test_the_migration_is_idempotent(tmp_path):
    p = tmp_path / "old.db"
    build_old(p)
    Index(p).close()
    ix = Index(p)                      # second open must be a no-op
    try:
        assert ix.stats(owner="boss")["images"] == 3
        assert not list(ix.conn.execute(
            "SELECT name FROM sqlite_master WHERE name='images_mig'"))
    finally:
        ix.close()


def test_a_backup_is_written_before_the_rebuild(tmp_path):
    p = tmp_path / "old.db"
    build_old(p)
    Index(p).close()
    assert (tmp_path / "old.db.pre-owner-migration.bak").exists()


def test_saved_searches_are_carried_over(tmp_path):
    p = tmp_path / "old.db"
    build_old(p)
    ix = Index(p)
    try:
        assert [s["name"] for s in ix.list_saved_searches("boss")] == ["old"]
        assert ix.list_saved_searches("later") == []
    finally:
        ix.close()


def test_a_fresh_index_needs_no_migration(tmp_path):
    ix = Index(tmp_path / "fresh.db")
    try:
        assert ix._has_unique_owner_path()
        assert not (tmp_path / "fresh.db.pre-owner-migration.bak").exists()
    finally:
        ix.close()


# ------------------------------------------------------------ federated accounts

def test_google_identity_creates_and_then_finds_one_account(tmp_path):
    ix = Index(tmp_path / "a.db")
    try:
        acc = Accounts(ix.conn)
        u1, created1 = acc.upsert_federated("google", "sub-1", "ann@example.com", "Ann")
        u2, created2 = acc.upsert_federated("google", "sub-1", "ann@example.com", "Ann")
        assert created1 is True and created2 is False and u1 == u2 == "ann"
        assert acc.count() == 1
        assert acc.is_admin(u1) is True, "the first account is still the administrator"
    finally:
        ix.close()


def test_a_federated_account_cannot_be_signed_into_with_a_password(tmp_path):
    ix = Index(tmp_path / "a.db")
    try:
        acc = Accounts(ix.conn)
        u, _ = acc.upsert_federated("google", "sub-1", "ann@example.com", "Ann")
        assert acc.check(u, "") is False
        assert acc.check(u, "anything") is False
    finally:
        ix.close()


def test_usernames_are_deduplicated(tmp_path):
    ix = Index(tmp_path / "a.db")
    try:
        acc = Accounts(ix.conn)
        acc.create("ann", "secret123")
        u, created = acc.upsert_federated("google", "sub-9", "ann@other.com", "Ann")
        assert created is True and u == "ann2", "a taken name must not be reused"
    finally:
        ix.close()


def test_a_verified_email_links_an_existing_password_account(tmp_path):
    ix = Index(tmp_path / "a.db")
    try:
        acc = Accounts(ix.conn)
        acc.create("ann", "secret123")
        ix.conn.execute("UPDATE users SET email='ann@example.com' WHERE username='ann'")
        ix.conn.commit()
        u, created = acc.upsert_federated("google", "sub-1", "ann@example.com", "Ann")
        assert u == "ann" and created is False
        assert acc.count() == 1
    finally:
        ix.close()
