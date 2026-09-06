"""Accounts and sessions.

The point is not to defend against a determined attacker - this is a local tool binding to a
private network - it is that **each user's taste model must be their own**. E14's whole result
is that preference is personal; sharing one 'default' bucket between people would blend
incompatible tastes into mush and quietly poison everyone's ranking.

It also matters for a second reason. Face embeddings are biometric data
(docs/LICENSING.md section 4), and the moment the server binds to anything other than
localhost, "who is asking" stops being a rhetorical question. Passwords are PBKDF2-SHA256
with a per-user salt; sessions are random 256-bit tokens.

What this is NOT: there is no TLS here, so on an untrusted network the token and password
cross the wire in the clear. Put it behind a VPN or an HTTPS reverse proxy - which is exactly
what serve.py warns about when you bind beyond localhost.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
import time

PBKDF2_ROUNDS = 200_000
SESSION_TTL = 60 * 60 * 24 * 30          # 30 days
USERNAME_RE = re.compile(r"^[a-zA-Z0-9._-]{2,32}$")


def hash_password(password: str, salt: bytes | None = None) -> tuple[str, str]:
    salt = salt or os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, PBKDF2_ROUNDS)
    return dk.hex(), salt.hex()


def verify_password(password: str, stored_hash: str, salt_hex: str) -> bool:
    dk, _ = hash_password(password, bytes.fromhex(salt_hex))
    return hmac.compare_digest(dk, stored_hash)     # constant time


def new_token() -> str:
    return secrets.token_urlsafe(32)


def valid_username(name: str) -> bool:
    return bool(USERNAME_RE.match(name or ""))


SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    username      TEXT PRIMARY KEY,
    password_hash TEXT NOT NULL,
    salt          TEXT NOT NULL,
    display_name  TEXT,
    created_at    REAL,
    -- The first account created becomes the administrator. There is no email here to send
    -- a reset link to, so "forgot password" means asking that person - which only works if
    -- the UI can tell you who they are.
    is_admin      INTEGER NOT NULL DEFAULT 0,
    must_change   INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS sessions (
    token      TEXT PRIMARY KEY,
    username   TEXT NOT NULL REFERENCES users(username) ON DELETE CASCADE,
    created_at REAL,
    expires_at REAL
);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(username);
"""


class Accounts:
    """User and session storage, over the same SQLite file as the index."""

    def __init__(self, conn):
        self.conn = conn
        self.conn.executescript(SCHEMA)
        self._migrate()
        self.conn.commit()

    def _migrate(self) -> None:
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(users)")}
        for col, ddl in (("is_admin", "INTEGER NOT NULL DEFAULT 0"),
                         ("must_change", "INTEGER NOT NULL DEFAULT 0")):
            if col not in cols:
                self.conn.execute(f"ALTER TABLE users ADD COLUMN {col} {ddl}")
        # An index created before roles existed has no admin; promote the earliest account
        # so the "contact the admin" path always has somebody to name.
        if cols and not self.conn.execute(
                "SELECT 1 FROM users WHERE is_admin=1").fetchone():
            r = self.conn.execute(
                "SELECT username FROM users ORDER BY created_at LIMIT 1").fetchone()
            if r:
                self.conn.execute("UPDATE users SET is_admin=1 WHERE username=?",
                                  (r["username"],))
        self.conn.commit()

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]

    def exists(self, username: str) -> bool:
        return self.conn.execute("SELECT 1 FROM users WHERE username=?",
                                 (username,)).fetchone() is not None

    def create(self, username: str, password: str, display_name: str | None = None,
               is_admin: bool | None = None) -> None:
        h, s = hash_password(password)
        admin = self.count() == 0 if is_admin is None else is_admin
        self.conn.execute(
            "INSERT INTO users(username,password_hash,salt,display_name,created_at,is_admin) "
            "VALUES(?,?,?,?,?,?)",
            (username, h, s, display_name or username, time.time(), int(admin)))
        self.conn.commit()

    def is_admin(self, username: str) -> bool:
        r = self.conn.execute("SELECT is_admin FROM users WHERE username=?",
                              (username,)).fetchone()
        return bool(r and r["is_admin"])

    def admins(self) -> list[str]:
        return [r["username"] for r in
                self.conn.execute("SELECT username FROM users WHERE is_admin=1")]

    def set_password(self, username: str, password: str, must_change: bool = True) -> None:
        h, s = hash_password(password)
        self.conn.execute(
            "UPDATE users SET password_hash=?, salt=?, must_change=? WHERE username=?",
            (h, s, int(must_change), username))
        # Any existing session is invalidated: a reset the user did not perform themselves
        # should not leave a live session behind.
        self.conn.execute("DELETE FROM sessions WHERE username=?", (username,))
        self.conn.commit()

    def set_admin(self, username: str, admin: bool) -> None:
        self.conn.execute("UPDATE users SET is_admin=? WHERE username=?",
                          (int(admin), username))
        self.conn.commit()

    def delete(self, username: str) -> None:
        self.conn.execute("DELETE FROM sessions WHERE username=?", (username,))
        self.conn.execute("DELETE FROM users WHERE username=?", (username,))
        self.conn.commit()

    def clear_must_change(self, username: str) -> None:
        self.conn.execute("UPDATE users SET must_change=0 WHERE username=?", (username,))
        self.conn.commit()

    def must_change(self, username: str) -> bool:
        r = self.conn.execute("SELECT must_change FROM users WHERE username=?",
                              (username,)).fetchone()
        return bool(r and r["must_change"])

    def details(self) -> list[dict]:
        return [dict(r) for r in self.conn.execute(
            "SELECT username, display_name, created_at, is_admin, must_change "
            "FROM users ORDER BY is_admin DESC, username")]

    def check(self, username: str, password: str) -> bool:
        r = self.conn.execute("SELECT password_hash, salt FROM users WHERE username=?",
                              (username,)).fetchone()
        if r is None:
            # Hash anyway so a missing user and a wrong password take the same time.
            hash_password(password)
            return False
        return verify_password(password, r["password_hash"], r["salt"])

    def start_session(self, username: str) -> str:
        tok = new_token()
        now = time.time()
        self.conn.execute(
            "INSERT INTO sessions(token,username,created_at,expires_at) VALUES(?,?,?,?)",
            (tok, username, now, now + SESSION_TTL))
        self.conn.execute("DELETE FROM sessions WHERE expires_at < ?", (now,))
        self.conn.commit()
        return tok

    def resolve(self, token: str | None) -> str | None:
        if not token:
            return None
        r = self.conn.execute(
            "SELECT username, expires_at FROM sessions WHERE token=?", (token,)).fetchone()
        if r is None or r["expires_at"] < time.time():
            return None
        return r["username"]

    def end_session(self, token: str) -> None:
        self.conn.execute("DELETE FROM sessions WHERE token=?", (token,))
        self.conn.commit()

    def info(self, username: str) -> dict:
        r = self.conn.execute(
            "SELECT username, display_name, created_at FROM users WHERE username=?",
            (username,)).fetchone()
        return dict(r) if r else {}

    def list_users(self) -> list[str]:
        return [r["username"] for r in
                self.conn.execute("SELECT username FROM users ORDER BY username")]
