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


# Split deliberately. `CREATE TABLE IF NOT EXISTS` is a no-op against a table that already
# exists, so on an upgrade the new columns are absent when the index that references them is
# created - which fails with "no such column: provider" and takes every auth route with it.
# Tables first, then _migrate() adds the columns, then the indexes.
SCHEMA_TABLES = """
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
    must_change   INTEGER NOT NULL DEFAULT 0,
    -- password | google. A federated account has no usable password hash, so the password
    -- path must refuse it outright rather than fail a comparison against an empty string.
    provider      TEXT NOT NULL DEFAULT 'password',
    provider_sub  TEXT,
    email         TEXT,
    avatar_url    TEXT,
    last_seen     REAL
);
CREATE TABLE IF NOT EXISTS sessions (
    token      TEXT PRIMARY KEY,
    username   TEXT NOT NULL REFERENCES users(username) ON DELETE CASCADE,
    created_at REAL,
    expires_at REAL
);
"""

SCHEMA_INDEXES = """
CREATE UNIQUE INDEX IF NOT EXISTS idx_users_provider
    ON users(provider, provider_sub) WHERE provider_sub IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(username);
"""


class Accounts:
    """User and session storage, over the same SQLite file as the index."""

    def __init__(self, conn):
        self.conn = conn
        self.conn.executescript(SCHEMA_TABLES)
        self._migrate()                       # must precede the indexes - see SCHEMA_TABLES
        self.conn.executescript(SCHEMA_INDEXES)
        self.conn.commit()

    def _migrate(self) -> None:
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(users)")}
        for col, ddl in (("is_admin", "INTEGER NOT NULL DEFAULT 0"),
                         ("must_change", "INTEGER NOT NULL DEFAULT 0"),
                         ("provider", "TEXT NOT NULL DEFAULT 'password'"),
                         ("provider_sub", "TEXT"),
                         ("email", "TEXT"),
                         ("avatar_url", "TEXT"),
                         ("last_seen", "REAL")):
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
            "SELECT username, display_name, created_at, is_admin, must_change, "
            "provider, email, last_seen FROM users ORDER BY is_admin DESC, username")]

    def check(self, username: str, password: str) -> bool:
        r = self.conn.execute(
            "SELECT password_hash, salt, provider FROM users WHERE username=?",
            (username,)).fetchone()
        if r is None or not r["password_hash"] or r["provider"] != "password":
            # Hash anyway so a missing user, a federated user and a wrong password all take
            # the same time - otherwise the response time enumerates accounts.
            hash_password(password)
            return False
        return verify_password(password, r["password_hash"], r["salt"])

    # ------------------------------------------------------- federated identity

    def find_by_provider(self, provider: str, sub: str) -> str | None:
        r = self.conn.execute(
            "SELECT username FROM users WHERE provider=? AND provider_sub=?",
            (provider, sub)).fetchone()
        return r["username"] if r else None

    def find_by_email(self, email: str) -> str | None:
        if not email:
            return None
        r = self.conn.execute("SELECT username FROM users WHERE lower(email)=lower(?)",
                              (email,)).fetchone()
        return r["username"] if r else None

    def upsert_federated(self, provider: str, sub: str, email: str,
                         display_name: str | None = None,
                         avatar_url: str | None = None) -> tuple[str, bool]:
        """Find or create the account behind a verified provider identity.

        Matching is on `(provider, sub)` - the provider's own immutable id - and never on
        the email address alone. Emails get reassigned and can be changed at the provider;
        treating one as a key is how federated logins end up handing a stranger somebody
        else's account. An existing *password* account with the same address is linked only
        because that address arrived verified from the provider.
        """
        existing = self.find_by_provider(provider, sub)
        if existing:
            self.conn.execute(
                "UPDATE users SET email=?, avatar_url=?, last_seen=? WHERE username=?",
                (email, avatar_url, time.time(), existing))
            self.conn.commit()
            return existing, False

        linked = self.find_by_email(email)
        if linked:
            self.conn.execute(
                "UPDATE users SET provider=?, provider_sub=?, avatar_url=?, last_seen=? "
                "WHERE username=?", (provider, sub, avatar_url, time.time(), linked))
            self.conn.commit()
            return linked, False

        username = self.available_username(email.split("@")[0] if email else "user")
        admin = self.count() == 0
        self.conn.execute(
            "INSERT INTO users(username,password_hash,salt,display_name,created_at,"
            "is_admin,provider,provider_sub,email,avatar_url,last_seen) "
            "VALUES(?,'','',?,?,?,?,?,?,?,?)",
            (username, display_name or email or username, time.time(), int(admin),
             provider, sub, email, avatar_url, time.time()))
        self.conn.commit()
        return username, True

    def available_username(self, seed: str) -> str:
        base = re.sub(r"[^a-zA-Z0-9._-]", "", (seed or "user"))[:24].strip("._-") or "user"
        if len(base) < 2:
            base = f"{base}user"[:24]
        if not self.exists(base):
            return base
        for i in range(2, 1000):
            cand = f"{base[:26]}{i}"
            if not self.exists(cand):
                return cand
        return f"{base[:20]}{secrets.token_hex(4)}"

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
            "SELECT username, display_name, created_at, provider, email, avatar_url "
            "FROM users WHERE username=?", (username,)).fetchone()
        return dict(r) if r else {}

    def touch(self, username: str) -> None:
        self.conn.execute("UPDATE users SET last_seen=? WHERE username=?",
                          (time.time(), username))
        self.conn.commit()

    def list_users(self) -> list[str]:
        return [r["username"] for r in
                self.conn.execute("SELECT username FROM users ORDER BY username")]
