"""Sign in with Google (OpenID Connect authorization-code flow).

Passwords were the right choice for a tool on a VPN. They are the wrong choice for something
hosted: a password store is a liability you inherit forever, "forgot password" needs a mail
server this project does not have, and the accounts here are attached to biometric data, so
the weakest credential in the system sets the security of the whole thing. Delegating to
Google removes the store, the reset flow and most of the credential-stuffing surface.

**It is optional.** With no client configured, `enabled()` is False, the button never renders
and password sign-in continues to work exactly as before. That keeps a self-hosted install
running with no internet access at all, which is the deployment the licensing note in
docs/LICENSING.md section 4 was written for.

On signature verification: the ID token is fetched by this server directly from Google's
token endpoint over TLS, in exchange for a code plus the client secret. OpenID Connect Core
section 3.1.3.7 rule 6 says a token obtained that way may be trusted without re-verifying its
signature, because the TLS channel and the client secret already authenticate the issuer.
`iss`, `aud`, `exp` and `nonce` are still checked - those defend against a *valid* token
issued for somebody else being replayed here, which TLS does nothing about.
"""
from __future__ import annotations

import base64
import json
import os
import secrets
import threading
import time
from dataclasses import dataclass

AUTH_URI = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URI = "https://oauth2.googleapis.com/token"
ISSUERS = {"https://accounts.google.com", "accounts.google.com"}
SCOPES = "openid email profile"
#: How long a half-finished sign-in stays valid. Long enough to pick an account, short
#: enough that a leaked state parameter is worthless by the time anyone finds it.
STATE_TTL = 600.0


@dataclass
class GoogleConfig:
    client_id: str = ""
    client_secret: str = ""
    #: Public origin the browser reaches this server on, e.g. https://facet.example.com.
    #: Google requires an exact redirect-URI match, so this cannot be guessed from the
    #: request - a Host header is attacker-controlled.
    public_url: str = ""

    @classmethod
    def from_env(cls) -> "GoogleConfig":
        return cls(
            client_id=os.environ.get("FACET_GOOGLE_CLIENT_ID", "").strip(),
            client_secret=os.environ.get("FACET_GOOGLE_CLIENT_SECRET", "").strip(),
            public_url=os.environ.get("FACET_PUBLIC_URL", "").strip().rstrip("/"),
        )

    def enabled(self) -> bool:
        return bool(self.client_id and self.client_secret and self.public_url)

    @property
    def redirect_uri(self) -> str:
        return f"{self.public_url}/api/auth/google/callback"

    def why_disabled(self) -> str:
        missing = [n for n, v in (("FACET_GOOGLE_CLIENT_ID", self.client_id),
                                  ("FACET_GOOGLE_CLIENT_SECRET", self.client_secret),
                                  ("FACET_PUBLIC_URL", self.public_url)) if not v]
        return ("Google sign-in is off: set " + ", ".join(missing)) if missing else ""


class StateStore:
    """Short-lived CSRF states for in-flight sign-ins.

    In memory on purpose: a state that does not survive a restart is a state that cannot be
    replayed after one. The cost is that a sign-in started before a deploy has to be
    restarted, which is the right trade for a ten-minute window.
    """

    def __init__(self, ttl: float = STATE_TTL):
        self.ttl = ttl
        self._lock = threading.Lock()
        self._states: dict[str, tuple[float, str, str]] = {}

    def issue(self, next_url: str = "/") -> tuple[str, str]:
        state, nonce = secrets.token_urlsafe(24), secrets.token_urlsafe(16)
        with self._lock:
            self._prune()
            self._states[state] = (time.time() + self.ttl, nonce, next_url)
        return state, nonce

    def take(self, state: str) -> tuple[str, str] | None:
        """Consume a state. Single use - a replayed callback finds nothing."""
        with self._lock:
            self._prune()
            got = self._states.pop(state or "", None)
        if got is None or got[0] < time.time():
            return None
        return got[1], got[2]

    def _prune(self) -> None:
        now = time.time()
        for k in [k for k, v in self._states.items() if v[0] < now]:
            self._states.pop(k, None)


def authorize_url(cfg: GoogleConfig, state: str, nonce: str) -> str:
    from urllib.parse import urlencode
    return AUTH_URI + "?" + urlencode({
        "client_id": cfg.client_id,
        "redirect_uri": cfg.redirect_uri,
        "response_type": "code",
        "scope": SCOPES,
        "state": state,
        "nonce": nonce,
        # Ask every time rather than silently reusing whichever account the browser
        # happens to be signed into - this app holds photographs, and landing in the
        # wrong account is not a recoverable mistake.
        "prompt": "select_account",
    })


def exchange_code(cfg: GoogleConfig, code: str, timeout: float = 15.0) -> dict:
    import requests
    r = requests.post(TOKEN_URI, timeout=timeout, data={
        "code": code,
        "client_id": cfg.client_id,
        "client_secret": cfg.client_secret,
        "redirect_uri": cfg.redirect_uri,
        "grant_type": "authorization_code",
    })
    if r.status_code != 200:
        raise OAuthError(f"Google rejected the sign-in ({r.status_code}). "
                         f"{_brief(r.text)}")
    return r.json()


def decode_id_token(id_token: str) -> dict:
    """Decode the JWT payload. See the module docstring on why the signature is not
    re-verified here - and note this function must only ever be handed a token that came
    straight back from `exchange_code`."""
    parts = id_token.split(".")
    if len(parts) != 3:
        raise OAuthError("malformed ID token")
    pad = "=" * (-len(parts[1]) % 4)
    try:
        return json.loads(base64.urlsafe_b64decode(parts[1] + pad))
    except Exception as e:  # noqa: BLE001
        raise OAuthError("unreadable ID token") from e


def verify_claims(claims: dict, cfg: GoogleConfig, nonce: str, leeway: float = 120.0) -> dict:
    """Check the claims that TLS cannot vouch for, and return the identity."""
    if claims.get("iss") not in ISSUERS:
        raise OAuthError("ID token came from the wrong issuer")
    aud = claims.get("aud")
    if aud != cfg.client_id and cfg.client_id not in (aud or []):
        raise OAuthError("ID token was issued for a different application")
    if float(claims.get("exp", 0)) < time.time() - leeway:
        raise OAuthError("ID token has expired")
    if nonce and claims.get("nonce") != nonce:
        raise OAuthError("ID token does not match this sign-in attempt")
    sub = claims.get("sub")
    if not sub:
        raise OAuthError("ID token carries no subject")
    email = (claims.get("email") or "").strip()
    # An unverified address must not be allowed to link to an existing account: that is
    # exactly the account-takeover path `upsert_federated`'s email-linking branch opens.
    if email and not claims.get("email_verified", False):
        email = ""
    return {"sub": str(sub), "email": email,
            "name": claims.get("name") or email or None,
            "picture": claims.get("picture")}


class OAuthError(RuntimeError):
    pass


def _brief(text: str, n: int = 200) -> str:
    t = " ".join((text or "").split())
    return t[:n] + ("…" if len(t) > n else "")
