"""Google sign-in: configuration, state handling and claim verification.

No network. Every test here is about the checks that happen after Google's response has
already arrived - which is where an OIDC client is actually attacked, since TLS covers the
transport but says nothing about who a valid token was issued for.
"""
import base64
import json
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from facet.api import oauth  # noqa: E402

CFG = oauth.GoogleConfig(client_id="cid.apps.googleusercontent.com",
                         client_secret="secret", public_url="https://facet.example")


def claims(**over):
    base = {"iss": "https://accounts.google.com", "aud": CFG.client_id,
            "exp": time.time() + 600, "sub": "12345", "email": "a@b.com",
            "email_verified": True, "name": "A B", "nonce": "N"}
    base.update(over)
    return base


# ---------------------------------------------------------------- configuration

def test_disabled_without_full_configuration(monkeypatch):
    for k in ("FACET_GOOGLE_CLIENT_ID", "FACET_GOOGLE_CLIENT_SECRET", "FACET_PUBLIC_URL"):
        monkeypatch.delenv(k, raising=False)
    cfg = oauth.GoogleConfig.from_env()
    assert cfg.enabled() is False
    assert "FACET_GOOGLE_CLIENT_ID" in cfg.why_disabled()


def test_enabled_when_configured(monkeypatch):
    monkeypatch.setenv("FACET_GOOGLE_CLIENT_ID", "x")
    monkeypatch.setenv("FACET_GOOGLE_CLIENT_SECRET", "y")
    monkeypatch.setenv("FACET_PUBLIC_URL", "https://f.example/")
    cfg = oauth.GoogleConfig.from_env()
    assert cfg.enabled() and cfg.why_disabled() == ""
    assert cfg.redirect_uri == "https://f.example/api/auth/google/callback"


def test_authorize_url_carries_state_and_nonce():
    url = oauth.authorize_url(CFG, "STATE", "NONCE")
    assert url.startswith(oauth.AUTH_URI + "?")
    for part in ("state=STATE", "nonce=NONCE", "response_type=code",
                 "scope=openid+email+profile", "prompt=select_account"):
        assert part in url


# ----------------------------------------------------------------------- state

def test_state_is_single_use():
    store = oauth.StateStore()
    state, nonce = store.issue("/somewhere")
    assert store.take(state) == (nonce, "/somewhere")
    assert store.take(state) is None, "a replayed callback must find nothing"


def test_state_expires():
    store = oauth.StateStore(ttl=-1)
    state, _ = store.issue()
    assert store.take(state) is None


def test_unknown_state_is_rejected():
    assert oauth.StateStore().take("not-a-state") is None


# ---------------------------------------------------------------------- claims

def test_valid_claims_produce_an_identity():
    got = oauth.verify_claims(claims(), CFG, "N")
    assert got == {"sub": "12345", "email": "a@b.com", "name": "A B", "picture": None}


@pytest.mark.parametrize("over,why", [
    ({"iss": "https://evil.example"}, "issuer"),
    ({"aud": "someone-elses-client-id"}, "different application"),
    ({"exp": time.time() - 6000}, "expired"),
    ({"nonce": "different"}, "does not match"),
    ({"sub": None}, "no subject"),
])
def test_bad_claims_are_rejected(over, why):
    with pytest.raises(oauth.OAuthError, match=why):
        oauth.verify_claims(claims(**over), CFG, "N")


def test_an_unverified_email_is_dropped_not_trusted():
    """Account linking keys on the email address. An unverified one would let anybody who
    can set a Google profile address claim an existing account."""
    got = oauth.verify_claims(claims(email_verified=False), CFG, "N")
    assert got["email"] == ""


def test_id_token_payload_is_decoded():
    payload = base64.urlsafe_b64encode(json.dumps(claims()).encode()).rstrip(b"=").decode()
    got = oauth.decode_id_token(f"header.{payload}.signature")
    assert got["sub"] == "12345"


@pytest.mark.parametrize("tok", ["", "a.b", "not a jwt", "a.!!!!.c"])
def test_malformed_id_tokens_raise(tok):
    with pytest.raises(oauth.OAuthError):
        oauth.decode_id_token(tok)


def test_aud_may_be_a_list():
    assert oauth.verify_claims(claims(aud=[CFG.client_id, "other"]), CFG, "N")["sub"] == "12345"
