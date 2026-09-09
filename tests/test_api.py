"""API tests against a synthetic index, using FastAPI's TestClient.

No models, no GPU, no network: the index rows are written by hand so the expected responses
are known exactly.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from facet.api.app import create_app  # noqa: E402
from facet.pipeline.db import Index  # noqa: E402


ENC, CROP, DIM = "test-enc", "test-crop", 16
#: (beauty, age, p_female, ood) for the three synthetic faces every library gets.
FACES = [(4.5, 30, 0.99, False), (2.1, 55, 0.02, False), (4.2, 28, 0.95, True)]


def seed_library(db_path, feats_path, owner: str, tag: str = "a"):
    """Give `owner` a three-face library. Returns the face ids.

    Every test that needs results needs a library of its own now: search is scoped to the
    caller, so a second account starts empty rather than inheriting the first one's photos.
    That is the whole point of the tenancy work, and a fixture that quietly shared one
    library between accounts would test the behaviour that was removed.
    """
    import numpy as np

    from facet.pipeline.store import FeatureStore

    store = FeatureStore(feats_path, ENC, CROP, dim=DIM)
    rows = store.append(
        np.random.default_rng(abs(hash(owner)) % 2**31).normal(size=(3, DIM)).astype(np.float32))
    ix = Index(db_path)
    ids = []
    for i, (beauty, age, pf, ood) in enumerate(FACES):
        iid = ix.upsert_image(owner=owner, path=f"/{tag}img{i}.jpg", content_hash=f"{tag}h{i}",
                              status="ok", n_faces=1, width=200, height=200)
        fid = ix.insert_face(image_id=iid, face_idx=0, x1=10, y1=10, x2=110, y2=110,
                             det_score=0.9, face_px=100, quality=0.6,
                             feature_row=int(rows[i]), quality_json='{"blur": 500.0}',
                             encoder_version=ENC, crop_version=CROP)
        ids.append(fid)
        ix.upsert_predictions([
            {"face_id": fid, "model": "beauty", "model_version": "b1", "value": beauty,
             "confidence": None if ood else 0.8, "std": 0.3,
             "distribution": [0.1, 0.1, 0.2, 0.3, 0.3],
             "extra": {"p_ge4": beauty / 5, "ood": ood, "warnings": [],
                       "aleatoric": 0.5, "epistemic": 0.05, "source": "SCUT-FBP5500 scale"}},
            {"face_id": fid, "model": "age", "model_version": "m1", "value": age},
            {"face_id": fid, "model": "gender", "model_version": "m1", "value": pf,
             "confidence": 0.95},
        ])
    ix.bump_generation()
    ix.close()
    return ids


@pytest.fixture
def paths(tmp_path):
    return {"db": tmp_path / "api.db", "feats": tmp_path / "feats",
            "uploads": tmp_path / "uploads"}


@pytest.fixture
def client(paths, monkeypatch):
    """A synthetic index WITH a real feature store, owned by the open-mode profile.

    The preference model is fitted over cached embeddings, so a fixture without a feature
    store silently disables personalisation and any test of it would pass vacuously.
    """
    monkeypatch.delenv("FACET_PUBLIC_URL", raising=False)
    monkeypatch.delenv("FACET_REQUIRE_AUTH", raising=False)
    monkeypatch.delenv("FACET_ALLOW_SERVER_PATHS", raising=False)
    seed_library(paths["db"], paths["feats"], "default")
    return TestClient(create_app(str(paths["db"]), str(paths["feats"]),
                                 upload_root=str(paths["uploads"])))


def test_about_carries_the_disclaimer(client):
    d = client.get("/api/about").json()["disclaimer"]
    assert "not a property of the person" in d["estimates_not_measurements"]
    assert "2.2x" in d["measured_skew"], "the measured skew must be stated, not hidden"


def test_stats(client):
    s = client.get("/api/stats").json()
    assert s["images"] == 3 and s["faces"] == 3
    assert s["by_model"]["beauty"] == 3


def test_search_ranks_and_explains(client):
    r = client.post("/api/search", json={
        "preferences": {"age": {"range": [25, 35], "weight": 0.3},
                        "gender": {"value": "female", "weight": 0.2},
                        "attractiveness": {"min_percentile": 0.0, "weight": 0.5}},
        "limit": 10}).json()
    assert r["total_matched"] == 3
    top = r["results"][0]
    assert top["path"] == "/aimg0.jpg"
    assert {c["criterion"] for c in top["contributions"]} == {"age", "gender", "attractiveness"}
    assert "disclaimer" in r


def test_search_response_includes_every_result_note(client):
    r = client.post("/api/search", json={"limit": 5}).json()
    for res in r["results"]:
        assert "SCUT-FBP5500" in res["_note"]


def test_bad_query_is_a_400_not_a_500(client):
    r = client.post("/api/search", json={"preferences": {"age": {"nonsense": 1}}})
    assert r.status_code == 400


def test_face_detail_has_predictions_and_quality(client):
    fid = client.post("/api/search", json={"limit": 1}).json()["results"][0]["face_id"]
    d = client.get(f"/api/face/{fid}").json()
    assert set(d["predictions"]) == {"beauty", "age", "gender"}
    assert d["quality_json"]["blur"] == 500.0
    assert isinstance(d["predictions"]["beauty"]["distribution"], list)


def test_missing_face_is_404(client):
    assert client.get("/api/face/99999").status_code == 404


def test_ood_exclusion_is_counted(client):
    r = client.post("/api/search", json={
        "preferences": {"attractiveness": {"min_percentile": 0.0}},
        "filters": {"exclude_ood": True}, "limit": 10}).json()
    assert r["diagnostics"]["excluded_ood"] == 1
    assert all(not x["ood"] for x in r["results"])


def test_feedback_roundtrip(client):
    fid = client.post("/api/search", json={"limit": 1}).json()["results"][0]["face_id"]
    assert client.post("/api/feedback", json={"face_id": fid, "kind": "like"}
                       ).json()["feedback"] == ["like"]
    got = client.post("/api/search", json={"limit": 10}).json()["results"]
    assert any(x["face_id"] == fid and "like" in x["feedback"] for x in got)
    assert client.post("/api/feedback", json={"face_id": fid, "kind": "like", "remove": True}
                       ).json()["feedback"] == []


def test_feedback_rejects_unknown_kind(client):
    assert client.post("/api/feedback", json={"face_id": 1, "kind": "nope"}).status_code == 400


def test_saved_searches_roundtrip(client):
    spec = {"preferences": {"age": {"range": [20, 30], "weight": 1.0}}}
    client.post("/api/searches", json={"name": "x", "spec": spec})
    assert [s["name"] for s in client.get("/api/searches").json()] == ["x"]
    client.delete("/api/searches/x")
    assert client.get("/api/searches").json() == []


def test_csv_export_carries_provenance(client):
    r = client.post("/api/export?fmt=csv", json={"limit": 5})
    assert r.status_code == 200
    body = r.text
    assert body.startswith("#"), "the export must lead with the estimates disclaimer"
    assert "2.2x" in body, "the measured skew must travel with exported numbers"
    assert "face_id,path,relevance" in body


def test_index_rejects_a_nonexistent_directory(client, monkeypatch):
    monkeypatch.setenv("FACET_ALLOW_SERVER_PATHS", "1")
    assert client.post("/api/index", json={"roots": ["/definitely/not/here"]}).status_code == 400


def test_server_paths_are_refused_by_default(client):
    """Indexing a server directory is an arbitrary filesystem read for anyone with an
    account. On a hosted deployment that is the whole game, so it is off unless the
    operator turns it on."""
    r = client.post("/api/index", json={"roots": ["/tmp"]})
    assert r.status_code == 403 and "upload" in r.json()["detail"]
    r2 = client.post("/api/preference/references", json={"paths": ["/tmp"], "kind": "like"})
    assert r2.status_code == 403


def test_index_status_reports_idle(client):
    s = client.get("/api/index/status").json()
    assert s["status"] == "idle" and s["running"] is False


def test_ui_is_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "Facet" in r.text


# ---------------------------------------------------------------- personalisation

def test_preference_starts_untaught(client):
    d = client.get("/api/preference").json()
    assert d["trained"] is False and d["alpha"] == 0.0
    assert "Not taught yet" in d["note"]


def test_feedback_teaches_the_preference_model(client):
    """Liking and disliking indexed faces must produce a trained model."""
    ids = [r["face_id"] for r in client.post("/api/search", json={"limit": 3}).json()["results"]]
    client.post("/api/feedback", json={"face_id": ids[0], "kind": "like"})
    client.post("/api/feedback", json={"face_id": ids[1], "kind": "dislike"})
    d = client.get("/api/preference").json()
    assert d["feedback"].get("like") == 1 and d["feedback"].get("dislike") == 1


def test_reset_clears_learned_preferences(client):
    ids = [r["face_id"] for r in client.post("/api/search", json={"limit": 2}).json()["results"]]
    client.post("/api/feedback", json={"face_id": ids[0], "kind": "like"})
    client.post("/api/preference/reset")
    assert client.get("/api/preference").json()["feedback"] == {}


def test_reference_rejects_bad_kind(client, monkeypatch):
    monkeypatch.setenv("FACET_ALLOW_SERVER_PATHS", "1")
    assert client.post("/api/preference/references",
                       json={"paths": [], "kind": "nonsense"}).status_code == 400


def test_search_reports_personalisation_state(client):
    r = client.post("/api/search", json={"personalisation": {"enabled": True}, "limit": 3}).json()
    assert "personalisation_alpha" in r["diagnostics"]


# ------------------------------------------------------------ undo & accounts

def test_feedback_has_an_undo_window_before_it_teaches(client):
    fid = client.post("/api/search", json={"limit": 1}).json()["results"][0]["face_id"]
    r = client.post("/api/feedback",
                    json={"face_id": fid, "kind": "dislike", "undo_seconds": 30}).json()
    assert r["undo"] and r["undo"]["commit_at"] is not None
    assert client.get("/api/feedback/undo").json()["undo"], "should be undoable"
    # inside the window the model has learned nothing
    assert client.get("/api/preference").json()["trained"] is False


def test_undo_restores_the_face_and_leaves_no_trace(client):
    fid = client.post("/api/search", json={"limit": 1}).json()["results"][0]["face_id"]
    client.post("/api/feedback", json={"face_id": fid, "kind": "dislike", "undo_seconds": 30})
    hidden = [r["face_id"] for r in client.post("/api/search", json={"limit": 50}).json()["results"]]
    assert fid not in hidden, "a rejected face must leave the results at once"
    client.post("/api/feedback/undo", json={"face_id": fid})
    back = [r["face_id"] for r in client.post("/api/search", json={"limit": 50}).json()["results"]]
    assert fid in back, "undo must bring it back"
    assert client.get("/api/preference").json()["trained"] is False


def test_zero_window_commits_immediately(client):
    fid = client.post("/api/search", json={"limit": 1}).json()["results"][0]["face_id"]
    client.post("/api/feedback", json={"face_id": fid, "kind": "like", "undo_seconds": 0})
    assert client.get("/api/feedback/undo").json()["undo"] is None
    assert client.get("/api/preference").json()["trained"] is True


def test_disliked_faces_are_hidden_and_counted(client):
    ids = [r["face_id"] for r in client.post("/api/search", json={"limit": 3}).json()["results"]]
    client.post("/api/feedback", json={"face_id": ids[0], "kind": "dislike", "undo_seconds": 0})
    r = client.post("/api/search", json={"limit": 50}).json()
    assert ids[0] not in [x["face_id"] for x in r["results"]]
    assert r["diagnostics"]["hidden_by_you"] == 1
    # ...unless explicitly asked for
    r2 = client.post("/api/search",
                     json={"filters": {"exclude_disliked": False}, "limit": 50}).json()
    assert ids[0] in [x["face_id"] for x in r2["results"]]


def test_open_mode_until_the_first_account(client):
    s = client.get("/api/auth/status").json()
    assert s["open_mode"] is True and s["accounts_exist"] is False


def test_registration_then_auth_is_required(client):
    r = client.post("/api/auth/register",
                    json={"username": "u1", "password": "secret123"}).json()
    assert r["token"]
    assert client.post("/api/search", json={"limit": 1}).status_code == 401
    ok = client.post("/api/search", json={"limit": 1},
                     headers={"Authorization": f"Bearer {r['token']}"})
    assert ok.status_code == 200


def test_credentials_are_validated(client):
    client.post("/api/auth/register", json={"username": "u1", "password": "secret123"})
    assert client.post("/api/auth/register",
                       json={"username": "u1", "password": "secret123"}).status_code == 409
    assert client.post("/api/auth/register",
                       json={"username": "x", "password": "secret123"}).status_code == 400
    assert client.post("/api/auth/register",
                       json={"username": "ok", "password": "123"}).status_code == 400
    assert client.post("/api/auth/login",
                       json={"username": "u1", "password": "wrong"}).status_code == 401


def test_users_cannot_read_each_others_taste(client, paths):
    a = client.post("/api/auth/register", json={"username": "aaa", "password": "secret123"}).json()
    b = client.post("/api/auth/register", json={"username": "bbb", "password": "secret123"}).json()
    ha = {"Authorization": f"Bearer {a['token']}"}
    hb = {"Authorization": f"Bearer {b['token']}"}
    bb = seed_library(paths["db"], paths["feats"], "bbb", tag="b")

    a_ids = [r["face_id"] for r in
             client.post("/api/search", json={"limit": 50}, headers=ha).json()["results"]]
    client.post("/api/feedback", json={"face_id": a_ids[0], "kind": "dislike",
                                       "undo_seconds": 0}, headers=ha)
    assert a_ids[0] not in [r["face_id"] for r in
                            client.post("/api/search", json={"limit": 50},
                                        headers=ha).json()["results"]]
    # bbb's own library is untouched by what aaa rejected
    b_ids = [r["face_id"] for r in
             client.post("/api/search", json={"limit": 50}, headers=hb).json()["results"]]
    assert sorted(b_ids) == sorted(bb)
    # and a client cannot borrow another profile by naming it in the request body
    spoof = client.post("/api/search",
                        json={"limit": 50, "owner": "aaa", "personalisation": {"user": "aaa"}},
                        headers=hb).json()
    assert sorted(r["face_id"] for r in spoof["results"]) == sorted(bb), \
        "the server must override both the owner and the profile named by the client"


def test_one_library_is_invisible_to_another_account(client, paths):
    """The tenancy boundary, stated directly: an image uploaded by one person must not
    appear in anybody else's results, crops, face details or statistics."""
    a = client.post("/api/auth/register", json={"username": "aaa", "password": "secret123"}).json()
    b = client.post("/api/auth/register", json={"username": "bbb", "password": "secret123"}).json()
    ha = {"Authorization": f"Bearer {a['token']}"}
    hb = {"Authorization": f"Bearer {b['token']}"}

    a_face = client.post("/api/search", json={"limit": 1},
                         headers=ha).json()["results"][0]["face_id"]
    a_img = client.get(f"/api/face/{a_face}", headers=ha).json()["image_id"]

    assert client.post("/api/search", json={"limit": 50}, headers=hb).json()["results"] == []
    assert client.get("/api/stats", headers=hb).json()["faces"] == 0
    # 404, not 403: a 403 would confirm the id exists, which is a fact about aaa's library
    assert client.get(f"/api/face/{a_face}", headers=hb).status_code == 404
    assert client.get(f"/api/crop/{a_face}?t={b['token']}").status_code == 404
    assert client.get(f"/api/image/{a_img}?t={b['token']}").status_code == 404
    assert client.post("/api/feedback",
                       json={"face_id": a_face, "kind": "like"}, headers=hb).status_code == 404


def test_saved_searches_are_private(client):
    a = client.post("/api/auth/register", json={"username": "aaa", "password": "secret123"}).json()
    b = client.post("/api/auth/register", json={"username": "bbb", "password": "secret123"}).json()
    ha = {"Authorization": f"Bearer {a['token']}"}
    hb = {"Authorization": f"Bearer {b['token']}"}
    client.post("/api/searches", json={"name": "mine", "spec": {"limit": 1}}, headers=ha)
    assert [x["name"] for x in client.get("/api/searches", headers=ha).json()] == ["mine"]
    assert client.get("/api/searches", headers=hb).json() == []
    # the same name in two accounts is two different saved searches, not a collision
    assert client.post("/api/searches", json={"name": "mine", "spec": {"limit": 9}},
                       headers=hb).status_code == 200


# ---------------------------------------------------------------------- admin

def test_first_account_becomes_the_administrator(client):
    r = client.post("/api/auth/register",
                    json={"username": "boss", "password": "secret123"}).json()
    h = {"Authorization": f"Bearer {r['token']}"}
    assert client.get("/api/auth/me", headers=h).json()["is_admin"] is True
    assert client.get("/api/auth/status").json()["admins"] == ["boss"]


def test_second_account_is_not_an_administrator(client):
    a = client.post("/api/auth/register", json={"username": "boss", "password": "secret123"}).json()
    b = client.post("/api/auth/register", json={"username": "bob", "password": "secret123"}).json()
    hb = {"Authorization": f"Bearer {b['token']}"}
    assert client.get("/api/auth/me", headers=hb).json()["is_admin"] is False
    assert client.get("/api/admin/users", headers=hb).status_code == 403


def test_admin_can_create_and_reset(client):
    a = client.post("/api/auth/register", json={"username": "boss", "password": "secret123"}).json()
    ha = {"Authorization": f"Bearer {a['token']}"}
    assert client.post("/api/admin/users",
                       json={"username": "newbie", "password": "temp1234"},
                       headers=ha).status_code == 200
    tok = client.post("/api/auth/login",
                      json={"username": "newbie", "password": "temp1234"}).json()["token"]
    hn = {"Authorization": f"Bearer {tok}"}
    assert client.get("/api/auth/me", headers=hn).json()["must_change_password"] is True
    # an admin reset must invalidate the user's live sessions
    client.post("/api/auth/password",
                json={"username": "newbie", "new_password": "reset9999"}, headers=ha)
    assert client.get("/api/auth/me", headers=hn).status_code == 401


def test_self_service_password_change_requires_the_current_one(client):
    a = client.post("/api/auth/register", json={"username": "boss", "password": "secret123"}).json()
    ha = {"Authorization": f"Bearer {a['token']}"}
    assert client.post("/api/auth/password",
                       json={"current_password": "wrong", "new_password": "another1"},
                       headers=ha).status_code == 401
    r = client.post("/api/auth/password",
                    json={"current_password": "secret123", "new_password": "another1"},
                    headers=ha)
    assert r.status_code == 200 and r.json()["token"]


def test_admin_guard_rails(client):
    a = client.post("/api/auth/register", json={"username": "boss", "password": "secret123"}).json()
    ha = {"Authorization": f"Bearer {a['token']}"}
    assert client.delete("/api/admin/users/boss", headers=ha).status_code == 400
    assert client.post("/api/admin/role",
                       json={"username": "boss", "make_admin": False},
                       headers=ha).status_code == 400


def test_deleting_a_user_removes_their_library_and_preferences(client, paths):
    a = client.post("/api/auth/register", json={"username": "boss", "password": "secret123"}).json()
    b = client.post("/api/auth/register", json={"username": "bob", "password": "secret123"}).json()
    ha = {"Authorization": f"Bearer {a['token']}"}
    hb = {"Authorization": f"Bearer {b['token']}"}
    seed_library(paths["db"], paths["feats"], "bob", tag="b")
    fid = client.post("/api/search", json={"limit": 1}, headers=hb).json()["results"][0]["face_id"]
    client.post("/api/feedback",
                json={"face_id": fid, "kind": "like", "undo_seconds": 0}, headers=hb)
    assert client.delete("/api/admin/users/bob", headers=ha).status_code == 200
    # deletion must actually delete (docs/LICENSING.md 4.2)
    assert "bob" not in [u["username"] for u in
                         client.get("/api/admin/users", headers=ha).json()["users"]]
    ix = Index(paths["db"])
    try:
        assert ix.conn.execute("SELECT COUNT(*) FROM images WHERE owner='bob'").fetchone()[0] == 0
        assert ix.conn.execute("SELECT COUNT(*) FROM feedback WHERE user='bob'").fetchone()[0] == 0
    finally:
        ix.close()


def test_self_service_deletion_erases_everything(client, paths):
    r = client.post("/api/auth/register",
                    json={"username": "solo", "password": "secret123"}).json()
    h = {"Authorization": f"Bearer {r['token']}"}
    # the only administrator cannot delete themselves and strand the deployment
    assert client.post("/api/account/delete",
                       json={"confirm": "delete", "password": "secret123"},
                       headers=h).status_code == 400
    client.post("/api/auth/register", json={"username": "other", "password": "secret123"})
    client.post("/api/admin/role", json={"username": "other", "make_admin": True}, headers=h)
    assert client.post("/api/account/delete",
                       json={"confirm": "nope", "password": "secret123"},
                       headers=h).status_code == 400
    assert client.post("/api/account/delete",
                       json={"confirm": "delete", "password": "wrong"},
                       headers=h).status_code == 401
    assert client.post("/api/account/delete",
                       json={"confirm": "delete", "password": "secret123"},
                       headers=h).status_code == 200
    assert client.get("/api/auth/me", headers=h).status_code == 401


def test_account_export_lists_what_is_held(client):
    r = client.post("/api/auth/register",
                    json={"username": "solo", "password": "secret123"}).json()
    h = {"Authorization": f"Bearer {r['token']}"}
    d = client.get("/api/account/export", headers=h).json()
    for key in ("account", "library", "judgements", "reference_faces", "saved_searches"):
        assert key in d, f"export is missing {key}"
    assert len(d["library"]) == 3


def test_auth_401_reports_the_real_reason(client):
    """A wrong password must say so. The UI used to intercept every 401 and report
    'signed out', which describes the wrong problem entirely."""
    client.post("/api/auth/register", json={"username": "boss", "password": "secret123"})
    r = client.post("/api/auth/login", json={"username": "boss", "password": "wrong"})
    assert r.status_code == 401
    assert "wrong username or password" in r.json()["detail"]


def test_error_details_are_human_readable(client):
    client.post("/api/auth/register", json={"username": "boss", "password": "secret123"})
    for payload, expect in (
        ({"username": "boss", "password": "secret123"}, "taken"),
        ({"username": "ok", "password": "abc"}, "at least 8"),
        ({"username": "x", "password": "secret123"}, "2-32"),
    ):
        d = client.post("/api/auth/register", json=payload).json()["detail"]
        assert expect in d, f"unhelpful message for {payload}: {d}"


# ------------------------------------------------------- authentication surface

def test_no_endpoint_leaks_face_data_without_a_session(client):
    """Face crops, source images, predictions and stats were all anonymously readable.
    A client-side gate is decoration; this is the actual boundary."""
    client.post("/api/auth/register", json={"username": "boss", "password": "secret123"})
    for method, path in [
        ("get", "/api/stats"), ("get", "/api/face/1"),
        ("get", "/api/crop/1"), ("get", "/api/image/1"),
        ("get", "/api/searches"), ("get", "/api/index/status"),
        ("get", "/api/preference"), ("get", "/api/admin/users"),
    ]:
        r = getattr(client, method)(path)
        assert r.status_code == 401, f"{path} answered {r.status_code} with no credentials"
    assert client.post("/api/search", json={"limit": 1}).status_code == 401
    assert client.post("/api/index", json={"roots": ["/tmp"]}).status_code == 401


def test_media_accepts_the_token_as_a_query_parameter(client):
    """<img src> cannot set headers, so media takes ?t=<token> - the same token, not a
    weaker side door."""
    r = client.post("/api/auth/register",
                    json={"username": "boss", "password": "secret123"}).json()
    assert client.get(f"/api/crop/1?t={r['token']}").status_code in (200, 410)
    assert client.get("/api/crop/1?t=not-a-real-token").status_code == 401


def test_model_status_reports_the_whole_stack(client):
    """A ranking that cannot say which models produced it is not auditable."""
    r = client.post("/api/auth/register",
                    json={"username": "boss", "password": "secret123"}).json()
    d = client.get("/api/models", headers={"Authorization": f"Bearer {r['token']}"}).json()
    for key in ("detector", "encoder", "attractiveness", "age_gender", "quality",
                "your_model", "faces_indexed"):
        assert key in d, f"model status missing {key}"
    # every component says which experiment chose it
    for key in ("detector", "encoder", "attractiveness", "age_gender", "quality"):
        assert d[key].get("decided_by"), f"{key} does not record why it was chosen"
    assert d["attractiveness"]["trained_on"].startswith("SCUT-FBP5500")
    assert d["your_model"]["trained"] is False


def test_model_status_requires_a_session(client):
    client.post("/api/auth/register", json={"username": "boss", "password": "secret123"})
    assert client.get("/api/models").status_code == 401


# --------------------------------------------------------------- undo, reworked

def test_only_one_judgement_is_undoable_at_a_time(client):
    """Marking a second face closes the first one's window.

    The old behaviour queued a countdown per judgement, which made undo something you had
    to chase and left the taste model several decisions behind the user.
    """
    ids = [r["face_id"] for r in client.post("/api/search", json={"limit": 3}).json()["results"]]
    first = client.post("/api/feedback",
                        json={"face_id": ids[0], "kind": "dislike", "undo_seconds": 60}).json()
    assert first["undo"]["face_id"] == ids[0]
    assert first["committed_face_id"] is None

    second = client.post("/api/feedback",
                         json={"face_id": ids[1], "kind": "like", "undo_seconds": 60}).json()
    assert second["committed_face_id"] == ids[0], "marking another face commits the first"
    assert client.get("/api/feedback/undo").json()["undo"]["face_id"] == ids[1]

    # the first judgement has now taught the model; undoing it is no longer free
    assert client.get("/api/preference").json()["trained"] is True


def test_undoing_the_open_judgement_leaves_no_trace(client):
    ids = [r["face_id"] for r in client.post("/api/search", json={"limit": 3}).json()["results"]]
    client.post("/api/feedback", json={"face_id": ids[0], "kind": "dislike",
                                       "undo_seconds": 60})
    client.post("/api/feedback/undo", json={"face_id": ids[0]})
    assert client.get("/api/feedback/undo").json()["undo"] is None
    assert client.get("/api/preference").json()["trained"] is False
    assert ids[0] in [r["face_id"] for r in
                      client.post("/api/search", json={"limit": 50}).json()["results"]]


def test_commit_closes_the_window_on_demand(client):
    """The client calls this when the card leaves the screen, so a judgement is never left
    half-made if the user simply walks away."""
    fid = client.post("/api/search", json={"limit": 1}).json()["results"][0]["face_id"]
    client.post("/api/feedback", json={"face_id": fid, "kind": "like", "undo_seconds": 60})
    assert client.post("/api/feedback/commit").json()["committed"] == 1
    assert client.get("/api/feedback/undo").json()["undo"] is None
    assert client.get("/api/preference").json()["trained"] is True


def test_undo_window_is_bounded(client):
    """A client asking for a ten-hour undo would be asking for a judgement that never
    teaches anything."""
    fid = client.post("/api/search", json={"limit": 1}).json()["results"][0]["face_id"]
    r = client.post("/api/feedback",
                    json={"face_id": fid, "kind": "like", "undo_seconds": 99999}).json()
    assert r["undo"]["seconds"] == 120.0


def test_feedback_on_someone_elses_face_is_a_404(client, paths):
    a = client.post("/api/auth/register", json={"username": "aaa", "password": "secret123"}).json()
    b = client.post("/api/auth/register", json={"username": "bbb", "password": "secret123"}).json()
    ha = {"Authorization": f"Bearer {a['token']}"}
    hb = {"Authorization": f"Bearer {b['token']}"}
    fid = client.post("/api/search", json={"limit": 1}, headers=ha).json()["results"][0]["face_id"]
    assert client.post("/api/feedback", json={"face_id": fid, "kind": "like"},
                       headers=hb).status_code == 404
    assert client.post("/api/feedback/undo", json={"face_id": fid},
                       headers=hb).status_code == 404


# ------------------------------------------------------------------- rate limits

def test_repeated_bad_passwords_are_throttled(client):
    client.post("/api/auth/register", json={"username": "boss", "password": "secret123"})
    codes = [client.post("/api/auth/login",
                         json={"username": "boss", "password": f"wrong{i}"}).status_code
             for i in range(12)]
    assert 429 in codes, "an unthrottled password endpoint is a free guessing oracle"
    # a correct password is still refused while the window is open, by design
    assert client.post("/api/auth/login",
                       json={"username": "boss", "password": "secret123"}).status_code == 429


# ------------------------------------------------------------- security headers

def test_responses_carry_framing_and_sniffing_defences(client):
    h = client.get("/").headers
    assert h["X-Frame-Options"] == "DENY"
    assert h["X-Content-Type-Options"] == "nosniff"
    assert "frame-ancestors 'none'" in h["Content-Security-Policy"]
    assert h["Referrer-Policy"] == "no-referrer"


def test_hosted_deployments_have_no_open_mode(paths, monkeypatch):
    """A configured public URL means strangers can reach this. Open mode - a signed-in
    session for anyone who loads the page - stops being available."""
    monkeypatch.setenv("FACET_PUBLIC_URL", "https://facet.example")
    seed_library(paths["db"], paths["feats"], "default")
    c = TestClient(create_app(str(paths["db"]), str(paths["feats"]),
                              upload_root=str(paths["uploads"])))
    assert c.get("/api/auth/status").json()["open_mode"] is False
    assert c.post("/api/search", json={"limit": 1}).status_code == 401


def test_google_sign_in_is_advertised_only_when_configured(client, paths, monkeypatch):
    assert client.get("/api/auth/status").json()["google"]["enabled"] is False
    assert client.get("/api/auth/google/start").status_code == 503
    monkeypatch.setenv("FACET_GOOGLE_CLIENT_ID", "cid")
    monkeypatch.setenv("FACET_GOOGLE_CLIENT_SECRET", "sec")
    monkeypatch.setenv("FACET_PUBLIC_URL", "https://facet.example")
    c = TestClient(create_app(str(paths["db"]), str(paths["feats"]),
                              upload_root=str(paths["uploads"])))
    assert c.get("/api/auth/status").json()["google"]["enabled"] is True
    r = c.get("/api/auth/google/start", follow_redirects=False)
    assert r.status_code == 302 and r.headers["location"].startswith(
        "https://accounts.google.com/")


def test_google_start_refuses_an_offsite_next(paths, monkeypatch):
    """An open redirect on a sign-in route lends a real domain's credibility to a
    phishing page."""
    monkeypatch.setenv("FACET_GOOGLE_CLIENT_ID", "cid")
    monkeypatch.setenv("FACET_GOOGLE_CLIENT_SECRET", "sec")
    monkeypatch.setenv("FACET_PUBLIC_URL", "https://facet.example")
    c = TestClient(create_app(str(paths["db"]), str(paths["feats"]),
                              upload_root=str(paths["uploads"])))
    for bad in ("https://evil.example", "//evil.example"):
        r = c.get(f"/api/auth/google/start?next={bad}", follow_redirects=False)
        assert "evil.example" not in r.headers["location"]


# ------------------------------------------------------- the import job's lock

def test_index_job_lock_is_reentrant():
    """A nested helper inside the worker once shadowed `set`, so `queued.pop(owner, set())`
    written inside `with self.lock:` re-entered the same non-reentrant lock and deadlocked
    the entire server - every request, not just the import. The rename fixed that instance;
    re-entrancy removes the whole class of failure."""
    import threading

    from facet.api.app import IndexJob
    job = IndexJob()
    assert isinstance(job.lock, type(threading.RLock()))
    with job.lock:
        with job.lock:                      # would hang forever on a plain Lock
            job.states["x"] = {"status": "running"}
    assert job.state_for("x")["status"] == "running"


def test_index_job_worker_defines_no_shadowing_helpers():
    """Cheap source guard: the worker must not name a helper after a builtin it also uses."""
    import inspect
    import re

    from facet.api import app as appmod
    src = inspect.getsource(appmod.IndexJob.start)
    assert not re.search(r"^\s+def (set|list|dict|type|id)\(", src, re.M), \
        "the worker shadows a builtin it relies on elsewhere"


def test_a_second_import_is_queued_not_refused(client, monkeypatch):
    """Uploading again while the first batch is still indexing must not 409: the bytes are
    already stored by then, so a refusal would leave them permanently unindexed."""
    from facet.api.app import IndexJob
    job = IndexJob()
    job.threads["u"] = type("T", (), {"is_alive": lambda self: True})()
    out = job.start("db", "feats", ["/a"], 0, False, owner="u")
    assert out["status"] == "queued"
    assert job.queued["u"] == {"/a"}
