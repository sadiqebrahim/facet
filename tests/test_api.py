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


@pytest.fixture
def client(tmp_path):
    """A synthetic index WITH a real feature store.

    The preference model is fitted over cached embeddings, so a fixture without a feature
    store silently disables personalisation and any test of it would pass vacuously.
    """
    import numpy as np

    from facet.pipeline.store import FeatureStore

    db = tmp_path / "api.db"
    feats = tmp_path / "feats"
    ENC, CROP, DIM = "test-enc", "test-crop", 16
    store = FeatureStore(feats, ENC, CROP, dim=DIM)
    rng = np.random.default_rng(0)
    store.append(rng.normal(size=(3, DIM)).astype(np.float32))
    ix = Index(db)
    for i, (beauty, age, pf, ood) in enumerate([
        (4.5, 30, 0.99, False), (2.1, 55, 0.02, False), (4.2, 28, 0.95, True),
    ]):
        iid = ix.upsert_image(path=f"/img{i}.jpg", content_hash=f"h{i}", status="ok",
                              n_faces=1, width=200, height=200)
        fid = ix.insert_face(image_id=iid, face_idx=0, x1=10, y1=10, x2=110, y2=110,
                             det_score=0.9, face_px=100, quality=0.6, feature_row=i,
                             quality_json='{"blur": 500.0}',
                             encoder_version=ENC, crop_version=CROP)
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
    ix.close()
    return TestClient(create_app(str(db), str(feats)))


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
    assert top["path"] == "/img0.jpg"
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


def test_index_rejects_a_nonexistent_directory(client):
    assert client.post("/api/index", json={"roots": ["/definitely/not/here"]}).status_code == 400


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


def test_reference_paths_are_validated(client):
    r = client.post("/api/preference/references",
                    json={"paths": ["/definitely/not/a/real/image.jpg"], "kind": "like"})
    assert r.status_code == 200
    assert r.json()["added"] == 0 and r.json()["skipped"]


def test_reference_rejects_bad_kind(client):
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
    assert r["commit_at"] is not None
    assert client.get("/api/feedback/pending").json()["pending"], "should be undoable"
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
    assert client.get("/api/feedback/pending").json()["pending"] == []
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


def test_users_cannot_read_each_others_taste(client):
    a = client.post("/api/auth/register", json={"username": "aaa", "password": "secret123"}).json()
    b = client.post("/api/auth/register", json={"username": "bbb", "password": "secret123"}).json()
    ha = {"Authorization": f"Bearer {a['token']}"}
    hb = {"Authorization": f"Bearer {b['token']}"}
    fid = client.post("/api/search", json={"limit": 1}, headers=ha).json()["results"][0]["face_id"]
    client.post("/api/feedback", json={"face_id": fid, "kind": "dislike", "undo_seconds": 0},
                headers=ha)
    assert fid not in [r["face_id"] for r in
                       client.post("/api/search", json={"limit": 50}, headers=ha).json()["results"]]
    assert fid in [r["face_id"] for r in
                   client.post("/api/search", json={"limit": 50}, headers=hb).json()["results"]]
    # and a client cannot borrow another profile by naming it in the request body
    spoof = client.post("/api/search",
                        json={"limit": 50, "personalisation": {"user": "aaa"}},
                        headers=hb).json()
    assert fid in [r["face_id"] for r in spoof["results"]], "server must override the user"
