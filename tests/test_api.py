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
    db = tmp_path / "api.db"
    ix = Index(db)
    for i, (beauty, age, pf, ood) in enumerate([
        (4.5, 30, 0.99, False), (2.1, 55, 0.02, False), (4.2, 28, 0.95, True),
    ]):
        iid = ix.upsert_image(path=f"/img{i}.jpg", content_hash=f"h{i}", status="ok",
                              n_faces=1, width=200, height=200)
        fid = ix.insert_face(image_id=iid, face_idx=0, x1=10, y1=10, x2=110, y2=110,
                             det_score=0.9, face_px=100, quality=0.6, feature_row=i,
                             quality_json='{"blur": 500.0}',
                             encoder_version="e", crop_version="c")
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
    return TestClient(create_app(str(db), str(tmp_path / "feats")))


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
