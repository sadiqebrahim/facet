"""Pipeline tests on a purpose-built fixture directory.

The fixture deliberately contains the cases a real photo directory contains and a happy-path
test would miss: an exact duplicate, a corrupt file, a file too small to be a photograph, and
a file with no detectable face.
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from facet.pipeline.db import Index
from facet.pipeline.discovery import IMAGE_EXTENSIONS, load_image, plan_scan
from facet.pipeline.duplicates import near_duplicate_groups, union_find
from facet.pipeline.store import FeatureStore

FF = ROOT / "data/raw/FairFace/val_images"
needs_data = pytest.mark.skipif(not FF.exists(), reason="FairFace not downloaded")


@pytest.fixture(scope="module")
def fixture_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("photos")
    srcs = sorted(FF.glob("*.jpg"))[:4] if FF.exists() else []
    for i, s in enumerate(srcs):
        shutil.copy(s, d / f"photo_{i}.jpg")
    if srcs:
        shutil.copy(srcs[0], d / "exact_duplicate.jpg")   # same bytes, different name
    (d / "corrupt.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"garbage" * 500)
    (d / "tiny.jpg").write_bytes(b"\xff\xd8")             # below MIN_BYTES
    (d / "notes.txt").write_text("not an image")
    return d


def test_discovery_finds_only_images_and_skips_tiny(fixture_dir):
    plan = plan_scan([fixture_dir], {})
    names = {Path(c.path).name for c in plan.new}
    assert "notes.txt" not in names, "non-image extension was picked up"
    assert "tiny.jpg" in {Path(p).name for p in plan.too_small}
    assert "corrupt.jpg" in names, "corrupt files must be attempted, then recorded"


def test_discovery_is_incremental(fixture_dir):
    plan = plan_scan([fixture_dir], {})
    fps = {c.path: (c.mtime, c.size_bytes, "ok") for c in plan.new}
    again = plan_scan([fixture_dir], fps)
    assert again.new == [] and again.changed == []
    assert len(again.unchanged) == len(plan.new)


def test_discovery_detects_modification(fixture_dir, tmp_path):
    d = tmp_path / "mod"
    d.mkdir()
    f = d / "a.jpg"
    f.write_bytes(b"\xff\xd8" + b"x" * 4000)
    plan = plan_scan([d], {})
    fps = {c.path: (c.mtime, c.size_bytes, "ok") for c in plan.new}
    f.write_bytes(b"\xff\xd8" + b"y" * 8000)          # size changes
    assert len(plan_scan([d], fps).changed) == 1


def test_overlapping_roots_do_not_double_index(fixture_dir):
    plan = plan_scan([fixture_dir, fixture_dir], {})
    paths = [c.path for c in plan.new]
    assert len(paths) == len(set(paths))


def test_corrupt_file_returns_reason_not_exception(fixture_dir):
    img, err = load_image(fixture_dir / "corrupt.jpg")
    assert img is None and err


def test_union_find_groups_transitively():
    labels = union_find(5, [(0, 1), (1, 2), (3, 4)])
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4]
    assert labels[0] != labels[3]


def test_near_duplicate_groups_finds_identical_rows():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(6, 32)).astype(np.float32)
    feats = np.vstack([base, base[:2]])              # rows 6,7 duplicate rows 0,1
    labels = near_duplicate_groups(feats, threshold=0.92)
    assert labels[0] == labels[6]
    assert labels[1] == labels[7]
    assert labels[2] != labels[0]


def test_feature_store_roundtrip_and_growth(tmp_path):
    s = FeatureStore(tmp_path, "enc:v1", "crop:v1", dim=8)
    a = np.random.rand(5, 8).astype(np.float32)
    rows = s.append(a)
    np.testing.assert_allclose(s.take(rows), a, rtol=1e-6)
    b = np.random.rand(3, 8).astype(np.float32)
    rows_b = s.append(b)
    assert list(rows_b) == [5, 6, 7] and len(s) == 8
    reopened = FeatureStore(tmp_path, "enc:v1", "crop:v1", dim=8)
    np.testing.assert_allclose(reopened.take([0, 5]), np.vstack([a[0], b[0]]), rtol=1e-6)


def test_feature_store_rejects_incompatible_dim(tmp_path):
    FeatureStore(tmp_path, "enc:v1", "crop:v1", dim=8)
    with pytest.raises(ValueError):
        FeatureStore(tmp_path, "enc:v1", "crop:v1", dim=16)


def test_feature_store_separates_shards_by_version(tmp_path):
    """A crop-protocol change must start a new shard, not mix incomparable vectors."""
    a = FeatureStore(tmp_path, "enc:v1", "crop:m0.00", dim=4)
    b = FeatureStore(tmp_path, "enc:v1", "crop:m0.25", dim=4)
    a.append(np.ones((2, 4), dtype=np.float32))
    assert len(a) == 2 and len(b) == 0
    assert a.bin != b.bin


def test_index_records_failures_rather_than_dropping_them(tmp_path):
    ix = Index(tmp_path / "t.db")
    ix.upsert_image(path="/bad.jpg", status="corrupt", error="decode failed", n_faces=0)
    assert ix.stats()["by_status"]["corrupt"] == 1
    ix.close()


def test_reindexing_replaces_faces_without_duplicating(tmp_path):
    ix = Index(tmp_path / "t.db")
    iid = ix.upsert_image(path="/a.jpg", status="ok", n_faces=1)
    ix.insert_face(image_id=iid, face_idx=0, x1=0, y1=0, x2=1, y2=1)
    ix.clear_faces(iid)
    ix.insert_face(image_id=iid, face_idx=0, x1=0, y1=0, x2=1, y2=1)
    assert ix.stats()["faces"] == 1
    ix.close()


def test_lazy_queue_requeues_on_version_bump(tmp_path):
    ix = Index(tmp_path / "t.db")
    iid = ix.upsert_image(path="/a.jpg", status="ok", n_faces=1)
    fid = ix.insert_face(image_id=iid, face_idx=0, x1=0, y1=0, x2=1, y2=1)
    assert len(ix.faces_missing_prediction("beauty", "v1")) == 1
    ix.upsert_predictions([{"face_id": fid, "model": "beauty",
                            "model_version": "v1", "value": 3.0}])
    assert len(ix.faces_missing_prediction("beauty", "v1")) == 0
    # a version bump must re-queue exactly the affected faces
    assert len(ix.faces_missing_prediction("beauty", "v2")) == 1
    ix.close()
