"""Dataset loader tests against counts verified from the dataset's own paper/README."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from facet.data.scut_fbp5500 import ScutFbp5500

DATA = ROOT / "data/raw/SCUT-FBP5500_v2"
pytestmark = pytest.mark.skipif(not DATA.exists(), reason="SCUT-FBP5500 not downloaded")


@pytest.fixture(scope="module")
def ds():
    return ScutFbp5500(DATA)


def test_image_count(ds):
    assert len(ds.filenames) == 5500


def test_subgroup_counts_match_the_paper(ds):
    counts = ds.labels["subgroup"].value_counts().to_dict()
    assert counts == {"AF": 2000, "AM": 2000, "CF": 750, "CM": 750}


def test_every_image_rated_by_all_sixty_raters(ds):
    assert ds.ratings["Rater"].nunique() == 60
    assert len(ds.ratings) == 5500 * 60
    assert (ds.labels["n_ratings"] == 60).all()


def test_rating_histograms_are_normalised(ds):
    P = ds.histogram_matrix(ds.filenames)
    assert P.shape == (5500, 5)
    np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-5)


def test_expectation_of_histogram_equals_mean_score(ds):
    """The LDL target and the regression target must be mutually consistent."""
    P = ds.histogram_matrix(ds.filenames).astype(np.float64)
    expectation = P @ np.array([1.0, 2, 3, 4, 5])
    np.testing.assert_allclose(expectation, ds.target(ds.filenames, "mean"), atol=1e-4)


def test_rater_disagreement_matches_the_published_range(ds):
    """The paper states per-image rating std is mostly within [0.6, 0.7]. Verify it."""
    assert 0.60 <= ds.labels["std"].median() <= 0.70


def test_official_splits_are_complete_and_disjoint(ds):
    for name, split in ds.splits.items():
        assert not (set(split.train) & set(split.test)), f"{name} train/test overlap"
        assert len(set(split.train) | set(split.test)) == 5500, f"{name} does not cover all images"
    assert (len(ds.splits["cv1"].train), len(ds.splits["cv1"].test)) == (4400, 1100)
    assert (len(ds.splits["split6040"].train), len(ds.splits["split6040"].test)) == (3300, 2200)


def test_landmarks_shape_and_the_one_known_bad_file(ds):
    assert ds.landmarks("AF1.jpg").shape == (86, 2)
    lms, valid = ds.all_landmarks(ds.filenames)
    assert lms.shape == (5500, 86, 2)
    # Exactly one file in the release is unusable - documented in docs/DATASETS.md.
    bad = [n for n, v in zip(ds.filenames, valid) if not v]
    assert bad == ["CM152.jpg"]
