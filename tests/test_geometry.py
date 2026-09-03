import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from facet.models.geometry import generalized_procrustes, geometry_features, symmetry_residual


def _square():
    return np.array([[-1.0, -1], [1, -1], [1, 1], [-1, 1]])


def test_procrustes_is_invariant_to_similarity_transforms():
    """Shape features must not change under translation, scale or rotation."""
    base = np.random.default_rng(0).normal(size=(20, 2))
    theta = 0.7
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    transformed = (base * 3.5) @ R.T + np.array([10.0, -4.0])
    aligned, _ = generalized_procrustes(np.stack([base, transformed]), iters=5)
    # Both shapes should collapse onto (nearly) the same canonical form.
    assert np.abs(aligned[0] - aligned[1]).max() < 1e-6


def test_symmetric_shape_has_low_residual():
    sym = _square()
    asym = np.array([[-1.0, -1], [1, -1], [1, 1], [-0.2, 1]])
    assert symmetry_residual(sym) < symmetry_residual(asym)


def test_feature_matrix_shape():
    lms = np.random.default_rng(1).normal(size=(7, 86, 2))
    feats, names = geometry_features(lms)
    assert feats.shape == (7, 86 * 2 + 6)
    assert len(names) == feats.shape[1]
    assert np.isfinite(feats).all()
