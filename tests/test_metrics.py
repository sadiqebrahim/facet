"""Metric tests against analytically known values.

An evaluation harness that cannot reproduce a known answer cannot be trusted to measure
a new one.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from facet.evaluation.metrics import (
    calibration_metrics,
    kl_divergence,
    ndcg_at_k,
    pairwise_accuracy,
    precision_at_k,
    regression_metrics,
)


def test_perfect_prediction():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    m = regression_metrics(y, y)
    assert m["mae"] == pytest.approx(0.0)
    assert m["rmse"] == pytest.approx(0.0)
    assert m["pc"] == pytest.approx(1.0)
    assert m["spearman"] == pytest.approx(1.0)


def test_known_errors():
    y = np.array([1.0, 2.0, 3.0])
    p = np.array([2.0, 3.0, 4.0])  # constant +1 offset
    m = regression_metrics(y, p)
    assert m["mae"] == pytest.approx(1.0)
    assert m["rmse"] == pytest.approx(1.0)
    assert m["pc"] == pytest.approx(1.0)  # offset does not change correlation


def test_pairwise_accuracy_bounds():
    rng = np.random.default_rng(0)
    y = rng.normal(size=500)
    assert pairwise_accuracy(y, y, n_pairs=5000) == pytest.approx(1.0)
    assert pairwise_accuracy(y, -y, n_pairs=5000) == pytest.approx(0.0)


def test_ndcg_perfect_and_reversed():
    y = np.arange(50, dtype=float)
    assert ndcg_at_k(y, y, k=10) == pytest.approx(1.0)
    assert ndcg_at_k(y, -y, k=10) < 0.5


def test_precision_at_k():
    y = np.arange(100, dtype=float)
    # A perfect ranker's top 10 are exactly the true top 10 percent.
    assert precision_at_k(y, y, k=10, quantile=0.9) == pytest.approx(1.0)


def test_kl_divergence_zero_for_identical():
    p = np.array([[0.1, 0.2, 0.4, 0.2, 0.1]])
    assert kl_divergence(p, p) == pytest.approx(0.0, abs=1e-9)
    q = np.array([[0.2, 0.2, 0.2, 0.2, 0.2]])
    assert kl_divergence(p, q) > 0


def test_calibration_of_correctly_specified_gaussian():
    """A correctly specified sigma must give ~68% / ~95% coverage."""
    rng = np.random.default_rng(42)
    n, sigma = 20000, 0.5
    y = rng.normal(3.0, 1.0, n)
    pred = y + rng.normal(0, sigma, n)
    m = calibration_metrics(y, pred, np.full(n, sigma))
    assert m["coverage_68"] == pytest.approx(0.68, abs=0.02)
    assert m["coverage_95"] == pytest.approx(0.95, abs=0.02)
