"""Per-user preference model tests.

E14's findings are the spec here: the residual blend must never fully replace the population
model, a single example must still work (cold start), and more labels must mean more
personal weight.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from facet.models.preference import MAX_ALPHA, PreferenceModel, blend


def taste(d=32, seed=0):
    """A synthetic user who likes vectors near +axis0 and dislikes those near -axis0."""
    rng = np.random.default_rng(seed)
    ax = np.zeros(d); ax[0] = 1
    return ax, rng


def test_cold_start_from_a_single_example():
    ax, rng = taste()
    m = PreferenceModel(32).fit(ax[None, :], np.zeros((0, 32)))
    assert m.method == "centroid", "one example must not fit a linear model"
    assert m.alpha() < 0.1, "one example must barely move the ranking"
    liked = m.score((ax + rng.normal(0, .1, (4, 32))))
    other = m.score((-ax + rng.normal(0, .1, (4, 32))))
    assert liked.mean() > other.mean()


def test_switches_to_ridge_once_both_classes_have_enough():
    ax, rng = taste()
    m = PreferenceModel(32).fit(ax + rng.normal(0, .3, (5, 32)),
                                -ax + rng.normal(0, .3, (5, 32)))
    assert m.method == "ridge"
    assert m.score((ax + rng.normal(0, .2, (5, 32)))).mean() > \
           m.score((-ax + rng.normal(0, .2, (5, 32)))).mean()


def test_alpha_grows_with_labels_but_is_capped():
    ax, rng = taste()
    a = []
    for n in (2, 10, 50, 500):
        m = PreferenceModel(32).fit(ax + rng.normal(0, .3, (n, 32)),
                                    -ax + rng.normal(0, .3, (n, 32)))
        a.append(m.alpha())
    assert a == sorted(a), "more labels must mean more personal weight"
    assert a[-1] <= MAX_ALPHA
    assert a[-1] < 1.0, "the population model must never be fully replaced (E14)"


def test_untaught_model_has_no_influence():
    m = PreferenceModel(32).fit(np.zeros((0, 32)), np.zeros((0, 32)))
    assert m.method == "none" and m.alpha() == 0.0
    np.testing.assert_allclose(m.score(np.random.rand(3, 32)), 0)


def test_blend_is_a_convex_combination():
    pop = np.array([0.9, 0.1]); per = np.array([0.1, 0.9])
    np.testing.assert_allclose(blend(pop, per, 0.0), pop)
    np.testing.assert_allclose(blend(pop, per, 1.0), per)
    np.testing.assert_allclose(blend(pop, per, 0.5), [0.5, 0.5])


def test_dislikes_push_scores_down():
    ax, rng = taste()
    liked = ax + rng.normal(0, .2, (5, 32))
    with_dis = PreferenceModel(32).fit(liked, -ax + rng.normal(0, .2, (5, 32)))
    probe = -ax + rng.normal(0, .1, (5, 32))
    without = PreferenceModel(32).fit(liked, np.zeros((0, 32)))
    assert with_dis.score(probe).mean() < without.score(probe).mean()


def test_roundtrip_serialisation(tmp_path):
    ax, rng = taste()
    m = PreferenceModel(32).fit(ax + rng.normal(0, .3, (6, 32)),
                                -ax + rng.normal(0, .3, (6, 32)))
    p = tmp_path / "pref.npz"; m.save(p)
    back = PreferenceModel.load(p)
    probe = rng.normal(size=(5, 32))
    np.testing.assert_allclose(m.score(probe), back.score(probe), rtol=1e-9)
    assert back.method == m.method and back.alpha() == m.alpha()


def test_status_explains_itself():
    ax, rng = taste()
    for n, expect in ((1, "Learning"), (40, "Blending")):
        m = PreferenceModel(32).fit(ax + rng.normal(0, .3, (n, 32)),
                                    -ax + rng.normal(0, .3, (n, 32)) if n > 2
                                    else np.zeros((0, 32)))
        assert expect.lower() in m.status().note.lower()


def test_dislike_only_still_steers_the_ranking():
    """Rejecting is far less effort than curating examples, so many users will only ever
    press ✕. That must move the ranking, not do nothing."""
    ax, rng = taste()
    m = PreferenceModel(32).fit(np.zeros((0, 32)), ax + rng.normal(0, .2, (4, 32)))
    assert m.method == "avoid"
    assert m.alpha() > 0, "rejections alone must still influence ranking"
    away = m.score(-ax + rng.normal(0, .1, (5, 32))).mean()
    toward = m.score(ax + rng.normal(0, .1, (5, 32))).mean()
    assert away > toward
    assert "rejected" in m.status().note.lower()
