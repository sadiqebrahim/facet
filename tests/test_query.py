"""Query engine tests against a synthetic index.

Built from hand-written rows rather than a real index so the expected ranking is known
exactly and the tests need no models or GPU.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from facet.pipeline.db import Index
from facet.query.engine import SearchEngine
from facet.query.scoring import attractiveness_match, percentile_ranks, trapezoid
from facet.query.spec import QuerySpec


def _face(ix, path, beauty, age, p_female, quality=0.6, ood=False, p_ge4=None, gconf=0.95):
    iid = ix.upsert_image(path=path, content_hash=path, status="ok", n_faces=1)
    fid = ix.insert_face(image_id=iid, face_idx=0, x1=0, y1=0, x2=100, y2=100,
                         det_score=0.9, face_px=100, quality=quality, feature_row=fid_counter(),
                         encoder_version="e", crop_version="c")
    rows = [{"face_id": fid, "model": "beauty", "model_version": "v1", "value": beauty,
             "confidence": None if ood else 0.8, "std": 0.3,
             "distribution": [0.1, 0.2, 0.3, 0.2, 0.2],
             "extra": {"p_ge4": p_ge4 if p_ge4 is not None else beauty / 5.0, "ood": ood,
                       "warnings": ["ood"] if ood else [], "aleatoric": 0.5,
                       "epistemic": 0.05, "ood_score": 0.9 if ood else 0.1}}]
    if age is not None:
        rows.append({"face_id": fid, "model": "age", "model_version": "v1", "value": age})
    if p_female is not None:
        rows.append({"face_id": fid, "model": "gender", "model_version": "v1",
                     "value": p_female, "confidence": gconf})
    ix.upsert_predictions(rows)
    return fid


_counter = [0]
def fid_counter():
    _counter[0] += 1
    return _counter[0] - 1


@pytest.fixture
def engine(tmp_path):
    ix = Index(tmp_path / "q.db")
    _counter[0] = 0
    # a spread of ages, genders and attractiveness, plus the awkward cases
    _face(ix, "/a_ideal.jpg", beauty=4.5, age=30, p_female=0.99)
    _face(ix, "/b_old.jpg", beauty=4.4, age=65, p_female=0.98)
    _face(ix, "/c_male.jpg", beauty=4.3, age=30, p_female=0.02)
    _face(ix, "/d_plain.jpg", beauty=2.0, age=30, p_female=0.97)
    _face(ix, "/e_noage.jpg", beauty=4.2, age=None, p_female=None)      # lazy pass gap
    _face(ix, "/f_ood.jpg", beauty=4.6, age=30, p_female=0.99, ood=True)
    _face(ix, "/g_lowq.jpg", beauty=4.4, age=30, p_female=0.99, quality=0.05)
    _face(ix, "/h_uncertain.jpg", beauty=4.1, age=30, p_female=0.55, gconf=0.55)
    ix.close()
    eng = SearchEngine(tmp_path / "q.db")
    yield eng
    eng.close()


def _names(resp):
    return [Path(r.path).name for r in resp.results]


def test_ranking_prefers_the_matching_face(engine):
    spec = QuerySpec.from_dict({"preferences": {
        "age": {"range": [25, 35], "weight": 0.3},
        "gender": {"value": "female", "weight": 0.2},
        "attractiveness": {"min_percentile": 0.5, "weight": 0.5}}, "limit": 10})
    names = _names(engine.search(spec))
    assert names[0] == "a_ideal.jpg"
    # wrong age, wrong gender and low attractiveness must all rank below it
    for n in ("b_old.jpg", "c_male.jpg", "d_plain.jpg"):
        assert names.index(n) > 0


def test_missing_attribute_is_reported_not_treated_as_non_matching(engine):
    spec = QuerySpec.from_dict({"preferences": {"age": {"range": [25, 35], "weight": 1.0}},
                                "limit": 10})
    resp = engine.search(spec)
    assert resp.diagnostics["missing_age"] == 1
    assert "e_noage.jpg" in _names(resp), "a face awaiting the lazy pass must still appear"


def test_required_flag_excludes_and_counts(engine):
    spec = QuerySpec.from_dict({"preferences": {
        "age": {"range": [25, 35], "weight": 1.0, "required": True}}, "limit": 10})
    resp = engine.search(spec)
    assert "e_noage.jpg" not in _names(resp)
    assert resp.diagnostics["excluded_required_missing"] == 1


def test_strict_gender_reports_low_confidence_exclusions(engine):
    spec = QuerySpec.from_dict({"preferences": {
        "gender": {"value": "female", "weight": 1.0, "mode": "strict",
                   "min_confidence": 0.6}}, "limit": 10})
    resp = engine.search(spec)
    assert resp.diagnostics["excluded_low_gender_confidence"] == 1
    assert "h_uncertain.jpg" not in _names(resp)
    assert "c_male.jpg" not in _names(resp)


def test_soft_gender_keeps_uncertain_faces(engine):
    spec = QuerySpec.from_dict({"preferences": {
        "gender": {"value": "female", "weight": 1.0, "mode": "soft"}}, "limit": 10})
    assert "h_uncertain.jpg" in _names(engine.search(spec))


def test_ood_faces_can_be_excluded_and_are_counted(engine):
    spec = QuerySpec.from_dict({"preferences": {"attractiveness": {"min_percentile": 0.0}},
                                "filters": {"exclude_ood": True}, "limit": 10})
    resp = engine.search(spec)
    assert resp.diagnostics["excluded_ood"] == 1
    assert "f_ood.jpg" not in _names(resp)


def test_quality_filter_applies_before_scoring(engine):
    spec = QuerySpec.from_dict({"preferences": {"attractiveness": {"min_percentile": 0.0}},
                                "filters": {"min_quality": 0.3}, "limit": 10})
    assert "g_lowq.jpg" not in _names(engine.search(spec))


def test_every_result_explains_itself(engine):
    spec = QuerySpec.from_dict({"preferences": {
        "age": {"range": [25, 35], "weight": 0.3},
        "gender": {"value": "female", "weight": 0.2},
        "attractiveness": {"min_percentile": 0.5, "weight": 0.5}}, "limit": 3})
    for r in engine.search(spec).results:
        assert {c.criterion for c in r.contributions} == {"age", "gender", "attractiveness"}
        # relevance must equal the weighted sum it reports, not an unrelated number
        expected = sum(c.contribution for c in r.contributions) / 1.0
        assert r.relevance == pytest.approx(expected, abs=1e-9)


def test_pagination_is_contiguous_and_non_overlapping(engine):
    base = {"preferences": {"attractiveness": {"min_percentile": 0.0}}}
    p1 = _names(engine.search(QuerySpec.from_dict({**base, "limit": 3, "offset": 0})))
    p2 = _names(engine.search(QuerySpec.from_dict({**base, "limit": 3, "offset": 3})))
    both = _names(engine.search(QuerySpec.from_dict({**base, "limit": 6, "offset": 0})))
    assert not set(p1) & set(p2)
    assert p1 + p2 == both


def test_sort_keys_change_the_order(engine):
    base = {"preferences": {"attractiveness": {"min_percentile": 0.0}}, "limit": 8}
    by_attr = _names(engine.search(QuerySpec.from_dict({**base, "sort_by": "attractiveness"})))
    by_qual = _names(engine.search(QuerySpec.from_dict({**base, "sort_by": "quality"})))
    assert by_attr != by_qual
    assert by_qual[-1] == "g_lowq.jpg"


def test_confidence_weighting_downweights_uncertain_criteria(engine):
    """An uncertain gender prediction should contribute less than a confident one."""
    spec = QuerySpec.from_dict({"preferences": {"gender": {"value": "female", "weight": 1.0}},
                                "limit": 10})
    by_name = {Path(r.path).name: r for r in engine.search(spec).results}
    conf = next(c for c in by_name["a_ideal.jpg"].contributions if c.criterion == "gender")
    unc = next(c for c in by_name["h_uncertain.jpg"].contributions if c.criterion == "gender")
    assert conf.contribution > unc.contribution


def test_percentile_is_relative_to_the_collection():
    p = percentile_ranks(np.array([1.0, 2.0, 3.0, 4.0]))
    assert p[0] < p[-1] and 0 < p[0] < 1 and 0 < p[-1] < 1
    assert attractiveness_match(0.9, 0.8) == pytest.approx(0.5)
    assert attractiveness_match(0.5, 0.8) == 0.0


def test_trapezoid_shoulders():
    assert trapezoid(30, 25, 35, 5) == 1.0
    assert trapezoid(40, 25, 35, 5) == 0.0
    assert trapezoid(37.5, 25, 35, 5) == pytest.approx(0.5)
