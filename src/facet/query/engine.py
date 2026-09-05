"""Search engine: hard filters -> soft scoring -> rank -> paginate.

Section 9.1 keeps two ranking problems apart, and this file is the second one. Ranking faces
by predicted attractiveness is machine learning; ranking *results by match to a query* is a
transparent scoring function that has to be able to explain itself. Conflating them would
destroy the "why was this selected" panel the brief asks for.

Three behaviours the research phase forced, all visible in the diagnostics this returns:

* **Unknown is reported, not silently dropped.** Age and gender come from a lazy pass, so many
  faces have no prediction. Those criteria contribute nothing rather than scoring zero, and
  the count is surfaced.
* **Strict filters report what they excluded.** Section 8: a hard gender filter on a ~96%
  classifier discards true matches unevenly, and the user should be able to see how many.
* **Attractiveness is a percentile of the collection** (E7), computed over all scored faces
  rather than the filtered subset, so the meaning of "top 20%" does not shift with the filters.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..pipeline.db import Index
from .scoring import (
    Contribution, age_match, attractiveness_match, combine, gender_match, percentile_ranks,
)
from .spec import QuerySpec

BEAUTY_SOURCE_NOTE = (
    "Attractiveness values are model estimates on the SCUT-FBP5500 rating scale (60 raters "
    "aged 18-27, 2017), not measurements. Ranking across demographic groups carries a "
    "measured skew - see docs/RESEARCH.md 13.5."
)


@dataclass
class Result:
    face_id: int
    image_id: int
    path: str
    bbox: tuple[float, float, float, float]
    relevance: float
    quality: float
    attractiveness: float | None
    attractiveness_percentile: float | None
    p_ge4: float | None
    age: float | None
    gender: str | None
    gender_confidence: float | None
    confidence: float | None
    interval: tuple[float, float] | None
    ood: bool
    warnings: list[str]
    contributions: list[Contribution] = field(default_factory=list)

    def as_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "contributions"}
        d["contributions"] = [c.as_dict() for c in self.contributions]
        d["_note"] = BEAUTY_SOURCE_NOTE
        return d


@dataclass
class SearchResponse:
    results: list[Result]
    total_matched: int
    diagnostics: dict[str, Any]
    spec: dict[str, Any]

    def as_dict(self) -> dict:
        return {
            "results": [r.as_dict() for r in self.results],
            "total_matched": self.total_matched,
            "diagnostics": self.diagnostics,
            "spec": self.spec,
        }


class SearchEngine:
    def __init__(self, index_path: str | Path):
        self.index = Index(index_path)
        self._pct_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    # ------------------------------------------------------------ collection stats

    def _percentile_lookup(self, key: str):
        """Empirical CDF over the whole collection, computed once and cached."""
        if key in self._pct_cache:
            return self._pct_cache[key]
        col = "value" if key == "mean" else "p_ge4"
        if key == "mean":
            rows = self.index.conn.execute(
                "SELECT value v FROM predictions WHERE model='beauty' AND value IS NOT NULL")
            vals = np.array([r["v"] for r in rows], dtype=np.float64)
        else:
            rows = self.index.conn.execute(
                "SELECT extra FROM predictions WHERE model='beauty' AND extra IS NOT NULL")
            vals = np.array([json.loads(r["extra"]).get("p_ge4", np.nan) for r in rows],
                            dtype=np.float64)
            vals = vals[np.isfinite(vals)]
        order = np.sort(vals)
        pct = percentile_ranks(order) if len(order) else order
        self._pct_cache[key] = (order, pct)
        return order, pct

    def _to_percentile(self, key: str, value: float | None) -> float | None:
        if value is None:
            return None
        order, pct = self._percentile_lookup(key)
        if len(order) == 0:
            return None
        i = int(np.clip(np.searchsorted(order, value), 0, len(order) - 1))
        return float(pct[i])

    # --------------------------------------------------------------------- search

    def search(self, spec: QuerySpec) -> SearchResponse:
        f = spec.filters
        where, params = ["f.feature_row IS NOT NULL"], []
        if f.min_quality > 0:
            where.append("COALESCE(f.quality,0) >= ?"); params.append(f.min_quality)
        if f.min_face_px > 0:
            where.append("COALESCE(f.face_px,0) >= ?"); params.append(f.min_face_px)
        if f.min_det_score > 0:
            where.append("COALESCE(f.det_score,0) >= ?"); params.append(f.min_det_score)
        if f.paths_like:
            where.append("i.path LIKE ?"); params.append(f"%{f.paths_like}%")

        sql = f"""
            SELECT f.id face_id, f.image_id, i.path, f.x1, f.y1, f.x2, f.y2,
                   f.quality, f.face_px, f.det_score,
                   b.value beauty, b.confidence bconf, b.interval_lo, b.interval_hi,
                   b.extra bextra,
                   a.value age, g.value p_female, g.confidence gconf,
                   d.group_id dup_group
            FROM faces f
            JOIN images i ON i.id = f.image_id
            LEFT JOIN predictions b ON b.face_id=f.id AND b.model='beauty'
            LEFT JOIN predictions a ON a.face_id=f.id AND a.model='age'
            LEFT JOIN predictions g ON g.face_id=f.id AND g.model='gender'
            LEFT JOIN duplicates d ON d.face_id=f.id AND d.kind='near'
            WHERE {' AND '.join(where)}
        """
        rows = list(self.index.conn.execute(sql, params))

        diag = {
            "candidates_before_filters": len(rows),
            "missing_age": 0, "missing_gender": 0, "missing_attractiveness": 0,
            "excluded_ood": 0, "excluded_low_gender_confidence": 0,
            "excluded_required_missing": 0, "collapsed_near_duplicates": 0,
        }

        seen_dup: set[int] = set()
        results: list[Result] = []
        total_w = spec.total_weight()

        for r in rows:
            extra = json.loads(r["bextra"]) if r["bextra"] else {}
            ood = bool(extra.get("ood", False))
            if f.exclude_ood and ood:
                diag["excluded_ood"] += 1
                continue
            if f.exclude_near_duplicates and r["dup_group"] is not None:
                if r["dup_group"] in seen_dup:
                    diag["collapsed_near_duplicates"] += 1
                    continue
                seen_dup.add(r["dup_group"])

            contribs: list[Contribution] = []
            drop = False

            # ---- age -----------------------------------------------------------
            if spec.age is not None:
                c = spec.age
                unc = _age_uncertainty(r["age"])
                m = age_match(r["age"], c.min_age, c.max_age, c.softness, unc,
                              c.scale_softness_by_uncertainty)
                if m is None:
                    diag["missing_age"] += 1
                    if c.required:
                        diag["excluded_required_missing"] += 1
                        drop = True
                    else:
                        contribs.append(Contribution(
                            "age", 0.0, 0.0, c.weight, 0.0, "no age prediction (lazy pass)"))
                else:
                    conf = 1.0 if not spec.confidence_weighting else _age_confidence(unc)
                    contribs.append(Contribution(
                        "age", m, conf, c.weight, c.weight * m * conf,
                        f"predicted {r['age']:.0f}, wanted {c.min_age:.0f}-{c.max_age:.0f}"))

            # ---- gender --------------------------------------------------------
            if spec.gender is not None and not drop:
                c = spec.gender
                m = gender_match(r["p_female"], c.value)
                if m is None:
                    diag["missing_gender"] += 1
                    if c.required:
                        diag["excluded_required_missing"] += 1
                        drop = True
                    else:
                        contribs.append(Contribution(
                            "gender", 0.0, 0.0, c.weight, 0.0,
                            "no gender prediction (lazy pass)"))
                else:
                    conf = float(r["gconf"] or max(m, 1 - m))
                    if c.mode == "strict":
                        if conf < c.min_confidence:
                            diag["excluded_low_gender_confidence"] += 1
                            drop = True
                        elif m < 0.5:
                            drop = True
                    if not drop:
                        w = conf if spec.confidence_weighting else 1.0
                        contribs.append(Contribution(
                            "gender", m, conf, c.weight, c.weight * m * w,
                            f"{'female' if (r['p_female'] or 0) >= 0.5 else 'male'} "
                            f"(p={float(r['p_female']):.2f}), wanted {c.value}"))

            # ---- attractiveness -------------------------------------------------
            pct = None
            if spec.attractiveness is not None and not drop:
                c = spec.attractiveness
                key = "p_ge4" if c.use_p_ge4 else "mean"
                raw = extra.get("p_ge4") if c.use_p_ge4 else r["beauty"]
                pct = self._to_percentile(key, raw)
                m = attractiveness_match(pct, c.min_percentile)
                if m is None:
                    diag["missing_attractiveness"] += 1
                    contribs.append(Contribution(
                        "attractiveness", 0.0, 0.0, c.weight, 0.0, "no attractiveness prediction"))
                else:
                    # E12: an OOD face's confidence was suppressed at predict time, so it
                    # contributes on match alone rather than on a number we do not trust.
                    conf = 1.0
                    if spec.confidence_weighting:
                        conf = 0.5 if ood else float(r["bconf"] or 0.5)
                    contribs.append(Contribution(
                        "attractiveness", m, conf, c.weight, c.weight * m * conf,
                        f"percentile {pct:.2f} of collection, wanted top "
                        f"{100*(1-c.min_percentile):.0f}%"
                        + (" (out-of-distribution: confidence suppressed)" if ood else "")))
            if drop:
                continue

            results.append(Result(
                face_id=r["face_id"], image_id=r["image_id"], path=r["path"],
                bbox=(r["x1"], r["y1"], r["x2"], r["y2"]),
                relevance=combine(contribs, total_w),
                quality=float(r["quality"] or 0.0),
                attractiveness=r["beauty"], attractiveness_percentile=pct,
                p_ge4=extra.get("p_ge4"),
                age=r["age"],
                gender=(None if r["p_female"] is None
                        else ("female" if r["p_female"] >= 0.5 else "male")),
                gender_confidence=r["gconf"],
                confidence=r["bconf"],
                interval=((r["interval_lo"], r["interval_hi"])
                          if r["interval_lo"] is not None else None),
                ood=ood, warnings=list(extra.get("warnings", [])),
                contributions=contribs,
            ))

        keyfns = {
            "relevance": lambda x: -x.relevance,
            "attractiveness": lambda x: -(x.attractiveness or -np.inf),
            "confidence": lambda x: -(x.confidence or 0.0),
            "quality": lambda x: -x.quality,
            "age_match": lambda x: -next((c.match for c in x.contributions
                                          if c.criterion == "age"), 0.0),
            "random": lambda x: x.face_id,
        }
        results.sort(key=keyfns.get(spec.sort_by, keyfns["relevance"]))
        total = len(results)
        page = results[spec.offset : spec.offset + spec.limit]

        diag["total_matched"] = total
        diag["returned"] = len(page)
        return SearchResponse(page, total, diag, _spec_dict(spec))

    def close(self):
        self.index.close()


def _age_uncertainty(age):
    """MiVOLO gives a point estimate. E4 measured its error by bucket, so we widen the
    shoulder using that measured profile rather than pretending the estimate is exact."""
    if age is None:
        return None
    a = float(age)
    if a < 10:
        return 3.0
    if a < 20:
        return 5.8
    if a < 40:
        return 5.2
    if a < 60:
        return 6.0
    return 6.9


def _age_confidence(unc):
    if unc is None:
        return 0.0
    return float(np.clip(1.0 - unc / 15.0, 0.1, 1.0))


def _spec_dict(spec: QuerySpec) -> dict:
    out: dict[str, Any] = {"sort_by": spec.sort_by, "limit": spec.limit,
                           "offset": spec.offset,
                           "confidence_weighting": spec.confidence_weighting,
                           "filters": spec.filters.__dict__}
    for name, c in spec.active():
        out[name] = c.__dict__
    return out
