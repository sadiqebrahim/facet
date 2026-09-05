"""Membership functions and the relevance score.

    relevance = sum_c  w_c * match_c * conf_c  /  sum_c w_c

Multiplying match by confidence is the design decision that makes the whole thing behave: an
uncertain criterion contributes little in either direction rather than contributing a
confident-looking wrong answer. It also gives the UI its explanation for free, because every
term in the sum is a row in the "why this result" panel.

Nothing here is learned. Section 9.1 keeps query relevance (B) separate from attractiveness
prediction (A) precisely so the ranking stays explainable - a learned end-to-end scorer could
not tell a user why a result placed where it did.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Contribution:
    """One criterion's effect on one result - the unit the UI explains with."""

    criterion: str
    match: float
    confidence: float
    weight: float
    contribution: float
    detail: str

    def as_dict(self) -> dict:
        return {
            "criterion": self.criterion, "match": round(self.match, 4),
            "confidence": round(self.confidence, 4), "weight": self.weight,
            "contribution": round(self.contribution, 4), "detail": self.detail,
        }


def trapezoid(x: float, lo: float, hi: float, shoulder: float) -> float:
    """1.0 inside [lo, hi], falling linearly to 0 across `shoulder` on each side."""
    if shoulder <= 0:
        return 1.0 if lo <= x <= hi else 0.0
    if lo <= x <= hi:
        return 1.0
    d = (lo - x) if x < lo else (x - hi)
    return float(max(0.0, 1.0 - d / shoulder))


def age_match(pred_age, lo, hi, shoulder, uncertainty=None, scale_by_uncertainty=True):
    if pred_age is None:
        return None
    s = shoulder
    if scale_by_uncertainty and uncertainty:
        # A face the model is unsure about gets a gentler penalty for falling outside the
        # range, because the distance itself is uncertain.
        s = shoulder + float(uncertainty)
    return trapezoid(float(pred_age), lo, hi, s)


def gender_match(p_female, want):
    if p_female is None:
        return None
    p = float(p_female)
    return p if want == "female" else 1.0 - p


def percentile_ranks(values: np.ndarray) -> np.ndarray:
    """Empirical percentile of each value within the collection.

    Label-free, so it works on a user's directory where conformal calibration cannot (E12).
    This is why attractiveness is expressed as a percentile rather than an absolute score.
    """
    from scipy.stats import rankdata

    v = np.asarray(values, dtype=np.float64)
    if len(v) == 0:
        return v
    return (rankdata(v, method="average") - 0.5) / len(v)


def attractiveness_match(pct: float | None, min_pct: float) -> float | None:
    """Ramp from 0 at `min_pct` to 1 at the top of the collection."""
    if pct is None:
        return None
    if min_pct >= 1.0:
        return 1.0 if pct >= 1.0 else 0.0
    return float(np.clip((pct - min_pct) / (1.0 - min_pct), 0.0, 1.0))


def combine(contributions: list[Contribution], total_weight: float) -> float:
    if not contributions:
        return 0.0
    return float(sum(c.contribution for c in contributions) / max(total_weight, 1e-9))
