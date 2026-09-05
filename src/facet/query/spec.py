"""Query specification.

The brief's example query is the target:

    Age: 25-35, importance 30%
    Gender: Female, importance 20%
    Attractiveness: High, importance 50%

Three things the research phase changed about how that gets expressed:

* **Attractiveness is a percentile, not a threshold.** E7 showed absolute scores do not
  survive a domain change - two reasonable rater pools disagreed on 80% of a top-100 - so
  `attractiveness > 4.0` is not a meaningful request. "Top 20% of this collection" is, and it
  is label-free, which matters because a user's directory has no ground truth to calibrate
  against.
* **Gender is a soft preference by default** (section 8). A hard filter on a classifier that
  E4 measured at 0.96 (and E11 at 0.81 for the retired baseline) silently discards true
  matches, unevenly across groups. Strict mode exists, but it reports what it excluded.
* **Unknown is not the same as unmatched.** Age and gender come from a lazy pass (E4: MiVOLO
  is ~190x the cost of everything else), so many faces legitimately have no prediction.
  Treating "not yet predicted" as "does not match" would be an invisible failure of exactly
  the kind section 13.2 warns about.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

SortKey = Literal["relevance", "attractiveness", "confidence", "quality", "age_match", "random"]


@dataclass
class AgeCriterion:
    """Trapezoidal membership over a requested age range."""

    min_age: float
    max_age: float
    weight: float = 0.30
    #: Width of the soft shoulder in years. 36 should not score zero when 25-35 was asked for.
    softness: float = 5.0
    #: E4 measured age error varying 5x more by age bucket than by race (2.7 years for
    #: 20-29, 6.9 for 70+). A face the model is less sure about should not be penalised as
    #: sharply, so the shoulder widens with that face's own predicted uncertainty.
    scale_softness_by_uncertainty: bool = True
    required: bool = False          # if True, faces with no age prediction are excluded


@dataclass
class GenderCriterion:
    value: Literal["female", "male"]
    weight: float = 0.20
    mode: Literal["soft", "strict"] = "soft"
    min_confidence: float = 0.6
    required: bool = False


@dataclass
class AttractivenessCriterion:
    """Percentile-based, per E7. `min_percentile=0.8` means the top 20% of the collection."""

    min_percentile: float = 0.8
    weight: float = 0.50
    #: Rank by P(rating >= 4) from the distribution head rather than by the mean. Section 9.3:
    #: it answers the question actually being asked and accounts for the model's own spread.
    use_p_ge4: bool = True


@dataclass
class Filters:
    """Hard constraints, applied before scoring."""

    min_quality: float = 0.0
    min_face_px: float = 0.0
    min_det_score: float = 0.0
    exclude_ood: bool = False
    exclude_near_duplicates: bool = True
    paths_like: str | None = None


@dataclass
class QuerySpec:
    age: AgeCriterion | None = None
    gender: GenderCriterion | None = None
    attractiveness: AttractivenessCriterion | None = None
    filters: Filters = field(default_factory=Filters)
    sort_by: SortKey = "relevance"
    limit: int = 100
    offset: int = 0
    #: Multiply each criterion's match by the model's confidence in it, so an uncertain
    #: criterion contributes little in either direction rather than a confident wrong answer.
    confidence_weighting: bool = True

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "QuerySpec":
        d = dict(d or {})
        prefs = d.get("preferences", d)
        spec = cls(
            filters=Filters(**(d.get("filters") or {})),
            sort_by=d.get("sort_by", d.get("ranking", {}).get("sort_by", "relevance")),
            limit=int(d.get("limit", d.get("ranking", {}).get("limit", 100))),
            offset=int(d.get("offset", 0)),
            confidence_weighting=bool(d.get("confidence_weighting", True)),
        )
        if "age" in prefs and prefs["age"]:
            a = dict(prefs["age"])
            rng = a.pop("range", None)
            if rng:
                a["min_age"], a["max_age"] = float(rng[0]), float(rng[1])
            spec.age = AgeCriterion(**a)
        if "gender" in prefs and prefs["gender"]:
            g = dict(prefs["gender"])
            if "value" not in g and "label" in g:
                g["value"] = g.pop("label")
            spec.gender = GenderCriterion(**g)
        if "attractiveness" in prefs and prefs["attractiveness"]:
            spec.attractiveness = AttractivenessCriterion(**dict(prefs["attractiveness"]))
        return spec

    def active(self) -> list[tuple[str, Any]]:
        return [(n, c) for n, c in
                (("age", self.age), ("gender", self.gender),
                 ("attractiveness", self.attractiveness)) if c is not None]

    def total_weight(self) -> float:
        return sum(c.weight for _, c in self.active()) or 1.0
