"""Model protocols.

Every model sits behind one of these so it can be swapped by configuration. Nothing
outside `facet.models` may import insightface / onnxruntime / torch directly - that is
what keeps both the "easy to replace models in" requirement and the licensing surface
(docs/LICENSING.md) mechanically enforceable rather than aspirational.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass
class Detection:
    """One detected face."""

    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    score: float
    keypoints: np.ndarray | None = None  # (5, 2) - eyes, nose, mouth corners


@dataclass
class Prediction:
    """A model estimate, carrying its own uncertainty and provenance.

    This type exists to make docs/RESEARCH.md section 11.4 structurally enforceable: it
    is impossible to hold a prediction without also holding the caveats that make it
    honest. `source` records what the number actually means - for attractiveness that is
    "the SCUT-FBP5500 rating scale as defined by 60 raters aged 18-27 in 2017", not
    "beauty".
    """

    value: float
    confidence: float | None = None
    std: float | None = None
    interval: tuple[float, float] | None = None
    distribution: np.ndarray | None = None
    model_version: str = ""
    source: str = ""
    warnings: list[str] = field(default_factory=list)

    def is_estimate(self) -> bool:  # pragma: no cover - documentation in code form
        return True


@runtime_checkable
class FaceDetector(Protocol):
    name: str

    def detect(self, image: np.ndarray) -> list[Detection]: ...


@runtime_checkable
class FaceAligner(Protocol):
    name: str
    output_size: int

    def align(self, image: np.ndarray, detection: Detection) -> np.ndarray: ...


@runtime_checkable
class FeatureExtractor(Protocol):
    """The expensive, cached half of the pipeline.

    Features are keyed on (name, version, crop protocol) so that a cached feature is
    never silently reused across incompatible preprocessing.
    """

    name: str
    version: str
    dim: int

    def encode(self, crops: np.ndarray) -> np.ndarray: ...


@runtime_checkable
class AttributeHead(Protocol):
    """The cheap, replaceable half - runs over cached features."""

    name: str

    def predict(self, features: np.ndarray) -> list[Prediction]: ...
