"""Face image quality signals that are free or nearly free.

docs/RESEARCH.md section 2.4 argues we should not start by training a FIQA model, but build a
composite from signals we already compute, then validate it against a real one (E9). This
module is that composite.

Every signal here costs either nothing (it falls out of a forward pass we already do) or one
cheap OpenCV call:

    det_score      detector confidence            - free
    face_pixels    detected face area             - free
    blur           variance of Laplacian          - one cv2 call
    exposure       clipping + mean luminance      - numpy
    contrast       pixel standard deviation       - numpy
    embedding_norm pre-normalisation L2 norm      - free from the embedder (MagFace-style)

Quality matters here for a specific reason beyond hygiene: a blurry face regresses toward the
training mean, which produces a confident-looking mid-range attractiveness score. Quality is
therefore an input to CONFIDENCE (section 11), not merely a filter.
"""
from __future__ import annotations

import cv2
import numpy as np


def blur_score(crop: np.ndarray) -> float:
    """Variance of the Laplacian. Higher = sharper."""
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def exposure_stats(crop: np.ndarray) -> dict[str, float]:
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    g = g.astype(np.float32)
    return {
        "luminance": float(g.mean() / 255.0),
        "contrast": float(g.std() / 255.0),
        "clipped_dark": float((g <= 2).mean()),
        "clipped_bright": float((g >= 253).mean()),
    }


def per_crop_signals(crop: np.ndarray) -> dict[str, float]:
    out = {"blur": blur_score(crop)}
    out.update(exposure_stats(crop))
    return out


def composite(
    signals: dict[str, np.ndarray],
    blur_ref: float = 500.0,
    size_ref: float = 112.0,
) -> np.ndarray:
    """Combine signals into a 0-1 quality score.

    The reference constants are deliberately explicit and configurable rather than tuned:
    E9 validates this composite against a real FIQA model, and until that runs these are
    documented guesses, not fitted parameters.
    """
    def unit(x, ref):
        return np.clip(np.asarray(x, dtype=np.float64) / ref, 0.0, 1.0)

    parts = [
        unit(signals["blur"], blur_ref),
        unit(signals.get("face_pixels", size_ref), size_ref),
        np.clip(signals.get("det_score", 1.0), 0.0, 1.0),
        1.0 - np.clip(signals.get("clipped_dark", 0.0) + signals.get("clipped_bright", 0.0),
                      0.0, 1.0),
        np.clip(np.asarray(signals.get("contrast", 0.2)) / 0.25, 0.0, 1.0),
    ]
    return np.clip(np.mean(parts, axis=0), 0.0, 1.0)
