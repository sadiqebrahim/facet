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


def composite_v2(signals: dict[str, np.ndarray]) -> np.ndarray:
    """Repaired quality composite. Supersedes `composite`, which E9 showed was broken.

    E9 evaluated the original composite by the field's functional criterion - does
    rejecting low-quality faces reduce face-recognition error (Error-vs-Reject on LFW's
    6,000-pair protocol)? It beat random rejection (AUERC 0.0136 vs 0.0229) but was
    **beaten by one of its own components**, raw face size (0.0096). Diagnosis:

    * `unit(face_pixels, size_ref=112)` SATURATES - almost every detected face is larger
      than 112px, so the term was 1.0 for nearly everything and contributed no ranking
      information. Face size is the single strongest signal and the composite was
      discarding it.
    * The equal-weight mean was therefore dominated by blur: corr(composite, blur) =
      +0.914 versus +0.065 for face size. It was, in effect, a blur detector - and blur
      is one of the WEAKER signals (AUERC 0.0193).
    * `contrast` is worse than random rejection (0.0248 vs 0.0229), i.e. actively
      harmful, and was being averaged in at equal weight.
    * `embedding_norm` - the free MagFace-style proxy - genuinely works (0.0158, 31%
      better than random) and was not in the composite at all: corr was -0.030.

    The repair is structural rather than fitted: log-scale the size term so it keeps
    discriminating over the whole useful range, drop contrast, include embedding norm when
    available, and weight by measured usefulness.
    """
    def logsize(px, lo=24.0, hi=400.0):
        px = np.clip(np.asarray(px, dtype=np.float64), 1.0, None)
        return np.clip((np.log(px) - np.log(lo)) / (np.log(hi) - np.log(lo)), 0.0, 1.0)

    def rank01(x):
        """Rank-normalise, so heavy-tailed signals like Laplacian variance are usable."""
        x = np.asarray(x, dtype=np.float64)
        if x.size == 0 or np.ptp(x) == 0:
            return np.zeros_like(x)
        from scipy.stats import rankdata
        return (rankdata(x) - 1) / max(len(x) - 1, 1)

    terms, weights = [], []
    if "face_pixels" in signals:
        terms.append(logsize(signals["face_pixels"])); weights.append(0.45)
    if "embedding_norm" in signals:
        terms.append(rank01(signals["embedding_norm"])); weights.append(0.25)
    if "blur" in signals:
        terms.append(rank01(signals["blur"])); weights.append(0.15)
    if "det_score" in signals:
        terms.append(np.clip(signals["det_score"], 0.0, 1.0)); weights.append(0.15)
    # Clipping is a hard defect rather than a graded one, so it multiplies rather than adds.
    penalty = 1.0 - np.clip(
        np.asarray(signals.get("clipped_dark", 0.0))
        + np.asarray(signals.get("clipped_bright", 0.0)), 0.0, 1.0
    )
    w = np.array(weights) / np.sum(weights)
    return np.clip(np.tensordot(w, np.stack(terms), axes=1) * penalty, 0.0, 1.0)
