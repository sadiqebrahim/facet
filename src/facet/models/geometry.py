"""Landmark / geometric features.

This is the classical facial-geometry approach (docs/RESEARCH.md section 6.5). The
dataset's own published geometric baselines reach PC 0.5948-0.6738, far below deep
features - so this exists for two reasons, neither of which is top-line accuracy:

1. It is the only *interpretable* signal in the system. "High facial symmetry" is
   something the UI can actually show in a "why was this selected" panel.
2. It is nearly free once landmarks are computed, so fusion with deep features is cheap
   to test (and a null result is worth documenting).

NOTE ON SEMANTICS: the index-to-facial-part mapping for SCUT-FBP5500's 86 points is not
documented in the release and we have not verified it. Rather than guess indices and
silently compute wrong "eye spacing" ratios, everything here is semantics-free:
Procrustes shape coordinates plus scalars derived from the point cloud as a whole.
Adding named ratios is a TODO gated on verifying the index map.
"""
from __future__ import annotations

import numpy as np


def _centre_scale(shapes: np.ndarray) -> np.ndarray:
    """Centre each shape at its centroid and scale to unit Frobenius norm."""
    centred = shapes - shapes.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centred.reshape(len(centred), -1), axis=1, keepdims=True)
    return centred / np.clip(norms, 1e-9, None)[:, :, None]


def _best_rotation(shape: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Optimal 2-D rotation aligning `shape` to `ref` (orthogonal Procrustes)."""
    u, _, vt = np.linalg.svd(shape.T @ ref)
    r = u @ vt
    if np.linalg.det(r) < 0:  # reflection - flip the smaller singular direction
        u[:, -1] *= -1
        r = u @ vt
    return shape @ r


def generalized_procrustes(shapes: np.ndarray, iters: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Align a set of (N, P, 2) shapes to a common frame.

    Returns (aligned shapes, mean shape). Removes translation, scale and rotation, so
    what remains is pure shape - which is exactly what a geometric beauty model wants.
    """
    aligned = _centre_scale(shapes.astype(np.float64))
    mean = aligned[0]
    for _ in range(iters):
        aligned = np.stack([_best_rotation(s, mean) for s in aligned])
        mean = aligned.mean(axis=0)
        mean = mean / np.linalg.norm(mean)
    return aligned, mean


def symmetry_residual(shape: np.ndarray) -> float:
    """Bilateral asymmetry, without needing to know which point is which.

    Reflects the (already centred/aligned) shape about its vertical axis, matches each
    reflected point to its nearest original point, and returns the mean residual. Lower
    is more symmetric. Semantics-free by construction.
    """
    reflected = shape.copy()
    reflected[:, 0] *= -1
    d = np.linalg.norm(shape[:, None, :] - reflected[None, :, :], axis=-1)
    return float(d.min(axis=1).mean())


def shape_scalars(shape: np.ndarray) -> dict[str, float]:
    """Interpretable scalars from an aligned shape. These are what the UI can explain."""
    x, y = shape[:, 0], shape[:, 1]
    width = float(x.max() - x.min())
    height = float(y.max() - y.min())
    r = np.linalg.norm(shape, axis=1)
    return {
        "symmetry": symmetry_residual(shape),
        "aspect_ratio": width / max(height, 1e-9),
        "radial_mean": float(r.mean()),
        "radial_std": float(r.std()),
        "spread_x": float(x.std()),
        "spread_y": float(y.std()),
    }


SCALAR_NAMES = ("symmetry", "aspect_ratio", "radial_mean", "radial_std", "spread_x", "spread_y")


def geometry_features(landmarks: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """(N, P, 2) raw landmarks -> (N, 2P + 6) feature matrix and its column names.

    Columns are the Procrustes-aligned coordinates followed by the interpretable scalars.
    """
    aligned, _ = generalized_procrustes(landmarks)
    coords = aligned.reshape(len(aligned), -1)
    scalars = np.array(
        [[shape_scalars(s)[k] for k in SCALAR_NAMES] for s in aligned], dtype=np.float64
    )
    names = [f"pt{i}_{a}" for i in range(landmarks.shape[1]) for a in ("x", "y")]
    names += list(SCALAR_NAMES)
    return np.hstack([coords, scalars]).astype(np.float32), names
