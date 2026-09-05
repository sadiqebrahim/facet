"""The production attractiveness head, and everything needed to report it honestly.

Assembled from what the experiments decided:

* **E6** - a label-distribution (LDL) head, not regression. Not because it is more accurate
  (it is not, within seed noise) but because it is the only objective that yields a rating
  distribution, which sections 9.3 and 11 both require.
* **E5** - trained on features from the m0.25 crop protocol.
* **exp001/E5** - over frozen ArcFace + CLIP features.
* **E12** - raw model spread is not a confidence. Intervals come from split-conformal
  calibration, and they are only valid in-domain.
* **E7/E12** - out-of-distribution faces get their numeric confidence SUPPRESSED rather than
  reported, because a nominal 90% interval delivered 43% coverage under domain shift.

A limitation worth stating plainly rather than burying: E12 concluded conformal calibration
should be *per collection*, but conformal needs labels and a user's photo directory has none.
So the shipped intervals are calibrated on SCUT-FBP5500 and are trustworthy only for faces
resembling it. That is precisely why the OOD gate exists, and why the label-free
percentile-within-collection is the primary ranking signal rather than the absolute score.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

LEVELS = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

SOURCE_OF_TRUTH = (
    "Predicted rating on the SCUT-FBP5500 scale, as defined by 60 volunteer raters aged "
    "18-27 (mean 21.6) in 2017. This is an estimate of how that group would have rated this "
    "face. It is not a measurement of beauty."
)


@dataclass
class BeautyPrediction:
    mean: float
    distribution: np.ndarray
    p_ge4: float
    aleatoric: float          # predicted rater disagreement
    epistemic: float          # ensemble spread
    interval: tuple[float, float] | None
    confidence: float | None  # None when the OOD gate fires
    ood_score: float
    ood: bool
    warnings: list[str]
    source: str = SOURCE_OF_TRUTH


class BeautyHead:
    """Ensemble of linear LDL heads over frozen features, with conformal intervals."""

    name = "beauty_ldl_arcface_clip"

    def __init__(self, mean, scale, weights, biases, conformal, ood_ref, ood_thresh,
                 version, config_hash, metrics=None):
        self.mean, self.scale = np.asarray(mean), np.asarray(scale)
        self.W = [np.asarray(w) for w in weights]     # each (dim, 5)
        self.b = [np.asarray(x) for x in biases]
        self.conformal = {float(k): float(v) for k, v in conformal.items()}
        self.ood_ref = np.asarray(ood_ref, dtype=np.float32)
        self.ood_thresh = float(ood_thresh)
        self.version = version
        self.config_hash = config_hash
        self.metrics = metrics or {}

    # ---------------------------------------------------------------- inference

    def _members(self, X):
        Z = (X - self.mean) / self.scale
        out = []
        for W, b in zip(self.W, self.b):
            logits = Z @ W + b
            e = np.exp(logits - logits.max(axis=1, keepdims=True))
            out.append(e / e.sum(axis=1, keepdims=True))
        return np.stack(out)                      # (M, N, 5)

    def ood_score(self, X: np.ndarray) -> np.ndarray:
        """Distance to the nearest training feature (cosine). Label-free, so it works on a
        user's collection where conformal calibration cannot."""
        Xn = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-9, None)
        sims = Xn.astype(np.float32) @ self.ood_ref.T
        return 1.0 - sims.max(axis=1)

    def predict(self, X: np.ndarray, level: float = 0.90) -> list[BeautyPrediction]:
        P = self._members(X)
        p = P.mean(axis=0)
        means = P @ LEVELS                                   # (M, N)
        mu = means.mean(axis=0)
        epi = means.std(axis=0)
        var = (p @ (LEVELS**2)) - (p @ LEVELS) ** 2
        alea = np.sqrt(np.clip(var, 0, None))
        combined = np.sqrt(alea**2 + epi**2)
        q = self.conformal.get(level, 1.0)
        ood = self.ood_score(X)

        out = []
        for i in range(len(X)):
            is_ood = bool(ood[i] > self.ood_thresh)
            half = q * combined[i]
            warn = []
            if is_ood:
                warn.append(
                    "face is unlike the training distribution; confidence suppressed "
                    "(E12: a nominal 90% interval delivered 43% coverage under domain shift)"
                )
            out.append(BeautyPrediction(
                mean=float(mu[i]), distribution=p[i].astype(float),
                p_ge4=float(p[i][3:].sum()),
                aleatoric=float(alea[i]), epistemic=float(epi[i]),
                interval=None if is_ood else (float(mu[i] - half), float(mu[i] + half)),
                confidence=None if is_ood else float(np.clip(1.0 - half / 2.0, 0.0, 1.0)),
                ood_score=float(ood[i]), ood=is_ood, warnings=warn,
            ))
        return out

    # ------------------------------------------------------------ (de)serialise

    def save(self, path: str | Path) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, mean=self.mean, scale=self.scale,
            W=np.stack(self.W), b=np.stack(self.b), ood_ref=self.ood_ref,
            conformal_keys=np.array(list(self.conformal)),
            conformal_vals=np.array(list(self.conformal.values())),
            ood_thresh=self.ood_thresh,
            meta=np.array(json.dumps({
                "version": self.version, "config_hash": self.config_hash,
                "metrics": self.metrics, "source": SOURCE_OF_TRUTH,
            })),
        )

    @classmethod
    def load(cls, path: str | Path) -> "BeautyHead":
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        return cls(
            mean=z["mean"], scale=z["scale"], weights=list(z["W"]), biases=list(z["b"]),
            conformal=dict(zip(z["conformal_keys"].tolist(), z["conformal_vals"].tolist())),
            ood_ref=z["ood_ref"], ood_thresh=float(z["ood_thresh"]),
            version=meta["version"], config_hash=meta["config_hash"],
            metrics=meta.get("metrics", {}),
        )
