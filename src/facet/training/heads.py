"""Prediction heads that run over cached frozen features.

These are deliberately cheap. If experiment 001 confirms that frozen features carry the
signal, then swapping, retraining or personalising the beauty model is a matter of
refitting one of these - seconds on CPU, no GPU, no re-encoding. That property is the
whole argument for the encode/predict split in docs/RESEARCH.md section 15.1.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler

RATING_LEVELS = np.array([1.0, 2.0, 3.0, 4.0, 5.0])


class RidgeRegressionHead:
    """Scalar regression. The mandatory baseline (RESEARCH.md section 6.1).

    Alpha is chosen by generalised cross-validation on the TRAINING split only.
    """

    name = "ridge"

    def __init__(self, alphas: np.ndarray | None = None):
        self.alphas = alphas if alphas is not None else np.logspace(-2, 5, 30)
        self.scaler = StandardScaler()
        self.model: RidgeCV | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeRegressionHead":
        Xs = self.scaler.fit_transform(X)
        self.model = RidgeCV(alphas=self.alphas).fit(Xs, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(self.scaler.transform(X))

    @property
    def chosen_alpha(self) -> float:
        return float(self.model.alpha_)


class DistributionHead:
    """Label-distribution head: predict the full 5-bin rating histogram.

    This is the approach argued for in RESEARCH.md section 6.3. Instead of regressing the
    mean of 60 ratings, it regresses the histogram itself and then derives:

        mean     - expectation over the rating levels (comparable to the scalar baseline)
        std      - ALEATORIC uncertainty: genuine disagreement between human raters,
                   which a mean-regression model cannot represent at all
        p_ge4    - P(rating >= 4), a better ranking target for "find the best faces"
                   than the mean, because it is the question the user is actually asking

    Implemented as independent ridge regressors per bin followed by a clip-and-renormalise.
    That is intentionally the simplest thing that could work: the point of this experiment
    is to measure what the FEATURES carry, not to tune a head.
    """

    name = "ridge_ldl"

    def __init__(self, alphas: np.ndarray | None = None, levels: np.ndarray = RATING_LEVELS):
        self.alphas = alphas if alphas is not None else np.logspace(-2, 5, 30)
        self.levels = levels
        self.scaler = StandardScaler()
        self.models: list[RidgeCV] = []

    def fit(self, X: np.ndarray, P: np.ndarray) -> "DistributionHead":
        """`P`: (N, n_levels) normalised rating histograms."""
        Xs = self.scaler.fit_transform(X)
        self.models = [RidgeCV(alphas=self.alphas).fit(Xs, P[:, j]) for j in range(P.shape[1])]
        return self

    def predict_distribution(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform(X)
        raw = np.column_stack([m.predict(Xs) for m in self.models])
        raw = np.clip(raw, 1e-6, None)
        return raw / raw.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Expected rating - directly comparable to the scalar regression baseline."""
        return self.predict_distribution(X) @ self.levels

    def predict_std(self, X: np.ndarray) -> np.ndarray:
        """Aleatoric spread: how much the rater pool is predicted to disagree."""
        p = self.predict_distribution(X)
        mu = p @ self.levels
        var = p @ (self.levels**2) - mu**2
        return np.sqrt(np.clip(var, 0, None))

    def predict_p_ge(self, X: np.ndarray, threshold: float = 4.0) -> np.ndarray:
        p = self.predict_distribution(X)
        return p[:, self.levels >= threshold].sum(axis=1)


class BaggedRidgeHead:
    """Bootstrap ensemble of ridge heads -> EPISTEMIC uncertainty.

    The deep-ensemble recipe (RESEARCH.md section 11.2), which is essentially free once
    features are frozen: N linear fits over cached features cost milliseconds. The spread
    across members measures model ignorance, as distinct from the rater disagreement that
    DistributionHead measures.
    """

    name = "bagged_ridge"

    def __init__(self, n_members: int = 10, alpha: float = 100.0, seed: int = 0):
        self.n_members = n_members
        self.alpha = alpha
        self.seed = seed
        self.scaler = StandardScaler()
        self.members: list[Ridge] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaggedRidgeHead":
        Xs = self.scaler.fit_transform(X)
        rng = np.random.default_rng(self.seed)
        n = len(Xs)
        self.members = []
        for _ in range(self.n_members):
            idx = rng.integers(0, n, n)
            self.members.append(Ridge(alpha=self.alpha).fit(Xs[idx], y[idx]))
        return self

    def _member_preds(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform(X)
        return np.stack([m.predict(Xs) for m in self.members])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._member_preds(X).mean(axis=0)

    def predict_with_std(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        preds = self._member_preds(X)
        return preds.mean(axis=0), preds.std(axis=0)
