"""Per-user preference model: learn *this* user's taste, not the crowd's.

Grounded in E14, which simulated users from individual raters and found:

* **The residual formulation is mandatory.** Training on a user's own labels alone is
  catastrophic in the cold-start regime (Spearman 0.21 vs 0.51 at 10 labels). The population
  model has to carry the weight until the user has said enough for their own signal to beat
  it, so personalisation blends *toward* a personal score rather than replacing anything.
* **Gains start at ~10 labels** and are solid by ~100 - a budget a real person will actually
  spend.
* **It only pays off when the user differs from the training pool.** SCUT's 60 raters agree
  with each other so strongly that personalisation actively *hurt* there; MEBeauty's diverse
  pool is where it helped. A user's own taste is by definition the diverse case.

Two ways to teach it, because they fail at different times:

    references   example faces the user picks. Works from ONE image, so it solves cold start.
    feedback     like / reject on results. Slower to accumulate but reflects real judgements
                 on real candidates rather than idealised examples.

Both reduce to labelled feature vectors, so they train one model together.

Everything runs over the cached frozen embeddings, which is what makes this practical: fitting
takes milliseconds on CPU, so the ranking can update the instant a user clicks reject.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: How fast personalisation takes over from the population model as labels accumulate.
#: n/(n+HALF) reaches 0.5 at HALF labels. E14 saw gains from 10 and solid ones by 100.
HALF_LABELS = 25.0
#: Never fully replace the population model. Even a well-taught personal model is fitted on
#: far less data, and E14's own failure case was a personal model confidently fitting noise.
MAX_ALPHA = 0.85


@dataclass
class PreferenceStatus:
    n_likes: int
    n_dislikes: int
    n_references: int
    alpha: float
    trained: bool
    method: str
    note: str


class PreferenceModel:
    """Ridge on liked/disliked embeddings, with a centroid fallback for tiny label counts.

    With one or two examples a fitted linear model is meaningless, so the score is cosine
    similarity to the liked centroid (minus the disliked centroid). Once there are enough
    labels on both sides, ridge takes over - it can weight the directions in feature space
    that actually separate this user's likes from their dislikes.
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.w: np.ndarray | None = None
        self.b: float = 0.0
        self.like_centroid: np.ndarray | None = None
        self.dislike_centroid: np.ndarray | None = None
        self.n_likes = self.n_dislikes = self.n_references = 0
        self.method = "none"

    # ------------------------------------------------------------------ fitting

    def fit(self, liked: np.ndarray, disliked: np.ndarray, ref_weight: float = 2.0,
            n_references: int = 0, alpha: float = 1.0) -> "PreferenceModel":
        liked = _unit(np.asarray(liked, dtype=np.float64).reshape(-1, self.dim))
        disliked = _unit(np.asarray(disliked, dtype=np.float64).reshape(-1, self.dim))
        self.n_likes, self.n_dislikes = len(liked), len(disliked)
        self.n_references = n_references

        if len(liked):
            self.like_centroid = _unit(liked.mean(axis=0, keepdims=True))[0]
        if len(disliked):
            self.dislike_centroid = _unit(disliked.mean(axis=0, keepdims=True))[0]

        # Ridge needs both classes and enough examples to be better than a centroid.
        if len(liked) >= 3 and len(disliked) >= 3:
            X = np.vstack([liked, disliked])
            y = np.concatenate([np.ones(len(liked)), -np.ones(len(disliked))])
            # References are idealised examples rather than judgements on real candidates,
            # so they get more pull but do not drown out actual feedback.
            sw = np.concatenate([
                np.full(len(liked), ref_weight if n_references else 1.0),
                np.ones(len(disliked))])
            Xw = X * sw[:, None]
            A = Xw.T @ X + alpha * np.eye(self.dim)
            self.w = np.linalg.solve(A, Xw.T @ y)
            self.b = float(-(X @ self.w).mean())
            self.method = "ridge"
        elif self.like_centroid is not None or self.dislike_centroid is not None:
            # Dislike-only is a first-class case, not an edge case: rejecting is far less
            # effort than curating examples, so many users will only ever press ✕. With no
            # likes we cannot say what they want, but "less like these" is still a real
            # signal and must move the ranking.
            self.w = None
            self.method = "centroid" if self.like_centroid is not None else "avoid"
        else:
            self.method = "none"
        return self

    # ---------------------------------------------------------------- inference

    def score(self, X: np.ndarray) -> np.ndarray:
        """Higher = closer to this user's taste. Arbitrary scale; use percentiles."""
        Xu = _unit(np.asarray(X, dtype=np.float64).reshape(-1, self.dim))
        if self.method == "ridge" and self.w is not None:
            return Xu @ self.w + self.b
        if self.like_centroid is not None:
            s = Xu @ self.like_centroid
            if self.dislike_centroid is not None:
                s = s - Xu @ self.dislike_centroid
            return s
        if self.dislike_centroid is not None:
            return -(Xu @ self.dislike_centroid)      # steer away from what was rejected
        return np.zeros(len(Xu))

    def alpha(self) -> float:
        """How much to trust the personal model, from how much it has been taught."""
        n = self.n_likes + self.n_dislikes
        if self.method == "none" or n == 0:
            return 0.0
        return float(min(MAX_ALPHA, n / (n + HALF_LABELS)))

    def status(self) -> PreferenceStatus:
        a = self.alpha()
        n = self.n_likes + self.n_dislikes
        if self.method == "none":
            note = "Not taught yet. Add reference faces or start rating results."
        elif self.method == "avoid":
            note = (f"Steering away from {self.n_dislikes} rejected face"
                    f"{'s' if self.n_dislikes != 1 else ''}. Like a few too and it can learn "
                    f"what you DO want, not just what you don't.")
        elif n < 10:
            note = (f"Learning from {n} example{'s' if n != 1 else ''}. E14 found personal "
                    f"signal becomes reliable around 10 and solid by 100 - keep rating.")
        elif a < MAX_ALPHA:
            note = f"Blending {a*100:.0f}% your taste with {100-a*100:.0f}% the population model."
        else:
            note = (f"Weighted {MAX_ALPHA*100:.0f}% to your taste - the cap. The population "
                    f"model always keeps a share, because your labels are far fewer.")
        return PreferenceStatus(self.n_likes, self.n_dislikes, self.n_references,
                                a, self.method != "none", self.method, note)

    # ------------------------------------------------------------ (de)serialise

    def save(self, path: str | Path) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, dim=self.dim,
            w=self.w if self.w is not None else np.zeros(0),
            b=self.b,
            like=self.like_centroid if self.like_centroid is not None else np.zeros(0),
            dislike=self.dislike_centroid if self.dislike_centroid is not None else np.zeros(0),
            meta=np.array(json.dumps({"n_likes": self.n_likes, "n_dislikes": self.n_dislikes,
                                      "n_references": self.n_references,
                                      "method": self.method})))

    @classmethod
    def load(cls, path: str | Path) -> "PreferenceModel":
        z = np.load(path, allow_pickle=False)
        m = cls(int(z["dim"]))
        meta = json.loads(str(z["meta"]))
        m.w = z["w"] if z["w"].size else None
        m.b = float(z["b"])
        m.like_centroid = z["like"] if z["like"].size else None
        m.dislike_centroid = z["dislike"] if z["dislike"].size else None
        m.n_likes, m.n_dislikes = meta["n_likes"], meta["n_dislikes"]
        m.n_references, m.method = meta["n_references"], meta["method"]
        return m


def blend(population_pct: np.ndarray, personal_pct: np.ndarray, alpha: float) -> np.ndarray:
    """The residual blend E14 validated: personal taste on top of, never instead of, the crowd."""
    return (1.0 - alpha) * np.asarray(population_pct) + alpha * np.asarray(personal_pct)


def _unit(X: np.ndarray) -> np.ndarray:
    return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-9, None)
