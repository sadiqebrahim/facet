"""Training objectives for the beauty head (experiment E6).

Five ways to turn the same frozen features into a score, differing ONLY in the loss:

    (a) regression   MSE against the mean rating - the conventional baseline
    (b) ordinal      CORAL-style cumulative logits, P(rating > k) for k = 1..4
    (c) distribution softmax over the rating scale, KL against the real 60-rater histogram
    (d) pairwise     Bradley-Terry on empirical preference probabilities
    (e) hybrid       KL + lambda * pairwise

Two design points worth stating, because they are what make this a fair comparison and
not a hyperparameter race:

* Every objective uses the SAME head architecture, optimiser, schedule, seed and
  early-stopping criterion. Only the loss differs.
* The ordinal and pairwise targets are built from the ACTUAL per-rater ratings rather
  than derived from the mean. For (b) the target is the empirical P(rating > k). For (d)
  the target is the empirical fraction of raters who scored face i above face j - a real
  preference probability, not a hard label manufactured from two means. Most published
  pairwise FBP work has to synthesise these; SCUT-FBP5500 ships the raw ratings, so we
  do not.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

RATING_LEVELS = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])


class ScoreHead(nn.Module):
    """One linear head over frozen features; output width depends on the objective.

    Deliberately linear: E6 is about the loss, and a linear map keeps every arm's
    capacity identical so differences are attributable to the objective alone.
    """

    def __init__(self, in_dim: int, out_dim: int = 1):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        nn.init.zeros_(self.fc.bias)
        nn.init.normal_(self.fc.weight, std=0.01)

    def forward(self, x):
        return self.fc(x)


# --------------------------------------------------------------------------- losses


def loss_regression(out, batch):
    return F.mse_loss(out.squeeze(-1), batch["mean"])


def loss_ordinal(out, batch):
    """CORAL-style: K-1 cumulative logits, BCE against empirical P(rating > k).

    Soft targets rather than hard ones, because we know the rating distribution.
    """
    return F.binary_cross_entropy_with_logits(out, batch["cum"])


def loss_distribution(out, batch):
    """KL( empirical histogram || predicted softmax ) over the rating scale."""
    logp = F.log_softmax(out, dim=-1)
    return F.kl_div(logp, batch["hist"], reduction="batchmean")


def loss_pairwise(out, batch):
    """Bradley-Terry: sigmoid(s_i - s_j) against the empirical preference probability."""
    s = out.squeeze(-1)
    i, j, p = batch["pair_i"], batch["pair_j"], batch["pair_p"]
    return F.binary_cross_entropy_with_logits(s[i] - s[j], p)


def loss_hybrid(out, batch, lam: float = 1.0):
    """KL on the distribution + lambda * Bradley-Terry on its expectation."""
    kl = loss_distribution(out, batch)
    p = F.softmax(out, dim=-1)
    s = p @ RATING_LEVELS.to(out.device)
    i, j, pp = batch["pair_i"], batch["pair_j"], batch["pair_p"]
    bt = F.binary_cross_entropy_with_logits((s[i] - s[j]) * 4.0, pp)
    return kl + lam * bt


#: name -> (output width, loss fn, needs sampled pairs)
OBJECTIVES = {
    "regression": (1, loss_regression, False),
    "ordinal": (4, loss_ordinal, False),
    "distribution": (5, loss_distribution, False),
    "pairwise": (1, loss_pairwise, True),
    "hybrid": (5, loss_hybrid, True),
}


# ---------------------------------------------------------------------- decoding


def to_score(name: str, out: torch.Tensor) -> torch.Tensor:
    """Collapse any head's output to a single ranking score."""
    if name in ("regression", "pairwise"):
        return out.squeeze(-1)
    if name == "ordinal":
        # E[rating] = 1 + sum_k P(rating > k)
        return 1.0 + torch.sigmoid(out).sum(dim=-1)
    p = F.softmax(out, dim=-1)
    return p @ RATING_LEVELS.to(out.device)


def to_distribution(name: str, out: torch.Tensor) -> torch.Tensor | None:
    """Predicted distribution over the rating scale, where the objective provides one."""
    if name in ("distribution", "hybrid"):
        return F.softmax(out, dim=-1)
    if name == "ordinal":
        # differentiate the cumulative probabilities into a pmf
        cum = torch.sigmoid(out)  # P(r > 1..4)
        ones = torch.ones_like(cum[:, :1])
        upper = torch.cat([ones, cum], dim=1)          # P(r > 0..4)
        lower = torch.cat([cum, torch.zeros_like(cum[:, :1])], dim=1)
        return (upper - lower).clamp_min(1e-6)
    return None


def aleatoric_std(name: str, out: torch.Tensor) -> torch.Tensor | None:
    """Predicted spread of the rating distribution - the rater-disagreement estimate."""
    p = to_distribution(name, out)
    if p is None:
        return None
    p = p / p.sum(dim=-1, keepdim=True)
    lv = RATING_LEVELS.to(out.device)
    mu = p @ lv
    var = (p @ (lv**2)) - mu**2
    return var.clamp_min(0).sqrt()


# ------------------------------------------------------------------------ targets


def build_targets(hist: np.ndarray, levels=(1, 2, 3, 4, 5)) -> dict[str, np.ndarray]:
    """Derive every objective's target from the empirical rating histogram."""
    hist = np.asarray(hist, dtype=np.float64)
    hist = hist / hist.sum(axis=1, keepdims=True)
    lv = np.asarray(levels, dtype=np.float64)
    mean = hist @ lv
    # P(rating > k) for k = 1..4, i.e. the tail sums
    cum = np.stack([hist[:, k + 1 :].sum(axis=1) for k in range(len(levels) - 1)], axis=1)
    return {"hist": hist, "mean": mean, "cum": cum}


def empirical_preference(R: np.ndarray, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    """P(a rater scores face i above face j), from the raw per-rater matrix.

    Ties count half, which is the standard Bradley-Terry treatment and matters here
    because ratings are integers on a 5-point scale, so ties are common.
    """
    ri, rj = R[i], R[j]
    valid = np.isfinite(ri) & np.isfinite(rj)
    wins = (ri > rj).astype(np.float64) + 0.5 * (ri == rj)
    wins[~valid] = np.nan
    n = valid.sum(axis=1)
    with np.errstate(invalid="ignore"):
        p = np.nansum(np.where(valid, wins, 0.0), axis=1) / np.maximum(n, 1)
    p[n == 0] = 0.5
    return p
