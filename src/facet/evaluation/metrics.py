"""Evaluation metrics.

Covers the four families the brief asks for - regression, classification, ranking and
calibration - plus the two that matter most for this particular product:

* `ndcg_at_k` / `precision_at_k`: the user only ever sees the top N results, so
  top-of-list quality is the metric that tracks product quality. A model with mediocre
  global correlation but excellent top-100 precision is the better product.
* `pairwise_accuracy`: scale-free, and the metric that survives cross-dataset comparison
  when rating scales differ (see docs/RESEARCH.md section 9.2).
"""
from __future__ import annotations

import numpy as np
from scipy import stats

# --------------------------------------------------------------------- regression


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """MAE, RMSE, Pearson, Spearman - the standard SCUT-FBP5500 reporting set."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    resid = y_pred - y_true
    return {
        "mae": float(np.abs(resid).mean()),
        "rmse": float(np.sqrt((resid**2).mean())),
        "pc": float(stats.pearsonr(y_true, y_pred).statistic),
        "spearman": float(stats.spearmanr(y_true, y_pred).statistic),
    }


# ------------------------------------------------------------------------ ranking


def pairwise_accuracy(
    y_true: np.ndarray, y_pred: np.ndarray, n_pairs: int = 200_000, seed: int = 0
) -> float:
    """Fraction of comparable pairs ordered correctly.

    Samples pairs rather than enumerating O(n^2). Ties in y_true are excluded, since
    they are not comparable and would otherwise inflate or deflate the score depending
    on tie-breaking.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    n = len(y_true)
    rng = np.random.default_rng(seed)
    i = rng.integers(0, n, n_pairs)
    j = rng.integers(0, n, n_pairs)
    comparable = y_true[i] != y_true[j]
    if comparable.sum() == 0:
        return float("nan")
    i, j = i[comparable], j[comparable]
    correct = (y_true[i] > y_true[j]) == (y_pred[i] > y_pred[j])
    return float(correct.mean())


def kendall_tau(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(stats.kendalltau(y_true, y_pred).statistic)


def ndcg_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int = 100) -> float:
    """NDCG@k with relevance = true score shifted to be non-negative.

    This is the metric closest to "did the top-k the user actually sees contain the
    genuinely best faces".
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    rel = y_true - y_true.min()
    k = min(k, len(rel))
    discount = 1.0 / np.log2(np.arange(2, k + 2))

    order = np.argsort(-y_pred, kind="stable")[:k]
    dcg = float((rel[order] * discount).sum())
    ideal = np.sort(rel)[::-1][:k]
    idcg = float((ideal * discount).sum())
    return dcg / idcg if idcg > 0 else float("nan")


def precision_at_k(
    y_true: np.ndarray, y_pred: np.ndarray, k: int = 100, quantile: float = 0.9
) -> float:
    """Of the top-k we return, what fraction are truly in the top `quantile` of the set."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    k = min(k, len(y_true))
    threshold = np.quantile(y_true, quantile)
    top = np.argsort(-y_pred, kind="stable")[:k]
    return float((y_true[top] >= threshold).mean())


def ranking_metrics(y_true: np.ndarray, y_pred: np.ndarray, k: int = 100) -> dict[str, float]:
    return {
        "pairwise_acc": pairwise_accuracy(y_true, y_pred),
        "kendall_tau": kendall_tau(y_true, y_pred),
        f"ndcg@{k}": ndcg_at_k(y_true, y_pred, k),
        f"precision@{k}": precision_at_k(y_true, y_pred, k),
    }


# -------------------------------------------------------------------- distribution


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """Mean KL(p || q) over rows. p = true rating histogram, q = predicted."""
    p = np.clip(np.asarray(p, dtype=np.float64), eps, None)
    q = np.clip(np.asarray(q, dtype=np.float64), eps, None)
    p = p / p.sum(axis=-1, keepdims=True)
    q = q / q.sum(axis=-1, keepdims=True)
    return float((p * np.log(p / q)).sum(axis=-1).mean())


# --------------------------------------------------------------------- calibration


def calibration_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_std: np.ndarray
) -> dict[str, float]:
    """How trustworthy is a predicted uncertainty?

    `coverage_*`  - fraction of true values inside the nominal Gaussian interval.
                    Well-calibrated means coverage_68 ~ 0.68, coverage_95 ~ 0.95.
    `err_std_corr`- Spearman correlation between predicted uncertainty and actual error.
                    This is the one that matters for the UI: a confidence number is only
                    useful if it actually predicts when the model is wrong.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    y_std = np.clip(np.asarray(y_std, dtype=np.float64).ravel(), 1e-9, None)
    err = np.abs(y_pred - y_true)
    z = err / y_std
    # A constant sigma carries no per-sample information, so the rank correlation is
    # undefined rather than zero - report NaN instead of letting scipy warn.
    corr = (
        float(stats.spearmanr(y_std, err).statistic)
        if np.ptp(y_std) > 0
        else float("nan")
    )
    return {
        "coverage_68": float((z <= 1.0).mean()),
        "coverage_95": float((z <= 1.96).mean()),
        "mean_std": float(y_std.mean()),
        "err_std_corr": corr,
    }


# ------------------------------------------------------------------- convenience


def full_report(
    y_true: np.ndarray, y_pred: np.ndarray, k: int = 100, y_std: np.ndarray | None = None
) -> dict[str, float]:
    out = regression_metrics(y_true, y_pred)
    out.update(ranking_metrics(y_true, y_pred, k=k))
    if y_std is not None:
        out.update(calibration_metrics(y_true, y_pred, y_std))
    return out


def group_report(
    y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray, k: int = 100
) -> dict[str, dict[str, float]]:
    """Per-group metrics. The fairness audit (E11) is built on this."""
    groups = np.asarray(groups)
    out = {}
    for g in sorted(set(groups.tolist())):
        m = groups == g
        if m.sum() < 10:
            continue
        out[str(g)] = {"n": int(m.sum()), **full_report(y_true[m], y_pred[m], k=k)}
    return out
