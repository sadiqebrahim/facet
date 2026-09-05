#!/usr/bin/env python
"""Train and serialise the production attractiveness head.

Assembles the experiments' conclusions into one artifact:
  E6  -> label-distribution objective (for its outputs, not its accuracy)
  E5  -> m0.25 crop features
  E12 -> deep ensemble + split-conformal intervals, plus an OOD reference set
  E7  -> the OOD gate exists because calibration does not survive domain shift

Writes a single .npz holding weights, calibration constants, the OOD reference and the
provenance string that every downstream display must carry.

Usage:
    python scripts/train_beauty_head.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from facet.data.scut_fbp5500 import ScutFbp5500  # noqa: E402
from facet.models.beauty_head import BeautyHead  # noqa: E402
from facet.training.objectives import (  # noqa: E402
    ScoreHead, build_targets, loss_distribution, to_score,
)
from facet.utils.hashing import hash_obj  # noqa: E402
from facet.utils.seed import seed_everything  # noqa: E402

FEATURE_DIR = ROOT / "artifacts/features"
PARTS = ("arcface_buffalo_l", "clip")
MARGIN = 0.25


def load(parts, margin=MARGIN, size=112):
    mats, order = [], None
    for p in parts:
        st = f"{p}__m{margin:g}__s{size}"
        arr = np.load(FEATURE_DIR / f"{st}.npy").astype(np.float32)
        names = json.loads((FEATURE_DIR / f"{st}.json").read_text())["filenames"]
        order = order or names
        arr /= np.clip(np.linalg.norm(arr, axis=1, keepdims=True), 1e-9, None)
        mats.append(arr)
    return np.hstack(mats), order


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default=str(ROOT / "data/raw/SCUT-FBP5500_v2"))
    ap.add_argument("--out", default=str(ROOT / "artifacts/models/beauty_head.npz"))
    ap.add_argument("--members", type=int, default=5)
    ap.add_argument("--ood-ref-size", type=int, default=2000)
    ap.add_argument("--ood-percentile", type=float, default=99.0)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = ScutFbp5500(args.data_root)
    X, order = load(PARTS)
    tgt = build_targets(ds.histogram_matrix(order).astype(np.float64))
    y = tgt["mean"]

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(X))
    n_va, n_cal = len(X) // 10, len(X) // 5
    va, cal, fit = perm[:n_va], perm[n_va:n_va + n_cal], perm[n_va + n_cal:]
    print(f"fit={len(fit)} calib={len(cal)} val={len(va)}  dim={X.shape[1]}  device={device}")

    mean, scale = X[fit].mean(0), X[fit].std(0) + 1e-6
    def norm(A): return ((A - mean) / scale).astype(np.float32)

    Ws, bs = [], []
    for m in range(args.members):
        torch.manual_seed(args.seed + m)
        head = ScoreHead(X.shape[1], 5).to(device)
        opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
        xt = torch.from_numpy(norm(X[fit])).to(device)
        xv = torch.from_numpy(norm(X[va])).to(device)
        batch = {"hist": torch.from_numpy(tgt["hist"][fit]).float().to(device)}
        best, state, bad = -np.inf, None, 0
        for ep in range(args.epochs):
            head.train()
            loss = loss_distribution(head(xt), batch)
            opt.zero_grad(); loss.backward(); opt.step()
            if ep % 5 == 0:
                head.eval()
                with torch.no_grad():
                    s = to_score("distribution", head(xv)).cpu().numpy()
                rho = stats.spearmanr(y[va], s).statistic
                if rho > best:
                    best, bad = rho, 0
                    state = {k: v.detach().clone() for k, v in head.state_dict().items()}
                else:
                    bad += 1
                    if bad > 10:
                        break
        head.load_state_dict(state)
        Ws.append(head.fc.weight.detach().cpu().numpy().T.copy())
        bs.append(head.fc.bias.detach().cpu().numpy().copy())
        print(f"  member {m}: val spearman {best:.4f}")

    ood_idx = rng.choice(fit, size=min(args.ood_ref_size, len(fit)), replace=False)
    ref = X[ood_idx] / np.clip(np.linalg.norm(X[ood_idx], axis=1, keepdims=True), 1e-9, None)

    head = BeautyHead(mean, scale, Ws, bs, conformal={0.68: 1.0, 0.90: 1.0, 0.95: 1.0},
                      ood_ref=ref, ood_thresh=1.0, version="", config_hash="")

    # split-conformal on the held-out calibration split (E12)
    preds = head.predict(X[cal])
    mu = np.array([p.mean for p in preds])
    sig = np.array([np.sqrt(p.aleatoric**2 + p.epistemic**2) for p in preds])
    err = np.abs(mu - y[cal])
    conformal = {}
    for lvl, alpha in ((0.68, 0.32), (0.90, 0.10), (0.95, 0.05)):
        n = len(err)
        qq = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
        conformal[lvl] = float(np.quantile(err / np.clip(sig, 1e-9, None), qq))
    head.conformal = conformal

    # OOD threshold: the training set's own tail, so in-domain faces rarely trip the gate
    head.ood_thresh = float(np.percentile(head.ood_score(X[cal]), args.ood_percentile))

    preds = head.predict(X[va])
    mu_va = np.array([p.mean for p in preds])
    cov = float(np.mean(np.abs(mu_va - y[va]) <=
                        conformal[0.90] * np.array([np.sqrt(p.aleatoric**2 + p.epistemic**2)
                                                    for p in preds])))
    head.metrics = {
        "val_spearman": float(stats.spearmanr(y[va], mu_va).statistic),
        "val_pearson": float(np.corrcoef(y[va], mu_va)[0, 1]),
        "val_mae": float(np.abs(mu_va - y[va]).mean()),
        "val_coverage_90": cov,
        "conformal": conformal,
        "ood_threshold": head.ood_thresh,
        "n_members": args.members,
    }
    head.version = f"beauty_ldl_ens{args.members}:scut5500:m{MARGIN:g}"
    head.config_hash = hash_obj({**vars(args), "parts": PARTS, "margin": MARGIN})
    head.save(args.out)

    print(f"\nval: PC={head.metrics['val_pearson']:.4f} rho={head.metrics['val_spearman']:.4f} "
          f"MAE={head.metrics['val_mae']:.4f}")
    print(f"conformal q: {conformal}")
    print(f"coverage@90 on val: {cov:.4f}  (nominal 0.90)")
    print(f"OOD threshold (p{args.ood_percentile:g} of calib): {head.ood_thresh:.4f}")
    print(f"[ok] -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
