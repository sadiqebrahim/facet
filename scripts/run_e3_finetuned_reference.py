#!/usr/bin/env python
"""Experiment E3 - reproduce the published fine-tuned baselines.

docs/RESEARCH.md 14: "An experiment harness that cannot reproduce a known result cannot be
trusted to measure a new one." Every number this project has produced rests on the loaders,
splits and metrics in `facet/`. This checks them against ground truth that was published
before we existed.

Targets, verified from arXiv:1801.06345 Table VII/VIII and reproduced in Appendix A of the
research report:

    5-fold CV     AlexNet 0.8634 | ResNet-18 0.8900 | ResNeXt-50 0.8997   (PC)
    60/40 split   AlexNet 0.8298 | ResNet-18 0.8513 | ResNeXt-50 0.8777

Protocol follows the paper: ImageNet-pretrained backbone, fine-tuned end to end on the raw
SCUT-FBP5500 images (NOT our SCRFD-aligned crops - the point is to reproduce their setup,
not ours), MSE against the mean rating, evaluated on the dataset's own official splits.

There are three possible outcomes and all three are informative:

  * we land near the published numbers -> the harness is sound and every other result stands.
  * we land far BELOW -> our training recipe is undertuned, which would mean exp001's claim
    that frozen features beat fine-tuning was measured against a weak opponent.
  * we land far ABOVE -> something leaks. Given E1 found ~5% identity leakage in the official
    splits, that is a live possibility and would need chasing down.

Usage:
    python scripts/run_e3_finetuned_reference.py
    python scripts/run_e3_finetuned_reference.py --models resnet18 --splits cv1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from facet.data.scut_fbp5500 import ScutFbp5500  # noqa: E402
from facet.evaluation.metrics import regression_metrics  # noqa: E402
from facet.utils.run import RunManifest  # noqa: E402
from facet.utils.seed import seed_everything  # noqa: E402

#: Per-model fine-tuning learning rates, found by probing rather than assumed. AlexNet's
#: large fully-connected classifier is far more sensitive: at 1e-3 its loss reached 2.8e6
#: within 25 iterations, while ResNet-18 trains stably there. Using one shared lr would
#: have meant reporting a "bad" AlexNet that was really a diverged one.
LR = {"alexnet": 3e-4, "resnet18": 1e-3, "resnext50": 1e-3}

PUBLISHED = {
    "5fold": {"alexnet": 0.8634, "resnet18": 0.8900, "resnext50": 0.8997},
    "split6040": {"alexnet": 0.8298, "resnet18": 0.8513, "resnext50": 0.8777},
}


class FaceRegressionDataset(Dataset):
    def __init__(self, ds: ScutFbp5500, names: list[str], train: bool):
        from torchvision import transforms as T

        self.ds, self.names = ds, names
        self.y = ds.labels.loc[names, "mean"].to_numpy(np.float32)
        norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.tf = (
            T.Compose([T.Resize(256), T.RandomCrop(224), T.RandomHorizontalFlip(),
                       T.ToTensor(), norm])
            if train else
            T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), norm])
        )

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        img = Image.open(self.ds.image_path(self.names[i])).convert("RGB")
        return self.tf(img), self.y[i]


def build(name: str):
    from torchvision import models

    if name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(m.fc.in_features, 1)
    elif name == "resnext50":
        m = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(m.fc.in_features, 1)
    elif name == "alexnet":
        m = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
        m.classifier[6] = nn.Linear(m.classifier[6].in_features, 1)
    else:
        raise ValueError(name)
    return m


def train_eval(ds, model_name, split, device, epochs, bs, lr, workers, seed):
    torch.manual_seed(seed)
    tr = DataLoader(FaceRegressionDataset(ds, split.train, True), batch_size=bs, shuffle=True,
                    num_workers=workers, pin_memory=True, drop_last=True, persistent_workers=True)
    te = DataLoader(FaceRegressionDataset(ds, split.test, False), batch_size=bs * 2,
                    shuffle=False, num_workers=workers, pin_memory=True, persistent_workers=True)

    model = build(model_name).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    crit = nn.MSELoss()

    t0 = time.time()
    for ep in range(epochs):
        model.train()
        for x, y in tr:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                loss = crit(model(x).squeeze(-1), y)
            # A diverged run otherwise reports a silent `nan` PC, which reads like a bad
            # result rather than a broken one. Fail loudly instead.
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"{model_name}/{split.name}: loss became {loss.item()} at epoch {ep}. "
                    "Training diverged - lower --lr."
                )
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sched.step()
    train_sec = time.time() - t0

    model.eval()
    preds, ys = [], []
    with torch.no_grad():
        for x, y in te:
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                p = model(x.to(device)).squeeze(-1)
            preds.append(p.float().cpu().numpy())
            ys.append(y.numpy())
    m = regression_metrics(np.concatenate(ys), np.concatenate(preds))
    m["train_sec"] = round(train_sec, 1)
    return m


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default=str(ROOT / "data/raw/SCUT-FBP5500_v2"))
    ap.add_argument("--out-dir", default=str(ROOT / "experiments/e3_finetuned_reference"))
    ap.add_argument("--models", nargs="*", default=["alexnet", "resnet18", "resnext50"])
    ap.add_argument("--splits", nargs="*",
                    default=["cv1", "cv2", "cv3", "cv4", "cv5", "split6040"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    # Default is per-model (see LR above); --lr overrides it for every model.
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    manifest = RunManifest(
        experiment="e3_finetuned_reference",
        description="Reproduce the published fine-tuned SCUT-FBP5500 baselines",
        config=vars(args), seed=args.seed,
        dataset="SCUT-FBP5500_v2",
        split_methodology=(
            "The dataset's own official 5-fold CV and 60/40 split. Raw images (not our "
            "SCRFD-aligned crops), ImageNet-pretrained backbones, MSE on the mean rating - "
            "the paper's protocol, so the numbers are comparable to theirs."
        ),
    )

    ds = ScutFbp5500(args.data_root)
    results: dict = {}
    lr_desc = args.lr if args.lr is not None else LR
    print(f"device={device} epochs={args.epochs} bs={args.batch_size} lr={lr_desc}\n")
    print(f"{'model':<11} {'split':<10} {'PC':>8} {'MAE':>8} {'RMSE':>8} {'train s':>8}")
    print("-" * 58)

    for name in args.models:
        results[name] = {}
        for sname in args.splits:
            lr = args.lr if args.lr is not None else LR[name]
            m = train_eval(ds, name, ds.splits[sname], device, args.epochs,
                           args.batch_size, lr, args.workers, args.seed)
            m["lr"] = lr
            results[name][sname] = m
            print(f"{name:<11} {sname:<10} {m['pc']:>8.4f} {m['mae']:>8.4f} "
                  f"{m['rmse']:>8.4f} {m['train_sec']:>8.1f}")
        cv = [k for k in results[name] if k.startswith("cv")]
        if cv:
            results[name]["cv_mean"] = {
                k: float(np.mean([results[name][c][k] for c in cv]))
                for k in ("pc", "mae", "rmse")
            }
        print("-" * 58)

    print(f"\n{'model':<11} {'ours 5-fold':>12} {'published':>10} {'delta':>8} | "
          f"{'ours 60/40':>11} {'published':>10} {'delta':>8}")
    verdict = {}
    for name in args.models:
        r = results[name]
        ours5 = r.get("cv_mean", {}).get("pc")
        pub5 = PUBLISHED["5fold"].get(name)
        ours6 = r.get("split6040", {}).get("pc")
        pub6 = PUBLISHED["split6040"].get(name)
        line = f"{name:<11}"
        line += f" {ours5:>12.4f} {pub5:>10.4f} {ours5-pub5:>+8.4f}" if ours5 else " " * 32
        line += f" | {ours6:>11.4f} {pub6:>10.4f} {ours6-pub6:>+8.4f}" if ours6 else ""
        print(line)
        verdict[name] = {"ours_5fold": ours5, "published_5fold": pub5,
                         "delta_5fold": (ours5 - pub5) if ours5 else None,
                         "ours_6040": ours6, "published_6040": pub6,
                         "delta_6040": (ours6 - pub6) if ours6 else None}
    results["_verdict"] = verdict

    # Status is reported PER MODEL. A single max over all models conflates "the harness is
    # wrong" with "one recipe is undertuned" - and those call for completely different
    # responses. A model we had to train at a lower learning rate (AlexNet needed 3e-4
    # because 1e-3 diverged) is expected to undershoot under a shared epoch budget; that
    # says nothing about the loaders, splits or metrics.
    def status_of(d):
        if d is None:
            return "n/a"
        return "REPRODUCED" if abs(d) <= 0.02 else "CLOSE" if abs(d) <= 0.04 else "OFF"

    print()
    for name, v in verdict.items():
        v["status_5fold"] = status_of(v["delta_5fold"])
        v["status_6040"] = status_of(v["delta_6040"])
        print(f"  {name:<11} 5-fold {v['status_5fold']:<11} 60/40 {v['status_6040']}")

    # The harness verdict rests on the modern backbones, which are the ones our recipe
    # actually suits and the ones later experiments are compared against.
    core = [verdict[m]["delta_5fold"] for m in ("resnet18", "resnext50")
            if m in verdict and verdict[m]["delta_5fold"] is not None]
    if core:
        worst = max(abs(d) for d in core)
        over = [d for d in core if d > 0.02]
        results["_max_abs_delta_core"] = worst
        results["_status"] = ("HARNESS VALIDATED" if worst <= 0.02 and not over
                              else "NEEDS REVIEW")
        print(f"\n  core (resnet18, resnext50) max |delta| = {worst:.4f} -> {results['_status']}")
        print("  Coming in ABOVE published would suggest leakage; below suggests a weaker "
              "recipe.\n  E1 found ~5% identity leakage in these official splits, so the "
              "sign matters.")

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2, default=float))
    manifest.metrics = results
    manifest.finish(out)
    print(f"\n[ok] -> {out/'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
