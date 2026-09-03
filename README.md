# Facet

Face analysis, filtering and ranking. Point it at a directory of images, describe the
faces you're looking for, and get back a ranked, explained result set.

**Status: Phase 1 (research) complete, Phase 4 (baselines) underway. There is no UI yet —
by design.** The project is empirical: models get selected by measurement, not by
reputation.

---

## What this is, and what it is not

This system predicts **how a specific, narrow group of human raters would have rated a
face**. It does not measure beauty. Beauty is not a property of faces that can be
measured.

That is not boilerplate — it is the central finding of the research phase and it shapes
the architecture. The dataset everything is trained on (SCUT-FBP5500) was rated by 60
volunteers aged 18–27 at one Chinese university in 2017, and **those raters correlate
with their own crowd average at only r ≈ 0.77**. Published models hit r ≈ 0.93 against
that average — which means they are excellent at predicting *that specific group's mean*,
not at perceiving beauty. See [`docs/RESEARCH.md §1.1`](docs/RESEARCH.md).

Three consequences run through the whole design:

1. Attractiveness is always reported **with its distribution**, never as a bare number.
2. **Rater disagreement is signal, not noise** — a face everyone rates 3.5 and a face
   split between 2 and 5 are different results, and the system distinguishes them.
3. Personal preference is a first-class feature, because r ≈ 0.77 means there is a great
   deal of taste that no population model can explain.

## Licensing — read before building anything on this

**The current model stack cannot be used commercially.** SCUT-FBP5500 is non-commercial
research only, and InsightFace's pretrained weights are too. A model trained on
non-commercial data inherits that restriction. A commercially-clean path exists (FairFace
+ MiVOLO + self-collected ratings) and is documented, but is not what is built here.
Full register: [`docs/LICENSING.md`](docs/LICENSING.md).

All processing is local. No image, crop, embedding or prediction leaves the machine.

---

## Results so far

| Experiment | Question | Outcome |
|---|---|---|
| [`exp001`](experiments/exp001_frozen_linear_probe/FINDINGS.md) | Do frozen features + a linear head work? | **Yes, better than expected.** Frozen CLIP+ArcFace + ridge reaches **PC 0.9398** on the official 5-fold splits, beating fine-tuned ResNeXt-50 (0.8997) and the reported 2025 SOTA range (0.932–0.935) — with no backbone training. The label-distribution head beat mean-regression on every representation. Ensemble spread was found to be a *useless* confidence signal (22 % coverage at 68 % nominal). |
| [`e1_identity_audit`](experiments/e1_identity_audit/FINDINGS.md) | Are the official splits subject-disjoint? | **No.** ~5 % of every official test set shares an identity with training (near-duplicates up to cosine 0.998). Measured impact on the headline: +0.003 PC — disclosed, but it does not explain the result. |

Next up is the gate that matters: **E7, cross-dataset generalisation to MEBeauty.**
Nothing here says the system works on images outside this one benchmark.

## Documentation

| Document | Contents |
|---|---|
| [`docs/RESEARCH.md`](docs/RESEARCH.md) | **The Phase 1 report.** Models, papers, datasets, beauty/age/gender/ranking/ensemble/uncertainty approaches, licensing, biases, recommended experiments, architecture, roadmap |
| [`docs/DATASETS.md`](docs/DATASETS.md) | Per-dataset cards: size, demographics, rating methodology, biases, licensing, identity-overlap risk |
| [`docs/LICENSING.md`](docs/LICENSING.md) | License register, the two viable paths, privacy/biometric obligations, responsible-use rules |
| [`experiments/`](experiments/) | One directory per experiment: results, run manifest, findings |

---

## Quick start

Requires the SCUT-FBP5500 dataset (`make data`) and a CUDA GPU (CPU works, ~5× slower).

```bash
make features   # detect, align, encode; cached and keyed by (encoder, crop protocol)
make exp001     # frozen representation + linear probe on the official splits
make test
```

## Repository layout

```
docs/            research report, dataset cards, licensing register
configs/         model registry, crop protocol, ranking weights - no magic numbers in code
src/facet/
  data/          dataset loaders, official splits, identity-disjoint split construction
  models/        detectors, encoders, geometry - all behind swappable protocols
  training/      prediction heads and losses
  evaluation/    regression, ranking, calibration and fairness metrics
  pipeline/      (Phase 8) discovery, dedup, incremental indexing
  query/         (Phase 9) filters, scoring, ranking
scripts/         runnable entry points
experiments/     results.json + manifest.json per run - the experimental record
tests/
```

### The one architectural decision that matters

The pipeline is split into **encode** (expensive, cached forever) and **predict** (cheap,
replaceable):

```
image → detect → align → [ ENCODE: 512-d embedding, cached ] → [ PREDICT: linear heads ]
                          ~8 ms/face, once per image ever        ~microseconds, swappable
```

If frozen features carry the signal — which is exactly what `exp001` measures — then
changing the beauty model never requires re-indexing, and a **per-user preference model is
just another linear head over the same cached features**. That makes personalisation
nearly free instead of a research project. See [`docs/RESEARCH.md §15.1`](docs/RESEARCH.md).

## Reproducibility

Every experiment writes a `manifest.json` recording dataset version, split methodology,
config hash, seed, hardware, library versions, timings and metrics. A metric without its
split methodology is not a result — the age-estimation literature learned this the hard
way (`arXiv:2307.04570`), and it is why experiment **E1** (identity-leakage audit of the
official splits) gates everything downstream.

## Citation

If you use this work, cite the underlying datasets and models — SCUT-FBP5500
(Liang et al., ICPR 2018), InsightFace (Deng et al.), MiVOLO (Kuprashevich & Tolstykh).
