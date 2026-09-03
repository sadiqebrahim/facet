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

That is not boilerplate — it is the central finding of the research phase, it shapes the
architecture, and it is now measured: on a demographically **balanced** test set, this system
selects White faces into its top-100 at 2.2× their population share and Southeast Asian faces
at 0.23× ([E11](experiments/e11_fairness/FINDINGS.md)). That skew is a property of the rater
pool the model was trained on. A different pool would give a different skew — not no skew. The dataset everything is trained on (SCUT-FBP5500) was rated by 60
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

| [`e7_cross_dataset`](experiments/e7_cross_dataset/FINDINGS.md) | **The gate.** Does any of it transfer off-benchmark? | **Conditional pass.** Trained on SCUT, tested on 2,520 held-out in-the-wild MEBeauty faces: ranking transfers (ρ **0.608**, 72 % of within-dataset performance, vs a 0.872 human ceiling). But **cross-group calibration does not** — the top-100 shares only **20/100** members with the one MEBeauty's own raters would pick. Also: training on 2,520 *diverse* images transfers better than 5,500 posed ones. |

| [`e14_personalisation`](experiments/e14_personalisation/FINDINGS.md) | Can we learn an individual's taste, and at what label cost? | **Depends entirely on rater-pool diversity.** Seven SCUT raters rated all 5,500 images twice, giving a real ceiling: people agree with the crowd (ρ 0.766) *more than with their own earlier judgment* (ρ 0.575). The population model already sits at **95.5 % of the achievable maximum**, so on that homogeneous pool personalisation **hurts** at every budget. On diverse MEBeauty it helps — **+0.025 Spearman from 100 labels**, 68 % of users — and 5.4× more for users the consensus fits worst. |

| [`e5_crop_sensitivity`](experiments/e5_crop_sensitivity/FINDINGS.md) | Does the crop protocol matter more than the backbone? | **No — my prior was wrong.** Backbone choice spans 4–10× more than crop margin. But margin matters *hugely for ArcFace alone* (transfer +61 % relative), and the in-benchmark and transfer optima **disagree** — so the production margin moved 0.00 → **0.25**, worth +0.033 transfer ρ for free. |

| [`e6_objectives`](experiments/e6_objectives/FINDINGS.md) | Which training objective — regression, ordinal, LDL, pairwise, hybrid? | **None of them, on accuracy.** All five sit within **1.3 seed standard deviations** (Welch p = 0.068); a first single-seed run showed a 0.028 "win" that was pure noise. They differ ~8× on *auxiliary* outputs, so the LDL head wins for producing a usable rating distribution, not for being more accurate. Also corrects an over-claim from exp001. |

| [`e12_uncertainty`](experiments/e12_uncertainty/FINDINGS.md) | Can we show an honest confidence number? | **In-domain yes, out-of-domain no.** Every *raw* uncertainty channel is badly miscalibrated — 0.998 to 0.065 coverage at a 68 % nominal level, failing in opposite directions. Conformal fixes all of them in-domain (0.90 at nominal 0.90). But calibrated on SCUT and applied to MEBeauty, **a 90 % interval delivers 43 %**. Calibration must be per-collection and OOD-gated. |

| [`e11_fairness`](experiments/e11_fairness/FINDINGS.md) | Where does demographic bias actually enter? | **Not at detection** (recall spread 0.0006 across 7 balanced groups) — it enters late and hard. On a *balanced* set the system puts White faces in its top-100 at **2.2× their share** and Southeast Asian faces at **0.23×**. Gender accuracy spread 0.174; age error varies 5× more by age than by race. The photo-quality confound was tested and **not supported** (R² = 0.017) — though styling remains untested. |

**What E7 changed.** Ranking works well enough to build on, so the project proceeds — but
with a narrower scope than the brief assumed: absolute thresholds like "attractiveness above
4.0" are not supportable across domains and are replaced by percentile-within-collection;
production training moves to in-the-wild data; and personalisation is promoted from a
Phase 10 extra to a core requirement, because two reasonable rater pools disagreeing on 80 %
of a result set means no single population ranking can be the honest answer.

**What E14 added.** Personalisation is worth building, but it is a *refinement, not the
resolution* — a +0.025 gain does not offset 80 % of a top-100 changing between rater pools.
It must use the residual formulation (`population + user_residual`, since training on a
user's labels alone is catastrophic cold-start), and it should be gated on how poorly the
population model already fits that user. The bigger lever remains **whose ratings you train
on in the first place.**

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
conda env create -f environment.yml && conda activate facet

make data       # download SCUT-FBP5500 (prompts for licence acceptance)
make features   # detect, align, encode; cached and keyed by (encoder, crop protocol)
make exp001     # frozen representation + linear probe on the official splits
make test

# MEBeauty (the cross-dataset test set) is fetched separately - see docs/DATASETS.md:
#   git clone https://github.com/fbplab/MEBeauty-database
python scripts/run_e7_cross_dataset.py
```

`make` uses whatever `python` is on your PATH; override with `make PY=/path/to/python <target>`.

**Neither dataset is redistributed here** — both are non-commercial research only and
prohibit redistribution. The repo ships download scripts and pointers, never data.

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
