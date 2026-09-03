# E12 — Uncertainty and calibration

exp001 found a concrete defect: bootstrap-ensemble spread gave **22 % coverage at a 68 %
nominal level** and correlated with actual error at 0.10. Showing that as "confidence" would
put an authoritative-looking meaningless number in front of the user — the exact failure
`docs/RESEARCH.md §11.4` warns against. This experiment finds a method that works, and
establishes where it stops working.

Run: 2026-09-03 · RTX A6000 · seed 1337 · raw numbers in `results.json`

## Setup

Five uncertainty channels on the E6 production head (LDL) over the E5 production crop
(m0.25), SCUT official cv1, with the training split further divided into fit (3,080) /
conformal calibration (880) / early-stopping validation (440):

| channel | what it measures |
|---|---|
| `aleatoric` | spread of the predicted rating distribution — irreducible rater disagreement |
| `ensemble` | spread across 5 independently seeded heads — epistemic |
| `mc_dropout` | dropout active at inference, 30 passes |
| `tta` | horizontal-flip test-time augmentation, 2 views |
| `combined` | √(aleatoric² + epistemic²) |

## Result 1 — every raw uncertainty channel is badly miscalibrated

Nominal 68 / 90 / 95 % Gaussian intervals, in-domain:

| method | cov@68 | cov@90 | cov@95 | mean σ | err corr |
|---|---:|---:|---:|---:|---:|
| aleatoric | **0.998** | 1.000 | 1.000 | 1.0670 | 0.082 |
| ensemble | **0.065** | 0.119 | 0.143 | 0.0229 | 0.140 |
| mc_dropout | 0.285 | 0.466 | 0.543 | 0.0898 | 0.124 |
| tta | 0.074 | 0.135 | 0.159 | 0.0271 | 0.084 |
| combined | 0.998 | 1.000 | 1.000 | 1.0673 | 0.082 |
| *(true MAE)* | | | | *0.1921* | |

Not one is close. They fail in **opposite directions**, which is diagnostic:

- **Aleatoric massively over-covers** (σ = 1.07 against an MAE of 0.19). It is measuring the
  right thing for the wrong question: rater disagreement is ≈0.64 on the 1–5 scale, and that
  is a property of *the raters*, not of the model's error.
- **Ensemble and TTA massively under-cover** (σ = 0.02–0.03). Five linear heads on 3,080
  samples agree almost perfectly, so member spread captures parameter uncertainty while
  ignoring the dominant irreducible noise. This generalises exp001's bagging result: **the
  defect is not specific to bootstrapping, it is inherent to raw model spread.**

**No raw model output should ever be shown as confidence.** That is now measured across five
methods, not inferred from one.

## Result 2 — conformal prediction fixes coverage for all of them

Split-conformal, scale factor fitted on the held-out calibration set:

| method | cov@68 | cov@90 | cov@95 | **width@90** |
|---|---:|---:|---:|---:|
| **aleatoric** | 0.688 | 0.907 | 0.944 | **0.821** |
| **combined** | 0.688 | 0.907 | 0.944 | 0.822 |
| mc_dropout | 0.690 | 0.898 | 0.946 | 0.837 |
| ensemble | 0.663 | 0.893 | 0.955 | 1.071 |
| tta | 0.696 | 0.893 | 0.935 | **2.505** |

Every channel now hits its nominal level — which is exactly what conformal guarantees, and a
good check that the implementation is right. **Coverage stops being the discriminator and
sharpness becomes one.** Aleatoric and combined give the tightest usable intervals (±0.41 on
a 1–5 scale at 90 %); TTA is 3× wider for the same coverage and is not worth its extra
forward pass.

Note the fitted scale factors: 0.21 for aleatoric but **11.5 for ensemble and 14–88 for TTA**.
A method needing a 88× correction is not measuring uncertainty in any meaningful unit; it is
being rescued by the calibration set.

## Result 3 — ⚠ conformal calibration does NOT survive the domain shift

Intervals calibrated on SCUT, applied to held-out MEBeauty (MEBeauty's 1–10 scale mapped onto
SCUT's 1–5 by rank position, the only scale-free way to compare absolute intervals):

| method | cov@68 | cov@90 | cov@95 |
|---|---:|---:|---:|
| aleatoric | 0.236 | 0.428 | 0.525 |
| combined | 0.236 | 0.428 | 0.526 |
| mc_dropout | 0.329 | 0.566 | 0.687 |
| **ensemble** | **0.478** | **0.757** | **0.876** |
| **tta** | 0.470 | **0.781** | **0.879** |
| *nominal* | *0.68* | *0.90* | *0.95* |

**A "90 % confidence interval" delivers 43 % coverage on out-of-domain images.** This is not a
bug: conformal guarantees coverage only under *exchangeability*, and a domain shift violates
that by construction. §11.2 recommended conformal for the UI, and that recommendation needed
checking rather than assuming — this is the check, and it comes with a large caveat attached.

The ordering under shift is the interesting part and it **inverts** the in-domain ranking. The
epistemic channels (ensemble 0.757, TTA 0.781 at nominal 0.90) degrade *least*; the aleatoric
channel — sharpest in-domain — degrades *most* (0.428). Mechanistically that is what should
happen: epistemic uncertainty is supposed to grow off-distribution, whereas the predicted
rating spread is a learned function that has no way to know it is being asked about an
unfamiliar face.

**So the two channels are good at different jobs**: aleatoric for sharp in-domain intervals,
epistemic for noticing that the intervals should not be trusted at all.

## Result 4 — the aleatoric channel is a poor "is this face polarising?" detector

`docs/RESEARCH.md §6.3` and `§11.1` claimed the LDL head's predicted spread would answer
"do humans genuinely disagree about this face" — presented as the strongest argument for LDL
over regression.

Measured: **corr(predicted aleatoric, true rater σ) = 0.139.**

That is weak, and consistent with E6, which found the same head at r = 0.154 while the
**hybrid** (KL + pairwise) head reached **r = 0.344** — 2.2× better. The claim is not refuted,
but as implemented by the plain LDL head it is only weakly supported, and §6.3's framing was
too strong.

Note this is a *different* property from interval calibration: after conformal rescaling the
aleatoric channel produces excellent interval widths while still barely ranking which faces
humans argue about. Sharp intervals and good disagreement-detection are separate jobs and
should be measured separately.

## Decisions taken

1. **Never expose a raw model uncertainty as confidence.** Measured across five methods, all
   miscalibrated, two of them by more than 10×.
2. **Ship conformal prediction** on top of the `combined` channel — nominal coverage, tightest
   intervals, and it inherits the ensemble's shift-sensitivity.
3. **Report intervals as in-domain-only, and gate them on an OOD signal.** A 90 % interval
   becomes a 43 % interval off-distribution. The UI must either widen intervals or drop the
   numeric confidence entirely when the OOD detector fires. This makes §11.1's OOD channel a
   requirement, not an optional extra.
4. **Maintain a per-collection calibration set.** Because calibration is domain-specific, the
   honest implementation calibrates on a sample from the *user's own* indexed directory rather
   than shipping SCUT-derived constants.
5. **Use the hybrid head, not the LDL head, for the "polarising face" signal** (r = 0.344 vs
   0.139). The two heads can share a backbone, so this costs one extra linear layer.
6. **Drop TTA.** 3× wider intervals for the same coverage, plus a second forward pass. Keep
   it only for off-the-shelf models that cannot be ensembled (e.g. MiVOLO).

## Threats to validity

- One split (cv1) and one shifted domain (MEBeauty).
- Linear heads: a deep ensemble of linear models is unusually tight, which likely exaggerates
  how badly `ensemble` under-covers. The conclusion that raw spread needs recalibration is
  robust; the specific 11.5× factor is not.
- The MEBeauty rank-mapping makes absolute intervals comparable across incommensurable scales,
  but it is an approximation and imports MEBeauty's own score distribution.
- Conformal coverage is marginal, not conditional: 90 % overall does not mean 90 % for every
  subgroup. Per-group conditional coverage belongs in E11.

## Reproduce

```bash
python scripts/extract_features.py --dataset scut --encoder clip --margin 0.25 --flip
python scripts/run_e12_uncertainty.py
```
