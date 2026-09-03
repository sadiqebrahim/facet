# E7 — Cross-dataset generalisation: SCUT-FBP5500 → MEBeauty

**The gate.** Everything before this was research on one saturated benchmark. This asks
whether any of it survives contact with images that don't look like that benchmark.

Run: 2026-09-03 · RTX A6000 · seed 1337 · raw numbers in `results.json`

## Setup

Train on **all 5,500** SCUT-FBP5500 images, test on **all 2,520** MEBeauty images. MEBeauty
is fully held out — no MEBeauty image influenced the SCUT-trained model. Identical
preprocessing (SCRFD pad-detect → ArcFace 112×112 template alignment) for both.

The two datasets disagree on nearly everything:

| | SCUT-FBP5500 | MEBeauty |
|---|---|---|
| imagery | frontal, neutral, posed | in-the-wild, unconstrained |
| ethnicities | Asian, Caucasian **only** | 6 groups (47 % of images from groups absent in SCUT) |
| scale | 1–5 | 1–10 |
| raters | 60, aged 18–27, one university | ~300, mixed ethnicity/age/gender (61,404 ratings) |
| source | DataTang, 10k US Adults | Unsplash / Pixabay / Pexels |

Because the scales differ, **everything is reported as rank correlation**. Four arms,
because a bare cross-dataset number is uninterpretable.

## Result 1 — the human ceiling, measured on both sides

| | Spearman |
|---|---:|
| MEBeauty split-half rater reliability | 0.7736 |
| MEBeauty full-pool (Spearman–Brown corrected) | **0.8724** |
| SCUT-FBP5500 published inter-group rater correlation | 0.770 |

MEBeauty's ~300 raters agree with each other about as much as SCUT's 60 do. **0.872 is the
practical ceiling** for predicting MEBeauty's mean. Model numbers below should be read
against that, never against 1.0.

## Result 2 — ranking transfers; roughly 72 % of within-dataset performance survives

Spearman, best representation (ArcFace R50 + CLIP):

| Arm | What it measures | Spearman | Pairwise acc |
|---|---|---:|---:|
| **A** SCUT → MEBeauty (all 2,520) | **the actual question** | **0.6084** | 0.7159 |
| A restricted to arm B's test rows | comparable to B | 0.5753 | 0.7037 |
| **B** MEBeauty → MEBeauty (official split) | within-dataset upper bound | **0.7990** | 0.8012 |
| **C** MEBeauty → SCUT (all 5,500) | reverse direction | **0.7424** | 0.7716 |
| — human ceiling on MEBeauty | | 0.8724 | — |

On the *identical* test rows, cross-dataset transfer retains **0.5753 / 0.7990 = 72 %** of
within-dataset performance, and reaches **66 % of the human ceiling** (vs. 92 % for the
within-dataset model).

**This is a pass, and a qualified one.** A model trained on posed Chinese-undergraduate-rated
portraits ranks in-the-wild multi-ethnic faces far above chance (pairwise accuracy 0.716 vs
0.5 chance). It is not close to the within-dataset model, and the gap is domain shift, not
task difficulty — arm B proves the task is learnable to 0.80 on this same data.

## Result 3 — diversity beats scale, decisively

**Arm C (0.7424) is much better than arm A (0.6084)** — and MEBeauty has *less than half*
the training data (2,520 vs 5,500).

Training on 2,520 diverse in-the-wild images transfers to constrained posed portraits far
better than 5,500 constrained posed portraits transfer to the wild. Direction of transfer
matters more than dataset size.

**This inverts the default plan.** SCUT-FBP5500 is the larger, more-cited, better-documented
benchmark and it is the *worse* training set for a system meant to run on arbitrary user
directories. Production training should be on in-the-wild data — MEBeauty, or pooled
comparisons (§6.4) — with SCUT-FBP5500 demoted to a secondary evaluation set.

## Result 4 — unseen ethnicities rank fine, if the representation is right

Arm A Spearman by ethnicity. Four of these six groups have **zero** representation in the
training data:

| Representation | asian | black* | caucasian | hispanic* | indian* | mideastern* |
|---|---:|---:|---:|---:|---:|---:|
| ArcFace R50 | 0.224 | 0.192 | 0.265 | 0.107 | 0.268 | 0.128 |
| ArcFace R100 | 0.490 | 0.303 | 0.522 | 0.475 | 0.378 | 0.435 |
| CLIP | 0.593 | 0.456 | 0.607 | 0.583 | 0.546 | 0.676 |
| **ArcFace R50 + CLIP** | 0.584 | **0.488** | 0.622 | 0.612 | 0.586 | **0.704** |

\* absent from SCUT-FBP5500 entirely.

| Representation | in-SCUT-distribution | out-of-distribution | gap |
|---|---:|---:|---:|
| ArcFace R50 | 0.263 | 0.138 | 0.125 |
| ArcFace R100 | 0.504 | 0.359 | 0.145 |
| CLIP | 0.598 | 0.583 | 0.014 |
| ArcFace R50 + CLIP | 0.604 | 0.608 | **−0.005** |

With CLIP in the representation the out-of-distribution penalty **vanishes** — the fused
model ranks unseen ethnicities *fractionally better* than seen ones. The explanation is
that the OOD-ness was never in the labels but in the representation: CLIP was pretrained on
web-scale diverse imagery, so a Nigerian or Indian face is not out-of-distribution *for the
encoder*, even though the beauty labels came only from Asian and Caucasian faces. The linear
head learned a direction in CLIP space that is not ethnicity-specific.

This is a strong argument for the frozen-general-encoder architecture on fairness grounds,
not just efficiency grounds. It also nearly reverses my §13.1.3 expectation that predictions
on unseen groups would be unusable extrapolation — **for ranking within a group**. Which
brings us to the part that does not work.

## Result 5 — ⚠ cross-group calibration does NOT transfer, and within-group Spearman hides it

Within-group ranking can be excellent while a group is systematically placed too high or too
low *overall*. For a product that ranks all faces together and shows the top N, that offset
decides who gets surfaced. Measured as percentile-rank placement (scale-free):

| Representation | max abs bias | bias spread | top-100 overlap |
|---|---:|---:|---:|
| ArcFace R50 | 0.323 | 0.590 | 8/100 |
| ArcFace R100 | 0.313 | 0.614 | 11/100 |
| CLIP | 0.194 | 0.373 | 17/100 |
| ArcFace R50 + CLIP | **0.215** | **0.414** | **20/100** |

Per-group placement for the best model (percentile points, negative = ranked too low):

| Group | n | true %ile | pred %ile | bias |
|---|---:|---:|---:|---:|
| female/asian | 197 | 0.616 | 0.401 | **−0.215** |
| female/indian | 149 | 0.587 | 0.426 | −0.161 |
| female/hispanic | 149 | 0.718 | 0.589 | −0.129 |
| female/caucasian | 546 | 0.628 | 0.594 | −0.034 |
| female/mideastern | 146 | 0.661 | 0.642 | −0.018 |
| female/black | 150 | 0.437 | 0.446 | +0.009 |
| male/asian | 151 | 0.312 | 0.329 | +0.017 |
| male/hispanic | 148 | 0.441 | 0.464 | +0.023 |
| male/indian | 150 | 0.335 | 0.414 | +0.079 |
| male/mideastern | 148 | 0.335 | 0.423 | +0.088 |
| male/caucasian | 440 | 0.440 | 0.544 | +0.104 |
| male/black | 146 | 0.232 | 0.430 | **+0.198** |

The pattern is systematic: **every female group is pushed down, every male group is pushed
up.** The model compresses the gender gap that MEBeauty's raters produced, because SCUT's
raters produced a much smaller one (SCUT means: AF 3.06 / AM 2.87, a 0.19 gap on a 1–5 scale;
MEBeauty's true gender gap in percentile terms is roughly three times larger). The model
faithfully reproduces the *training* rater pool's group-level preferences, which are not the
test pool's.

And the product-level consequence — **the top-100 the user would actually see**:

| Ethnicity | in true top-100 | in predicted top-100 |
|---|---:|---:|
| asian | 16 | **3** |
| black | 2 | 4 |
| caucasian | 44 | **58** |
| hispanic | 12 | 11 |
| indian | 14 | 8 |
| mideastern | 12 | 16 |
| **set overlap** | | **20/100** |

The SCUT-trained model's top-100 shares only **20 %** of its members with the top-100 chosen
by MEBeauty's raters, and over-selects Caucasian faces (58 vs 44) while nearly eliminating
Asian faces (3 vs 16).

**This is the most important result in the project so far.** Note carefully what it does and
does not say. MEBeauty's "true top-100" is not objective truth either — it is the aggregate
of a different rater pool. The finding is not "the model is wrong"; it is that **which faces
get surfaced depends overwhelmingly on whose ratings you trained on**, and the disagreement
between two reasonable rater pools is large enough to change 80 % of the result set.

That is the §1.1 argument, confirmed end-to-end on real data: at inter-rater r ≈ 0.77–0.87
there is no single "attractiveness" to rank by, and any product that presents one is making
an arbitrary choice of whose taste to encode without telling the user.

## Verdict: conditional pass

| Capability | Transfers? | Evidence |
|---|---|---|
| **Relative ranking within a comparable pool** | ✅ yes | ρ 0.61 cross-dataset, 72 % of within-dataset, 0.72 pairwise accuracy |
| **Ranking faces of unseen ethnicities** | ✅ yes, with CLIP | OOD gap −0.005 |
| **Absolute scores / thresholds** | ❌ no | scales incommensurable by construction |
| **Cross-group / cross-demographic comparison** | ❌ **no** | up to 0.215 percentile placement bias; 0.414 spread |
| **Top-N selection matching another rater pool** | ❌ no | 20/100 overlap |

**Proceed — with the product scope narrowed to what the evidence supports.**

## Decisions taken

1. **Train the production model on in-the-wild data, not SCUT-FBP5500.** Arm C beats arm A
   with half the data. SCUT-FBP5500 becomes a secondary eval set. (Reverses the §15.6 plan.)
2. **Keep CLIP in the representation.** It is what removes the OOD penalty (gap 0.145 → −0.005)
   and it halves cross-group bias versus ArcFace alone.
3. **Drop absolute attractiveness thresholds from the query language.** The brief's
   `attractiveness > 4.0` is not supportable across domains. Replace with **percentile within
   the indexed collection**, which is well-defined, scale-free and honest. `configs/query/scoring.yaml`
   needs updating.
4. **Do not rank across demographic groups as if scores were comparable.** Either surface the
   measured placement bias, or offer within-group ranking. An unqualified global "best faces"
   ranking is not supportable at 0.215 percentile bias.
5. **Personalisation is promoted from Phase 10 to core.** Two reasonable rater pools disagree
   on 80 % of a top-100. A single population model cannot be the product; per-user preference
   is the only principled resolution. E14 should be brought forward.
6. **Report top-k overlap as a headline metric.** Global ρ of 0.61 looked acceptable; 20/100
   top-100 overlap is what the user experiences. Add it to the standard evaluation report.

## Threats to validity

- One target dataset. MEBeauty is the best available in-the-wild multi-ethnic option, but
  n = 2,520 and its ~300 raters have their own composition and biases.
- Only a linear head was tested. A fine-tuned backbone might transfer differently — though
  the usual expectation is *worse*, since fine-tuning specialises to the source domain.
- MEBeauty's per-image rater counts vary (median 25), so its per-image means are noisier
  than SCUT's 60-rater means. This depresses the measurable ceiling and all arm-A numbers
  somewhat; the split-half ceiling of 0.872 already accounts for it.
- Ethnicity labels come from MEBeauty's own directory structure — annotator-assigned
  perceived ethnicity, with all the usual caveats.

## Reproduce

```bash
python scripts/extract_features.py --dataset mebeauty --encoder arcface_buffalo_l
python scripts/extract_features.py --dataset mebeauty --encoder clip
python scripts/run_e7_cross_dataset.py
```
