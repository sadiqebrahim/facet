# E6 — Which training objective should the beauty head use?

The core beauty experiment. Five objectives over identical frozen features (ArcFace R50 +
CLIP at the E5 production crop, margin 0.25), identical head architecture, optimiser,
schedule, seed and early-stopping criterion. **Only the loss differs**, so any difference is
attributable to the objective.

Run: 2026-09-03 · RTX A6000 · seed 1337 · raw numbers in `results.json`

## Objectives

| | Loss | Target built from |
|---|---|---|
| **regression** | MSE | mean rating |
| **ordinal** | CORAL-style cumulative logits, BCE | empirical P(rating > k) |
| **distribution** | softmax + KL | the real 60-rater histogram |
| **pairwise** | Bradley–Terry, BCE on score differences | empirical P(rater scores i above j) |
| **hybrid** | KL + λ·Bradley–Terry | both |

Two things make this a fair test rather than a hyperparameter race. Every arm uses the same
linear head and is early-stopped on the **same** criterion (validation Spearman), so no
objective gets a home-field advantage in model selection. And the ordinal and pairwise
targets come from the **actual per-rater ratings**, not from the mean — for pairwise, the
target is the real fraction of the 60 raters who scored face *i* above face *j*. Most
published pairwise FBP work has to synthesise those labels; SCUT-FBP5500 ships the raw
ratings, so we don't.

Scores are affine-calibrated on the training split before MAE, so the pairwise head — which
has no inherent scale — remains comparable. Rank metrics are unaffected by that.

## Results

| objective | PC | MAE | Spearman | pairwise | NDCG@100 | **transfer ρ** | KL to true hist | aleatoric r |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| regression | 0.9345 | 0.1904 | 0.9257 | 0.8819 | 0.9594 | 0.6261 ± 0.0070 | — | — |
| ordinal | 0.9349 | 0.1902 | 0.9259 | 0.8821 | 0.9588 | 0.6152 ± 0.0072 | 2.4817 | 0.238 |
| **distribution** | **0.9387** | **0.1830** | **0.9280** | **0.8848** | 0.9598 | 0.6213 ± 0.0061 | **0.3192** | 0.154 |
| pairwise | 0.9353 | 0.1889 | 0.9273 | 0.8834 | **0.9605** | **0.6262 ± 0.0088** | — | — |
| hybrid | 0.9335 | 0.1898 | 0.9255 | 0.8819 | 0.9568 | 0.6117 ± 0.0105 | 0.3666 | **0.344** |

## Result 1 — on accuracy, the objectives are indistinguishable

This is the headline, and it took a variance check to see it.

The first run used a single transfer seed and appeared to show regression winning at 0.6385
against distribution's 0.6103 — a 0.028 gap that would have been easy to write up as
"regression transfers best". Repeating the transfer arm over **five seeds** dissolves it:

| | best | worst | gap |
|---|---|---|---:|
| transfer ρ | pairwise 0.6262 | hybrid 0.6117 | **0.0145** |

with a pooled seed standard deviation of **0.0109**. The best-to-worst gap is **1.3 seed
standard deviations**, Welch *p* = 0.068 — not significant, and that is the gap between the
extremes of five objectives, so no pair is distinguishable.

In-benchmark the spread is likewise tiny: PC ranges 0.9335–0.9387, a span of 0.005, against
a backbone-choice span of 0.076 (E5).

**The loss function is not where the performance is.** Representation choice (E5: 0.076 PC /
0.305 ρ) and crop protocol for ArcFace (E5: 0.124 ρ) both dominate it by an order of
magnitude.

## Result 2 — this corrects an over-claim from exp001

exp001 reported that the label-distribution head beat mean-regression on **every**
representation, by +0.002 to +0.018 PC, and I described that as supporting LDL as the
production choice. That comparison used ridge heads on a single configuration with no
variance estimate.

Under matched training with proper seed variance, the direction survives in-benchmark
(distribution 0.9387 vs regression 0.9345, +0.004) but the effect is small, and **on transfer
it does not survive at all** (0.6213 vs 0.6261 — distribution is nominally *behind*, well
within noise). The honest statement is that **LDL and regression are equally accurate**, not
that LDL is more accurate.

## Result 3 — so choose on the auxiliary outputs, where the differences are enormous

Accuracy is a tie; what each objective *additionally* gives you is not:

| objective | usable rating distribution? | rater-disagreement estimate |
|---|---|---|
| regression | ✗ none | ✗ none |
| pairwise | ✗ none (and no absolute scale) | ✗ none |
| ordinal | ✗ **KL 2.48** — the differentiated cumulative is a poor pmf | r = 0.238 |
| **distribution** | ✅ **KL 0.319** | r = 0.154 |
| **hybrid** | ✅ KL 0.367 | ✅ **r = 0.344** |

The KL spread is nearly **8×** between ordinal and distribution. And regression and pairwise
produce no distribution at all, so they cannot support §11's aleatoric/epistemic split, the
"is this face polarising?" signal, or `P(rating ≥ 4)` — which §9.3 makes the ranking target.

The hybrid's disagreement estimate (r = 0.344) is more than double the pure distribution
head's (0.154), at a cost of 0.005 PC and ~0.010 transfer ρ — both inside noise. That is a
real trade and worth knowing.

## Result 4 — the "order learning transfers better" hypothesis is not confirmed

`docs/RESEARCH.md §6.4` argued, following UOL (arXiv:2409.00603), that ranking objectives
should transfer better across datasets because absolute score scales are incommensurable.

Measured: pairwise 0.6262 ± 0.0088 vs regression 0.6261 ± 0.0070. **Identical.**

That does not refute UOL — this is one dataset pair, a linear head over frozen features, and
scale-free evaluation by rank correlation, which already neutralises the scale-mismatch
problem that motivates order learning. But at this scale and in this setup, the predicted
advantage is not there, and §6.4's claim is downgraded accordingly.

## Decisions taken

1. **Use the `distribution` (LDL) head in production.** Not because it is more accurate — it
   is not, within noise — but because it is the only arm that is simultaneously top-of-pack
   in-benchmark and produces a well-formed rating distribution (KL 0.319, ~8× better than
   ordinal). §11 and §9.3 both need that distribution.
2. **Keep `hybrid` as the alternative** if rater-disagreement estimation becomes the priority:
   2.2× better aleatoric correlation for an accuracy cost inside seed noise.
3. **Reject `ordinal`.** No accuracy advantage and its implied distribution is bad (KL 2.48).
4. **Stop optimising the loss function.** The measured spread is ~1 seed-std. Effort belongs
   in representation and data (E5, E7), not in objective design.
5. **Report seed variance for every future single-run comparison.** A 0.028 single-seed gap
   here was entirely noise; without the repeat it would have become a documented "finding".

## Threats to validity

- Linear heads only. A higher-capacity head might separate the objectives — though it would
  also weaken the "only the loss differs" control.
- λ = 1.0 for the hybrid was not tuned; a sweep might improve it.
- Five transfer seeds gives a coarse variance estimate; the non-significance at *p* = 0.068 is
  suggestive rather than conclusive, and a larger repeat count would tighten it.
- One transfer target (MEBeauty), as in E7.
- Early stopping on validation Spearman is uniform and therefore fair, but it does mildly
  favour objectives whose training signal aligns with rank correlation.

## Reproduce

```bash
python scripts/run_e6_objectives.py
```
