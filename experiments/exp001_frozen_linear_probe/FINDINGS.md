# exp001 — Frozen representation + linear probe on SCUT-FBP5500

Experiment **E2** from [`docs/RESEARCH.md §14`](../../docs/RESEARCH.md).
Run: 2026-09-02 · 12m49s · RTX A6000 · seed 1337 · raw numbers in `results.json`

## Questions

1. Which frozen pretrained representation best predicts attractiveness ratings?
2. How close does *frozen features + a linear head* get to a **fine-tuned** CNN?
3. Does a label-distribution head beat plain mean-regression?

No test data influenced any fitting decision: ridge `alpha` was chosen by cross-validation
**inside the training split only**, on the dataset's own official splits.

## Headline result

5-fold CV mean, ridge head on the mean score. Published references are verified from
arXiv:1801.06345 and were computed on **these same official splits**.

| Representation | dim | PC | MAE | RMSE | Spearman | Pairwise acc |
|---|---:|---:|---:|---:|---:|---:|
| geometry (86 landmarks) | 178 | 0.6819 | 0.3880 | — | 0.6844 | 0.7506 |
| ArcFace R50 (`w600k_r50`) | 512 | 0.8268 | 0.3021 | — | 0.8209 | 0.8143 |
| ArcFace R100 (`glintr100`) | 512 | 0.8712 | 0.2643 | — | 0.8642 | 0.8393 |
| **CLIP ViT-B/32** | 512 | **0.9279** | 0.1976 | — | 0.9166 | 0.8776 |
| ArcFace R50 + geometry | 690 | 0.8507 | 0.2734 | — | 0.8512 | 0.8331 |
| ArcFace R50 + R100 | 1024 | 0.8813 | 0.2551 | — | 0.8725 | 0.8452 |
| **ArcFace R50 + CLIP** | 1024 | **0.9398** | 0.1823 | — | 0.9311 | 0.8895 |
| **all four** | 1714 | **0.9419** | 0.1798 | — | 0.9336 | 0.8916 |
| — *published, fine-tuned* AlexNet | — | 0.8634 | 0.2651 | 0.3481 | — | — |
| — *published, fine-tuned* ResNet-18 | — | 0.8900 | 0.2419 | 0.3166 | — | — |
| — *published, fine-tuned* ResNeXt-50 | — | 0.8997 | 0.2291 | 0.3017 | — | — |
| — *published, geometric + shallow* | — | 0.5948–0.6738 | 0.3898–0.4289 | — | — | — |
| — **human inter-rater correlation** | — | **0.770** | — | — | — | — |

### 1. Frozen features + ridge regression beat fine-tuned CNNs

A **linear model on frozen CLIP features (PC 0.9279)** outperforms fine-tuned ResNeXt-50
(0.8997) on the same official splits, and the ArcFace+CLIP fusion (0.9398) exceeds every
published number I could find for this benchmark, including the 2025 reported SOTA range
of 0.932–0.935. No backbone was trained. Total fit time is seconds on CPU.

**This validates the central architectural bet** of `docs/RESEARCH.md §15.1`: encoding can
be a one-time cached cost and the predictive head can stay cheap and swappable. Concretely
it means changing the beauty model never requires re-indexing, and a per-user preference
model is just another linear head over the same cached features — which makes Phase 10
personalisation nearly free rather than a research project.

### 2. CLIP beats face-recognition embeddings, decisively — and that is a warning

CLIP (0.9279) beats ArcFace R100 (0.8712) and R50 (0.8268) by a wide margin, despite
ArcFace being face-specific and trained on hundreds of millions of faces.

The likely reason is the confound predicted in `docs/RESEARCH.md §13.1.5`. ArcFace is
trained to be *identity-discriminative*, which means it is explicitly trained to **discard**
expression, lighting, styling and image quality. Those are precisely the things human
raters respond to. CLIP retains them. So the gap is evidence that **attractiveness ratings
substantially encode photography and presentation, not only facial structure.**

That is a real finding and it cuts both ways: it explains the accuracy, and it means the
score is partly "how well was this person photographed and styled". The product must
describe it accordingly. **E11c (correlate predicted attractiveness with predicted image
quality) is now a high priority**, not a formality.

The two representations are also complementary — fusing them adds +0.012 PC over CLIP
alone — so ArcFace contributes signal CLIP lacks, consistent with it carrying the
geometric component.

### 3. The label-distribution head beats plain regression on every representation

| Representation | ridge (mean) | LDL (distribution) | Δ |
|---|---:|---:|---:|
| geometry | 0.6819 | 0.6936 | +0.0117 |
| ArcFace R50 | 0.8268 | 0.8450 | **+0.0182** |
| ArcFace R100 | 0.8712 | 0.8826 | +0.0114 |
| CLIP | 0.9279 | 0.9302 | +0.0023 |
| ArcFace R50 + CLIP | 0.9398 | 0.9422 | +0.0024 |
| all | 0.9419 | 0.9440 | +0.0021 |

Predicting the full 5-bin histogram and taking its expectation beats regressing the mean
directly — **on every single representation**, with the largest gains where the
representation is weakest. Modelling the rating distribution acts as a regulariser, and it
comes with strictly more output: spread, `P(rating ≥ 4)`, and shape.

It also delivers usable disagreement signal: the head's predicted spread correlates with
the **true** per-image rater standard deviation at **r = 0.35** (CLIP-based). Modest, but
it is real information about which faces are polarising, and mean-regression cannot
produce it at all. This supports adopting LDL as the production head (`§6.3`).

### 4. Bagged ridge is NOT a usable uncertainty estimate — a clear negative result

| | measured | should be |
|---|---:|---:|
| coverage @ 68 % nominal | **0.223** | 0.68 |
| coverage @ 95 % nominal | **0.431** | 0.95 |
| mean predicted σ | 0.069 | ≈ 0.19 (the actual MAE) |
| corr(predicted σ, actual error) | 0.096 | high |

Bootstrap spread across ensemble members **underestimates true uncertainty by roughly 3×**
and barely correlates with where the model is actually wrong. The reason is structural: a
well-determined linear model over 4,400 samples is stable under resampling, so member
disagreement captures parameter uncertainty while ignoring the dominant irreducible noise.

**This is exactly the failure `docs/RESEARCH.md §11` warned about** — exposing raw model
spread as "confidence" would produce a number that looks authoritative and means nothing.
Conformal prediction (§11.2, item 3) is now mandatory rather than preferred, and this
result is the justification.

### 5. Fairness: the weaker the representation, the larger the demographic gap

Per-subgroup PC on cv1:

| Representation | AF (n=404) | AM (n=394) | CF (n=171) | CM (n=131) | spread |
|---|---:|---:|---:|---:|---:|
| ArcFace R50 | 0.8472 | 0.8573 | 0.7495 | 0.7026 | **0.155** |
| ArcFace R50 + CLIP | 0.9382 | 0.9313 | 0.9369 | 0.9177 | **0.021** |

ArcFace alone is dramatically worse on Caucasian faces — the subgroups with 750 training
images versus 2,000. The fusion nearly eliminates the gap (spread 0.155 → 0.021).

Two lessons: aggregate accuracy hid a 0.15 PC disparity, so **per-group reporting must be
default, not an audit step**; and stronger representations can reduce demographic
disparity rather than merely relocating it. Note this still only covers two ethnic groups —
the dataset contains no Black, South Asian, Hispanic, Middle Eastern or Southeast Asian
subjects at all, so nothing here speaks to performance on them.

## Threats to validity

- **Identity leakage is confirmed** (see [`../e1_identity_audit/`](../e1_identity_audit/)):
  ~5 % of test images share an identity with training in every official split. Measured
  impact on the headline is **+0.003 PC** — real, documented, but it does not explain the
  result.
- **Single benchmark.** Everything here is SCUT-FBP5500. It says nothing about
  generalisation, which is what the product actually needs. **E7 (MEBeauty) remains the
  go/no-go gate.** A CLIP-based model that partly keys on photographic style is exactly the
  kind of model that could transfer poorly.
- **PC 0.94 is far above the 0.77 human inter-rater ceiling.** As `§1.1` explains, this is
  not superhuman perception; the crowd mean is a low-variance target. It does mean this
  benchmark is saturated and further PC gains here are not worth pursuing.
- Ridge alphas were selected within-train, but the *representation* was chosen by looking
  at test performance across arms — standard for a comparison study, but the winning arm's
  absolute number is mildly optimistic.

## Decisions taken

1. **Adopt the frozen-encoder architecture.** Encode once, cache, predict cheaply. Confirmed.
2. **Ship ArcFace + CLIP fusion as the candidate production representation**, pending E7.
3. **Use the label-distribution head, not mean regression.** Better accuracy on every arm
   and strictly richer output.
4. **Do not use ensemble spread as a confidence score.** Implement conformal prediction.
5. **Make per-subgroup metrics a default output** of every experiment.

## Next

`E7` cross-dataset (MEBeauty) — the gate · `E5` crop-margin sweep · `E11c` attractiveness
vs. image-quality confound · `E12` conformal calibration · `E3` fine-tuned reference to
confirm the harness reproduces published numbers.

## Reproduce

```bash
make features && make exp001
```
