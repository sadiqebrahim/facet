# E5 — Crop and alignment sensitivity

How much does the crop protocol matter, and what should be frozen into the feature store?

Run: 2026-09-03 · RTX A6000 · seed 1337 · raw numbers in `results.json`

## Setup

Crop margin (0 / 10 / 25 / 40 %) × alignment (ArcFace 5-point similarity transform vs. a
plain square bbox crop) × three representations. Two numbers per configuration, because E7
established they can disagree:

- **in-benchmark** — SCUT-FBP5500 official 5-fold Pearson correlation
- **transfer** — train on all 5,500 SCUT, test on all 2,520 held-out MEBeauty (Spearman)

## Results

| encoder | crop | 5-fold PC | 5-fold MAE | transfer ρ | transfer pairwise |
|---|---|---:|---:|---:|---:|
| arcface_r50 | m0.00 template | 0.8268 | 0.3021 | 0.2030 | 0.5673 |
| arcface_r50 | m0.10 template | 0.8337 | 0.2970 | 0.2080 | 0.5698 |
| arcface_r50 | m0.25 template | 0.8515 | 0.2810 | 0.2638 | 0.5886 |
| arcface_r50 | **m0.40 template** | **0.8641** | **0.2706** | **0.3272** | 0.6104 |
| arcface_r50 | m0.25 bbox (no align) | 0.8537 | 0.2805 | **0.3364** | 0.6138 |
| clip | m0.00 template | 0.9279 | 0.1976 | 0.5938 | 0.7096 |
| clip | m0.10 template | 0.9280 | 0.1968 | 0.6081 | 0.7152 |
| clip | **m0.25 template** | 0.9274 | 0.1979 | **0.6240** | 0.7225 |
| clip | m0.40 template | 0.9255 | 0.2002 | 0.6223 | 0.7225 |
| clip | m0.25 bbox (no align) | 0.9248 | 0.2008 | 0.6156 | 0.7197 |
| **arcface_r50+clip** | m0.00 template | **0.9398** | **0.1823** | 0.6084 | 0.7159 |
| arcface_r50+clip | m0.10 template | 0.9389 | 0.1834 | 0.6215 | 0.7202 |
| **arcface_r50+clip** | **m0.25 template** | 0.9390 | 0.1827 | **0.6418** | **0.7293** |
| arcface_r50+clip | m0.40 template | 0.9357 | 0.1879 | 0.6296 | 0.7243 |
| arcface_r50+clip | m0.25 bbox (no align) | 0.9353 | 0.1887 | 0.6184 | 0.7205 |

## Result 1 — my prior was wrong: the backbone matters far more than the crop

`docs/RESEARCH.md §2.2` recorded the expectation that crop margin "may matter more than the
choice of backbone", and noted that would be a cheap and important finding. It is cheap. It
is also **false**:

| Source of variation | PC range | transfer ρ range |
|---|---:|---:|
| crop margin (arcface_r50) | 0.0374 | 0.1242 |
| crop margin (clip) | 0.0024 | 0.0302 |
| crop margin (fusion) | 0.0041 | 0.0334 |
| **backbone choice** | **0.0756** | **0.3053** |

Choosing the representation is worth roughly **2× more in-benchmark and 2.5–10× more in
transfer** than choosing the crop margin. The prior is retired; §2.2 has been corrected.

## Result 2 — but crop margin matters enormously *for ArcFace specifically*

ArcFace transfer improves **0.2030 → 0.3272** from margin 0 to 0.40 — a **61 % relative
gain** — while CLIP moves only 0.594 → 0.622.

The mechanism is straightforward and consistent with exp001. ArcFace's canonical 112×112
crop is deliberately *tight*: it is framed for identity, and crops away hair, jawline, ears
and head shape. Those are exactly the features human raters use for attractiveness. Giving
ArcFace more context restores information its own preprocessing was designed to discard.
CLIP already sees a broader, semantic view and is correspondingly insensitive.

This is the same story exp001 told (ArcFace is trained to throw away what raters respond to),
observed through a different lens.

## Result 3 — in-benchmark and transfer optima disagree, and transfer should win

For the production fusion:

| | best config | value |
|---|---|---:|
| in-benchmark PC | m0.00 | 0.9398 |
| transfer ρ | **m0.25** | **0.6418** |

Optimising on SCUT-FBP5500 alone would have selected **m0.00** and given up **0.033 transfer
Spearman** (≈5 % relative) — the number that actually predicts whether the product works on a
user's directory. The in-benchmark cost of choosing m0.25 instead is 0.0008 PC, i.e. nothing.

This is a concrete vindication of reporting both numbers, and it is the reason the crop
protocol is being changed rather than left at its accidental default.

**Note this improves E7's headline.** E7 ran at the then-default m0.00 and reported transfer
ρ = 0.6084. At m0.25 the same model reaches **0.6418**. E7's conclusions are unchanged in
kind — ranking transfers, calibration does not — but its central number was pessimistic by
about 5 %.

## Result 4 — alignment is worth less than expected, and is actively harmful for ArcFace transfer

Template alignment vs. plain bbox crop, both at margin 0.25:

| encoder | PC (bbox → template) | transfer ρ (bbox → template) |
|---|---|---|
| arcface_r50 | 0.8537 → 0.8515 (**−0.0022**) | 0.3364 → 0.2638 (**−0.0727**) |
| clip | 0.9248 → 0.9274 (+0.0026) | 0.6156 → 0.6240 (+0.0084) |
| **arcface_r50+clip** | 0.9353 → 0.9390 (**+0.0037**) | 0.6184 → 0.6418 (**+0.0233**) |

For **ArcFace alone, a plain bbox crop transfers substantially better than the canonical
5-point similarity transform** (+0.073 ρ). The likely cause is that MEBeauty is in-the-wild:
under real pose variation a similarity transform fitted to five keypoints introduces
distortion, whereas a bbox crop simply keeps whatever the detector found.

For the production **fusion**, alignment still helps (+0.023 transfer), so it stays — but the
effect is far smaller than the "alignment is essential" folklore implies, and it is not
uniformly positive. Worth revisiting if ArcFace is ever used alone.

## Decisions taken

1. **Change the production crop margin from 0.00 to 0.25.** Free: +0.033 transfer Spearman,
   −0.0008 in-benchmark PC. `configs/pipeline/align.yaml` bumped to `version: v2`, which
   correctly invalidates the feature cache.
2. **Keep template alignment** for the fusion (+0.023 transfer), while recording that it is a
   small effect and harmful for ArcFace alone.
3. **Retire the §2.2 prior.** Backbone choice dominates crop protocol; effort belongs in
   representation selection.
4. **Select crop protocol on transfer, not in-benchmark accuracy.** The two optima differ.
5. **Re-run E7 at m0.25** when convenient; its reported transfer numbers are ~5 % pessimistic.

## Engineering note

Detection results are now cached per dataset (`artifacts/detections/*.npz`). Detection is
~7 ms/image and does not depend on the crop protocol, so caching it turned this sweep from
quadratic into near-free: re-extraction dropped from **15.1 ms/image to 2.5 ms/image (~6×)**.
The pipeline needs this cache anyway for incremental indexing (Phase 8).

The refactor was validated by re-running exp001 on the rebuilt features: cv1 PC reproduced
at **0.8203**, exactly matching the recorded value.

## Threats to validity

- One target dataset for transfer (MEBeauty), as in E7.
- Margins beyond 0.40 were not tested; ArcFace was still improving at the top of the range,
  so its optimum may lie further out. Worth extending for an ArcFace-only configuration.
- Only 112×112 input was tested. CLIP internally resizes to 224, so its effective resolution
  is fixed by its own preprocessing; a native-224 pathway was not evaluated.
- Alignment was tested only at margin 0.25.

## Reproduce

```bash
for m in 0 0.1 0.25 0.4; do
  for ds in scut mebeauty; do
    python scripts/extract_features.py --dataset $ds --encoder arcface_buffalo_l --margin $m
    python scripts/extract_features.py --dataset $ds --encoder clip --margin $m
  done
done
python scripts/run_e5_crop_sensitivity.py
```
