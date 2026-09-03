# E4 — Off-the-shelf age and gender models

`docs/RESEARCH.md §2.5` recommends adopting MiVOLO rather than training an age model, on the
strength of published numbers and an Apache-2.0 licence. §1.2 says published numbers are not a
valid basis for selection. So this measures both candidates on one balanced set, per group.

Run: 2026-09-03 · RTX A6000 · FairFace validation (10,954 images, 7 balanced race groups)

## ⚠ Read this before the numbers

**MiVOLO v2's model card says it was trained on "proprietary and open-source datasets" without
enumerating them, so FairFace may be in its training data.** The InsightFace baseline certainly
is not. That asymmetry favours MiVOLO and cannot be resolved from outside the model. Treat the
gap below as an **upper bound** on MiVOLO's advantage, not a clean comparison.

Crop protocol is swept per model rather than fixed: the two were trained on different framings,
and feeding one model the other's preferred crop measures the crop, not the model.

## Results

| model | crop | gender acc | age MAE | gender spread (race) | age spread (race) | img/s |
|---|---|---:|---:|---:|---:|---:|
| **mivolo_v2** | **full** | **0.9609** | **5.14** | **0.0393** | **1.00** | 24.4 |
| mivolo_v2 | bbox | 0.9453 | 5.28 | 0.0285 | 0.57 | 30.2 |
| mivolo_v2 | align | 0.9432 | 5.34 | 0.0324 | 0.60 | 22.8 |
| insightface | bbox | 0.8088 | 10.99 | 0.1101 | 2.82 | 4628 |
| insightface | align | 0.7805 | 10.04 | 0.1010 | 2.79 | 4747 |

MiVOLO wins on every accuracy and fairness axis, decisively:

- **Gender: 0.9609 vs 0.8088** (+0.152)
- **Age MAE: 5.14 vs 10.99 years** — less than half the error
- **Gender disparity across races: 0.039 vs 0.110** — ~3× fairer
- **Age disparity across races: 1.00 vs 2.82 years** — ~3× fairer

## Per-group detail

Gender accuracy by race — MiVOLO is better for **every** group, and by the largest margin
exactly where the baseline was weakest:

| race | MiVOLO | InsightFace | Δ |
|---|---:|---:|---:|
| Indian | 0.9769 | 0.7784 | **+0.199** |
| **Black** | 0.9377 | **0.7526** | **+0.185** |
| Southeast Asian | 0.9597 | 0.7972 | +0.163 |
| East Asian | 0.9568 | 0.8077 | +0.149 |
| Latino_Hispanic | 0.9643 | 0.8250 | +0.139 |
| White | 0.9597 | 0.8379 | +0.122 |
| Middle Eastern | 0.9752 | 0.8627 | +0.113 |

E11 found the baseline's worst group was Black subjects at 0.753. MiVOLO lifts that to 0.938
and compresses the whole spread from 0.110 to 0.039. **The fairness improvement is larger than
the accuracy improvement**, which is the more important half.

Age MAE by bucket — E11's finding was that the baseline's error varied 5× more by age than by
race. MiVOLO largely fixes that:

| bucket | MiVOLO | InsightFace | Δ |
|---|---:|---:|---:|
| 0-2 | **2.68** | 14.84 | −12.16 |
| 3-9 | **2.98** | 19.70 | −16.72 |
| 10-19 | 5.80 | 15.36 | −9.56 |
| 20-29 | 4.97 | 9.01 | −4.03 |
| 30-39 | 5.48 | 7.07 | −1.58 |
| 40-49 | 6.25 | 9.09 | −2.84 |
| 50-59 | 5.72 | 11.05 | −5.33 |
| 60-69 | 5.97 | 11.66 | −5.69 |
| 70+ | 6.94 | 13.02 | −6.08 |

Bucket spread falls from **12.63 years to 4.26**. The baseline collapsed toward the young-adult
mode (19.7 years of error on children); MiVOLO is nearly flat across the lifespan.

## The cost: MiVOLO is ~190× slower

24.4 img/s against 4,628. That is a real constraint for a directory scanner: 100,000 images
would take ~70 minutes of age/gender inference alone, versus ~20 seconds for the baseline.

It is tolerable because it is a **one-time indexing cost** under the cached-encoder architecture
(§15.1) — but it makes MiVOLO the pipeline bottleneck by a wide margin, and it argues for
running it lazily (only on faces that survive quality and query filters) rather than eagerly on
every detected face.

## MiVOLO wants a *wider* crop than the rest of the pipeline

MiVOLO scored best on the **full image** (0.9609) rather than on our detector crops
(bbox 0.9453, aligned 0.9432). FairFace's 1.25-padding images include head and shoulders, and
MiVOLO v2 is a face+body model — even in face-only mode it benefits from that context, which
our tight ArcFace-framed crop removes.

**Pipeline consequence:** age/gender needs its own crop protocol, wider than the embedding
crop. One detection, two crops. This is the same lesson E5 and E8 taught in different form —
crop protocol is model-specific and must not be shared by default.

Note the tension: the *bbox* crop gives slightly better fairness (gender spread 0.0285 vs
0.0393, age spread 0.57 vs 1.00) at slightly worse accuracy. If disparity matters more than
mean accuracy, bbox is the better choice — and given §13, that is a defensible reading.

## Decisions taken

1. **Adopt MiVOLO v2 for age and gender**, confirming §2.5. Better on accuracy *and* fairness,
   and Apache-2.0.
2. **Give it a wide crop** (full frame or generous bbox), separate from the 112 px ArcFace crop.
3. **Run it lazily**, not on every detected face — at 190× the baseline cost it dominates
   indexing time.
4. **Retire the InsightFace `genderage` baseline.** 0.81 gender / 11 years MAE on diverse input
   is not shippable, and it carries a research-only licence besides.
5. **Update E11's conclusions**: the disparities E11 measured for age/gender were largely
   properties of the weak baseline, not irreducible. Detection remains clean, attractiveness
   remains severe, and age/gender moves from "⚠ substantial" to "acceptable, still non-zero".

## Licensing work this required

MiVOLO's repo depends on Ultralytics YOLOv8 (**AGPL-3.0**, network-viral) for its detector.
Only `yolo_detector.py` touches it, so this vendors just the model definition
(`mivolo_model.py`, `cross_bottleneck_attn.py`, Apache-2.0, attributed) under
`facet/third_party/mivolo/` and drives it with our SCRFD detector — the plan recorded in
`docs/LICENSING.md §1`. Four tests now enforce this, including an AST check that no source file
imports ultralytics.

Residual issue, unchanged: MiVOLO's weights are Apache-2.0 but its training data includes
IMDB-clean, which derives from IMDb under non-commercial terms. Apache-2.0 on weights does not
launder training-data provenance. Irrelevant for research use; a question for a lawyer before
any commercial use.

## Threats to validity

- **Possible FairFace contamination in MiVOLO's training data** (see the top). This is the
  dominant caveat.
- Age is compared against FairFace **bucket midpoints**, which adds an irreducible error floor
  of ~2 years and penalises both models roughly equally.
- Only two candidates. DEX, CORAL, MWR and DLDL-v2 were not run; §7 argued they are not worth
  the effort given MiVOLO's licence and performance, and this does not test that.
- MiVOLO is run in **face-only mode** — we detect faces, not people, so its body input is the
  zero-fill its own code uses. Published numbers use body crops and are correspondingly better.
- Speed measured on single-model sequential batches; both would improve with pipelining.

## Reproduce

```bash
python scripts/run_e4_age_gender.py
```
