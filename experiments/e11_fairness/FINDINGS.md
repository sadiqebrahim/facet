# E11 — End-to-end fairness audit, and the attractiveness/quality confound

`docs/RESEARCH.md §13.3` argued that per-component bias **compounds** through the pipeline. So
this audits the whole chain on one balanced set rather than each component separately:
**FairFace validation, 10,954 images, balanced across seven perceived-race groups, CC BY 4.0.**

Run: 2026-09-03 · RTX A6000 · seed 1337 · raw numbers in `results.json`

FairFace's labels are annotator-*perceived* race, gender and age bucket. They are used here
only to slice our own error rates — never as prediction targets, never as a user-facing facet.

## Summary: bias does not compound — it enters late and it is large

| Stage | Disparity | Verdict |
|---|---|---|
| (a) detection | recall spread **0.0006** | ✅ clean |
| (b) gender | accuracy spread **0.101** by race, **0.174** by race × gender | ⚠ substantial |
| (c) age | MAE spread **2.79 years** by race, **13.9 years** by age bucket | ⚠ substantial |
| (d) attractiveness | percentile spread **0.240**; top-100 skew **2.2× / 4.3×** | 🔴 severe |
| (e) photo-quality confound | R² = **0.017** | ✅ hypothesis not supported |

The §13.3 compounding story is **half right**. There is no detector bias to compound *from* —
recall is essentially perfect and uniform. But disparity grows sharply through the attribute
stages and is worst exactly where the research report predicted it would be worst: the
attractiveness model, trained on two ethnic groups, applied to seven.

## (a) Detection — no measurable bias

| race | n | recall | mean det score |
|---|---:|---:|---:|
| East Asian | 1550 | 1.0000 | 0.755 |
| Indian | 1516 | 1.0000 | 0.758 |
| Latino_Hispanic | 1623 | 1.0000 | 0.771 |
| Middle Eastern | 1209 | 1.0000 | 0.759 |
| Southeast Asian | 1415 | 1.0000 | 0.759 |
| White | 2085 | 0.9995 | 0.765 |
| Black | 1556 | 0.9994 | 0.764 |

Overall recall **0.99982**, spread **0.00064**, and detector confidence is flat across groups
(0.755–0.771). SCRFD with E5's padding shows **no demographic recall disparity here.**

⚠ **This is a weaker result than it looks.** FairFace images are already face-centred crops,
so this measures detection on an easy, well-framed setting. It does *not* test the case that
matters for a directory scanner: small, occluded, profile or poorly-lit faces in cluttered
scenes, which is where `§13.2`'s "a missed face is an invisible failure" concern lives. That
requires WIDER FACE-style evaluation (E8) and remains open.

## (b) Gender — 0.174 accuracy spread across race × gender

Overall accuracy **0.7805**.

| race | n | accuracy |
|---|---:|---:|
| Middle Eastern | 1209 | 0.8304 |
| Latino_Hispanic | 1623 | 0.8182 |
| White | 2085 | 0.8019 |
| Southeast Asian | 1415 | 0.7654 |
| East Asian | 1550 | 0.7594 |
| Indian | 1516 | 0.7592 |
| **Black** | 1556 | **0.7294** |

Best group **Middle Eastern/Male 0.857**; worst **East Asian/Male 0.683** — a **0.174** gap.

The directional finding (worst on Black subjects) is consistent with *Gender Shades*. The
overall 78 % is far below the 97–99 % usually quoted for gender classification, so I checked
the setup rather than reporting a model as biased on a possible harness error: the label
mapping is right (the inverse mapping scores 0.21), and margin 0.25 is the *better* crop for
this model (0.792 vs 0.749 at margin 0.0). **The weakness is real**, and reflects the cheap
on-disk InsightFace `genderage` model meeting a genuinely diverse in-the-wild set. It
strengthens the §2.5 case for adopting MiVOLO rather than this baseline.

Product consequence: `§8`'s decision to make gender a **soft, confidence-gated preference**
rather than a hard filter is vindicated. A hard filter on a 78 %-accurate classifier would
silently discard roughly one in five true matches, disproportionately East Asian and Black
subjects.

## (c) Age — error varies far more by age than by race

Overall MAE **10.04 years** against FairFace bucket midpoints.

| race | MAE | | age bucket | n | MAE |
|---|---:|---|---|---:|---:|
| East Asian | 9.03 | | 3-9 | 1356 | **16.74** |
| Middle Eastern | 9.25 | | 70+ | 118 | **19.87** |
| White | 9.95 | | 60-69 | 321 | 17.45 |
| Southeast Asian | 10.01 | | 50-59 | 796 | 14.92 |
| Latino_Hispanic | 10.02 | | 10-19 | 1181 | 12.60 |
| Indian | 10.04 | | 0-2 | 199 | 12.42 |
| **Black** | **11.83** | | 40-49 | 1353 | 10.95 |
| | | | 30-39 | 2330 | 6.74 |
| | | | **20-29** | 3300 | **5.92** |

Race spread is 2.79 years; **age-bucket spread is 13.9 years** — five times larger. The model
collapses toward the young-adult mode: it is nearly 3× worse on children (16.74) and the
elderly (19.87) than on 20–29-year-olds (5.92).

Some of that MAE is an artefact of comparing point predictions to bucket midpoints (a true
9-year-old in the "3-9" bucket scores against a midpoint of 6), but that inflates every bucket
by a similar small amount and cannot explain a 14-year spread.

**Product consequence:** a single global "±N years" claim would be dishonest. `§11`'s
per-face interval must widen with predicted age, and the UI should not offer age filtering at
the extremes with the same apparent precision it offers for 20–35.

## (d) 🔴 Attractiveness — the severe finding

The SCUT-trained LDL head applied to the balanced FairFace set:

| race | n | mean score | mean percentile |
|---|---:|---:|---:|
| **Southeast Asian** | 1415 | 2.532 | **0.357** |
| East Asian | 1550 | 2.677 | 0.446 |
| Indian | 1516 | 2.712 | 0.473 |
| Latino_Hispanic | 1623 | 2.763 | 0.506 |
| Black | 1556 | 2.771 | 0.513 |
| Middle Eastern | 1209 | 2.878 | 0.579 |
| **White** | 2085 | 2.908 | **0.597** |

Percentile spread **0.240** across races, **0.278** across race × gender.

And what the user would actually see — the **top 100** of a balanced set, where every group
should contribute roughly its population share:

| race | in top-100 | expected | ratio |
|---|---:|---:|---:|
| **White** | **41** | ~19 | **2.2× over** |
| East Asian | 20 | ~14 | 1.4× over |
| Middle Eastern | 13 | ~11 | 1.2× over |
| Black | 8 | ~14 | 0.6× under |
| Latino_Hispanic | 8 | ~15 | 0.5× under |
| Indian | 7 | ~14 | 0.5× under |
| **Southeast Asian** | **3** | ~13 | **4.3× under** |

A model trained on 4,000 Asian and 1,500 Caucasian faces, rated by 60 Chinese undergraduates
in 2017, ranks **White faces into the top 100 at 2.2× their share and Southeast Asian faces at
0.23× theirs.** On a set that is balanced by construction.

This is not a subtle statistical artefact; it is the dominant behaviour of the system on
diverse input. It confirms `§13.1.3` — predictions on groups absent from the training data are
extrapolation — and it goes further: the extrapolation is not merely noisy, it is *directionally
biased*, and it survives into the only output the user ever sees.

Note East Asian is over-represented in the top-100 (20 vs 14) despite a below-average mean
percentile (0.446). The group distributions differ in **shape**, not just location, so a single
per-group offset correction would not fix this.

## (e) The photography confound — hypothesis NOT supported

exp001 and E5 both pointed here: CLIP beats ArcFace, and ArcFace's tight identity crop
transfers worst, which suggested attractiveness ratings might substantially encode
*photography* rather than faces. Tested directly:

| signal | Spearman with predicted attractiveness |
|---|---:|
| contrast | +0.089 |
| quality composite | +0.085 |
| blur (sharpness) | +0.074 |
| face pixel size | +0.055 |
| luminance | +0.050 |
| detector confidence | −0.018 |

**R² of attractiveness predicted from photography signals alone: 0.017.**

Technical image quality explains **1.7 %** of the variance. The confound hypothesis, as
stated, is **not supported** — and the direction is at least benign (sharper, better-exposed
images score slightly higher, which is what one would want).

⚠ **Be precise about what this rules out.** It tests *technical capture quality* — sharpness,
exposure, contrast, resolution. It does **not** test **styling**: makeup, hairstyle, grooming,
clothing, expression, professional vs. casual photography. Those are exactly what CLIP encodes
and ArcFace discards, and they are not measurable with a Laplacian. The styling hypothesis
remains open and needs a different probe (e.g. CLIP text-prompt similarity to styling
descriptors). §13.1.5 should be narrowed, not deleted.

## Decisions taken

1. **Ship the measured disparities with the product.** §13.4 promised bias would be published
   rather than buried; these numbers are that publication, and the top-100 table is the honest
   headline: this system over-selects White faces 2.2× and under-selects Southeast Asian faces
   4.3× on balanced input.
2. **Do not offer an unqualified global "best faces" ranking.** Already forced by E7 (0.215
   placement bias); E11 raises it to 0.240 on a balanced set with a 4.3× top-100 skew.
   `configs/query/scoring.yaml` keeps `cross_group_ranking.surface_measured_bias: true`.
3. **Adopt MiVOLO for age/gender.** The on-disk baseline is 78 % gender / 10 years MAE on
   diverse input — too weak to build on, quite apart from its disparities.
4. **Widen age intervals by predicted age bucket**, not globally: the age-bucket MAE spread
   (13.9 years) is 5× the race spread.
5. **Keep gender soft and confidence-gated** (§8). A hard filter here would silently drop ~1 in
   5 true matches, unevenly.
6. **Narrow §13.1.5, don't drop it.** Technical quality is not the confound; styling is untested.
7. **E8 (detection in cluttered scenes) is still required.** (a) tested detection on pre-cropped
   faces, which is the easy case.

## Threats to validity

- FairFace's labels are annotator-perceived; the groups are social constructs measured with
  error, and "race" is not a biological variable the model could be unbiased with respect to.
- Detection was measured on face-centred crops, so (a) is close to a best case.
- Age MAE against bucket midpoints carries a small irreducible penalty.
- The attractiveness head is one model (LDL over ArcFace+CLIP at m0.25) trained on one dataset;
  the *direction* of the skew is a property of SCUT-FBP5500's rater pool, and a different
  training pool would produce a different skew — not no skew.
- Gender is evaluated as a binary because the label set is binary. Non-binary presentation is
  unrepresented in the data and therefore unmeasured here.

## Reproduce

```bash
python scripts/download_fairface.py
python scripts/extract_features.py --dataset fairface --encoder arcface_buffalo_l --margin 0.25
python scripts/extract_features.py --dataset fairface --encoder clip --margin 0.25
python scripts/run_e11_fairness.py
```
