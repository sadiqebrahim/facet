# Dataset register

Per-dataset cards covering everything Phase 3 of the brief asks for.
**[V]** = verified against the primary source. **[S]** = secondary/reported, needs verification.

Status: `HELD` = on this machine · `ACQUIRE` = wanted · `AVOID` = do not use.

---

## SCUT-FBP5500 — `HELD` — primary beauty training set

| Field | Value |
|---|---|
| Path | `data/raw/SCUT-FBP5500_v2/` |
| Images | 5,500 [V] |
| Subjects | **unknown — assumed ≈5,500 but NOT verified. See E1.** |
| Composition | Asian F 2,000 · Asian M 2,000 · Caucasian F 750 · Caucasian M 750 [V] |
| Demographics | Asian + Caucasian only. **No Black, South Asian, Hispanic, Middle Eastern, SE Asian subjects.** [V] |
| Age range | 15–60 [V] |
| Gender labels | implicit in filename prefix (AF/AM/CF/CM) [V] |
| Attractiveness labels | mean score ∈ [1,5]; **plus all 60 individual ratings** (`All_Ratings.xlsx`) [V] |
| Raters | 60 volunteers, aged 18–27, mean 21.6 [V] |
| Rating method | web GUI, crowdsourced, 4 subsets rated separately, random order; ~10 % of faces re-shown, re-rated if the two ratings correlated < 0.7 [V] |
| Per-image rating σ | mostly [0.6, 0.7] on the 1–5 scale [V] |
| Inter-rater correlation | 0.770–0.785 overall; lowest for Caucasian male (0.743) [V] |
| Landmarks | 86 manual points per face, `.pts` binary (int32 count + 172 float32) [V] |
| Image conditions | frontal, unoccluded, neutral expression [V] |
| Splits supplied | official 5-fold CV (4400/1100) and 60/40 (3300/2200) [V] |
| Provenance | Internet; some Asian faces from DataTang and GuangZhouXiangSu; some Caucasian faces from the **10k US Adults Faces Database** [V] |
| Identity overlap | **unknown**; the 10k US Adults source is widely redistributed → plausible overlap with other corpora. **E1** |
| License | **non-commercial research only** [V] |
| Redistribution | prohibited |
| Citation | Liang, Lin, Jin, Xie, Li — ICPR 2018 |

**Biases:** narrow rater pool (one university, ~5-year age band, 2017); own-group agreement bias;
two ethnic groups only; constrained frontal imagery; ratings confounded with makeup, styling and
photo quality; score distribution concentrated mid-range (two-component Gaussian mixture [V]).

**Verdict:** train on it, but treat it as *"how 60 Chinese undergraduates rated 5,500 constrained
frontal photos in 2017"*, which is what it is. Never the sole evaluation.

---

## MEBeauty — `HELD` — cross-dataset generalisation test set

Acquired 2026-09-03 from the authors' repository (`github.com/fbplab/MEBeauty-database`).
All values below are [V], measured from the release itself.

| Field | Value |
|---|---|
| Path | `data/raw/MEBeauty/` |
| Images on disk | 2,544 originals; **2,520 with usable scores** [V] |
| Demographics | 6 ethnic groups: caucasian 986, asian 348, indian 299, hispanic 297, black 296, mideastern 294 [V] |
| Gender | roughly balanced within each group [V] |
| Rating scale | **1–10** [V] — **not comparable to SCUT-FBP5500's 1–5** |
| Raters | **360 raters, 61,404 individual ratings**, median 25 per image [V] |
| Per-rater scores | **supplied** (`scores/generic_scores_all_2022.xlsx`, 2,607 × 363) [V] |
| Mean score | 6.10; range 1.00–9.63 [V] |
| Mean per-image rater σ | 1.87 on the 1–10 scale (≈0.83 rescaled to 1–5, vs SCUT's 0.64) [V] |
| **Human ceiling** | split-half rater ρ **0.7736**; Spearman–Brown full-pool **0.8724** [V, measured in E7] |
| Conditions | unconstrained pose, expression, lighting, background [V] |
| Splits | official train/val/test supplied (1,990 / 530 after filtering to available images) [V] |
| Provenance | Unsplash, Pixabay, Pexels [V] |
| License | **non-commercial research only** [V, from the release README] |
| Contact | irina.val.lebedeva@gmail.com |

**Overlap with SCUT-FBP5500:** four of its six ethnic groups (black, hispanic, indian,
mideastern — **1,186 images, 47 %**) have *zero* representation in SCUT-FBP5500. This makes
it a direct test of out-of-distribution behaviour, not merely a domain-shift test.

**Verdict:** this turned out to be the most informative dataset in the project. E7 showed
that (a) ranking transfers from SCUT at ρ 0.608, (b) training on MEBeauty transfers to SCUT
*better* (ρ 0.742) despite being less than half the size, and (c) cross-group calibration
does not transfer at all. **Recommendation: promote MEBeauty from evaluation set to primary
training set**, and demote SCUT-FBP5500 to secondary evaluation. Compare across the two by
**rank correlation only**.

---

## FairFace — `ACQUIRE` — fairness evaluation + the commercially-clean option

| Field | Value |
|---|---|
| Images | ~108,501, from YFCC-100M (Flickr, CC-licensed) [V] |
| Labels | race (7 groups), gender, age (9 buckets) [V] |
| Design | deliberately balanced across race [V] |
| License | **CC BY 4.0 — commercial use permitted, attribution required** [V] |

**Verdict:** acquire. Two roles: (1) the backbone of the E11 fairness audit — it is balanced by
construction, which is exactly what a bias audit needs; (2) the only permissively-licensed face
attribute dataset available, so it anchors any commercial track (§12.2 Path B).
Labels are annotator-perceived race/gender — a limitation to state, not to ignore.

---

## Other beauty datasets

| Dataset | Size | Notes | Verdict |
|---|---|---|---|
| SCUT-FBP (original) | 500 Asian female | 5-point scale; predecessor | `ACQUIRE` — third generalisation probe |
| HotOrNot | ~2k in-the-wild | used by ComboLoss; availability/terms unclear | investigate |
| CelebA `Attractive` attr | 202k | **binary, very noisy, single-perspective**; non-commercial [V] | `AVOID` for evaluation; weak pretraining only |
| PFBP-SCUT500 / -SCUT5500 / -US10K | — | MetaFBP's personalisation benchmarks, built from the above | `ACQUIRE` for E14 |

---

## Age / gender datasets

| Dataset | Size | Labels | License | Verdict |
|---|---|---|---|---|
| **FairFace** | 108k | age bucket, gender, race | **CC BY 4.0** [V] | primary eval |
| UTKFace | ~20k | exact age, gender, race | non-commercial [V] | secondary eval |
| IMDB-Clean | ~200k | age | IMDb non-commercial [V] | ⚠ **MiVOLO trained on it — never evaluate MiVOLO here** |
| MORPH-II | 55k academic | age, longitudinal | academic free / commercial paid (UNCW) [V] | skip unless needed |
| AgeDB | 16k | age | research | secondary |
| APPA-REAL | 7.6k | **apparent age + per-rater spread** | research | valuable — its σ matches our uncertainty framing |
| LAGENDA | — | age, gender | released with MiVOLO [V] | eval only (MiVOLO trained on it) |

---

## Detection / landmark / quality datasets

| Dataset | Purpose | License | Verdict |
|---|---|---|---|
| WIDER FACE | detection benchmark | CC BY-NC-ND 4.0 [V] | evaluation only; no derivatives |
| 300W, WFLW, COFW | landmarks | research | only if we train a landmarker — we should not |
| LFW / CFP-FP / CPLFW / CALFW / AgeDB-30 | recognition verification | research | sanity-check the embedding model |
| XQLFW | low-quality recognition | research | FIQA validation (E9) |
| AVA | *image* aesthetics, ~250k images, ~210 ratings each [S] | unclear | reference for rating-distribution methodology |

---

## Rules for combining datasets

1. **Never pool attractiveness scores across datasets.** The scales measure different constructs
   with different instruments (1–5 from 60 Chinese undergraduates vs. 1–10 from a multi-ethnic
   pool vs. a binary flag).
2. **Pool comparisons, not scores.** Within-dataset pairwise preferences are scale-free and *can*
   be pooled (Bradley–Terry). This is the defensible route to a larger training signal.
3. **Always report cross-dataset results by rank correlation.**
4. **Verify subject-disjointness before every split**, and across datasets before combining.
5. **Record dataset version and split methodology in every run manifest.** A metric without its
   split methodology is not a result.
