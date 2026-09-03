# E9 — Is the "free" quality composite actually a quality signal?

`docs/RESEARCH.md §2.4` proposed *not* training a FIQA model, and instead building a composite
from signals we already compute — blur, exposure, contrast, face size, detector confidence,
embedding norm — then validating it. `src/facet/models/quality.py` was that composite, and its
reference constants were explicitly documented as guesses. This validates them, and finds
them wrong.

Run: 2026-09-03 · RTX A6000 · seed 1337 · `results.json`

## Method: the functional test, not a correlation

The obvious evaluation — correlate against CR-FIQA — would only measure agreement with another
estimator. The field's actual ground truth is functional: **a face quality metric is good if
rejecting low-quality faces reduces face-recognition error.** That is the Error-versus-Reject
Curve (ERC), and it needs identity labels, so this runs on **LFW's official 6,000-pair
verification protocol** (7,701 unique images).

Baseline verification with our own pipeline: **AUC 0.9894, FNMR 0.0240 at FMR 0.01.** Pairs are
progressively rejected by the quality of their *worse* face; the threshold stays fixed. Lower
AUERC is better; random rejection is the floor to beat.

## Result 1 — the composite worked, and was beaten by one of its own components

| signal | 0 % | 10 % | 20 % | 40 % | **AUERC** | vs random |
|---|---:|---:|---:|---:|---:|---:|
| **face_pixels** | 0.0240 | 0.0100 | 0.0074 | 0.0086 | **0.0096** | **+57.9 %** |
| composite (v1) | 0.0240 | 0.0139 | 0.0122 | 0.0128 | 0.0136 | +40.6 % |
| embedding_norm | 0.0240 | 0.0200 | 0.0162 | 0.0114 | 0.0158 | +30.8 % |
| det_score | 0.0240 | 0.0198 | 0.0189 | 0.0154 | 0.0184 | +19.4 % |
| blur | 0.0240 | 0.0197 | 0.0200 | 0.0183 | 0.0193 | +15.5 % |
| random | 0.0240 | 0.0244 | 0.0251 | 0.0215 | 0.0229 | — |
| **contrast** | 0.0240 | 0.0237 | 0.0237 | 0.0261 | **0.0248** | **−8.2 %** |

The §2.4 hypothesis is directionally right — the composite beats random by 41 %, and rejecting
20 % of pairs halves FNMR (0.0240 → 0.0122). **But raw face size alone beats it.** A composite
that is worse than one of its own inputs is not a composite, it is a dilution.

Two other things fall out. **`embedding_norm` genuinely works** (+30.8 % over random),
confirming §2.4's MagFace-style "free quality proxy" hypothesis. And **`contrast` is worse than
random** — actively harmful, yet it was being averaged in at equal weight.

## Result 2 — diagnosis: the composite was a blur detector

| | corr with v1 |
|---|---:|
| blur | **+0.914** |
| det_score | +0.091 |
| face_pixels | **+0.065** |
| embedding_norm | **−0.030** |

The composite was almost perfectly correlated with blur, and almost uncorrelated with the two
signals that actually work. The cause is a saturation bug:

```python
unit(signals["face_pixels"], size_ref=112)   # clips at 112
```

Nearly every detected face is larger than 112 px, so this term was **1.0 for almost every
image** and carried no ranking information at all. The equal-weight mean then defaulted to the
one term with real variance — blur — which is among the *weakest* signals (AUERC 0.0193).
Meanwhile embedding norm was not in the composite at all.

So the guessed constants did not merely mis-weight the signals; one of them silently deleted
the strongest signal.

## Result 3 — the repair

`composite_v2` makes four structural changes, each traceable to a measurement above:

| change | why |
|---|---|
| face size on a **log scale** (24 → 400 px) instead of clipped at 112 | fixes the saturation that deleted the strongest signal |
| **drop contrast** | measured worse than random rejection |
| **add embedding norm** (rank-normalised) | +30.8 % over random and previously absent |
| **rank-normalise** blur | Laplacian variance is heavy-tailed; raw values are unusable in a mean |
| clipping as a **multiplicative penalty** | over/under-exposure is a hard defect, not a graded one |

| signal | AUERC | vs random |
|---|---:|---:|
| **composite_v2** | **0.0034** | **+85.1 %** |
| face_pixels | 0.0096 | +57.9 % |
| composite_v1 | 0.0136 | +40.6 % |
| random | 0.0229 | — |

v2 now beats every individual component, which is what a composite is supposed to do.

## Result 4 — two validity checks on that number

v2's weights were chosen *after* seeing which signals worked on LFW, so the headline is
optimistic by an unknown amount. Two checks bound the concern:

**Fold split** (LFW's 10 official folds, first 5 vs last 5):

| signal | folds 1–5 | folds 6–10 | all |
|---|---:|---:|---:|
| composite_v2 | 0.0040 | 0.0028 | 0.0034 |
| face_pixels | 0.0089 | 0.0106 | 0.0096 |
| composite_v1 | 0.0125 | 0.0147 | 0.0136 |
| random | 0.0262 | 0.0244 | 0.0252 |

Stable, and v2 beats face_pixels on **both** halves — the result is not driven by a subset.

**Transductive check.** v2 rank-normalises two signals, so a face's score depends on the other
faces present. Recomputing those ranks using only folds 6–10 gives **0.0028**, identical to
ranking over the full set. Rank normalisation is not doing the work.

Neither check removes the design-time selection effect — both halves informed the repair — so
**0.0034 should be read as an upper bound**, with 0.0096 (face size alone, chosen without
reference to LFW) as the conservative floor. A second identity-labelled dataset would settle it.

## Decisions taken

1. **Adopt `composite_v2`; retain `composite` only as the superseded reference** that E9's
   numbers refer to.
2. **Do not train or download a FIQA model.** §2.4's bet pays off: a free composite reaches
   85 % of the way from random to a strong rejection curve, using signals already computed. CR-FIQA
   remains worth a comparison, but is no longer on the critical path.
3. **Face size is the dominant quality signal** — weight it accordingly (0.45) and never let it
   saturate. This is also a cheap product rule: small faces are bad results regardless of what
   any model says.
4. **Drop `contrast` from quality entirely.** It is worse than random.
5. **Rank-normalisation must use fixed reference quantiles in production**, computed once from a
   sample of the user's collection, not from whatever batch happens to be in memory. Otherwise a
   face's quality score changes depending on what it is indexed alongside.
6. **Quality feeds confidence, not just filtering** (§11): E12 established that intervals are
   only valid in-domain, and low quality is one of the ways a face leaves the domain.

## Threats to validity

- **LFW is easy and homogeneous** — mostly well-lit frontal celebrity photographs. The quality
  range here is narrower than a user's directory, and the baseline FNMR (0.024) is already low.
  XQLFW exists precisely for this and would be the better follow-up.
- v2's weights were informed by these LFW measurements (see Result 4).
- ERC measures quality's value *for recognition*. Quality for **attractiveness** prediction is a
  related but distinct target and is not validated here.
- LFW has known demographic skew, so per-group quality behaviour is unmeasured.
- E11 reported a `quality_composite` correlation using **v1**; that row now refers to the
  superseded metric. E11's conclusion is unaffected — its R² = 0.017 was computed from the raw
  individual signals, not the composite.

## Reproduce

```bash
python scripts/run_e9_quality.py     # LFW must be present (sklearn's lfw_home)
```
