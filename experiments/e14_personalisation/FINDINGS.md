# E14 — Personalisation: is there learnable individual taste?

E7 promoted this from a Phase-10 extra to a core question: two rater pools disagreed on 80 %
of a top-100, so a single population ranking cannot be the honest answer. The obvious fix is
a per-user model. This asks whether that actually works, and at what label cost.

Run: 2026-09-03 · seed 1337 · raw numbers in `results.json`

## Setup

Simulated users = individual raters, which both datasets make possible:

- **SCUT-FBP5500**: 60 raters × 5,500 images, **complete** matrix (330,000 ratings). A
  demographically homogeneous pool — Chinese undergraduates aged 18–27.
- **MEBeauty**: 360 raters × 2,520 images, sparse (61,404 ratings). A deliberately diverse
  pool — mixed ethnicity, age and gender.

Three models over the same cached frozen features (ArcFace R50 + CLIP):

| Model | Trained on |
|---|---|
| **population** | mean of all **other** raters (leave-one-rater-out, so the target rater's own labels never leak in) |
| **personal** | only this rater's *n* labels |
| **residual** | population prediction **+** ridge on (this rater's rating − population prediction) |

Evaluated by Spearman against that rater's own held-out ratings. **The population baseline
is recomputed on the same user cohort available at each support size** — users rated
different numbers of images, so the cohort shrinks as *n* grows, and comparing a large-*n*
mean against the all-users baseline would compare different populations.

## The ceiling, measured rather than assumed

SCUT re-showed images to raters during annotation; seven raters ended up rating **all 5,500
images twice**. That gives a real test–retest reliability:

| | Spearman |
|---|---:|
| A rater vs. **themselves**, re-rating the same faces | **0.5749** |
| A rater vs. the **consensus of the other 59** | **0.7662** |
| Attenuation limit — √reliability, the max *any* predictor can reach against a single noisy rating | **0.7582** |

**People agree with the crowd more than they agree with their own earlier judgment.** That is
not a paradox: a single rating is noisy, the mean of 59 is not, so the consensus predicts the
*stable* part of someone's preference better than a second noisy sample of it does.

The consequence is decisive. Since individual ratings have reliability 0.575, no model can
correlate above **0.758** with a single one of them. Our population model reaches **0.7238**
— **95.5 % of the theoretical maximum**, leaving 0.0345 of headroom, all of which is rater
noise rather than personal taste.

## Result 1 — in a homogeneous pool, personalisation actively hurts

SCUT-FBP5500, 60 simulated users:

| labels | population (matched) | personal-only | **residual** | gain | users helped |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.7238 | — | 0.7238 | — | — |
| 10 | 0.7238 | 0.4703 | 0.7215 | −0.0023 | 23 % |
| 50 | 0.7238 | 0.6304 | 0.7171 | −0.0067 | 17 % |
| 100 | 0.7238 | 0.6560 | 0.7133 | −0.0105 | 17 % |
| 500 | 0.7238 | 0.6819 | 0.6980 | **−0.0258** | 7 % |
| 2000 | 0.7238 | 0.7197 | 0.7211 | −0.0027 | 28 % |

**Personalisation never beats the population model, at any label budget**, and it helps only
7–28 % of users. The damage peaks around 250–500 labels and then *recovers* toward baseline
at 2,000 — the signature of fitting noise: with a few hundred labels there is enough data to
confidently learn a spurious residual, and only with thousands does it average back out.

This is not a failure of the method. It is the ceiling doing its job: at 95.5 % of the
attenuation limit there is essentially nothing left to personalise. **In a homogeneous rater
pool, "individual taste" is mostly measurement noise.**

## Result 2 — in a diverse pool, personalisation works, and cheaply

MEBeauty, 128 simulated users:

| labels | population (matched) | personal-only | **residual** | gain | users helped |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.5054 | — | 0.5054 | — | — |
| 10 | 0.5054 | 0.2066 | 0.5091 | +0.0037 | 60 % |
| 25 | 0.5054 | 0.3295 | 0.5185 | +0.0131 | 67 % |
| 50 | 0.5054 | 0.4062 | 0.5226 | +0.0172 | 62 % |
| **100** | 0.5066 | 0.4613 | **0.5320** | **+0.0254** | **68 %** |
| 250 | 0.4826 | 0.4922 | 0.5135 | +0.0309 | 71 % |
| 500 | 0.4717 | 0.5027 | 0.5125 | +0.0407 | 71 % |

Gains are positive from **10 labels**, grow monotonically, and help ~two-thirds of users.
The magnitude is modest — **+0.025 Spearman at 100 labels** — but it is real, consistent, and
achieved at a label budget a user would actually tolerate (100 like/dislike judgements).

**The difference between the two datasets is the answer to the question.** Personalisation
headroom is a property of **rater-pool diversity**, not of the algorithm. The same code, the
same features and the same architecture help in one pool and hurt in the other.

## Result 3 — the residual formulation is the right architecture

Compare `personal-only` against `residual` at low label counts:

| | 10 labels | 100 labels |
|---|---:|---:|
| SCUT personal-only | 0.4703 | 0.6560 |
| SCUT residual | **0.7215** | **0.7133** |
| MEBeauty personal-only | 0.2066 | 0.4613 |
| MEBeauty residual | **0.5091** | **0.5320** |

Training a model on a user's labels alone is catastrophic in the cold-start regime — 0.21 vs
0.51 at 10 labels. The residual decomposition of `docs/RESEARCH.md §15.5` degrades
gracefully to the population model when the user has said little, and only departs from it
where there is evidence to. **Confirmed as the production design.**

## Result 4 — personalise selectively: it helps most those the consensus fits worst

On MEBeauty at 100 labels, correlating "how well the population model already fits this user"
against "how much personalisation gains them": **ρ = −0.182**.

| Cohort | mean gain |
|---|---:|
| Users the consensus fits **worst** (bottom 25 %) | **+0.0483** |
| Users the consensus fits **best** (top 25 %) | +0.0089 |

A **5.4× difference**. Personalisation should be *gated on population-model fit*, not applied
uniformly: it is nearly free to skip for well-served users, and worth ~5× more for the
poorly-served ones. This also mitigates the SCUT failure mode — the users who would be
harmed by noise-fitting are exactly the ones the gate would exclude.

## Verdict

**Personalisation is worth building, with three qualifications.**

1. It only pays off when the user base is heterogeneous relative to the training pool. Real
   users of a photo-directory tool resemble MEBeauty's pool far more than SCUT's, so the
   MEBeauty result is the relevant one — but this should be re-measured on real feedback.
2. The gain is **modest**: +0.025 Spearman at 100 labels, ~5 % relative. It does **not** close
   the gap that E7 opened (80 % of a top-100 changing between rater pools). Personalisation
   is a refinement, not the resolution.
3. It must be the **residual** formulation, and it should be **gated** on population fit.

## Decisions taken

1. **Ship the residual formulation**, `final = population + α(n)·user_residual`, over the
   same cached frozen features — confirmed by Result 3.
2. **Gate personalisation on population-model fit.** Enable it for users the population model
   serves poorly; leave well-served users on the population model.
3. **Target ~50–100 user labels** before enabling the personal component. Gains are visible
   at 10 and solid by 100.
4. **Prefer pairwise feedback in the UI.** These simulations used absolute ratings, whose
   test–retest reliability is only 0.575 — that noise is the binding constraint, and pairwise
   comparisons are the standard remedy. Worth measuring directly (see below).
5. **Do not claim personalisation solves the E7 problem.** Which rater pool the population
   model was trained on still dominates.

## Threats to validity

- MEBeauty's per-user test sets are small (raters averaged ~170 ratings each), so per-user
  Spearman is noisy. The matched-cohort correction handles the *comparison* bias but not the
  variance; the ≥250 rows rest on 41 and 21 users.
- The two datasets differ in more than pool diversity — image domain, rating scale and
  ratings-per-image all differ — so "diversity" is the most plausible explanation for the
  divergence, not a proven one. A controlled test would subsample MEBeauty's raters into
  homogeneous and heterogeneous cohorts.
- Test–retest reliability comes from only 7 raters (though each re-rated all 5,500 images).
- Simulated raters are not users. Real feedback is pairwise, sparse, self-selected toward
  liked items, and collected over time. The label budgets here are optimistic.
- Only linear residual heads were tested.

## Reproduce

```bash
python scripts/run_e14_personalisation.py
```
