# Facet — Phase 1 Research Report

**Face analysis, filtering and ranking system**
Status: Phase 1 (research) complete · Phase 4 (baselines) started
Author: generated for `sadiqebrahim13@gmail.com`
Date: 2026-09-02
Target hardware: 1× NVIDIA RTX A6000 (48 GB), 64× Xeon Gold 6426Y, 125 GB RAM

---

## 0. How to read this document

This report answers the Phase-1 brief. It is organised as the brief asked:

| § | Topic |
|---|---|
| 1 | Executive summary, headline findings, and Phase 4 results |
| 2 | Existing models (detection, alignment, embedding, quality, age, gender, beauty) |
| 3 | Relevant papers |
| 4 | Relevant datasets |
| 5 | Available pretrained models |
| 6 | Beauty / attractiveness modelling approaches |
| 7 | Age estimation approaches |
| 8 | Gender / category classification approaches |
| 9 | Ranking approaches |
| 10 | Ensemble approaches |
| 11 | Confidence / uncertainty approaches |
| 12 | Dataset and model licensing |
| 13 | Known limitations and biases |
| 14 | Recommended experiments |
| 15 | Proposed technical architecture + development roadmap |

Two conventions are used throughout:

- **[V]** — *verified*: I fetched the primary source (paper PDF, repository README, dataset
  README, license file) and read the number or clause myself. Where a number is [V] it is
  reproduced exactly.
- **[S]** — *secondary*: reported by a search summary or a third-party page, not confirmed
  against the primary source. **[S] numbers must not be used to make a go/no-go decision
  without re-verification.** They are included because they establish the shape of the field.

Anything marked [S] is a task in §14, not a fact to build on.

---

## 1. Executive summary

### 1.1 The single most important finding

The facial-beauty-prediction (FBP) literature reports Pearson correlations against ground
truth of **0.90–0.935** on SCUT-FBP5500. The same dataset's own paper reports that the
**correlation between two groups of human raters is 0.770–0.785** [V].

| Quantity | Value | Source |
|---|---|---|
| Correlation, female labelers vs. ground truth, all faces | **0.785** | SCUT-FBP5500 paper, Table III [V] |
| Correlation, male labelers vs. ground truth, all faces | **0.781** | same [V] |
| Correlation, all labelers, all faces | **0.770** | same [V] |
| ResNeXt-50 5-fold PC (dataset paper baseline) | **0.8997** | same, Table VII [V] |
| Modern reported SOTA PC | 0.9142 – 0.935 | various [S] |

A model at PC 0.93 is **not** better than a human at judging beauty. It is better than a
human at **predicting the average of 60 specific Chinese undergraduates**, which is a
different and much easier task: the mean of 60 ratings has roughly `1/sqrt(60)` the variance
of a single rating, so it is an intrinsically low-noise target. Individual raters disagree
enormously — the paper reports per-image rating standard deviations concentrated in
**[0.6, 0.7]** on a 1–5 scale [V], i.e. roughly ±15 % of the full range, per image.

**Consequences for this project, and they are structural, not cosmetic:**

1. The system must never present attractiveness as a measurement. The honest statement is
   *"this face is predicted to score X on the SCUT-FBP5500 rating scale, as defined by 60
   Chinese undergraduates in 2017"*. §13 and §15.7 make this a hard product requirement.
2. **Modelling rater disagreement is not a nice-to-have, it is the main signal.** The spread
   of the 60 ratings is the model's best available proxy for "is this face polarising or
   universally agreed". A face at mean 3.5 with σ=0.3 and a face at mean 3.5 with σ=1.2 are
   completely different results, and a mean-regression model cannot tell them apart. This is
   the strongest argument for label-distribution learning (§6.3) over plain regression.
3. Chasing PC beyond ~0.90 on SCUT-FBP5500 is **fitting the annotator pool, not the concept**.
   Effort is better spent on cross-dataset generalisation (§14, E7) and personalisation (§6.9).
4. Personalisation (Phase 10) is not an "advanced feature" bolted on at the end. Given
   inter-rater r≈0.77, a per-user model has a *large* amount of variance available to explain.
   It should be designed for from the start (§15.5).

### 1.2 Second finding: the evaluation literature is partly unreliable

`A Call to Reflect on Evaluation Practices for Age Estimation` (arXiv 2307.04570) reports that
for age estimation, once a **unified subject-exclusive (identity-disjoint) split** is used, the
apparent gains of specialised architectures (CORAL, MWR, Mean-Variance, DLDL-v2) largely
**disappear**, and all methods yield comparable results [S — high priority to verify, see §14 E0].

This is exactly the identity-leakage failure the brief warns about, confirmed in the
neighbouring literature. It has a direct implication: **published leaderboard numbers are not a
valid basis for model selection here.** We must re-benchmark candidates ourselves under our own
splits. The report therefore treats published numbers as *candidate generation*, and §14 as the
actual selection mechanism.

SCUT-FBP5500 has a related problem: the official 5-fold split is a random image split, and the
dataset provenance (§4.1) includes three third-party sources, at least one of which
(10k US Adults Faces) is known to contain repeated identities across research corpora. **We do
not yet know whether SCUT-FBP5500's official splits are subject-disjoint.** Measuring this is
experiment E1 in §14 and is a precondition for trusting any number we produce.

### 1.3 Recommended system shape (provisional, to be confirmed by experiment)

Do **not** build one model that does everything. The tasks have different data, different
licenses, different failure modes and different maturity levels:

```
                       ┌── age        →  MiVOLO (Apache-2.0, strong, off-the-shelf)
detect + align         ├── gender     →  MiVOLO / InsightFace genderage
(SCRFD + 5-pt / 106-pt)├── quality    →  CR-FIQA-style, or embedding-norm proxy
                       ├── identity   →  ArcFace embedding (dedup, "more like this")
                       └── beauty     →  OUR model: frozen face embedding + distribution head
                                          (this is the only part we must train)
```

The reasoning:

- **Age and gender are solved well enough off the shelf.** MiVOLO is Apache-2.0 for both code
  and weights [V], reports 4.22–4.24 MAE on IMDB-clean and 99.46 % gender accuracy [V]. Training
  our own would burn weeks to match it.
- **Beauty is the only component with no good permissively-licensed pretrained model**, and the
  only one where the "right" target is genuinely open. It is where our research effort belongs.
- **A frozen face-recognition embedding is a very strong beauty feature.** ArcFace embeddings are
  trained to be identity-discriminative on tens of millions of faces, which requires encoding
  precisely the fine-grained geometry and texture that attractiveness ratings key off. The
  dataset paper's own hand-crafted geometric baseline reaches only PC 0.5948–0.6738 [V], while a
  fine-tuned ResNeXt-50 reaches 0.8997 [V]. Experiment E2 (§14, already implemented as
  `exp001`) tests whether a *frozen* embedding plus a linear head closes most of that gap at
  ~1/100th the training cost. If it does — and the prior is that it does — it changes the whole
  engineering plan, because a linear head can be retrained per-user in milliseconds, which is
  exactly what Phase 10 personalisation needs.

### 1.4 Phase 4 results — what the experiments actually showed

The predictions in §1.3 have now been tested. Two experiments have run; both are written
up in `experiments/`. Summary, because it changes several recommendations below:

**exp001 (E2) — frozen features + a linear head beat fine-tuned CNNs.**

| Representation (frozen, ridge head) | 5-fold PC | vs. published fine-tuned |
|---|---:|---|
| 86-landmark geometry | 0.6819 | ≈ published geometric baseline (0.5948–0.6738) ✓ harness sane |
| ArcFace R50 | 0.8268 | below AlexNet (0.8634) |
| ArcFace R100 | 0.8712 | ≈ ResNet-18 (0.8900) |
| **CLIP ViT-B/32** | **0.9279** | **beats ResNeXt-50 (0.8997)** |
| **ArcFace R50 + CLIP** | **0.9398** | **beats reported 2025 SOTA (0.932–0.935)** |
| all four fused | 0.9419 | — |

The §1.3 hypothesis was that frozen ArcFace + ridge would reach 0.85–0.90. It reached
0.827 — and **CLIP, which I expected to be the weaker of the two, won outright.** The
architectural bet in §15.1 is confirmed; my prediction about *which* representation would
win was wrong, in an informative way (see below).

**Three findings that change recommendations in this document:**

1. **CLIP > ArcFace by a wide margin (0.928 vs. 0.827)**, despite ArcFace being
   face-specific and trained on hundreds of millions of faces. The most likely explanation
   is the confound predicted in §13.1.5: ArcFace is trained to be identity-discriminative,
   which means trained to *discard* expression, lighting, styling and image quality — the
   very things raters respond to. **This is evidence that attractiveness ratings
   substantially encode photography and presentation, not only facial structure.**
   It raises E11c (attractiveness vs. image-quality correlation) from a formality to a
   priority, and it means the product must describe the score accordingly.
2. **The label-distribution head (§6.3) beat mean-regression on every representation
   tested**, by +0.002 to +0.018 PC, with the largest gains where the representation is
   weakest. Its predicted spread correlates with true rater disagreement at r = 0.35.
   §6.3's recommendation stands, now with evidence.
3. **Bagged-ensemble spread is not a usable confidence signal.** Measured coverage was
   0.22 at 68 % nominal and 0.43 at 95 % nominal — underestimating uncertainty by ~3× —
   with correlation to actual error of only 0.10. This is precisely the failure §11.4
   warns about. **Conformal prediction is now mandatory, not preferred.**

Also: per-subgroup reporting revealed that ArcFace alone has a 0.155 PC spread across the
four demographic subgroups (worst on Caucasian faces, which have 750 training images vs.
2,000), while the fused representation reduces that spread to 0.021. Aggregate accuracy
concealed a large disparity, so **per-group metrics must be a default output, not an audit
step** (§13.3).

**E1 — the official splits are NOT subject-disjoint.** ~5 % of every official test set
shares an identity with its training set (max pairwise similarity 0.9976 — near-duplicates
exist). This confirms the concern raised in §1.2 and §4.1 with hard numbers. Measured
impact on the headline metric is **+0.003 PC**: real, must be disclosed, but not large
enough to explain the exp001 result. Consequence: build identity-disjoint splits and report
both, and reuse the clustering for the pipeline's duplicate detection.

**What has not changed:** none of this says anything about generalisation. Everything above
is one benchmark whose human ceiling is 0.77 and which is now demonstrably saturated.
**E7 (cross-dataset, MEBeauty) remains the go/no-go gate**, and a CLIP-based model that
partly keys on photographic style is exactly the kind of model that might transfer poorly.

### 1.5 The licensing problem, stated early because it constrains everything

| Asset | License | Commercial? |
|---|---|---|
| InsightFace **code** | MIT | Yes |
| InsightFace **pretrained weights** (buffalo_l, antelopev2) | research only | **No** [V] |
| SCUT-FBP5500 dataset | non-commercial research only | **No** [V] |
| CelebA | non-commercial research only | **No** [V] |
| WIDER FACE | CC BY-NC-ND 4.0 | **No** [V] |
| UTKFace | non-commercial | **No** [V] |
| IMDB-WIKI / IMDB-Clean | IMDb non-commercial terms | **No** [V] |
| MORPH | academic vs. paid commercial (UNCW) | paid [V] |
| **FairFace** | **CC BY 4.0** | **Yes** [V] |
| **MiVOLO** code + weights | **Apache-2.0** | **Yes** [V] |

**A beauty model trained on SCUT-FBP5500 cannot be commercialised.** Neither can anything built
on InsightFace's released weights. This is fine for a research project and for personal use, and
it is fatal for a product, so it must be a conscious decision made now rather than discovered
later. §12 lays out the two viable paths (stay research-only, or build a clean-room commercial
track on FairFace + MiVOLO + own-collected preference data). The repository is structured so
that the license status of every artifact is machine-readable (`configs/models/*.yaml` carries a
`license:` and `commercial_use:` field) and so that a "commercial-safe subset" can be selected by
configuration rather than by rewriting code.

---

## 2. Existing models

### 2.1 Face detection

| Model | Arch | Train data | WIDER FACE (E/M/H) | Speed | Pretrained | License | Notes |
|---|---|---|---|---|---|---|---|
| **SCRFD** (`det_10g`) | ResNet-ish + sample/compute redistribution | WIDER FACE | SOTA in all subsets at its FLOP budget [S] | 10 GF; ~5 ms GPU | **Yes, on this machine** | code MIT / weights research-only [V] | Ships in `buffalo_l`. Outputs bbox + 5 keypoints, which is exactly what alignment needs. Default choice. |
| RetinaFace (R50) | FPN + context module, multi-task | WIDER FACE | very strong, slower | ~15–25 ms GPU | Yes | research-only | Best accuracy tier with DSFD [S] |
| YOLO5Face / YOLOv8-face | YOLO detector + 5-kpt head | WIDER FACE | YOLOv5x6: 96.67 / 95.08 / 86.55 [S] | fast | Yes | GPL-3.0 (YOLOv5/v8) ⚠ | GPL is a real problem for a shipped product |
| **YuNet** | tiny CNN, 75 k params | WIDER FACE | lower, but "several times faster than most" [S] | ~1 ms CPU | Yes, in OpenCV | **Apache-2.0 via OpenCV Zoo** | The commercial-safe fallback, and the CPU path |
| MTCNN | 3-stage cascade | — | weak by modern standards | slow | Yes | MIT | Legacy; do not use |
| BlazeFace / MediaPipe | mobile SSD | — | short-range only | very fast | Yes | Apache-2.0 | Good CPU option, weak on small/rotated faces |

**Recommendation:** SCRFD (`det_10g`, already on disk) for the research track; YuNet as the
commercially-clean and CPU-only alternative. Both emit 5 keypoints so the alignment stage is
identical and they are swappable behind one interface (`FaceDetector` protocol, §15.2).

**Detector choice matters more than it looks.** For a directory-scanning product, recall on
small, blurred, profile and partially-occluded faces determines what the user can find at all. A
missed face is invisible in the UI — there is no score to inspect. This makes detection recall a
*product* metric, and it is why §14 E8 evaluates the detector separately rather than assuming it.

### 2.2 Face alignment / landmarks

| Model | Points | Speed | Available | Purpose here |
|---|---|---|---|---|
| SCRFD 5-point (from detector) | 5 | free | Yes | Standard ArcFace similarity-transform to 112×112. **This is the alignment we need.** |
| InsightFace `2d106det` | 106 | ~2 ms | **Yes, on disk** | Geometry features; pose/expression estimation; quality heuristics |
| InsightFace `1k3d68` | 68 (3D) | ~3 ms | **Yes, on disk** | Yaw/pitch/roll → pose-based quality gating and fairness slicing |
| SCUT-FBP5500 supplied | 86 | n/a (labels) | **Yes, in dataset** | Ground-truth geometry for the classical geometric baseline (exp002) |
| PIPNet / SynergyNet / 3DDFA-V2 | 68–68k | fast | Yes | Only if 106-pt proves insufficient |
| MediaPipe FaceMesh | 468 | fast | Yes, Apache-2.0 | Dense mesh; commercially clean; overkill for ratios |

Alignment is not a detail. FBP models are highly sensitive to crop margin and in-plane rotation
because attractiveness ratings partly encode face *shape*, and a bad crop changes apparent
shape. **The crop protocol must be fixed and versioned** (`configs/pipeline/align.yaml`) and
recorded in the feature store, or features extracted on different days will not be comparable.
Experiment E5 (§14) quantifies exactly how much crop margin matters — my expectation is that it
is worth more than the choice of backbone, which would be an important and cheap finding.

### 2.3 Face embedding / recognition

| Model | Backbone | Train data | Benchmarks | On disk | License |
|---|---|---|---|---|---|
| **`w600k_r50`** (buffalo_l) | IResNet-50 | WebFace600K | LFW 99.83, CFP-FP 99.33, AgeDB-30 98.23, MR-ALL 91.25 [V] | **Yes** | research-only [V] |
| **`glintr100`** (antelopev2) | IResNet-100 | Glint360K | higher accuracy tier [V] | **Yes** | research-only [V] |
| AdaFace | IResNet-100 | WebFace4M/12M | quality-adaptive margin; strong on low-quality | download | MIT code / research weights |
| TransFace | ViT | Glint360K | ViT-based FR | **repo present on this machine** | research |
| LVFace | large ViT | — | **repo present on this machine** | research |
| CLIP ViT-B/32 | ViT | WIT-400M | general semantics, not face-specific | **in HF cache** | MIT |
| **FaRL** | ViT-B/16 | LAION-Face 20M | face-specific vision-language pretraining; strong few-shot on attributes [V] | download | MIT |
| DINOv2 / DINOv3 | ViT | LVD-142M | strong dense correspondence on faces without face training [S] | download | DINOv2 Apache-2.0; DINOv3 restrictive ⚠ |

The interesting empirical question — and it is genuinely open — is **which frozen representation
best predicts beauty**. There are three families with different inductive biases:

- **Identity embeddings (ArcFace)** encode fine geometry and texture but are *trained to discard*
  exactly what varies within an identity: expression, lighting, and to some extent age. Some of
  that discarded variance is attractiveness-relevant.
- **Semantic embeddings (CLIP)** encode "what kind of image is this", including makeup, styling,
  photo quality and aesthetic conventions — plausibly a large part of what raters respond to,
  and a known confound.
- **Face-specific SSL (FaRL, DINOv2)** sit in between.

One comparative study reports frozen-encoder ordering **CLIP > DINOv2 = BLIP > FaRL > MAE** on
facial tasks [S]. That ordering is surprising enough (CLIP beating a face-specific model on face
tasks) that it should be treated as a hypothesis to test, not a result to adopt. Experiment E2
tests ArcFace-R50, ArcFace-R100, CLIP and geometry on identical splits with identical heads,
which is the only way to answer this for *our* target.

### 2.4 Face image quality assessment (FIQA)

| Method | Idea | Needs training? | Notes |
|---|---|---|---|
| **CR-FIQA** (CVPR 2023) | predicts sample's *relative classifiability* — how separable it is from other identities | yes (or use released weights) | "performs best overall, followed closely by FaceQAN" [S]; current default recommendation |
| FaceQAN | adversarial-perturbation stability | yes | close second [S] |
| MagFace (CVPR 2021) | **embedding magnitude ∝ quality**, learned jointly with recognition | free if using MagFace weights | Elegant: quality comes out of the recognition model at zero extra cost |
| SDD-FIQA | similarity-distribution distance | yes | strong [S] |
| SER-FIQ | embedding stability under dropout | **no training, but N forward passes** | Expensive at index time; also doubles as an uncertainty estimate (§11) |
| CLIB-FIQA (CVPR 2024) | quality + confidence calibration | yes | Directly relevant to §11 |
| ViT-FIQA (2025) | ViT backbone | yes | Recent [S] |
| BRISQUE / NIQE | classical no-reference IQA | no | Not face-specific; still useful as a cheap *image*-level (not face-level) signal |

**Practical recommendation for v1:** do not start by training a FIQA model. Build a composite
quality score from signals that are **free** because we already compute them:

```
quality = f( detector confidence,
             face bbox pixel area,
             |yaw|,|pitch|,|roll|   (from 1k3d68),
             Laplacian-variance blur on the crop,
             exposure / clipping stats,
             ArcFace embedding L2 norm    ← MagFace-style free quality proxy
             ... )
```

Then validate that composite against a real FIQA model (E9). The embedding-norm proxy is the
interesting one: even for models not trained with MagFace's objective, embedding norm correlates
with quality, and we get it for free from a forward pass we are already doing. If the composite
tracks CR-FIQA at r>0.7, we save a model from the pipeline permanently.

Quality matters here for a specific reason beyond hygiene: **beauty predictions on low-quality
faces are unreliable in a way the model will not otherwise tell us.** A blurry face regresses
toward the training mean, which produces confident-looking mid-range scores. Quality is therefore
an input to the *confidence* estimate (§11), not merely a filter.

### 2.5 Age estimation

| Model | Approach | Reported | License | Notes |
|---|---|---|---|---|
| **MiVOLO / MiVOLO-v2** | multi-input transformer (face + body), joint age+gender | IMDB-clean **4.22–4.24 MAE**, gender **99.46 %**; UTKFace 4.23 / 97.69 %; Lagenda 3.65–3.99 / 97.36–97.99 %; AgeDB 5.55–5.58 / 98.3 % [V] | **Apache-2.0** [V] | Uses body when face is small/occluded — a real advantage for in-the-wild directories |
| InsightFace `genderage` | small CNN | fast, lower accuracy | research-only | **Already on disk**; fine as a cheap first pass |
| DLDL-v2 | label-distribution + expectation | MORPH 1.97 MAE [S] | varies | MORPH numbers are suspect under random splits (§1.2) |
| CORAL | consistent-rank ordinal regression | — | MIT | Clean ordinal formulation; good baseline |
| MWR | moving-window regression, mean+variance | — | — | Gives a variance → useful for §11 |
| C3AE | compact cascade | — | — | Edge/CPU option |
| OrdinalCLIP | language-guided rank prompts | — | — | Interesting for zero-shot / few-shot ordinal tasks |

**Recommendation:** adopt **MiVOLO** as the age+gender model. It is accurate, fast, permissively
licensed, and handles the face-not-visible case. Benchmark it on our own held-out data (E4)
rather than trusting published MAE, and specifically measure MAE *by decade and by
predicted-race bucket*, because age MAE is famously non-uniform (worst at the extremes and for
under-represented groups) and a flat "±4 years" claim in the UI would be dishonest.

One caveat to record: MiVOLO's weights are Apache-2.0, but they were trained partly on
IMDB-clean, which derives from IMDb data under non-commercial terms. Apache-2.0 on the weights
does not launder the training data's provenance. For a research/personal system this is a
non-issue; for a commercial product it is a question for a lawyer, and it is logged in
`docs/LICENSING.md` rather than hand-waved.

### 2.6 Gender / category classification

Accuracy here is ~97–99 % for the binary task on standard benchmarks [V], so the modelling is
not the hard part. The hard parts are definitional and ethical:

- Every available model predicts **perceived binary presentation**, not identity. The label in
  the training data was assigned by an annotator looking at a photo.
- Accuracy is systematically lower for some groups — this is the Buolamwini/Gebru
  *Gender Shades* result [S], where error rates were highest for darker-skinned subjects and
  higher for women than men in commercial systems.
- A hard gender filter that silently drops 3 % of matching faces is a *product* bug the user
  cannot see.

**Design decision (§15.4):** gender is exposed as a **soft, confidence-weighted preference by
default**, not a hard filter. The UI labels it "presenting as", and low-confidence faces are
surfaced rather than hidden unless the user explicitly opts into strict filtering. This is both
more honest and more useful — a strict filter on a noisy classifier loses recall the user never
finds out about.

### 2.7 Beauty / attractiveness

Covered in depth in §6. Summary of the model landscape:

| Model | Approach | SCUT-FBP5500 PC | Weights | License |
|---|---|---|---|---|
| ResNeXt-50 (dataset baseline) | regression | **0.8997** (5-fold) [V] | yes (Caffe/PyTorch) | research |
| ComboLoss + SENet | improved expectation loss | SOTA at publication [S] | yes (GitHub) | research |
| R3CNN | ranking-guided regression | 0.9142 [S] | — | research |
| MD-Net | multi-branch | 0.9235 [S] | — | — |
| VM-BeautyNet | ViT + Mamba ensemble | 0.9212 [S] | — | — |
| Hybrid VMamba-ViT | hybrid SSM/attention | 0.9261 [S] | — | — |
| Diff-FBP | diffusion features | 0.932 [S] | — | — |
| SCAT | — | 0.935 [S] | — | — |
| UOL | uncertainty-oriented order learning | better *cross-dataset* [V, qualitative] | — | — |
| MetaFBP | meta-learned personalisation | (personalised task) | yes (GitHub) | research |

Note the shape of this table: a ~4-point PC spread between the 2018 baseline and 2025 SOTA,
achieved with progressively heavier architectures, on a benchmark whose human ceiling is 0.77.
**This is a saturated benchmark.** The productive directions are the two that are not measured by
it — cross-dataset generalisation (UOL's contribution) and personalisation (MetaFBP's) — which is
why §6 and §14 weight those heavily rather than pursuing another PC decimal.

---

## 3. Relevant papers

Grouped by what they contribute to this project. Verification status refers to whether I read
the primary source.

**Datasets and benchmarks**
- Liang, Lin, Jin, Xie, Li. *SCUT-FBP5500: A Diverse Benchmark Dataset for Multi-Paradigm Facial
  Beauty Prediction.* ICPR 2018. arXiv:1801.06345. **[V — full text read]** The foundational
  dataset for this project. Source of every [V] number in §4.1.
- Lebedeva, Guo et al. *MEBeauty: a multi-ethnic facial beauty dataset in-the-wild.*
  Neural Computing and Applications, 2021. 2550 in-the-wild images, six ethnic groups [S].
  **The primary cross-dataset generalisation test set for us** (§14 E7).
- Kärkkäinen & Joo. *FairFace: Face Attribute Dataset for Balanced Race, Gender, and Age.*
  arXiv:1908.04913. ~108 k images from YFCC-100M, CC BY 4.0 [V].
- Yang, Luo, Loy, Tang. *WIDER FACE: A Face Detection Benchmark.* CVPR 2016.
- Murray, Marchesotti, Perronnin. *AVA: A Large-Scale Database for Aesthetic Visual Analysis.*
  CVPR 2012. ~250 k images, ~210 ratings each [S] — the model for *image* aesthetics, and the
  source of the rating-distribution idea.

**Beauty modelling**
- Xu, Jin et al. *Label distribution based facial attractiveness computation by deep residual
  learning.* arXiv:1609.00496. **The LDL formulation** central to §6.3.
- Gao et al. *Learning Expectation of Label Distribution for Facial Age and Attractiveness
  Estimation* (DLDL-v2). arXiv:2007.01771. Unifies age and beauty under one distribution-learning
  head — directly relevant to our multi-task design (§10.4).
- Xu, Lu et al. *ComboLoss for Facial Attractiveness Analysis with Squeeze-and-Excitation
  Networks.* Code: `github.com/lucasxlu/ComboLoss`. Claims SOTA on SCUT-FBP, HotOrNot and
  SCUT-FBP5500 [S].
- Lin, Liang, Jin. *R3CNN / Regression-guided ranking.* Ranking + regression jointly [S].
- *Uncertainty-oriented Order Learning for Facial Beauty Prediction* (UOL). arXiv:2409.00603.
  **[V — abstract/method read]** Argues absolute-score regression generalises poorly across
  datasets because beauty *scales* are inconsistent between annotation efforts, and that learning
  *order* plus explicit uncertainty transfers better. **This is the most directly load-bearing
  paper for our cross-dataset problem.**
- Zhu et al. *MetaFBP: Learning to Learn High-Order Predictor for Personalized Facial Beauty
  Prediction.* ACM MM 2023. arXiv:2311.13929. Code available. Benchmarks PFBP-SCUT500,
  PFBP-SCUT5500, PFBP-US10K. **The blueprint for Phase 10.**
- *Ethically aligned Deep Learning: Unbiased Facial Aesthetic Prediction.* arXiv:2111.05149.
  Required reading for §13.
- Altwaijry & Belongie. *Relative ranking of facial attractiveness.* WACV 2013. Early pairwise
  formulation; cited in the SCUT-FBP5500 paper [V].

**Age / ordinal**
- Kuprashevich & Tolstykh. *MiVOLO: Multi-input Transformer for Age and Gender Estimation.*
  arXiv:2307.04616. **[V — repo/README read]**
- Kuprashevich et al. *Beyond Specialization: Assessing the Capabilities of MLLMs in Age and
  Gender Estimation.* 2024. Relevant to whether a VLM can replace specialists (§14 E10).
- **`A Call to Reflect on Evaluation Practices for Age Estimation.` arXiv:2307.04570.**
  **The methodological warning of §1.2. Read this before designing any split.**
- Cao, Mirjalili, Raschka. *Rank consistent ordinal regression (CORAL).*
- Li et al. *Unimodal-Concentrated Loss: Fully Adaptive Label Distribution Learning for Ordinal
  Regression.* arXiv:2204.00309. Fixes a real weakness of fixed-variance LDL.

**Representation**
- Zheng et al. *General Facial Representation Learning in a Visual-Linguistic Manner (FaRL).*
  CVPR 2022. arXiv:2112.03109.
- Deng et al. *ArcFace: Additive Angular Margin Loss.* CVPR 2019.
- Guo et al. *Sample and Computation Redistribution for Efficient Face Detection (SCRFD).*
  arXiv:2105.04714.
- Meng et al. *MagFace.* CVPR 2021. Quality ∝ embedding magnitude.

**Quality / uncertainty / calibration**
- Boutros et al. *CR-FIQA.* CVPR 2023. **[V — paper located]**
- Ou et al. *CLIB-FIQA: Face Image Quality Assessment with Confidence Calibration.* CVPR 2024.
- Shi & Jain. *Probabilistic Face Embeddings (PFE).* ICCV 2019. Embeddings as distributions.
- Lakshminarayanan, Pritzel, Blundell. *Simple and Scalable Predictive Uncertainty Estimation
  using Deep Ensembles.* NeurIPS 2017. arXiv:1612.01474.
- Guo et al. *On Calibration of Modern Neural Networks.* ICML 2017. Temperature scaling.
- Angelopoulos & Bates. *A Gentle Introduction to Conformal Prediction.* The distribution-free
  coverage guarantee we want for the UI (§11.4).

**Ranking**
- Burges et al. *Learning to Rank using Gradient Descent (RankNet).* ICML 2005.
- Bradley & Terry (1952); Plackett–Luce. The statistical foundation for pairwise preference.
- Cao et al. *Learning to Rank: From Pairwise Approach to Listwise Approach (ListNet).* ICML 2007.

**Bias**
- Buolamwini & Gebru. *Gender Shades.* FAT* 2018.
- *Anatomizing Bias in Facial Analysis.* arXiv:2112.06522.
- *Review of Demographic Bias in Face Recognition.* arXiv:2502.02309.

---

## 4. Relevant datasets

Full dataset cards, including everything the brief asked for per dataset, are in
[`docs/DATASETS.md`](DATASETS.md). Summary and the decisions that follow from it:

### 4.1 SCUT-FBP5500 — primary training set (obtained, verified)

Downloaded from the authors' own Google Drive link and extracted to
`data/raw/SCUT-FBP5500_v2/`. Everything below is [V] from the dataset's own README and paper.

| Property | Value |
|---|---|
| Images | 5,500 frontal, unoccluded, neutral expression, ages 15–60 |
| Subsets | Asian female 2,000 · Asian male 2,000 · Caucasian female 750 · Caucasian male 750 |
| Beauty labels | continuous mean score in [1, 5] |
| **Raters** | **60 volunteers, aged 18–27, mean age 21.6** |
| **Individual ratings** | **all 60 raters' per-image scores supplied** (`All_Ratings.xlsx`, 20.8 MB) |
| Rating protocol | web GUI, crowdsourced, four subsets rated separately, images shown in random order |
| Consistency check | ~10 % of faces re-shown; if the two ratings correlated < 0.7 the rater re-rated |
| Per-image rating σ | mostly within **[0.6, 0.7]** on the 1–5 scale |
| Inter-group rater correlation | **0.770–0.785** overall; AF 0.785, AM 0.782, CF 0.788, **CM 0.743** |
| Landmarks | **86 manual points per face** (ASM-initialised, hand-corrected) |
| Splits | official 5-fold CV (4400/1100) **and** official 60/40 (3300/2200) — both supplied |
| Provenance | Internet; some Asian faces from **DataTang** and **GuangZhouXiangSu**; some Caucasian faces from the **10k US Adults Faces Database** |
| License | **non-commercial research only** |

The `All_Ratings.xlsx` file is the most valuable asset in this dataset and is usually thrown
away. It makes rating-distribution prediction (§6.3), rater-disagreement modelling, per-rater
personalisation simulation (§6.9) and a genuine human noise ceiling all possible from data we
already have. **No experiment in this project should train on the mean score alone without at
least comparing against a distribution-based alternative.**

Three concerns to carry forward:

1. **Rater pool.** Sixty people in a ~5-year age band from one university. The "ground truth"
   is one demographic's aesthetic consensus, circa 2017. Note the CM (Caucasian male) inter-rater
   correlation is the lowest at 0.743 [V] — consistent with the paper's own observation that
   raters agree more on own-group faces. Our model will inherit all of this.
2. **Provenance and identity leakage.** Three third-party sources are named. The 10k US Adults
   Faces Database is widely redistributed, so **cross-dataset identity overlap is plausible**.
   The official splits are random image splits with no stated subject-disjointness guarantee.
   → **E1 in §14: run ArcFace over all 5,500 images and cluster; if any cluster spans the
   train/test boundary in the official split, every published number on this benchmark
   (including the ones in this document) is optimistically biased.** This is cheap to do and we
   have the embedding model on disk. It should be the *first* thing run after the baseline.
3. **Constrained imagery.** Frontal, unoccluded, neutral. The target application is arbitrary
   user directories: profiles, group shots, motion blur, sunglasses, harsh light. **The
   train/deploy domain gap is severe and is the single biggest threat to the product working
   at all.** MEBeauty (in-the-wild) is the check on this (E7).

### 4.2 Other datasets — decisions

| Dataset | Use | Decision |
|---|---|---|
| **MEBeauty** (2,550 in-the-wild, 6 ethnicities, diverse raters) | beauty, in-the-wild | **Acquire. Use as held-out cross-dataset test, not training.** Different rating scale (1–10) → compare by *rank* correlation, which is exactly UOL's argument |
| **SCUT-FBP** (500, original) | beauty | Small; useful as a third generalisation probe |
| HotOrNot | beauty, in-the-wild | Historic; used by ComboLoss; check availability/terms |
| **CelebA** (`Attractive` binary attr) | beauty, weak label | ⚠ **Binary, single-annotator-ish, notoriously noisy.** Use only for large-scale weak pretraining if at all, never for evaluation. Non-commercial |
| **FairFace** (108 k, CC BY 4.0) | age/gender/race, **fairness slicing** | **Acquire. This is the fairness evaluation backbone (§13) and the only commercially-usable face attribute set** |
| UTKFace (20 k) | age/gender | Non-commercial; useful cross-check for age |
| IMDB-Clean | age | Non-commercial; MiVOLO's training set — do not evaluate MiVOLO on it (train/test contamination) |
| AgeDB / APPA-REAL / MORPH-II | age | MORPH needs a license; APPA-REAL has *apparent* age + per-rater σ, which matches our uncertainty framing well |
| WIDER FACE | detection | CC BY-NC-ND; evaluation only |
| 300W / WFLW | landmarks | Only if we train our own landmarker (we should not) |
| LFW / CFP-FP / AgeDB-30 | recognition | Sanity-check the embedding model only |

**On combining datasets — the brief explicitly warns against doing it blindly, and it is right.**
The scales are not commensurable: SCUT-FBP5500 is 1–5 from 60 Chinese undergraduates; MEBeauty
is 1–10 from a multi-ethnic pool; CelebA is a binary flag. Naively pooling them means fitting a
single regression to three different latent constructs measured on three different instruments.

The defensible options, in increasing order of ambition:
1. **Train per-dataset, evaluate cross-dataset by rank correlation only.** Safest. Do this first.
2. **Per-dataset calibration heads on a shared backbone** — shared representation, dataset-specific
   output affine transform. Handles scale mismatch explicitly.
3. **Learn on pairwise comparisons within each dataset** and pool the *comparisons*, not the
   scores. Bradley–Terry over the union. This is scale-free by construction and is the
   theoretically clean answer (§9.2).

Option 3 is the interesting research bet and follows directly from UOL's finding. E6 in §14
tests options 1 and 3 head to head.

---

## 5. Available pretrained models

### 5.1 Already on this machine (verified by inspection)

```
~/.insightface/models/buffalo_l/     det_10g.onnx      SCRFD-10GF detector + 5 kpts
                                     w600k_r50.onnx    ArcFace IResNet-50 @ WebFace600K, 512-d
                                     2d106det.onnx     106-point 2D landmarks
                                     1k3d68.onnx       68-point 3D landmarks (→ pose)
                                     genderage.onnx    gender + age
~/.insightface/models/antelopev2/    scrfd_10g_bnkps.onnx
                                     glintr100.onnx    ArcFace IResNet-100 @ Glint360K, 512-d
                                     2d106det.onnx, 1k3d68.onnx, genderage.onnx
~/.cache/huggingface/hub/            openai/clip-vit-base-patch32
                                     timm/swin_large_patch4_window7_224.ms_in22k_ft_in1k
~/sadiq/                             LVFace/, TransFace/, insightface_wheels/
```

This is a substantial head start: detection, alignment, two tiers of face embedding, landmarks,
pose and a baseline age/gender model are all available offline right now, with GPU ONNX Runtime
(CUDA EP confirmed available). **Nothing needs to be downloaded to run experiment 001.**

### 5.2 To acquire

| Model | Why | Size | License |
|---|---|---|---|
| **MiVOLO v2** (`mivolov2_d1_384x384`) + `yolov8x_person_face.pt` | age + gender production model | ~500 MB | Apache-2.0 ⚠ detector is YOLOv8 → **AGPL-3.0, check before shipping** |
| FaRL ViT-B/16 | candidate frozen representation (E2) | ~350 MB | MIT |
| DINOv2 ViT-B/14 | candidate frozen representation (E2) | ~350 MB | Apache-2.0 |
| CR-FIQA R50 | quality reference for E9 | ~170 MB | research |
| LAION aesthetic predictor V2 | *image* aesthetics (distinct from facial) | ~4 MB | MIT |
| ComboLoss / SCUT-FBP5500 baseline weights | reproduce published FBP numbers | ~90–230 MB | research |

Note the trap in row 1: MiVOLO's own weights are Apache-2.0, but the recommended person/face
detector shipped with it is YOLOv8-based, and Ultralytics YOLOv8 is **AGPL-3.0**. We already have
SCRFD, so the clean move is to use MiVOLO's *age/gender head* with *our* detector and never
introduce the AGPL dependency. Recorded in `docs/LICENSING.md`.

---

## 6. Beauty / attractiveness modelling approaches

The brief asks for nine approaches to be investigated. Here is each one, what it buys, what it
costs, and my prior on whether it will win — recorded now so that the experiments can *falsify*
it rather than confirm it.

### 6.1 Direct score regression

Predict a scalar; minimise MSE / L1 / Huber against the mean rating.

- **Pros:** simplest; matches how every benchmark is scored; head is one linear layer.
- **Cons:** throws away the entire rating distribution; MSE against a mean is exactly the
  objective that cannot distinguish "everyone says 3.5" from "half say 2, half say 5"; regression
  to the mean at the score extremes, which is precisely where a "find the best faces" product
  operates. **This last point is underrated: our product ranks the top-N, so tail accuracy is the
  only accuracy that matters, and it is where plain MSE is weakest.**
- **Verdict:** the mandatory baseline, unlikely to be the final answer.

### 6.2 Beauty classification

Bin the score (e.g. 5 classes) and use cross-entropy.

- **Pros:** gives a probability vector → a natural (if poorly calibrated) confidence; robust to
  label noise.
- **Cons:** discards ordinality (predicting 1 when truth is 5 costs the same as predicting 4);
  bin edges are arbitrary.
- **Verdict:** only worth it as *ordinal* classification (CORAL-style cumulative-logit), which
  keeps the probability vector and restores order. Test as 6.2b.

### 6.3 Rating-distribution prediction (label-distribution learning) — **recommended primary**

Predict `p(rating = r | face)` over the rating scale; train with KL divergence against the
empirical histogram of the 60 raters. Optionally add DLDL-v2's *expectation loss* so that the
distribution's mean is also directly supervised.

This is the approach the brief intuited, and I think it is correct here, for a reason specific to
our data: **we actually have all 60 individual ratings** [V], so the target distribution is real
empirical data, not a synthetic Gaussian smeared around the mean (which is what most LDL papers
are forced to use). That is an unusual advantage and we should exploit it.

What it gives us that regression cannot:
- **Mean** → the ranking score.
- **Spread (σ, entropy)** → "is this face polarising?" — a genuinely useful, honest UI signal, and
  a first-class input to §11's confidence.
- **Shape** → bimodality detection. A face rated 2-or-5-but-never-3 is a *specific*, interesting
  category that a mean of 3.5 completely hides.
- **P(rating ≥ 4)** → a far better ranking target for "find me the best faces" than the mean,
  because it directly answers the user's actual question.

- **Cons:** needs per-rater labels (we have them for SCUT-FBP5500, but not for most datasets);
  slightly more complex head; must handle the fact that not every rater rated every image.
- **Verdict:** **primary candidate.** Cheap to add (the head is `Linear → softmax` over ~5–9 bins),
  strictly more informative than regression, and reduces to regression whenever we want a scalar.

### 6.4 Pairwise ranking

Learn `s(A) > s(B)` from pairs; RankNet / margin ranking loss; Bradley–Terry as the generative
model.

- **Pros:** **scale-free** — this is the key property. Our end product *ranks*; it does not need
  calibrated absolute scores. It sidesteps the incommensurability of SCUT-FBP5500's 1–5 and
  MEBeauty's 1–10 entirely (§4.2, option 3). UOL's cross-dataset finding [V] is direct evidence
  that order transfers where absolute scores do not. It is also the natural format for
  **user feedback** (Phase 10): "I like this one better" is a pair, and it is far easier and more
  reliable to elicit from a human than "rate this 1–5".
- **Cons:** O(n²) pairs (mitigated by sampling); produces a score with no inherent units, so the
  UI needs a calibration layer (map to a percentile — which is arguably more honest anyway); no
  absolute threshold like "above 4.0" without post-hoc calibration.
- **Verdict:** **primary candidate alongside 6.3.** These two are complementary, not competing:
  LDL for the honest per-face distribution, ranking for the cross-dataset-robust ordering.
  A combined objective (KL + λ·pairwise) is the specific thing I most want to test (E6).

### 6.5 Landmark / geometric approaches

Ratios and distances from the 86 supplied landmarks: symmetry, thirds, fifths, eye spacing,
jaw width, golden-ratio proxies.

- **Verified performance:** the dataset paper's own geometric baselines reach
  **PC 0.5948 (linear) / 0.6738 (Gaussian) / 0.6668 (SVR)** on the whole dataset [V]. Compare to
  0.8997 for a fine-tuned CNN [V].
- **Verdict:** **weak alone, but keep it.** Two reasons that survive its weak accuracy: (a) it is
  the only *interpretable* component — the UI can say "high facial symmetry", which is a real
  product feature for the "why was this selected" panel; (b) it is nearly free once landmarks are
  computed, and it may be complementary to appearance features (§6.7). Do not expect it to move
  the top-line metric.

### 6.6 Deep visual features (frozen or fine-tuned)

Either fine-tune a CNN/ViT end-to-end, or freeze a pretrained encoder and train a head.

- Fine-tuned reference points [V]: AlexNet 0.8634, ResNet-18 0.8900, ResNeXt-50 0.8997 (5-fold PC).
- **Frozen + linear head is experiment 001 and it is the pivotal experiment of Phase 4.** If a
  frozen ArcFace embedding plus ridge regression lands near 0.88–0.90, then:
  - feature extraction becomes a **one-time** cost per image, cached forever;
  - swapping the beauty model = retraining a linear head in seconds, no GPU;
  - **per-user personalisation becomes trivially cheap** (a per-user linear head over cached
    features), which is the whole of Phase 10;
  - the index does not need reprocessing when the beauty model changes — only when the *encoder*
    changes. This is an enormous engineering simplification for Phase 8's "reprocess when models
    change" requirement.
  That combination of consequences is why this experiment runs first.
- 5,500 images is a **small** dataset for full fine-tuning; heavy backbones will overfit, and the
  0.90→0.935 SOTA climb is at least partly benchmark-specific capacity. Frozen features are the
  better-conditioned starting point.

### 6.7 Geometry + visual fusion

Concatenate landmark-derived features with the deep embedding, or fuse at a deeper level.

- **Verdict:** cheap to test (concatenation + ridge is one line once both feature sets exist),
  so test it — but with a calibrated expectation. A 512-d ArcFace embedding almost certainly
  already encodes the geometry that 18 hand-picked ratios capture. My prior is a small or null
  gain. It is included because it is nearly free to falsify and because a null result is itself
  worth documenting (it justifies dropping geometry from the production path and keeping it only
  for the explanation UI).

### 6.8 Transformer-based approaches

ViT / Swin / Mamba-hybrid backbones. This is where the reported SOTA sits (VM-BeautyNet 0.9212,
VMamba-ViT 0.9261, Diff-FBP 0.932, SCAT 0.935 — all [S]).

- **Verdict:** worth testing as *frozen feature extractors* (FaRL, DINOv2, CLIP already fit the
  exp001 harness). Full custom transformer training on 5,500 images is **not** where this
  project's effort should go — the gains are within the range that split methodology alone can
  produce (§1.2), and none of it addresses the generalisation problem that will actually break
  the product.

### 6.9 Ensembles and personalisation

Covered in §10 and §15.5. The key structural decision, made now: **the general model and the
per-user model must be separate objects.** The general model predicts the crowd distribution; the
user model predicts *this user's deviation from it*. Keeping them separate means:
- the expensive general model is trained once and shared;
- the user model is a small residual model over cached features, trainable online;
- a user's preferences never contaminate the general model;
- we can always show both ("popular opinion: 3.8 · your taste: 4.4"), which is a better product
  than either alone.

MetaFBP's meta-learning framing (commonality + individuality) is exactly this decomposition and
is the reference implementation to study.

### 6.10 Summary of priors, to be tested not assumed

| Approach | Expected SCUT-FBP5500 PC | Expected cross-dataset | Cost | Prior |
|---|---|---|---|---|
| Geometric + ridge | 0.60–0.68 [V, published] | poor | trivial | keep for interpretability only |
| **Frozen ArcFace + ridge** | **0.85–0.90 (hypothesis)** | fair | trivial | **pivotal — run first** |
| Frozen CLIP + ridge | 0.80–0.88 (hypothesis) | fair | trivial | tests the "styling confound" question |
| Fine-tuned CNN | 0.89–0.90 [V] | poor | hours | reference point |
| LDL head | ≈ regression on PC, **richer output** | fair | low | **primary** |
| Pairwise ranking | ≈ regression on rank metrics | **best** | low | **primary** |
| Ensemble | +0.01–0.02 | best | medium | final production system |

The single number to watch is not PC on SCUT-FBP5500 — it is **Spearman on MEBeauty for a model
that never saw MEBeauty.** That is the number that predicts whether the product works on a user's
photo directory.

---

## 7. Age estimation approaches

| Approach | Description | Uncertainty output | Notes |
|---|---|---|---|
| Regression | direct scalar | none | Simple; suffers at distribution tails |
| Classification (1-year bins) | ~100 classes | full distribution | Ignores ordinality |
| **DEX / expectation over softmax** | classify, take expectation | distribution → σ | Long-standing strong baseline; free variance estimate |
| **Ordinal (CORAL)** | K−1 binary "older than k?" | monotone CDF | Rank-consistent by construction |
| **LDL / DLDL-v2** | Gaussian-smoothed target + expectation loss | full distribution | Same machinery as §6.3 → **one head design serves both age and beauty** |
| Mean-variance / MWR | predict μ and σ | explicit σ | Directly usable in §11 |
| MiVOLO | multi-input transformer, face+body | via ensembling | **Chosen production model** |

**Decision:** use **MiVOLO** off the shelf. Do not train an age model. Rationale: it is
Apache-2.0 [V], reports 4.22–4.24 MAE on IMDB-clean and 3.65 on Lagenda [V], and no realistic
in-house effort on our data budget beats that. Our work on age is limited to:

1. **Independent benchmarking** on our own data (E4) — never on IMDB-clean, which MiVOLO trained on.
2. **Uncertainty**: MiVOLO gives a point estimate. We add σ via test-time augmentation
   (horizontal flip + small crop jitter, 4–8 passes) and calibrate the resulting interval against
   held-out data (§11). TTA is cheap, needs no retraining, and gives an honest spread.
3. **Per-group error reporting** on FairFace (§13). A global MAE hides the failure modes; the UI
   should widen the stated interval for groups where we measure it to be wider.

The DLDL-v2 insight — that age and attractiveness can share one distribution-learning head — is
worth noting for §10.4's multi-task option, but it is a later optimisation, not a v1 decision.

---

## 8. Gender / category classification

Technically the easiest task in the system (97–99 % on benchmarks [V]) and the one with the most
design traps. The modelling recommendation is short: **use MiVOLO's gender head** (99.46 % on
IMDB-clean [V]), with InsightFace `genderage` already on disk as a cross-check and a second
ensemble member.

The design work is in how it is used:

1. **Language.** The system predicts *perceived presentation*, not identity. The UI must say
   "presenting as" or "appears". This is not politeness, it is accuracy: the model has no access
   to identity and was trained on annotator perceptions.
2. **Soft by default.** Expose it as a weighted preference (`gender: female, importance: 20%`),
   as the brief's own example does. A hard filter on a 98 %-accurate classifier silently discards
   ~2 % of true matches, biased toward androgynous presentation and toward the demographic groups
   where the classifier is weakest — a compounding, invisible failure.
3. **Confidence-gated strictness.** If the user *does* want a hard filter, apply it only above a
   confidence threshold and report the count of excluded-but-uncertain faces
   ("142 matches · 8 excluded as low-confidence"). The user can then choose to see them. This
   turns an invisible failure into a visible, actionable one.
4. **Measure and publish per-group accuracy** on FairFace (§13). Gender Shades [S] is the standing
   warning that aggregate accuracy conceals large disparities.
5. **Never store predicted race as a user-facing filter.** We compute it *only* for internal
   fairness auditing (§13), it stays out of the query language and out of the UI. Predicting race
   to *measure our own bias* is defensible; offering it as a search facet is not, and the two are
   separated by an explicit config flag that is off in the app and on in the eval harness.

---

## 9. Ranking approaches

The end product is a ranking system, so this section describes the actual output layer of the
whole project.

### 9.1 Two distinct ranking problems

They are often conflated and must not be:

- **(A) Ranking faces by predicted attractiveness** — a learned model over face appearance.
- **(B) Ranking results by *match to the user's stated query*** — a scoring function combining age
  fit, gender fit, attractiveness, quality and confidence, with user-supplied weights.

(A) is machine learning. (B) is a transparent, configurable scoring function that must be
**explainable** — the UI has to say *why* a result ranked where it did, and no learned black box
can do that credibly. Conflating them (e.g. learning (B) end-to-end) would destroy the
explainability the brief asks for in Phase 11's detail view. They stay separate.

### 9.2 For (A): learning to rank

| Method | Formulation | Fit here |
|---|---|---|
| Pointwise regression | predict score, sort | Baseline. Ignores that only order matters |
| **Pairwise (RankNet / Bradley–Terry)** | `P(A>B) = σ(s(A) − s(B))` | **Best fit.** Scale-free, matches user feedback format, transfers across datasets |
| Listwise (ListNet / Plackett–Luce) | model the permutation | More data-hungry; use only if we get list-level feedback |
| LambdaRank / LambdaMART | gradients weighted by NDCG change | Optimises top-of-list directly — **very relevant, since we only ever show top-N** |

Bradley–Terry deserves emphasis because it unifies three separate needs: it is the generative
model for pairwise comparisons, it converts our 60 per-rater SCUT-FBP5500 ratings into ~millions
of within-rater pairwise comparisons (a much larger effective training set than 5,500 scalars),
and it is the natural model for Phase 10 like/dislike feedback. One formulation, three uses.

**Metrics for (A):** pairwise accuracy, Spearman ρ, Kendall's τ, and **NDCG@k** — with the
argument that NDCG@k is the metric that actually correlates with product quality, because the
user sees `k` results and nothing else. A model with mediocre global ρ but excellent top-100
precision is a *better product* than the reverse, and only NDCG@k and precision@k will tell us so.

### 9.3 For (B): the query scoring function

Relevance is a transparent weighted combination, kept in configuration
(`configs/query/scoring.yaml`), never hard-coded:

```
relevance(face, query) = Σ_c  w_c · match_c(face, query) · conf_c(face)
                       ─────────────────────────────────────────────────
                                       Σ_c  w_c
```

with per-criterion match functions, each in [0,1]:

- **age** — trapezoidal membership over the requested range, soft shoulders so 36 does not fall
  off a cliff when the user asked 25–35, shoulder width scaled by the *predicted uncertainty* of
  that face's age. A face with age 35±2 and one with 35±9 should not match a 25–35 query equally.
- **gender** — predicted probability of the requested class.
- **attractiveness** — `P(rating ≥ threshold)` from the LDL head (§6.3), which is better than a
  thresholded mean because it accounts for the model's own spread.
- **quality** — from §2.4; also acts as a gate.
- **hard filters** — applied *before* scoring, and only where the user asks for them.

Multiplying each criterion's match by its confidence is the design decision that makes the whole
thing behave sensibly: an uncertain criterion contributes little in either direction rather than
contributing a confident-looking wrong answer. It also gives the UI its explanation for free —
each term in the sum is a row in the "why this result" panel.

**Everything about this is configuration**, including the membership function shapes. The brief's
"keep ranking logic configurable rather than hard-coding arbitrary weights" is satisfied by
making the scoring function a declarative document that the API returns alongside results, so the
UI can render the exact arithmetic that produced the ranking.

---

## 10. Ensemble approaches

### 10.1 What to ensemble

- **Multiple representations** — ArcFace-R50, ArcFace-R100, CLIP, FaRL, geometry. Diverse
  *inputs* are usually a better source of ensemble gain than diverse random seeds.
- **Multiple objectives** — regression head + LDL head + ranking head on the *same* frozen
  features. Nearly free (three small heads, one forward pass) and genuinely decorrelated because
  the losses differ.
- **Multiple seeds / bagging** — the classic deep-ensemble recipe; also our best uncertainty
  estimator (§11.2).
- **Test-time augmentation** — flip, small crop jitter; cheap, and yields a spread that doubles
  as an uncertainty signal.

### 10.2 How to combine

| Method | Notes |
|---|---|
| Simple averaging | Baseline; surprisingly hard to beat |
| Weighted average | Weights fitted on validation; risk of overfitting with few members |
| **Stacking** | Meta-learner over member predictions. **Must be fitted on out-of-fold predictions** or it silently overfits — the most common ensembling bug |
| Learned fusion (gating) | A small net that weights members *per sample* (e.g. trust geometry more when quality is high). Elegant, data-hungry |
| Multi-task single model | One backbone, several heads. Not an ensemble but shares the benefit; cheapest at inference |

### 10.3 The discipline

The brief's warning — "do not assume that adding more models automatically improves the system" —
is the operative constraint. Each additional member costs inference time on every image in a
potentially very large directory, and the marginal PC gain from member 4 onward is typically
under 0.005. The rule for this project:

> **A model earns its place in the production ensemble only if it improves the *cross-dataset*
> ranking metric (MEBeauty Spearman) by more than its added latency costs, measured, not assumed.
> Gains on SCUT-FBP5500 alone are not sufficient grounds for inclusion.**

That criterion is deliberately harsh and will probably reduce the production system to two or
three members. That is the correct outcome for a system that must scan large directories.

### 10.4 Multi-task alternative

One frozen backbone → heads for {beauty distribution, age distribution, gender, quality}. Cheapest
possible inference, and the tasks plausibly share structure (DLDL-v2 unified age and
attractiveness under one formulation). The catch is that our labels come from *different datasets*
— we have no single corpus labelled for all four — so multi-task training needs either
partial-label masking or a pseudo-labelling step, both of which add failure modes. **Deferred:
revisit after E2–E7 establish whether frozen features are sufficient.** If they are, multi-task
training is unnecessary, because the "shared backbone" already exists and is frozen.

---

## 11. Confidence and uncertainty

The brief asks the system to distinguish: strong prediction · weak prediction · poor-quality
image · ambiguous face · model disagreement. Those are **five different things**, and the central
design point of this section is that they need **separate signals**, not one blended number.
Collapsing them loses exactly the information that makes the UI honest.

### 11.1 A taxonomy that maps to the five requirements

| Signal | Source | Answers |
|---|---|---|
| **Image / face quality** | §2.4 composite (blur, size, pose, exposure, detector conf, embedding norm) | *"poor-quality image"* |
| **Aleatoric — inherent ambiguity** | spread of the predicted LDL distribution (§6.3) | *"ambiguous face — humans genuinely disagree about this one"* |
| **Epistemic — model ignorance** | disagreement across ensemble members / TTA | *"model disagreement"* / *"weak prediction — out of distribution"* |
| **Out-of-distribution** | distance from training-set feature distribution (Mahalanobis / kNN in embedding space) | *"this face is unlike anything I was trained on"* |
| **Calibrated interval** | conformal prediction on a held-out calibration set | *"age 28, 90 % interval [24, 33]"* |

The aleatoric/epistemic split is the crux, and it is what makes attractiveness different from
age. For age there is a true answer and uncertainty is mostly epistemic. For attractiveness
**there is no true answer** — the irreducible aleatoric spread *is the phenomenon*. A face where
raters split 50/50 is not a model failure; it is a real property of the face, and it is arguably
more interesting to a user than a face everyone rates 3.5. **Only a distribution-predicting model
can report this** (§6.3), which is the strongest single argument for choosing LDL over regression.

### 11.2 Recommended methods, in implementation order

1. **Deep ensembles** — the strongest general-purpose method; "deep ensembles generally outperform
   MC dropout due to more decorrelated inference models" [S]. With frozen features and linear
   heads, an ensemble of 5–10 heads costs essentially nothing (§6.6). This is a *free* benefit of
   the frozen-feature architecture and reinforces that decision.
2. **Test-time augmentation** — flip + jitter; cheap; applies to off-the-shelf models like MiVOLO
   that we cannot ensemble by retraining.
3. **Conformal prediction** — gives distribution-free coverage guarantees ("90 % of true values
   fall in this interval") from a calibration set. Search results are consistent that conformal
   "approximates the desired coverage level best for all methods, regardless of the initial
   coverage they obtain" [S]. **This is what the UI should display**, because it is the only
   method whose stated confidence means what a user thinks it means.
4. **Temperature scaling** for the classification heads — one parameter, fitted on validation,
   removes most miscalibration.
5. **MC dropout** — cheaper than ensembles but weaker, and it degrades the base prediction [S].
   Low priority given that ensembling is nearly free here.

### 11.3 Turning signals into a UI number

The brief's example output is the right target:

```
Estimated age:    28        Confidence: 91%
Attractiveness:   4.1 / 5   Confidence: 74%
Face quality:     62%
```

Concretely, per face:

```
age:            μ, conformal 90% interval, confidence = f(interval width)
attractiveness: full distribution over the rating scale,
                mean, P(≥4), σ_aleatoric (rater disagreement),
                σ_epistemic (ensemble disagreement)
gender:         calibrated probability
quality:        composite score + specific warnings ("heavy blur", "extreme pose")
ood:            flag when the face is far from the training distribution
```

The UI shows a simple number; the detail view exposes the decomposition, and the API always
returns all of it. **Confidence must be derived from the calibrated interval, not from a raw
softmax**, which is the specific mistake the brief warns against.

### 11.4 Honesty requirements (non-negotiable, and they belong in code)

- Every attractiveness number carries its provenance: *"predicted rating on the SCUT-FBP5500
  scale (60 raters, aged 18–27, 2017)"*. Not a footnote — attached to the value in the API
  response.
- Attractiveness is displayed with its distribution/spread, never as a bare scalar.
- Low-confidence and low-quality results are visually marked, not silently ranked.
- The word "beauty" should not appear unqualified in the UI. "Predicted attractiveness rating" is
  accurate; "beauty score" implies measurement.
- The system says *"this face is predicted to be rated X by that group"* and never
  *"this face is X attractive"*. The grammatical difference is the entire ethical difference, and
  it is cheap to get right if it is decided now — and expensive to retrofit later, because it
  changes API field names, not just copy.

---

## 12. Dataset and model licensing

Full register with clause-level detail: [`docs/LICENSING.md`](LICENSING.md). Summary and the
decision it forces:

### 12.1 The blocking constraint

| Asset | License | Commercial | Attribution | Redistribute |
|---|---|---|---|---|
| SCUT-FBP5500 | non-commercial research only [V] | ❌ | cite ICPR 2018 paper | ❌ |
| CelebA | non-commercial research only [V] | ❌ | cite | ❌ |
| WIDER FACE | CC BY-NC-ND 4.0 [V] | ❌ | required | ❌ (no derivatives) |
| UTKFace | non-commercial [V] | ❌ | cite | ❌ |
| IMDB-WIKI / IMDB-Clean | IMDb non-commercial terms [V] | ❌ | — | ❌ |
| MORPH | academic free / commercial paid via UNCW [V] | 💰 | per agreement | ❌ |
| **FairFace** | **CC BY 4.0** [V] | ✅ | **required** | ✅ |
| **MiVOLO** code + weights | **Apache-2.0** [V] | ✅ | NOTICE file | ✅ |
| InsightFace **code** | MIT [V] | ✅ | — | ✅ |
| InsightFace **weights** | research only [V] | ❌ | — | ❌ |
| YuNet (OpenCV Zoo) | Apache-2.0 | ✅ | — | ✅ |
| YOLOv5 / YOLOv8 (Ultralytics) | **AGPL-3.0** ⚠ | viral | — | source disclosure |
| CLIP (OpenAI) | MIT | ✅ | — | ✅ |
| DINOv2 | Apache-2.0 | ✅ | — | ✅ |
| FaRL | MIT | ✅ | — | ✅ |

### 12.2 What this means, plainly

**A beauty model trained on SCUT-FBP5500 cannot be used commercially.** Nor can anything using
InsightFace's released weights. There is no clever way around this: the restriction attaches to
the trained model as a derivative of the data.

Two coherent paths, and the project should pick one *consciously and now*:

**Path A — research / personal use (recommended default).**
Use everything above under its research terms. The system is for the user's own images, not a
product. Document the restriction prominently. This is the assumed path, and everything in the
roadmap works under it.

**Path B — commercially clean track (kept viable, not built yet).**
- Detection: **YuNet** (Apache-2.0) or SCRFD *code* with weights we train ourselves on a
  permissive corpus.
- Age/gender: **MiVOLO** (Apache-2.0), with the residual IMDb-provenance question flagged.
- Embeddings: train ArcFace ourselves on a permissively-licensed corpus, or use a licensed
  commercial model (InsightFace sells one [V]).
- Beauty: **collect our own ratings** — which, given §1.1, is arguably the *better* product
  anyway, because our own rater pool can be demographically appropriate to the users, and we
  would own the individual ratings needed for §6.3 and §11.
- Never touch: SCUT-FBP5500, CelebA, WIDER FACE, UTKFace, IMDB-*.

The repository supports this by making license a first-class, machine-readable field on every
model and dataset config, so `--commercial-safe` can mechanically exclude non-compliant
components rather than relying on someone remembering.

### 12.3 Privacy and biometric law

Face embeddings are **biometric data**. Depending on jurisdiction (GDPR Art. 9 special-category
data; Illinois BIPA; Texas CUBI; and similar), processing them can require consent, notice,
retention limits and deletion rights. Even for a local personal tool, the engineering
implications are cheap to honour now and expensive to retrofit:

- **All processing stays local by default.** No image or embedding leaves the machine. No
  telemetry. This is the single most important privacy property and it is free.
- **Deletion must actually delete** — removing an image from the index must purge its embeddings,
  crops, cached features and thumbnails, not just an index row.
- **Embeddings are re-identifiable**, so the database is sensitive even without the source images.
  Treat `artifacts/` and the index as sensitive at rest.
- **No dataset redistribution.** `data/raw/` is `.gitignore`d, permanently. The repo ships
  *download scripts*, never data.
- Provide an explicit "delete everything" path and document exactly what it removes.

---

## 13. Known limitations and biases

Recorded here so they can be measured (§14 E11) rather than discovered by a user.

### 13.1 In the attractiveness labels themselves

1. **Rater demographics.** 60 volunteers aged 18–27 (mean 21.6), from one Chinese university [V].
   The model learns *that* group's aesthetic consensus as of 2017 — not "beauty".
2. **Own-group bias, measured in the data.** Inter-rater correlation is higher for Asian faces
   than Caucasian ones, lowest for Caucasian males at 0.743 [V]. The paper itself notes this is
   consistent with psychological findings on own-group perception. Model accuracy will
   correspondingly differ by group.
3. **Class imbalance.** 4,000 Asian vs. 1,500 Caucasian faces [V]; no Black, South Asian,
   Hispanic, Middle Eastern or Southeast Asian subjects **at all**. The model has literally never
   seen these groups. **Predictions on them are extrapolation and should be flagged as OOD, not
   silently returned** — this is a concrete use for §11's OOD signal, not a hypothetical one.
4. **Constrained imagery.** Frontal, unoccluded, neutral expression, ages 15–60 [V]. Real
   directories are none of these.
5. **Confounds.** Ratings encode makeup, hairstyle, photo quality, lighting and image resolution,
   not just facial structure. A model may be partly learning *photography quality*. This is
   testable: correlate predicted beauty with our quality score; a high correlation is evidence of
   the confound (E11c). It also has a UI consequence — "attractiveness" that is partly
   "well-photographed" should be described as such.
6. **Score compression.** Ratings concentrate near the middle (the paper fits a two-component
   Gaussian mixture [V]); the extremes — which is exactly where a top-N ranking product
   operates — are sparsely supported.

### 13.2 In the other components

- **Gender:** binary only; higher error for darker-skinned subjects and for women in commercial
  systems (Gender Shades [S]); no non-binary representation in any available training data.
- **Age:** MAE is not uniform — worse at both extremes and for under-represented groups. A single
  "±4 years" claim in the UI would be misleading; the interval must be per-face (§11).
- **Detection:** lower recall on dark skin, extreme pose, small faces, occlusion. **A missed face
  is an invisible failure** — the user cannot tell the difference between "no matching faces" and
  "the detector didn't see them".
- **Recognition embeddings:** demographic differentials in FR accuracy are well documented
  (arXiv:2502.02309); this affects our dedup and "more like this" features.

### 13.3 Compounding

These stack multiplicatively through the pipeline. A dark-skinned face is more likely to be
missed by the detector; if detected, more likely to be mis-gendered; if classified, its
attractiveness prediction is pure extrapolation because no such faces were in the beauty
training set. **The end-to-end disparity is larger than any single component's.** This is why
§14 E11 evaluates fairness *end-to-end on FairFace*, not per component.

### 13.4 The honest framing

> This system predicts how a specific, narrow group of human raters would have rated a face.
> It does not measure beauty. Beauty is not a property of faces that can be measured.

That sentence belongs in the README, in the UI's about panel, and in the API response metadata.
It is not a disclaimer to be minimised — it is the accurate description of what was built, and
stating it plainly is what makes the rest of the system defensible.

---

## 14. Recommended experiments

Ordered by information gained per unit of compute. Every experiment has a falsifiable question, a
decision it informs, and a stopping condition. Full specs in `experiments/`.

### Tier 0 — validity of the benchmark itself (do these first)

**E0 · Split methodology audit.** Reproduce the "unified subject-exclusive split" finding
(arXiv:2307.04570) on our own data. *Question:* how much of published performance is split
leakage? *Decision:* whether to trust any published number. *Cost:* low.

**E1 · Identity leakage in SCUT-FBP5500.** ⚠ **Highest priority.** Run ArcFace (`w600k_r50`, on
disk) over all 5,500 images, cluster at a verification threshold, and check whether any identity
cluster spans the train/test boundary of the official splits. *Question:* are the official splits
subject-disjoint? *Decision:* if not, we build our own identity-disjoint splits and **all our
numbers become non-comparable to published ones — which we then report honestly**. *Cost:* ~2
minutes of GPU. **There is no excuse for not running this before anything else.**

### Tier 1 — baselines (Phase 4)

**E2 · Frozen representation comparison.** ✅ **Implemented as `exp001`.** ArcFace-R50 vs.
ArcFace-R100 vs. CLIP ViT-B/32 vs. 86-point geometry vs. concatenations, each with an identical
ridge head, on the official 5-fold and 60/40 splits. *Question:* which frozen representation best
predicts attractiveness, and how close does frozen+linear get to fine-tuned? *Decision:* the
entire feature-store architecture (§6.6). *Cost:* ~10 GPU-minutes total.

**E3 · Fine-tuned reference.** ResNet-18 / ResNeXt-50 fine-tuned, to reproduce the published
0.8900 / 0.8997 [V] and confirm our harness is sound. *Question:* is our evaluation code
correct? *Decision:* validates every subsequent number. **An experiment harness that cannot
reproduce a known result cannot be trusted to measure a new one.**

**E4 · Off-the-shelf age/gender benchmark.** MiVOLO vs. InsightFace `genderage` on FairFace and
UTKFace (never on IMDB-clean). Report MAE by decade and by group. *Decision:* production age model.

### Tier 2 — beauty modelling (Phase 2)

**E5 · Crop/alignment sensitivity.** Sweep crop margin (0 %, 10 %, 25 %, 40 %), alignment on/off,
input resolution. *Question:* how much does preprocessing matter relative to model choice?
*Decision:* freezes the crop protocol for the feature store. My prior is that this matters more
than backbone choice, which would be a cheap and important finding.

**E6 · Objective comparison — the core beauty experiment.** On the best frozen features from E2,
compare: (a) MSE regression · (b) ordinal classification · (c) **LDL against the real 60-rater
histograms** · (d) pairwise ranking (Bradley–Terry over within-rater comparisons) · (e) combined
KL + λ·pairwise. Evaluate on regression *and* ranking *and* calibration metrics. *Question:*
which objective produces the best ranking and the most useful uncertainty? *Decision:* the
production beauty head.

**E7 · Cross-dataset generalisation.** ⚠ **The most important experiment in the project.** Train
on SCUT-FBP5500, evaluate on MEBeauty and SCUT-FBP-500 by Spearman/Kendall only (scales differ).
*Question:* does anything we learn transfer off-benchmark? *Decision:* go/no-go on the whole
approach. **If nothing transfers, the product does not work, and no amount of in-benchmark PC
fixes it.**

### Tier 3 — system

**E8 · Detector evaluation** on realistic imagery (recall on small/blurred/profile/occluded faces;
speed at scale). **E9 · Quality-signal validation** — does the free composite (§2.4) track CR-FIQA?
**E10 · Can a VLM replace specialists?** Zero/few-shot age, gender, attractiveness from a
multimodal model, as a "one model for everything" check the brief asks us not to assume away.
**E11 · Fairness audit** — end-to-end per-group performance on FairFace: (a) detection recall,
(b) gender accuracy, (c) age MAE, (d) attractiveness score *distributions* by group, and
(e) correlation between predicted attractiveness and predicted image quality (the confound test).
**E12 · Uncertainty methods** — ensembles vs. TTA vs. MC dropout vs. conformal, scored by coverage
and by whether the confidence actually predicts error. **E13 · Ensemble selection** under the §10.3
rule. **E14 · Personalisation** — simulate users with the 60 individual raters we already have;
measure how many labels a new user must give before a personal model beats the population model.
This is a Phase-10 question we can answer *today* with data already on disk, which makes it
unusually cheap.

### The order

```
E1 → E2(exp001) → E3 → E5 → E6 → E7 → E4 → E9 → E11 → E12 → E13 → E14 → E8 → E10
└ validity ┘   └──── beauty modelling ────┘   └──── system & responsibility ────┘
```

E1 first because it can invalidate everything downstream. E7 is the gate: nothing ships until it
passes.

---

## 15. Proposed technical architecture

### 15.1 Revision to the brief's architecture

The brief's pipeline is close to right. Three changes, each justified by the research above:

1. **Split "attribute models" into `encode` (expensive, cached) and `predict` (cheap, replaceable).**
   If E2 confirms frozen features work, the encoder runs **once per face, ever**. Beauty, age,
   gender and personalisation heads then run over cached features in milliseconds. This makes
   "reprocess when models change" (Phase 8) almost always unnecessary — only an *encoder* change
   forces reprocessing, and heads can be swapped, retrained or personalised for free. This is the
   single highest-leverage architectural decision in the project and it falls straight out of E2.
2. **Quality assessment must run *after* alignment, not before it**, because pose and crop quality
   are only measurable once landmarks exist. It also feeds the confidence stage, so it is not a
   simple gate — a low-quality face is *scored with wide intervals*, not discarded, because
   discarding it is an invisible failure (§13.2).
3. **Add an explicit uncertainty stage**, and make confidence a first-class column in the
   database — not something recomputed at query time. The query engine multiplies match by
   confidence (§9.3), so it must be indexed.

```
  image directory
        │
        ▼
  ┌─────────────────┐  walk · hash · dedup (perceptual + exact) · EXIF · corrupt-file quarantine
  │  DISCOVERY      │  incremental: skip unchanged (path, mtime, size, content-hash)
  └────────┬────────┘
           ▼
  ┌─────────────────┐  SCRFD det_10g → bbox, 5 kpts, det score      [swappable: YuNet, RetinaFace]
  │  DETECTION      │  multiple faces per image, all retained
  └────────┬────────┘
           ▼
  ┌─────────────────┐  similarity transform → 112×112 (ArcFace canonical)
  │  ALIGNMENT      │  + 106-pt 2D and 68-pt 3D landmarks → pose
  └────────┬────────┘  ** crop protocol is versioned; features are keyed on it **
           ▼
  ┌─────────────────┐  blur · exposure · size · pose · det-conf · embedding-norm
  │  QUALITY        │  → composite score + specific warnings
  └────────┬────────┘
           ▼
  ╔═════════════════╗  ArcFace 512-d  [+ CLIP / FaRL if E2 says they help]
  ║  ENCODE  (slow) ║  ** computed once, cached forever, keyed by (model_version, crop_version) **
  ╚════════╤════════╝
           ▼
  ┌─────────────────┐  beauty distribution head · age head · gender head · geometry features
  │  PREDICT (fast) │  ** cheap, replaceable, retrainable, personalisable **
  └────────┬────────┘
           ▼
  ┌─────────────────┐  aleatoric (rating spread) · epistemic (ensemble) · OOD · conformal intervals
  │  UNCERTAINTY    │
  └────────┬────────┘
           ▼
  ┌─────────────────┐  SQLite (metadata, predictions, confidence) + vector index (embeddings)
  │  STORE          │  model_version stamped on every row → full reproducibility
  └────────┬────────┘
           ▼
  ┌─────────────────┐  hard filters → soft scoring → rank → paginate
  │  QUERY + RANK   │  configurable weights; returns per-criterion explanation
  └────────┬────────┘
           ▼
  ┌─────────────────┐  + personal preference model (separate, per-user, over cached features)
  │  API  →  UI     │
  └─────────────────┘
```

### 15.2 Module boundaries

```
src/facet/
  data/        dataset loaders, official splits, identity-disjoint split construction
  models/      thin adapters behind protocols:
                 FaceDetector      → detect(image)  -> [Detection]
                 FaceAligner       → align(image, det) -> crop, landmarks, pose
                 FeatureExtractor  → encode(crops) -> np.ndarray        (the expensive, cached part)
                 AttributeHead     → predict(features) -> Prediction+uncertainty  (cheap)
                 QualityAssessor   → assess(...) -> QualityReport
  training/    heads, losses (MSE, ordinal, KL/LDL, Bradley-Terry), training loops
  evaluation/  metrics (regression, ranking, classification, calibration), fairness slicing
  pipeline/    discovery, dedup, batching, incremental indexing, caching, error handling
  query/       filters, scoring functions, ranking, pagination
  utils/       config, seeding, hashing, logging, versioning
```

Every model sits behind a protocol so it is swappable by config. Nothing outside `models/` may
import `insightface`, `onnxruntime` or `torch` directly — that keeps the licensing surface (§12.2)
and the swap-a-model requirement both mechanically enforceable rather than aspirational.

### 15.3 Storage

- **SQLite** for images, faces, predictions, runs, model versions, user feedback. Single-file,
  zero-ops, transactional, more than adequate for millions of faces, and trivially backed up.
- **A separate vector index** (FAISS — already installed in several envs here) for embeddings.
- **Feature cache** as memory-mapped `.npy` shards keyed by `(encoder_version, crop_version)`,
  so heads can be retrained instantly without re-encoding.
- **Every prediction row carries `model_version` and `config_hash`.** This is what makes
  "reproduce a prediction" and "reprocess only what changed" both possible, and it is the concrete
  mechanism behind the brief's Phase 13 requirement.

### 15.4 Query language

```yaml
filters:                       # hard — applied before scoring
  min_quality: 0.4
  min_face_size: 80
preferences:                   # soft — weighted, confidence-multiplied
  age:            {range: [25, 35], weight: 0.30, softness: 5}
  gender:         {value: female,   weight: 0.20, min_confidence: 0.6}
  attractiveness: {target: high, threshold: 4.0, weight: 0.50}
ranking:
  sort_by: relevance
  limit: 100
```

Gender is a *preference* with a confidence floor, not a filter, for the reasons in §8.

### 15.5 Personalisation (designed for now, built in Phase 10)

Two separate models, never merged:

```
final_score(face, user) = population_score(face) + α(n) · user_residual(face, user)
```

`user_residual` is a small model (linear, or a low-rank adapter) over the **same cached features**,
trained on that user's likes/dislikes/comparisons. `α(n)` grows with the number of labels the user
has given, so a new user gets the population model and a heavy user gets their own taste — a
clean cold-start solution requiring no special-casing.

The frozen-feature architecture is what makes this cheap: no re-encoding, training is seconds on
CPU, and E14 tells us the label budget before we build any of the UI for it. **Pairwise feedback
("prefer A over B") is the preferred elicitation format** — easier for humans, more reliable than
absolute ratings, and it plugs straight into the Bradley–Terry machinery from §9.2.

### 15.6 Development roadmap

| Stage | Deliverable | Gate to proceed |
|---|---|---|
| **0 ✅** | Research report; repo skeleton; env; dataset acquired | — |
| **1 ✅** | `exp001` — frozen features + linear heads, reproducible harness | ✅ done: frozen CLIP+ArcFace reaches PC 0.9398, beating fine-tuned baselines |
| **2 ◑** | E1 identity-leakage audit; identity-disjoint splits | ✅ audit done: ~5 % leakage, +0.003 PC impact. Disjoint splits still TODO |
| **3** | E3 fine-tuned reference | Reproduce published 0.89/0.90 → harness trusted |
| **4** | E5, E6 — crop protocol frozen, beauty objective chosen | LDL/ranking beats regression on ranking metrics |
| **5** | **E7 cross-dataset** | **⚠ GO/NO-GO: does it transfer to MEBeauty?** |
| **6** | E4, E9, E12 — age/gender/quality/uncertainty selected | Calibrated intervals achieve nominal coverage |
| **7** | E11 fairness audit | Disparities measured and documented |
| **8** | Inference pipeline + incremental indexer | Indexes 100 k images without re-doing work |
| **9** | Query + ranking engine | Brief's example queries work |
| **10** | API | Stable contract, versioned |
| **11** | UI | Phase-11 feature list |
| **12** | E14 personalisation | Beats population model within a realistic label budget |
| **13** | Optimisation, batching, ONNX/TensorRT | Throughput target on A6000 |

Stage 5 is the real gate. Everything before it is research; everything after assumes the research
succeeded. If E7 fails, the honest response is to change the product — for example, drop the
absolute attractiveness claim entirely and ship a *purely personalised* ranker trained only on the
user's own feedback, which needs no transferable population model at all. **That fallback is a
good product too**, and knowing it exists means a negative result at stage 5 is informative rather
than fatal.

### 15.7 Engineering requirements (Phase 13)

- **Config over code.** Model paths, thresholds, weights, crop parameters in YAML. No magic
  numbers in source.
- **Versioning.** Every model has a version; every prediction records the version and config hash
  that produced it.
- **Reproducibility.** Every experiment records dataset + version, split methodology, model,
  backbone, hyperparameters, loss, augmentation, seed, hardware, timings, metrics, checkpoint path
  and config hash — written as a JSON manifest per run.
- **Separation.** `research/` ≠ `training/` ≠ `inference/` ≠ `pipeline/` ≠ `api/` ≠ `ui/`.
- **Tests.** Metrics against known values; loaders against known counts; pipeline against a
  fixture directory including deliberately corrupt files.
- **Honesty in the data model.** Estimates are typed as estimates. A `Prediction` object carries
  its uncertainty and its provenance, so it is *structurally impossible* to render a bare
  attractiveness scalar without also having its distribution and its source-of-truth caveat
  available. §11.4 is enforced by the type system rather than by discipline.

---

## Appendix A — verified reference numbers

Use these as the reproduction targets for E3. All [V], read from arXiv:1801.06345.

**SCUT-FBP5500, 5-fold cross-validation (average over folds):**

| Model | PC | MAE | RMSE |
|---|---|---|---|
| AlexNet | 0.8634 | 0.2651 | 0.3481 |
| ResNet-18 | 0.8900 | 0.2419 | 0.3166 |
| ResNeXt-50 | **0.8997** | **0.2291** | **0.3017** |

**SCUT-FBP5500, 60 % train / 40 % test:**

| Model | PC | MAE | RMSE |
|---|---|---|---|
| AlexNet | 0.8298 | 0.2938 | 0.3819 |
| ResNet-18 | 0.8513 | 0.2818 | 0.3703 |
| ResNeXt-50 | **0.8777** | **0.2518** | **0.3325** |

**Geometric features + shallow predictors (whole dataset, 10-fold):**

| Predictor | PC | MAE | RMSE |
|---|---|---|---|
| Linear regression | 0.5948 | 0.4289 | 0.5531 |
| Gaussian regression | 0.6738 | 0.3914 | 0.5085 |
| SVR | 0.6668 | 0.3898 | 0.5132 |

**Gabor appearance features + shallow predictors (whole dataset):**

| Sampling | Predictor | PC | MAE | RMSE |
|---|---|---|---|---|
| 86-keypoints | Gaussian regression | 0.7472 | 0.3554 | 0.4599 |
| 86-keypoints | SVR | 0.6691 | 0.3891 | 0.5065 |
| 64UniSample | Gaussian regression | 0.6764 | 0.4014 | 0.5177 |
| 64UniSample | SVR | 0.8065 | 0.3976 | 0.5126 |

⚠ The last row is anomalous: PC 0.8065 with MAE 0.3976 is internally inconsistent with the rest
of the table (better correlation, worse error than the 86-keypoint GR row). Treat as a probable
typo in the source and do not use as a target.

**Human agreement ceiling (Table III):**

| Rater group | CF | AF | CM | AM | All faces |
|---|---|---|---|---|---|
| Female labelers | 0.785 | 0.800 | 0.747 | 0.793 | **0.785** |
| Male labelers | 0.791 | 0.795 | 0.763 | 0.797 | **0.781** |
| All labelers | 0.788 | 0.785 | 0.743 | 0.782 | **0.770** |

**Read this table next to the one above it.** Models correlate with the crowd mean at 0.90+;
humans correlate with it at 0.77. §1.1 explains why this is not superhuman perception.

---

## Appendix B — sources

Primary sources read in full or in substantial part **[V]**:
- arXiv:1801.06345 — SCUT-FBP5500 (full PDF)
- SCUT-FBP5500_v2 `README.txt` and dataset contents (local, from authors' Drive)
- github.com/HCIILAB/SCUT-FBP5500-Database-Release — README, official split files
- github.com/WildChlamydia/MiVOLO — README, benchmarks, license
- github.com/deepinsight/insightface — model_zoo README, license notes
- arXiv:2409.00603 — Uncertainty-oriented Order Learning
- laion.ai/blog/laion-aesthetics + LAION-AI/aesthetic-predictor
- FairFace, UTKFace, CelebA, WIDER FACE, MORPH license statements

Secondary **[S]** (search summaries — each is a verification task in §14):
SCAT / Diff-FBP / MD-Net / VM-BeautyNet / VMamba-ViT reported PCs · CR-FIQA vs. FaceQAN vs.
MagFace rankings · DLDL-v2 MORPH 1.97 MAE · CLIP > DINOv2 > FaRL frozen-encoder ordering ·
Gender Shades error rates · deep-ensembles vs. MC-dropout vs. conformal comparisons ·
arXiv:2307.04570's subject-exclusive-split finding.
