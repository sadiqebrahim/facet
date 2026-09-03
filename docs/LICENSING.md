# License, privacy and responsible-use register

Phase 14 deliverable. Every model and dataset the project may touch, with its terms.
**[V]** = read from the primary source.

---

## 1. Models

| Asset | Code license | Weights license | Commercial | Attribution | Redistribute |
|---|---|---|---|---|---|
| InsightFace (SCRFD, ArcFace, genderage, landmarks) | MIT [V] | **research only** [V] | ❌ weights | — | ❌ weights |
| — `buffalo_l`, `antelopev2` (on disk) | — | research only [V] | ❌ | — | ❌ |
| **MiVOLO / MiVOLO-v2** | Apache-2.0 [V] | Apache-2.0 [V] | ✅ | NOTICE | ✅ |
| ⚠ `yolov8x_person_face.pt` (MiVOLO's detector) | **AGPL-3.0** (Ultralytics) | AGPL-3.0 | ⚠ viral | — | source disclosure |
| YuNet (OpenCV Zoo) | Apache-2.0 | Apache-2.0 | ✅ | — | ✅ |
| YOLO5Face / YOLOv8-face | AGPL-3.0 | AGPL-3.0 | ⚠ viral | — | source disclosure |
| CLIP (OpenAI) | MIT | MIT | ✅ | — | ✅ |
| DINOv2 | Apache-2.0 | Apache-2.0 | ✅ | — | ✅ |
| DINOv3 | restrictive research license ⚠ | ⚠ | ❌ | — | ❌ |
| FaRL | MIT | MIT | ✅ | — | ✅ |
| CR-FIQA | research | research | ❌ | — | ❌ |
| LAION aesthetic predictor V2 | MIT | MIT | ✅ | — | ✅ |
| SCUT-FBP5500 baseline weights | research | research | ❌ | cite | ❌ |
| **Any model we train on SCUT-FBP5500** | ours | **inherits non-commercial** | ❌ | cite | ❌ |

### The AGPL trap
MiVOLO's weights are Apache-2.0, but its recommended detector is Ultralytics YOLOv8 (AGPL-3.0).
AGPL is network-viral: shipping a service that uses it can require releasing your source.
**We already have SCRFD.** Use MiVOLO's age/gender head with our own detector and never introduce
the AGPL dependency. Enforced by keeping detector selection in config with a `license:` field.

### The derivative-work rule
A model trained on non-commercial data is a derivative of that data. Training our own beauty head
on SCUT-FBP5500 produces **non-commercial weights**. There is no way around this short of
different data.

---

## 2. Datasets

| Dataset | License | Commercial | Train on it? | Redistribute |
|---|---|---|---|---|
| SCUT-FBP5500 | non-commercial research only [V] | ❌ | research only | ❌ |
| CelebA | non-commercial research only [V] | ❌ | research only | ❌ |
| WIDER FACE | CC BY-NC-ND 4.0 [V] | ❌ | ❌ (no derivatives) | ❌ |
| UTKFace | non-commercial [V] | ❌ | research only | ❌ |
| IMDB-WIKI / IMDB-Clean | IMDb non-commercial terms [V] | ❌ | research only | ❌ |
| MORPH | academic free / commercial paid via UNCW [V] | 💰 | per agreement | ❌ |
| **FairFace** | **CC BY 4.0** [V] | ✅ | ✅ | ✅ with attribution |
| MEBeauty | **unverified — check before use** | ? | ? | ? |
| LAGENDA | released with MiVOLO; cite [V] | ? | ? | ? |
| AVA | unclear (dpchallenge.com sourced) | ? | ? | ? |

**Required citations if used:** SCUT-FBP5500 → Liang et al., ICPR 2018 · FairFace →
Kärkkäinen & Joo · MEBeauty → Lebedeva et al. · MiVOLO → Kuprashevich & Tolstykh · InsightFace →
Deng et al. (ArcFace), Guo et al. (SCRFD).

---

## 3. Two paths

**Path A — research / personal (current default).** Everything above under research terms. Not a
product. Restriction stated in the README and in the UI.

**Path B — commercially clean (not built).**
- Detection: YuNet (Apache-2.0), or SCRFD *code* with self-trained weights.
- Age/gender: MiVOLO (Apache-2.0) — with the residual note that its training data derives from
  IMDb; a lawyer's question, not an engineer's.
- Embeddings: self-trained on a permissive corpus, or a purchased commercial license
  (InsightFace sells one).
- Beauty: **our own collected ratings.** Given the rater-pool bias documented in RESEARCH.md §1.1,
  this is the better product regardless of licensing — we would control the rater demographics and
  own the individual ratings that §6.3 and §11 need.
- Excluded entirely: SCUT-FBP5500, CelebA, WIDER FACE, UTKFace, IMDB-*.

Every entry in `configs/models/*.yaml` carries `license:` and `commercial_use:` so a
`--commercial-safe` run can mechanically refuse non-compliant components.

---

## 4. Privacy and biometric data

Face embeddings are **biometric identifiers**. GDPR Art. 9 (special-category data), Illinois BIPA,
Texas CUBI and comparable regimes may require consent, notice, retention limits and deletion —
including for images of people who never interacted with the system.

**Engineering commitments (cheap now, expensive to retrofit):**
1. **Local-only by default.** No image, crop, embedding or prediction leaves the machine. No
   telemetry, no remote inference.
2. **Deletion means deletion.** Removing an image purges its embeddings, crops, thumbnails, cached
   features and index rows — not just a database row.
3. **The index is sensitive at rest.** Embeddings are re-identifiable without the source images.
4. **No dataset redistribution.** `data/` is permanently gitignored; the repo ships download
   scripts, never data.
5. **Retention limits** are configurable and enforced.
6. **A documented purge path** the user can actually run.

---

## 5. Responsible use

Enforced in the product, not just documented:

1. **Estimates are labelled as estimates.** A `Prediction` carries its uncertainty and provenance;
   the type system makes it structurally impossible to render a bare attractiveness scalar.
2. **Attractiveness is never presented as objective.** Every such value is accompanied by
   *"predicted rating on the SCUT-FBP5500 scale — 60 raters aged 18–27, 2017"*.
3. **Predicted race is never a user-facing facet.** Computed only for internal fairness auditing,
   behind a config flag that is off in the application and on in the eval harness.
4. **Out-of-distribution faces are flagged, not silently scored.** The beauty training set contains
   no Black, South Asian, Hispanic, Middle Eastern or SE Asian subjects; predictions on such faces
   are extrapolation and must say so.
5. **Measured disparities are published** in the repo (E11), not buried.
6. **Gender is a soft, confidence-gated preference**, and excluded-but-uncertain counts are shown
   so filtering failures are visible rather than silent.

### Scope statement
This system predicts how a specific, narrow group of human raters would have rated a face. It does
not measure beauty. Beauty is not a property of faces that can be measured. Any use that presents
its output as objective assessment of a person is a misuse of the system.
