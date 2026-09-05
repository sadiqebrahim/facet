# The indexing pipeline (Phase 8)

Point it at directories; get a queryable index of faces with predictions, uncertainty and
provenance. Every design choice below traces to an experiment — this is the research phase's
conclusions assembled into something that runs.

## Shape

```
directories
    │
    ├─ DISCOVERY      stat → content hash → decode        cheap-to-expensive, so an
    │                 corrupt/tiny/non-image recorded     unchanged rerun touches no pixels
    ▼
┌───────────────────────────────────────────────┐
│  INDEX PASS  (expensive, cached forever)      │   scripts/index_directory.py
│    detect      SCRFD, det_size+pad adaptive   │   E8
│    quality     composite_v2 from free signals │   E9
│    align       ArcFace template, margin 0.25  │   E5
│    encode      ArcFace ⊕ CLIP → 1024-d        │   exp001
│    store       shard keyed (encoder, crop)    │   E5/E8
└───────────────────────┬───────────────────────┘
                        ▼
┌───────────────────────────────────────────────┐
│  PREDICT PASSES  (cheap, replaceable)         │   scripts/predict_attributes.py
│    beauty      LDL ensemble + conformal + OOD │   E6, E12    ~8,400 faces/s
│    age/gender  MiVOLO, wide crop, LAZY        │   E4         ~40 faces/s
│    dupes       exact bytes + near-dup faces   │   E1
└───────────────────────────────────────────────┘
                        ▼
              SQLite index + feature store
```

The split is the point. Re-running a prediction pass after changing a head touches **no
pixels**: beauty over 1,121 cached faces takes 0.1 s, against 16 s to index the 800 images
they came from. That is `docs/RESEARCH.md §15.1`, measured.

## Usage

```bash
# index (resumable, incremental, safe to re-run)
python scripts/index_directory.py ~/Photos ~/Pictures --index facet.db --features feats/

# predictions — independently re-runnable
python scripts/predict_attributes.py --index facet.db --features feats/ --pass beauty
python scripts/predict_attributes.py --index facet.db --features feats/ --pass age --min-quality 0.4 --limit 5000
python scripts/predict_attributes.py --index facet.db --features feats/ --pass dupes
```

## What each experiment contributed

| Component | Setting | Why |
|---|---|---|
| detector | `det_size="auto"`, `pad_frac="auto"` | **E8**: the best scene config has *zero* recall on cropped portraits; both axes must adapt |
| crop | template align, margin **0.25** | **E5**: selected on cross-dataset transfer, not in-benchmark PC — the two optima disagree |
| encoder | ArcFace ⊕ CLIP, 1024-d | **exp001**: frozen features beat fine-tuned CNNs by +0.040 (**E3** confirmed against our own baselines) |
| quality | `composite_v2` | **E9**: validated by Error-vs-Reject on LFW (AUERC 0.0034, +85 % vs random) |
| beauty | LDL ensemble | **E6**: objectives tie on accuracy; LDL wins on what it *reports* |
| intervals | split-conformal + OOD gate | **E12**: raw spread is not confidence; conformal fails under domain shift |
| age/gender | MiVOLO, wide crop, lazy | **E4**: ~3× fairer than the baseline, ~190× slower |
| duplicates | exact hash + cosine > 0.92 | **E1**: built to audit splits, reused here |

## Handling the cases a happy path misses

Verified end to end on a fixture directory:

| input | outcome |
|---|---|
| corrupt JPEG (valid header, garbage payload) | `status=corrupt`, `error="decode failed"` — **recorded, not dropped** |
| flat-colour image with no face | `status=no_faces` |
| 2-byte file | skipped as `too_small` before decoding |
| `notes.txt` | ignored by extension |
| exact byte-for-byte copy | exact duplicate group |
| re-saved JPEG (quality 88) | near-duplicate group (cosine > 0.92) |
| unchanged directory, re-run | **0 processed, 7 skipped** |

Failures are rows with a status, never silent omissions — `§13.2`'s "a missed face is an
invisible failure" applies equally to a file we could not open.

## Reproducibility and re-processing

Every row carries the versions that produced it: `detector_version` on images,
`encoder_version` + `crop_version` on faces, `model_version` + `config_hash` on predictions.
Consequences:

- A **head** change re-queues only predictions (`faces_missing_prediction` finds them).
- A **crop or encoder** change starts a new feature shard rather than mixing incomparable
  vectors — the exact failure the detection cache had before E8 caught it.
- A **detector** change is visible per image, so re-detection can be scoped.

## Measured throughput (RTX A6000)

| stage | rate |
|---|---|
| index (detect + quality + align + ArcFace + CLIP) | **~50 img/s** |
| beauty over cached features | **~8,400 faces/s** |
| age/gender (MiVOLO, lazy) | **~40 faces/s** |

100k images ≈ 33 min to index; re-predicting beauty across all of them ≈ 15 s. MiVOLO is the
bottleneck by two orders of magnitude, which is why it is lazy and quality-gated.

## Honest limitations

- **Conformal intervals are calibrated on SCUT-FBP5500, not per collection.** E12 concluded
  calibration should be per-collection, but conformal needs labels and a photo directory has
  none. The OOD gate is the mitigation: on FairFace **55.7 % of faces tripped it** and had
  their numeric confidence suppressed — high, and correct, since E7/E11 showed the model is
  genuinely unreliable off-distribution. Percentile-within-collection (label-free) should be
  the primary ranking signal, not the absolute score.
- **The demographic skew E11 measured is present in this index.** Nothing here corrects it.
- Age/gender predictions exist only for the faces the lazy pass covered; absence is not a
  failed prediction and the query layer must distinguish those.
- Single-process. Batching is per-stage, not pipelined across stages, so CPU decode and GPU
  inference do not overlap.
