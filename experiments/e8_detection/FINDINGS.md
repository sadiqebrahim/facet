# E8 — Detector evaluation on real scenes

E11 measured detection recall at **0.9998** with essentially no demographic disparity — but on
FairFace, whose images are pre-cropped and face-centred. I flagged that as close to a best
case. This runs the same detector on **WIDER FACE validation** (3,226 images, 39,697 faces,
median face height **20 px**, 68 % under 32 px) and asks what a directory scanner actually
loses.

`docs/RESEARCH.md §13.2`: a missed face is an **invisible** failure — the user cannot tell
"no matching faces in this directory" from "the detector never saw them". Recall is a product
metric here, not an infrastructure one.

Run: 2026-09-03 · RTX A6000 · IoU 0.5, greedy highest-score-first matching · `results.json`

## Result 1 — recall depends overwhelmingly on face size

Best configuration (`buffalo_l`, det_size 1024, unpadded):

| face height | n | recall |
|---|---:|---:|
| tiny < 16 px | 15,266 | 0.351 |
| small 16–32 | 11,184 | 0.749 |
| medium 32–64 | 7,553 | 0.873 |
| large 64–128 | 3,191 | **0.940** |
| huge ≥ 128 | 1,918 | 0.908 |

Overall recall is **0.641**, but that number is dominated by the 39 % of annotations under
16 px, which are not useful search results even when found. On faces ≥ 32 px — the ones a
product would actually return — recall is **0.895**.

Note the non-monotonicity: **≥128 px scores *lower* than 64–128 px** (0.908 vs 0.940). That
is the same frame-filling failure E5 found on portraits, showing up inside a scene benchmark.

## Result 2 — occlusion and blur are the real killers

| axis | recall |
|---|---:|
| occlusion: none | 0.805 |
| occlusion: partial | 0.494 |
| occlusion: heavy | **0.312** |
| blur: clear | 0.893 |
| blur: normal | 0.894 |
| blur: heavy | **0.470** |
| pose: typical | 0.643 |
| pose: atypical | 0.587 |

Heavy occlusion costs **0.49 recall** and heavy blur **0.42**. Pose costs surprisingly little
(0.056) — modern anchor-based detectors handle profile faces far better than the folklore
suggests.

One result I checked before believing: "extreme illumination" appeared to *help* (0.771 vs
0.633). It is a **size confound** — extreme-illumination faces have median height 24 px vs
20 px for normal, so they are simply bigger. Reported as a confound, not a lighting effect.

## Result 3 — E5's padding decision does not generalise, and E8's own winner is worse

E5 chose `pad_frac = 0.25` because SCRFD's recall on frame-filling portrait crops collapses
to 46 % without it. Padding shrinks faces relative to the frame — the wrong direction for a
scene full of 20-pixel faces. Measured at det_size 1024:

| face height | pad 0.00 | pad 0.25 | Δ |
|---|---:|---:|---:|
| tiny < 16 px | 0.3508 | 0.1949 | **−0.156** |
| small 16–32 | 0.7487 | 0.6875 | −0.061 |
| medium 32–64 | 0.8725 | 0.8650 | −0.008 |
| large 64–128 | 0.9398 | 0.9304 | −0.009 |
| **huge ≥ 128** | 0.9077 | **0.9447** | **+0.037** |

Padding helps **exactly one bucket** — the frame-filling one E5 was fixing — and hurts every
other, worst on the smallest faces. The two regimes want opposite things.

## Result 4 — ⚠ the config sweep's winner is blind to portraits

This is the finding that matters most, and it only appeared because both regimes were tested:

| config | **portrait recall** | scene recall_all | scene ≥32px | img/s |
|---|---:|---:|---:|---:|
| d640 pad0.25 *(E5 production)* | 0.9975 | 0.3916 | 0.8570 | 60.6 |
| d1024 pad0.00 *(E8 sweep winner)* | **0.0000** | 0.6580 | 0.8964 | 31.3 |
| d1024 pad=auto | 0.3550 | 0.6588 | 0.8995 | 29.6 |
| **det=auto pad=auto** *(new)* | **0.9975** | **0.6588** | **0.8995** | 30.0 |

**The configuration that won E8's own sweep detects zero faces in tight portrait crops.**

The cause is an interaction I had not considered: InsightFace resizes every image so its
longest side equals `det_size`, which means small images are **upscaled**. A 350×350 portrait
at det_size 1024 is blown up ~3×, the face exceeds SCRFD's scale range, and recall goes to
zero. At det_size 640 the same image works perfectly.

Had E8 been run only on WIDER FACE — the obvious thing to do for a detection experiment — its
recommendation would have silently broken every avatar, ID photo and pre-cropped image in a
user's directory. E5's finding and E8's finding are each correct *and each wrong as a global
default*.

## The fix: adapt both axes

Implemented in `InsightFaceDetector` and now the default:

- **`det_size="auto"`** — keep a detector at 640 and at 1024, route per image by longest side
  (≤ 700 px → 640, else 1024). Small images are never upscaled beyond the detector's range.
- **`pad_frac="auto"`** — detect unpadded first; retry padded only when nothing was found or
  the top detection fills the frame. The retry is rare on scenes, so it costs ~nothing there,
  and it fully recovers the portrait case.

The result is **strictly dominant**: it matches E5's portrait recall (0.9975) *and* the best
scene recall (0.6588 / 0.8995) at essentially the same speed as the d1024 configs.

## Also found: a cache that would have returned stale data

The detection cache introduced in E5 keyed on `{dataset}__{pack}.npz` and **did not include
the detector version**. Changing the detector — exactly what this experiment concluded we
should do — would have silently reused the old boxes. Fixed: the key now includes
`detector.version`. A cache that confidently returns wrong data is worse than no cache.

## Decisions taken

1. **Adopt `det_size="auto"`, `pad_frac="auto"` as the detector default.** Strictly better on
   both regimes.
2. **Report recall on faces ≥ 32 px as the headline** (0.895), with overall recall (0.641) as
   context. Sub-16 px faces inflate the denominator without being useful results.
3. **Surface detection limits in the UI.** At 0.31 recall under heavy occlusion and 0.47 under
   heavy blur, "no matching faces" is frequently wrong. The indexer should report faces-found
   per image so a user can tell a sparse directory from a failing detector.
4. **`antelopev2`'s detector is bit-identical to `buffalo_l`'s** (recall 0.640673 and 8.129
   det/img on both). Drop it from the registry as a detection option; keep it only for its
   R100 embedding.
5. **E11's clean detection result stands but is now correctly bounded**: no demographic
   disparity on face-centred crops; scene-level recall is a separate, much harder problem, and
   the per-group behaviour of *that* is still unmeasured (WIDER FACE has no demographic labels).

## Threats to validity

- WIDER FACE has no demographic annotations, so this measures *what* we miss but not *whom*.
  Combining E8's difficulty axes with E11's demographic axes would need a set with both.
- Only IoU 0.5 and the default detector threshold (0.5) were evaluated; a lower threshold
  trades precision for recall and might suit an indexer that can afford false positives.
- Scene throughput (~30 img/s) was measured on an A6000 with sequential single-image calls;
  batching would improve it substantially and is a Phase 8 concern.
- The portrait regime is represented by SCUT-FBP5500 crops, which are uniform 350×350; real
  avatars vary more.

## Reproduce

```bash
python scripts/download_wider_face.py
python scripts/run_e8_detection.py
```
