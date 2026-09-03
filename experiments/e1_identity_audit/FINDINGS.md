# E1 — Identity-leakage audit of SCUT-FBP5500's official splits

The highest-priority experiment in [`docs/RESEARCH.md §14`](../../docs/RESEARCH.md),
because it determines whether any number computed on this benchmark can be trusted.

## Why

SCUT-FBP5500 ships official 5-fold and 60/40 splits, but they are random **image** splits.
The dataset is assembled from three third-party sources (DataTang, GuangZhouXiangSu, and
the 10k US Adults Faces Database), and the last is widely redistributed. If the same person
appears in both train and test, a model can memorise identities instead of learning
attractiveness. The age-estimation literature found exactly this (arXiv:2307.04570): under
unified subject-exclusive splits, the reported gains of specialised architectures largely
vanished.

## Method

Cosine similarity over cached ArcFace R100 (`glintr100`) embeddings for all 5,500 images;
pairs above 0.5 (a conventional verification operating point) joined into connected
components; components that straddle a train/test boundary counted.

## Result: leakage confirmed in every official split

```
pairs above 0.50             : 172
identity clusters            : 5332
clusters with >1 image       : 164   (332 images)
max pairwise similarity      : 0.9976   <- near-duplicate images exist
```

| Split | Straddling clusters | Leaked test images | % of test |
|---|---:|---:|---:|
| cv1 | 57 | 58 | 5.27 % |
| cv2 | 51 | 51 | 4.64 % |
| cv3 | 59 | 59 | 5.36 % |
| cv4 | 52 | 53 | 4.82 % |
| cv5 | 55 | 55 | 5.00 % |
| split6040 | 83 | 87 | 3.95 % |

**Roughly 5 % of every official test set shares an identity with its training set.** The
dataset does not have 5,500 distinct subjects, and its splits are not subject-disjoint.

## But the impact on the headline metric is small

Fitting the exp001 ArcFace+CLIP model and scoring the leaked and clean test subsets
separately:

| | PC | MAE |
|---|---:|---:|
| leaked test images | 0.9440 | 0.1798 |
| clean test images | 0.9370 | 0.1824 |
| all test images | 0.9398 | 0.1823 |

Leaked images are predicted slightly better (+0.007 PC), and removing them moves the
headline from **0.9398 → 0.9370, a bias of +0.003 PC**.

## Verdict

Both halves matter and neither should be reported without the other:

1. **The leakage is real and must be disclosed.** Every published SCUT-FBP5500 number,
   including ours, is computed on splits that are not subject-disjoint and is therefore
   optimistically biased.
2. **It is not large enough to explain exp001's result.** +0.003 PC does not account for
   frozen features outperforming fine-tuned CNNs by ~0.04.

Had this not been run, the exp001 result would have rested on an unexamined assumption.
It cost about two minutes because the embeddings were already cached.

## Consequences

- Build **identity-disjoint splits** (group by the clusters found here) and report both
  official and subject-disjoint numbers going forward. Official splits stay for
  comparability with published work; disjoint splits are the honest number.
- Reuse this clustering for the pipeline's **duplicate detection** (Phase 8) — near-
  duplicates at cosine 0.998 are exactly what the indexer must collapse.
- Treat all published SCUT-FBP5500 leaderboard numbers as upper bounds.

## Reproduce

```bash
python scripts/run_e1_identity_audit.py --threshold 0.5
```
