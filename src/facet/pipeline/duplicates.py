"""Duplicate detection: exact files and near-duplicate faces.

Two different problems the brief lists together:

* **Exact duplicates** - the same bytes at two paths. Free: the content hash is already
  computed during discovery.
* **Near duplicates** - the same person in near-identical shots (burst frames, crops,
  re-saves). E1 built this machinery to audit SCUT-FBP5500's splits and found near-duplicates
  at cosine 0.998; the same clustering is what an indexer needs to avoid showing a user ten
  copies of one photo.

The face-embedding half deliberately reuses the ArcFace features already in the store, so
duplicate detection costs one matrix multiply rather than a second model.
"""
from __future__ import annotations

import numpy as np


def union_find(n: int, edges) -> np.ndarray:
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, j in edges:
        ri, rj = find(int(i)), find(int(j))
        if ri != rj:
            parent[ri] = rj
    return np.array([find(i) for i in range(n)])


def near_duplicate_groups(features: np.ndarray, threshold: float = 0.92,
                          block: int = 4096) -> np.ndarray:
    """Group rows whose cosine similarity exceeds `threshold`.

    Blocked so memory stays bounded: a full N x N matrix is 800 GB at N = 300k, which is the
    scale "very large directories" implies. Returns a group label per row.

    The default 0.92 is above a verification operating point (~0.5, E1) because the target
    here is *near-identical images*, not the same identity in different photos - a user
    usually wants both of those shown, just not the same shot twice.
    """
    n = len(features)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    F = features / np.clip(np.linalg.norm(features, axis=1, keepdims=True), 1e-9, None)
    F = F.astype(np.float32)
    edges = []
    for s in range(0, n, block):
        e = min(s + block, n)
        sims = F[s:e] @ F.T
        for local, row in enumerate(sims):
            i = s + local
            row[: i + 1] = -1.0            # upper triangle only
            for j in np.nonzero(row > threshold)[0]:
                edges.append((i, int(j)))
    return union_find(n, edges)
