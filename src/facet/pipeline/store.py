"""Append-only feature store.

The expensive half of docs/RESEARCH.md 15.1. Embeddings are written once and read many
times, so this is a flat binary file plus a small JSON sidecar, memory-mapped for reads.

Shards are keyed by **(encoder version, crop version)**. That is not tidiness: E5 changed the
crop margin from 0.00 to 0.25 and E8 changed the detector, and features computed under
different preprocessing are not comparable. Keying on both means a protocol change starts a
new shard instead of silently mixing incompatible vectors - the same failure the detection
cache had before E8 fixed it.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)


class FeatureStore:
    def __init__(self, root: str | Path, encoder_version: str, crop_version: str,
                 dim: int, dtype: str = "float32"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.key = f"{_slug(encoder_version)}__{_slug(crop_version)}"
        self.bin = self.root / f"{self.key}.bin"
        self.meta_path = self.root / f"{self.key}.json"
        self.dim, self.dtype = dim, dtype
        if self.meta_path.exists():
            m = json.loads(self.meta_path.read_text())
            if m["dim"] != dim or m["dtype"] != dtype:
                raise ValueError(
                    f"feature store {self.key} holds dim={m['dim']} dtype={m['dtype']}, "
                    f"caller expects dim={dim} dtype={dtype}"
                )
            self.count = m["count"]
        else:
            self.count = 0
            self._write_meta(encoder_version, crop_version)
        self._mm = None

    def _write_meta(self, encoder_version: str, crop_version: str) -> None:
        self.meta_path.write_text(json.dumps({
            "encoder_version": encoder_version, "crop_version": crop_version,
            "dim": self.dim, "dtype": self.dtype, "count": self.count,
        }, indent=2))

    def append(self, arr: np.ndarray) -> np.ndarray:
        """Append (N, dim) rows; returns the row indices assigned."""
        arr = np.ascontiguousarray(arr, dtype=self.dtype)
        if arr.ndim != 2 or arr.shape[1] != self.dim:
            raise ValueError(f"expected (N,{self.dim}), got {arr.shape}")
        start = self.count
        with open(self.bin, "ab") as fh:
            fh.write(arr.tobytes())
        self.count += len(arr)
        m = json.loads(self.meta_path.read_text())
        m["count"] = self.count
        self.meta_path.write_text(json.dumps(m, indent=2))
        self._mm = None                       # invalidate the cached map
        return np.arange(start, self.count)

    @property
    def matrix(self) -> np.ndarray:
        """Memory-mapped (count, dim) view. Never loads the whole store into RAM."""
        if self.count == 0:
            return np.zeros((0, self.dim), dtype=self.dtype)
        if self._mm is None:
            self._mm = np.memmap(self.bin, dtype=self.dtype, mode="r",
                                 shape=(self.count, self.dim))
        return self._mm

    def take(self, rows) -> np.ndarray:
        rows = np.asarray(rows, dtype=np.int64)
        return np.asarray(self.matrix[rows])

    def __len__(self) -> int:
        return self.count
