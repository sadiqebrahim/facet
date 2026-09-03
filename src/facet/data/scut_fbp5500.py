"""SCUT-FBP5500 loader.

Everything this module exposes is documented in docs/DATASETS.md.

The important design point: this loader exposes the FULL 60-rater rating histogram for
every image, not just the mean. Per docs/RESEARCH.md section 1.1, the mean is a
low-variance target that hides the phenomenon we actually care about (rater
disagreement), so the mean is offered as a convenience, not as the only target.

License: the SCUT-FBP5500 dataset is NON-COMMERCIAL RESEARCH ONLY.
Cite: Liang, Lin, Jin, Xie, Li. SCUT-FBP5500. ICPR 2018.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

# Filename prefix -> (race, gender). Verified against the dataset README.
SUBGROUPS: dict[str, tuple[str, str]] = {
    "AF": ("Asian", "female"),
    "AM": ("Asian", "male"),
    "CF": ("Caucasian", "female"),
    "CM": ("Caucasian", "male"),
}

RATING_LEVELS: tuple[int, ...] = (1, 2, 3, 4, 5)
N_RATERS = 60
N_IMAGES = 5500
N_LANDMARKS = 86


@dataclass(frozen=True)
class Split:
    """One train/test partition, as filename lists."""

    name: str
    train: list[str]
    test: list[str]

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Split({self.name!r}, train={len(self.train)}, test={len(self.test)})"


class MissingLandmarksError(FileNotFoundError):
    """Raised for landmark files that exist but contain no usable points.

    SCUT-FBP5500 ships exactly one such file (CM152.pts is 4 bytes: a header claiming
    0 points and no data). Surfacing this explicitly rather than letting it become a
    shape mismatch downstream is the difference between a documented data-quality fact
    and a confusing crash.
    """


def _read_pts(path: Path) -> np.ndarray:
    """Read a SCUT-FBP5500 .pts file.

    Format (verified by inspection): int32 little-endian point count, then count*2
    float32 (x, y) pairs in image pixel coordinates. Files are 692 bytes = 4 + 86*2*4.
    """
    raw = path.read_bytes()
    if len(raw) < 8:
        raise MissingLandmarksError(f"{path}: {len(raw)} bytes, contains no landmark data")
    n = struct.unpack("<i", raw[:4])[0]
    pts = np.frombuffer(raw[4:], dtype="<f4").reshape(-1, 2)
    if n != len(pts):
        raise ValueError(f"{path}: header says {n} points, found {len(pts)}")
    return pts.astype(np.float32)


class ScutFbp5500:
    """Access to images, ratings, landmarks and the official splits."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        if not (self.root / "Images").is_dir():
            raise FileNotFoundError(
                f"{self.root} does not look like SCUT-FBP5500_v2 (no Images/ directory)"
            )
        self.images_dir = self.root / "Images"
        self.landmarks_dir = self.root / "facial landmark"
        self.splits_dir = self.root / "train_test_files"

    # ---------------------------------------------------------------- labels

    @cached_property
    def ratings(self) -> pd.DataFrame:
        """Raw per-rater ratings: columns Rater, Filename, Rating, original Rating.

        330,000 rows = 5,500 images x 60 raters. Every rater rated every image.
        """
        df = pd.read_excel(self.root / "All_Ratings.xlsx")
        df["Filename"] = df["Filename"].astype(str)
        return df

    @cached_property
    def labels(self) -> pd.DataFrame:
        """One row per image: mean/std/histogram of the 60 ratings plus subgroup metadata.

        Columns:
            filename, race, gender, mean, std, n_ratings,
            p1..p5   - normalised rating histogram (the LDL target)
            p_ge4    - P(rating >= 4), the "would be rated attractive" target
            entropy  - rater disagreement, in nats
        """
        g = self.ratings.groupby("Filename")["Rating"]
        hist = np.zeros((len(g), len(RATING_LEVELS)), dtype=np.float64)
        names = []
        means, stds, counts = [], [], []
        for i, (name, vals) in enumerate(g):
            names.append(name)
            arr = vals.to_numpy()
            counts.append(len(arr))
            means.append(arr.mean())
            stds.append(arr.std(ddof=1))
            for j, level in enumerate(RATING_LEVELS):
                hist[i, j] = (arr == level).sum()

        probs = hist / hist.sum(axis=1, keepdims=True)
        safe = np.where(probs > 0, probs, 1.0)
        ent = -(probs * np.log(safe)).sum(axis=1)

        df = pd.DataFrame(
            {
                "filename": names,
                "mean": means,
                "std": stds,
                "n_ratings": counts,
                "p_ge4": probs[:, 3:].sum(axis=1),
                "entropy": ent,
            }
        )
        for j, level in enumerate(RATING_LEVELS):
            df[f"p{level}"] = probs[:, j]

        prefix = df["filename"].str[:2]
        df["race"] = prefix.map(lambda p: SUBGROUPS[p][0])
        df["gender"] = prefix.map(lambda p: SUBGROUPS[p][1])
        df["subgroup"] = prefix
        return df.set_index("filename").sort_index()

    def histogram_matrix(self, filenames: list[str]) -> np.ndarray:
        """(N, 5) normalised rating histograms, ordered to match `filenames`."""
        return self.labels.loc[filenames, [f"p{l}" for l in RATING_LEVELS]].to_numpy(np.float32)

    def target(self, filenames: list[str], kind: str = "mean") -> np.ndarray:
        """Fetch a 1-D regression target ordered to match `filenames`."""
        if kind not in {"mean", "std", "p_ge4", "entropy"}:
            raise ValueError(f"unknown target {kind!r}")
        return self.labels.loc[filenames, kind].to_numpy(np.float32)

    # ---------------------------------------------------------------- splits

    def _read_split_file(self, path: Path) -> list[str]:
        names = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if line:
                names.append(line.split()[0])
        return names

    @cached_property
    def splits(self) -> dict[str, Split]:
        """The dataset's own official splits.

        `cv1`..`cv5`   - official 5-fold cross-validation (4400 train / 1100 test)
        `split6040`    - official 60% train / 40% test (3300 / 2200)

        NOTE: these are random IMAGE splits. Whether they are subject-disjoint is
        unverified - see experiment E1 in docs/RESEARCH.md section 14. Treat any metric
        computed on them as provisional until E1 has run.
        """
        out: dict[str, Split] = {}
        cv_root = self.splits_dir / "5_folders_cross_validations_files"
        for k in range(1, 6):
            d = cv_root / f"cross_validation_{k}"
            out[f"cv{k}"] = Split(
                name=f"cv{k}",
                train=self._read_split_file(d / f"train_{k}.txt"),
                test=self._read_split_file(d / f"test_{k}.txt"),
            )
        d6040 = self.splits_dir / "split_of_60%training and 40%testing"
        out["split6040"] = Split(
            name="split6040",
            train=self._read_split_file(d6040 / "train.txt"),
            test=self._read_split_file(d6040 / "test.txt"),
        )
        return out

    @cached_property
    def filenames(self) -> list[str]:
        """All 5,500 image filenames, sorted."""
        return sorted(p.name for p in self.images_dir.glob("*.jpg"))

    def image_path(self, filename: str) -> Path:
        return self.images_dir / filename

    # ------------------------------------------------------------ landmarks

    def landmarks(self, filename: str) -> np.ndarray:
        """(86, 2) hand-annotated landmarks in image pixel coordinates."""
        return _read_pts(self.landmarks_dir / (Path(filename).stem + ".pts"))

    def all_landmarks(
        self, filenames: list[str] | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """(N, 86, 2) landmarks and an (N,) boolean validity mask.

        Images with unusable landmark files are filled with NaN and marked invalid.
        Exactly one image in the release is affected (CM152.jpg). Callers decide whether
        to drop or impute - this method will not silently do either.
        """
        names = filenames or self.filenames
        out = np.full((len(names), N_LANDMARKS, 2), np.nan, dtype=np.float32)
        valid = np.ones(len(names), dtype=bool)
        for i, n in enumerate(names):
            try:
                out[i] = self.landmarks(n)
            except (MissingLandmarksError, ValueError):
                valid[i] = False
        return out, valid

    # ------------------------------------------------------------- summary

    def summary(self) -> dict:
        """Facts used in docs/DATASETS.md - recomputed so the docs stay honest."""
        lab = self.labels
        return {
            "n_images": len(self.filenames),
            "n_labelled": len(lab),
            "n_raters": int(self.ratings["Rater"].nunique()),
            "n_ratings": int(len(self.ratings)),
            "subgroup_counts": lab["subgroup"].value_counts().to_dict(),
            "mean_score": float(lab["mean"].mean()),
            "score_range": [float(lab["mean"].min()), float(lab["mean"].max())],
            "mean_rater_std": float(lab["std"].mean()),
            "median_rater_std": float(lab["std"].median()),
            "splits": {k: (len(v.train), len(v.test)) for k, v in self.splits.items()},
        }
