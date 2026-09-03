"""MEBeauty loader - the cross-dataset generalisation test set (experiment E7).

MEBeauty exists in this project for one purpose: to answer whether anything learned on
SCUT-FBP5500 transfers to images that do not look like SCUT-FBP5500. It differs from
SCUT-FBP5500 on every axis that matters:

    axis            SCUT-FBP5500              MEBeauty
    ------------    ----------------------    ---------------------------------
    imagery         frontal, neutral, posed   in-the-wild, unconstrained
    ethnicities     Asian, Caucasian only     6 groups incl. Black, Indian,
                                              Hispanic, Middle Eastern
    rating scale    1-5                       1-10
    raters          60, aged 18-27, one uni   ~300, mixed ethnicity/age/gender
    source          DataTang, 10k US Adults   Unsplash / Pixabay / Pexels

Because the SCALES DIFFER, cross-dataset performance must be compared by RANK
correlation only. Comparing MAE across a 1-5 and a 1-10 scale is meaningless. This is
exactly the argument for order-learning objectives in docs/RESEARCH.md section 6.4.

Four of the six ethnic groups here are ABSENT from SCUT-FBP5500 entirely, which makes
this dataset a direct test of the out-of-distribution claim in section 13.1.3.

License: NON-COMMERCIAL RESEARCH ONLY (stated in the dataset's own README).
Cite: Lebedeva, Guo et al. MEBeauty: a multi-ethnic facial beauty dataset in-the-wild.
Neural Computing and Applications, 2021.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

ETHNICITIES = ("asian", "black", "caucasian", "hispanic", "indian", "mideastern")
GENDERS = ("female", "male")

#: Ethnic groups with no representation at all in SCUT-FBP5500. Predictions from a
#: SCUT-trained model on these are extrapolation, not interpolation.
NOT_IN_SCUT = ("black", "hispanic", "indian", "mideastern")

RATING_MIN, RATING_MAX = 1.0, 10.0

_PATH_RE = re.compile(r"images/(female|male)/([a-z]+)/(.+)$")


@dataclass(frozen=True)
class Split:
    name: str
    train: list[str]
    test: list[str]


class MEBeauty:
    """Access to MEBeauty images, mean scores, per-rater scores and official splits.

    Keys are `"<gender>/<ethnicity>/<filename>"`, which is stable across the several
    path conventions used inside the release (absolute /home/ubuntu paths in some score
    files, ./cropped_images/... in others).
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.images_dir = self.root / "original_images"
        self.scores_dir = self.root / "scores"
        if not self.images_dir.is_dir():
            raise FileNotFoundError(f"{self.root} has no original_images/ directory")

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _key_from_path(path: str) -> str | None:
        """Normalise any of the release's path conventions to gender/ethnicity/file."""
        p = str(path).replace("\\", "/")
        m = _PATH_RE.search(p)
        if m:
            return f"{m.group(1)}/{m.group(2)}/{Path(m.group(3)).name}"
        # cropped_images/images_crop_align_mtcnn/<gender>/<ethnicity>/<file>
        parts = p.split("/")
        for i, part in enumerate(parts):
            if part in GENDERS and i + 2 < len(parts) + 1:
                try:
                    gender, eth, fname = part, parts[i + 1], parts[i + 2]
                except IndexError:
                    return None
                if eth in ETHNICITIES:
                    return f"{gender}/{eth}/{Path(fname).name}"
        return None

    # ------------------------------------------------------------------- labels

    @cached_property
    def _raw_scores(self) -> pd.DataFrame:
        return pd.read_excel(self.scores_dir / "generic_scores_all_2022.xlsx")

    @cached_property
    def labels(self) -> pd.DataFrame:
        """One row per image that exists on disk AND has a score.

        Columns: mean, std, n_ratings, gender, ethnicity, group, in_scut_distribution.
        `std` is the disagreement among the raters who actually rated that image, and is
        the MEBeauty analogue of the SCUT-FBP5500 rater spread.
        """
        df = self._raw_scores.copy()
        rater_cols = [c for c in df.columns if c not in ("mean", "image", "path")]
        df["key"] = df["path"].map(self._key_from_path)
        df = df[df["key"].notna()].copy()

        R = df[rater_cols].to_numpy(dtype=np.float64)
        df["n_ratings"] = np.isfinite(R).sum(axis=1)
        with np.errstate(invalid="ignore"):
            df["std"] = np.nanstd(R, axis=1, ddof=1)
            recomputed = np.nanmean(R, axis=1)
        # Prefer the release's own mean, fall back to recomputing it.
        df["mean_score"] = df["mean"].where(df["mean"].notna(), pd.Series(recomputed, index=df.index))

        out = df[["key", "mean_score", "std", "n_ratings"]].rename(
            columns={"mean_score": "mean"}
        )
        out = out.drop_duplicates("key").set_index("key")
        parts = out.index.to_series().str.split("/", expand=True)
        out["gender"] = parts[0].to_numpy()
        out["ethnicity"] = parts[1].to_numpy()
        out["group"] = out["gender"] + "/" + out["ethnicity"]
        out["in_scut_distribution"] = ~out["ethnicity"].isin(NOT_IN_SCUT)

        exists = [k for k in out.index if (self.images_dir / k).exists()]
        return out.loc[exists].sort_index()

    def rater_matrix(self) -> tuple[np.ndarray, list[str]]:
        """(N_images, N_raters) matrix with NaN for unrated pairs, ordered like `labels`.

        Used to compute MEBeauty's own human inter-rater ceiling, which is the only fair
        yardstick for a model evaluated on this data.
        """
        df = self._raw_scores.copy()
        rater_cols = [c for c in df.columns if c not in ("mean", "image", "path")]
        df["key"] = df["path"].map(self._key_from_path)
        df = df[df["key"].notna()].drop_duplicates("key").set_index("key")
        keys = list(self.labels.index)
        return df.loc[keys, rater_cols].to_numpy(dtype=np.float64), rater_cols

    # ------------------------------------------------------------------- splits

    def _read_split(self, fname: str) -> list[str]:
        path = self.scores_dir / fname
        keys = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            key = self._key_from_path(line.rsplit(" ", 1)[0])
            if key is not None:
                keys.append(key)
        return keys

    @cached_property
    def splits(self) -> dict[str, Split]:
        """The release's own train/val/test split, restricted to images we actually have."""
        have = set(self.labels.index)
        train = [k for k in self._read_split("train_2022.txt") if k in have]
        val = [k for k in self._read_split("val_2022.txt") if k in have]
        test = [k for k in self._read_split("test_2022.txt") if k in have]
        return {"official": Split("official", train + val, test)}

    # -------------------------------------------------------------------- access

    @cached_property
    def keys(self) -> list[str]:
        return list(self.labels.index)

    def image_path(self, key: str) -> Path:
        return self.images_dir / key

    def target(self, keys: list[str], kind: str = "mean") -> np.ndarray:
        return self.labels.loc[keys, kind].to_numpy(np.float32)

    def summary(self) -> dict:
        lab = self.labels
        return {
            "n_images": len(lab),
            "rating_scale": [RATING_MIN, RATING_MAX],
            "mean_score": float(lab["mean"].mean()),
            "score_range": [float(lab["mean"].min()), float(lab["mean"].max())],
            "mean_rater_std": float(lab["std"].mean()),
            "median_ratings_per_image": float(lab["n_ratings"].median()),
            "group_counts": lab["group"].value_counts().to_dict(),
            "ethnicity_counts": lab["ethnicity"].value_counts().to_dict(),
            "n_outside_scut_distribution": int((~lab["in_scut_distribution"]).sum()),
            "splits": {k: (len(v.train), len(v.test)) for k, v in self.splits.items()},
        }
