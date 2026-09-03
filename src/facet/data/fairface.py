"""FairFace loader - the fairness audit set (experiment E11).

FairFace is balanced by construction across seven race groups, which is what a bias audit
needs: an imbalanced audit set confounds "the model is worse on group X" with "there are
fewer examples of group X".

Two things to keep straight about the labels:

* They are annotator-PERCEIVED race, gender and age bucket, not self-reported identity.
* We use them ONLY to slice our own error rates. Predicted race never becomes a user-facing
  search facet (docs/LICENSING.md section 5.3). Measuring your own bias requires the
  attribute; offering it as a filter is a different thing.

License: CC BY 4.0 - commercial use permitted, attribution required. The only face dataset
in this project without a non-commercial restriction.
Cite: Karkkainen & Joo, FairFace (WACV 2021).
"""
from __future__ import annotations

from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

AGE_BUCKETS = ("0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+")
#: Midpoints for turning bucket labels into a number an age MAE can be computed against.
#: The open-ended top bucket is assigned 75, which is a convention, not a measurement.
AGE_MIDPOINTS = np.array([1.0, 6.0, 14.5, 24.5, 34.5, 44.5, 54.5, 64.5, 75.0])
GENDERS = ("Male", "Female")
RACES = (
    "East Asian", "Indian", "Black", "White",
    "Middle Eastern", "Latino_Hispanic", "Southeast Asian",
)

#: Race groups with no representation at all in SCUT-FBP5500 (Asian + Caucasian only).
NOT_IN_SCUT = ("Indian", "Black", "Middle Eastern", "Latino_Hispanic")


class FairFace:
    def __init__(self, root: str | Path, split: str = "val"):
        self.root = Path(root)
        self.images_dir = self.root / f"{split}_images"
        if not self.images_dir.is_dir():
            raise FileNotFoundError(
                f"{self.images_dir} not found - run scripts/download_fairface.py"
            )

    @cached_property
    def labels(self) -> pd.DataFrame:
        df = pd.read_csv(self.root / "labels.csv")
        df["age_bucket"] = df["age"].map(lambda i: AGE_BUCKETS[int(i)])
        df["age_mid"] = df["age"].map(lambda i: AGE_MIDPOINTS[int(i)])
        df["gender_label"] = df["gender"].map(lambda i: GENDERS[int(i)])
        df["race_label"] = df["race"].map(lambda i: RACES[int(i)])
        df["group"] = df["race_label"] + "/" + df["gender_label"]
        df["in_scut_distribution"] = ~df["race_label"].isin(NOT_IN_SCUT)
        return df.set_index("file").sort_index()

    @cached_property
    def filenames(self) -> list[str]:
        return list(self.labels.index)

    def image_path(self, name: str) -> Path:
        return self.images_dir / name

    def summary(self) -> dict:
        lab = self.labels
        return {
            "n_images": len(lab),
            "race_counts": lab["race_label"].value_counts().to_dict(),
            "gender_counts": lab["gender_label"].value_counts().to_dict(),
            "age_bucket_counts": lab["age_bucket"].value_counts().to_dict(),
            "n_outside_scut_distribution": int((~lab["in_scut_distribution"]).sum()),
        }
