"""Step 8: torch Dataset over the frozen split manifest.

Every load re-verifies the manifest's SHA256 against split_manifest.sha256
before reading it -- the split must never be silently regenerated or
hand-edited; see splits.py's module docstring.

LFW's 40 LFWA+ attribute columns (the actual multi-label training target)
live in lfw_records.csv, not in split_manifest.csv -- the manifest
intentionally only carries the compact fairness/report columns -- so this
rejoins them by image_path for LFW specifically.

Unknown/missing race, gender or age_group labels (FairFace/UTKFace) are
encoded as -1, the standard ignore_index for nn.CrossEntropyLoss, rather
than being dropped or given their own class. ita_value/ita_group and the
demographic labels are exposed only via `metadata`, never `targets`: they
must never reach the model as input.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from label_encoders import load_labels

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENTS_DIR = Path(__file__).resolve().parents[2] / "experiments"
RECORDS_DIR = Path(__file__).resolve().parents[2] / "data" / "records"

IGNORE_INDEX = -1


class ManifestHashMismatchError(RuntimeError):
    pass


def load_frozen_manifest(dataset: str) -> pd.DataFrame:
    split_dir = EXPERIMENTS_DIR / dataset
    manifest_path = split_dir / "split_manifest.csv"
    saved_hash = (split_dir / "split_manifest.sha256").read_text().strip()
    actual_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if actual_hash != saved_hash:
        raise ManifestHashMismatchError(
            f"{manifest_path} does not match its frozen split_manifest.sha256 -- "
            "the split must never be silently regenerated or edited."
        )
    return pd.read_csv(manifest_path)


class FaceDataset(Dataset):
    """dataset in {'fairface', 'utkface', 'lfw'}; split in {'train','val','test'}.
    __getitem__ returns (image_tensor, targets, metadata)."""

    def __init__(self, dataset: str, split: str, transform=None):
        if dataset not in ("fairface", "utkface", "lfw"):
            raise ValueError(f"unknown dataset {dataset!r}")
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be train/val/test, got {split!r}")

        self.dataset = dataset
        self.split = split
        self.transform = transform
        self.labels = load_labels(dataset)

        manifest = load_frozen_manifest(dataset)
        df = manifest[manifest["split"] == split].reset_index(drop=True)

        if dataset == "lfw":
            attr_cols = [f"attr_{a}" for a in self.labels["attributes"]]
            records = pd.read_csv(RECORDS_DIR / "lfw_records.csv", usecols=["image_path", *attr_cols])
            df = df.merge(records, on="image_path", how="left", validate="one_to_one")
            missing = df[attr_cols].isna().any(axis=1)
            if missing.any():
                # lfw_att_40.mat (the real LFWA+ release) only covers 13,143 of
                # LFW's 13,233 images -- these ~90 rows have no attribute label
                # at all and can't serve as training examples for this head.
                # The frozen split itself is untouched; only this dataset's
                # usable rows shrink.
                print(
                    f"[lfw/{split}] {int(missing.sum())}/{len(df)} rows have no LFWA+ "
                    "attribute match (not in lfw_att_40.mat's 13,143-image coverage) -- excluded"
                )
                df = df[~missing].reset_index(drop=True)
        else:
            self._race_index = {c: i for i, c in enumerate(self.labels.get("race", []))}
            self._gender_index = {c: i for i, c in enumerate(self.labels.get("gender", []))}
            self._age_index = {c: i for i, c in enumerate(self.labels.get("age_group", []))}

        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def _label_index(self, index_map: dict, value) -> int:
        return index_map.get(value, IGNORE_INDEX)

    def _targets(self, row: pd.Series) -> dict:
        if self.dataset == "lfw":
            attrs = [row[f"attr_{a}"] for a in self.labels["attributes"]]
            return {"attributes": torch.tensor(attrs, dtype=torch.float32)}
        return {
            "race": torch.tensor(self._label_index(self._race_index, row["race_label"]), dtype=torch.long),
            "gender": torch.tensor(self._label_index(self._gender_index, row["gender_label"]), dtype=torch.long),
            "age_group": torch.tensor(self._label_index(self._age_index, row["age_group"]), dtype=torch.long),
        }

    def _metadata(self, row: pd.Series) -> dict:
        return {
            "image_path": row["image_path"],
            "identity_id": row.get("identity_id"),
            "race_label": row["race_label"],
            "gender_label": row["gender_label"],
            "age_group": row["age_group"],
            "ita_value": row["ita_value"],
            "ita_group": row["ita_group"],
        }

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = Image.open(REPO_ROOT / row["image_path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, self._targets(row), self._metadata(row)
