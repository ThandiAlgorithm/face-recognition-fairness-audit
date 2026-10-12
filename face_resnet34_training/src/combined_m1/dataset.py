"""Combined-M1: one dataset spanning FairFace + UTKFace + LFW train splits.

Reuses each dataset's own frozen split_manifest.csv (never regenerates or
touches val/test) and joins LFW's LFWA+ attribute columns the same way
FaceDataset does. See configs/race_mapping.yaml and configs/age_mapping.yaml
for the harmonization rules this applies, and their docstrings for why each
ambiguous source category becomes null rather than a guess.

Every task below is independently masked: an image missing a task's label
contributes IGNORE_INDEX (classification) or an all-zero mask row
(multi-label), so it drops out of that task's loss term entirely rather
than injecting a wrong or invented target.

Heads and their sources:
  gender          -- gender_label, read the same way for all three datasets.
                     For LFW specifically this is NOT the LFWA+ attr_Male
                     flag (13,143/13,233 images) -- gender_label there comes
                     from the original male_names.txt/female_names.txt
                     identity lists (gender_source ==
                     'derived_lfw_name_lists' in lfw_records.csv), covering
                     13,224/13,233 images, and is the same source already
                     used everywhere else in this project for gender
                     auditing and for M2/M3-MULTI's sample weighting.
  age_harmonized  -- FairFace/UTKFace age_group, mapped via age_mapping.yaml;
                     LFW: always missing (no numeric age label exists).
  race_harmonized -- FairFace/UTKFace race_label, mapped via
                     race_mapping.yaml; LFW: derived from its four native
                     Asian/Black/Indian/White attribute flags (exactly one
                     set -> that category, else missing) -- this is NOT
                     switched to LFW's own derived race_label the way gender
                     was, because race_label there is less complete
                     (11,104/13,233: it abstains whenever the underlying
                     attribute scores don't clearly favour one race) than
                     the raw attribute-flag approach already used here.
  skin_tone       -- ita_group (<28 / >=28), already identical across all
                     three manifests (the frozen GOV-002 rule) -- no mapping
                     file needed, just missing when ita_group is 'unknown'.
  lfw_attributes  -- LFW's LFWA+ attributes MINUS {Male, Asian, Black,
                     Indian, White} (those five still shouldn't be
                     double-supervised: Male duplicates the gender head's
                     signal even though it's no longer that head's literal
                     source, and Asian/Black/Indian/White already feed the
                     race head). Always fully missing for FairFace/UTKFace,
                     which have no appearance-attribute labels at all.

All five targets, plus every metadata field, are computed ONCE in __init__
as plain numpy arrays, not per-row inside __getitem__: pandas' .iloc/Series
row access (the first version of this file used it) carries enough
per-call overhead that, multiplied by ~108k images x several column
lookups each x every epoch, it became the dominant cost of training --
confirmed empirically (both Combined-M1 runs stuck well past their
benchmarked epoch time with CPU still pegged, i.e. doing real but wasted
work, not stalled). __getitem__ now only does a numpy index (O(1)) per
field, so the real cost per image goes back to what it should be: opening
and decoding that one image file.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import Dataset
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "datasets"))
from face_dataset import load_frozen_manifest, REPO_ROOT, RECORDS_DIR, IGNORE_INDEX
from label_encoders import load_labels

CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"
SOURCE_DATASETS = ["fairface", "utkface", "lfw"]

GENDER_CATEGORIES = ["female", "male"]
SKIN_TONE_CATEGORIES = ["<28", "≥28"]  # == ita_group's two known values
LFW_DEMOGRAPHIC_ATTRS = ["Male", "Asian", "Black", "Indian", "White"]


def _load_yaml(name: str) -> dict:
    return yaml.safe_load((CONFIGS_DIR / name).read_text())


def build_race_harmonized_vocab() -> list[str]:
    return list(_load_yaml("race_mapping.yaml")["harmonized_categories"])


def build_age_harmonized_vocab() -> list[str]:
    return list(_load_yaml("age_mapping.yaml")["harmonized_categories"])


def lfw_attribute_vocab() -> list[str]:
    all_attrs = load_labels("lfw")["attributes"]
    return [a for a in all_attrs if a not in LFW_DEMOGRAPHIC_ATTRS]


class CombinedDataset(Dataset):
    """split in {'train','val','test'}. __getitem__ returns
    (image_tensor, targets, metadata); targets always carries all five
    task keys, with IGNORE_INDEX / zero-mask rows for tasks that don't
    apply to a given image."""

    def __init__(self, split: str, transform=None):
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be train/val/test, got {split!r}")
        self.split = split
        self.transform = transform

        self.race_map = _load_yaml("race_mapping.yaml")
        self.age_map = _load_yaml("age_mapping.yaml")
        self.race_vocab = build_race_harmonized_vocab()
        self.age_vocab = build_age_harmonized_vocab()
        self.lfw_attr_vocab = lfw_attribute_vocab()
        self._race_index = {c: i for i, c in enumerate(self.race_vocab)}
        self._age_index = {c: i for i, c in enumerate(self.age_vocab)}
        self._gender_index = {c: i for i, c in enumerate(GENDER_CATEGORIES)}
        self._skin_index = {c: i for i, c in enumerate(SKIN_TONE_CATEGORIES)}

        frames = []
        for dataset in SOURCE_DATASETS:
            manifest = load_frozen_manifest(dataset)
            df = manifest[manifest["split"] == split].reset_index(drop=True)
            if dataset == "lfw":
                attr_cols = [f"attr_{a}" for a in load_labels("lfw")["attributes"]]
                records = pd.read_csv(RECORDS_DIR / "lfw_records.csv", usecols=["image_path", *attr_cols])
                df = df.merge(records, on="image_path", how="left", validate="one_to_one")
                missing = df[attr_cols].isna().any(axis=1)
                if missing.any():
                    print(f"[combined_m1/{split}] lfw: {int(missing.sum())}/{len(df)} rows have no "
                          "LFWA+ attribute match -- excluded")
                    df = df[~missing].reset_index(drop=True)
            df["source_dataset"] = dataset
            frames.append(df)

        self.df = pd.concat(frames, ignore_index=True)
        print(f"[combined_m1/{split}] {len(self.df)} images "
              f"({', '.join(f'{d}={len(f)}' for d, f in zip(SOURCE_DATASETS, frames))})")
        self._precompute_arrays()

    def __len__(self) -> int:
        return len(self.df)

    def _precompute_arrays(self) -> None:
        df = self.df
        n = len(df)
        source = df["source_dataset"].to_numpy()
        is_lfw = source == "lfw"

        # --- metadata (plain arrays, never fed to the model) ---
        self.image_paths = df["image_path"].to_numpy()
        self.source_dataset = source
        self.identity_id = df.get("identity_id", pd.Series([None] * n)).fillna("none").astype(str).to_numpy()
        self.race_original = df["race_label"].fillna("unknown").astype(str).to_numpy()
        self.gender_label_str = df["gender_label"].fillna("unknown").astype(str).to_numpy()
        self.age_group_str = df["age_group"].fillna("unknown").astype(str).to_numpy()
        self.ita_value = df["ita_value"].to_numpy()
        self.ita_group_str = df["ita_group"].fillna("unknown").astype(str).to_numpy()

        # --- gender ---
        # gender_label (not the LFWA+ attr_Male flag) is used uniformly
        # across all three datasets: for LFW specifically, gender_label is
        # sourced from the original male_names.txt/female_names.txt identity
        # lists (gender_source == 'derived_lfw_name_lists' in
        # lfw_records.csv), which cover 13,224/13,233 images -- more
        # complete than attr_Male's 13,143, and the project's own
        # established authoritative gender source (used everywhere else for
        # fairness auditing and for M2/M3-MULTI's sample weighting already).
        self.gender_target = df["gender_label"].map(lambda v: self._gender_index.get(v, IGNORE_INDEX)).to_numpy().astype(np.int64)

        # --- age (FairFace/UTKFace only -- LFW has no numeric age label) ---
        age_target = np.full(n, IGNORE_INDEX, dtype=np.int64)
        for dataset in SOURCE_DATASETS:
            mapping = self.age_map[dataset]
            if dataset == "lfw" or mapping is None:
                continue
            mask = source == dataset
            harmonized = df.loc[mask, "age_group"].map(mapping)
            age_target[mask] = harmonized.map(lambda h: self._age_index.get(h, IGNORE_INDEX)).to_numpy()
        self.age_target = age_target

        # --- race (FairFace/UTKFace via race_mapping.yaml; LFW via its own
        # four native attribute flags, exactly one set -> that category) ---
        race_target = np.full(n, IGNORE_INDEX, dtype=np.int64)
        for dataset in SOURCE_DATASETS:
            if dataset == "lfw":
                continue
            mask = source == dataset
            mapping = self.race_map[dataset]
            harmonized = df.loc[mask, "race_label"].map(mapping)
            race_target[mask] = harmonized.map(lambda h: self._race_index.get(h, IGNORE_INDEX)).to_numpy()
        if is_lfw.any():
            names = ["White", "Black", "Asian", "Indian"]
            flags = np.stack([df[f"attr_{a}"].to_numpy(dtype=float) for a in names], axis=1)  # (n, 4)
            set_flags = (flags >= 0.5) & ~np.isnan(flags)
            exactly_one = set_flags.sum(axis=1) == 1
            which = set_flags.argmax(axis=1)
            for i, name in enumerate(names):
                sel = is_lfw & exactly_one & (which == i)
                race_target[sel] = self._race_index.get(name, IGNORE_INDEX)
        self.race_target = race_target

        # --- skin tone (identical ita_group rule across all three datasets) ---
        self.skin_target = df["ita_group"].map(lambda v: self._skin_index.get(v, IGNORE_INDEX)).to_numpy().astype(np.int64)

        # --- LFW appearance attributes (zero + masked-out for non-LFW rows) ---
        n_attr = len(self.lfw_attr_vocab)
        attr_values = np.zeros((n, n_attr), dtype=np.float32)
        attr_mask = np.zeros((n, n_attr), dtype=np.float32)
        if is_lfw.any():
            attr_cols = [f"attr_{a}" for a in self.lfw_attr_vocab]
            attr_values[is_lfw] = df.loc[is_lfw, attr_cols].to_numpy(dtype=np.float32)
            attr_mask[is_lfw] = 1.0
        self.lfw_attr_values = attr_values
        self.lfw_attr_mask = attr_mask

    def __getitem__(self, idx: int):
        image = Image.open(REPO_ROOT / self.image_paths[idx]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        targets = {
            "gender": torch.tensor(self.gender_target[idx], dtype=torch.long),
            "age_harmonized": torch.tensor(self.age_target[idx], dtype=torch.long),
            "race_harmonized": torch.tensor(self.race_target[idx], dtype=torch.long),
            "skin_tone": torch.tensor(self.skin_target[idx], dtype=torch.long),
            "lfw_attributes": torch.from_numpy(self.lfw_attr_values[idx]),
            "lfw_attributes_mask": torch.from_numpy(self.lfw_attr_mask[idx]),
        }
        metadata = {
            "image_path": self.image_paths[idx],
            "source_dataset": self.source_dataset[idx],
            "identity_id": self.identity_id[idx],
            "race_original": self.race_original[idx],
            "gender_label": self.gender_label_str[idx],
            "age_group": self.age_group_str[idx],
            "ita_value": self.ita_value[idx],
            "ita_group": self.ita_group_str[idx],
        }
        return image, targets, metadata
