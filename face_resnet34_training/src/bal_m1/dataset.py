"""BAL-M1 dataset: reads harmoface_manifest_enriched.csv (built by
build_manifest.py from the HarmoFace dataset's own frozen
records_labeled.csv + this project's age harmonization and
fitzpatrick_approx). One row per image; four possible targets
(race, gender, age, skin_tone), each independently missing or present --
never fabricated, never imputed (project-wide rule).

Label sources, and why:
  race:       candidate_demographic_label -- the dataset's OWN harmonized
              race target (race_label is the original, preserved
              separately, never used as the training target itself).
  gender:     gender_clean (female/male; "unknown" -> ignored).
  age:        age_group_harmonized (this project's fix for the frozen
              manifest's un-harmonized age_group -- see build_manifest.py).
  skin_tone:  fitzpatrick_approx (6-band, ITA-derived approximation,
              documented as such -- not a bare "fitzpatrick" claim).

A row contributes to a head's loss only when that head's label is valid
for it; __getitem__ returns IGNORE_INDEX (-1) for every missing one, the
same convention used everywhere else in this project.
"""
from __future__ import annotations

import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

IGNORE_INDEX = -1

RACE_VOCAB = {"White": 0, "Black": 1, "Indian": 2, "East Asian": 3,
              "Southeast Asian": 4, "Middle Eastern": 5, "Latino": 6}
GENDER_VOCAB = {"female": 0, "male": 1}
AGE_VOCAB = {"0-19": 0, "20-39": 1, "40-59": 2, "60+": 3}
# Built lazily from whatever fitzpatrick_approx values are actually present
# in the manifest (approximate_fitzpatrick's bands), sorted for a stable,
# reproducible index order -- never hardcoded, since "unknown" rows exist
# and the exact band set depends on the ITA distribution actually seen.


def _load_rows(manifest_path: Path, split: str) -> list[dict]:
    with open(manifest_path, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == split and r["file_present"] == "1"]
    return rows


def build_skin_tone_vocab(rows: list[dict]) -> dict[str, int]:
    values = sorted({r["fitzpatrick_approx"] for r in rows if r["fitzpatrick_approx"] not in ("", "unknown")})
    return {v: i for i, v in enumerate(values)}


class BalM1Dataset(Dataset):
    """split in {'train','val','test'}. Only rows with file_present=='1'
    are included -- see build_manifest.py's completeness report for how
    much of each split that currently is. __getitem__ returns
    (image_tensor, {"race":.., "gender":.., "age":.., "skin_tone":..},
    row_dict) -- row_dict carries source_dataset/image_id/etc for
    per-source and fairness-audit breakdowns downstream, same pattern
    FaceDataset's `meta` dict already uses elsewhere in this project."""

    def __init__(self, manifest_path: Path, split: str, transform, skin_tone_vocab: dict[str, int] | None = None):
        self.rows = _load_rows(manifest_path, split)
        self.transform = transform
        self.skin_tone_vocab = skin_tone_vocab if skin_tone_vocab is not None else build_skin_tone_vocab(self.rows)

    def __len__(self) -> int:
        return len(self.rows)

    def _encode(self, row: dict) -> dict[str, int]:
        race = RACE_VOCAB.get(row["candidate_demographic_label"], IGNORE_INDEX)
        gender = GENDER_VOCAB.get(row["gender_clean"], IGNORE_INDEX)
        age = AGE_VOCAB.get(row["age_group_harmonized"], IGNORE_INDEX)
        skin = self.skin_tone_vocab.get(row["fitzpatrick_approx"], IGNORE_INDEX)
        return {"race": race, "gender": gender, "age": age, "skin_tone": skin}

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        img = Image.open(row["harmoface_abs_path"]).convert("RGB")
        img = self.transform(img)
        targets = {k: torch.tensor(v, dtype=torch.long) for k, v in self._encode(row).items()}
        return img, targets, row
