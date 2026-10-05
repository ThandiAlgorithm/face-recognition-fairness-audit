"""Step 6: frozen per-dataset label vocabularies.

Built once per dataset and saved to configs/<dataset>_labels.json. Every
later run loads the saved vocabulary rather than re-deriving category order
from whatever happens to be in the data at the time, so a class's integer
index can never silently shift between runs.

'unknown' / missing values are excluded from the trainable vocabulary --
FaceDataset encodes them as -1 (the standard ignore_index for
CrossEntropyLoss) rather than inventing a class for "no label". The
n_unknown counts recorded here exist so that choice is auditable.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"
RECORDS_DIR = Path(__file__).resolve().parents[2] / "data" / "records"

LFW_ATTR_PREFIX = "attr_"
UNKNOWN_VALUES = {"unknown"}


class LabelsAlreadyFrozenError(RuntimeError):
    pass


def _category_field(series: pd.Series) -> dict:
    valid = series.dropna()
    valid = valid[~valid.isin(UNKNOWN_VALUES)]
    categories = sorted(valid.unique().tolist())
    return {
        "categories": categories,
        "n_unknown_or_missing": int(len(series) - len(valid)),
    }


def build_fairface_labels() -> dict:
    df = pd.read_csv(RECORDS_DIR / "fairface_records.csv")
    return {
        "race": _category_field(df["race_label"]),
        "gender": _category_field(df["gender_label"]),
        "age_group": _category_field(df["age_group"]),
    }


def build_utkface_labels() -> dict:
    df = pd.read_csv(RECORDS_DIR / "utkface_records.csv")
    return {
        "race": _category_field(df["race_label"]),
        "gender": _category_field(df["gender_label"]),
        "age_group": _category_field(df["age_group"]),
    }


def build_lfw_labels() -> dict:
    df = pd.read_csv(RECORDS_DIR / "lfw_records.csv")
    attributes = sorted(c[len(LFW_ATTR_PREFIX):] for c in df.columns if c.startswith(LFW_ATTR_PREFIX))
    return {"attributes": attributes}


BUILDERS = {
    "fairface": build_fairface_labels,
    "utkface": build_utkface_labels,
    "lfw": build_lfw_labels,
}


def freeze_labels(dataset: str, force: bool = False) -> Path:
    out_path = CONFIGS_DIR / f"{dataset}_labels.json"
    if out_path.exists() and not force:
        raise LabelsAlreadyFrozenError(
            f"{out_path} already exists. Label vocabularies are frozen once written -- "
            "pass force=True (and document why) to replace them deliberately."
        )
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    labels = BUILDERS[dataset]()
    out_path.write_text(json.dumps(labels, indent=2))
    return out_path


def load_labels(dataset: str) -> dict:
    """Returns the flat {task: [categories]} shape FaceDataset expects
    (LFW's 'attributes' list is already flat)."""
    path = CONFIGS_DIR / f"{dataset}_labels.json"
    raw = json.loads(path.read_text())
    if dataset == "lfw":
        return raw
    return {task: field["categories"] for task, field in raw.items()}


def main() -> None:
    for dataset in BUILDERS:
        path = freeze_labels(dataset)
        labels = json.loads(path.read_text())
        print(f"{dataset}: {path.name}")
        for task, field in labels.items():
            if isinstance(field, dict):
                print(f"  {task}: {len(field['categories'])} classes, {field['n_unknown_or_missing']} unknown/missing")
            else:
                print(f"  {task}: {len(field)} items")


if __name__ == "__main__":
    main()
