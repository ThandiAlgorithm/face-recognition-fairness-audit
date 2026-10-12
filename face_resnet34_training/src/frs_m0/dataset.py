"""M0 dataset: loads LFW's already-Phase-0-aligned crops with identity
labels (for training) and full demographic/appearance metadata (for
fairness analysis downstream). Reuses data/records/lfw_records.csv (built
by src/datasets/records.py) and experiments/lfw/split_manifest.csv (the
frozen, identity-disjoint 80/10/10 split already used by M1) rather than
rebuilding either -- see data_audit.py's docstring for why reusing the
existing split is deliberate, not a shortcut.

Missing demographic values stay NaN/"unknown" -- never inferred, same
policy as the rest of this project.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
from utils import approximate_fitzpatrick  # noqa: E402  (this dir's utils.py, i.e. frs_m0/utils.py)
from demographic_weights import compute_multi_demographic_weights  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
FACE_TRAINING_ROOT = Path(__file__).resolve().parents[2]
RECORDS_PATH = FACE_TRAINING_ROOT / "data" / "records" / "lfw_records.csv"
SPLIT_MANIFEST_PATH = FACE_TRAINING_ROOT / "experiments" / "lfw" / "split_manifest.csv"

DEMOGRAPHIC_COLS = ["race_label", "gender_label", "age_group", "ita_value", "ita_group"]
# A representative, defensible subset of LFWA+ appearance attributes for
# the "other overlooked skews" analysis (spec section 30) -- not all 73,
# just the ones with a clear appearance-condition interpretation.
APPEARANCE_ATTR_COLS = [
    "attr_Eyeglasses", "attr_Sunglasses", "attr_No_Eyewear",
    "attr_Mustache", "attr_No_Beard", "attr_5_o_Clock_Shadow", "attr_Goatee", "attr_Sideburns",
    "attr_Heavy_Makeup", "attr_Wearing_Lipstick",
    "attr_Wearing_Hat", "attr_Bald",
    "attr_Blurry", "attr_Harsh_Lighting", "attr_Flash",
]


def load_m0_table() -> pd.DataFrame:
    """One row per LFW image: identity_id, split, image_path, every
    DEMOGRAPHIC_COLS + APPEARANCE_ATTR_COLS field, fitzpatrick_approx."""
    records = pd.read_csv(RECORDS_PATH)
    split = pd.read_csv(SPLIT_MANIFEST_PATH)[["image_path", "split"]]
    df = records.merge(split, on="image_path", how="inner", validate="one_to_one")
    df["fitzpatrick_approx"] = df["ita_value"].apply(approximate_fitzpatrick)
    keep_cols = ["image_id", "image_path", "identity_id", "split"] + DEMOGRAPHIC_COLS + ["fitzpatrick_approx"] + APPEARANCE_ATTR_COLS
    return df[keep_cols].reset_index(drop=True)


def build_train_identity_vocab(df: pd.DataFrame, min_images: int) -> tuple[dict[str, int], list[str]]:
    """Train-only identity -> class-index mapping, restricted to identities
    with >= min_images in TRAIN (spec section 6). Returns (vocab,
    removed_identity_ids) so callers can log what got dropped instead of
    silently discarding it."""
    train = df[df["split"] == "train"]
    counts = train["identity_id"].value_counts()
    kept = sorted(counts[counts >= min_images].index.tolist())
    removed = sorted(counts[counts < min_images].index.tolist())
    vocab = {identity: i for i, identity in enumerate(kept)}
    return vocab, removed


IGNORE_INDEX = -1


def _build_demo_vocab(series: pd.Series) -> dict[str, int]:
    """Sorted category -> index map, excluding NaN/'unknown' (those get
    IGNORE_INDEX at lookup time, never a fabricated class). Returns an
    empty dict if the column has no real values at all (e.g. LFW's
    age_group, which is 100% missing -- see dataset.py's module docstring
    on never inferring missing demographics)."""
    valid = series.dropna()
    valid = valid[valid != "unknown"]
    return {cat: i for i, cat in enumerate(sorted(valid.unique()))}


class M0TrainDataset(Dataset):
    """Identity-classification training set: TRAIN split images whose
    identity survived the >=min_images filter (build_train_identity_vocab).

    compute_weights=True computes the same combined race+gender+age+
    Fitzpatrick rarity weight M2-MULTI/M3-MULTI use (demographic_weights.py),
    from THIS dataset's own train split -- never pooled with FairFace/
    UTKFace.

    Also always builds race/gender/age label vocabularies and exposes them
    (self.race_vocab etc) for M0AdversarialEmbeddingModel's adversary
    heads -- built unconditionally (cheap), used only when
    --mitigation adversarial. A demographic with no real values at all for
    this dataset (LFW's age_group) gets an empty vocab and every row's
    label is IGNORE_INDEX -- the adversary head still exists architecturally
    (for a uniform model across datasets) but never receives a real
    training signal, which is the correct behavior, not a bug: there is
    nothing to adversarially erase if nothing was ever encoded.

    __getitem__ always returns (img, identity_label, weight, race_label,
    gender_label, age_label); weight is a uniform 1.0 and the demographic
    labels are unused when their respective mitigation isn't active, so
    the "none" mitigation path is numerically identical to never having
    this feature."""

    def __init__(self, df: pd.DataFrame, identity_vocab: dict[str, int], transform, compute_weights: bool = False):
        sub = df[(df["split"] == "train") & (df["identity_id"].isin(identity_vocab))].reset_index(drop=True)
        self.df = sub
        self.identity_vocab = identity_vocab
        self.transform = transform
        if compute_weights:
            self.weights, self.weight_diagnostics = compute_multi_demographic_weights(sub)
        else:
            self.weights = np.ones(len(sub), dtype=np.float64)
            self.weight_diagnostics = None

        self.race_vocab = _build_demo_vocab(sub["race_label"])
        self.gender_vocab = _build_demo_vocab(sub["gender_label"])
        self.age_vocab = _build_demo_vocab(sub["age_group"])
        self._race_labels = sub["race_label"].map(lambda v: self.race_vocab.get(v, IGNORE_INDEX)).to_numpy()
        self._gender_labels = sub["gender_label"].map(lambda v: self.gender_vocab.get(v, IGNORE_INDEX)).to_numpy()
        self._age_labels = sub["age_group"].map(lambda v: self.age_vocab.get(v, IGNORE_INDEX)).to_numpy()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(REPO_ROOT / row["image_path"]).convert("RGB")
        img = self.transform(img)
        label = self.identity_vocab[row["identity_id"]]
        weight = float(self.weights[idx])
        return (
            img,
            torch.tensor(label, dtype=torch.long),
            torch.tensor(weight, dtype=torch.float32),
            torch.tensor(int(self._race_labels[idx]), dtype=torch.long),
            torch.tensor(int(self._gender_labels[idx]), dtype=torch.long),
            torch.tensor(int(self._age_labels[idx]), dtype=torch.long),
        )


class M0EvalDataset(Dataset):
    """val/test images for embedding extraction -- ALL images in the split,
    including identity-singletons (they're valid impostor-pair members even
    though they can't form a genuine pair; see spec section 6)."""

    def __init__(self, df: pd.DataFrame, split: str, transform):
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(REPO_ROOT / row["image_path"]).convert("RGB")
        img = self.transform(img)
        return img, row["image_id"]
