"""Shared per-sample 'how demographically rare is this image' weight,
combining race, gender, age and skin-tone marginals computed from ONE
dataset's own training distribution -- never pooled across datasets (each
of FairFace/UTKFace/LFW calls this separately, on its own train split, so
FairFace's weights never leak into UTKFace's or LFW's). Used identically
by M2-MULTI (as a WeightedRandomSampler weight) and M3-MULTI (as a loss
weight) so both interventions start from the same notion of "rare" and
differ only in where that number is injected into training.

Skin tone here is Fitzpatrick (approximated from ITA via the published
breakpoints in frs_m0/utils.py), not raw ITA -- the brief names Fitzpatrick
specifically, and it gives six bands instead of a coarse <28/>=28 split.
The existing M2-ITA/M3-ITA/M4-ITA experiments keep using ita_group
directly; this is a deliberate, separate choice for the MULTI experiments.

Formula, per attribute a in {race_label, gender_label, age_group, fitzpatrick_approx}:
  r_a(i) = (1 / count(group of image i in a)), normalised so mean(r_a) == 1
           across the training set -- keeps every attribute's contribution
           on the same scale regardless of its class count or skew, so
           e.g. FairFace's 7-way race and 2-way gender pull equally hard.
Combined:
  weight(i) = mean of r_a(i) over whichever attributes have a non-missing
              value for image i (a missing attribute is left out of that
              image's mean, never imputed or penalised), then capped at
              MAX_WEIGHT so one person who happens to be rare on every
              single attribute can't dominate a batch or the loss -- the
              brief's own warning against letting combined weights
              "explode" if multiplied unchecked.

A person in several underrepresented groups gets a higher combined weight
than someone in only one, because each rare-group term pulls the mean up
-- without ever forming one combinatorial race x gender x age x skin-tone
class, which would create the tiny/empty cells the brief explicitly warns
about.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "frs_m0"))
from utils import approximate_fitzpatrick  # noqa: E402

ATTRS = ["race_label", "gender_label", "age_group", "fitzpatrick_approx"]
MAX_WEIGHT = 4.0  # mean weight is 1.0 by construction, so this caps any sample at 4x average


def _bucket(value):
    if value is None:
        return None
    try:
        if value != value:  # NaN
            return None
    except TypeError:
        pass
    if value in ("≥28", ">=28"):
        return "≥28"
    return str(value)


def compute_multi_demographic_weights(df: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """df: one row per training image, with ATTRS columns present (every
    split_manifest.csv already has all four -- see face_dataset.py).
    Returns (weight per row as float64 array aligned to df's row order,
    diagnostics dict for logging to weight_diagnostics.json)."""
    n = len(df)
    if "fitzpatrick_approx" not in df.columns:
        df = df.copy()
        df["fitzpatrick_approx"] = df["ita_value"].apply(approximate_fitzpatrick)

    per_attr_rarity: dict[str, np.ndarray] = {}
    group_counts_log: dict[str, dict] = {}

    for attr in ATTRS:
        buckets = df[attr].map(_bucket)
        counts = buckets.value_counts()
        group_counts_log[attr] = {str(k): int(v) for k, v in counts.to_dict().items()}
        raw = buckets.map(lambda b: (1.0 / counts[b]) if b is not None else np.nan).astype(float)
        mean_raw = raw.mean(skipna=True)
        per_attr_rarity[attr] = (raw / mean_raw).to_numpy() if mean_raw else np.full(n, np.nan)

    stacked = np.vstack([per_attr_rarity[a] for a in ATTRS])  # (4, n)
    pre_cap_mean = np.nanmean(stacked, axis=0)
    weight = np.nan_to_num(pre_cap_mean, nan=1.0)  # only possible if ALL 4 attrs missing for a row
    clip_mask = weight > MAX_WEIGHT
    n_clipped = int(clip_mask.sum())
    weight = np.minimum(weight, MAX_WEIGHT)

    diagnostics = {
        "attributes": ATTRS,
        "group_counts": group_counts_log,
        "n_images": n,
        "mean_weight": float(weight.mean()),
        "min_weight": float(weight.min()),
        "max_weight_before_cap": float(np.nanmax(pre_cap_mean)) if n else None,
        "max_weight_cap": MAX_WEIGHT,
        "n_clipped_at_cap": n_clipped,
    }
    return weight, diagnostics
