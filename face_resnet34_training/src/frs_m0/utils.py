"""Shared helpers for the M0 face-verification baseline.

Deliberately separate from src/datasets/transforms.py: that file's
train_transform() includes ColorJitter (brightness/contrast/saturation),
which M0's spec explicitly forbids -- skin-tone and acquisition-condition
analysis are part of this investigation, so the training augmentation
must not perturb exactly the signal being measured.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def m0_train_transform() -> T.Compose:
    return T.Compose(
        [
            T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def m0_eval_transform() -> T.Compose:
    return T.Compose(
        [
            T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


# Same approximate ITA(deg) -> Fitzpatrick-band mapping used in
# src/evaluation/metrics.py (duplicated, not imported, to keep frs_m0/
# self-contained -- see that file's comment for the dermatology-literature
# justification). NOT a clinical Fitzpatrick assignment.
FITZPATRICK_ITA_BREAKPOINTS = [
    (55.0, "I-II (very light)"),
    (41.0, "II-III (light)"),
    (28.0, "III-IV (intermediate)"),
    (10.0, "IV-V (tan)"),
    (-30.0, "V (brown)"),
    (float("-inf"), "VI (dark)"),
]


def approximate_fitzpatrick(ita_value) -> str:
    if ita_value is None or (isinstance(ita_value, float) and np.isnan(ita_value)):
        return "unknown"
    for lower, label in FITZPATRICK_ITA_BREAKPOINTS:
        if ita_value > lower:
            return label
    return "unknown"


def l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, p=2, dim=-1)


def cosine_similarity_matrix_pairs(emb_a: np.ndarray, emb_b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity for two equal-length, already L2-normalized
    embedding arrays -- for normalized vectors this is just the dot product."""
    return np.sum(emb_a * emb_b, axis=1)


def compute_roc(scores: np.ndarray, labels: np.ndarray):
    """labels: 1 = genuine (same identity), 0 = impostor. Returns
    (thresholds, fmr, fnmr) swept across the full score range, sorted by
    threshold ascending. FMR = impostor pairs scoring >= threshold (wrongly
    accepted); FNMR = genuine pairs scoring < threshold (wrongly rejected)."""
    thresholds = np.unique(scores)
    thresholds = np.concatenate([[thresholds[0] - 1e-6], thresholds, [thresholds[-1] + 1e-6]])
    genuine = scores[labels == 1]
    impostor = scores[labels == 0]
    n_gen, n_imp = len(genuine), len(impostor)
    fmr = np.array([(impostor >= t).sum() / n_imp if n_imp else np.nan for t in thresholds])
    fnmr = np.array([(genuine < t).sum() / n_gen if n_gen else np.nan for t in thresholds])
    return thresholds, fmr, fnmr


def eer_from_roc(thresholds: np.ndarray, fmr: np.ndarray, fnmr: np.ndarray) -> tuple[float, float]:
    """Returns (eer, threshold_at_eer) -- the point where FMR and FNMR cross,
    linearly interpolated between the two bracketing thresholds."""
    diff = fmr - fnmr
    sign_changes = np.where(np.diff(np.sign(diff)) != 0)[0]
    if len(sign_changes) == 0:
        idx = int(np.argmin(np.abs(diff)))
        return float((fmr[idx] + fnmr[idx]) / 2), float(thresholds[idx])
    i = sign_changes[0]
    x0, x1 = diff[i], diff[i + 1]
    t0, t1 = thresholds[i], thresholds[i + 1]
    f0, f1 = fmr[i], fmr[i + 1]
    w = x0 / (x0 - x1) if (x0 - x1) != 0 else 0.5
    eer_threshold = t0 + w * (t1 - t0)
    eer = f0 + w * (f1 - f0)
    return float(eer), float(eer_threshold)


def tar_at_fmr(thresholds: np.ndarray, fmr: np.ndarray, fnmr: np.ndarray, target_fmr: float) -> float | None:
    """TAR (= 1 - FNMR) at the operating point where FMR is closest to (and
    <=) target_fmr, scanning from the high-threshold/low-FMR end. Returns
    None if no operating point reaches that low an FMR (too few impostor
    pairs to resolve it) -- caller must not fabricate a number here."""
    valid = fmr <= target_fmr
    if not valid.any():
        return None
    idx = np.where(valid)[0][0]  # tightest threshold that still meets the target
    return float(1.0 - fnmr[idx])


def bootstrap_ci(values_fn, n_items: int, n_boot: int = 1000, seed: int = 42, alpha: float = 0.05):
    """values_fn(indices) -> scalar metric computed on a resample. Resamples
    whole indices (identities or pairs, caller's choice) WITH replacement.
    Returns (point_estimate, ci_lo, ci_hi)."""
    rng = np.random.default_rng(seed)
    point = values_fn(np.arange(n_items))
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n_items, n_items)
        boots[i] = values_fn(idx)
    lo, hi = np.nanpercentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(point), float(lo), float(hi)
