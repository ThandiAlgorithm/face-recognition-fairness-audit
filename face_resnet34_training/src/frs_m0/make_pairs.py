"""M0 spec sections 14-17: genuine + impostor pair generation.

Genuine pair: same identity, two different images.
Impostor pair: two different identities (one image each).

Both use a fixed pair_seed so pairs are reproducible and, once written to
disk, frozen -- never regenerated for the same split (spec section 17).
Impostor pairs also get same/different flags for every demographic axis
(spec section 16), so the controlled same-race/same-gender/etc. impostor
analysis doesn't need the raw images again later.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import load_m0_table

OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "M0_LFW"
DEMOGRAPHIC_PAIR_COLS = ["race_label", "gender_label", "age_group", "fitzpatrick_approx", "ita_group"]


def _genuine_pairs(df: pd.DataFrame, rng: np.random.Generator) -> list[tuple]:
    pairs = []
    for identity, g in df.groupby("identity_id"):
        if len(g) < 2:
            continue
        idx = g.index.to_numpy()
        # every unique unordered pair within this identity -- LFW identities
        # with 2+ images are typically small (median 1-2 even restricted to
        # this subset), so this is never combinatorially large in practice.
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                pairs.append((idx[i], idx[j]))
    rng.shuffle(pairs)
    return pairs


def _impostor_pairs(df: pd.DataFrame, n_pairs: int, rng: np.random.Generator) -> list[tuple]:
    idx = df.index.to_numpy()
    identity = df["identity_id"].to_numpy()
    pairs = []
    seen = set()
    attempts = 0
    max_attempts = n_pairs * 20
    while len(pairs) < n_pairs and attempts < max_attempts:
        attempts += 1
        i, j = rng.choice(idx, size=2, replace=False)
        if identity[np.where(idx == i)[0][0]] == identity[np.where(idx == j)[0][0]]:
            continue
        key = (min(i, j), max(i, j))
        if key in seen:
            continue
        seen.add(key)
        pairs.append(key)
    return pairs


def _row_for_pair(df: pd.DataFrame, a_idx: int, b_idx: int, same_identity: int, pair_id: int) -> dict:
    a, b = df.loc[a_idx], df.loc[b_idx]
    row = {
        "pair_id": pair_id,
        "image_a": a["image_path"], "image_b": b["image_path"],
        "image_id_a": a["image_id"], "image_id_b": b["image_id"],
        "identity_a": a["identity_id"], "identity_b": b["identity_id"],
        "same_identity": same_identity,
    }
    for col in DEMOGRAPHIC_PAIR_COLS:
        row[f"{col}_a"] = a[col]
        row[f"{col}_b"] = b[col]
        row[f"same_{col}"] = int(a[col] == b[col]) if pd.notna(a[col]) and pd.notna(b[col]) else None
    row["ita_a"] = a["ita_value"]
    row["ita_b"] = b["ita_value"]
    return row


def generate_pairs(df: pd.DataFrame, split: str, n_impostor: int, pair_seed: int) -> pd.DataFrame:
    sub = df[df["split"] == split]
    rng = np.random.default_rng(pair_seed)
    genuine = _genuine_pairs(sub, rng)
    impostor = _impostor_pairs(sub, n_impostor, rng)

    rows = []
    pid = 0
    for a, b in genuine:
        rows.append(_row_for_pair(sub, a, b, 1, pid))
        pid += 1
    for a, b in impostor:
        rows.append(_row_for_pair(sub, a, b, 0, pid))
        pid += 1
    return pd.DataFrame(rows)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-impostor-val", type=int, default=20000)
    parser.add_argument("--n-impostor-test", type=int, default=20000)
    parser.add_argument("--pair-seed", type=int, default=42)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "pairs").mkdir(exist_ok=True)
    df = load_m0_table()

    for split, n_imp in [("val", args.n_impostor_val), ("test", args.n_impostor_test)]:
        pairs = generate_pairs(df, split, n_imp, args.pair_seed)
        out_path = OUT_DIR / "pairs" / f"{split}_pairs.csv"
        pairs.to_csv(out_path, index=False)
        n_genuine = int((pairs["same_identity"] == 1).sum())
        n_impostor_actual = int((pairs["same_identity"] == 0).sum())
        print(f"{split}: {n_genuine} genuine pairs, {n_impostor_actual} impostor pairs -> {out_path}")


if __name__ == "__main__":
    main()
