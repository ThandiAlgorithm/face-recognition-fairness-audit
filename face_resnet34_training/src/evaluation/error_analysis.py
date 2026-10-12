"""Error analysis for a binary classification head: what, if anything,
distinguishes the images the model gets WRONG from the ones it gets right,
within one true class. Built to investigate LFW M1's "Male" head
specifically -- why true-female images get wrongly predicted Male=1 20% of
the time -- but works for LFW (any attribute), and for FairFace/UTKFace's
gender task (the one binary task they have).

For LFW: compares misclassified vs correctly-classified true-negative
(true-female) rows across every OTHER LFWA+ attribute in the same
test_predictions.parquet (hairstyle, makeup, lighting, pose, etc.), so a
systematic visual confound shows up as a rate difference between the two
groups. FairFace/UTKFace have no such secondary attribute set, so for them
this instead compares the CATEGORICAL distribution of race/age/skin-tone
between the two groups -- a narrower check, but the only one their saved
metadata supports.

Also reports each misclassified row's image_path and model confidence, so
a handful can be pulled and looked at directly afterward. Purely
descriptive -- no hypothesis assumed going in.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "datasets"))
from label_encoders import load_labels


def _extract(df: pd.DataFrame, dataset: str, attribute: str, labels: dict):
    if dataset == "lfw":
        idx = labels["attributes"].index(attribute)
        y_true = df["attributes_true"].apply(lambda v: v[idx])
        y_prob = df["attributes_probs"].apply(lambda v: v[idx])
        return y_true, y_prob
    task = attribute
    male_idx = labels[task].index("male")
    y_true = df[f"{task}_true"]
    y_prob = df[f"{task}_probs"].apply(lambda v: v[male_idx])
    return y_true, y_prob


def _categorical_rate_comparison(df: pd.DataFrame, misclassified, correct, min_rate_n: int) -> list[dict]:
    rows = []
    for col in ["race_label", "age_group", "ita_group"]:
        if col not in df.columns:
            continue
        for val in df[col].dropna().unique():
            if val in ("unknown",):
                continue
            mis_n = int((df.loc[misclassified, col] == val).sum())
            total_n = int(((df[col] == val) & (misclassified | correct)).sum())
            if total_n < min_rate_n:
                continue
            rows.append({
                "field": col, "value": str(val), "n_in_group": total_n,
                "n_misclassified": mis_n, "rate": round(mis_n / total_n, 4),
            })
    return sorted(rows, key=lambda r: r["rate"], reverse=True)


def analyze(dataset: str, eval_dir: Path, attribute: str = "Male", threshold: float = 0.5,
            min_rate_n: int = 10) -> dict:
    df = pd.read_parquet(eval_dir / "test_predictions.parquet")
    labels = load_labels(dataset)
    y_true, y_prob = _extract(df, dataset, attribute, labels)
    y_pred = (y_prob >= threshold).astype(int)

    true_negative_mask = y_true == 0  # true female
    misclassified = true_negative_mask & (y_pred == 1)  # predicted male, actually female
    correct = true_negative_mask & (y_pred == 0)

    result = {
        "dataset": dataset, "attribute": attribute, "threshold": threshold,
        "n_true_negative_total": int(true_negative_mask.sum()),
        "n_misclassified": int(misclassified.sum()),
        "n_correct": int(correct.sum()),
        "misclassification_rate": float(misclassified.sum() / true_negative_mask.sum()) if true_negative_mask.sum() else None,
    }

    if dataset == "lfw":
        attrs = labels["attributes"]
        other_attrs = [a for a in attrs if a != attribute]
        rate_rows = []
        for a in other_attrs:
            a_idx = attrs.index(a)
            vals = df["attributes_true"].apply(lambda v, a_idx=a_idx: v[a_idx])
            mis_vals = vals[misclassified]
            cor_vals = vals[correct]
            if len(mis_vals) < min_rate_n or len(cor_vals) < min_rate_n:
                continue
            mis_rate = float(mis_vals.mean())
            cor_rate = float(cor_vals.mean())
            rate_rows.append({"attribute": a, "rate_in_misclassified": mis_rate,
                               "rate_in_correct": cor_rate, "gap": mis_rate - cor_rate})
        if rate_rows:
            rate_df = pd.DataFrame(rate_rows).sort_values("gap", key=lambda s: s.abs(), ascending=False)
            result["attribute_rate_comparison"] = rate_df.to_dict("records")
        else:
            # Too few misclassified/correct cases to compare against any other
            # attribute at min_rate_n -- happens for very rare attributes (e.g.
            # LFW's Indian/Black cells), not a crash-worthy condition.
            result["attribute_rate_comparison"] = []
            result["attribute_rate_comparison_note"] = (
                f"No other attribute had >= {min_rate_n} cases in both the misclassified "
                f"and correctly-classified groups -- sample too small for this comparison."
            )
    else:
        # No secondary attribute set for FairFace/UTKFace -- compare race/age/skin-tone
        # category rates instead (which categories are over-represented among the
        # misclassifications relative to their own group size).
        result["category_rate_comparison"] = _categorical_rate_comparison(df, misclassified, correct, min_rate_n)

    if "race_label" in df.columns:
        race_counts = df.loc[misclassified, "race_label"].value_counts().to_dict()
        race_totals = df.loc[true_negative_mask, "race_label"].value_counts().to_dict()
        result["misclassified_by_race"] = {
            r: {"n_misclassified": int(race_counts.get(r, 0)), "n_total_female": int(race_totals.get(r, 0)),
                "rate": round(race_counts.get(r, 0) / race_totals[r], 4) if race_totals.get(r) else None}
            for r in race_totals
        }

    sample_cols = [c for c in ["image_path", "race_label", "age_group", "ita_group"] if c in df.columns]
    sample = df.loc[misclassified, sample_cols].copy()
    sample["model_confidence_male"] = y_prob[misclassified]
    sample = sample.sort_values("model_confidence_male", ascending=False)
    result["sample_misclassified_rows"] = sample.head(20).to_dict("records")

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["lfw", "fairface", "utkface"])
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--attribute", default=None,
                         help="LFW: attribute name (default 'Male'). fairface/utkface: task name (default 'gender').")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    attribute = args.attribute or ("Male" if args.dataset == "lfw" else "gender")

    result = analyze(args.dataset, args.eval_dir, attribute, args.threshold)
    out = args.out or (args.eval_dir / f"error_analysis_{attribute}.json")
    out.write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps({k: v for k, v in result.items() if k != "sample_misclassified_rows"}, indent=2, default=str))


if __name__ == "__main__":
    main()
