"""Master comparison table: one row per dataset x model, built purely by
aggregating what metrics.py already computed (metrics.json,
fairness_metrics.csv) -- no new metrics, no re-running any model. This is
the single table meant to drive the mitigation/model-selection decision
before the final frozen benchmark, not another exploratory view.

Only covers UTKFace and FairFace for now: both share the same
race/gender/age_group classification-head structure and the same
evaluate.py/metrics.py pipeline this reads from. LFW (40 independent
binary attributes, no race/age/skin-tone task structure) and M0 (identity
verification, not classification) do NOT map cleanly onto this table's
columns and are deliberately left out rather than forced into a misleading
row -- see their own deep-dive write-ups for comparison. BAL-M1
(HarmoFace) will be added once it has its own evaluate.py-equivalent
pipeline producing metrics.json/fairness_metrics.csv in the same shape --
it currently has neither (still training, and no eval script exists yet).

Column conventions (document, don't just compute -- these are judgment
calls a reader needs to be able to check):
  Macro-F1 / Balanced Acc.: mean across the task heads present for that
      dataset (race, gender, age_group for UTKFace/FairFace; race, gender,
      age, skin_tone for BAL-M1).
  <Attribute> gap: mean, across every task head, of that head's
      equalized_odds_diff broken down by the given sensitive attribute
      (race_label/gender_label/age_group/fitzpatrick_approx) -- "how much
      does this demographic attribute associate with outcome disparity
      across everything the model does," not just within its own task.
  Worst intersection: the single largest |equalized_odds_diff| found across
      every intersectional sensitive_attribute (race_gender_label,
      race_age_label, gender_age_label, race_ita_label, ...) and every task
      head, reported as "attribute=value (gap)".
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paired_comparison import _discover_variants  # noqa: E402

EXPERIMENTS_DIR = Path(__file__).resolve().parents[2] / "experiments"

DATASET_TASKS = {
    "utkface": ["race", "gender", "age_group"],
    "fairface": ["race", "gender", "age_group"],
}
GAP_ATTRS = {
    "race": "race_label", "gender": "gender_label",
    "age": "age_group", "skin_tone": "fitzpatrick_approx",
}
INTERSECTIONAL_PREFIXES = ("race_gender", "race_age", "gender_age", "race_ita", "gender_ita", "age_ita")


def _eval_dir_for(path: Path) -> Path:
    return path.parent


def build_row(dataset: str, method: str, eval_dir: Path) -> dict | None:
    metrics_path = eval_dir / "metrics.json"
    fairness_path = eval_dir / "fairness_metrics.csv"
    if not metrics_path.exists() or not fairness_path.exists():
        return None
    classification = json.loads(metrics_path.read_text())
    fairness = pd.read_csv(fairness_path)
    tasks = [t for t in DATASET_TASKS[dataset] if t in classification]
    if not tasks:
        return None

    macro_f1 = sum(classification[t].get("macro_f1", float("nan")) for t in tasks) / len(tasks)
    bal_acc = sum(classification[t].get("balanced_accuracy", float("nan")) for t in tasks) / len(tasks)

    row = {"dataset": dataset, "model": method, "macro_f1": macro_f1, "balanced_accuracy": bal_acc}
    for gap_name, sensitive_col in GAP_ATTRS.items():
        sub = fairness[(fairness["task"].isin(tasks)) & (fairness["sensitive_attribute"] == sensitive_col)]
        row[f"{gap_name}_gap"] = sub["equalized_odds_diff"].mean() if not sub.empty else float("nan")

    inter = fairness[
        fairness["task"].isin(tasks)
        & fairness["sensitive_attribute"].str.startswith(INTERSECTIONAL_PREFIXES, na=False)
    ]
    if not inter.empty:
        worst = inter.loc[inter["equalized_odds_diff"].abs().idxmax()]
        row["worst_intersection"] = f"{worst['sensitive_attribute']}={worst['class']} (gap={worst['equalized_odds_diff']:.3f})"
    else:
        row["worst_intersection"] = ""
    return row


def run() -> pd.DataFrame:
    rows = []
    for dataset in DATASET_TASKS:
        variants = _discover_variants(dataset)
        if not variants:
            print(f"[{dataset}] no evaluated variants found -- skipping")
            continue
        for method, pred_path in variants.items():
            row = build_row(dataset, method, _eval_dir_for(pred_path))
            if row is not None:
                rows.append(row)
            else:
                print(f"[{dataset}/{method}] metrics.json or fairness_metrics.csv missing -- not yet evaluated, skipped")
    df = pd.DataFrame(rows)
    out_path = EXPERIMENTS_DIR / "master_comparison.csv"
    df.to_csv(out_path, index=False)
    print(f"wrote {len(df)} rows -> {out_path}")
    return df


if __name__ == "__main__":
    run()
