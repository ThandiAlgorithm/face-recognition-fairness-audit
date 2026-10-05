"""Step 5: frozen 80/10/10 train/val/test splits.

Split seed is fixed at 42 and is intentionally independent of the later
per-run training seeds (seed.py) -- the split never changes when the
training seed does.

LFW: identity-disjoint. Every image of a person goes to exactly one split;
stratified (where group size allows) by ITA group x LFWA+ Male attribute,
taken as the per-identity majority vote across that identity's images.

FairFace / UTKFace: no reliable identity IDs, so the split is image-level,
stratified by race x gender x age_group with progressive backoff
(race x gender x age_group -> race x gender -> race -> unstratified) when a
stratum is too small to divide 80/10/10 without starving a split. Every
backoff is logged, never silently absorbed.

All datasets: known duplicate clusters (Phase 0's own exact_dup_group /
near_dup_group) are merged into a single unit via union-find before
splitting, so near-duplicate images can never land in different splits.
For LFW this union is combined with identity grouping, so a duplicate pair
that happened to carry two different identity labels still can't leak.

Once written, a split is frozen: rerunning refuses to overwrite an existing
manifest unless the caller explicitly asks to replace it. Every later
experiment must re-verify the manifest's saved SHA256 before training, not
regenerate the split.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

RECORDS_DIR = Path(__file__).resolve().parents[2] / "data" / "records"
EXPERIMENTS_DIR = Path(__file__).resolve().parents[2] / "experiments"

SPLIT_SEED = 42
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.8, 0.1, 0.1
# A stratum needs enough groups that 10% rounds to >= 1 in both val and
# test; below this it can't be split 80/10/10 without starving a split, so
# it backs off to a coarser stratification level instead.
MIN_STRATUM_SIZE = 10
# Cross-tab cells in the split-quality report below this count are flagged
# as poorly represented, but never used to trigger a re-split -- re-rolling
# until a preferred result appears would make split selection itself a form
# of optimization.
SMALL_CELL_THRESHOLD = 5


class SplitAlreadyFrozenError(RuntimeError):
    pass


def ita_group_label(skin_group_ita: str | float) -> str | None:
    """Relabels Phase 0's frozen GOV-002 binary ITA split (skin_group_ita:
    'lighter'/'darker', threshold ITA > 28) to the '>=28'/'<28' convention
    used in this pipeline's reports. Same frozen boundary, different label
    text -- not a recomputation."""
    if pd.isna(skin_group_ita):
        return None
    return "≥28" if skin_group_ita == "lighter" else "<28"


def build_groups(df: pd.DataFrame, group_key_cols: list[str]) -> pd.Series:
    """Union-find over rows: two rows land in the same group if they share
    a non-null value in any of group_key_cols. Rows touched by none of the
    keys get their own singleton group."""
    n = len(df)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for col in group_key_cols:
        seen: dict = {}
        for i, val in enumerate(df[col].to_numpy()):
            if pd.isna(val):
                continue
            if val in seen:
                union(seen[val], i)
            else:
                seen[val] = i

    roots = [find(i) for i in range(n)]
    return pd.Series([f"g{r}" for r in roots], index=df.index)


def _mode_or_na(series: pd.Series):
    series = series.dropna()
    if series.empty:
        return np.nan
    return series.mode(dropna=True).iloc[0]


def group_level_frame(df: pd.DataFrame, group_col: str, strat_cols: list[str]) -> pd.DataFrame:
    """Per-group stratification key via majority vote. The vast majority of
    groups are singletons (no duplicate cluster), where "majority vote of
    one row" is just that row -- handled by a vectorised fast path so this
    doesn't have to pay pandas' slow per-group Python-callback cost for
    ~100k singleton groups to resolve a handful of real multi-row clusters."""
    sizes = df.groupby(group_col)[group_col].transform("size")
    singletons = df.loc[sizes == 1, [group_col, *strat_cols]]
    multi = df.loc[sizes > 1]

    parts = [singletons]
    if not multi.empty:
        parts.append(multi.groupby(group_col)[strat_cols].agg(_mode_or_na).reset_index())
    return pd.concat(parts, ignore_index=True)


def stratified_group_split(
    groups: pd.DataFrame,
    candidate_levels: list[list[str]],
    group_col: str = "group_id",
    seed: int = SPLIT_SEED,
    min_stratum_size: int = MIN_STRATUM_SIZE,
) -> tuple[dict[str, str], list[dict]]:
    """Assigns every group to train/val/test. Tries candidate_levels[0]
    first; any stratum with fewer than min_stratum_size groups is pooled
    and retried at the next (coarser) level; anything still too small after
    the last level is split unstratified. Returns (assignments, fallback_log)."""
    rng = np.random.default_rng(seed)
    assignments: dict[str, str] = {}
    fallback_log: list[dict] = []
    remaining = groups.copy()

    def assign(group_ids: list[str]) -> None:
        shuffled = list(group_ids)
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = round(n * TRAIN_FRAC)
        n_val = round(n * VAL_FRAC)
        n_train = min(n_train, n)
        n_val = min(n_val, n - n_train)
        for gid in shuffled[:n_train]:
            assignments[gid] = "train"
        for gid in shuffled[n_train : n_train + n_val]:
            assignments[gid] = "val"
        for gid in shuffled[n_train + n_val :]:
            assignments[gid] = "test"

    for level_idx, cols in enumerate(candidate_levels):
        if remaining.empty:
            break
        key = (
            list(zip(*[remaining[c].astype(str) for c in cols]))
            if cols
            else [("__all__",)] * len(remaining)
        )
        remaining = remaining.assign(_stratum=key)
        sizes = remaining.groupby("_stratum")[group_col].transform("count")

        big_enough = remaining[sizes >= min_stratum_size]
        for stratum_key, stratum_df in big_enough.groupby("_stratum"):
            assign(stratum_df[group_col].tolist())

        too_small = remaining[sizes < min_stratum_size]
        is_last_level = level_idx == len(candidate_levels) - 1
        if len(too_small) and not is_last_level:
            fallback_log.append(
                {
                    "from_level": cols,
                    "to_level": candidate_levels[level_idx + 1],
                    "n_small_strata": too_small["_stratum"].nunique(),
                    "n_groups_affected": len(too_small),
                    "affected_strata": sorted({str(s) for s in too_small["_stratum"]}),
                }
            )
            remaining = too_small.drop(columns="_stratum")
        elif len(too_small) and is_last_level:
            fallback_log.append(
                {
                    "from_level": cols,
                    "to_level": "unstratified",
                    "n_small_strata": too_small["_stratum"].nunique(),
                    "n_groups_affected": len(too_small),
                    "affected_strata": sorted({str(s) for s in too_small["_stratum"]}),
                }
            )
            assign(too_small[group_col].tolist())
            remaining = remaining.iloc[0:0]
        else:
            remaining = remaining.iloc[0:0]

    return assignments, fallback_log


def quality_report(manifest: pd.DataFrame) -> dict:
    report: dict = {}
    dims = {
        "race": ["race_label"],
        "gender": ["gender_label"],
        "age_group": ["age_group"],
        "ita_group_report": ["ita_group_report"],
        "race_x_gender": ["race_label", "gender_label"],
        "ita_x_gender": ["ita_group_report", "gender_label"],
        "ita_x_age": ["ita_group_report", "age_group"],
        "race_x_ita": ["race_label", "ita_group_report"],
    }
    small_cells: list[dict] = []
    for name, cols in dims.items():
        if any(c not in manifest.columns for c in cols):
            continue
        table = (
            manifest.groupby(["split", *cols], dropna=False)
            .size()
            .rename("count")
            .reset_index()
        )
        table["pct_of_split"] = table["count"] / table.groupby("split")["count"].transform("sum") * 100
        report[name] = table.to_dict(orient="records")
        for row in table.to_dict(orient="records"):
            if row["split"] in ("val", "test") and row["count"] < SMALL_CELL_THRESHOLD:
                small_cells.append({"dimension": name, **row})
    report["small_cell_warnings"] = small_cells
    return report


def freeze_check(manifest_path: Path, force: bool = False) -> None:
    if manifest_path.exists() and not force:
        raise SplitAlreadyFrozenError(
            f"{manifest_path} already exists. The split is frozen once written -- "
            "pass force=True (and document why) to replace it deliberately."
        )


def write_manifest(manifest: pd.DataFrame, out_dir: Path, fallback_log: list[dict], report: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "split_manifest.csv"
    freeze_check(manifest_path)

    manifest.to_csv(manifest_path, index=False)
    sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    (out_dir / "split_manifest.sha256").write_text(sha256 + "\n")
    (out_dir / "split_fallback_log.json").write_text(json.dumps(fallback_log, indent=2, default=str))
    (out_dir / "split_quality_report.json").write_text(json.dumps(report, indent=2, default=str))


def split_lfw(records: pd.DataFrame) -> tuple[pd.DataFrame, list[dict], dict]:
    df = records.copy()
    df["group_id"] = build_groups(df, ["exact_dup_group", "near_dup_group", "identity_id"])
    df["ita_group_report"] = df["ita_group"].map(ita_group_label)

    groups = group_level_frame(df, "group_id", ["ita_group_report", "attr_Male"])
    assignments, fallback_log = stratified_group_split(
        groups,
        candidate_levels=[["ita_group_report", "attr_Male"], ["attr_Male"], []],
    )

    df["split"] = df["group_id"].map(assignments)
    manifest = df[["image_path", "identity_id", "group_id", "split", "race_label",
                    "gender_label", "age_group", "ita_value", "ita_group_report"]].rename(
        columns={"ita_group_report": "ita_group"}
    )
    report = quality_report(df.rename(columns={"ita_group_report": "ita_group_report"}))
    return manifest, fallback_log, report


def split_image_level(records: pd.DataFrame) -> tuple[pd.DataFrame, list[dict], dict]:
    df = records.copy()
    df["group_id"] = build_groups(df, ["exact_dup_group", "near_dup_group"])
    df["ita_group_report"] = df["ita_group"].map(ita_group_label)

    groups = group_level_frame(df, "group_id", ["race_label", "gender_label", "age_group"])
    assignments, fallback_log = stratified_group_split(
        groups,
        candidate_levels=[
            ["race_label", "gender_label", "age_group"],
            ["race_label", "gender_label"],
            ["race_label"],
            [],
        ],
    )

    df["split"] = df["group_id"].map(assignments)
    manifest = df[["image_path", "identity_id", "group_id", "split", "race_label",
                    "gender_label", "age_group", "ita_value", "ita_group_report"]].rename(
        columns={"ita_group_report": "ita_group"}
    )
    report = quality_report(df)
    return manifest, fallback_log, report


def main() -> None:
    for dataset in ["lfw", "fairface", "utkface"]:
        print(f"Splitting {dataset}...")
        records = pd.read_csv(RECORDS_DIR / f"{dataset}_records.csv")
        usable = records[records["image_path"].notna()].copy()

        if dataset == "lfw":
            manifest, fallback_log, report = split_lfw(usable)
        else:
            manifest, fallback_log, report = split_image_level(usable)

        out_dir = EXPERIMENTS_DIR / dataset
        write_manifest(manifest, out_dir, fallback_log, report)

        split_counts = manifest["split"].value_counts().to_dict()
        print(f"  {len(manifest)} images -> {split_counts}")
        if dataset == "lfw":
            id_counts = manifest.groupby("split")["identity_id"].nunique().to_dict()
            print(f"  identities -> {id_counts}")
        if fallback_log:
            print(f"  {len(fallback_log)} stratification fallback(s) -- see split_fallback_log.json")
        if report["small_cell_warnings"]:
            print(f"  {len(report['small_cell_warnings'])} small-cell warning(s) -- see split_quality_report.json")
        print(f"  written to {out_dir.relative_to(EXPERIMENTS_DIR.parent)}")


if __name__ == "__main__":
    main()
