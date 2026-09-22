from pathlib import Path
from collections import defaultdict
import hashlib
import shutil

import pandas as pd


FLAGS_PATH = Path("metadata/duplicate_flags.parquet")
REVIEW_PATH = Path("metadata/near_duplicate_review.csv")
BACKUP_PATH = Path(
    "metadata/duplicate_flags_before_human_review_finalize.parquet"
)


# Union-find for confirmed duplicate connected components

class UnionFind:
    def __init__(self):
        self.parent = {}

    def add(self, x):
        if x not in self.parent:
            self.parent[x] = x

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a, b):
        self.add(a)
        self.add(b)

        ra = self.find(a)
        rb = self.find(b)

        if ra != rb:
            # deterministic root
            if ra < rb:
                self.parent[rb] = ra
            else:
                self.parent[ra] = rb


def make_group_id(members):
    """
    Stable group ID based only on sorted member paths.
    """
    text = "|".join(sorted(members))
    digest = hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()[:16]

    return f"near-{digest}"


def bool_series(series):
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )


def main():

    flags = pd.read_parquet(FLAGS_PATH)
    review = pd.read_csv(REVIEW_PATH)

    required_flags = {
        "original_path",
        "exact_dup_group",
        "near_dup_group",
        "near_dup_status",
        "cross_dataset_dup",
        "cross_dataset_dup_status",
    }

    required_review = {
        "original_path_a",
        "original_path_b",
        "cross_dataset",
        "review_status",
    }

    missing = required_flags - set(flags.columns)
    if missing:
        raise RuntimeError(
            f"duplicate_flags missing columns: {sorted(missing)}"
        )

    missing = required_review - set(review.columns)
    if missing:
        raise RuntimeError(
            f"review CSV missing columns: {sorted(missing)}"
        )

    # Basic integrity checks

    if len(flags) != 135039:
        raise RuntimeError(
            f"Expected 135039 flag rows, got {len(flags)}"
        )

    if flags["original_path"].duplicated().any():
        raise RuntimeError(
            "duplicate_flags contains duplicate original_path values"
        )

    review["review_status"] = (
        review["review_status"]
        .fillna("candidate")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    allowed = {
        "confirmed",
        "false_alarm",
        "uncertain",
        "candidate",
    }

    unknown = (
        set(review["review_status"].unique())
        - allowed
    )

    if unknown:
        raise RuntimeError(
            f"Unknown review statuses: {sorted(unknown)}"
        )

    print("Review status:")
    print(
        review["review_status"]
        .value_counts()
        .to_string()
    )

    # Current completed review state.
    expected = {
        "false_alarm": 31009,
        "candidate": 29328,
        "confirmed": 1210,
        "uncertain": 6,
    }

    actual = (
        review["review_status"]
        .value_counts()
        .to_dict()
    )

    if actual != expected:
        raise RuntimeError(
            "Review counts differ from the completed review state.\n"
            f"Expected: {expected}\n"
            f"Actual:   {actual}"
        )

    # Ensure every reviewed path belongs to canonical flags

    canonical_paths = set(flags["original_path"])

    review_paths = (
        set(review["original_path_a"])
        | set(review["original_path_b"])
    )

    missing_paths = sorted(
        review_paths - canonical_paths
    )

    if missing_paths:
        raise RuntimeError(
            "Review file contains paths missing from "
            f"duplicate_flags. First examples: {missing_paths[:10]}"
        )

    # Preserve exact duplicate results

    exact_before = flags["exact_dup_group"].notna().sum()

    # Build near-duplicate groups only from confirmed pairs

    confirmed = review[
        review["review_status"].eq("confirmed")
    ].copy()

    uf = UnionFind()

    for row in confirmed.itertuples(index=False):
        uf.union(
            row.original_path_a,
            row.original_path_b,
        )

    components = defaultdict(list)

    for path in uf.parent:
        components[uf.find(path)].append(path)

    # Stable order for reproducibility.
    component_list = sorted(
        [
            sorted(members)
            for members in components.values()
        ],
        key=lambda members: members[0],
    )

    path_to_group = {}

    for members in component_list:
        group_id = make_group_id(members)

        for path in members:
            path_to_group[path] = group_id

    # Reset old perceptual fields.
    # Exact duplicate information is NOT changed.
    flags["near_dup_group"] = pd.Series(
        pd.NA,
        index=flags.index,
        dtype="string",
    )

    flags["near_dup_status"] = "none"

    flags["cross_dataset_dup"] = False
    flags["cross_dataset_dup_status"] = "none"

    # Write confirmed group membership.
    flags["near_dup_group"] = (
        flags["original_path"]
        .map(path_to_group)
        .astype("string")
    )

    # Image-level review status.
    # Precedence: confirmed > uncertain > candidate > none.
    # false_alarm does not make an image a duplicate.

    precedence = {
        "none": 0,
        "candidate": 1,
        "uncertain": 2,
        "confirmed": 3,
    }

    near_status = defaultdict(lambda: "none")

    for row in review.itertuples(index=False):

        status = row.review_status

        if status == "false_alarm":
            continue

        for path in [
            row.original_path_a,
            row.original_path_b,
        ]:
            if (
                precedence[status]
                > precedence[near_status[path]]
            ):
                near_status[path] = status

    flags["near_dup_status"] = (
        flags["original_path"]
        .map(near_status)
        .fillna("none")
    )

    # Cross-dataset status

    cross_mask = bool_series(
        review["cross_dataset"]
    )

    cross_review = review[cross_mask].copy()

    cross_status = defaultdict(lambda: "none")

    for row in cross_review.itertuples(index=False):

        status = row.review_status

        if status == "false_alarm":
            continue

        for path in [
            row.original_path_a,
            row.original_path_b,
        ]:
            if (
                precedence[status]
                > precedence[cross_status[path]]
            ):
                cross_status[path] = status

    flags["cross_dataset_dup_status"] = (
        flags["original_path"]
        .map(cross_status)
        .fillna("none")
    )

    flags["cross_dataset_dup"] = (
        flags["cross_dataset_dup_status"]
        .eq("confirmed")
    )

    # Final validation

    exact_after = flags["exact_dup_group"].notna().sum()

    if exact_before != exact_after:
        raise RuntimeError(
            "Exact duplicate information changed unexpectedly"
        )

    if len(flags) != 135039:
        raise RuntimeError(
            "Row count changed during finalization"
        )

    if flags["original_path"].duplicated().any():
        raise RuntimeError(
            "Duplicate original_path created during finalization"
        )

    # Backup old file once.
    if not BACKUP_PATH.exists():
        shutil.copy2(
            FLAGS_PATH,
            BACKUP_PATH,
        )

    # Atomic-ish replacement.
    tmp = FLAGS_PATH.with_suffix(".tmp.parquet")

    flags.to_parquet(
        tmp,
        index=False,
    )

    tmp.replace(FLAGS_PATH)

    # Report

    print()
    print("=" * 60)
    print("DUPLICATE FINALIZATION COMPLETE")
    print("=" * 60)

    print(
        "Rows:",
        len(flags),
    )

    print(
        "Exact duplicate image rows:",
        exact_after,
    )

    print(
        "Confirmed near-duplicate pairs:",
        len(confirmed),
    )

    print(
        "Confirmed near-duplicate groups:",
        len(component_list),
    )

    print(
        "Images in confirmed near-duplicate groups:",
        flags["near_dup_group"].notna().sum(),
    )

    print()
    print("Image-level near duplicate status:")
    print(
        flags["near_dup_status"]
        .value_counts()
        .to_string()
    )

    print()
    print("Image-level cross-dataset status:")
    print(
        flags["cross_dataset_dup_status"]
        .value_counts()
        .to_string()
    )

    print()
    print(
        "Confirmed cross-dataset duplicate images:",
        int(flags["cross_dataset_dup"].sum()),
    )

    print()
    print("Saved:", FLAGS_PATH)
    print("Backup:", BACKUP_PATH)


if __name__ == "__main__":
    main()
