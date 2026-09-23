from collections import defaultdict
from pathlib import Path
import random

import pandas as pd


MASTER_PATH = Path("metadata/master_annotations.parquet")
OUTPUT_PATH = Path("metadata/utkface_split_grouped.csv")

SEED = 26041

TARGETS = {
    "train": 16876,
    "val": 3616,
    "test": 3616,
}


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

        if ra == rb:
            return

        # Deterministic root choice.
        if ra < rb:
            self.parent[rb] = ra
        else:
            self.parent[ra] = rb


def union_group(df, col, uf):
    """
    Union every image sharing the same non-null duplicate group.
    """

    grouped = (
        df.dropna(subset=[col])
        .groupby(col)["image_id"]
    )

    for _, ids in grouped:
        ids = list(ids)

        if len(ids) < 2:
            continue

        first = ids[0]

        for other in ids[1:]:
            uf.union(first, other)


def leakage_count(df, group_col):
    grouped = (
        df.dropna(subset=[group_col])
        .groupby(group_col)["split"]
        .nunique()
    )

    return int((grouped > 1).sum())


def main():

    master = pd.read_parquet(
        MASTER_PATH,
        columns=[
            "image_id",
            "dataset",
            "exact_dup_group",
            "near_dup_group",
        ],
    )

    utk = master[
        master["dataset"]
        .astype(str)
        .str.lower()
        .eq("utkface")
    ].copy()

    if len(utk) != 24108:
        raise RuntimeError(
            f"Expected 24108 UTKFace rows, got {len(utk)}"
        )

    if utk["image_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate UTKFace image_id values found"
        )

    # Build one connected-component graph using both exact duplicates
    # and confirmed near-duplicate groups, to handle transitive overlap.

    uf = UnionFind()

    for image_id in utk["image_id"]:
        uf.add(image_id)

    union_group(
        utk,
        "exact_dup_group",
        uf,
    )

    union_group(
        utk,
        "near_dup_group",
        uf,
    )

    components = defaultdict(list)

    for image_id in utk["image_id"]:
        root = uf.find(image_id)
        components[root].append(image_id)

    component_list = [
        sorted(members)
        for members in components.values()
    ]

    component_list.sort(
        key=lambda x: x[0]
    )

    multi = [
        x for x in component_list
        if len(x) > 1
    ]

    singletons = [
        x for x in component_list
        if len(x) == 1
    ]

    print("UTKFace images:", len(utk))
    print("Connected components:", len(component_list))
    print("Multi-image components:", len(multi))
    print("Singleton components:", len(singletons))

    print(
        "Largest component:",
        max(len(x) for x in component_list),
    )

    # Deterministic ordering: largest duplicate components placed first,
    # random tie-breaking uses the frozen seed.

    rng = random.Random(SEED)

    decorated = [
        (
            -len(component),
            rng.random(),
            component,
        )
        for component in multi
    ]

    decorated.sort(
        key=lambda x: (
            x[0],
            x[1],
        )
    )

    multi = [
        x[2]
        for x in decorated
    ]

    remaining = TARGETS.copy()

    assignment = {}

    # Place multi-image components, choosing the split with the
    # greatest proportional remaining capacity that can fit it.

    for component in multi:

        size = len(component)

        eligible = [
            split
            for split in TARGETS
            if remaining[split] >= size
        ]

        if not eligible:
            raise RuntimeError(
                "Could not place duplicate component "
                f"of size {size} without exceeding targets"
            )

        chosen = max(
            eligible,
            key=lambda split: (
                remaining[split] / TARGETS[split],
                remaining[split],
                split,
            ),
        )

        for image_id in component:
            assignment[image_id] = chosen

        remaining[chosen] -= size

    # Singletons fill the exact remaining counts.

    singleton_ids = [
        x[0]
        for x in singletons
    ]

    rng.shuffle(singleton_ids)

    slots = []

    for split in ["train", "val", "test"]:
        slots.extend(
            [split] * remaining[split]
        )

    if len(slots) != len(singleton_ids):
        raise RuntimeError(
            "Remaining split capacity does not equal "
            "number of singleton images"
        )

    rng.shuffle(slots)

    for image_id, split in zip(
        singleton_ids,
        slots,
    ):
        assignment[image_id] = split

    # Build output.

    out = utk[
        ["image_id"]
    ].copy()

    out["split"] = (
        out["image_id"]
        .map(assignment)
    )

    if out["split"].isna().any():
        raise RuntimeError(
            "Some images received no split"
        )

    counts = (
        out["split"]
        .value_counts()
        .to_dict()
    )

    if counts != TARGETS:
        raise RuntimeError(
            "Final counts differ from targets.\n"
            f"Expected: {TARGETS}\n"
            f"Actual:   {counts}"
        )

    # Leakage validation before writing.

    audit = utk.merge(
        out,
        on="image_id",
        how="left",
        validate="one_to_one",
    )

    exact_leaks = leakage_count(
        audit,
        "exact_dup_group",
    )

    near_leaks = leakage_count(
        audit,
        "near_dup_group",
    )

    print()
    print("Final split counts:")
    print(
        out["split"]
        .value_counts()
        .to_string()
    )

    print()
    print(
        "Exact duplicate groups crossing splits:",
        exact_leaks,
    )

    print(
        "Confirmed near-duplicate groups crossing splits:",
        near_leaks,
    )

    if exact_leaks != 0:
        raise RuntimeError(
            "Exact duplicate leakage remains"
        )

    if near_leaks != 0:
        raise RuntimeError(
            "Confirmed near-duplicate leakage remains"
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print("=" * 70)
    print("GROUP-AWARE UTKFACE SPLIT PASSED")
    print("=" * 70)
    print("Seed:", SEED)
    print("Rows:", len(out))
    print("Exact leakage: 0")
    print("Confirmed near-duplicate leakage: 0")
    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
