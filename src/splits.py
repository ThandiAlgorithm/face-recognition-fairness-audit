from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import random

import pandas as pd


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def add(self, x: str) -> None:
        if x not in self.parent:
            self.parent[x] = x

    def find(self, x: str) -> str:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: str, b: str) -> None:
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


def _union_group(
    df: pd.DataFrame,
    group_col: str,
    uf: _UnionFind,
) -> None:
    """
    Join all images belonging to the same duplicate group.
    """
    grouped = (
        df.dropna(subset=[group_col])
        .groupby(group_col)["image_id"]
    )

    for _, ids in grouped:
        ids = list(ids)

        if len(ids) < 2:
            continue

        first = ids[0]

        for other in ids[1:]:
            uf.union(first, other)


def _leakage_count(
    df: pd.DataFrame,
    group_col: str,
) -> int:
    grouped = (
        df.dropna(subset=[group_col])
        .groupby(group_col)["split"]
        .nunique()
    )

    return int((grouped > 1).sum())


def build_utkface_split(
    master: pd.DataFrame,
    preprocessing_cfg: dict,
) -> pd.DataFrame:
    """
    Build the deterministic UTKFace split.

    Exact duplicates and confirmed near-duplicate groups are
    kept wholly inside one split.

    The frozen random seed is read from preprocessing config.
    """

    cfg = preprocessing_cfg["splits"]["utkface"]

    seed = int(cfg["seed"])

    fractions = cfg.get(
        "fractions",
        {
            "train": 0.70,
            "val": 0.15,
            "test": 0.15,
        },
    )

    required_fraction_keys = {
        "train",
        "val",
        "test",
    }

    if set(fractions) != required_fraction_keys:
        raise RuntimeError(
            "UTKFace split fractions must contain exactly "
            "train, val and test."
        )

    total_fraction = sum(
        float(value)
        for value in fractions.values()
    )

    if abs(total_fraction - 1.0) > 1e-9:
        raise RuntimeError(
            "UTKFace split fractions must sum to 1."
        )

    required_columns = {
        "image_id",
        "dataset",
        "exact_dup_group",
        "near_dup_group",
    }

    missing = required_columns - set(master.columns)

    if missing:
        raise RuntimeError(
            "Master table missing columns required for "
            f"group-aware splitting: {sorted(missing)}"
        )

    subset = master[
        master["dataset"].eq("utkface")
    ][
        [
            "image_id",
            "exact_dup_group",
            "near_dup_group",
        ]
    ].copy()

    if subset["image_id"].isna().any():
        raise RuntimeError(
            "UTKFace contains null image_id values."
        )

    if subset["image_id"].duplicated().any():
        raise RuntimeError(
            "UTKFace contains duplicate image_id values."
        )

    n_total = len(subset)

    n_train = int(
        round(
            n_total
            * float(fractions["train"])
        )
    )

    n_val = int(
        round(
            n_total
            * float(fractions["val"])
        )
    )

    n_test = (
        n_total
        - n_train
        - n_val
    )

    targets = {
        "train": n_train,
        "val": n_val,
        "test": n_test,
    }

    # --------------------------------------------------------
    # Build connected components using BOTH exact duplicates
    # and confirmed near-duplicate groups.
    #
    # near_dup_group contains only confirmed near duplicates.
    # --------------------------------------------------------

    uf = _UnionFind()

    for image_id in subset["image_id"]:
        uf.add(image_id)

    _union_group(
        subset,
        "exact_dup_group",
        uf,
    )

    _union_group(
        subset,
        "near_dup_group",
        uf,
    )

    components: dict[str, list[str]] = defaultdict(list)

    for image_id in subset["image_id"]:
        root = uf.find(image_id)
        components[root].append(image_id)

    component_list = [
        sorted(members)
        for members in components.values()
    ]

    component_list.sort(
        key=lambda members: members[0]
    )

    multi = [
        members
        for members in component_list
        if len(members) > 1
    ]

    singletons = [
        members
        for members in component_list
        if len(members) == 1
    ]

    # --------------------------------------------------------
    # Deterministic ordering.
    #
    # This deliberately matches the validated grouped-split
    # generation procedure.
    # --------------------------------------------------------

    rng = random.Random(seed)

    decorated = [
        (
            -len(component),
            rng.random(),
            component,
        )
        for component in multi
    ]

    decorated.sort(
        key=lambda item: (
            item[0],
            item[1],
        )
    )

    multi = [
        item[2]
        for item in decorated
    ]

    remaining = targets.copy()

    assignment: dict[str, str] = {}

    # --------------------------------------------------------
    # Place duplicate-connected components first.
    #
    # A component may never be split.
    # --------------------------------------------------------

    for component in multi:
        size = len(component)

        eligible = [
            split
            for split in targets
            if remaining[split] >= size
        ]

        if not eligible:
            raise RuntimeError(
                "Could not place duplicate component "
                f"of size {size} without exceeding "
                "target split counts."
            )

        chosen = max(
            eligible,
            key=lambda split: (
                remaining[split]
                / targets[split],
                remaining[split],
                split,
            ),
        )

        for image_id in component:
            assignment[image_id] = chosen

        remaining[chosen] -= size

    # --------------------------------------------------------
    # Singletons fill the exact remaining capacities.
    # --------------------------------------------------------

    singleton_ids = [
        component[0]
        for component in singletons
    ]

    rng.shuffle(singleton_ids)

    slots: list[str] = []

    for split in [
        "train",
        "val",
        "test",
    ]:
        slots.extend(
            [split] * remaining[split]
        )

    if len(slots) != len(singleton_ids):
        raise RuntimeError(
            "Remaining split capacity does not match "
            "the number of singleton images."
        )

    rng.shuffle(slots)

    for image_id, split in zip(
        singleton_ids,
        slots,
    ):
        assignment[image_id] = split

    output = subset[
        ["image_id"]
    ].copy()

    output["split"] = (
        output["image_id"]
        .map(assignment)
    )

    if output["split"].isna().any():
        raise RuntimeError(
            "Some UTKFace images received no split."
        )

    counts = (
        output["split"]
        .value_counts()
        .to_dict()
    )

    if counts != targets:
        raise RuntimeError(
            "Generated UTKFace split counts differ "
            "from target counts.\n"
            f"Expected: {targets}\n"
            f"Actual:   {counts}"
        )

    # --------------------------------------------------------
    # Mandatory leakage check.
    # --------------------------------------------------------

    audit = subset.merge(
        output,
        on="image_id",
        how="left",
        validate="one_to_one",
    )

    exact_leaks = _leakage_count(
        audit,
        "exact_dup_group",
    )

    near_leaks = _leakage_count(
        audit,
        "near_dup_group",
    )

    if exact_leaks:
        raise RuntimeError(
            "Generated UTKFace split contains "
            f"{exact_leaks} exact-duplicate groups "
            "crossing splits."
        )

    if near_leaks:
        raise RuntimeError(
            "Generated UTKFace split contains "
            f"{near_leaks} confirmed near-duplicate "
            "groups crossing splits."
        )

    return output


def lock_utkface_split(
    master: pd.DataFrame,
    preprocessing_cfg: dict,
    split_path: Path,
) -> pd.DataFrame:
    """
    Generate the deterministic group-aware split and ensure
    any existing locked split matches it exactly.
    """

    generated = build_utkface_split(
        master,
        preprocessing_cfg,
    )

    if split_path.exists():
        existing = pd.read_csv(split_path)

        required = {
            "image_id",
            "split",
        }

        missing = required - set(existing.columns)

        if missing:
            raise RuntimeError(
                "Existing UTKFace split missing columns: "
                f"{sorted(missing)}"
            )

        existing = existing[
            ["image_id", "split"]
        ].copy()

        check = generated.merge(
            existing,
            on="image_id",
            suffixes=("_new", "_old"),
            how="outer",
            indicator=True,
            validate="one_to_one",
        )

        mismatch = check[
            check["_merge"].ne("both")
            | check["split_new"].ne(
                check["split_old"]
            )
        ]

        if len(mismatch):
            raise RuntimeError(
                "Existing UTKFace split does not match "
                "the deterministic duplicate-group-aware "
                "frozen split."
            )

        return existing

    split_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    generated.to_csv(
        split_path,
        index=False,
    )

    return generated
