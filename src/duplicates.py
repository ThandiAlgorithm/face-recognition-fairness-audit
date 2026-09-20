from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from PIL import Image

from src import image_integrity, paths


def exact_duplicate_flags(manifest: pd.DataFrame) -> pd.DataFrame:
    images = manifest[manifest["file_role"].eq("image_candidate")].copy()
    sizes = images.groupby("sha256")["sha256"].transform("size")
    duplicate_hashes = sorted(images.loc[sizes > 1, "sha256"].unique())
    hash_to_group = {
        value: f"EXACT-{value[:16]}"
        for value in duplicate_hashes
    }
    images["exact_dup_group"] = images["sha256"].map(hash_to_group)

    dataset_sets = images.groupby("sha256")["dataset"].agg(lambda values: set(values))
    cross_hashes = {
        hash_value
        for hash_value, datasets in dataset_sets.items()
        if len(datasets) > 1
    }
    images["cross_dataset_exact"] = images["sha256"].isin(cross_hashes)
    return images[["original_path", "exact_dup_group", "cross_dataset_exact"]]


def _require_imagehash():
    try:
        import imagehash
    except ImportError as exc:
        raise RuntimeError(
            "Near duplicate screening requires imagehash. Install the version pinned in models.yaml."
        ) from exc
    return imagehash


def compute_perceptual_hashes(
    manifest: pd.DataFrame,
    integrity: pd.DataFrame,
    preprocessing_cfg: dict,
    checkpoint_path: Path,
) -> pd.DataFrame:
    imagehash = _require_imagehash()
    images = manifest[manifest["file_role"].eq("image_candidate")][
        ["dataset", "original_path"]
    ].merge(
        integrity[["original_path", "decode_success"]],
        on="original_path",
        how="left",
        validate="one_to_one",
    )
    images = images[images["decode_success"].fillna(False)].copy()

    expected_paths = set(
        images["original_path"].astype(str)
    )

    prior = (
        pd.read_parquet(checkpoint_path)
        if checkpoint_path.exists()
        else pd.DataFrame()
    )

    if not prior.empty:
        prior = prior[
            prior["original_path"]
            .astype(str)
            .isin(expected_paths)
        ].copy()

    completed = (
        set(prior["original_path"].astype(str))
        if not prior.empty
        else set()
    )

    rows = (
        prior.to_dict("records")
        if not prior.empty
        else []
    )

    near_cfg = preprocessing_cfg["duplicates"]["near"]
    phash_size = int(near_cfg["phash_size"])
    crop_cfg = near_cfg["crop_resistant"]

    pending = images[~images["original_path"].isin(completed)]
    for count, item in enumerate(pending.itertuples(index=False), start=1):
        rgb, _ = image_integrity.load_rgb_for_analysis(paths.data_abs(item.original_path))
        pil = Image.fromarray(rgb, mode="RGB")
        phash = imagehash.phash(pil, hash_size=phash_size)
        crop_hash = imagehash.crop_resistant_hash(
            pil,
            limit_segments=int(crop_cfg["limit_segments"]),
            min_segment_size=int(crop_cfg["min_segment_size"]),
            segmentation_image_size=int(crop_cfg["segmentation_image_size"]),
        )
        rows.append(
            {
                "dataset": item.dataset,
                "original_path": item.original_path,
                "phash": str(phash),
                "crop_hash": str(crop_hash),
            }
        )
        if count % 1000 == 0:
            pd.DataFrame(rows).to_parquet(checkpoint_path, index=False)
            print(f"Perceptual hash checkpoint: {len(rows)} rows")

    output = pd.DataFrame(rows)
    output.to_parquet(checkpoint_path, index=False)
    return output


@dataclass
class BKNode:
    value: int
    children: dict[int, "BKNode"] = field(default_factory=dict)


class BKTree:
    def __init__(self):
        self.root: BKNode | None = None

    @staticmethod
    def distance(left: int, right: int) -> int:
        return (left ^ right).bit_count()

    def add(self, value: int) -> None:
        if self.root is None:
            self.root = BKNode(value)
            return
        node = self.root
        while True:
            distance = self.distance(value, node.value)
            if distance == 0:
                return
            child = node.children.get(distance)
            if child is None:
                node.children[distance] = BKNode(value)
                return
            node = child

    def query(self, value: int, radius: int) -> list[tuple[int, int]]:
        if self.root is None:
            return []
        output = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            distance = self.distance(value, node.value)
            if distance <= radius:
                output.append((node.value, distance))
            low = distance - radius
            high = distance + radius
            for edge, child in node.children.items():
                if low <= edge <= high:
                    stack.append(child)
        return output


def _hash_int(text: str) -> int:
    return int(text, 16)


def phash_candidate_pairs(hashes: pd.DataFrame, threshold: int) -> dict[tuple[int, int], dict]:
    values_to_rows = defaultdict(list)
    for index, value in enumerate(hashes["phash"].astype(str)):
        values_to_rows[_hash_int(value)].append(index)

    tree = BKTree()
    for value in values_to_rows:
        tree.add(value)

    candidates: dict[tuple[int, int], dict] = {}
    unique_values = sorted(values_to_rows)
    for left_value in unique_values:
        for right_value, distance in tree.query(left_value, threshold):
            if right_value < left_value:
                continue
            for left_index in values_to_rows[left_value]:
                for right_index in values_to_rows[right_value]:
                    if right_index <= left_index:
                        continue
                    key = (left_index, right_index)
                    candidates[key] = {
                        "phash_distance": int(distance),
                        "crop_matching_regions": 0,
                        "crop_distance_sum": None,
                    }
    return candidates


def crop_candidate_pairs(hashes: pd.DataFrame, preprocessing_cfg: dict) -> dict[tuple[int, int], tuple[int, int]]:
    imagehash = _require_imagehash()
    near_cfg = preprocessing_cfg["duplicates"]["near"]["crop_resistant"]
    segment_cutoff = int(near_cfg["segment_hamming_cutoff"])
    region_cutoff = int(near_cfg["region_cutoff"])

    segment_to_rows = defaultdict(set)
    row_segments = []
    for index, text in enumerate(hashes["crop_hash"].astype(str)):
        multihash = imagehash.hex_to_multihash(text)
        segments = [_hash_int(str(item)) for item in multihash.segment_hashes]
        row_segments.append(segments)
        for segment in segments:
            segment_to_rows[segment].add(index)

    tree = BKTree()
    for segment in segment_to_rows:
        tree.add(segment)

    counts = defaultdict(int)
    distance_sums = defaultdict(int)
    for segment in sorted(segment_to_rows):
        matches = tree.query(segment, segment_cutoff)
        left_rows = segment_to_rows[segment]
        for other_segment, distance in matches:
            if other_segment < segment:
                continue
            right_rows = segment_to_rows[other_segment]
            for left_index in left_rows:
                for right_index in right_rows:
                    if right_index <= left_index:
                        continue
                    key = (left_index, right_index)
                    counts[key] += 1
                    distance_sums[key] += int(distance)

    return {
        key: (counts[key], distance_sums[key])
        for key in counts
        if counts[key] >= region_cutoff
    }


def build_near_duplicate_review(
    hashes: pd.DataFrame,
    preprocessing_cfg: dict,
    existing_review: pd.DataFrame | None = None,
) -> pd.DataFrame:
    threshold = int(preprocessing_cfg["duplicates"]["near"]["phash_hamming_threshold"])
    phash_pairs = phash_candidate_pairs(hashes, threshold)
    crop_pairs = crop_candidate_pairs(hashes, preprocessing_cfg)
    keys = sorted(set(phash_pairs) | set(crop_pairs))

    existing_lookup = {}
    if existing_review is not None and not existing_review.empty:
        for row in existing_review.itertuples(index=False):
            key = tuple(sorted((row.original_path_a, row.original_path_b)))
            existing_lookup[key] = getattr(row, "review_status", "candidate")

    rows = []
    for left_index, right_index in keys:
        left = hashes.iloc[left_index]
        right = hashes.iloc[right_index]
        if left["original_path"] == right["original_path"]:
            continue
        key = tuple(sorted((left["original_path"], right["original_path"])))
        phash = phash_pairs.get((left_index, right_index), {})
        crop = crop_pairs.get((left_index, right_index))
        rows.append(
            {
                "original_path_a": left["original_path"],
                "original_path_b": right["original_path"],
                "dataset_a": left["dataset"],
                "dataset_b": right["dataset"],
                "cross_dataset": left["dataset"] != right["dataset"],
                "phash_distance": phash.get("phash_distance"),
                "crop_matching_regions": crop[0] if crop else 0,
                "crop_distance_sum": crop[1] if crop else None,
                "candidate_method": (
                    "phash+crop" if phash and crop else "phash" if phash else "crop"
                ),
                "review_status": existing_lookup.get(key, "candidate"),
            }
        )
    return pd.DataFrame(rows)


def duplicate_flags_from_review(
    manifest: pd.DataFrame,
    review: pd.DataFrame,
) -> pd.DataFrame:
    exact = exact_duplicate_flags(manifest)
    paths_all = manifest.loc[manifest["file_role"].eq("image_candidate"), "original_path"].tolist()
    result = pd.DataFrame({"original_path": paths_all}).merge(
        exact,
        on="original_path",
        how="left",
        validate="one_to_one",
    )
    result["near_dup_group"] = None
    result["near_dup_status"] = "none"
    result["cross_dataset_dup"] = result["cross_dataset_exact"].fillna(False)
    result["cross_dataset_dup_status"] = result["cross_dataset_exact"].map(
        {True: "confirmed", False: "none"}
    )

    if review.empty:
        return result.drop(columns=["cross_dataset_exact"])

    parent = {path: path for path in paths_all}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    accepted_statuses = {"confirmed"}
    for row in review.itertuples(index=False):
        if row.review_status in accepted_statuses:
            union(row.original_path_a, row.original_path_b)

    groups = defaultdict(list)
    for path in paths_all:
        groups[find(path)].append(path)
    path_to_group = {}
    group_number = 1
    for members in groups.values():
        if len(members) > 1:
            group_id = f"NEAR-{group_number:06d}"
            group_number += 1
            for path in members:
                path_to_group[path] = group_id

    status_priority = {
        "none": 0,
        "false_alarm": 1,
        "candidate": 2,
        "uncertain": 3,
        "confirmed": 4,
    }
    path_status = defaultdict(lambda: "none")
    cross_status = defaultdict(lambda: "none")
    for row in review.itertuples(index=False):
        for path in [row.original_path_a, row.original_path_b]:
            if status_priority[row.review_status] > status_priority[path_status[path]]:
                path_status[path] = row.review_status
        if row.cross_dataset:
            for path in [row.original_path_a, row.original_path_b]:
                if status_priority[row.review_status] > status_priority[cross_status[path]]:
                    cross_status[path] = row.review_status

    result["near_dup_group"] = result["original_path"].map(path_to_group)
    result["near_dup_status"] = result["original_path"].map(path_status)

    for index, row in result.iterrows():
        cross = cross_status[row["original_path"]]
        if cross != "none":
            result.at[index, "cross_dataset_dup_status"] = cross
            result.at[index, "cross_dataset_dup"] = cross == "confirmed"

    return result.drop(columns=["cross_dataset_exact"])
