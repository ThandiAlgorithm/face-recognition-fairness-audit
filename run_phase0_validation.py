from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config, master_table, paths


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()




def table_content_hash(path: Path) -> str:
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    else:
        return file_sha256(path)

    frame = frame.copy()
    for column in frame.columns:
        if frame[column].dtype == "object":
            frame[column] = frame[column].map(
                lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
                if isinstance(value, (list, dict, tuple))
                else "<NULL>" if pd.isna(value)
                else str(value)
            )
    frame = frame.reindex(sorted(frame.columns), axis=1)
    sort_columns = [
        column
        for column in ["image_id", "pair_id", "original_path"]
        if column in frame.columns
    ]
    if sort_columns:
        frame = frame.sort_values(sort_columns).reset_index(drop=True)
    payload = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

def add(checks: list[dict], name: str, passed: bool, detail: object) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": str(detail)})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-reference", action="store_true")
    parser.add_argument("--compare-reference", action="store_true")
    args = parser.parse_args()

    paths.ensure_dirs()
    preprocessing = config.load_preprocessing()
    config.validate_preprocessing(preprocessing)
    exclusions = config.load_exclusions()

    checks = []
    required = {
        "file_manifest": paths.METADATA_DIR / "file_manifest.parquet",
        "master_annotations": paths.METADATA_DIR / "master_annotations.parquet",
        "lfw_pairs": paths.METADATA_DIR / "lfw_pairs.parquet",
        "utkface_split": paths.METADATA_DIR / "utkface_split.csv",
        "duplicate_flags": paths.METADATA_DIR / "duplicate_flags.parquet",
        "near_duplicate_review": paths.METADATA_DIR / "near_duplicate_review.csv",
        "alignment_qa_review": paths.METADATA_DIR / "alignment_qa_review.csv",
        "environment_runtime": paths.METADATA_DIR / "environment_runtime.json",
        "environment_lock": paths.DOCS_ENV_DIR / "requirements-lock.txt",
    }
    for name, path in required.items():
        add(checks, f"{name} exists", path.exists(), path)

    if not all(path.exists() for path in required.values()):
        report = {"status": "NOT_READY", "checks": checks}
        print(json.dumps(report, indent=2))
        raise RuntimeError("Required Phase 0 outputs are missing.")

    manifest = pd.read_parquet(required["file_manifest"])
    master = pd.read_parquet(required["master_annotations"])
    pairs = pd.read_parquet(required["lfw_pairs"])
    review = pd.read_csv(required["near_duplicate_review"])
    qa = pd.read_csv(required["alignment_qa_review"])

    image_manifest = manifest[manifest["file_role"].eq("image_candidate")]
    add(checks, "one master row per canonical image candidate", len(master) == len(image_manifest), f"master={len(master)} manifest={len(image_manifest)}")
    add(checks, "image_id unique", master["image_id"].is_unique, master["image_id"].nunique())
    add(checks, "original_path unique", master["original_path"].is_unique, master["original_path"].nunique())

    expected_ids = master.apply(
        lambda row: master_table.stable_image_id(row["dataset"], row["dataset_version"], row["original_path"]),
        axis=1,
    )
    add(checks, "stable image_id exact GOV-005 formula", bool(expected_ids.equals(master["image_id"])), "dataset prefix plus 16 hex digest")

    forbidden_hits = []
    for dataset in ["lfw", "utkface"]:
        dataset_cfg = preprocessing["datasets"][dataset]
        tokens = [token.lower() for token in dataset_cfg.get("forbidden_path_tokens", [])]
        for path in master.loc[master["dataset"].eq(dataset), "original_path"]:
            lowered = path.lower()
            if any(token in lowered for token in tokens):
                forbidden_hits.append(path)
    add(checks, "no aligned or cropped canonical LFW or UTKFace paths", len(forbidden_hits) == 0, forbidden_hits[:5])

    archive_checks_path = paths.DATA_PREP_OUTPUTS_DIR / "tables" / "archive_checks.csv"
    if archive_checks_path.exists():
        archive_checks = pd.read_csv(archive_checks_path)
        add(checks, "archives version pinned and checksums match", bool(len(archive_checks) and archive_checks["checksum_matches"].all()), archive_checks.to_dict("records"))
    else:
        add(checks, "archive checks exist", False, archive_checks_path)

    unresolved_near = review[review["review_status"].eq("candidate")]
    add(checks, "near duplicate candidate review complete", len(unresolved_near) == 0, len(unresolved_near))

    unresolved_cross = review[review["cross_dataset"] & review["review_status"].eq("candidate")]
    add(checks, "cross dataset perceptual review complete", len(unresolved_cross) == 0, len(unresolved_cross))

    qa_required = str(preprocessing["alignment_qa"]["required_status"])
    add(checks, "alignment QA sample reviewed", bool(len(qa) and qa["qa_status"].eq(qa_required).all()), qa["qa_status"].value_counts(dropna=False).to_dict())

    raw_datasets = {
        dataset
        for dataset, item in preprocessing["datasets"].items()
        if item["project_detection_required"]
    }
    raw = master[master["dataset"].isin(raw_datasets) & ~master["corrupt"]]
    add(checks, "raw images have detection outcomes", bool(raw["n_detections"].notna().all()), int(raw["n_detections"].notna().sum()))
    add(checks, "alignment transform stored for successful project alignments", bool(raw.loc[raw["alignment_valid"].fillna(False), "alignment_transform"].notna().all()), int(raw["alignment_valid"].fillna(False).sum()))

    add(checks, "LFW pair row count", len(pairs) == 6000, len(pairs))
    add(checks, "LFW pair image IDs resolve", bool(pairs["image_id_1"].isin(master["image_id"]).all() and pairs["image_id_2"].isin(master["image_id"]).all()), "both members resolve")

    source_review_path = paths.METADATA_DIR / "source_review.yaml"
    source_review = yaml.safe_load(source_review_path.read_text(encoding="utf-8")) if source_review_path.exists() else {}
    source_review_ok = all(
        source_review.get(dataset, {}).get(field) is True
        for dataset in ["lfw", "fairface", "utkface"]
        for field in ["official_source_reviewed", "errata_reviewed", "identity_metadata_reviewed"]
    )
    add(checks, "published errata and identity metadata review complete", source_review_ok, source_review_path)

    models = config.load_models()
    unfilled_libraries = {
        key: value
        for key, value in models.get("libraries", {}).items()
        if value is None or "FILL" in str(value)
    }
    add(checks, "models.yaml environment versions filled", len(unfilled_libraries) == 0, unfilled_libraries)

    detector_threshold = float(preprocessing["detector"]["confidence_threshold"])
    checkpoint_cfg = models["checkpoints"]["retinaface"]
    checkpoint_threshold = float(checkpoint_cfg["detection_threshold"])
    add(checks, "RetinaFace threshold consistent across configs", detector_threshold == checkpoint_threshold, f"preprocessing={detector_threshold} models={checkpoint_threshold}")

    checkpoint_path = Path(str(checkpoint_cfg["file_location"])).expanduser()
    checkpoint_exists = checkpoint_path.exists()
    add(checks, "RetinaFace checkpoint exists", checkpoint_exists, checkpoint_path)
    if checkpoint_exists:
        actual_checkpoint_hash = file_sha256(checkpoint_path)
        add(checks, "RetinaFace checkpoint SHA-256 matches models.yaml", actual_checkpoint_hash == checkpoint_cfg["sha256"], actual_checkpoint_hash)
        add(checks, "RetinaFace checkpoint size matches models.yaml", checkpoint_path.stat().st_size == int(checkpoint_cfg["file_size_bytes"]), checkpoint_path.stat().st_size)

    counts = master.groupby("dataset").size().to_dict()
    add(checks, "all three canonical datasets present", set(counts) == {"lfw", "fairface", "utkface"}, counts)

    reference_path = paths.METADATA_DIR / "phase0_reference_hashes.json"
    hash_targets = {
        "file_manifest.parquet": required["file_manifest"],
        "master_annotations.parquet": required["master_annotations"],
        "lfw_pairs.parquet": required["lfw_pairs"],
        "utkface_split.csv": required["utkface_split"],
    }
    current_hashes = {name: table_content_hash(path) for name, path in hash_targets.items()}

    if args.write_reference:
        reference_path.write_text(json.dumps(current_hashes, indent=2), encoding="utf-8")
        print("Wrote reference hashes:", reference_path)

    if args.compare_reference:
        if not reference_path.exists():
            add(checks, "reference hashes exist", False, reference_path)
        else:
            expected = json.loads(reference_path.read_text(encoding="utf-8"))
            add(checks, "rebuild matches reference hashes", current_hashes == expected, {"expected": expected, "current": current_hashes})

    failed = [item for item in checks if not item["passed"]]
    status = "PHASE0_REBUILD_READY" if not failed else "NOT_READY"
    report = {
        "status": status,
        "dataset_counts": counts,
        "checks": checks,
    }
    report_path = paths.DATA_PREP_OUTPUTS_DIR / "reports" / "phase0_closure_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

    if failed:
        raise RuntimeError(f"{len(failed)} Phase 0 closure checks failed.")


if __name__ == "__main__":
    main()
