from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
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




def normalise_for_hash(value):
    """Convert nested table values into deterministic JSON-compatible data."""

    if isinstance(value, np.ndarray):
        return normalise_for_hash(value.tolist())

    if isinstance(value, (list, tuple)):
        return [normalise_for_hash(item) for item in value]

    if isinstance(value, dict):
        return {
            str(key): normalise_for_hash(item)
            for key, item in sorted(
                value.items(),
                key=lambda pair: str(pair[0]),
            )
        }

    if isinstance(value, np.generic):
        return normalise_for_hash(value.item())

    if value is None or value is pd.NA or value is pd.NaT:
        return None

    if pd.api.types.is_scalar(value):
        missing = pd.isna(value)

        if isinstance(missing, (bool, np.bool_)) and missing:
            return None

    if isinstance(value, (str, int, float, bool)):
        return value

    return str(value)


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
                lambda value: json.dumps(
                    normalise_for_hash(value),
                    sort_keys=True,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                )
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
        "lfw_pairs": paths.METADATA_DIR / "lfw_view2_pairs.parquet",
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

    # Check current archive provenance rather than relying on
    # a potentially stale archive_checks.csv.

    archive_results = []

    # This project's preprocessing.yaml stores archive information
    # as a top-level list of records keyed by "dataset", not nested
    # under each dataset's own config block.
    archives_by_dataset = {
        entry["dataset"]: entry
        for entry in preprocessing.get("archives", [])
    }

    for dataset in ["lfw", "fairface", "utkface"]:
        archive_cfg = archives_by_dataset.get(dataset)

        if not isinstance(archive_cfg, dict):
            raise RuntimeError(
                f"{dataset}: cannot locate the archive "
                "configuration. Inspect preprocessing.yaml."
            )

        archive_path = archive_cfg.get("path")
        provenance_status = archive_cfg.get(
            "provenance_status"
        )

        expected_sha = archive_cfg.get(
            "sha256",
            archive_cfg.get("expected_sha256"),
        )

        if provenance_status == "archive_not_available":
            # This is a documented provenance limitation,
            # not a claim that a checksum was verified.
            accepted = archive_path is None

            archive_results.append({
                "dataset": dataset,
                "provenance_status": provenance_status,
                "archive_path": None,
                "archive_exists": False,
                "checksum_verified": False,
                "accepted_with_limitation": accepted,
            })

        else:
            path = (
                Path(str(archive_path)).expanduser()
                if archive_path
                else None
            )

            exists = path is not None and path.is_file()

            actual_sha = (
                file_sha256(path)
                if exists
                else None
            )

            checksum_verified = bool(
                expected_sha
                and actual_sha
                and actual_sha.lower()
                    == str(expected_sha).lower()
            )

            archive_results.append({
                "dataset": dataset,
                "provenance_status": provenance_status,
                "archive_path": str(path),
                "archive_exists": exists,
                "checksum_verified": checksum_verified,
                "accepted_with_limitation": False,
            })

    archive_df = pd.DataFrame(archive_results)

    archive_ok = bool(
        (
            archive_df["checksum_verified"]
            | archive_df["accepted_with_limitation"]
        ).all()
    )

    add(
        checks,
        "archive provenance verified or limitation documented",
        archive_ok,
        archive_results,
    )

    # Refresh the previously stale report using the
    # configuration and evidence checked above.
    archive_checks_path = (
        paths.DATA_PREP_OUTPUTS_DIR
        / "tables"
        / "archive_checks.csv"
    )

    archive_checks_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    archive_df.to_csv(
        archive_checks_path,
        index=False,
    )

    # Exhaustive-review is no longer the frozen acceptance criterion for
    # within-dataset candidates -- AMEND-20260922-01 splits them into a
    # strong-evidence stratum (reviewed exhaustively) and a weak-evidence
    # stratum (reviewed via a deterministic seed-26041 sample of 1,000).
    # Formal RZ/TM/Supervisor approval of that amendment is still
    # [PENDING] and is NOT asserted here; this only checks that the
    # technical review work matches what the amendment documents.
    strong_stratum_path = paths.METADATA_DIR / "near_duplicate_review_strong.csv"
    weak_sample_path = paths.METADATA_DIR / "near_duplicate_review_weak_sample.csv"

    if strong_stratum_path.exists() and weak_sample_path.exists():
        strong_stratum = pd.read_csv(strong_stratum_path)
        weak_sample = pd.read_csv(weak_sample_path)

        strong_unresolved = strong_stratum[strong_stratum["review_status"].eq("candidate")]
        add(
            checks,
            "strong-evidence duplicate stratum fully reviewed",
            len(strong_stratum) == 4513 and len(strong_unresolved) == 0,
            f"rows={len(strong_stratum)}, unresolved={len(strong_unresolved)}",
        )

        weak_seed_ok = bool((weak_sample["sampling_seed"] == 26041).all())
        weak_population_ok = bool((weak_sample["weak_population_size"] == 30328).all())
        add(
            checks,
            "weak-evidence sample matches documented seed and population",
            len(weak_sample) == 1000 and weak_seed_ok and weak_population_ok,
            f"sample_rows={len(weak_sample)}, seed_matches={weak_seed_ok}, population_matches={weak_population_ok}",
        )

        within = review.reset_index().rename(columns={"index": "row_id"})
        within = within[~within["cross_dataset"].fillna(False)]
        weak_population_live = within[
            within["candidate_method"].eq("crop")
            & within["crop_matching_regions"].eq(2)
        ]

        sample_keys = set(zip(weak_sample["original_path_a"], weak_sample["original_path_b"]))
        in_documented_sample = [
            (a, b) in sample_keys
            for a, b in zip(
                weak_population_live["original_path_a"],
                weak_population_live["original_path_b"],
            )
        ]
        weak_population_live = weak_population_live.assign(in_documented_sample=in_documented_sample)

        # Check 3 of 4: the 29,328 pairs the amendment says were left
        # unreviewed must still be "candidate" today, not silently
        # resolved by anything downstream.
        weak_unresolved_live = weak_population_live[
            weak_population_live["review_status"].eq("candidate")
        ]
        add(
            checks,
            "unsampled weak-evidence pairs retain candidate status",
            len(weak_unresolved_live) == 29328,
            f"unresolved={len(weak_unresolved_live)}, expected=29328",
        )

        # Check 4 of 4: any resolved weak-evidence pair outside the
        # documented sample must trace to real, timestamped human
        # review evidence in the journal, predating stratification --
        # not an unexplained or unaudited status change. See the
        # addendum in AMEND-20260922-01.md.
        bulk_resolved = weak_population_live[
            weak_population_live["review_status"].ne("candidate")
            & ~weak_population_live["in_documented_sample"]
        ]

        journal_path = paths.METADATA_DIR / "near_duplicate_review.csv.journal.csv"
        strong_journal_path = paths.METADATA_DIR / "near_duplicate_review_strong.csv.journal.csv"

        bulk_all_false_alarm = bool(len(bulk_resolved)) and bool(
            (bulk_resolved["review_status"] == "false_alarm").all()
        )
        bulk_journal_traceable = False
        bulk_predates_stratification = False

        if journal_path.exists() and strong_journal_path.exists() and len(bulk_resolved):
            journal = pd.read_csv(journal_path)
            bulk_journal_traceable = bool(
                set(bulk_resolved["row_id"]).issubset(set(journal["row_id"]))
            )

            strata_start = pd.to_datetime(
                pd.read_csv(strong_journal_path)["timestamp"]
            ).min()
            bulk_journal_rows = journal[journal["row_id"].isin(bulk_resolved["row_id"])]
            bulk_predates_stratification = bool(
                (pd.to_datetime(bulk_journal_rows["timestamp"]) < strata_start).all()
            )

        add(
            checks,
            "additional weak-stratum bulk decisions have a verified audit trail (protocol status pending GOV-010)",
            (
                len(bulk_resolved) == 2755
                and bulk_all_false_alarm
                and bulk_journal_traceable
                and bulk_predates_stratification
            ),
            (
                f"bulk_rows={len(bulk_resolved)}, all_false_alarm={bulk_all_false_alarm}, "
                f"journal_traceable={bulk_journal_traceable}, "
                f"predates_stratification={bulk_predates_stratification}"
            ),
        )
    else:
        add(
            checks,
            "duplicate-review stratification evidence files exist",
            False,
            {"strong_stratum_path": str(strong_stratum_path), "weak_sample_path": str(weak_sample_path)},
        )

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
    add(
        checks,
        "LFW pair image IDs resolve",
        bool(
            pairs["image_id_a"].isin(master["image_id"]).all()
            and pairs["image_id_b"].isin(master["image_id"]).all()
        ),
        "both members resolve",
    )

    source_review_path = paths.METADATA_DIR / "source_review.yaml"
    source_review = yaml.safe_load(source_review_path.read_text(encoding="utf-8")) if source_review_path.exists() else {}
    source_review_ok = all(
        source_review.get(dataset, {}).get(field) is True
        for dataset in ["lfw", "fairface", "utkface"]
        for field in ["official_source_reviewed", "errata_reviewed", "identity_metadata_reviewed"]
    )
    add(checks, "published errata and identity metadata review complete", source_review_ok, source_review_path)

    models = config.load_models()
    # Phase 0 requires the preprocessing, data and
    # image-processing environment to be reproducible.
    # Phase 1 statistical and fairness packages are
    # recorded separately and are not mandatory here.

    phase0_required_libraries = [
        "python",
        "torch",
        "torchvision",
        "numpy",
        "pandas",
        "scipy",
        "scikit_learn",
        "scikit_image",
        "pillow",
        "opencv",
        "imagehash",
    ]

    libraries = models.get("libraries", {})

    missing_phase0_versions = {
        name: libraries.get(name)
        for name in phase0_required_libraries
        if (
            libraries.get(name) is None
            or "FILL" in str(libraries.get(name))
        )
    }

    add(
        checks,
        "Phase 0 required library versions recorded",
        len(missing_phase0_versions) == 0,
        missing_phase0_versions,
    )

    # Retain an accurate record of packages not installed.
    phase1_libraries = {
        name: libraries.get(name)
        for name in [
            "statsmodels",
            "fairlearn",
            "aif360",
        ]
    }

    add(
        checks,
        "Phase 1 library availability documented",
        all(name in libraries for name in phase1_libraries),
        phase1_libraries,
    )

    # Verify the actual, current Phase 0 RetinaFace configuration.
    detector_cfg = preprocessing["detector"]
    checkpoint_cfg = models["checkpoints"]["retinaface"]

    # Check all settings shared by the two configuration files.
    shared_settings = [
        "implementation",
        "framework",
        "backbone",
        "confidence_threshold",
        "nms_threshold",
        "top_k",
        "keep_top_k",
    ]

    for setting in shared_settings:
        preprocessing_value = detector_cfg.get(setting)
        models_value = checkpoint_cfg.get(setting)

        add(
            checks,
            f"RetinaFace {setting} agrees across configs",
            preprocessing_value is not None
            and models_value is not None
            and preprocessing_value == models_value,
            f"preprocessing={preprocessing_value}, "
            f"models={models_value}",
        )

    # Verify the frozen detection threshold independently.
    add(
        checks,
        "RetinaFace confidence threshold is 0.90",
        detector_cfg.get("confidence_threshold") == 0.90,
        detector_cfg.get("confidence_threshold"),
    )

    # The actual checksum keys differ between these configs.
    preprocessing_sha = detector_cfg.get("checkpoint_sha256")
    models_sha = checkpoint_cfg.get("weights_sha256")

    add(
        checks,
        "RetinaFace checkpoint SHA-256 agrees across configs",
        bool(preprocessing_sha)
        and bool(models_sha)
        and preprocessing_sha == models_sha,
        f"preprocessing={preprocessing_sha}, models={models_sha}",
    )

    # Verify the real checkpoint on disk.
    location = checkpoint_cfg.get("file_location")

    checkpoint_path = (
        Path(location).expanduser()
        if location
        else None
    )

    checkpoint_exists = (
        checkpoint_path is not None
        and checkpoint_path.is_file()
    )

    add(
        checks,
        "RetinaFace checkpoint exists",
        checkpoint_exists,
        str(checkpoint_path),
    )

    if checkpoint_exists:
        actual_checkpoint_hash = file_sha256(checkpoint_path)

        add(
            checks,
            "RetinaFace checkpoint SHA-256 matches models.yaml",
            actual_checkpoint_hash == models_sha,
            actual_checkpoint_hash,
        )

        add(
            checks,
            "RetinaFace checkpoint SHA-256 matches preprocessing.yaml",
            actual_checkpoint_hash == preprocessing_sha,
            actual_checkpoint_hash,
        )

        actual_size = checkpoint_path.stat().st_size
        expected_size = checkpoint_cfg.get("file_size_bytes")

        add(
            checks,
            "RetinaFace checkpoint size matches models.yaml",
            expected_size is not None
            and actual_size == int(expected_size),
            f"actual={actual_size}, expected={expected_size}",
        )

    add(
        checks,
        "RetinaFace alignment size agrees across configs",
        preprocessing["alignment"].get("output_size")
        == checkpoint_cfg.get("alignment_output_size"),
        f"preprocessing={preprocessing['alignment'].get('output_size')}, "
        f"models={checkpoint_cfg.get('alignment_output_size')}",
    )

    counts = master.groupby("dataset").size().to_dict()
    add(checks, "all three canonical datasets present", set(counts) == {"lfw", "fairface", "utkface"}, counts)

    reference_path = paths.METADATA_DIR / "phase0_reference_hashes.json"
    hash_targets = {
        "file_manifest.parquet": required["file_manifest"],
        "master_annotations.parquet": required["master_annotations"],
        "lfw_view2_pairs.parquet": required["lfw_pairs"],
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
