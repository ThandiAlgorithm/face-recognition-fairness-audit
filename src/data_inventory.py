from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from src import paths


KNOWN_METADATA_SUFFIXES = {".csv", ".tsv", ".txt", ".json", ".yaml", ".yml"}
KNOWN_DOC_SUFFIXES = {".md", ".rst", ".pdf", ".html", ".htm"}
KNOWN_ARCHIVE_SUFFIXES = {".zip", ".gz", ".tgz", ".tar", ".bz2", ".7z"}
KNOWN_SYSTEM_FILENAMES = {
    ".ds_store",
    "thumbs.db",
    "desktop.ini",
}


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _classify_file(
    path: Path,
    image_roots: list[Path],
    label_files: set[Path],
) -> str:
    resolved = path.resolve()

    if resolved in label_files:
        return "metadata"

    if path.name.lower() in KNOWN_SYSTEM_FILENAMES:
        return "system_file"

    if any(
        _is_under(path, root)
        for root in image_roots
    ):
        if path.suffix.lower() in (
            KNOWN_METADATA_SUFFIXES
            | KNOWN_DOC_SUFFIXES
        ):
            return "metadata_or_documentation"

        return "image_candidate"

    suffix = path.suffix.lower()

    if suffix in KNOWN_ARCHIVE_SUFFIXES:
        return "archive"

    if suffix in KNOWN_METADATA_SUFFIXES:
        return "metadata"

    if suffix in KNOWN_DOC_SUFFIXES:
        return "documentation"

    return "other"


def validate_dataset_variant(dataset: str, dataset_cfg: dict) -> list[str]:
    warnings = []
    variant = str(dataset_cfg["variant"]).lower()
    roots = [str(item).lower() for item in dataset_cfg["image_roots"]]
    forbidden = [str(item).lower() for item in dataset_cfg.get("forbidden_path_tokens", [])]

    for root in roots:
        for token in forbidden:
            if token and token in root:
                raise RuntimeError(
                    f"{dataset} canonical root contains forbidden variant token '{token}': {root}"
                )

    if dataset in {"lfw", "utkface"} and "in_the_wild" not in variant and "original" not in variant:
        warnings.append(f"{dataset} variant does not explicitly say original or in_the_wild: {variant}")

    if dataset == "fairface" and dataset_cfg.get("source_preprocessed") is not True:
        raise RuntimeError("FairFace must be marked source_preprocessed=true for the official release.")

    return warnings


def build_file_manifest(preprocessing_cfg: dict) -> pd.DataFrame:
    rows = []

    for dataset, dataset_cfg in preprocessing_cfg["datasets"].items():
        validate_dataset_variant(dataset, dataset_cfg)

        dataset_root = paths.DATA_DIR / dataset_cfg["root"]

        image_roots = [
            paths.DATA_DIR / item
            for item in dataset_cfg["image_roots"]
        ]

        label_files = {
            (paths.DATA_DIR / item).resolve()
            for item in dataset_cfg.get("label_files", [])
        }

        if not dataset_root.exists():
            raise FileNotFoundError(
                f"Dataset root not found for {dataset}: {dataset_root}"
            )

        files_to_manifest: set[Path] = set()

        # Scan every file inside the canonical image roots.
        #
        # Do not filter by extension here. Files with unusual or malformed
        # extensions must still enter the manifest and integrity stage.
        for image_root in image_roots:
            if not image_root.exists():
                raise FileNotFoundError(
                    f"Canonical image root not found for "
                    f"{dataset}: {image_root}"
                )

            if not image_root.is_dir():
                raise RuntimeError(
                    f"Canonical image root is not a directory: "
                    f"{image_root}"
                )

            for file_path in image_root.rglob("*"):
                if file_path.is_file():
                    files_to_manifest.add(
                        file_path.resolve()
                    )

        # Explicitly configured metadata/label files may sit outside the
        # canonical image roots, so include them separately.
        for label_file in label_files:
            if not label_file.exists():
                raise FileNotFoundError(
                    f"Configured label file not found for "
                    f"{dataset}: {label_file}"
                )

            files_to_manifest.add(
                label_file
            )

        for file_path in sorted(
            files_to_manifest,
            key=lambda item: item.as_posix().lower(),
        ):
            role = _classify_file(
                file_path,
                image_roots,
                label_files,
            )

            rows.append(
                {
                    "dataset": dataset,
                    "dataset_version": dataset_cfg["dataset_version"],
                    "variant": dataset_cfg["variant"],
                    "original_path": paths.data_rel(file_path),
                    "file_size_bytes": int(
                        file_path.stat().st_size
                    ),
                    "sha256": sha256_file(file_path),
                    "archive_version": dataset_cfg.get(
                        "archive_version",
                        "unknown",
                    ),
                    "file_role": role,
                    "suffix": file_path.suffix.lower(),
                    "source_preprocessed": bool(
                        dataset_cfg["source_preprocessed"]
                    ),
                    "project_detection_required": bool(
                        dataset_cfg["project_detection_required"]
                    ),
                    "project_alignment_required": bool(
                        dataset_cfg["project_alignment_required"]
                    ),
                }
            )

    manifest = pd.DataFrame(rows)

    if manifest.empty:
        raise RuntimeError(
            "File manifest is empty."
        )

    if manifest["original_path"].duplicated().any():
        duplicates = manifest.loc[
            manifest["original_path"].duplicated(),
            "original_path",
        ].head().tolist()

        raise RuntimeError(
            f"Manifest contains duplicate paths: {duplicates}"
        )

    return manifest


def canonical_image_candidates(manifest: pd.DataFrame) -> pd.DataFrame:
    return manifest[manifest["file_role"].eq("image_candidate")].copy()


def check_archives(preprocessing_cfg: dict) -> pd.DataFrame:
    rows = []
    archives = preprocessing_cfg.get("archives", [])
    for item in archives:
        raw_path = item.get("path")
        expected = item.get("sha256")

        if raw_path is None:
            path = None
            exists = False
        else:
            path = paths.PROJECT_ROOT / raw_path
            exists = path.exists()

        actual = sha256_file(path) if exists else None
        pinned = bool(expected and expected not in {"FILL", "TODO", "unknown"})
        rows.append(
            {
                "dataset": item["dataset"],
                "variant": item["variant"],
                "archive_path": raw_path,
                "exists": exists,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "checksum_pinned": pinned,
                "checksum_matches": bool(exists and pinned and actual == expected),
                "provenance_status": item.get("provenance_status", "archive_pinned"),
            }
        )
    return pd.DataFrame(rows)


def dataset_count_report(manifest: pd.DataFrame) -> pd.DataFrame:
    images = canonical_image_candidates(manifest)
    return (
        images.groupby(["dataset", "dataset_version", "variant"], dropna=False)
        .size()
        .rename("source_files_in_canonical_image_roots")
        .reset_index()
    )
