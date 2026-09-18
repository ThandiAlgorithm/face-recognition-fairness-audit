
from __future__ import annotations

import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
METADATA_EXTENSIONS = {".csv", ".txt", ".json"}


def scan_directory(root: Path) -> dict:
    counts = {
        "all_files": 0,
        "image_files": 0,
        "metadata_files": 0,
        "other_files": 0,
        "directories": 0,
    }
    if not root.exists():
        return counts

    for path in root.rglob("*"):
        if path.is_dir():
            counts["directories"] += 1
            continue
        counts["all_files"] += 1
        suffix = path.suffix.lower()
        if suffix in IMAGE_EXTENSIONS:
            counts["image_files"] += 1
        elif suffix in METADATA_EXTENSIONS:
            counts["metadata_files"] += 1
        else:
            counts["other_files"] += 1
    return counts


def count_identities(root: Path) -> Optional[int]:
    if not root.exists():
        return None
    return sum(1 for p in root.iterdir() if p.is_dir())


def list_image_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]


README_STEMS = {"readme", "license", "licence", "version", "citation"}


def find_provenance_files(root: Path) -> list[str]:
    if not root.exists():
        return []
    return sorted(
        str(p.relative_to(root))
        for p in root.rglob("*")
        if p.is_file() and p.stem.lower() in README_STEMS
    )


def find_metadata_files(root: Path, max_depth: int = 2) -> list[str]:
    if not root.exists():
        return []
    out = []
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in METADATA_EXTENSIONS:
            continue
        if len(p.relative_to(root).parts) > max_depth:
            continue
        out.append(str(p.relative_to(root)))
    return sorted(out)


def find_image_directories(root: Path) -> list[str]:
    if not root.exists():
        return []
    out = []
    for d in [root, *(p for p in root.rglob("*") if p.is_dir())]:
        if any(p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS for p in d.iterdir()):
            rel = str(d.relative_to(root))
            out.append(rel if rel != "." else ".")
    return sorted(out)


def build_dataset_inventory(data_dir: Path, dataset_dirs: dict[str, str]) -> pd.DataFrame:
    scanned_date = datetime.date.today().isoformat()
    rows = []
    for name, dirname in dataset_dirs.items():
        root = data_dir / dirname
        prov = find_provenance_files(root)
        meta = find_metadata_files(root)
        imgdirs = find_image_directories(root)
        rows.append({
            "dataset": name,
            "detected_path": str(root),
            "exists": root.exists(),
            "date_scanned": scanned_date,
            "provenance_files_found": "; ".join(prov) if prov else "none found",
            "n_provenance_files": len(prov),
            "metadata_files_found": "; ".join(meta) if meta else "none found",
            "n_metadata_files": len(meta),
            "n_image_directories": len(imgdirs),
            "image_directories_sample": "; ".join(imgdirs[:5]),
            **scan_directory(root),
        })
    return pd.DataFrame(rows)


def build_file_inventory(data_dir: Path, locations: dict[tuple, str]) -> pd.DataFrame:
    rows = []
    for (dataset, location), rel in locations.items():
        root = data_dir / rel
        rows.append({
            "dataset": dataset,
            "location": location,
            "path": str(root),
            "exists": root.exists(),
            **scan_directory(root),
            "identities": count_identities(root),
        })
    return pd.DataFrame(rows)
