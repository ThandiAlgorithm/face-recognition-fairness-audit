from __future__ import annotations

import hashlib
import json
import platform
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import cv2
import numpy as np
import pandas as pd

from src import phase1_paths as paths


def project_path(value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else paths.PROJECT_ROOT / p


def load_audit_config(path: str | Path = "configs/phase1_audit.json") -> tuple[dict, Path]:
    p = project_path(path)
    return json.loads(p.read_text(encoding="utf-8")), p


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_table(path: str | Path) -> pd.DataFrame:
    p = project_path(path)
    if not p.exists():
        raise FileNotFoundError(f"Table not found: {p}")
    if p.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(p)
    return pd.read_csv(p)


def _parquet_safe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object:
            sample = out[c].dropna().head(50)
            if len(sample) and not all(isinstance(v, str) for v in sample):
                if all(isinstance(v, (bool, np.bool_)) for v in sample):
                    continue
                out[c] = out[c].map(lambda v: None if v is None or (isinstance(v, float) and np.isnan(v)) else json.dumps(v, default=_json_default) if not isinstance(v, str) else v)
    return out


def _json_default(v: Any):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return str(v)


def write_table(df: pd.DataFrame, path: Path, parquet: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    if parquet:
        try:
            _parquet_safe(df).to_parquet(path.with_suffix(".parquet"), index=False)
        except Exception as exc:
            path.with_suffix(".parquet_warning.txt").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
    return path


def write_json(obj: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default), encoding="utf-8")
    return path


def output_dirs(cfg: dict, module: str) -> dict[str, Path]:
    root = project_path(cfg.get("outputs_root", "outputs/phase1")) / module
    out = {k: root / k for k in ["tables", "figures", "metadata", "qa"]}
    for p in out.values():
        p.mkdir(parents=True, exist_ok=True)
    out["root"] = root
    return out


def module_table(cfg: dict, module: str, name: str) -> Path:
    return project_path(cfg.get("outputs_root", "outputs/phase1")) / module / "tables" / name


def load_master(cfg: dict) -> tuple[pd.DataFrame, Path]:
    p = project_path(cfg["master_table"])
    master = read_table(p)
    for col in ["image_id", "dataset"]:
        if col not in master.columns:
            raise ValueError(f"Master table missing required column: {col}")
    if master["image_id"].duplicated().any():
        raise ValueError("image_id is not unique in the master table.")
    return master, p


def is_missing(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and np.isnan(v):
        return True
    if isinstance(v, str) and v.strip().lower() in {"", "nan", "none", "null"}:
        return True
    return False


def as_array(value: Any, shape: Optional[tuple[int, ...]] = None) -> Optional[np.ndarray]:
    if not isinstance(value, (list, tuple, np.ndarray)) and is_missing(value):
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:
            return None
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        try:
            arr = np.asarray([np.asarray(v, dtype=float) for v in value], dtype=float)
        except Exception:
            return None
    if arr.size == 0 or not np.isfinite(arr).all():
        return None
    if shape is not None:
        if arr.size != int(np.prod(shape)):
            return None
        arr = arr.reshape(shape)
    return arr


def resolve_file(value: Any) -> Optional[Path]:
    if is_missing(value):
        return None
    p = Path(str(value))
    candidates = [p] if p.is_absolute() else [paths.PROJECT_ROOT / p, paths.DATA_DIR / p, paths.ALIGNED_DIR / p]
    for c in candidates:
        if c.exists():
            return c
    return None


def as_bool(v: Any) -> bool:
    if is_missing(v):
        return False
    if isinstance(v, str):
        return v.strip().lower() in {"true", "1", "yes"}
    return bool(v)


def valid_pixel_mask(record: dict, aligned_shape: tuple[int, int]) -> tuple[np.ndarray, str]:
    h, w = aligned_shape
    transform = as_array(record.get("alignment_transform"), (2, 3))
    src_w = record.get("width_px")
    src_h = record.get("height_px")
    if transform is None or is_missing(src_w) or is_missing(src_h):
        return np.ones((h, w), dtype=bool), "full_image_no_transform"
    ones = np.ones((int(src_h), int(src_w)), dtype=np.uint8)
    mask = cv2.warpAffine(ones, transform.astype(np.float64), (w, h), flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    return mask, "warped_source_extent"


def erode(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask
    k = np.ones((2 * px + 1, 2 * px + 1), np.uint8)
    return cv2.erode(mask.astype(np.uint8), k).astype(bool)


def image_number(original_path: str) -> Optional[int]:
    import re
    m = re.match(r"^.+_(\d+)$", Path(str(original_path)).stem)
    return int(m.group(1)) if m else None


def analysis_flags(df: pd.DataFrame, eligible: pd.Series, dedup_cfg: dict) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["eligible"] = eligible.fillna(False).astype(bool)
    reason = pd.Series(np.where(out["eligible"], None, "not_eligible"), index=df.index, dtype=object)
    order = df.sort_values("image_id").index
    for col, key in [("exact_dup_group", "keep_one_per_exact_group"), ("near_dup_group", "keep_one_per_confirmed_near_group")]:
        if col in df.columns and dedup_cfg.get(key, True):
            g = df.loc[order, col]
            has = g.notna() & g.astype(str).str.len().gt(0) & out.loc[order, "eligible"] & reason.loc[order].isna()
            dup = g[has].duplicated(keep="first")
            reason.loc[dup[dup].index] = f"duplicate_copy:{col}"
    out["in_analysis"] = out["eligible"] & reason.isna()
    out["analysis_exclusion"] = reason
    return out


def _run_one(func: Callable, cfg: dict, record: dict) -> dict:
    try:
        return func(record, cfg)
    except Exception as exc:
        return {"image_id": record.get("image_id"), "status": "exception", "error": f"{type(exc).__name__}: {exc}"}


def process_records(
    func: Callable[[dict, dict], dict],
    records: list[dict],
    cfg: dict,
    checkpoint: Path,
    workers: int = 1,
    chunk: int = 2000,
) -> pd.DataFrame:
    done = pd.DataFrame()
    if checkpoint.exists():
        done = pd.read_csv(checkpoint, low_memory=False)
        wanted = {r["image_id"] for r in records}
        done = done[done["image_id"].isin(wanted)]
    finished = set(done["image_id"]) if not done.empty else set()
    todo = [r for r in records if r["image_id"] not in finished]
    parts = [done] if not done.empty else []
    runner = partial(_run_one, func, cfg)
    pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        for start in range(0, len(todo), chunk):
            batch = todo[start:start + chunk]
            rows = list(pool.map(runner, batch, chunksize=64)) if pool else [runner(r) for r in batch]
            parts.append(pd.DataFrame(rows))
            pd.concat(parts, ignore_index=True).to_csv(checkpoint, index=False)
            print(f"{checkpoint.stem}: {len(finished) + start + len(batch)} / {len(records)}", flush=True)
    finally:
        if pool:
            pool.shutdown()
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def save_run_metadata(module: str, cfg: dict, cfg_path: Path, master_path: Path, od: dict, extra: dict) -> None:
    import scipy
    import skimage
    meta = {
        "module": module,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "opencv": cv2.__version__,
        "scipy": scipy.__version__,
        "scikit_image": skimage.__version__,
        "config_path": str(cfg_path),
        "config_sha256": sha256_file(cfg_path),
        "master_table": str(master_path),
        "master_sha256": sha256_file(master_path) if master_path.exists() else None,
        "manual_review": False,
        **extra,
    }
    write_json(meta, od["metadata"] / "run_metadata.json")
    write_json(cfg, od["metadata"] / "config_snapshot.json")


def register_results(cfg: dict, cfg_path: Path, module: str, files: Iterable[Path], script: str) -> None:
    """Register every generated file for a module, not only named tables.

    This scans the module output directory after the run has created tables, Parquet
    copies, PNG figures, QA images, JSON metadata, markdown drafts, and warnings.
    The explicit ``files`` argument is retained for backwards compatibility and is
    unioned with the scan.
    """
    outputs_root = project_path(cfg.get("outputs_root", "outputs/phase1"))
    module_root = outputs_root / module
    reg = outputs_root / "result_registry.csv"
    now = datetime.now(timezone.utc).isoformat()
    cfg_sha = sha256_file(cfg_path)

    candidates = {Path(f) for f in files if Path(f).exists()}
    if module_root.exists():
        candidates.update(p for p in module_root.rglob("*") if p.is_file())
    ordered = sorted(candidates, key=lambda p: str(p))
    rows = [{
        "registered_utc": now,
        "module": module,
        "script": script,
        "file": f.as_posix(),
        "file_type": f.suffix.lower().lstrip(".") or "none",
        "file_sha256": sha256_file(f),
        "config_sha256": cfg_sha,
    } for f in ordered]
    new = pd.DataFrame(rows)

    if reg.exists():
        old = pd.read_csv(reg, low_memory=False)
        # Support older registries that predate file_type.
        if "file_type" not in old.columns:
            old["file_type"] = old["file"].map(lambda x: Path(str(x)).suffix.lower().lstrip(".") or "none")
        old = old[~old["module"].eq(module)]
        new = pd.concat([old, new], ignore_index=True)
    reg.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(reg, index=False)


def contact_sheet(items: list[tuple[Path, str, Optional[Callable[[np.ndarray], np.ndarray]]]], path: Path, tile: int = 160, cols: int = 6) -> None:
    if not items:
        return
    rows = (len(items) + cols - 1) // cols
    sheet = np.full((rows * (tile + 22), cols * tile, 3), 255, np.uint8)
    for i, (img_path, label, draw) in enumerate(items):
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        if draw is not None:
            img = draw(img)
        img = cv2.resize(img, (tile, tile))
        r, c = divmod(i, cols)
        y, x = r * (tile + 22), c * tile
        sheet[y:y + tile, x:x + tile] = img
        cv2.putText(sheet, label[:24], (x + 3, y + tile + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), sheet)
