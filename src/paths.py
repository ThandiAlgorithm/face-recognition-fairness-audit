from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CONFIGS_DIR = PROJECT_ROOT / "configs"
METADATA_DIR = PROJECT_ROOT / "metadata"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DATA_PREP_OUTPUTS_DIR = OUTPUTS_DIR / "data_prep"
DERIVED_DIR = DATA_DIR / "derived"
ALIGNED_DIR = DERIVED_DIR / "aligned"
CACHE_DIR = DERIVED_DIR / "cache"
DOCS_ENV_DIR = PROJECT_ROOT / "docs" / "env"


def ensure_dirs() -> None:
    for path in [
        METADATA_DIR,
        DATA_PREP_OUTPUTS_DIR / "reports",
        DATA_PREP_OUTPUTS_DIR / "tables",
        DATA_PREP_OUTPUTS_DIR / "figures",
        DATA_PREP_OUTPUTS_DIR / "logs",
        DATA_PREP_OUTPUTS_DIR / "qa",
        DATA_PREP_OUTPUTS_DIR / "config",
        DERIVED_DIR,
        ALIGNED_DIR,
        CACHE_DIR,
        DOCS_ENV_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def data_rel(path: Path) -> str:
    return path.resolve().relative_to(DATA_DIR.resolve()).as_posix()


def data_abs(relative_path: str) -> Path:
    return DATA_DIR / Path(relative_path)


def project_rel(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def project_abs(relative_path: str) -> Path:
    return PROJECT_ROOT / Path(relative_path)
