
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
CONFIGS_DIR = PROJECT_ROOT / "configs"
METADATA_DIR = PROJECT_ROOT / "metadata"

PROCESSED_DATA_DIR = DATA_DIR / "processed"
INTERIM_DATA_DIR = PROCESSED_DATA_DIR / "interim"

OUTPUTS_DATA_PREP_DIR = PROJECT_ROOT / "outputs" / "data_prep"

OUTPUT_SUBDIRS = ["reports", "tables", "figures", "logs", "qa", "config"]


def ensure_output_dirs() -> None:
    for name in OUTPUT_SUBDIRS:
        (OUTPUTS_DATA_PREP_DIR / name).mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)


def find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "src").is_dir() and (candidate / "data").is_dir():
            return candidate
    raise RuntimeError(f"Could not locate project root above {start}")


def to_rel(p) -> str:
    return Path(p).resolve().relative_to(DATA_DIR.resolve()).as_posix()


def to_abs(rel: str) -> Path:
    return DATA_DIR / rel
