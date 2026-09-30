from pathlib import Path
import yaml

from src import paths


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data or {}


def load_preprocessing(path: str | Path | None = None) -> dict:
    p = Path(path) if path is not None else paths.CONFIGS_DIR / "preprocessing.yaml"
    if not p.is_absolute():
        p = paths.PROJECT_ROOT / p
    return _load_yaml(p)


def load_categories(path: str | Path | None = None) -> dict:
    p = Path(path) if path is not None else paths.CONFIGS_DIR / "categories.yaml"
    if not p.is_absolute():
        p = paths.PROJECT_ROOT / p
    return _load_yaml(p)
