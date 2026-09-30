from pathlib import Path

# Standalone defaults. Tests or host projects may monkeypatch/override these.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CONFIGS_DIR = PROJECT_ROOT / "configs"
ALIGNED_DIR = DATA_DIR / "derived" / "aligned"
METADATA_DIR = PROJECT_ROOT / "metadata"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
