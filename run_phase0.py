from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

STEPS = [
    "run_capture_environment.py",
    "run_build_file_manifest.py",
    "run_image_integrity.py",
    "run_parse_labels.py",
    "run_duplicate_detection.py",
    "run_build_master_table.py",
    "run_full_feature_extraction.py",
    "run_lfw_protocol_validation.py",
    "run_phase0_validation.py",
]


def main() -> None:
    for script in STEPS:
        path = PROJECT_ROOT / script
        print()
        print("Running", script)
        print()
        completed = subprocess.run([sys.executable, str(path)], cwd=PROJECT_ROOT)
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
