from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import paths

PACKAGES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "statsmodels": "statsmodels",
    "scikit_learn": "scikit-learn",
    "scikit_image": "scikit-image",
    "pillow": "Pillow",
    "opencv": "opencv-python",
    "face_detection": "face-detection",
    "torch": "torch",
    "torchvision": "torchvision",
    "tensorflow": "tensorflow",
    "tf_keras": "tf-keras",
    "retina_face": "retina-face",
    "imagehash": "ImageHash",
    "fairlearn": "fairlearn",
    "aif360": "aif360",
}


def package_version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--update-models", action="store_true")
    args = parser.parse_args()
    paths.ensure_dirs()

    libraries = {"python": platform.python_version()}
    for key, distribution in PACKAGES.items():
        libraries[key] = package_version(distribution)

    report = {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "libraries": libraries,
    }

    try:
        import torch

        report["torch_cuda_available"] = bool(torch.cuda.is_available())
        report["torch_cuda_version"] = torch.version.cuda
        report["torch_device_count"] = int(torch.cuda.device_count())
    except Exception:
        report["torch_cuda_available"] = False
        report["torch_cuda_version"] = None
        report["torch_device_count"] = 0

    runtime_path = paths.METADATA_DIR / "environment_runtime.json"
    runtime_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    freeze_path = paths.DOCS_ENV_DIR / "requirements-lock.txt"
    freeze = subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze"],
        text=True,
    )
    freeze_path.write_text(freeze, encoding="utf-8")
    (paths.DOCS_ENV_DIR / "python_version.txt").write_text(
        platform.python_version() + "\n",
        encoding="utf-8",
    )

    if args.update_models:
        models_path = paths.CONFIGS_DIR / "models.yaml"
        with open(models_path, "r", encoding="utf-8") as handle:
            models = yaml.safe_load(handle)
        models.setdefault("libraries", {})
        models["libraries"].update(libraries)
        with open(models_path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(models, handle, sort_keys=False)
        print("Updated:", models_path)

    print(json.dumps(report, indent=2))
    print("Saved:", runtime_path)
    print("Saved:", freeze_path)


if __name__ == "__main__":
    main()
