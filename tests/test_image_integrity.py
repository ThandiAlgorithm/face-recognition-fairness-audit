import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest  # noqa: E402
from PIL import Image, ImageCms  # noqa: E402

from src import paths  # noqa: E402
from src.image_integrity import inspect_image  # noqa: E402

# Orientation 6 swaps image width and height.
ORIENTATION_ROTATE_90_CW = 6

# Real Phase 0 greyscale settings from configs/preprocessing.yaml.
PREPROCESSING_CFG = {
    "greyscale": {
        "chroma_tolerance": 2.0,
        "sample_max_pixels": 10000,
        "sample_seed": 26041,
    },
}


@pytest.fixture(autouse=True)
def _isolated_data_dir(tmp_path, monkeypatch):
    # inspect_image() resolves paths relative to paths.DATA_DIR;
    # point it at this test's own tmp_path so fixture images don't
    # need to live under the real data/ directory.
    monkeypatch.setattr(paths, "DATA_DIR", tmp_path)


def _make_test_image_with_exif_and_icc(path: Path, width: int = 40, height: int = 20) -> None:
    img = Image.new("RGB", (width, height), color=(200, 50, 50))

    exif = img.getexif()
    exif[274] = ORIENTATION_ROTATE_90_CW  # 274 = Orientation tag

    srgb_profile = ImageCms.createProfile("sRGB")
    icc_bytes = ImageCms.ImageCmsProfile(srgb_profile).tobytes()

    img.save(path, exif=exif, icc_profile=icc_bytes)


def test_exif_orientation_is_detected(tmp_path):
    img_path = tmp_path / "exif_icc_test.jpg"
    _make_test_image_with_exif_and_icc(img_path)

    result = inspect_image(img_path, PREPROCESSING_CFG)

    assert result["decode_success"] is True
    assert result["exif_orientation"] == ORIENTATION_ROTATE_90_CW


def test_icc_profile_is_detected(tmp_path):
    img_path = tmp_path / "exif_icc_test.jpg"
    _make_test_image_with_exif_and_icc(img_path)

    result = inspect_image(img_path, PREPROCESSING_CFG)

    assert result["colour_profile"] is not None


def test_dimensions_reflect_exif_rotation(tmp_path):
    # EXIF transpose changes 40 by 20 to 20 by 40.
    img_path = tmp_path / "exif_icc_test.jpg"
    _make_test_image_with_exif_and_icc(img_path, width=40, height=20)

    result = inspect_image(img_path, PREPROCESSING_CFG)

    assert result["header_width_px"] == 40
    assert result["header_height_px"] == 20
    assert result["width_px"] == 20
    assert result["height_px"] == 40


def test_plain_image_has_no_exif_or_icc(tmp_path):
    img_path = tmp_path / "plain.png"
    Image.new("RGB", (10, 10), color=(0, 0, 0)).save(img_path)

    result = inspect_image(img_path, PREPROCESSING_CFG)

    assert result["decode_success"] is True
    assert result["exif_orientation"] is None
    assert result["colour_profile"] is None


def test_chroma_stats_are_populated_on_success(tmp_path):
    img_path = tmp_path / "plain.png"
    Image.new("RGB", (10, 10), color=(100, 100, 100)).save(img_path)

    result = inspect_image(img_path, PREPROCESSING_CFG)

    assert result["width_px"] == 10
    assert result["height_px"] == 10
    # A neutral grey has no chroma: a* and b* should be near zero.
    assert result["mean_a_star"] == pytest.approx(0.0, abs=0.5)
    assert result["mean_b_star"] == pytest.approx(0.0, abs=0.5)


def test_truncated_png_is_recorded_as_failure_not_padded(tmp_path):
    real_path = tmp_path / "real.png"
    Image.new("RGB", (50, 50), color=(1, 2, 3)).save(real_path)
    data = real_path.read_bytes()

    truncated_path = tmp_path / "truncated.png"
    truncated_path.write_bytes(data[: len(data) // 2])

    result = inspect_image(truncated_path, PREPROCESSING_CFG)

    assert result["decode_success"] is False
    assert result["decode_error"] is not None
    assert result["width_px"] is None
    assert result["height_px"] is None


def test_missing_file_is_recorded_as_failure(tmp_path):
    result = inspect_image(tmp_path / "does_not_exist.jpg", PREPROCESSING_CFG)

    assert result["decode_success"] is False
    assert result["decode_error"] is not None


def test_achromatic_greyscale_content_is_flagged(tmp_path):
    # Grey RGB pixels have no chroma.
    img_path = tmp_path / "grey_rgb.png"
    Image.new("RGB", (20, 20), color=(128, 128, 128)).save(img_path)

    result = inspect_image(img_path, PREPROCESSING_CFG)

    assert result["original_mode"] == "RGB"
    assert result["is_greyscale"] is True


def test_saturated_colour_image_is_not_greyscale(tmp_path):
    img_path = tmp_path / "red_rgb.png"
    Image.new("RGB", (20, 20), color=(220, 20, 20)).save(img_path)

    result = inspect_image(img_path, PREPROCESSING_CFG)

    assert result["is_greyscale"] is False
