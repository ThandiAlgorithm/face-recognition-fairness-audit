
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image, ImageCms  # noqa: E402

from src.image_integrity import decode_and_measure_image  # noqa: E402

# Orientation 6 swaps image width and height.
ORIENTATION_ROTATE_90_CW = 6


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

    result = decode_and_measure_image(img_path)

    assert result["decode_success"] is True
    assert result["exif_orientation"] == ORIENTATION_ROTATE_90_CW


def test_icc_profile_is_detected(tmp_path):
    img_path = tmp_path / "exif_icc_test.jpg"
    _make_test_image_with_exif_and_icc(img_path)

    result = decode_and_measure_image(img_path)

    assert result["icc_profile_present"] is True


def test_array_shape_reflects_exif_rotation(tmp_path):
    # EXIF transpose changes 40 by 20 to 20 by 40.
    img_path = tmp_path / "exif_icc_test.jpg"
    _make_test_image_with_exif_and_icc(img_path, width=40, height=20)

    result = decode_and_measure_image(img_path)

    assert result["header_width"] == 40
    assert result["header_height"] == 20
    assert result["decoded_width"] == 20
    assert result["decoded_height"] == 40
    assert result["header_decoded_size_match"] is True  # swap is the expected case


def test_plain_image_has_no_exif_or_icc(tmp_path):
    img_path = tmp_path / "plain.png"
    Image.new("RGB", (10, 10), color=(0, 0, 0)).save(img_path)

    result = decode_and_measure_image(img_path)

    assert result["decode_success"] is True
    assert result["exif_orientation"] is None
    assert result["icc_profile_present"] is False


def test_pixel_stats_are_populated_on_success(tmp_path):
    img_path = tmp_path / "plain.png"
    Image.new("RGB", (10, 10), color=(100, 100, 100)).save(img_path)

    result = decode_and_measure_image(img_path)

    assert result["array_shape"] == [10, 10, 3]
    assert result["pixel_mean"] == 100.0
    assert result["pixel_std"] == 0.0


def test_truncated_png_is_recorded_as_failure_not_padded(tmp_path):
    real_path = tmp_path / "real.png"
    Image.new("RGB", (50, 50), color=(1, 2, 3)).save(real_path)
    data = real_path.read_bytes()

    truncated_path = tmp_path / "truncated.png"
    truncated_path.write_bytes(data[: len(data) // 2])

    result = decode_and_measure_image(truncated_path)

    assert result["decode_success"] is False
    assert result["decode_error"] is not None
    assert result["array_shape"] is None


def test_missing_file_is_recorded_as_failure(tmp_path):
    result = decode_and_measure_image(tmp_path / "does_not_exist.jpg")

    assert result["decode_success"] is False
    assert result["decode_error"] is not None


def test_achromatic_greyscale_content_detected_even_in_rgb_mode(tmp_path):
    # Grey RGB pixels have no chroma.
    img_path = tmp_path / "grey_rgb.png"
    Image.new("RGB", (20, 20), color=(128, 128, 128)).save(img_path)

    result = decode_and_measure_image(img_path)

    assert result["original_mode"] == "RGB"
    assert result["achromatic_rgb"] is True


def test_saturated_colour_image_is_not_achromatic(tmp_path):
    img_path = tmp_path / "red_rgb.png"
    Image.new("RGB", (20, 20), color=(220, 20, 20)).save(img_path)

    result = decode_and_measure_image(img_path)

    assert result["achromatic_rgb"] is False
