import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import cv2
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import categories, config, detection, image_integrity, master_table, measurements, paths

CHECKPOINT_EVERY = 250


def save_checkpoint(rows: list[dict], path: Path) -> None:
    pd.DataFrame(rows).to_parquet(path, index=False)


def load_checkpoint(path: Path) -> tuple[list[dict], set[str]]:
    if not path.exists():
        return [], set()
    dataframe = pd.read_parquet(path)
    rows = dataframe.to_dict("records")
    done = set(dataframe["relative_path"])
    return rows, done


def align_output_path(relative_path: str) -> Path:
    parts = Path(relative_path).parts
    if len(parts) < 2:
        identity = "unknown"
        filename = Path(relative_path).stem
    else:
        identity = parts[-2]
        filename = Path(parts[-1]).stem

    output_dir = paths.PROCESSED_DATA_DIR / "aligned_lfw" / identity
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{filename}.png"


def empty_measurements() -> dict:
    return {
        "brightness": None, "contrast": None, "sharpness": None,
        "face_bbox_area_px": None, "face_area_ratio": None,
        "yaw": None, "pitch": None, "roll": None,
        "source_achromatic": None,
        "skin_roi_total_pixel_count": None, "skin_roi_valid_pixel_count": None,
        "skin_valid_fraction": None, "skin_roi_valid": False, "skin_roi_invalid_reason": None,
        "L_star": None, "a_star": None, "b_star": None,
        "ITA": None, "ita_undefined": None, "hue_angle": None,
    }


def main() -> None:
    paths.ensure_output_dirs()

    gov002 = config.load_gov002()
    gov006 = config.load_gov006()

    detector_cfg = config.detector_config(gov006)
    alignment_cfg = config.alignment_config(gov006)
    skin_cfg = config.skin_config(gov006)

    threshold = float(detector_cfg["confidence_threshold"])
    allow_upscaling = bool(detector_cfg["allow_upscaling"])
    output_size = int(alignment_cfg["output_size"])
    landmark_fractions = alignment_cfg["canonical_landmark_fractions"]

    roi_patches = skin_cfg["roi_patches"]
    skin_filter = skin_cfg["roi_skin_filter"]
    roi_min_valid_pixels = int(skin_cfg["roi_min_valid_pixels"])
    b_star_min_abs = float(skin_cfg["ita_guard"]["b_star_min_abs"])

    master = pd.read_parquet(paths.PROCESSED_DATA_DIR / "master_annotations_base.parquet")

    checkpoint_path = paths.INTERIM_DATA_DIR / "full_feature_extraction_checkpoint.parquet"
    rows, done = load_checkpoint(checkpoint_path)

    print("Already processed:", len(done))
    print("Total master rows:", len(master))

    lfw_pending = master[(master["dataset"] == "lfw") & (~master["relative_path"].isin(done))]

    model = None
    if len(lfw_pending):
        print("Building RetinaFace model")
        model = detection.build_retinaface_model()

    processed_since_checkpoint = 0

    for row_index, master_row in master.iterrows():
        relative_path = master_row["relative_path"]
        if relative_path in done:
            continue

        dataset = master_row["dataset"]
        absolute_path = paths.to_abs(relative_path)

        output = {
            "relative_path": relative_path,
            "dataset": dataset,
            "processing_success": False,
            "processing_error": None,
            "project_detection_applied": False,
            "project_alignment_applied": False,
            "detection_success": None,
            "number_of_faces": None,
            "detection_confidence": None,
            "bbox": None,
            "all_detections_json": None,
            "alignment_success": None,
            "alignment_transform": None,
            "aligned_path": None,
        }
        output.update(empty_measurements())

        try:
            source_achromatic = (
                bool(master_row["achromatic_rgb"]) if pd.notna(master_row["achromatic_rgb"]) else False
            )

            if dataset == "lfw":
                result = detection.detect_faces(
                    image_path=absolute_path, model=model, threshold=threshold, allow_upscaling=allow_upscaling
                )

                output["project_detection_applied"] = True
                output["detection_success"] = bool(result["inference_success"])
                output["number_of_faces"] = int(result["number_of_faces"])
                output["all_detections_json"] = json.dumps(result["detections"])

                primary = detection.select_primary_face(result["detections"])

                if not result["inference_success"]:
                    output["processing_success"] = False
                    output["processing_error"] = (
                        result["inference_error"] or "RetinaFace inference failed"
                    )
                elif primary is None:
                    output["processing_success"] = True
                    output["skin_roi_invalid_reason"] = "no face detected"
                else:
                    output["detection_confidence"] = float(primary["det_score"])
                    output["bbox"] = json.dumps(primary["bbox"])

                    image_bgr = cv2.imread(str(absolute_path))
                    if image_bgr is None:
                        raise RuntimeError("cv2 could not read LFW image")

                    original_height, original_width = image_bgr.shape[:2]
                    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

                    aligned_rgb, transform = detection.align_face(
                        image_rgb=image_rgb,
                        keypoints=primary["kps"],
                        output_size=output_size,
                        canonical_landmark_fractions=landmark_fractions,
                        return_transform=True,
                    )

                    output["project_alignment_applied"] = True

                    if aligned_rgb is None:
                        output["alignment_success"] = False
                        output["skin_roi_invalid_reason"] = "alignment failed"
                        output["processing_success"] = True
                    else:
                        output["alignment_success"] = True
                        output["alignment_transform"] = json.dumps(transform.tolist())

                        aligned_output = align_output_path(relative_path)
                        saved = cv2.imwrite(str(aligned_output), cv2.cvtColor(aligned_rgb, cv2.COLOR_RGB2BGR))
                        if not saved:
                            raise RuntimeError("could not save aligned LFW crop")

                        output["aligned_path"] = str(aligned_output.relative_to(PROJECT_ROOT).as_posix())

                        measured = measurements.measure_face_crop(
                            rgb=aligned_rgb,
                            source_achromatic=source_achromatic,
                            roi_patches=roi_patches,
                            skin_filter=skin_filter,
                            roi_min_valid_pixels=roi_min_valid_pixels,
                            b_star_min_abs=b_star_min_abs,
                            bbox=primary["bbox"],
                            original_width=original_width,
                            original_height=original_height,
                        )
                        output.update(measured)
                        output["processing_success"] = True

            else:
                rgb, color_info = image_integrity.load_rgb_for_analysis(absolute_path)

                measured = measurements.measure_face_crop(
                    rgb=rgb,
                    source_achromatic=source_achromatic,
                    roi_patches=roi_patches,
                    skin_filter=skin_filter,
                    roi_min_valid_pixels=roi_min_valid_pixels,
                    b_star_min_abs=b_star_min_abs,
                    bbox=None,
                    original_width=None,
                    original_height=None,
                )
                output.update(measured)
                output["color_conversion"] = color_info["color_conversion"]
                output["color_conversion_error"] = color_info["color_conversion_error"]
                output["processing_success"] = True

            output["ITA_group"] = categories.assign_ita_group(output["ITA"], gov002)
            output["ITA_binary"] = categories.assign_ita_binary(output["ITA"], gov002)

        except Exception as exc:
            output["processing_error"] = f"{type(exc).__name__}: {exc}"

        rows.append(output)
        done.add(relative_path)
        processed_since_checkpoint += 1

        if processed_since_checkpoint % CHECKPOINT_EVERY == 0:
            save_checkpoint(rows, checkpoint_path)
            print("Processed", len(done), "of", len(master))

    save_checkpoint(rows, checkpoint_path)

    updates = pd.DataFrame(rows)
    if updates["relative_path"].duplicated().any():
        raise RuntimeError("Full extraction checkpoint has duplicate paths.")

    enriched = master_table.enrich_master(master=master, updates=updates, key="relative_path")
    enriched["feature_extraction_complete"] = enriched["processing_success"].fillna(False)

    output_path = paths.PROCESSED_DATA_DIR / "master_annotations_features.parquet"
    enriched.to_parquet(output_path, index=False)

    summary = enriched.groupby("dataset").agg(
        rows=("image_id", "size"),
        feature_complete=("feature_extraction_complete", "sum"),
        skin_valid=("skin_roi_valid", "sum"),
        ita_available=("ITA", "count"),
    )
    summary.to_csv(paths.OUTPUTS_DATA_PREP_DIR / "tables" / "full_feature_extraction_summary.csv")

    print()
    print(summary)
    print()
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
