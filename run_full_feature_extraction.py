from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import categories, config, detection, image_integrity, master_table, measurements, paths

CHECKPOINT_EVERY = 250


def _aligned_output_path(dataset: str, image_id: str) -> Path:
    directory = paths.ALIGNED_DIR / dataset
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{image_id}.png"


def _load_checkpoint(path: Path) -> tuple[list[dict], set[str]]:
    if not path.exists():
        return [], set()
    frame = pd.read_parquet(path)
    return frame.to_dict("records"), set(frame["image_id"].astype(str))


def _save_checkpoint(rows: list[dict], path: Path) -> None:
    pd.DataFrame(rows).to_parquet(path, index=False)


def _to_serializable_detections(items: list[dict]) -> list[dict]:
    output = []
    for item in items:
        output.append(
            {
                "bbox": [float(value) for value in item["bbox"]],
                "score": float(item["score"]),
                "landmarks": [
                    [float(point[0]), float(point[1])]
                    for point in item["landmarks"]
                ],
            }
        )
    return output


def _create_qa_contact_sheet(review: pd.DataFrame, output_path: Path) -> None:
    pairs = []
    for row in review.itertuples(index=False):
        if row.aligned_path is None or pd.isna(row.aligned_path):
            continue
        try:
            source_rgb, _ = image_integrity.load_rgb_for_analysis(paths.data_abs(row.original_path))
            source = Image.fromarray(source_rgb).convert("RGB")
            aligned_candidate = paths.PROJECT_ROOT / str(row.aligned_path)
            if not aligned_candidate.exists():
                aligned_candidate = paths.data_abs(str(row.aligned_path))
            aligned = Image.open(aligned_candidate).convert("RGB")
            pairs.append((row.image_id, row.dataset, source, aligned))
        except Exception:
            continue

    if not pairs:
        return

    thumb_w = 220
    thumb_h = 220
    label_h = 34
    cell_w = thumb_w * 2
    cell_h = thumb_h + label_h
    canvas = Image.new("RGB", (cell_w, cell_h * len(pairs)), "white")
    draw = ImageDraw.Draw(canvas)

    for index, (image_id, dataset, source, aligned) in enumerate(pairs):
        source.thumbnail((thumb_w, thumb_h))
        aligned.thumbnail((thumb_w, thumb_h))
        y = index * cell_h + label_h
        canvas.paste(source, (0, y))
        canvas.paste(aligned, (thumb_w, y))
        draw.text((4, index * cell_h + 4), f"{dataset} {image_id}", fill="black")

    canvas.save(output_path)


def _build_qa_review(master: pd.DataFrame, manifest: pd.DataFrame, preprocessing: dict) -> pd.DataFrame:
    source = manifest.loc[
        manifest["file_role"].eq("image_candidate"),
        ["original_path", "project_alignment_required"],
    ]
    eligible = master.merge(source, on="original_path", how="left", validate="one_to_one")
    eligible = eligible[
        eligible["project_alignment_required"].fillna(False)
        & eligible["alignment_valid"].fillna(False)
    ].copy()

    seed = int(preprocessing["alignment_qa"]["seed"])
    per_dataset = int(preprocessing["alignment_qa"]["per_dataset"])
    samples = []
    for dataset, group in eligible.groupby("dataset"):
        samples.append(group.sample(n=min(per_dataset, len(group)), random_state=seed))
    review = pd.concat(samples, ignore_index=True) if samples else pd.DataFrame()
    if review.empty:
        return review

    review = review[["image_id", "dataset", "original_path", "aligned_path"]].copy()
    review["qa_status"] = "pending"
    review["qa_notes"] = ""

    review_path = paths.METADATA_DIR / "alignment_qa_review.csv"
    if review_path.exists():
        prior = pd.read_csv(review_path)
        prior_lookup = prior.set_index("image_id")[["qa_status", "qa_notes"]]
        for index, row in review.iterrows():
            if row["image_id"] in prior_lookup.index:
                review.at[index, "qa_status"] = prior_lookup.loc[row["image_id"], "qa_status"]
                review.at[index, "qa_notes"] = prior_lookup.loc[row["image_id"], "qa_notes"]
    return review


def main() -> None:
    paths.ensure_dirs()
    preprocessing = config.load_preprocessing()
    categories_cfg = config.load_categories()
    exclusions = config.load_exclusions()
    manifest = pd.read_parquet(paths.METADATA_DIR / "file_manifest.parquet")
    master = pd.read_parquet(paths.METADATA_DIR / "master_annotations.parquet")

    source = manifest.loc[
        manifest["file_role"].eq("image_candidate"),
        [
            "original_path",
            "source_preprocessed",
            "project_detection_required",
            "project_alignment_required",
        ],
    ]
    work = master.merge(source, on="original_path", how="left", validate="one_to_one")

    checkpoint_path = paths.CACHE_DIR / "detection_alignment_measurement.parquet"
    rows, done = _load_checkpoint(checkpoint_path)

    raw_pending = work[
        work["project_detection_required"].fillna(False)
        & ~work["image_id"].isin(done)
        & ~work["corrupt"].fillna(True)
    ]
    model = None
    if len(raw_pending):
        print("Building RetinaFace model")
        model = detection.build_retinaface_model()

    processed_since_checkpoint = 0
    for item in work.itertuples(index=False):
        if item.image_id in done:
            continue

        update = {
            "image_id": item.image_id,
            "bbox": None,
            "det_confidence": None,
            "n_detections": None,
            "no_detection": None,
            "multi_detection": None,
            "detections_all": None,
            "landmarks": None,
            "alignment_transform": None,
            "aligned_path": None,
            "alignment_valid": None,
            "brightness": None,
            "contrast": None,
            "sharpness": None,
            "face_px_area": None,
            "face_area_ratio": None,
            "yaw": None,
            "pitch": None,
            "roll": None,
            "L_star": None,
            "a_star": None,
            "b_star": None,
            "ITA": None,
            "hue_angle": None,
            "ita_group": None,
            "ita_ordinal": None,
            "fitz_classifier": None,
            "fitz_classifier_conf": None,
            "fitz_ordinal": None,
            "skin_group_ita": None,
            "skin_group_fitz": None,
            "skin_roi_valid": None,
            "ita_undefined": None,
            "inference_success": None,
            "inference_error": None,
            "colour_conversion_success": None,
            "colour_conversion_error": None,
            "skin_roi_total_pixel_count": None,
            "skin_roi_valid_pixel_count": None,
            "skin_valid_fraction": None,
            "skin_invalid_reason": None,
        }

        try:
            if item.corrupt:
                rows.append(update)
                done.add(item.image_id)
                continue

            rgb, colour_info = image_integrity.load_rgb_for_analysis(paths.data_abs(item.original_path))
            update["colour_conversion_success"] = colour_info["colour_conversion_success"]
            update["colour_conversion_error"] = colour_info["colour_conversion_error"]
            if not colour_info["colour_conversion_success"]:
                raise RuntimeError("Colour conversion failed")

            original_height, original_width = rgb.shape[:2]
            valid_mask = np.ones((original_height, original_width), dtype=bool)
            aligned_rgb = None
            bbox = None

            if item.project_detection_required:
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                result = detection.detect_faces(bgr, preprocessing, model)
                update["inference_success"] = result["inference_success"]
                update["inference_error"] = result["inference_error"]
                if not result["inference_success"]:
                    rows.append(update)
                    done.add(item.image_id)
                    continue

                detections_all = _to_serializable_detections(result["detections"])
                update["detections_all"] = detections_all
                update["n_detections"] = int(result["n_detections"])
                update["no_detection"] = result["n_detections"] == 0
                update["multi_detection"] = result["n_detections"] > 1
                primary = detection.select_primary_face(result["detections"], preprocessing)

                if primary is None:
                    update["alignment_valid"] = False
                    rows.append(update)
                    done.add(item.image_id)
                    continue

                bbox = [float(value) for value in primary["bbox"]]
                update["bbox"] = bbox
                update["det_confidence"] = float(primary["score"])
                update["landmarks"] = [
                    [float(point[0]), float(point[1])]
                    for point in primary["landmarks"]
                ]

                aligned_rgb, transform, valid_mask = detection.align_face(
                    rgb,
                    primary["landmarks"],
                    preprocessing,
                )
                if aligned_rgb is None or transform is None:
                    update["alignment_valid"] = False
                    rows.append(update)
                    done.add(item.image_id)
                    continue

                aligned_path = _aligned_output_path(item.dataset, item.image_id)
                saved = cv2.imwrite(
                    str(aligned_path),
                    cv2.cvtColor(aligned_rgb, cv2.COLOR_RGB2BGR),
                )
                if not saved:
                    raise RuntimeError("Failed to save aligned crop")
                update["alignment_valid"] = True
                update["alignment_transform"] = transform.tolist()
                update["aligned_path"] = paths.project_rel(aligned_path)

            elif item.source_preprocessed:
                aligned_rgb = rgb
                update["alignment_valid"] = True
                update["aligned_path"] = item.original_path
                update["inference_success"] = None
                valid_mask = np.ones(aligned_rgb.shape[:2], dtype=bool)

            else:
                raise RuntimeError("Image is neither raw for project detection nor source preprocessed")

            measure, diagnostics = measurements.measure_aligned_face(
                aligned_rgb=aligned_rgb,
                preprocessing_cfg=preprocessing,
                is_greyscale=bool(item.is_greyscale) if pd.notna(item.is_greyscale) else False,
                valid_mask=valid_mask,
                bbox=bbox,
                original_width=original_width if bbox is not None else None,
                original_height=original_height if bbox is not None else None,
            )
            update.update(measure)
            update.update(diagnostics)
            update["ita_group"] = categories.assign_ita_group(update["ITA"], categories_cfg)
            update["ita_ordinal"] = categories.assign_ita_ordinal(update["ITA"], categories_cfg)
            update["skin_group_ita"] = categories.assign_ita_binary(update["ITA"], categories_cfg)

        except Exception as exc:
            update["inference_error"] = update["inference_error"] or f"{type(exc).__name__}: {exc}"

        rows.append(update)
        done.add(item.image_id)
        processed_since_checkpoint += 1
        if processed_since_checkpoint % CHECKPOINT_EVERY == 0:
            _save_checkpoint(rows, checkpoint_path)
            print("Processed", len(done), "of", len(work))

    _save_checkpoint(rows, checkpoint_path)
    updates = pd.DataFrame(rows)
    if updates["image_id"].duplicated().any():
        raise RuntimeError("Detection checkpoint contains duplicate image IDs")

    diagnostics_path = paths.METADATA_DIR / "detection_alignment_diagnostics.parquet"
    updates.to_parquet(diagnostics_path, index=False)

    master_update_columns = [
        column
        for column in master.columns
        if column in updates.columns and column != "image_id"
    ]
    master_updates = updates[["image_id", *master_update_columns]].copy()
    enriched = master_table.apply_detection_updates(master, master_updates)
    enriched = master_table.apply_measurement_categories(enriched, categories_cfg)
    enriched = master_table.apply_eligibility(enriched, manifest, exclusions)
    enriched = master_table.finalise_master(enriched)
    enriched.to_parquet(paths.METADATA_DIR / "master_annotations.parquet", index=False)

    qa = _build_qa_review(enriched, manifest, preprocessing)
    qa_path = paths.METADATA_DIR / "alignment_qa_review.csv"
    qa.to_csv(qa_path, index=False)
    _create_qa_contact_sheet(
        qa,
        paths.DATA_PREP_OUTPUTS_DIR / "qa" / "alignment_contact_sheet.jpg",
    )

    model_info = detection.get_model_info(preprocessing)
    model_info_path = paths.DATA_PREP_OUTPUTS_DIR / "config" / "retinaface_runtime.json"
    model_info_path.write_text(json.dumps(model_info, indent=2), encoding="utf-8")

    print("Saved:", diagnostics_path)
    print("Saved:", qa_path)
    print("QA rows pending:", int(qa["qa_status"].eq("pending").sum()) if len(qa) else 0)


if __name__ == "__main__":
    main()
