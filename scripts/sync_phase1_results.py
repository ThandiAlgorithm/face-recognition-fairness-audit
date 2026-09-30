"""Sync Phase 1 audit outputs into the git-tracked results/phase1_audit/ folder.

outputs/phase1/ and outputs/data_prep/ are gitignored and regenerated whenever
run_phase1_audit.py (or the Phase 0 pipeline) reruns, so the Streamlit dashboard
reads a curated, git-tracked copy under results/phase1_audit/ instead. This
script is idempotent: rerun it any time the source outputs are refreshed.

Usage:
    python scripts/sync_phase1_results.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_PHASE1 = PROJECT_ROOT / "outputs" / "phase1"
OUTPUTS_DATA_PREP = PROJECT_ROOT / "outputs" / "data_prep"
DEST = PROJECT_ROOT / "results" / "phase1_audit"

MEA_TABLES = [
    "MEA_boundary_sensitivity.csv",
    "MEA_reliability_agreement.csv",
    "MEA_roi_failure_by_group.csv",
    "MEA_roi_failure_tests.csv",
    "MEA_skin_group_distribution.csv",
    "MEA_status_counts.csv",
    "MEA_subgroup_omnibus.csv",
    "MEA_subgroup_pairwise.csv",
    "MEA_subgroup_summary.csv",
    "MEA_summary_by_dataset.csv",
]
MEA_FIGURES = [
    "MEA_BA_fairface_rel_ITA_forehead_vs_rel_ITA_left_cheek.png",
    "MEA_BA_fairface_rel_ITA_forehead_vs_rel_ITA_right_cheek.png",
    "MEA_BA_fairface_rel_ITA_left_cheek_vs_rel_ITA_right_cheek.png",
    "MEA_BA_fairface_skin_ITA_deg_vs_phase0_ITA.png",
    "MEA_BA_fairface_skin_ITA_deg_vs_rel_ITA_mean_based.png",
    "MEA_BA_lfw_rel_ITA_forehead_vs_rel_ITA_left_cheek.png",
    "MEA_BA_lfw_rel_ITA_forehead_vs_rel_ITA_right_cheek.png",
    "MEA_BA_lfw_rel_ITA_left_cheek_vs_rel_ITA_right_cheek.png",
    "MEA_BA_lfw_skin_ITA_deg_vs_phase0_ITA.png",
    "MEA_BA_lfw_skin_ITA_deg_vs_rel_ITA_mean_based.png",
    "MEA_BA_utkface_rel_ITA_forehead_vs_rel_ITA_left_cheek.png",
    "MEA_BA_utkface_rel_ITA_forehead_vs_rel_ITA_right_cheek.png",
    "MEA_BA_utkface_rel_ITA_left_cheek_vs_rel_ITA_right_cheek.png",
    "MEA_BA_utkface_skin_ITA_deg_vs_phase0_ITA.png",
    "MEA_BA_utkface_skin_ITA_deg_vs_rel_ITA_mean_based.png",
    "MEA_hist_ITA.png",
    "MEA_hist_skin_L_star.png",
    "MEA_hist_skin_chroma.png",
    "MEA_hist_skin_hue_deg.png",
]

VIS_TABLES = [
    "VIS_brightness_vs_skin_L.csv",
    "VIS_qc_flag_counts.csv",
    "VIS_status_counts.csv",
    "VIS_subgroup_omnibus.csv",
    "VIS_subgroup_pairwise.csv",
    "VIS_subgroup_summary.csv",
    "VIS_summary_by_dataset.csv",
]
VIS_FIGURES = [
    "VIS_box_brightness_L_mean_by_skin_group_ita.png",
    "VIS_box_contrast_L_sd_by_skin_group_ita.png",
    "VIS_box_face_area_ratio_by_skin_group_ita.png",
    "VIS_box_face_short_side_px_by_skin_group_ita.png",
    "VIS_box_illumination_asymmetry_by_skin_group_ita.png",
    "VIS_box_pitch_deg_by_skin_group_ita.png",
    "VIS_box_roll_deg_by_skin_group_ita.png",
    "VIS_box_sharpness_lapvar_by_skin_group_ita.png",
    "VIS_box_yaw_deg_by_skin_group_ita.png",
    "VIS_hist_brightness_L_mean.png",
    "VIS_hist_contrast_L_sd.png",
    "VIS_hist_face_area_ratio.png",
    "VIS_hist_face_short_side_px.png",
    "VIS_hist_illumination_asymmetry.png",
    "VIS_hist_pitch_deg.png",
    "VIS_hist_roll_deg.png",
    "VIS_hist_sharpness_lapvar.png",
    "VIS_hist_yaw_deg.png",
]
VIS_QA = [
    "VIS_extremes_brightness_L_mean.png",
    "VIS_extremes_sharpness_lapvar.png",
    "VIS_extremes_yaw_deg.png",
]

SHIFT_TABLES = [
    "SHIFT_analysis_set_counts.csv",
    "SHIFT_categorical_distributions.csv",
    "SHIFT_categorical_tests.csv",
    "SHIFT_continuous_summary.csv",
    "SHIFT_continuous_tests.csv",
    "SHIFT_effective_balance_summary.csv",
    "SHIFT_effective_balance_tests.csv",
    "SHIFT_identity_depth.csv",
    "SHIFT_ita_within_native_race.csv",
    "SHIFT_nominal_balance.csv",
]

LFW_TABLES = [
    "LFW_composition_images_vs_pairs.csv",
    "LFW_composition_tests.csv",
    "LFW_genuine_visual_difference_by_skin_group.csv",
    "LFW_genuine_visual_difference_tests.csv",
    "LFW_impostor_same_group_vs_random.csv",
    "LFW_impostor_yoking_counts.csv",
    "LFW_pair_file_checks.csv",
    "LFW_pair_resolution_counts.csv",
    "LFW_visual_difference_median_by_yoking.csv",
]

DETECTION_FIGURES = [
    "alignment_outcomes.png",
    "confidence_distributions.png",
    "detector_stage_funnel.png",
    "failure_examples_contact_sheet.png",
    "utkface_age_band_confidence.png",
    "utkface_age_band_failure_rates.png",
    "utkface_gender_confidence.png",
    "utkface_gender_failure_rates.png",
    "utkface_race_confidence.png",
    "utkface_race_failure_rates.png",
    "visual_review_categories.png",
]


def copy_files(pairs: list[tuple[Path, Path]]) -> tuple[int, list[Path]]:
    copied = 0
    missing: list[Path] = []
    for src, dest in pairs:
        if not src.exists():
            missing.append(src)
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        copied += 1
    return copied, missing


def build_copy_map() -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []

    for name in MEA_TABLES:
        pairs.append((OUTPUTS_PHASE1 / "measurement_audit" / "tables" / name, DEST / "measurement_audit" / "tables" / name))
    for name in MEA_FIGURES:
        pairs.append((OUTPUTS_PHASE1 / "measurement_audit" / "figures" / name, DEST / "measurement_audit" / "figures" / name))
    pairs.append((OUTPUTS_PHASE1 / "measurement_audit" / "qa" / "MEA_roi_overlay_sample.png", DEST / "measurement_audit" / "qa" / "MEA_roi_overlay_sample.png"))
    pairs.append((OUTPUTS_PHASE1 / "measurement_audit" / "metadata" / "MEA_method.json", DEST / "measurement_audit" / "metadata" / "MEA_method.json"))
    pairs.append((OUTPUTS_PHASE1 / "measurement_audit" / "MEA_limitations_draft.md", DEST / "measurement_audit" / "MEA_limitations_draft.md"))

    for name in VIS_TABLES:
        pairs.append((OUTPUTS_PHASE1 / "visual_audit" / "tables" / name, DEST / "visual_audit" / "tables" / name))
    for name in VIS_FIGURES:
        pairs.append((OUTPUTS_PHASE1 / "visual_audit" / "figures" / name, DEST / "visual_audit" / "figures" / name))
    for name in VIS_QA:
        pairs.append((OUTPUTS_PHASE1 / "visual_audit" / "qa" / name, DEST / "visual_audit" / "qa" / name))
    pairs.append((OUTPUTS_PHASE1 / "visual_audit" / "metadata" / "VIS_method.json", DEST / "visual_audit" / "metadata" / "VIS_method.json"))

    for name in SHIFT_TABLES:
        pairs.append((OUTPUTS_PHASE1 / "shift_audit" / "tables" / name, DEST / "shift_audit" / "tables" / name))
    pairs.append((OUTPUTS_PHASE1 / "shift_audit" / "figures" / "SHIFT_skin_group_by_dataset.png", DEST / "shift_audit" / "figures" / "SHIFT_skin_group_by_dataset.png"))
    pairs.append((OUTPUTS_PHASE1 / "shift_audit" / "metadata" / "SHIFT_frozen_axes.json", DEST / "shift_audit" / "metadata" / "SHIFT_frozen_axes.json"))
    pairs.append((OUTPUTS_PHASE1 / "shift_audit" / "SHIFT_interpretation_limits_draft.md", DEST / "shift_audit" / "SHIFT_interpretation_limits_draft.md"))

    for name in LFW_TABLES:
        pairs.append((OUTPUTS_PHASE1 / "lfw_audit" / "tables" / name, DEST / "lfw_audit" / "tables" / name))
    pairs.append((OUTPUTS_PHASE1 / "lfw_audit" / "metadata" / "LFW_status.json", DEST / "lfw_audit" / "metadata" / "LFW_status.json"))
    pairs.append((OUTPUTS_PHASE1 / "lfw_audit" / "metadata" / "LFW_protocol_rules.json", DEST / "lfw_audit" / "metadata" / "LFW_protocol_rules.json"))

    for name in DETECTION_FIGURES:
        pairs.append((OUTPUTS_DATA_PREP / "detection_integrity" / "figures" / name, DEST / "detection_audit" / "figures" / name))

    qa = OUTPUTS_DATA_PREP / "qa"
    pairs.append((qa / "detector_pilot" / "detection_alignment_qa.png", DEST / "alignment_qa" / "detector_pilot" / "detection_alignment_qa.png"))
    pairs.append((qa / "detector_pilot" / "supplied_crop_qa.png", DEST / "alignment_qa" / "detector_pilot" / "supplied_crop_qa.png"))
    pairs.append((qa / "measurement_examples" / "skin_roi_qa.png", DEST / "alignment_qa" / "measurement_examples" / "skin_roi_qa.png"))
    pairs.append((OUTPUTS_DATA_PREP / "figures" / "achromatic_chroma_distribution.png", DEST / "alignment_qa" / "achromatic" / "achromatic_chroma_distribution.png"))
    pairs.append((OUTPUTS_DATA_PREP / "figures" / "achromatic_sample.png", DEST / "alignment_qa" / "achromatic" / "achromatic_sample.png"))
    pairs.append((OUTPUTS_DATA_PREP / "figures" / "mean_rule_only_flagged_sample.png", DEST / "alignment_qa" / "achromatic" / "mean_rule_only_flagged_sample.png"))

    for src in sorted((qa / "alignment").glob("*.aligned.png")):
        pairs.append((src, DEST / "alignment_qa" / "samples" / src.name))

    return pairs


def convert_jpg_to_png(src: Path, dest: Path) -> bool:
    if not src.exists():
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        im.convert("RGB").save(dest, "PNG")
    return True


def build_alignment_qa_manifest() -> pd.DataFrame:
    rows = []
    for path in sorted((DEST / "alignment_qa" / "samples").glob("*.aligned.png")):
        parts = path.stem.removesuffix(".aligned").split("__")
        dataset = parts[0] if parts else "unknown"
        original = parts[-1] if len(parts) > 1 else path.stem
        context = "/".join(parts[1:-1]) if len(parts) > 2 else ""
        rows.append({"dataset": dataset, "context": context, "original_filename": original, "filename": path.name})
    manifest = pd.DataFrame(rows)
    dest_path = DEST / "alignment_qa" / "samples_manifest.csv"
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(dest_path, index=False)
    return manifest


def build_brightness_vs_skin_tone() -> pd.DataFrame | None:
    src = OUTPUTS_PHASE1 / "visual_audit" / "tables" / "VIS_image_measurements.parquet"
    if not src.exists():
        return None
    df = pd.read_parquet(src, columns=[
        "image_id", "dataset", "gender", "race", "age_band",
        "brightness_L_mean", "contrast_L_sd", "illumination_asymmetry",
        "skin_L_star", "skin_ITA_deg", "ita_group", "skin_group_ita",
        "in_analysis",
    ])
    df = df.dropna(subset=["brightness_L_mean", "skin_L_star"])
    dest_path = DEST / "combined" / "brightness_vs_skin_tone.parquet"
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest_path, index=False)
    return df


def main() -> None:
    print(f"Source outputs/phase1: {OUTPUTS_PHASE1} ({'exists' if OUTPUTS_PHASE1.exists() else 'MISSING'})")
    print(f"Source outputs/data_prep: {OUTPUTS_DATA_PREP} ({'exists' if OUTPUTS_DATA_PREP.exists() else 'MISSING'})")
    print(f"Destination: {DEST}\n")

    pairs = build_copy_map()
    copied, missing = copy_files(pairs)
    total_bytes = sum((d.stat().st_size for _, d in pairs if d.exists()))
    print(f"Copied {copied}/{len(pairs)} files ({total_bytes / 1_000_000:.1f} MB)")
    if missing:
        print(f"WARNING: {len(missing)} source files were missing and skipped:")
        for m in missing:
            print(f"  - {m}")

    converted = 0
    for src, dest in [
        (OUTPUTS_DATA_PREP / "qa" / "alignment_contact_sheet.jpg", DEST / "alignment_qa" / "contact_sheets" / "alignment_contact_sheet.png"),
        (PROJECT_ROOT / "metadata" / "phase0_smoke_20" / "outputs" / "qa" / "alignment_contact_sheet.jpg", DEST / "alignment_qa" / "contact_sheets" / "alignment_contact_sheet_smoke20.png"),
    ]:
        if convert_jpg_to_png(src, dest):
            converted += 1
        else:
            print(f"WARNING: contact sheet source missing, skipped: {src}")
    print(f"Converted {converted}/2 jpg contact sheets to png")

    manifest = build_alignment_qa_manifest()
    print(f"Built alignment_qa/samples_manifest.csv ({len(manifest)} rows)")

    combined = build_brightness_vs_skin_tone()
    if combined is not None:
        print(f"Built combined/brightness_vs_skin_tone.parquet ({len(combined):,} rows)")
    else:
        print("WARNING: VIS_image_measurements.parquet missing, skipped brightness_vs_skin_tone derivation")

    print("\nDone.")


if __name__ == "__main__":
    main()
