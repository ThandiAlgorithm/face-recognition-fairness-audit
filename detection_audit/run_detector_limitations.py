from pathlib import Path

import pandas as pd


META = Path("metadata")

DET001 = META / "detection_integrity_by_dataset.csv"
DET002 = META / "subgroup_detection_failures.csv"
DET003 = META / "confidence_summary.csv"
DET004 = META / "alignment_failures_by_subgroup.csv"
QA = META / "alignment_qa_review.csv"

OUTPUT = META / "detector_limitations_draft.md"


def main():
    inputs = [DET001, DET002, DET003, DET004]

    missing = [
        str(path)
        for path in inputs
        if not path.exists()
    ]

    if missing:
        raise RuntimeError(
            "Complete previous detection audits first:\n"
            + "\n".join(missing)
        )

    integrity = pd.read_csv(DET001)
    failures = pd.read_csv(DET002)
    confidence = pd.read_csv(DET003)
    alignment = pd.read_csv(DET004)

    qa_text = (
        "Production visual QA file not available. "
        "Human review remains outstanding."
    )

    if QA.exists():
        qa = pd.read_csv(QA)

        qa_text = (
            f"QA rows: {len(qa)}\n\n"
            + qa["qa_status"]
                .value_counts(dropna=False)
                .to_string()
        )

    # Only use dataset-wide rows for this overview.
    detection_overview = failures[
        failures["attribute"].eq("ALL")
    ].copy()

    alignment_overview = alignment[
        alignment["attribute"].eq("ALL")
    ].copy()

    # Show actual calculated values as evidence tables.
    text = f"""# DET-005: Detector-stage limitations and exclusions

Status: DRAFT — requires researcher review.

## 1. Scope

Project-native RetinaFace with MobileNetV1 0.25 is used
on eligible canonical LFW and UTKFace images.

FairFace uses publisher-preprocessed face chips and
does not undergo the same project detector.

The two processing routes must be reported separately.

## 2. Detection integrity

{integrity.to_markdown(index=False)}

## 3. Detection failures by dataset

{detection_overview.to_markdown(index=False)}

Detection failure must be reported with its denominator.
A failure does not prove that an image contains no face.

## 4. Native demographic audit

The complete native-label analysis is retained in:
{DET002}

Unknown labels remain unknown. Native race categories are
not silently mapped between datasets.

Observed subgroup differences are descriptive until the
required statistical protocol is applied.

## 5. Confidence distributions

{confidence.to_markdown(index=False)}

These are retained-detection scores after the frozen
confidence threshold of 0.90.

They are not calibrated probabilities or an independent
measure of detection accuracy.

## 6. Landmark and alignment integrity

{alignment_overview.to_markdown(index=False)}

Automated alignment validity and visual acceptability
are different checks.

## 7. Production visual QA

{qa_text}

Outstanding visual QA or failures must be reviewed and
recorded before declaring this stage complete.

## 8. Row retention and downstream eligibility

All canonical image rows remain in the master inventory.

Corrupt images, inference failures, no-detection cases,
invalid landmarks and invalid alignments must retain
their observed status rather than disappearing.

Analyses requiring a valid aligned project-detected face
must restrict eligibility appropriately and report
their excluded counts.

The exact master eligibility flags and their documented
exclusion-reason precedence must be checked against
the frozen data dictionary before final use.

FairFace must not be credited with project-detector
successes or failures.

## 9. Interpretation limitations

Subgroup differences in observed failure rates do not
by themselves establish the cause of those differences.

Differences in image acquisition, quality, resolution,
pose, occlusion, source preprocessing and demographic
annotation coverage may affect interpretation.

Results describe this specific detector, checkpoint,
threshold, input pipeline and dataset population.

## 10. Required manual completion

- Review production visual QA findings.
- Describe any notable error categories from DET-001.
- Incorporate the final STAT-001 and STAT-004 analyses.
- Reconcile exclusions with the frozen data dictionary.
- Record any remaining unresolved limitations.
"""

    OUTPUT.write_text(
        text,
        encoding="utf-8",
    )

    print("Draft created:", OUTPUT)
    print("Review the text before using it in a report.")


if __name__ == "__main__":
    main()
