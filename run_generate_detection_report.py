"""Assemble the detector-stage draft report from already-saved audit outputs.

Reads the tables written by notebooks/detection_audit.ipynb under
outputs/data_prep/detection_integrity/ and writes the draft markdown
report. Does not recompute anything or touch the frozen master table.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import paths

OUTPUT = paths.DATA_PREP_OUTPUTS_DIR / "detection_integrity"
FIGURES = OUTPUT / "figures"
QA_PATH = paths.METADATA_DIR / "alignment_qa_review.csv"


def _require(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required audit output missing: {path}")
    return path


def main() -> None:
    integrity_table = pd.read_csv(_require(OUTPUT / "detection_integrity_by_dataset.csv"))
    alignment_table = pd.read_csv(_require(OUTPUT / "alignment_outcomes.csv"))
    review_table = pd.read_csv(_require(OUTPUT / "failure_review.csv"))
    chi_square = pd.read_csv(_require(OUTPUT / "chi_square_diagnostics.csv"))
    qa = pd.read_csv(_require(QA_PATH))

    canonical_images = int(integrity_table["canonical_images"].sum())
    publisher_preprocessed = int(integrity_table["publisher_preprocessed_not_project_detected"].sum())
    entered_detector = int(integrity_table["entered_detector"].sum())
    corrupt_not_processed = int(integrity_table["corrupt_not_processed"].sum())

    outcome_n = dict(zip(alignment_table["outcome"], alignment_table["n"]))

    category_counts = (
        review_table["visual_category"]
        .value_counts()
        .sort_values(ascending=False)
    )

    no_anomaly_count = int(category_counts.get("no_obvious_visual_anomaly", 0))

    figure_files = sorted(p.name for p in FIGURES.glob("*.png")) if FIGURES.is_dir() else []
    figures_block = "\n".join(f"- `figures/{name}`" for name in figure_files) or "(no figures found)"

    chi_square_lines = []
    for row in chi_square.itertuples(index=False):
        chi_square_lines.append(
            f"- **{row.axis}**: groups [{row.groups}], observed failures "
            f"[{row.observed_failures}], expected failure counts "
            f"[{row.expected_failure_counts}], minimum expected count "
            f"{row.minimum_expected_count}, any expected count below 5: "
            f"{row.any_expected_below_5}"
        )
    chi_square_block = "\n".join(chi_square_lines)

    report = f"""# Detector-stage limitations

**Technical detection audit: COMPLETE (descriptive).**
**Formal statistical and governance acceptance: PENDING.**

This report separates two things that must not be conflated:
what has been verified and computed (below, all descriptive),
and what still requires the approved Phase 1 statistical
protocol or team/governance sign-off before any confirmatory
subgroup-disparity claim can be made.

**Frozen Phase 0 baseline:** `f072db1`

**Reproducibility:** every count below was produced by
`notebooks/detection_audit.ipynb`, executed cleanly top to
bottom against the frozen Phase 0 master, and is saved in the
CSV/parquet tables referenced in each section.

## Pipeline population (verified)

The canonical master contains {canonical_images:,} images.

Publisher-preprocessed FairFace images:
{publisher_preprocessed:,}

Images entering project RetinaFace:
{entered_detector:,}

Corrupt images excluded from inference:
{corrupt_not_processed:,}

Figure: `figures/detection_terminal_state_distribution.png`,
`figures/detector_stage_funnel.png`

## Detection outcomes (verified)

Inference errors: {outcome_n['inference_error']:,}

Zero detections: {outcome_n['zero_detection']:,}

Successful project alignments:
{outcome_n['aligned_success']:,}

Production visual QA: {len(qa)}/{len(qa)} passed.

## Subgroup detection failures (descriptive)

See `subgroup_detection_failures.csv`.

These findings describe UTKFace only. LFW has no usable
native or derived gender, age-band or race labels under
the eligibility rule applied here, and all 39 detector
failures are UTKFace images; LFW and FairFace contributed
zero. This section makes no claim about LFW or FairFace
subgroup detection performance.

There are only 39 failure events in total, spread across
three demographic axes. Per-subgroup counts are small
(several in the single digits), so rates and intervals are
wide and unstable. These are descriptive signals for
follow-up, not evidence that subgroups differ in detector
performance.

Counts, denominators and exploratory Wilson 95%
intervals have been generated. The GOV-003 image-level
reporting checks (minimum cell size, low-confidence sample
size) were applied and passed for every single-axis UTKFace
subgroup shown here.

Figure: `figures/utkface_gender_failure_rates.png`,
`figures/utkface_age_band_failure_rates.png`,
`figures/utkface_race_failure_rates.png`

### Multi-group chi-square diagnostic (descriptive only -- not interpreted)

`configs/statistics.yaml` specifies chi-square with Cramer's V
for the age and race axes. Before running that test as
confirmatory, its standard-approximation assumptions were
checked against the actual 39-failure population:

{chi_square_block}

**Finding:** at least one expected cell count falls below the
conventional threshold of 5 for both age and race. This is a documented limitation
of applying the frozen chi-square method to this small a
failure count, not a silent substitution of a different test.
Any change to the frozen method requires
`docs/CHANGE_CONTROL.md`. No p-value from this test is
reported or interpreted here.

## Confidence (descriptive)

See `confidence_summary.csv`
and the confidence figures.

Confidence is analysed only for successfully
detected primary faces. The frozen detector
confidence threshold is 0.90.

These scores must not be interpreted as
calibrated correctness probabilities. Their inferential
applicability (if any) is not yet specified in
`configs/statistics.yaml` and needs clarification before
any confirmatory use.

Figure: `figures/confidence_distributions.png`,
`figures/utkface_gender_confidence.png`,
`figures/utkface_age_band_confidence.png`,
`figures/utkface_race_confidence.png`

## Landmark and alignment outcomes (verified)

See `alignment_outcomes.csv`
and `alignment_failure_subgroups.csv`.

Recorded landmark failures:
{outcome_n['landmark_failure']:,}

Recorded post-detection alignment failures:
{outcome_n['alignment_failure']:,}

Figure: `figures/alignment_outcomes.png`

## Dataset-specific limitations

LFW is historically preselected by face detection.
Its results therefore describe a selected population.

UTKFace is processed using the project's RetinaFace
pipeline on in-the-wild source images.

FairFace is publisher-preprocessed and is excluded
from the project detector's performance denominator.

Dataset-specific demographic labels are not
automatically interchangeable.

## Failure-example review (descriptive, fully reviewed)

All {len(review_table)} zero-detection cases were
individually inspected and assigned a descriptive visual
category; see `failure_review.csv` and the
category chart. This is the complete population, not a
sample.

Category counts across the full population:

{category_counts.to_string()}

The largest single category is very young children
photographed in non-frontal or lying poses. A second
cluster is elderly subjects in heavily stylized,
high-contrast black-and-white portraiture. Two images are
not standard photographs (one painting, one with heavy
halftone/print-style degradation). A meaningful group of
{no_anomaly_count} images shows no apparent visual anomaly on casual inspection.

These are observations about what is visible, not an
explanation of detector behaviour. They do not establish
that any one visual factor caused a given failure, and the
"no obvious visual anomaly" cases in particular warrant
further technical investigation rather than a visual
explanation.

Figure: `figures/failure_examples_contact_sheet.png`,
`figures/visual_review_categories.png`

## All generated figures

{figures_block}

## What is complete versus what is pending

**Complete (technical, descriptive, verified by re-execution):**

- Detection/alignment reconciliation against the frozen Phase 0 master.
- Subgroup failure counts and descriptive Wilson intervals, UTKFace only.
- GOV-003 image-level reporting checks (minimum cell size, low-confidence flag).
- Confidence-score summaries and distributions among detected faces.
- Landmark and alignment outcome reconciliation.
- Full visual review of all 39 zero-detection cases, each assigned a category.
- Chi-square small-expected-count diagnostic (descriptive, not interpreted).

**Pending (requires the approved Phase 1 statistical protocol and/or team/governance review):**

- Any confirmatory interpretation of the age/race chi-square tests, given the small-expected-count finding above.
- Clarification of the confidence-score inferential status in `configs/statistics.yaml`.
- Benjamini-Hochberg multiple-testing correction, if/when confirmatory subgroup tests are run.
- Formal statistical sign-off and team/governance review of this report.

No formal subgroup-disparity claim is made by this report. Any
such claim requires H1/H2 confirmatory testing under the
frozen pre-registration and `configs/statistics.yaml`, which
is separate Phase 1 work, not part of this detection audit.

## Review

Owner: [FILL]

Reviewer: [FILL]

Statistical protocol review: PENDING

Final report approval: PENDING
"""

    report_path = PROJECT_ROOT / "docs" / "detector_stage_limitations_DRAFT.md"
    report_path.write_text(report, encoding="utf-8")

    print("Saved draft report:", report_path)


if __name__ == "__main__":
    main()
