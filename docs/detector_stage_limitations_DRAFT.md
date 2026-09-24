# Detector-stage limitations

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

The canonical master contains 135,039 images.

Publisher-preprocessed FairFace images:
97,698

Images entering project RetinaFace:
37,326

Corrupt images excluded from inference:
15

Figure: `figures/detection_terminal_state_distribution.png`,
`figures/detector_stage_funnel.png`

## Detection outcomes (verified)

Inference errors: 0

Zero detections: 39

Successful project alignments:
37,287

Production visual QA: 60/60 passed.

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

- **age_band**: groups [0-19, 20-39, 40-59, 60+], observed failures [28, 3, 0, 8], expected failure counts [7.909, 19.281, 7.37, 4.441], minimum expected count 4.441, any expected count below 5: True
- **race**: groups [Asian, Black, Indian, Others, White], observed failures [17, 6, 3, 1, 12], expected failure counts [5.806, 7.379, 6.52, 2.77, 16.525], minimum expected count 2.77, any expected count below 5: True

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
0

Recorded post-detection alignment failures:
0

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

All 39 zero-detection cases were
individually inspected and assigned a descriptive visual
category; see `failure_review.csv` and the
category chart. This is the complete population, not a
sample.

Category counts across the full population:

visual_category
baby_lying_or_turned              15
no_obvious_visual_anomaly          8
elderly_high_contrast_stylized     7
extreme_crop_or_perspective        4
low_light_or_occluded              3
non_photographic_or_degraded       2

The largest single category is very young children
photographed in non-frontal or lying poses. A second
cluster is elderly subjects in heavily stylized,
high-contrast black-and-white portraiture. Two images are
not standard photographs (one painting, one with heavy
halftone/print-style degradation). A meaningful group of
8 images shows no apparent visual anomaly on casual inspection.

These are observations about what is visible, not an
explanation of detector behaviour. They do not establish
that any one visual factor caused a given failure, and the
"no obvious visual anomaly" cases in particular warrant
further technical investigation rather than a visual
explanation.

Figure: `figures/failure_examples_contact_sheet.png`,
`figures/visual_review_categories.png`

## All generated figures

- `figures/alignment_outcomes.png`
- `figures/confidence_distributions.png`
- `figures/detection_terminal_state_distribution.png`
- `figures/detector_stage_funnel.png`
- `figures/failure_examples_contact_sheet.png`
- `figures/utkface_age_band_confidence.png`
- `figures/utkface_age_band_failure_rates.png`
- `figures/utkface_gender_confidence.png`
- `figures/utkface_gender_failure_rates.png`
- `figures/utkface_race_confidence.png`
- `figures/utkface_race_failure_rates.png`
- `figures/visual_review_categories.png`

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
