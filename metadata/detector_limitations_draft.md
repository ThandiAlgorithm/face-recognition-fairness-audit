# DET-005: Detector-stage limitations and exclusions

Status: DRAFT — requires researcher review.

## 1. Scope

Project-native RetinaFace with MobileNetV1 0.25 is used
on eligible canonical LFW and UTKFace images.

FairFace uses publisher-preprocessed face chips and
does not undergo the same project detector.

The two processing routes must be reported separately.

## 2. Detection integrity

| dataset   | route                   | detection_status                |     n |
|:----------|:------------------------|:--------------------------------|------:|
| fairface  | publisher_preprocessing | not_applicable_publisher_route  | 97698 |
| lfw       | project_retinaface      | face_detected                   | 13233 |
| utkface   | project_retinaface      | face_detected                   | 24054 |
| utkface   | project_retinaface      | no_detection                    |    39 |
| utkface   | project_retinaface      | pre_inference_or_colour_failure |    15 |

## 3. Detection failures by dataset

| dataset   | attribute   | native_group   |   eligible_images |   detected |   failures |   failure_rate |   no_detection |   inference_failure |   pre_inference_or_colour_failure | thin_cell   |
|:----------|:------------|:---------------|------------------:|-----------:|-----------:|---------------:|---------------:|--------------------:|----------------------------------:|:------------|
| lfw       | ALL         | ALL            |             13233 |      13233 |          0 |     0          |              0 |                   0 |                                 0 | False       |
| utkface   | ALL         | ALL            |             24093 |      24054 |         39 |     0.00161873 |             39 |                   0 |                                 0 | False       |

Detection failure must be reported with its denominator.
A failure does not prove that an image contains no face.

## 4. Native demographic audit

The complete native-label analysis is retained in:
metadata\subgroup_detection_failures.csv

Unknown labels remain unknown. Native race categories are
not silently mapped between datasets.

Observed subgroup differences are descriptive until the
required statistical protocol is applied.

## 5. Confidence distributions

| dataset   | score_type   |   n_scores |      min |      p05 |   median |     mean |      p95 |      max |        std |
|:----------|:-------------|-----------:|---------:|---------:|---------:|---------:|---------:|---------:|-----------:|
| lfw       | all_retained |      15547 | 0.900085 | 0.977407 | 0.999749 | 0.996169 | 0.999934 | 0.999982 | 0.0129725  |
| lfw       | primary      |      13233 | 0.912591 | 0.998533 | 0.999796 | 0.999526 | 0.999938 | 0.999982 | 0.00147678 |
| utkface   | all_retained |      25436 | 0.900045 | 0.991653 | 0.999638 | 0.997177 | 0.999975 | 0.999999 | 0.0105258  |
| utkface   | primary      |      24054 | 0.900774 | 0.995599 | 0.999678 | 0.998623 | 0.999976 | 0.999999 | 0.00519438 |

These are retained-detection scores after the frozen
confidence threshold of 0.90.

They are not calibrated probabilities or an independent
measure of detection accuracy.

## 6. Landmark and alignment integrity

| dataset   | attribute   | native_group   |   images_in_group |   faces_detected |   missing_or_invalid_primary |   primary_valid |   alignment_failed_after_valid_primary |   valid_alignment |   primary_failure_rate |   alignment_failure_rate_given_valid_primary |   end_to_end_failure_rate_given_detection | thin_cell   |
|:----------|:------------|:---------------|------------------:|-----------------:|-----------------------------:|----------------:|---------------------------------------:|------------------:|-----------------------:|---------------------------------------------:|------------------------------------------:|:------------|
| lfw       | ALL         | ALL            |             13233 |            13233 |                            0 |           13233 |                                      0 |             13233 |                      0 |                                            0 |                                         0 | False       |
| utkface   | ALL         | ALL            |             24093 |            24054 |                            0 |           24054 |                                      0 |             24054 |                      0 |                                            0 |                                         0 | False       |

Automated alignment validity and visual acceptability
are different checks.

## 7. Production visual QA

QA rows: 60

qa_status
pass    60

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
