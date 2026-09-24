# Detector-stage limitations

**Status:** DRAFT — pending review of the full detector-stage audit.

## Detector-stage denominator

Project RetinaFace detection is evaluated only on
canonical images requiring project detection that were
successfully decoded.

Corrupt source images remain in the master table but do
not enter detector inference.

FairFace uses publisher-preprocessed face chips.
FairFace images are therefore not counted as project
RetinaFace detection or alignment successes.

## Failure states

The pipeline distinguishes:

- Inference errors
- Zero detections after successful inference
- Multiple detections
- Primary-face selection failures
- Missing or invalid five-point landmarks
- Project alignment failures
- Successful project alignments

Multiple detections are a diagnostic condition, not
automatically a failure.

Every failure retains its image-level record.

## Interpretation

Subgroup differences measured at the detector stage
describe the project preprocessing pipeline.

They do not, by themselves, establish bias in a later
recognition or attribute-classification model.

Excluding detection and alignment failures from later
analyses changes the population reaching those models.
Detector-stage attrition must therefore be reported.

## Dataset-specific limitations

LFW was originally selected using a face detector.
Its detection outcomes are conditioned on that
historically selected population.

UTKFace uses in-the-wild source images.

FairFace is publisher-preprocessed and does not enter
project RetinaFace inference.

Native race labels must not be pooled across
incompatible dataset-specific schemes.

## Results to insert

### Detection integrity
Insert the reconciled detector and alignment counts
for LFW, UTKFace and FairFace.

### Subgroup detection failures
Insert subgroup failure counts, denominators,
approved confidence intervals and statistical tests.

### Confidence distributions
Insert confidence distribution summaries,
approved effect sizes and relevant statistical tests.

Remember that these confidence values exist only
for successfully detected faces.

### Landmark and alignment outcomes
Insert landmark and alignment failure counts,
denominators and approved confidence intervals.

## Final denominator statement

[Insert the exact approved eligibility-mask wording.]

## Review

Owner: TM

Reviewer: RZ

Review status: PENDING
