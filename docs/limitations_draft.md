# Phase 0 methodology limitations

These limitations describe the completed Phase 0 pipeline.
Historical pilot implementations are distinguished from the
final implementation.

## LFW: original unaligned images

The canonical LFW source is the original unaligned variant,
not the deep-funneled variant. Results are not automatically
comparable with studies using different LFW preprocessing
variants.

The original LFW archive was obtained through a Figshare
mirror. Its MD5 was verified against the checksum documented
by torchvision, and its actual SHA-256 is pinned in the
acquisition and preprocessing records.

## LFW: prior face-detection filtering

LFW was originally constructed using a face-detection
selection process. Consequently, RetinaFace detection
outcomes measured on LFW describe this selected image
population, not performance on arbitrary photographs.

The project reports LFW and UTKFace detector outcomes
separately, with this limitation attached to LFW.

FairFace cannot be included in the project RetinaFace
detection-failure denominator, because its images were not
processed by the project detector.

## Final dataset-specific preprocessing

LFW: original unaligned images.

UTKFace: in-the-wild images, not the aligned or crop_part1
variants.

Both datasets underwent EXIF-orientation and colour handling,
followed by project-native PyTorch RetinaFace using the
MobileNetV1 0.25 backbone. The configured detection threshold
was 0.90. All detections were retained in the diagnostics;
the primary face was selected by largest bounding-box area,
with confidence as the tie-breaker.

Successful primary detections were aligned using five
landmarks and a similarity transformation to produce
224 x 224 images. The transformation was preserved.

FairFace: official publisher-preprocessed dlib face chips
with padding 0.25. These images were used directly.
Project RetinaFace was not run on FairFace.

These different source-processing routes limit direct
cross-dataset comparisons of detector outcomes.

## Historical detector implementations

An early pilot briefly used SCRFD. Later intermediate work
used TensorFlow RetinaFace and a PyTorch wrapper. Those
implementations were superseded.

The completed Phase 0 implementation uses project-native
PyTorch RetinaFace with the MobileNetV1 0.25 backbone.

Historical TensorFlow/Keras compatibility problems do not
describe a dependency of the final detector. The changes
are traceable in AMEND-20260919-01.

The final checkpoint passed strict state-dictionary loading
and a forward-pass test. Its SHA-256 and file size are
verified against the current configuration.

## Stable identifiers and row retention

The completed master uses the canonical GOV-005 identifier
formula, with the dataset prefix and the first 16 characters
of SHA-256(dataset_version + "|" + original_path).

The master contains 135,039 unique image IDs. Corrupt
images, failed detections and invalid measurements retain
their rows and recorded outcomes.

## Final detection and alignment outcomes

Of the 37,341 canonical LFW and UTKFace images, 15 UTKFace
images were corrupt and retained without inference.

RetinaFace inference succeeded on all 37,326 eligible raw
images. It returned zero faces for 39 images. The other
37,287 images received valid project alignments.

Together with 97,698 publisher-preprocessed FairFace
images, the final master records 134,985 valid alignments.

All 60 sampled production alignments passed visual QA.
This sample result does not establish that every production
alignment has been individually inspected.

## Skin-measurement limitations

Valid skin ROIs were recorded for 128,538 images.

There were 4,738 invalid ROIs due to greyscale or
insufficient chroma and 1,709 additional invalid ROIs.

Another 54 images had no skin-ROI measurement:
39 no-detection cases and 15 corrupt images.

Invalid or unavailable skin measurements remain recorded
as such. Skin-tone proxies are measurements, not
ground-truth demographic identities or accuracy measures.

## Source-archive availability

The actual LFW archive is retained and its SHA-256 verified.

The original FairFace and UTKFace archives are unavailable
on the project machine. Their extracted dataset versions
and provenance are documented, but an original-archive
checksum verification cannot be claimed for them.

## Duplicate-review limitations

The 4,513 strong-evidence duplicate pairs were fully
reviewed.

The weak-evidence review contains the documented
30,328-pair population and its reproducible, seed-26041
sample of 1,000 pairs. The 29,328 unsampled pairs retain
candidate status.

A further 2,755 weak-evidence pairs were marked
false_alarm in a pre-stratification bulk review. Their
journal entries are traceable. The acceptability of that
review method remains subject to the pending protocol
decision documented in AMEND-20260922-01.

Sampling evidence must not be interpreted as individual
review of the 29,328 remaining candidates.

## FairFace publisher-split overlap

The duplicate audit identified 21 exact-duplicate groups
and 72 confirmed near-duplicate groups crossing FairFace's
official publisher train/validation split.

The publisher's split was preserved rather than silently
modified. This overlap must be considered when interpreting
subsequent validation results and designing evaluation
safeguards.
