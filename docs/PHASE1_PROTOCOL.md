# Phase 1 Preprocessing Protocol

**Source issue:** GOV-006  
**Version:** 1.1.0  
**Status:** Frozen
**Freeze date:** 2026-09-17 
**Freeze commit:** PENDING_SHARED_REPO_FREEZE_COMMIT

Numerical settings are in `configs/preprocessing.yaml`. This document
fixes the order.

Rerunning the pipeline on the same raw input with the same config must produce
the same output table.

## Stage order

| # | Stage | Input | Output | Must precede |
|---|---|---|---|---|
| 1 | Read raw archive | downloaded archive | read-only tree | all |
| 2 | Build file manifest | raw tree | `metadata/file_manifest.parquet` | 3 |
| 3 | Decode and integrity check | raw files | dimensions, mode, format, corrupt flag | 4 |
| 4 | Apply EXIF orientation | decoded image | oriented image in memory | detection |
| 5 | Colour handling | oriented image | sRGB image, profile fields, greyscale flag | 15 |
| 6 | Parse native labels | label files | demographics plus provenance | 9 |
| 7 | Duplicate checks | manifest and images | exact, near, cross-dataset groups | 9 |
| 8 | Create stable image_id | manifest fields | `image_id` | 9 |
| 9 | Write master row | stages 3 to 8 | `metadata/master_annotations.parquet` | 10 |
| 10 | Lock splits | master table | split field, sealed View 2 | any development analysis |
| 11 | Run detection | oriented image | detections, confidence, landmarks | 12 |
| 12 | Resolve primary face | all detections | selected bbox and landmarks | 13 |
| 13 | Align | oriented image, landmarks | aligned crop, transform | 14, 15 |
| 14 | Visual measures | aligned crop, bbox | brightness, contrast, sharpness, size | 18 |
| 15 | Skin colour | aligned crop, pre-augmentation | L\*, a\*, b\*, ITA, hue | 18 |
| 16 | Soft attributes | aligned crop | attribute probabilities | 18 |
| 17 | Build derived tables | master table | identity table, LFW pair table | 18 |
| 18 | Statistical audit | all tables | registered results | report |

## The four ordering constraints that matter

EXIF orientation before detection. A sideways face is a detection failure with
no demographic cause, and it would contaminate the Layer 0 result.

Split locking before any confirmatory or model-development analysis. LFW View 2 is sealed. Exploratory Phase 0 QA may inspect dataset-wide structure, integrity and provenance, but any result intended for the formal audit is rerun from the locked master table. View 2 is never opened for development.

Alignment before any aligned-crop measurement. Visual measures and skin colour
are both defined on the aligned crop.

Skin colour before any photometric augmentation. Brightness, contrast and
colour jitter move ITA directly. Reversing this order corrupts H1 silently,
with no error raised.

## Formulas

ITA, from stored CIELAB values:

```
ITA = arctan((L_star - 50) / b_star) * 180 / pi
```

Guard: when `|b_star| < 1.0`, ITA is null with `ita_undefined = true`. The
quotient is unstable near zero and flips sign across the discontinuity.

Hue angle, from stored CIELAB values:

```
hue_angle = degrees(atan2(b_star, a_star)) mod 360
```

`atan2` preserves the quadrant, which `arctan(b/a)` does not. For skin, values
should fall roughly in 0 to 90 degrees. Values outside are flagged.

Both are derived from the stored L\*, a\*, b\* values, never recomputed from
the image, so the two remain consistent with each other.

Colour conversion: sRGB input, D65 illuminant, 2 degree observer, using
`skimage.color.rgb2lab` at the version pinned in
`configs/models.yaml`. The conversion is not unique and ITA depends on
which convention is used.

## Augmentation

No photometric augmentation is applied to LFW, FairFace or UTKFace before
measurement. Augmentation used to train the CelebA attribute classifier belongs
to that training pipeline alone.

## Output rule

No source record is deleted from the master table. Failure is stored as data,
with a machine-readable `exclusion_reason` and the relevant eligibility masks
set to false. Exclusion rules are in `configs/exclusions.yaml`.

## What this protocol cannot prove

A fixed pipeline is not automatically a fair pipeline. It makes the pipeline
repeatable, which is what allows Layer 0 to test whether it fails differently
across groups.
