# Master Data Dictionary

**Source issue:** GOV-005  
**Version:** 1.1.0  
**Status:** Frozen
**Freeze date:** 2026-09-17  
**Freeze commit:** PENDING_SHARED_REPO_FREEZE_COMMIT  
**Approved by:** RZ and TM  
**Approval date:** 2026-09-17

No column may be added to a master table without first being added here.

## Files

| File | One row is | Written by |
|---|---|---|
| `metadata/master_annotations.parquet` | one raw dataset image | Phase 0 to Layer 3 |
| `metadata/lfw_pairs.parquet` | one LFW protocol pair | Layer 5 |
| `metadata/file_manifest.parquet` | one raw file as downloaded | Phase 0 |
| `metadata/results_registry.csv` | one accepted reported result | throughout |

## Conventions

- `snake_case` throughout.
- **Unknown is not false.** Missing categorical values use the literal
  `unknown`; missing numerics use null. A boolean is never used to encode
  "we do not know".
- Every derived demographic value has a paired `*_source` field, and a
  `*_confidence` field where the producing method supplies one.
- Units are recorded here, never guessed from the column name.
- Every derived category must be reproducible from a raw field plus a config
  value. No category is stored without its underlying continuous value.
- Rows are never deleted. Failure is stored as data.

## Stable image identifier

```
image_id = "<DATASET>-" + sha256(dataset_version + "|" + original_path)[:16]
```

The identifier must not depend on row order, on processing success, or on
image content. Two distinct files with identical bytes receive distinct
identifiers, and are linked instead through `exact_dup_group`.

---


## file_manifest

| Field | Type | Meaning |
|---|---|---|
| `dataset` | category | lfw, fairface, utkface, celeba |
| `dataset_version` | string | acquisition-manifest identifier |
| `original_path` | string | relative path inside the extracted raw dataset |
| `filename` | string | original basename |
| `extension` | string | original suffix |
| `file_size_bytes` | int | byte size before processing |
| `file_sha256` | string | cryptographic hash of the raw file bytes |
| `archive_sha256` | string, null | source archive hash where an archive exists |
| `width_px` | int, null | decoded width before conversion |
| `height_px` | int, null | decoded height before conversion |
| `original_mode` | string, null | Pillow/original decoded mode |
| `image_format` | string, null | decoded file format |
| `decode_ok` | bool | whether the image decoded successfully |
| `decode_error` | string, null | exception/error text when decode fails |

The manifest contains every raw file. Image-level master rows are created from the manifest without deleting failures.

---

## master_annotations

### Identity and provenance

| Field | Type | Units / values | Stage | Rule |
|---|---|---|---|---|
| `image_id` | string | stable id | Phase 0 | never reused, never regenerated differently |
| `dataset` | category | lfw, fairface, utkface, celeba | Phase 0 | fixed vocabulary |
| `dataset_version` | string | manifest id | Phase 0 | required |
| `original_path` | string | relative path | Phase 0 | raw path only |
| `file_size_bytes` | int | bytes | Phase 0 | from raw file before processing |
| `file_sha256` | string | SHA-256 hex | Phase 0 | byte-level file hash; used for exact duplicates |
| `image_format` | string | JPEG, PNG, etc. | integrity | recorded before conversion |
| `split` | category | train, val, test, view1, view2, none | Phase 0 | locked before any development analysis |
| `subject_id` | string, null | identity label | Phase 0 | null where the dataset has none. Never invented |

### Integrity

| Field | Type | Units / values | Stage | Rule |
|---|---|---|---|---|
| `width_px` | int | pixels | integrity | before any resize |
| `height_px` | int | pixels | integrity | before any resize |
| `original_mode` | string | RGB, L, RGBA, P, CMYK | integrity | recorded before conversion |
| `colour_profile` | string, null | ICC profile name | integrity | null if absent |
| `colour_profile_assumed` | bool | true, false | integrity | true when sRGB is assumed because no profile was present |
| `is_greyscale` | bool | true, false | integrity | true when `original_mode` is L, or when a\* and b\* are both within the null-chroma tolerance |
| `corrupt` | bool | true, false | integrity | row is retained |
| `parse_failed` | bool | true, false | labels | row is retained |

### Demographics

| Field | Type | Units / values | Stage | Rule |
|---|---|---|---|---|
| `gender` | category, null | female, male, unknown | labels | `gender_source` required whenever non-null |
| `gender_source` | string | native, derived:\<method\>, unavailable | labels | required |
| `gender_confidence` | float, null | 0 to 1 | labels | null when the source provides none |
| `race` | category, null | dataset-specific vocabulary | labels | never mapped across datasets |
| `race_source` | string | native, derived:\<method\>, unavailable | labels | required |
| `race_confidence` | float, null | 0 to 1 | labels | null when unavailable |
| `age` | float, null | years | labels | raw value, never the band |
| `age_source` | string | native, derived:\<method\>, unavailable | labels | required |
| `age_confidence` | float, null | 0 to 1 | labels | null when unavailable |
| `age_band` | category, null | from `configs/categories.yaml` | analysis | derived, never stored instead of `age` |

### Duplicates

| Field | Type | Units / values | Stage | Rule |
|---|---|---|---|---|
| `exact_dup_group` | string, null | group id | duplicates | shared by all byte-identical files |
| `near_dup_group` | string, null | group id | duplicates | candidate grouping, requires review |
| `near_dup_status` | category | none, candidate, confirmed, false_alarm, uncertain | duplicates | never auto-deleted |
| `cross_dataset_dup` | bool | true, false | duplicates | confirmed overlap only |
| `cross_dataset_dup_status` | category | none, candidate, confirmed, false_alarm | duplicates | review outcome |

### Detection and alignment

| Field | Type | Units / values | Stage | Rule |
|---|---|---|---|---|
| `bbox` | array, null | x1, y1, x2, y2 in pixels of the EXIF-oriented image | detection | coordinate origin is top-left |
| `det_confidence` | float, null | 0 to 1 | detection | checkpoint recorded in `configs/models.yaml` |
| `n_detections` | int | count | detection | 0 is a valid value and is retained |
| `no_detection` | bool | true, false | detection | a finding, not a software error |
| `multi_detection` | bool | true, false | detection | all detections stored in `detections_all` |
| `detections_all` | array, null | list of bbox and score | detection | written before any primary face is chosen |
| `landmarks` | array, null | five points, pixels | detection | same coordinate origin as bbox |
| `alignment_transform` | array, null | 2x3 similarity matrix | alignment | must reproduce the saved crop exactly |
| `aligned_path` | string, null | derived path | alignment | image files are never committed to git |
| `alignment_valid` | bool | true, false | alignment | `exclusion_reason` required when false |

### Visual measures

| Field | Type | Units | Stage | Rule |
|---|---|---|---|---|
| `brightness` | float, null | mean luminance, 0 to 100 (CIELAB L\*) | visual | computed on the face region, not the whole image |
| `contrast` | float, null | standard deviation of L\* | visual | same region as brightness |
| `sharpness` | float, null | variance of Laplacian | visual | higher is sharper. Direction stated because it is not obvious |
| `face_px_area` | float, null | pixels squared | visual | from the raw bbox |
| `face_area_ratio` | float, null | 0 to 1 | visual | bbox area divided by image area |
| `yaw`, `pitch`, `roll` | float, null | degrees | visual | method fixed in `configs/preprocessing.yaml`. Null in Phase 1 if pose is deferred |

### Skin measures

| Field | Type | Units | Stage | Rule |
|---|---|---|---|---|
| `L_star` | float, null | 0 to 100 | measurement | computed pre-augmentation on the aligned crop |
| `a_star` | float, null | CIELAB | measurement | pre-augmentation |
| `b_star` | float, null | CIELAB | measurement | pre-augmentation |
| `ITA` | float, null | degrees | measurement | derived from stored L\* and b\*, never recomputed from the image |
| `hue_angle` | float, null | degrees, 0 to 360 | measurement | `atan2(b_star, a_star)` converted to degrees and wrapped to [0, 360) |
| `ita_group` | category, null | six ITA groups | analysis | derived from config thresholds |
| `ita_ordinal` | int, null | 1 to 6 | analysis | ordinal rank for agreement analysis |
| `fitz_classifier` | category, null | I to VI | measurement | proxy output, not ground truth |
| `fitz_classifier_conf` | float, null | 0 to 1 | measurement | null when unavailable |
| `fitz_ordinal` | int, null | 1 to 6 | analysis | ordinal rank for agreement analysis |
| `skin_group_ita` | category, null | lighter, darker | analysis | ITA > 28 is lighter |
| `skin_group_fitz` | category, null | lighter, darker | analysis | I to III is lighter |
| `skin_roi_valid` | bool | true, false | measurement | `exclusion_reason` required when false |
| `ita_undefined` | bool | true, false | measurement | true when b\* is within the guard tolerance of zero, which makes ITA unstable |

### Soft attributes

| Field | Type | Units | Stage | Rule |
|---|---|---|---|---|
| `attr_<name>_prob` | float, null | 0 to 1 | soft attributes | probability retained. Never thresholded in storage |

`Pale_Skin` is stored for completeness and is **excluded from any use as a
skin tone measure**, because manipulating it changes lightness and hue
together.

### Audit flags

| Field | Type | Units / values | Stage | Rule |
|---|---|---|---|---|
| `low_confidence` | bool | true, false | throughout | threshold from config |
| `thin_cell` | bool | true, false | analysis | applied per GOV-003 |
| `exclusion_reason` | string, null | primary controlled reason | throughout | retained for concise reporting; detailed boolean flags remain authoritative |
| `eligible_detection_audit` | bool | true, false | derived | rule in `configs/exclusions.yaml` |
| `eligible_visual_audit` | bool | true, false | derived | rule in `configs/exclusions.yaml` |
| `eligible_skin_audit` | bool | true, false | derived | rule in `configs/exclusions.yaml` |
| `eligible_demographic_audit` | bool | true, false | derived | analysis-specific rule in `configs/exclusions.yaml` |
| `eligible_cross_dataset_eval` | bool | true, false | derived | rule in `configs/exclusions.yaml` |
| `eligible_lfw_pair_eval` | bool | true, false | derived | rule in `configs/exclusions.yaml` |

### Controlled vocabulary for `exclusion_reason`

```
corrupt_file
decode_failure
label_parse_failure
missing_required_label
low_confidence_label
zero_detection
multi_detection_unresolved
alignment_failure
invalid_skin_roi
greyscale_no_chroma
ita_undefined
exact_duplicate
near_duplicate_confirmed
cross_dataset_duplicate_confirmed
thin_cell
```

---

## lfw_pairs

| Field | Type | Units / values | Rule |
|---|---|---|---|
| `pair_id` | string | stable id | never reused |
| `fold` | int | 1 to 10 | from the official protocol |
| `view` | category | view1, view2 | view2 sealed until final reporting |
| `is_genuine` | bool | true, false | true is a mated pair |
| `identity_1`, `identity_2` | string | identity label | from the official list |
| `image_id_1`, `image_id_2` | string | master table key | must resolve |
| `gender_1`, `gender_2` | category | female, male, unknown | derived, source in master table |
| `same_gender` | bool, null | true, false, null | null if either member is unknown; never convert unknown to false |
| `skin_group_1`, `skin_group_2` | category | lighter, darker, unknown | from `skin_group_ita` |
| `same_skin_group` | bool, null | true, false, null | null if either member is unknown |
| `age_band_1`, `age_band_2` | category | age band or unknown | as above |
| `same_age_band` | bool, null | true, false, null | null if either member is unknown |
| `pair_demog_complete` | bool | true, false | true only when both members have all three attributes known |
| `yoked` | category | fully_yoked, partially_yoked, cross_demographic, unknown | fully yoked means same gender, skin group and age band |
| `brightness_difference` | float | absolute difference in L\* | both members must be eligible |
| `pose_difference` | float, null | degrees | null in Phase 1 if pose is deferred |
| `model_name` | string | checkpoint id from `configs/models.yaml` | never a bare name such as "ArcFace" |
| `model_score` | float, null | similarity | scale documented per model |
| `threshold` | float, null | decision threshold | single value across all subgroups |
| `decision` | category | match, non_match, null | null until scoring is run |

---

## What this dictionary cannot prove

A clear column name does not make the underlying label correct. The dictionary
makes each value's source and meaning visible so that it can be audited.
