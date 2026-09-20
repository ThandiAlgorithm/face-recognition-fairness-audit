# Governance mismatches to resolve before closure

1. GOV-005 names `configs/categories.yaml`, `configs/preprocessing.yaml`, and `configs/models.yaml`, while older governance text still names the GOV numbered YAML files. Runtime code should use the current main filenames only.

2. FairFace supplies an age band, not an exact numerical age. GOV-005 says `age_band` is derived from `age`. This needs a documented FairFace exception or an `age_band_source` field approved in GOV-005. Now more concrete after item 12 was resolved: FairFace's native age band uses its own 9-way scheme (0-2, 3-9, 10-19, ..., 70+, via `categories.FAIRFACE_AGE_MAP`), while UTKFace's numeric age is now bucketed into GOV-002's 4-way coarse `age_bands` (0-19, 20-39, 40-59, 60+). The two datasets' `age_band` column values are therefore on different scales and are not directly comparable as written. Needs a decision: bucket FairFace down to the coarse scheme too, or keep both native and add an explicit `age_band_source`/`age_band_scheme` field so downstream analysis doesn't silently compare incompatible bands.

3. Native FairFace and UTKFace labels have no confidence value. The old GOV-007 rule `gender_confidence >= 0.70` would incorrectly exclude every native label. The proposed exclusions config treats native labels as eligible without a model confidence and applies the threshold only to derived labels.

4. FairFace is publisher cropped and aligned. It should remain explicitly flagged as source preprocessed. LFW and UTKFace should use original or in the wild sources. Any canonical aligned or cropped LFW or UTKFace path should fail validation.

5. The old detector checkpoint record used threshold 0.5 while preprocessing specifies 0.90. The proposed `models.yaml` uses 0.90 to match preprocessing.

6. The old greyscale code used tolerance 3.0. Current preprocessing specifies 2.0. Counts must be regenerated.

7. The old brightness and contrast code used RGB luma. GOV-005 defines mean CIELAB L star and standard deviation of CIELAB L star. The proposed measurement code follows GOV-005.

8. GOV-005 has one `exclusion_reason` field but does not state precedence when several reasons apply. The proposed exclusions config adds a deterministic precedence. This should be approved.

9. LFW View 1 and View 2 are pair protocol concepts, not a clean unique image partition. The proposed image level master uses `split=none` for LFW and stores View 2 in `metadata/lfw_pairs.parquet`.

10. NOT YET RESOLVED. Proposed final implementation: project-native PyTorch RetinaFace (`src/retinaface/`, MobileNetV1 0.25, `mobilenet0.25_Final.pth`, no external RetinaFace package at runtime). Technical compatibility test (checkpoint strict-load, forward pass) and a one-image smoke test both PASS — see `docs/protocol_amendments/AMEND-20260919-01.md` for the full specification (architecture, checkpoint SHA256, frozen inference settings: confidence_threshold 0.90, nms_threshold 0.40, top_k 5000, keep_top_k 750, resize 1.0). An earlier intermediate step used the PyPI package `face-detection` (elliottzheng) — that step is superseded, not the frozen choice. **GOV-010 approval: PENDING.** `configs/preprocessing.yaml` and `configs/models.yaml` still describe the intermediate `elliottzheng_pytorch` state and have not been updated to `project_native` — that update happens only after approval, not before.

11. The governance files still contain unfilled freeze metadata in places. Fill the frozen date, commit, and approvals when the team formally freezes the corrected configuration.

12. RESOLVED 2026-09-18. `configs/categories.yaml` previously carried `age_bands_fine` / `age_bands_coarse` / `age_band_usage`, a scheme added on the old `gov-002-categories` branch that was never part of the joint freeze. Replaced with main's actual frozen content verbatim (single coarse `age_bands`, `frozen: true`, commit `6a5bef9`, approved by RZ and TM). `src/config.py` and `src/metadata_parsers.py` updated to read `age_bands` instead of `age_bands_fine`; the now-unused `verify_fine_in_coarse_nesting`/`fine_to_coarse_map` helpers were removed from `src/categories.py`. This surfaced item 2 above as a sharper, now-concrete problem.
