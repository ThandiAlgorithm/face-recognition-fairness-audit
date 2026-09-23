# Governance mismatches to resolve before closure

1. GOV-005 names `configs/categories.yaml`, `configs/preprocessing.yaml`, and `configs/models.yaml`, while older governance text still names the GOV numbered YAML files. Runtime code should use the current main filenames only.

2. FairFace supplies an age band, not an exact numerical age. GOV-005 says `age_band` is derived from `age`. This needs a documented FairFace exception or an `age_band_source` field approved in GOV-005. Now more concrete after item 12 was resolved: FairFace's native age band uses its own 9-way scheme (0-2, 3-9, 10-19, ..., 70+, via `categories.FAIRFACE_AGE_MAP`), while UTKFace's numeric age is now bucketed into GOV-002's 4-way coarse `age_bands` (0-19, 20-39, 40-59, 60+). The two datasets' `age_band` column values are therefore on different scales and are not directly comparable as written. Needs a decision: bucket FairFace down to the coarse scheme too, or keep both native and add an explicit `age_band_source`/`age_band_scheme` field so downstream analysis doesn't silently compare incompatible bands.

3. Native FairFace and UTKFace labels have no confidence value. The old GOV-007 rule `gender_confidence >= 0.70` would incorrectly exclude every native label. The proposed exclusions config treats native labels as eligible without a model confidence and applies the threshold only to derived labels.

4. TECHNICALLY IMPLEMENTED. FairFace is flagged `source_preprocessed`; LFW and UTKFace use original/in-the-wild sources. `run_phase0_validation.py`'s "no aligned or cropped canonical LFW or UTKFace paths" check passes against the full 135,039-row production master with zero forbidden-path matches.

5. TECHNICALLY IMPLEMENTED. `configs/models.yaml` and `configs/preprocessing.yaml` both specify confidence threshold 0.90. Validator checks "RetinaFace confidence_threshold agrees across configs" and "RetinaFace confidence threshold is 0.90" both pass.

6. TECHNICALLY IMPLEMENTED. `configs/preprocessing.yaml` specifies `chroma_tolerance: 2.0`. The production master's greyscale counts come from the completed extraction run against this current config, not the old 3.0 tolerance -- no separate regeneration is needed.

7. TECHNICALLY IMPLEMENTED. `src/measurements.py`'s `brightness()` and `contrast()` compute the mean and standard deviation of CIELAB L\*, matching GOV-005 (verified by code inspection). The master-table rebuild test independently confirmed these columns match production exactly.

8. GOV-005 has one `exclusion_reason` field but does not state precedence when several reasons apply. The proposed exclusions config adds a deterministic precedence. This should be approved.

9. TECHNICALLY IMPLEMENTED. LFW image-level `split=none` is implemented, and View 2 is stored in `metadata/lfw_view2_pairs.parquet` (6,000 pairs, validated against the official protocol file's SHA-256; see `run_validate_lfw_protocol.py`). The frozen `docs/data_dictionary.md` still names the file `lfw_pairs.parquet` -- tracked as a documentation erratum pending approval to update the frozen dictionary (see item 13).

10. TECHNICALLY IMPLEMENTED; GOVERNANCE STATUS PENDING.

The completed pipeline uses project-native PyTorch RetinaFace,
MobileNetV1 0.25, and mobilenet0.25_Final.pth.

Both active configurations specify project_native and agree
on confidence threshold 0.90, NMS threshold 0.40,
top_k 5000 and keep_top_k 750.

The checkpoint SHA-256, file size, strict model loading,
forward pass and full production extraction have passed
their technical checks.

The previous TensorFlow and PyTorch-wrapper implementations
are superseded and retained in the historical record.

AMEND-20260919-01 remains the relevant protocol record.
Technical implementation must not be mistaken for any
governance approval that has not actually been recorded.

11. The governance files still contain unfilled freeze metadata in places. Fill the frozen date, commit, and approvals when the team formally freezes the corrected configuration.

12. RESOLVED 2026-09-18. `configs/categories.yaml` previously carried `age_bands_fine` / `age_bands_coarse` / `age_band_usage`, a scheme added on the old `gov-002-categories` branch that was never part of the joint freeze. Replaced with main's actual frozen content verbatim (single coarse `age_bands`, `frozen: true`, commit `6a5bef9`, approved by RZ and TM). `src/config.py` and `src/metadata_parsers.py` updated to read `age_bands` instead of `age_bands_fine`; the now-unused `verify_fine_in_coarse_nesting`/`fine_to_coarse_map` helpers were removed from `src/categories.py`. This surfaced item 2 above as a sharper, now-concrete problem.

13. PENDING GOVERNANCE DECISION. `docs/data_dictionary.md` is frozen (v1.1.0, 2026-09-17, approved by RZ and TM) and is not silently redefined here. Two discrepancies between that frozen text and the completed Phase 0 implementation are recorded below as a proposed clarification, pending team approval to update the frozen dictionary itself.

### Phase 0 implementation clarification — pending governance

The completed Phase 0 master contains only LFW, FairFace
and UTKFace. CelebA is not a member of this canonical
Phase 0 image population.

The official LFW View 2 pair artifact is:
metadata/lfw_view2_pairs.parquet

Age-band categories retain their dataset-specific meaning.
FairFace uses its publisher-provided native age bands.
UTKFace uses age bands derived from its numeric ages
according to the configured coarse scheme.

These values are not directly interchangeable across
datasets. Any common age-band mapping for later analysis
requires an explicit, documented protocol decision.

No new or recalculated `age_band` column has been added to
solve this wording problem. The frozen dictionary's text
remains unchanged until this is formally resolved.
