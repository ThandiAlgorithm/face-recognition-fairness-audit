# REP-006 — Missing and Unknown Label Audit

## Status

These results are provisional Phase 1 representation-audit outputs generated
from the Phase 0 master-table snapshot at commit:

`99efe8f`

They must be regenerated and revalidated against the final canonical Phase 0
hand-off.

## Scope

REP-006 audits missing and unknown demographic labels and records which
representation analyses are affected.

No source rows are deleted. Missing values and Phase 0 failure flags remain
explicitly represented.

## Label availability

### FairFace

FairFace contains complete categorical demographic labels for:

- gender: 97,698 / 97,698
- race: 97,698 / 97,698
- harmonised age label: 97,698 / 97,698

Exact numeric age is unavailable:

- numeric age: 0 / 97,698

This does not prevent the primary age-composition analyses because the native
FairFace age bands can be mapped into the frozen harmonised age bands.

### LFW

The current LFW Phase 0 hand-off contains no usable demographic labels:

- gender: 0 / 13,233
- race: 0 / 13,233
- harmonised age label: 0 / 13,233
- exact numeric age: 0 / 13,233

All 13,233 LFW images therefore have all three core demographic labels
unavailable.

These missing labels are not associated with metadata parse failures:
`parse_failed = 0` for LFW.

The absence of demographic labels therefore represents a metadata limitation
of the current hand-off rather than a parsing failure.

### UTKFace

UTKFace contains usable demographic labels for:

- gender: 24,104 / 24,108
- race: 24,104 / 24,108
- harmonised age: 24,104 / 24,108
- exact numeric age: 24,104 / 24,108

Four rows have all three core demographic labels missing.

The missing-label audit confirms that these four rows are exactly the four
rows already flagged `parse_failed`:

- any core demographic label missing: 4
- parse failures: 4
- both missing and parse failed: 4
- missing without parse failure: 0
- parse failure without missing labels: 0

## Effect on representation analyses

FairFace:

- full demographic denominator: 97,698 images
- REP-001 demographic composition fully supported
- REP-004 intersectional composition fully supported
- REP-005 base-rate analysis fully supported

LFW:

- demographic denominator: 0 images
- REP-001 demographic composition unavailable
- REP-004 intersectional composition unavailable
- REP-005 gender base-rate analysis unavailable

LFW identity structure remains available for REP-002 and REP-003 because
subject identifiers are present independently of demographic metadata.

UTKFace:

- full dataset: 24,108 images
- usable demographic denominator: 24,104 images
- four rows are excluded from demographic-analysis denominators because of
  missing labels associated with Phase 0 parsing failure

## Interpretation boundary

Missing demographic metadata should not be interpreted as demographic
membership.

No missing gender, race, or age values are inferred or imputed.

The absence of LFW demographic labels prevents demographic representation
analysis in the current hand-off but does not invalidate LFW identity-level
analysis.

## Outputs

- `rep006_label_missingness.csv`
- `rep006_missingness_patterns.csv`
- `rep006_affected_analyses.csv`
- `rep006_missingness_vs_parse_failure.csv`

## Regeneration

```powershell
python run_rep006.py `
    --master "<path-to-master_annotations.parquet>"
    