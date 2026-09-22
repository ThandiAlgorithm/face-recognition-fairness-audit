# REP-004 — Intersectional Composition Audit

## Status

These results are provisional Phase 1 representation-audit outputs generated
from the Phase 0 master-table snapshot at commit:

`99efe8f`

They must be regenerated and revalidated against the final canonical Phase 0
hand-off.

## Analysis population

Intersectional composition uses images satisfying:

`eligible_demographic_audit == True`

Current eligible counts:

- FairFace: 97,698 / 97,698
- UTKFace: 24,104 / 24,108
- LFW: 0 / 13,233

LFW therefore does not support the current demographic intersection analysis.

## Pre-registered intersections

- gender × race
- gender × harmonised age
- race × harmonised age
- gender × race × harmonised age

The harmonised age categories are taken from frozen GOV-002:

- 0-19
- 20-39
- 40-59
- 60+

Native race categories remain dataset-specific and are not pooled between
FairFace and UTKFace.

## Minimum cell-size policy

Frozen GOV-003 image-level reporting rules are applied:

- n < 30: suppressed / thin cell
- 30 <= n < 100: low confidence
- n >= 100: reportable

Small cells remain listed. Raw percentages are retained for reproducibility,
while `percentage_reported` is blank for suppressed cells.

## Current findings

All tested FairFace intersection cells contain at least 100 eligible images.

For UTKFace:

- gender × age: all cells reportable
- gender × race: all cells reportable
- race × age: 19 reportable cells and 1 suppressed cell
- gender × race × age: 34 reportable, 4 low-confidence and 2 suppressed cells

These results describe dataset representation. They do not by themselves
establish downstream model-performance bias.

## Regeneration

```powershell
python run_rep004.py `
    --master "<path-to-master_annotations.parquet>"
    