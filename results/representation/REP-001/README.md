# REP-001 — Image-Level Representation Audit

## Status

These results are provisional Phase 1 representation-audit outputs.

They were generated before the Phase 0 preprocessing branch was merged into
`main`. They must therefore be regenerated and revalidated against the final
canonical Phase 0 hand-off before being treated as final report results.

## Input provenance

Phase 0 source branch:

`origin/data-exploring`

Phase 0 source commit:

`99efe8f`

Source commit message:

`Complete DATA-009 stable IDs and master annotation table`

Input master table snapshot:

`master_annotations_99efe8f.parquet`

Master-table dimensions:

- Rows: 135,039
- Unique image IDs: 135,039
- FairFace: 97,698
- UTKFace: 24,108
- LFW: 13,233

## Governance

Age harmonisation uses the frozen GOV-002 configuration:

`configs/categories.yaml`

Frozen age bands:

- 0-19
- 20-39
- 40-59
- 60+

Native/source demographic categories are retained separately from harmonised
reporting categories.

No source rows are deleted for REP-001 composition counts.

## Regeneration

Run:

```powershell
python run_rep001.py `
    --master "<path-to-master_annotations.parquet>"
    