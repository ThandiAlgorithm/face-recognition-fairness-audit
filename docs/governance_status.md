# Governance Integration Status

This file maps the canonical repository files to the Phase 1 handbook issue IDs.

| Issue | Canonical file(s) | Integration status | Closure condition |
|---|---|---|---|
| GOV-001 | `docs/PRE_REGISTRATION.md` | content reconciled with Phase 1 scope | joint review, sign-off, freeze commit |
| GOV-002 | `configs/categories.yaml` | content complete | joint review, freeze metadata |
| GOV-003 | `configs/reporting.yaml` | content complete | RZ approval + reviewer, freeze metadata |
| GOV-004 | `configs/statistics.yaml` | content complete; stale H1 comment removed | RZ approval + reviewer, freeze metadata |
| GOV-005 | `docs/data_dictionary.md` | schema reconciled with current pipeline | joint review, freeze metadata |
| GOV-006 | `docs/PHASE1_PROTOCOL.md`, `configs/preprocessing.yaml` | pipeline frozen in order; skin ROI deliberately deferred to MEA-001 gate | joint review; ROI must be frozen before MEA-002, not silently invented here |
| GOV-007 | `configs/exclusions.yaml` | eligibility/reconciliation rules corrected | joint review, freeze metadata |
| GOV-008 | `docs/software_environment.md`, `configs/models.yaml`, `docs/env/requirements-lock.txt` | governance rules integrated | cannot close until environment/checkpoints are actually pinned and 20-image smoke test passes |
| GOV-009 | `metadata/results_registry.csv`, `metadata/analysis_runs/analysis_run_template.yaml` | created | RZ review and commit |
| GOV-010 | `docs/CHANGE_CONTROL.md`, `docs/protocol_amendments/TEMPLATE.md` | created | joint review and commit |


