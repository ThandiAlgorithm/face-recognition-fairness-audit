# Phase 1 Scope and Freeze Timeline

**Source issue:** GOV-011  
**Version:** 1.0.0  
**Status:** Frozen  
**Approved by:** RZ and TM  
**Freeze date:** 2026-09-17  
**Freeze commit:** PENDING_SHARED_REPO_FREEZE_COMMIT

## Project Dates

- Frontier go/no-go date: 2026-10-02
- Phase 1 analysis/feature-freeze date: 2026-09-12
- Provisional final-report target: 2026-10-26

## Frontier Go/No-Go Rule

At the frontier go/no-go date, frontier analyses are released only if the
core audit is sufficiently healthy and the critical Phase 0 / Phase 1
dependencies required by those analyses are complete.

If the core checklist is materially incomplete, frontier analyses are dropped rather than extending the project scope.

Synthetic frontier work remains blocked unless the core audit is healthy and the synthetic branch is explicitly released.

## Analysis-Freeze Rule

After the Phase 1 analysis-freeze date:

- no new analyses are introduced;
- no new research questions are added;
- no new category definitions or statistical families are introduced;
- only reruns, validation, sensitivity analyses already specified by protocol,
  interpretation, result registration and report writing are permitted.

Any necessary change to a frozen method follows the change-control process in
`docs/CHANGE_CONTROL.md`.

## Kill Criteria

A frontier analysis is dropped if:

- its required core dependencies are incomplete by the go/no-go date;
- it would delay completion or validation of a core Phase 1 result;
- the required data, checkpoint, ethics approval or reproducibility evidence is
  unavailable;
- implementation cannot be completed and independently reviewed before the
  analysis-freeze date.

Core work always takes priority over frontier work.