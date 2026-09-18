# Method Change Control

**Source issue:** GOV-010  
**Version:** 1.0.0  
**Status:** Frozen

## Rule

Methods may change when implementation reveals a real problem, but changes are never silent. Every substantive change to a frozen method, threshold, category boundary, test family, eligibility rule, split, model/checkpoint or preprocessing choice requires a protocol amendment in `docs/protocol_amendments/`.

Each amendment records:

- the original method;
- the proposed change;
- the reason for the change;
- the evidence that prompted it;
- whether the change was made before or after seeing confirmatory results;
- the effect on confirmatory analyses;
- affected configs, scripts, outputs and report sections;
- approver(s), date and git commit;
- required reruns and sensitivity analyses.

## Result-driven changes

If a change is prompted by observed confirmatory results, the original analysis remains the primary analysis. The changed method is reported as a sensitivity analysis unless the supervisor approves a formal protocol amendment that explicitly changes the primary specification.

## Approval

Changes to items owned by both researchers require RZ and TM approval. Any result-driven change to a confirmatory specification additionally requires supervisor approval before the primary specification is replaced.

## File naming

Use `AMEND-YYYYMMDD-short-description.md` and never overwrite an earlier amendment.
