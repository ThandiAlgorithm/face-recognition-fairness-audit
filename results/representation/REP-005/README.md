# REP-005 â€” Classification-Target Base-Rate Audit

## Status

These results are provisional Phase 1 representation-audit outputs generated
from the Phase 0 master-table snapshot at commit:

`f072db1c5906690a6ff10b711ccc5b42f8585037`

They must be regenerated and revalidated against the final canonical Phase 0
hand-off.

## Definition

REP-005 evaluates the binary classification-target base rate:

P(Y = 1 | A = g)

The positive-class coding is read from frozen GOV-002 rather than selected
after inspecting the results.

For the current audit:

- target: gender
- Y = 1: female
- Y = 0: male

This is a coding convention only. It does not imply that either class is
advantaged, preferred, or normative.

## Analysis population

Only images satisfying the Phase 0 demographic-audit eligibility rule and
having a known binary target label contribute to REP-005 denominators.

Current usable target counts:

- FairFace: 97,698
- UTKFace: 24,104
- LFW: 0

LFW therefore does not support REP-005 with the current Phase 0 hand-off.

## Overall base rates

- FairFace: 45,920 / 97,698 female = approximately 47.00%
- UTKFace: 11,522 / 24,104 female = approximately 47.80%

## Race-conditioned base rates

Race is interpreted only within each dataset. Native race categories are not
treated as directly equivalent across FairFace and UTKFace.

FairFace:

- Black: approximately 50.00%
- East Asian: approximately 49.97%
- Indian: approximately 48.23%
- Latino/Hispanic: approximately 50.33%
- Middle Eastern: approximately 31.11%
- Southeast Asian: approximately 48.02%
- White: approximately 47.22%

UTKFace:

- Asian: approximately 54.16%
- Black: approximately 48.73%
- Indian: approximately 43.26%
- Others: approximately 55.11%
- White: approximately 45.72%

## Age-conditioned base rates

Using the frozen harmonised age bands:

FairFace:

- 0-19: approximately 48.47%
- 20-39: approximately 50.58%
- 40-59: approximately 37.05%
- 60+: approximately 40.67%

UTKFace:

- 0-19: approximately 53.90%
- 20-39: approximately 53.35%
- 40-59: approximately 29.99%
- 60+: approximately 42.45%

All reported FairFace and UTKFace cells satisfy the frozen image-level
minimum-n requirement for normal reporting.

## Interpretation boundary

These values describe the prevalence of the frozen positive target class
within dataset subgroups.

They do not measure classifier accuracy, error rates, equalised odds, or
downstream recognition bias.

## Outputs

- `rep005_target_coding.csv`
- `rep005_analysis_coverage.csv`
- `rep005_overall_base_rate.csv`
- `rep005_base_rate_by_race.csv`
- `rep005_base_rate_by_age.csv`

## Regeneration

```powershell
python run_rep005.py `
    --master "<path-to-master_annotations.parquet>"
