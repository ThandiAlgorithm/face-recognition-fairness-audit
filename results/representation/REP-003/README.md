# REP-003 — Images-per-Identity Distribution Audit

## Status

These results are provisional Phase 1 representation-audit outputs generated
from the Phase 0 snapshot at commit:

`99efe8f`

They must be regenerated and revalidated against the final canonical Phase 0
hand-off once Phase 0 is merged into `main`.

## Scope

REP-003 evaluates the distribution of images per identity where reliable
identity identifiers are available.

FairFace and UTKFace do not contain populated `subject_id` values in the
current Phase 0 hand-off and therefore do not support identity-depth analysis.

LFW contains subject identifiers for all 13,233 images.

## LFW identity-depth results

LFW contains:

- 5,749 unique identities
- 13,233 images with identity information
- 4,069 singleton identities
- 1,680 repeated identities
- 70.78% of identities represented by one image
- 29.22% represented by more than one image
- median images per identity: 1
- mean images per identity: 2.30
- 75th percentile: 2
- 90th percentile: 4
- 95th percentile: 6
- 99th percentile: 20
- maximum images for one identity: 530

The distribution therefore shows substantial identity-depth imbalance and a
long upper tail.

This is an identity-representation finding and should not be interpreted as a
demographic-bias result.

## Demographic subgroup analysis

The current LFW hand-off contains no usable gender, race, or age annotations.

Therefore images-per-identity distributions cannot currently be stratified by
demographic subgroup.

FairFace and UTKFace contain demographic labels but no subject identifiers, so
they also cannot support demographic identity-depth analysis.

No missing demographic attributes are inferred or manufactured.

## Outputs

- `rep003_identity_image_counts.csv`
- `rep003_identity_depth_summary.csv`
- `rep003_identity_depth_distribution.csv`
- `rep003_subgroup_analysis_status.csv`
- `rep003_subgroup_depth_summary.csv`
- `rep003_subgroup_depth_distribution.csv`

## Regeneration

Run:

```powershell
python run_rep003.py `
    --master "<path-to-master_annotations.parquet>"
    