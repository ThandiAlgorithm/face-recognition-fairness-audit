# REP-002 — Identity-Level Representation Audit

## Status

These results are provisional Phase 1 representation-audit outputs generated
from the Phase 0 snapshot at commit:

`99efe8f`

They must be regenerated and revalidated against the final canonical Phase 0
hand-off once it is merged into `main`.

## Scope

REP-002 evaluates identity-level representation only where reliable subject
identifiers are available.

### Identity support

- FairFace: no subject identifiers available.
- UTKFace: no subject identifiers available.
- LFW: subject identifiers available for all 13,233 images.

LFW contains:

- 5,749 unique identities
- 4,069 singleton identities
- 1,680 identities represented by more than one image
- maximum of 530 images for one identity

## Demographic identity composition

The current Phase 0 hand-off contains no demographic age, gender, or race
annotations for LFW.

Therefore, although LFW supports identity-level structural analysis,
identity-level demographic composition cannot currently be estimated.

FairFace and UTKFace do not support identity-level analysis because no
`subject_id` field is populated for those datasets.

No demographic attributes are inferred or manufactured.

## Outputs

- `rep002_identity_support.csv`
- `rep002_identity_demographic_availability.csv`
- `rep002_identity_demographic_composition.csv`

The demographic composition file contains headers only because no dataset in
the current hand-off simultaneously provides both reliable identity IDs and
usable demographic labels.

## Regeneration

Run:

```powershell
python run_rep002.py `
    --master "<path-to-master_annotations.parquet>"
    