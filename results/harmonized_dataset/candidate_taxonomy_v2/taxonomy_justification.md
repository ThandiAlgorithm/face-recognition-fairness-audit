# Candidate Harmonised Taxonomy v2

## Candidate labels

- White
- Black
- Indian
- East Asian
- Southeast Asian
- Middle Eastern
- Latino
- Asian
- Other
- Unknown

## Dataset size

Total pooled source images: 135,039

## Mapping principles

1. Original source labels are preserved.
2. No demographic label is assigned from facial appearance.
3. FairFace East Asian remains East Asian.
4. FairFace Southeast Asian remains Southeast Asian.
5. UTKFace Asian remains Asian because its source metadata is broader.
6. LFW derived Asian remains Asian because the derived metadata is broader.
7. FairFace Indian remains Indian.
8. FairFace Middle Eastern remains Middle Eastern.
9. FairFace Latino_Hispanic is harmonised to Latino.
10. UTKFace Others remains Other.
11. Missing or unavailable labels remain Unknown.

## Design rationale

The taxonomy prioritises traceability over forced completeness.

A more specific demographic category is only used when supported by the
source annotation. Broad labels are not visually subdivided.

The final dataset will later be evaluated not only for demographic counts,
but also for:

- gender composition,
- ITA / skin-tone composition,
- age coverage,
- source diversity,
- image quality,
- brightness,
- contrast,
- sharpness,
- face size,
- identity concentration where identity labels exist,
- and downstream model performance.

This version is still a candidate taxonomy and does not modify any source
image or source annotation.
