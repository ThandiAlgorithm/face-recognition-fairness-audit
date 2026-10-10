# Candidate Balancing Strategies

These scenarios are exploratory only.

No image has been deleted, duplicated, copied or relabelled.

## Scenario 1 — Current pool

Uses all currently mapped demographic records.

Advantage:
- maximum volume.

Disadvantage:
- White representation remains dominant.

## Scenario 2 — Strict parity across all trainable groups

Every demographic category is reduced to the size of the
smallest category.

Advantage:
- exact numerical parity.

Disadvantage:
- potentially severe loss of training volume.
- does not guarantee diversity or better model performance.

## Scenario 3 — Strict parity across explicit fine-grained groups

Uses:
- White
- Black
- Indian
- East Asian
- Southeast Asian
- Middle Eastern
- Latino

Broad Asian, Other and Unknown records remain preserved in
the full dataset but are not used in this candidate subset.

Advantage:
- clean source-supported taxonomy.
- no fabricated demographic refinement.

Disadvantage:
- excludes useful images from the candidate training subset.

## Scenario 4 — Volume-aware capped dataset

Large groups are capped at the median explicit-group size.
Smaller groups retain every available unique image.

Advantage:
- reduces majority dominance.
- preserves substantially more data than strict parity.
- does not duplicate minority examples.

Disadvantage:
- demographic counts are not exactly equal.

## Next stage

Whichever scenario is selected must then be constrained by:

- gender
- ITA / measured skin tone
- age
- source dataset
- brightness
- contrast
- sharpness
- face size
- image quality
- duplicate status
- identity concentration where identity is known

Numerical balance alone is not treated as sufficient evidence
of dataset fairness.
