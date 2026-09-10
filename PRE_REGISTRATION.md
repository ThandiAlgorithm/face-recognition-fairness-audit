# Pre-Registration - GOV-001

## Research Question

**Main question:** How does bias in facial recognition training data lead to unequal performance across demographic groups, and can targeted mitigation reduce this gap without a significant loss in accuracy?

**Sub-questions:**

1. Which bias types are present in the data, and how do they interact with one another?
2. How large is the performance gap across race, gender, age, and skin tone, individually and intersectionally?
3. Which mitigation strategies reduce this gap, and at what cost to accuracy?
4. Do the best-performing strategies stay consistent across datasets and fairness criteria?

## Bias Taxonomy

This investigation diagnoses and targets six of Suresh and Guttag's seven bias types: historical, representation, measurement, aggregation, evaluation, and deployment bias. Learning bias is out of scope, as mitigation in this project modifies the loss function and training data rather than the optimisation procedure itself.

## Fairness Criteria

Three fairness criteria are reported simultaneously throughout this investigation: demographic parity, equalised odds, and equality of opportunity. No single criterion is treated as sufficient on its own, since the three are mathematically incompatible in general.

## Hypotheses

### H1 - Intersectional Disparity
**H0:** There is no significant difference in facial recognition error rates across demographic and skin-tone subgroups, on LFW, FairFace, or UTKFace.
**H1:** Significant intersectional disparities exist across all three datasets: darker-skinned women show consistently higher error rates than lighter-skinned men, and the size of this gap varies systematically with each dataset's demographic composition and labelling convention, not at random.
**Type:** Confirmatory. Evaluated in Phase 1.

### H2 - Bias-Type Interaction
**H0:** The bias types identified across the three datasets are independent of one another and can be diagnosed and treated separately.
**H1:** These bias types interact rather than act independently. Representation bias amplifies aggregation bias, since a model exposed to too few examples of a group cannot learn generalisable features for it. Measurement bias in demographic labels undermines in-processing mitigations that depend on those labels. Evaluation bias conceals the true extent of aggregation bias, because aggregate accuracy over a skewed test set masks subgroup-level disaggregation.
**Type:** Confirmatory. Evaluated in Phase 1.

### H3 - Mitigation Effectiveness
**H0:** None of the four mitigation strategies - re-sampling, adversarial de-biasing, maximum mean discrepancy penalty, and threshold calibration - significantly improves subgroup fairness metrics relative to the unmitigated baseline, on any of the three datasets or three fairness criteria.
**H1:** Each strategy significantly reduces the disparity associated with the specific bias type it targets. However, no single strategy improves all three fairness criteria simultaneously across all three datasets, and the accuracy-fairness trade-off differs systematically between strategies.
**Type:** Confirmatory. Evaluated in Phase 2.

### H4 - Ranking Stability
**H0:** The ranking of mitigation strategies by fairness improvement is consistent across all three datasets and all three fairness criteria.
**H1:** Strategy rankings are unstable: a strategy most effective on one dataset may underperform on another, and a strategy that satisfies one fairness criterion may worsen another. This instability is expected to reflect genuine cross-dataset differences in demographic composition and bias profile, not measurement error.
**Type:** Confirmatory. Evaluated in Phase 3.

## Confirmatory vs. Exploratory Analyses
**Confirmatory:** H1, H2, H3, and H4, tested exactly as specified above.
**Exploratory:** matched subgroup comparison controlling for visual covariates; dataset-source separability; sensitivity of results to category-boundary choices; the synthetic controlled visible-phenotype stress test. Findings from exploratory analyses are reported as such and are not used to support or reject H1-H4.

## Sign-off
Ruvimbo Zimbodza: _______________  Date: _______________
Thandeka Malasa: _______________  Date: _______________

## Freeze Record
Freeze date: _______________
Git commit hash: _______________
