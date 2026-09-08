 # Pre-Registration

## Investigation Question
How do the multiple, interacting bias types present in facial recognition datasets propagate into intersectional performance disparities across race, gender, age, and skin tone, and to what extent can mitigation strategies matched to specific bias types reduce these gaps without unacceptable cost to overall accuracy, when measured simultaneously under multiple fairness criteria and validated across three datasets — LFW, FairFace, and UTKFace — with different demographic compositions and labelling methodologies?

## Bias Taxonomy
This investigation diagnoses and targets six of Suresh and Guttag's seven bias types: historical, representation, measurement, aggregation, evaluation, and deployment bias. Learning bias is out of scope, as mitigation in this project modifies the loss function and training data rather than the optimisation procedure itself.

## Fairness Criteria
Three fairness criteria are reported simultaneously throughout this investigation: demographic parity, equalised odds, and equality of opportunity. No single criterion is treated as sufficient on its own, since the three are mathematically incompatible in general.

## Hypotheses

### H1 — Intersectional Disparity
**H0:** There is no significant difference in facial recognition error rates across demographic and skin-tone subgroups, on LFW, FairFace, or UTKFace.
**H1:** Significant intersectional disparities exist across all three datasets, with darker-skinned women consistently showing measurably higher error rates than lighter-skinned men, and the magnitude of this disparity varies across datasets in a pattern that reflects each dataset's own demographic composition and labelling convention rather than random variation.
**Type:** Confirmatory. Evaluated in Phase 1.

### H2 — Bias-Type Interaction
**H0:** The bias types identified across the three datasets are independent of one another and can be diagnosed and treated separately.
**H1:** These bias types interact and compound one another: representation bias amplifies aggregation bias, because a model that has seen too few examples of a group cannot learn features that generalise to that group; measurement bias in demographic labels undermines in-processing mitigation strategies that depend on that signal; and evaluation bias conceals the true degree of aggregation bias by reporting aggregate accuracy across a skewed test set rather than disaggregating by subgroup.
**Type:** Confirmatory. Evaluated in Phase 1.

### H3 — Mitigation Effectiveness
**H0:** None of the four mitigation strategies (re-sampling, adversarial de-biasing, maximum mean discrepancy penalty, threshold calibration) produce a statistically significant improvement in subgroup fairness metrics relative to the unmitigated baseline, across all three datasets and all three fairness criteria.
**H1:** Each strategy produces a significant reduction in the disparity associated with the specific bias type it targets, but no single strategy improves all three fairness criteria simultaneously across all three datasets, and the trade-off between accuracy and fairness differs systematically across strategies.
**Type:** Confirmatory. Evaluated in Phase 2.

### H4 — Ranking Stability
**H0:** The ranking of mitigation strategies by fairness improvement is consistent across all three datasets and all three fairness criteria.
**H1:** Strategy rankings are unstable: a strategy that appears most effective on one dataset may underperform on another, and a strategy that satisfies one fairness criterion may worsen another. This instability is expected to reflect genuine differences in the demographic composition and bias profile of each dataset rather than measurement error.
**Type:** Confirmatory. Evaluated in Phase 3.

## Confirmatory vs. Exploratory Analyses
**Confirmatory:** H1, H2, H3, and H4, tested exactly as specified above.
**Exploratory:** matched subgroup comparison controlling for visual covariates; dataset-source separability; sensitivity of results to category-boundary choices; the synthetic controlled visible-phenotype stress test. Findings from exploratory analyses are reported as such and are not used to support or reject H1–H4.

## Sign-off
Ruvimbo Zimbodza: _______________  Date: _______________
Thandeka Malasa: _______________  Date: _______________

## Freeze Record
Freeze date: _______________
Git commit hash: _______________