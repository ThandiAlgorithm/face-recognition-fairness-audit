# Pre-Registration

**Source issue:** GOV-001  
**Version:** 1.1.0  
**Status:** Frozen
**Freeze date:** 17/09/2026 
**Freeze commit:** 6a5bef9ff727c793f0e710231d98118ceffaab7d

## Overall project question

How does bias in facial-recognition data and evaluation lead to unequal outcomes across demographic groups, and can targeted mitigation reduce those gaps without unacceptable loss of task performance?

## Phase 1 scope

Phase 1 is a forensic audit. It does not train mitigation models. It determines where representation, visual/acquisition, measurement, pipeline and benchmark-design differences enter the LFW, FairFace and UTKFace system, with fixed pretrained-model inference used only where required for the LFW benchmark audit.

Phase 1 asks:

1. Who and what is represented in each dataset?
2. Are groups represented under comparable visual and acquisition conditions?
3. How reliable are the demographic and skin-colour measurements used by the audit itself?
4. How different are LFW, FairFace and UTKFace as data-generating environments?
5. Does the LFW verification protocol test difficult demographic cases, or does pair design make the benchmark easier?

## Bias taxonomy

The investigation tracks historical, representation, measurement, aggregation, evaluation and deployment bias. Claims about interaction between bias types are treated as diagnostic unless the analysis design supports a stronger causal interpretation.

## Fairness criteria

For later model analysis, **equalised odds is the primary fairness criterion**. Demographic parity and equality of opportunity may be reported as secondary criteria where appropriate, but they do not replace the declared primary criterion.

For the Phase 1 LFW benchmark audit, subgroup false-match and false-non-match behaviour is reported at one fixed operating point; subgroup-specific thresholds are not used in the primary analysis.

## Hypotheses and phase mapping

### H1 — Intersectional disparity

**H0:** Under the pre-registered Phase 1 analyses, there are no supported subgroup differences in representation/pipeline measures across LFW, FairFace and UTKFace, and no supported subgroup difference in LFW verification error at the fixed operating point.

**H1:** At least one pre-registered demographic or skin-tone subgroup shows a supported disparity in representation or pipeline measures, and/or LFW verification performance differs across at least one pre-registered subgroup comparison. Intersectional results are reported only when minimum-n rules are satisfied.

**Type:** Confirmatory in Phase 1.

### H2 — Bias-layer interaction

**H0:** Phase 1 conclusions are materially unchanged when representation, measurement, visual/acquisition and benchmark-design layers are considered separately and together.

**H1:** At least one substantive audit conclusion changes when another measured layer is considered, consistent with interacting bias mechanisms. This is interpreted as diagnostic association unless the design supports a causal claim.

**Type:** Confirmatory diagnostic hypothesis in Phase 1.

### H3 — Mitigation effectiveness

**H0:** None of the four mitigation strategies — re-sampling, adversarial de-biasing, maximum mean discrepancy penalty and threshold calibration — significantly improves subgroup fairness metrics relative to the unmitigated baseline.

**H1:** At least one mitigation strategy improves a pre-declared fairness outcome, with any accuracy–fairness trade-off reported explicitly.

**Type:** Confirmatory in Phase 2; not tested in Phase 1.

### H4 — Ranking stability

**H0:** The ordering of mitigation strategies by the declared evaluation criteria is stable across datasets and criteria.

**H1:** The ordering varies across datasets and/or fairness criteria.

**Type:** Confirmatory in Phase 3; not tested in Phase 1.

## Confirmatory versus exploratory work

**Confirmatory Phase 1:** H1 and H2, using frozen category boundaries, subgroup thresholds, test families, eligibility rules and pipeline settings.

**Exploratory/frontier:** matched subgroup comparison controlling for visual covariates; dataset-source separability; category-boundary sensitivity; synthetic controlled visible-phenotype stress test. These analyses do not replace the confirmatory analysis.

## Freeze rule

This file must be committed and jointly approved before confirmatory analysis begins. Any later change follows `docs/CHANGE_CONTROL.md`.

## Sign-off

Ruvimbo Zimbodza: ___________RZ_________  Date: 17/09/2026
Thandeka Malasa: ____________TM_________  Date: 17/09/2026
