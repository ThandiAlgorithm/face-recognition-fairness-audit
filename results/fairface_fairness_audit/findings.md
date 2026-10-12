# FairFace Fairness Audit: M1 Baseline vs Mitigation

Source data in `data/` (threshold calibration and error analysis JSON for
M1's gender head). All numbers are from FairFace's locked test set
(n = 9,758), never touched during training or model selection.

Scope note: this audit only covers the brief-compliant MULTI mitigation
variants (resampling/reweighting by the combined race+gender+age+skin-tone
weight) once they finish. The earlier ITA-only (skin-tone-only) M2/M3 runs
were a pre-clarification experiment and are intentionally excluded here.
**M2-MULTI and M3-MULTI are both still training as of this write-up** --
only M1's baseline numbers are below so far.

## FairFace

### M1 baseline

#### Race group accuracy (recall)

| Group | M1 |
|---|---|
| Black | 81.0% |
| East Asian | 71.3% |
| White | 64.5% |
| Southeast Asian | 60.3% |
| Indian | 58.3% |
| Middle Eastern | 50.4% |
| Latino/Hispanic | 49.7% |

Race gap (max − min, 7 groups): **31.3 pts**. This is the largest fairness
gap found anywhere in this project, and the one M2-MULTI/M3-MULTI are
most directly relevant to, since both target race as one of their four
combined weight attributes (unlike the excluded ITA-only runs, which did
not target race at all).

#### Overall accuracy

| Task | M1 |
|---|---|
| Race | 62.6% |
| Gender | 87.4% |
| Age | 53.4% |

### Root cause 1: the gender head's decision threshold is miscalibrated

`threshold_calibration_gender.json` -- post-hoc analysis on M1's saved
gender probabilities, no retraining.

At the default threshold (0.5): male recall = 0.952, female recall =
0.786, a 16.6-point gap. **M1 is over-predicting "male"** -- sweeping the
threshold up to 0.755 brings male and female recall to within a few
points of each other (balanced recall 0.894 both). This is a genuine
fairness-improvement technique that requires no retraining, only a
different decision rule applied to an already-trained model.

### Root cause 2: who gets misclassified, by race

`error_analysis_gender.json` -- FairFace has no secondary attribute set
like LFW's (no makeup/accessory labels), so this compares misclassification
rate (true-female, predicted-male) across race groups instead.

M1's overall female→male misclassification rate: 21.4%. Per-race rate:
Indian 15.9%, Middle Eastern 18.8%, Southeast Asian 25.9% -- roughly
similar order of magnitude across groups; unlike LFW, FairFace does not
show one race group being dramatically more error-prone than others on
gender specifically. The gender miscalibration here looks like a global
effect rather than a race-concentrated one.

### Not yet done for FairFace

- M2-MULTI and M3-MULTI are both still training -- their comparison
  against this M1 baseline (overall accuracy, the 31.3-point race gap,
  threshold calibration, error analysis) will be added here once they
  finish.
- Threshold calibration and error analysis were run on the **gender**
  head only, since race (7-way) and age (9-way) are multi-class and the
  binary threshold-sweep technique does not directly apply to them in
  this form. A multi-class analogue (`multiclass_calibration.py`, already
  built) would be the natural next step for the race task once M2-MULTI/
  M3-MULTI checkpoints exist.
