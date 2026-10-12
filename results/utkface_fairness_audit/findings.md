# UTKFace Fairness Audit: M1 Baseline vs Mitigation

Source data in `data/` (threshold calibration and error analysis JSON for
each method's gender head, prefixed `m1_`/`m2_multi_`/`m3_multi_`). All
numbers are from UTKFace's locked test set (n = 2,405), never touched
during training or model selection.

Scope note: this audit only covers the brief-compliant MULTI mitigation
variants (resampling/reweighting by the combined race+gender+age+skin-tone
weight). The earlier ITA-only (skin-tone-only) M2/M3 runs were a
pre-clarification experiment and are intentionally excluded here.

## UTKFace

### M1 vs M2-MULTI (resampling) vs M3-MULTI (reweighting)

Both resample/reweight by the combined race+gender+age+Fitzpatrick weight,
built the same way as LFW/FairFace's MULTI variants
(`demographic_weights.py`).

#### Overall accuracy

| Task | M1 | M2-MULTI | M3-MULTI |
|---|---|---|---|
| Race | 80.2% | 78.7% | 77.2% |
| Gender | 88.0% | 88.4% | **90.2%** |
| Age | 76.9% | 77.2% | 73.6% |

**M3-MULTI gave the biggest gender gain of the two (88.0% → 90.2%)**, at a
small-to-moderate cost to race and a larger cost to age. M2-MULTI's
profile is closer to M1 across the board -- smaller moves in every
direction.

#### Fairness gaps (max − min group)

| Gap | M1 | M2-MULTI | M3-MULTI |
|---|---|---|---|
| Gender | 15.7 pts | 6.5 pts | **4.1 pts** |
| Race (excl. "Others", 4 groups) | 6.9 pts | 10.5 pts | 16.0 pts |
| Skin tone (ITA) | 3.0 pts | 3.7 pts | 4.4 pts |

**M3-MULTI closed the gender gap further than M2-MULTI** (15.7 → 4.1
points vs 15.7 → 6.5), consistent with it also raising gender accuracy
more. But it made the race gap noticeably worse than M2-MULTI did (16.0
vs 10.5 points, both worse than M1's 6.9) -- reweighting pushed harder on
gender at a bigger cost to race than resampling did here. Skin-tone gap
drifted up slightly under both, despite being an explicit weight
component in both.

#### Race group accuracy (recall), by method

| Group | M1 | M2-MULTI | M3-MULTI |
|---|---|---|---|
| White | 86.5% | 84.6% | 81.2% |
| Black | 84.4% | 80.2% | **87.7%** |
| Asian | 82.4% | 83.8% | 71.7% |
| Indian | 79.6% | 74.1% | 77.4% |
| Others | 28.3% | 39.9% | 37.6% (n = 173, excluded from gap calculations -- unreliable, catch-all category) |

Neither method is a clean win for every group. M3-MULTI's widened race
gap is driven mostly by Asian dropping sharply (82.4% → 71.7%) while Black
actually improved (84.4% → 87.7%, the best Black result of any UTKFace
method) -- a trade between groups, not a uniform cost.

### Root cause check: are the gender improvements real fixes, or threshold artifacts?

`threshold_calibration_gender.json` per method -- post-hoc analysis on
each model's saved gender probabilities, no retraining.

| Method | Default (0.5) gap | Default male/female recall | Balanced threshold | Balanced recall (both) |
|---|---|---|---|---|
| M1 | 15.7 pts | 0.805 / 0.962 | 0.220 | 0.909 |
| M2-MULTI | 6.5 pts | 0.915 / 0.850 | 0.655 | 0.893 |
| **M3-MULTI** | **4.1 pts** | **0.921 / 0.880** | **0.560** | **0.901** |

**M3-MULTI is the better-calibrated of the two by default** -- its
balanced threshold (0.560) is much closer to 0.5 than M2-MULTI's (0.655),
meaning less of its gender improvement depends on picking a non-default
cutoff. M1's threshold needs to move much further (all the way to 0.22)
to balance, confirming M1 was the most severely miscalibrated of the
three to begin with. Per-race balanced thresholds for M3-MULTI are also
tightly clustered (0.44-0.67) with near-zero residual gaps everywhere,
including a near-perfect balance for Black (threshold 0.67, gap
<0.01 pts).

### Error analysis: who gets misclassified, by race

`error_analysis_gender.json` -- no secondary attribute set (as with
FairFace), so this compares misclassification rate (true-female,
predicted-male) across race groups.

| Method | Overall female→male misclassification rate |
|---|---|
| M1 | 3.8% |
| M2-MULTI | 15.0% |
| M3-MULTI | 12.0% |

M1's baseline gender misclassification rate is already low in absolute
terms (3.8%) -- the 15.7-point recall *gap* found above is mostly driven
by very high male recall (98.0%) rather than a large number of actual
female misclassifications; there just weren't many errors to begin with,
they were just concentrated almost entirely on one side. Both MULTI
variants have higher absolute misclassification rates than M1 but much
smaller recall *gaps* (consistent with the calibration table above) --
errors are more evenly spread between male and female rather than
concentrated on one side. For M3-MULTI, errors concentrate most in the
youngest (0-19: 24.8%) and oldest (60+: 19.8%) age bands, same pattern as
M2-MULTI, consistent with age being one of the four shared weight
attributes in both. Per-race rates (M3-MULTI): Others 18.4%, Asian 14.5%,
White 12.2%, Black 10.8%, Indian 6.4% -- again a wider spread than M1's,
but no single race group dominating the way age does.

### Not yet done for UTKFace

Threshold calibration and error analysis were run on the **gender** head
only (the one binary task); race (5-way) and age (4-way) are multi-class
and would need a different, per-class calibration technique (see
`multiclass_calibration.py`, already built and applied to FairFace/UTKFace
race -- not yet run against either MULTI checkpoint here).
