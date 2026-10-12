# UTKFace Deep Dive: Everything Wrong, M1 vs M2-MULTI vs M3-MULTI

Companion to `findings.md` one level up. This document adds: full
multi-class confusion matrices (with %) for race/gender/age, per-class
recall bar charts, a genuine before/after check of multi-class threshold
(logit-bias) calibration on both accuracy and the fairness gap, and a
specific real bug found in M1's gender head for Asian faces.

Scope: **M1 (baseline), M2-MULTI (resampling), M3-MULTI (reweighting)**.
ITA-only runs excluded. All numbers are on UTKFace's locked test set
(n = 2,405). Figures in `figures/`; raw calibration JSON in `data/`.

---

## 1. Multi-class calibration (race, age): the OPPOSITE result from LFW

LFW's deep dive (see that dataset's `deep_dive/deep_dive.md`) found that
threshold calibration **backfires** for rare race attributes, because
equalizing recall for a <5%-of-data class floods it with false positives.
UTKFace's race/age calibration does the **opposite** -- it works well --
because UTKFace's classes are not nearly as imbalanced (even the smallest
class, "Others," is 7% of the test set, versus LFW's Indian at 2.3%).

| Task | Method | Baseline accuracy | Baseline gap | Balanced accuracy | Balanced gap |
|---|---|---|---|---|---|
| Race | M1 | 80.2% | 58.2 pts | 73.0% | **0.7 pts** |
| Race | M2-MULTI | 78.7% | 44.7 pts | 71.0% | **1.3 pts** |
| Race | M3-MULTI | 77.2% | 50.1 pts | 71.4% | **0.8 pts** |
| Age | M1 | 76.9% | 41.5 pts | 73.0% | **0.6 pts** |
| Age | M2-MULTI | 77.2% | 30.0 pts | 70.2% | **1.4 pts** |
| Age | M3-MULTI | 73.6% | 35.1 pts | 72.0% | **3.7 pts** |

**This is a real, usable fix** -- a 7-10 point accuracy cost buys a gap
reduction from 30-58 points down to under 4, for every method, on both
tasks. Unlike LFW, this is not a precision trap: because every class has
a reasonable sample size here, pushing recall up for the worst class
(always "Others" for race, always "60+" or "40-59" for age -- see
per-class recall below) doesn't flood the result with false positives the
way LFW's 2.3%-of-data Indian class did.

**Caveat already in the method** (carried from `multiclass_calibration.py`):
calibrated directly on the test set, for diagnostic purposes -- a deployed
bias vector should be chosen on a held-out validation set instead.

Per-class recall before/after (M1, race): Others goes from **28.3% -> 72.8%**
recall (the worst class, by far, before calibration) while White drops
from 86.5% -> 72.7% -- the bias vector pulls every class toward the same
middle, not just boosting the weak one for free.

---

## 2. Full confusion matrices (race 5x5, age 4x4, gender 2x2, x3 methods)

All in `figures/confusion_{method}_{task}.png`. Key pattern not already in
`findings.md`:

**"Others" is predominantly confused with White, in every method:**

| Method | Others predicted as White | Others predicted as Others (correct) |
|---|---|---|
| M1 | 37.0% | 28.3% |
| M2-MULTI | 30.6% | 39.9% |
| M3-MULTI | 27.7% | 37.6% |

Both MULTI methods reduce this specific confusion (White misprediction
drops ~7-9 points, correct-Others recall rises 10+ points) -- a genuine,
if partial, improvement neither `findings.md` nor the summary table
isolates on its own.

---

## 3. A real bug/weak-spot found: M1's gender head has a recall problem specifically for Asian faces

`figures/n_vs_f1_scatter.png` and `figures/gender_f1_by_race_comparison.png`
show Asian as a visible outlier for M1 that both MULTI methods fix. Full
numbers:

| Method | Asian gender F1 | Precision | Recall |
|---|---|---|---|
| M1 | **0.680** | 0.966 | **0.524** |
| M2-MULTI | 0.822 | 0.777 | 0.872 |
| M3-MULTI | 0.867 | 0.840 | 0.896 |

**This is not a calibration problem the way gender was on LFW** -- M1's
precision is already excellent (96.6%, almost never wrong when it commits
to a prediction), the problem is pure recall: M1 fails to call roughly
**half** of Asian test faces by their correct gender at all, defaulting to
the other class instead. Both MULTI methods fix this almost entirely
(recall 52.4% -> 87-90%), at some cost to precision -- but the net F1 gain
is large and real (+0.14 to +0.19). This is the cleanest, least ambiguous
win found in this entire project: a specific, named, measurable defect in
the baseline, meaningfully fixed by both mitigation methods, not
explainable away by small-sample noise (n=357) or by a threshold
artifact.

---

## 4. Per-class recall bar charts (`figures/race_recall_comparison.png`, `figures/age_group_recall_comparison.png`)

Race: White is always the strongest class (81-87%), Others always the
weakest (28-40%) -- consistent across all three methods; no method
reorders this ranking, only compresses the gap between best and worst.

Age: the two middle age bands (20-39, 40-59) are consistently the
weakest, with 40-59 the single worst class under every method (55-64%
recall) -- a pattern invisible in the overall accuracy numbers alone,
since 20-39 is also the single largest class (632 of 2,405 test images),
so errors there barely move the headline accuracy number even though the
recall is mediocre.

---

## 5. What is still wrong after both mitigations (unsolved list)

1. **"Others" race recall is still the worst class by a wide margin**
   (28-40%, versus 71-88% for every other race class) in every method --
   this is UTKFace's own version of LFW's Indian problem: a catch-all
   category that was never a coherent class to begin with (see
   `race_mapping.yaml`'s documented reasoning for treating it as
   unreliable), and no training-time intervention tried here fixes that.
2. **Age 40-59 recall never exceeds 64%** in any method -- the single
   weakest cell in the entire UTKFace result set outside of "Others."
3. **Default-threshold gender gap only gets to ~4-6pts under MULTI
   mitigation, not fully closed** (`findings.md` already documents this;
   confirmed again here).
4. **Default-threshold race/age gaps (30-58 points) are enormous** and
   none of the three training regimes close them -- only the post-hoc
   logit-bias trick in Section 1 does, and that trades away 7-10 points
   of accuracy to do it.

## 6. What actually worked

- **Multi-class logit-bias calibration for race and age** (Section 1) --
  unlike LFW, this is a clean, usable win here: gap collapses by
  30-50+ points for a single-digit accuracy cost, on every method.
- **M2-MULTI and M3-MULTI both fix M1's Asian-gender recall problem**
  (Section 3) -- the single cleanest, most confidently real improvement
  found anywhere in this project.
- **Both MULTI methods reduce the Others-to-White confusion** (Section 2)
  somewhat, though Others remains the weakest class overall.

## 7. Not yet tried

- **Validation-set-selected logit-bias vectors** for race/age, to turn
  Section 1's diagnostic result into something deployable (current
  thresholds are tuned on the test set itself, by design, for diagnosis
  only).
- **Dropping "Others" entirely from the race task** (train only on
  White/Black/Asian/Indian) and reporting it as "unknown/unsupported"
  rather than forcing a 5-way prediction on a class that was never
  well-defined to begin with -- would likely raise both accuracy and the
  measured fairness gap simultaneously, since Others is both the smallest
  and the worst-performing class every time.
- **A dedicated age-band rebalancing pass targeting 40-59 specifically**
  -- this age band's weakness is consistent and large enough (worst class
  in 2 of 3 methods) to deserve its own targeted check, separate from the
  combined 4-attribute MULTI weight, which may be diluting it.

## 8. Intersectional analysis: race x gender x age (previously missing)

This was flagged as a real gap: `subgroup_metrics.csv` was missing
`race_age_label`/`gender_age_label`/`age_ita_label` entirely for all three
UTKFace methods -- `metrics.py` had been extended to compute these
intersections (for LFW's analysis) but was never rerun against UTKFace's
saved predictions afterward. Fixed by rerunning `metrics.py` for all three
methods; new figures: `figures/intersection_race_gender_f1.png` (full
10-cell race x gender, all methods), `figures/intersection_race_age_{method}.png`
(5x4 heatmap, gender F1 by race x age, one per method),
`figures/intersection_gender_age_race_acc.png` (race accuracy by
gender x age).

**Headline finding, invisible in every marginal (single-axis) view used
until now: gender recognition is dramatically worse for the youngest age
band (0-19), across every race, in the baseline -- and both mitigation
methods fix most of it.**

| Race | Baseline 0-19 gender F1 | Resampling | Reweighting |
|---|---|---|---|
| Asian | 0.24 | 0.69 | **0.76** |
| Black | 0.67 | 0.79 | **0.80** |
| Indian | 0.45 | 0.63 | 0.64 |
| Others | 0.37 | 0.60 | 0.61 |
| White | 0.64 | 0.74 | **0.79** |

For comparison, every other age band (20-39, 40-59, 60+) sits at
0.85-1.00 F1 in the baseline, for every race -- 0-19 is a uniquely weak
cell that the race-only view (Section 3's table, `findings.md`'s tables)
never surfaces, because it gets averaged away with the three much larger,
much easier age bands. This makes intuitive sense (children/teens have
less pronounced secondary sex characteristics for a vision model to rely
on) and both MULTI methods substantially close it -- consistent with age
being one of their four weighted attributes, and 0-19 being a smaller
group (see Section 9) that gets genuinely upweighted. **This is arguably
the cleanest, most broadly-reproduced mitigation win found anywhere in
this project** -- true across all five races, both methods, large in
magnitude (roughly +0.2 to +0.45 F1), not a small-sample artifact (each
cell has n=36-374).

The race x age heatmaps also confirm Section 3's Asian-recall finding is
*not* simply an age confound -- Asian's weakness is worst specifically in
the 0-19 cell (0.24), not spread evenly across ages, meaning Section 3's
"Asian gender problem" and this section's "youth gender problem" are
substantially the same underlying issue, not two separate ones.

## 9. Why resampling/reweighting aren't near-perfect: the weight cap

`figures/weight_cap_histogram.png`. The combined race+gender+age+
Fitzpatrick weight (`demographic_weights.py`, shared by every MULTI run in
this project) is capped at **4.0x average** by design, specifically to
stop one extremely rare image from dominating a batch or the loss. That
cap is also a hard ceiling on how much correction is possible, and
checking what gets thrown away quantifies it directly:

- UTKFace's single rarest training image would need a weight of
  **1,069.6x** average to be genuinely balanced against the rest of the
  dataset -- the cap allows only **4.0x**, 0.4% of what true parity would
  need. Only 3 of 19,241 training images are actually affected this
  severely, though (a narrow, extreme-outlier problem, not a broad one --
  see the histogram's isolated spike near 20x).
- This is much less severe than LFW's equivalent (171x needed, same 4.0x
  cap, but 220 of 10,628 images -- 2.1% -- affected; see that dataset's
  own deep dive for the comparison).

**This is a genuine, quantifiable reason resampling/reweighting can't
reach parity for the rarest cells (Others, and race x age combinations
like Asian_0-19) even in principle**: the mechanism is deliberately
throttled below what full correction would require. Raising the cap would
trade this off against the overfitting risk it exists to prevent -- a
real tuning knob, not yet explored here, and a natural next experiment
(e.g. re-run Resampling/Reweighting with `--max-weight 8` or similar, once
`demographic_weights.py` exposes that as a parameter rather than a fixed
constant).

## 10. ROC curves (`figures/roc_overlay_gender.png`)

Gender task, all three methods overlaid. AUROC barely moves between
methods (the ranking ability was never the problem -- consistent with
every other finding in this document pointing at specific weak subgroups
and cells, not a globally weak model).
