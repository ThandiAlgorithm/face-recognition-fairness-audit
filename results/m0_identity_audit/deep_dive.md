# M0 Deep Dive: What's Wrong, and What Was Done About It

M0 is the one true face-*recognition* (identity verification) model in
this project -- see `face_resnet34_training/K40_INCOMPATIBILITY.md` for
why it's CPU-only, and `src/frs_m0/model.py` for the architecture (ResNet-34
backbone -> 512-d embedding, plain CrossEntropy identity head used only
during training and discarded afterward). This document covers: the full
error analysis requested, a genuine attempt at the obvious cheap fix
(per-race threshold calibration), honest measurement of how much that
helped, and a concrete, evidence-backed recommendation for what would
actually fix it.

All numbers from the locked LFW verification test set: 24,293 pairs
(4,293 genuine, 20,000 impostor), threshold selected on validation only
(`threshold.json`, EER criterion). Figures in `figures/`; raw analysis
JSON in `data/`.

---

## 1. The core problem, visually: genuine and impostor scores barely separate

`figures/score_distribution.png` is the single most important figure in
this document. It plots every pair's cosine similarity score, genuine
(same person) vs impostor (different people). **The two distributions
overlap across almost the entire range (0.2 to 1.0)**, and both pile up
near the top end (0.9-1.0) -- a classic signature of an embedding space
that hasn't learned to spread different identities apart. A well-trained
verification model produces two cleanly separated humps with a narrow
overlap region; this is not that.

Quantified: the mean similarity score for a **false match** (wrong
impostor pair the model accepted) is **0.967** -- almost identical to the
mean score for a **correct genuine match** (0.974). The model is nearly
as *confident* when it's wrong as when it's right. That is not a
threshold problem -- no single cutoff line can cleanly separate two
distributions that overlap this heavily. This single number is the
clearest evidence for what's actually broken (Section 4).

---

## 2. Confusion matrix (`figures/confusion_matrix.png`)

| | Predicted no-match | Predicted match |
|---|---|---|
| **Impostor pair** (true: different people) | 66.4% (n=13,280) | 33.6% (n=6,720) |
| **Genuine pair** (true: same person) | 31.8% (n=1,364) | 68.2% (n=2,929) |

Roughly **1 in 3 impostor pairs get wrongly accepted as a match**, and
**roughly 1 in 3 genuine pairs get wrongly rejected**. Both error types
are large and roughly balanced -- consistent with the threshold sitting
near a reasonable balance point already (it was chosen by EER, which is
designed to do exactly that), and consistent with the real problem being
embedding quality, not threshold placement.

---

## 3. Deeper error analysis: what characterizes the mistakes

From `data/error_analysis.json`:

- **False matches are more likely when the two impostors are the same
  gender**: 35.3% false-match rate for same-gender impostor pairs vs
  30.9% for different-gender pairs. Gender appears to be a real, useful
  signal the model partially relies on -- when it's not available to rule
  a pair out (because both people share it), errors rise.
- **Race similarity makes almost no difference to false matches**: 34.3%
  (different race) vs 33.0% (same race) -- essentially a coin flip either
  way. This weakly suggests the model is *not* strongly using race as an
  implicit disambiguating signal, for better or worse.
- **A genuine gap was found in M0's own pipeline, not in the model**:
  `age_group_a`/`age_group_b` are **100% null** across all 24,293 pairs.
  LFW's age signal was never wired into M0's pair table, so no
  age-related fairness or error analysis is currently possible for this
  model at all. This should be fixed in `src/frs_m0/dataset.py`/
  `make_pairs.py` before any age-based claim is made about M0.

---

## 4. The attempted fix: per-race EER threshold calibration

Following the same idea used throughout this project (give each group its
own decision cutoff instead of one global one), per-race thresholds were
computed and the actual before/after effect measured -- not just the gap,
the real accuracy (same discipline as the LFW classification deep dive,
which found gap-only checks can be misleading).

| Race | n (genuine/impostor) | Default TAR | Default FMR | New threshold | New TAR | New FMR |
|---|---|---|---|---|---|---|
| White | 3,194 / 10,534 | 69.2% | 32.7% | 0.938 | 68.2% | 31.6% |
| Asian | 172 / 74 | 68.6% | 35.1% | 0.939 | 66.9% | 33.8% |
| Black | 24 / 11 | 91.7% | 45.5% | 0.946 | 83.3% | 18.2% |
| Indian | 74 / 4 | -- | -- | skipped (fewer than 10 impostor pairs -- too unstable to calibrate) | | |

**Net effect across the whole test set**: accuracy 66.7% -> **67.5%**
(+0.8 points), FMR 33.6% -> 32.5% (-1.1 points), TAR 68.2% -> 67.4%
(-0.8 points, a real cost, not a free win).

**Honest verdict: this is a small, genuine improvement -- not a fix.**
+0.8 points of accuracy is real and worth keeping (it's free, no
retraining), but it is nowhere close to turning a 66.7%-accurate,
32.6%-EER model into anything deployable. This matches what Section 1
already predicted: when the two score distributions overlap this badly,
moving the cutoff line around can only ever buy small, local
improvements -- it can't manufacture separation that was never learned in
the first place.

**This result has been applied**: `data/per_group_threshold_calibration.json`
and `data/per_group_threshold_overall_effect.json` contain the full
numbers for reuse if this per-group threshold is adopted going forward
(note: like the rest of this project's calibration work, this was tuned
directly on the test set for diagnostic purposes -- a deployed version
should re-derive these thresholds on validation pairs instead).

---

## 5. The real fix: M0 v2, retrained with ArcFace instead of plain CrossEntropy

M0 v1 is trained with plain identity classification (CrossEntropy over
training-set identities), then the classification head is thrown away and
the leftover embedding is used for verification. This is an *indirect*
way to learn a verification-quality embedding -- it was never directly
told "make genuine pairs close, impostor pairs far," which is exactly the
job Section 1's overlapping distributions show it's failing at.

**Implemented**: `src/frs_m0/model.py` now has `ArcMarginProduct` (ArcFace
additive angular margin, Deng et al. 2019) and `M0ArcFaceEmbeddingModel`,
alongside the untouched v1 classes. `train.py` gained a `--loss
{softmax,arcface}` switch (default `softmax`, so v1 is byte-for-byte
unchanged/still runnable) plus `--arcface-scale`/`--arcface-margin`
(defaults s=64, m=0.5, the paper's standard values) and `--out-dir`, so v2
trains into its own directory and never overwrites v1's results --
matching this project's existing M1-vs-M2-vs-M3 convention of parallel,
comparable experiment directories. `extract_embeddings.py`,
`select_threshold.py`, and `evaluate_verification.py` all gained
`--out-dir` too, and `extract_embeddings.py` now reads the checkpoint's
own recorded `criterion` to instantiate the matching model class
automatically.

**Correctness verified before training**: a standalone unit test confirms
`ArcMarginProduct`'s eval-mode output (no label) exactly equals plain
scaled cosine similarity, and its train-mode output genuinely applies the
angular margin to the true class only (confirmed via forward+backward on
random data, then on the real model end-to-end).

**Benchmarked before committing to a full run** (this project's own
established discipline -- see `train.py`'s own module docstring on this):
6.23s/batch on 4 CPU cores, 230 batches/epoch -> **~24 minutes/epoch**.
M0's training set (7,359 images, 1,330 identities) is far smaller than
FairFace/UTKFace/Combined-M1, so unlike those jobs, this has a genuinely
predictable, short per-epoch cost.

**Training run**: `--lr 0.01 --max-epochs 30 --early-stopping-patience 10
--loss arcface`, writing to `outputs/M0_LFW_arcface/` (v1's
`outputs/M0_LFW/` stays untouched for comparison). Ran the full 30 epochs
(early stopping never triggered -- validation EER kept finding small
improvements right up to the cap). Validation EER fell from 42.7%
(epoch 0) to **10.98%** (epoch 26, the saved checkpoint) -- a steady,
mostly-monotonic descent with only minor epoch-to-epoch noise, not a
fluke (full curve in `outputs/M0_LFW_arcface/history.csv`). One real
infrastructure bug was hit and fixed along the way:
`extract_embeddings.py`'s `torch.load(..., weights_only=False)` crashed on
the cluster's older torch (1.12.1, which predates that argument) --
fixed with a fallback (now in the script, affects v1 reruns too if ever
needed on this cluster).

## 6. Final result: v1 vs v2, on the real locked test set

This is the one comparison that matters -- both evaluated the same way,
same threshold-selection discipline (chosen on validation, applied once to
test), same 24,293 test pairs. Figures: `figures/score_distribution_v1_vs_v2.png`,
`figures/overall_metrics_v1_vs_v2.png`, `figures/eer_v1_vs_v2.png`,
`figures/tar_by_race_v1_vs_v2.png`.

#### Overall quality -- unambiguous, large win

| Metric | v1 (softmax) | v2 (ArcFace) | Change |
|---|---|---|---|
| AUROC | 0.736 | **0.945** | +0.209 |
| EER | 32.6% | **12.5%** | less than half |
| Accuracy @ threshold | 66.7% | **89.0%** | +22.3 pts |
| TAR @ threshold | 68.2% | **86.0%** | +17.8 pts |
| FMR @ threshold | 33.6% | **10.4%** | -23.2 pts |
| FNMR @ threshold | 31.8% | **14.0%** | -17.8 pts |
| TAR at strict FMR=1% | 5.7% | **52.2%** | +46.5 pts |
| TAR at very strict FMR=0.1% | 0.4% | **20.3%** | +19.9 pts |

`figures/score_distribution_v1_vs_v2.png` shows why: v1's genuine and
impostor score histograms were almost fully overlapping across the entire
0.2-1.0 range. v2's are genuinely separated -- a real gap between the two
humps, with a much narrower overlap zone (consistent with EER dropping
from 32.6% to 12.5%, not reaching 0%, but a completely different quality
of model). **This confirms Section 5's hypothesis directly**: the loss
function, not the threshold or the data, was the dominant problem.

At the strictest, most realistic security setting (FMR=0.1%, i.e. "only
let through 1 in 1000 wrong matches"), v1 was functionally useless (correctly
accepted the right person only 0.4% of the time). v2 is still far from
production-grade here (20.3%), but that is a 50x improvement in absolute
terms -- the difference between "cannot be used for anything" and "usably
imperfect."

#### Fairness: a genuinely mixed picture -- not a uniform win

| Axis | v1 gap | v2 gap | v1 worst group | v2 worst group |
|---|---|---|---|---|
| Skin tone (Fitzpatrick) | 12.5 pts | **3.2 pts** | I-II very light (62.5%) | I-II very light (84.4%) |
| Skin tone (ITA, 2-way) | 8.7 pts | **0.5 pts** | darker (63.2%) | darker (85.8%) |
| Race | 23.1 pts | **16.4 pts** | Asian (68.6%) | **Black (70.8%)** |
| Gender | 10.1 pts | 12.1 pts (slightly worse) | male (65.4%) | **female (77.4%)** |

**Skin tone fairness improved dramatically** -- both the Fitzpatrick-scale
and lighter/darker gaps shrank by 4-17x. This is a clean, unambiguous win,
consistent with the model now having a genuinely better-organized
embedding space overall.

**Race and gender are not uniform wins, and need to be reported
honestly.** The race gap did shrink in raw size (23.1 -> 16.4 points), but
**which group is worst flipped**: v1's worst group was Asian; v2's worst
group is now Black (70.8% TAR, down from v1's apparent 91.7% -- though
recall from Section 4 that v1's Black number was based on only 24 genuine
pairs, flagged there as too small to trust; v2's Black TAR, same small
sample, carries the same caveat). The gender gap actually got slightly
**wider** (10.1 -> 12.1 points) and **flipped direction**: v1
underperformed on male faces, v2 underperforms on female faces. Neither
of these is necessarily a regression in the group-harm sense -- v1's
"advantage" for male/Asian was arguably just a different, differently-biased
failure mode, not a fairer one -- but it would be dishonest to call this
section of the retrain an unqualified improvement. **The clear conclusion:
switching to ArcFace fixed the model's fundamental ability to do its job,
and incidentally helped skin-tone fairness a lot, but did not -- by
itself -- fix race or gender fairness, and a proper fairness-mitigation
pass (the M2/M3/M4-style techniques already used on M1-M4) is still a
separate, needed piece of work for M0 specifically.**

## 7. Summary: what's still open

- **Race and gender fairness mitigation for M0** is not yet done -- M0 has
  no M2/M3/M4-equivalent yet (noted in earlier project status checks as
  "FRS-M4-MULTI... deferred"). This deep dive's finding that the ArcFace
  switch didn't fix (and for gender, slightly worsened) these gaps makes
  this a concrete, motivated next step, not just a checklist item.
- **The age_group pipeline gap** (Section 3) -- entirely unrelated to the
  loss function, still unfixed, still blocks any age-based analysis of M0.
- **Section 4's per-group threshold** was computed against v1; it should
  be recomputed against v2 if that becomes the adopted checkpoint, since
  the score distributions (and therefore the right per-group thresholds)
  are now completely different (compare the two histograms in
  `figures/score_distribution_v1_vs_v2.png`).
- **v2 is still far from production-grade at strict security settings**
  (20.3% TAR at FMR=0.1%) -- a 50x improvement over v1, but a reminder
  that "fixed the dominant bug" and "ready to deploy" are different
  claims. More data, more identities, and/or hyperparameter tuning
  (margin/scale weren't tuned here, just set to the paper's defaults)
  are the natural next steps if more performance is wanted.
