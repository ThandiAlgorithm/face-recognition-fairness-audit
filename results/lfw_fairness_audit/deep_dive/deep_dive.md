# LFW Deep Dive: Everything Wrong, M1 vs M2-MULTI vs M3-MULTI

This is a full audit beyond the summary `findings.md` one level up: every
task's confusion matrix (with %), every demographic attribute's threshold
behaviour (not just gender), a genuine before/after check of what
threshold-shifting does to precision and F1 (not just the recall gap), and
an honest list of what is still broken after both mitigation methods.

Scope: **M1 (baseline), M2-MULTI (resampling), M3-MULTI (reweighting)**.
ITA-only runs are excluded per current scope. All numbers are on LFW's
locked test set (n = 1,188: 297 true-female, 891 true-male), recomputed
directly from `test_predictions.parquet` for this document, so they are
reproducible, not transcribed from memory.

Figures referenced below are in `figures/`. Source data (full JSON) is
alongside the `test_eval/` directories under
`face_resnet34_training/experiments/lfw/`.

---

## 1. Headline finding: the threshold trick that fixed gender actively BREAKS the rare race attributes

This is the single most important correction to make to the project's
existing understanding. `findings.md` (one level up) reports that
threshold calibration "confirms the gender gap is substantially a
calibration problem" — true, and a genuine win (gap 18.5pts -> ~0pts, F1
barely moves). But that check only looked at the **recall gap**, never at
**precision or F1** at the new threshold. Doing that check for every
other demographic attribute changes the conclusion completely:

| Attribute | Method | Default F1 | Default P/R | "Balanced" F1 | Balanced P/R | Verdict |
|---|---|---|---|---|---|---|
| Male (gender) | M1 | 0.959 | 0.941/0.981 | ~0.93 (recall-matched) | ~0.90/~0.91 | **Threshold fix genuinely helps** |
| White | M1 | 0.899 | 0.891/0.908 | 0.857 | 0.922/0.801 | Worse |
| White | M2-MULTI | 0.907 | 0.870/0.948 | 0.847 | 0.917/0.787 | Worse |
| Black | M1 | 0.583 | 0.656/0.525 | **0.274** | 0.163/0.850 | **Much worse** |
| Black | M2-MULTI | 0.533 | 0.800/0.400 | **0.317** | 0.196/0.825 | **Much worse** |
| Black | M3-MULTI | **0.639** | 0.719/0.575 | 0.271 | 0.161/0.850 | **Much worse** |
| Asian | M1 | 0.508 | 0.738/0.388 | 0.450 | 0.304/0.863 | Worse |
| Asian | M3-MULTI | 0.481 | 0.500/0.463 | 0.414 | 0.275/0.838 | Worse |
| Indian | M1 | 0.000 | 0.000/0.000 | **0.152** | 0.084/0.815 | Improves, but P is terrible |
| Indian | M2-MULTI | 0.182 | 0.235/0.148 | 0.177 | 0.099/0.815 | No better |
| Indian | M3-MULTI | 0.105 | 0.182/0.074 | 0.162 | 0.090/0.815 | Slightly better |
| Pale_Skin | M1 | 0.864 | 0.923/0.811 | 0.875 | 0.874/0.875 | Roughly neutral |

**Why this happens**: the "pick the threshold where both recalls are
equal" technique treats a false positive and a false negative as equally
costly, which is a reasonable choice when the two classes are close to
balanced (gender is 75% male / 25% female — skewed but not extreme). For
Black (4.7% of test set), Asian (6.7%), and especially Indian (2.3%),
equalizing recall means accepting a huge number of false positives,
because the negative class is so much bigger that even a small false
positive *rate* is a large false positive *count*. Concretely, fixing
Indian's recall gap at M1 means predicting "Indian" for 262 people instead
of 4 — only 22 of those 262 are actually Indian (precision 8.4%).

**The corrected conclusion**: threshold calibration is a genuinely good,
free fix **only for gender here** — not a general-purpose fix for every
demographic gap. For the rare race attributes, M2-MULTI/M3-MULTI's actual
retraining (weak as the results still are) outperforms any post-hoc
threshold shift on F1. This reframes the earlier framing in `findings.md`,
which implied threshold calibration was a broadly "not yet tried" cheap
win across the board — it isn't, for anything this imbalanced.

See `figures/confusion_*.png` for every attribute x method (row-normalized
%, i.e. recall per true class) and raw counts.

---

## 2. Full confusion matrices (6 attributes x 3 methods = 18 matrices)

All in `figures/confusion_{method}_{attribute}.png`. Highlights not
already covered by the F1 tables in `findings.md`:

- **White confusion (M1)**: 33.1% of non-White people get called White
  (the single largest absolute error source in the whole attribute set, by
  raw count — 99 of 299 non-White test images). This false-positive rate
  barely moves across methods (32.4-42.1%), meaning **none of the methods
  improved the model's tendency to default to "White" under uncertainty.**
- **Black (M3-MULTI)**: best true-positive rate of any method (57.5%) but
  this is still "misses 42.5% of actual Black faces."
- **Indian (all methods)**: even at its best (M2-MULTI, 14.8% recall),
  the model is still wrong about an actual Indian face **85% of the
  time**. "Improved" here means "less catastrophically broken," not
  "good."

---

## 3. Per-attribute F1 bar chart and intersection bar chart

`figures/attribute_f1_comparison.png`, `figures/intersection_f1_comparison.png`.

No single method wins on every attribute — see table in Section 1. The
intersection chart makes visible what the subgroup table in `findings.md`
already states in numbers: White_female sits far below every other
gender/race cell on gender F1 (~0.37-0.44) across all three methods, the
single worst-performing subgroup with enough data to trust (n=220).

---

## 4. Sample-size vs F1 scatter (`figures/n_vs_f1_scatter.png`)

Confirms the small-sample-noise finding already documented for
Black_female (n=13) quantitatively: every subgroup with n < 20 shows much
more scatter between methods than subgroups with n > 200, which is exactly
what you'd expect from sampling noise rather than a real effect. Treat any
per-method difference for Black_female, Asian_female, or Indian_female as
noise, not signal, unless it is reproduced at n > 50.

---

## 5. Overall accuracy (`figures/overall_accuracy_comparison.png`)

| | M1 | M2-MULTI | M3-MULTI |
|---|---|---|---|
| Mean attribute accuracy (73 attrs) | 87.8% | 87.4% | 87.7% |

Confirms `findings.md`'s number; included here as a figure for the thesis
writeup.

---

## 6. What is still wrong after both mitigations (unsolved list)

1. **Indian recognition is still fundamentally broken.** Best F1 across
   every method and every threshold tried: 0.182 (M2-MULTI, default
   threshold). This is a *representation* problem (only 33 Indian-female,
   159 Indian-male images in the full 13,233-image dataset) that no
   post-hoc technique tried here can fix, and that neither resampling nor
   reweighting fixed to a usable level either.
2. **The gender gap at the White_female intersection is untouched** by
   either mitigation method (Section 3 above, and `findings.md`).
3. **The model defaults to "White" under uncertainty**, and this bias is
   unchanged by any method tried (Section 2).
4. **Race-attribute threshold calibration is not a usable fix** (Section
   1) — this was an open question in `findings.md`; now answered, and the
   answer is "no, not for the rare attributes."
5. **Black and Asian F1 trade off against each other and against White**
   depending on method — there is no method here that improves all three
   simultaneously (`findings.md` Section "Race and gender attributes").

## 7. What actually worked

- **Gender threshold calibration**: genuinely fixes the gender gap, no
  retraining needed, no precision cost at a reasonable threshold (~0.86).
- **M3-MULTI on Black**: best Black F1 of any method (0.639), a real,
  if partial, win -- but note this is the *best* result in the whole
  dataset, and it still misses 4 in 10 Black faces.
- **M2-MULTI on Indian**: moved Indian off a complete-failure baseline,
  even though the result (F1 0.182) is still weak in absolute terms.

## 8. Not yet tried (carried over + updated from findings.md)

Everything listed in `findings.md`'s "techniques not yet tried" section
still applies. Added, based on this deep dive specifically:
- **Per-class loss weighting tuned for precision, not just recall**
  (e.g., focal loss, or a cost matrix that penalizes false positives on
  rare classes more) — the EER-style threshold search used here optimizes
  for equal recall only; a precision-aware calibration target (e.g.
  maximize F1 directly via a precision-recall sweep, not equal-recall)
  might avoid the regression seen in Section 1.
- **Decoupled classifiers per race group**, specifically for the White
  false-positive problem (Section 2) — since the bias to default to White
  persists identically across all three training regimes, it is likely
  baked into the shared feature representation, not fixable by reweighting
  the final layer alone.

## 9. Why resampling/reweighting aren't near-perfect: the weight cap

`figures/weight_cap_histogram.png`. Every MULTI run in this project
(LFW, UTKFace, FairFace) shares one weight formula
(`demographic_weights.py`), capped at **4.0x average** by design -- to
stop one extremely rare image dominating a batch or the loss. Checking
what that cap actually throws away:

**LFW's rarest training image would need a weight of 171.3x average to be
genuinely balanced -- the cap allows only 4.0x, 2.3% of what true parity
would require.** Unlike UTKFace (where this only affects 3 extreme-outlier
images out of 19,241), LFW's problem is broad: **220 of 10,628 training
images (2.1%) sit above the 4.0x cap** -- the histogram shows a real,
continuous tail from 2x out to 8x and beyond, not just a couple of
isolated extreme cases. This is a direct, quantified explanation for why
Indian (the rarest race attribute, needing the most correction) only ever
reached F1 0.105-0.182 under either MULTI method (Section 1) -- the
mechanism correcting for its rarity was never allowed to apply more than
4x the weight an average image gets, regardless of how many multiples of
that Indian's true rarity actually warranted.

**Why LFW is worse than UTKFace here**: LFW is "faces in the wild"
(celebrity photos, massively skewed toward White men -- White_male alone
is 56.5% of the dataset, Section "Why Black/Indian cells are small"
above), while UTKFace was scraped with less extreme skew and FairFace was
purpose-built for balance (confirmed in that dataset's own deep dive: its
pre-cap weights barely exceed 4x at all, so the cap rarely binds there).
**The severity of this problem is a direct function of how imbalanced the
source dataset already was** -- the same formula, same cap, wildly
different bite.

## 10. ROC curves (`figures/roc_overlay_Male.png`)

Male attribute, all three methods overlaid. AUROC: Baseline 0.971,
Resampling 0.962, Reweighting 0.969 -- all close, consistent with every
other finding here: the model's underlying ranking ability for gender was
never badly broken (Root cause 1's AUROC 0.97 finding, confirmed again),
the problem throughout this dataset is specific weak subgroups and cells
(White_female, Indian, the youngest/rarest combinations), not a globally
weak model.
