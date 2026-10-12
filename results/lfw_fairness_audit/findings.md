# LFW Fairness Audit: M1 Baseline vs Mitigation

Source data for every number below is in `data/` (raw JSON/CSV pulled
directly from the evaluation pipeline, `m1_`/`m2_multi_`/`m3_multi_`
prefixes) and the root-level `error_analysis_Male*.json` /
`threshold_calibration_Male*.json` files (unsuffixed = M1; `_m3_multi` =
that variant) -- the two post-hoc analyses this doc is built around. All
numbers are from LFW's locked test set (n = 1,188; 297 true-female, 891
true-male), never touched during training or model selection.

Scope note: this audit only covers the brief-compliant MULTI mitigation
variants (resampling/reweighting by the combined race+gender+age+skin-tone
weight). The earlier ITA-only (skin-tone-only) M2/M3 runs were a
pre-clarification experiment and are intentionally excluded here.

## LFW

### M1 vs M2-MULTI (resampling) vs M3-MULTI (reweighting)

Both resample/reweight by race, gender, age, and skin tone (Fitzpatrick)
combined into one per-image weight -- the brief-compliant version.

#### Overall accuracy

| | M1 | M2-MULTI | M3-MULTI |
|---|---|---|---|
| Mean attribute accuracy (73 attrs) | 87.8% | 87.4% | 87.7% |

Both mitigation variants cost a small amount of overall accuracy. Neither
is a free win.

#### Race and gender attributes (F1)

| Attribute | M1 | M2-MULTI | M3-MULTI |
|---|---|---|---|
| Male (gender) | 0.959 | 0.955 | 0.960 |
| White | 0.899 | 0.907 | 0.900 |
| Black | 0.583 | 0.533 | **0.639** |
| Asian | 0.508 | 0.441 | 0.481 |
| **Indian** | **0.000** | **0.182** | 0.105 |
| Pale_Skin (skin tone) | 0.864 | 0.852 | 0.855 |

**Headline result: both MULTI variants moved Indian off completely broken
(F1 = 0.000, worse than random) to a real, if still weak, signal** --
resampling (M2-MULTI) got it furthest (F1 0.182, MCC +0.172, balanced
accuracy 0.498 → 0.569), reweighting (M3-MULTI) a bit less (F1 0.105).
This is the first mitigation result anywhere in this project with a
genuine, multi-metric-confirmed improvement on the single worst-performing
group. Reweighting also gave the best Black result of any method here
(F1 0.639, versus M1's 0.583 and M2-MULTI's worse 0.533) -- the two
methods trade off differently even though both share the same combined
weight formula: resampling helps the rarest group (Indian) more,
reweighting helps the next-rarest group (Black) more.

The cost: whichever race group a given method helps least tends to drift
down a little (Asian under both, Black under M2-MULTI). Gender (Male) and
skin tone (Pale_Skin) both drifted slightly in one direction or the other
under both methods. Mitigation here is a trade, not a uniform improvement
-- gains for the worst group come partly at the expense of others.

#### Subgroup / intersection: gender by race×gender

| Group | n | M1 F1 | M2-MULTI F1 | M3-MULTI F1 |
|---|---|---|---|---|
| White_female | 220 | 0.400 | 0.377 | 0.375 |
| Black_female | 13 | 0.800 | 0.500 | 0.800 |
| Asian_female | 17 | 0.800 | 0.750 | 0.750 |
| Indian_female | 2 | 0.000 | 0.000 | 0.000 |
| White_male | 645 | 0.983 | 0.981 | 0.985 |
| Black_male | 21 | 0.976 | 0.950 | 0.950 |
| Asian_male | 51 | 0.990 | 0.970 | 0.980 |
| Indian_male | 15 | 1.000 | 0.966 | 1.000 |

**The gender gap was not fixed at the intersection level by either
method.** White_female -- the single worst gender subgroup, and by far
the largest female group in the dataset (n = 220) -- did not improve
under either variant; it got slightly worse both times. Every subgroup
drifted flat-to-down. Multi-demographic resampling/reweighting targeted
gender as one of its four attributes, but did not actually close the
gender gap anywhere.

**Black_female's apparent swing under M2-MULTI (0.800 → 0.500) is
small-sample noise, not a real finding** (M3-MULTI, for reference, stayed
at 0.800). All three runs share the same 3 true-positive cases in this
n=13 cell; M1 and M3-MULTI caught 2 of 3, M2-MULTI caught 1 of 3 -- a
single flipped prediction. At this sample size a single image swings F1
by 0.3. **Do not read this cell as evidence of anything beyond "too few
Black women in this dataset to measure reliably."**

#### Why Black/Indian cells are small: dataset representation, not a split artifact

Full-dataset (all 13,233 LFW images, not just the test split) race x
gender counts:

| Group | Count | % of dataset |
|---|---|---|
| White_male | 7,482 | 56.5% |
| White_female | 2,077 | 15.7% |
| Asian_male | 635 | 4.8% |
| Black_male | 339 | 2.6% |
| Asian_female | 233 | 1.8% |
| Indian_male | 159 | 1.2% |
| **Black_female** | **140** | **1.1%** |
| **Indian_female** | **33** | **0.2%** |

Black women are 1.1% of the entire dataset -- White men alone outnumber
them 53 to 1. This is a hard limit on what any mitigation technique can
achieve here: resampling/reweighting can make training pay more attention
to the ~140 Black-women images that exist, but cannot manufacture more of
them. It is a dataset-composition ceiling, not a training-technique
problem.

### Is the gender gap a race issue or a gender issue?

**Gender issue, demonstrated cleanly by holding race constant.** White is
the best-represented, highest-data race in LFW, and still shows a massive
male/female gap (White_male F1 0.983 vs White_female F1 0.400 in M1).
Since race and data volume are held fixed between these two groups, the
gap cannot be a representation problem -- it is a property of how the
model recognises gender itself, and the same male >> female pattern
repeats within every race group checked (Asian 0.990 vs 0.800, Black
0.976 vs 0.800 in M1).

Separately, race classification itself is weak in a way gender
classification is not (Indian F1 = 0.000 in M1, Asian/Black F1 ~0.5-0.6 vs
Male's typical F1 > 0.9), and that weakness tracks representation closely
(Indian = 1.5% of the dataset). These are two independent problems that
happen to show up in the same model: a gender-recognition bias (affects
women across every race), and a race-recognition weakness (driven by how
little data exists for non-White groups). A mitigation aimed at one does
not automatically fix the other -- confirmed empirically: both MULTI
variants moved Indian (race) but left the gender gap completely
untouched.

### Root cause 1: the 0.5 decision threshold, not the model's underlying ability

`threshold_calibration_Male.json` (M1) /
`threshold_calibration_Male_m3_multi.json` (M3-MULTI) -- post-hoc analysis
on each model's saved prediction probabilities, no retraining.

Sweeping every threshold from 0.01 to 0.99 and picking the point where male
and female recall are equal:

| Method | Default (0.5) gap | Default male/female recall | Balanced threshold | Balanced gap |
|---|---|---|---|---|
| M1 | 18.5 pts | 0.983 / 0.798 | 0.860 | 0.000 |
| M3-MULTI | 16.6 pts | 0.981 / 0.815 | 0.865 | 0.003 |

**Both collapse to a near-zero gap with a single post-hoc threshold change
around 0.86 -- neither needed retraining to fix this.** This held
per-race for M1 too (White, n=865: balances at 0.865, 90.8%/91.0%; Asian,
n=68: balances at 0.81, 92.6%/92.9%). **This confirms the gender gap is
substantially a calibration problem, not a representation or
model-capacity problem** -- the model's ranking ability was always
reasonable (AUROC 0.97 for Male); the single global threshold was simply
set too low, over-predicting "Male" across the board and disproportionately
mislabelling women. Reweighting (M3-MULTI) moved the optimal threshold
only slightly (0.865 vs M1's 0.86) -- unlike UTKFace, where M3-MULTI's
gender improvement was a genuine training-time change (see
`utkface_fairness_audit`), LFW's M3-MULTI did not meaningfully re-calibrate
the gender head on its own.

This is a genuine fairness-improvement technique not yet used anywhere else
in this project as a training-time fix (M2/M3/M4 are all training-time
interventions); it requires no retraining, only a different decision rule
applied to an already-trained model.

### Root cause 2: what visually distinguishes the misclassified women

`error_analysis_Male.json` (M1) / `error_analysis_Male_m3_multi.json`
(M3-MULTI) -- compares true-female images wrongly predicted "Male" (at the
default 0.5 threshold) against the correctly classified ones, across
every other LFWA+ attribute in the same image.

| Method | Female→male misclassification rate (n=297 true-female) |
|---|---|
| M1 | 20.2% (60 misclassified) |
| M3-MULTI | 18.5% (55 misclassified) |

M1's attribute-rate comparison (the largest sample, most reliable):

| Attribute | Rate when misclassified | Rate when correct | Gap |
|---|---|---|---|
| Wearing_Lipstick | 20% | 78% | -58 pts |
| Wearing_Earrings | 18% | 75% | -57 pts |
| Heavy_Makeup | 7% | 58% | -51 pts |
| Wearing_Necklace | 37% | 83% | -46 pts |
| Wearing_Necktie | 42% | 8% | **+33 pts** |

The misclassified women are overwhelmingly the ones without
stereotypically-feminine cosmetic/accessory markers (makeup, lipstick,
earrings), and are far more likely to be wearing a stereotypically
masculine item (a necktie). This is a specific, interpretable finding: the
model has learned a narrower, stereotype-linked visual shortcut for gender
rather than a more invariant one. Misclassification rate did not vary much
by race among groups with enough data (White 19.8%, Asian 21.4%, unknown
26.5%), consistent with this being a gender-presentation issue rather than
a race-driven one. The same lipstick/earrings/makeup/necktie pattern
reproduces in M3-MULTI's attribute-rate comparison (saved in full in its
JSON file) -- the misclassification signature does not change under
reweighting, consistent with Root cause 1's finding that M3-MULTI barely
shifted the underlying calibration either.

Each method's misclassified rows (image path + model confidence, sorted by
confidence) are saved in full inside the corresponding `error_analysis_*`
JSON file for direct visual spot-checking.

### Techniques tried vs not yet tried, for this specific gender gap

Already tried (training-time interventions): resampling (M2-MULTI),
reweighting (M3-MULTI) -- neither closed the gender gap, consistent with
the diagnosis above -- it is a calibration/learned-shortcut issue, not a
representation issue that resampling or reweighting could fix.

Not yet tried, and directly motivated by the two root causes above:
- **Per-group threshold calibration** (demonstrated above, ~zero
  implementation cost, no retraining)
- **Decoupled/group-specific classifiers** (Dwork et al. 2018) -- a
  natural next step from threshold calibration, giving each group its own
  learned decision boundary rather than one global one
- **Constrained optimisation / reductions approach** (Agarwal et al. 2018)
  -- a more principled version of M3's heuristic reweighting, with an
  explicit fairness constraint in the training objective
- **Auxiliary multi-task training** on correlated attributes
  (Heavy_Makeup, Wearing_Lipstick) to give the model richer gender-relevant
  signal beyond the narrow cosmetic shortcut it currently relies on
- **Supervised contrastive representation learning** to make the embedding
  space itself more gender-separable before classification, rather than
  only reweighting the final loss
