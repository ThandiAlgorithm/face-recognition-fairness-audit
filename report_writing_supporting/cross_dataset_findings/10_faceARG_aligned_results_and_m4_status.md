# FaceARG Aligned Results and M4 Status — Summary of This Investigation

This file pulls together two separate threads of work into one place: the
finished, aligned FaceARG external-benchmark results across all three
source datasets, and the current status of M4 (adversarial debiasing) on
each of FairFace, UTKFace and LFW. The detailed, blow-by-blow methodology
for the FaceARG work (leakage checks, the race-label mapping, the
unaligned ablation, the pipeline sanity test) lives in
`09_faceARG_external_benchmark.md` — this file is the destination, not a
replacement; read file 09 first for the "how," this file for the "what we
found."

## Part 1: Why realignment was necessary, in one paragraph

FaceARG's own crops were originally fed through this project's models
using only resize-and-normalise preprocessing — never through the actual
training-time pipeline, which is RetinaFace face detection followed by a
5-point landmark similarity-transform alignment to fixed canonical
positions (`configs/preprocessing.yaml`). The project's own code
(`face-demo-app/backend/detect.py`) documents that skipping this step is
"a real train/inference distribution mismatch," not a cosmetic one. A
pipeline sanity test (running 200 of FairFace's own test images through
the exact same FaceARG scoring code) confirmed the scoring code itself was
correct — Black recall came out at 75%, matching the known in-distribution
number, with no sign of the Black-to-Asian confusion that had shown up on
unaligned FaceARG. That ruled out a code bug and left preprocessing as the
explanation. Every FaceARG image was then re-processed through the actual
RetinaFace + landmark pipeline. 8,784 of the 8,998 leakage-clean
candidates (97.6%) were successfully detected and aligned; 214 had no face
found by RetinaFace and were excluded outright, with no substitute crop,
matching this project's own governance policy for detection failures
(`configs/exclusions.yaml`). One further image was dropped after the fact
for a truncated file write (8,784 → 8,783), for reasons unrelated to the
alignment pipeline itself.

Detection failure itself was not evenly spread across demographic groups:
afro-american faces failed detection at 4.88%, more than double the
asian/caucasian rate (~2.1% each) and about eight times the indian rate
(0.61%) — confirmed as a real effect, not sampling noise, by a chi-square
test (χ²=79.56, Yates-corrected, p=4.69×10⁻¹⁹). This is *not* the same
ranking this project's own UTKFace audit found for its own detector
(there, Asian failed most at 0.474%, Black least at 0.132%) — the honest
claim is that detection failure rate differs by group *and by dataset*,
not that any one group is always hardest. Because failures are dropped
rather than patched, the afro-american group shrinks proportionally more
under the aligned benchmark (2,255 → 2,145, a 4.9% reduction) than the
other three groups — any aligned afro-american number should carry that
caveat.

## Part 2: Aligned results — FairFace-trained models

All five FairFace-family models (M1, M2, M3, BAL-M1, and now M4 — see the
note below the table) were scored on the same 8,783 aligned images (one of
the originally-reported 8,784 was a truncated file, dropped from the
candidate set; M1/M2/M3/BAL-M1's numbers are unchanged by this correction,
confirmed by recomputing them at the corrected count), with calibration
(gender threshold, race bias vector) fit once on FairFace's own clean
validation subset and applied once — never fit on FaceARG itself.

| Model | Race acc. (default) | Race acc. (calibrated) | 95% CI (calibrated) | Gender acc. (calibrated) |
|---|---|---|---|---|
| M1 (plain baseline) | 67.5% | 61.4% | [60.3%, 62.4%] | 96.9% |
| M2 (resampling) | 52.1% | 57.3% | [56.3%, 58.2%] | 92.9% |
| M3 (reweighting) | 70.1% | 63.8% | [62.8%, 64.8%] | 95.6% |
| BAL-M1 (balanced-HarmoFace) | 65.5% | 61.0% | [60.0%, 62.0%] | 94.6% |
| **M4 (adversarial debiasing)** | **71.4%** | **67.6%** | **[66.7%, 68.6%]** | 96.2% |

**M4 was scored on FaceARG for the first time in this pass** (it was
excluded from the original run below because its checkpoint did not exist
yet at the time). Its calibrated-race-accuracy CI [66.7%, 68.6%] does not
overlap with any other model's — **M4 has the best race accuracy of all
five FairFace-family models on this external benchmark, by a clear,
non-overlapping margin**, both before and after calibration. This is a
genuinely different outcome from this project's own in-distribution
FairFace test set, where M4's race accuracy (63.0%) was close to but not
dramatically ahead of the baseline — on FaceARG specifically, M4 pulls
clearly ahead of every other method, including M3 which had been the
in-distribution-and-FaceARG leader before this result existed.

Paired bootstrap, Bonferroni-corrected across 3 comparisons (BAL-M1
against each baseline):

| Comparison | Observed diff | 95% CI (uncorrected) | 95% CI (Bonferroni n=3) | Significant? |
|---|---|---|---|---|
| BAL-M1 vs M1 | −0.4pp | [−1.6, +0.8] | [−1.9, +1.1] | No |
| BAL-M1 vs M2 | +3.7pp | [+2.3, +5.1] | [+2.0, +5.5] | Yes, BAL-M1 better |
| BAL-M1 vs M3 | −2.8pp | [−4.1, −1.5] | [−4.3, −1.3] | Yes, BAL-M1 worse |

**Among M1/M2/M3/BAL-M1, M3 (reweighting on FairFace's own, unbalanced
training data) has the best race accuracy, not BAL-M1** — and M4 now beats
all four, M3 included (see above). BAL-M1 is statistically
indistinguishable from the plain baseline. This is the third independent
test in this project (after the clean FairFace test and the unaligned
FaceARG ablation) to find that balanced training data alone does not
produce a clearly superior model — the "balanced data wins" headline
cannot be supported. Adversarial debiasing, not balanced training data, is
the method this external benchmark actually favours.

A striking, unexpected result in the same run: **calibration (fit on
FairFace's own validation split) makes race accuracy on FaceARG worse for
three of the four models** (M1 −6.2pp, M3 −6.3pp, BAL-M1 −4.5pp; only M2
gains, +5.2pp). Calibration fit on one distribution does not reliably
transfer to a differently-distributed external benchmark — this is itself
a reportable finding, not just a methodology footnote, and the same
pattern shows up again in Part 4 below with LFW's gender calibration.

### Why BAL-M1 specifically underperforms, with a verified mechanism

BAL-M1 is not uniformly worse — it has the best afro-american recall of
any model by a wide margin (95.9% default) and strong indian recall
(65.2%), but Asian recall collapses to 15.7% (default), the worst result
any model gets on any class in this entire investigation. The cause was
checked directly in HarmoFace-95K's own training manifest
(`data/New Dataset/HarmoFace-95K_v1/metadata/records.csv`):

| Race (HarmoFace train) | from FairFace | from LFW | from UTKFace |
|---|---|---|---|
| White | 5,847 | 3,019 | 3,203 |
| Black | 8,808 | 313 | 2,877 |
| Indian | 9,284 | 128 | 2,676 |
| East Asian | 11,113 | **0** | **0** |
| Southeast Asian | 9,702 | **0** | **0** |
| Latino/Hispanic | 12,021 | **0** | **0** |
| Middle Eastern | 7,404 | **0** | **0** |

East Asian and Southeast Asian training images are 100% FairFace-sourced
— not a sample-count shortage (East Asian has 11,113 images, a healthy
count) but a source-*diversity* shortage. White/Black/Indian all got
exposure to three distinct photographic styles during training (FairFace's
curated crops, LFW's press-photo style, UTKFace's casual-photo style),
which plausibly taught the model source-invariant features for those
groups. Asian faces only ever came from one photographic convention,
repeated over 11,000 times, and that does not transfer to FaceARG's very
different, internet-collected photographic style. **The actionable
finding is specific: HarmoFace-95K balanced sample counts per race class
but did not balance source diversity per class, and the under-diversified
classes are exactly the ones that failed to generalise.** This is a more
defensible and more useful claim for the report than either "balancing
works" or "balancing doesn't work."

## Part 3: Aligned results — UTKFace-trained models

Scored the same way, with calibration fit on UTKFace's own clean
validation subset.

| Model | Race acc. (default) | Race acc. (calibrated) | Gender acc. (calibrated) |
|---|---|---|---|
| M1 | 85.0% | 83.5% | 95.4% |
| M2 | 76.9% | 82.2% | 95.0% |
| M3 | 77.0% | 82.9% | 95.2% |
| **M4 (adversarial debiasing)** | 84.7% | **84.3%** | **95.9%** |

**M4 was scored on FaceARG for the first time in this pass** (excluded
from the original run for the same reason as FairFace's M4). Its default
race accuracy (84.7%) is essentially tied with M1's (85.0%, no CI computed
for this comparison, so "essentially tied" rather than a formal
significance claim), but after calibration M4 has the **best race accuracy
of all four UTKFace-family models** (84.3% vs. M1's 83.5%), and the best
gender accuracy of all four, both before and after calibration. Combined
with Part 2's FairFace result, adversarial debiasing is now the best or
joint-best method on FaceARG for **both** source datasets it was tested
on — a genuinely different picture from UTKFace's own in-distribution test
set, where M4 trailed the plain baseline.

**UTKFace-trained models score 18-25 points higher on FaceARG than the
equivalent FairFace-trained models**, despite UTKFace being the smaller,
less-curated of the two source datasets. The likely explanation: UTKFace
is itself an uncurated, scraped, "in-the-wild" collection, which is
photographically closer to FaceARG's own internet-collected celebrity
photos than FairFace's more controlled curation protocol is. This is a
genuine, citable cross-dataset generalisation finding — the more
carefully-curated dataset is not automatically the one that generalises
best to real-world, uncurated test conditions.

## Part 4: Aligned results — LFW-trained models (gender only)

LFW's M1-M4 have no single categorical race head the way FairFace/UTKFace
do — but LFW's attribute set (LFWA+) does include binary Asian/White/
Black/Indian attributes as part of the same 73-way multi-label head "Male"
belongs to, so LFW is not fully race-blind (that race-attribute signal is
not used in this FaceARG comparison, which sticks to gender, but should
not be described as entirely absent). Only gender is scored here.

| Model | Gender acc. (default) | Gender gap (default) | Gender acc. (calibrated) | Gender gap (calibrated) |
|---|---|---|---|---|
| M1 | 92.7% | 0.005 | 91.0% | **0.204 — far worse** |
| M2 | 92.9% | 0.018 | 92.3% | 0.141 — worse |
| M3 | 92.8% | 0.025 | 91.2% | 0.187 — far worse |
| M4 (adversarial debiasing) | **38.4%** | **1.000 — degenerate** | 68.3% | 0.163 |

All three of M1/M2/M3 are already close to gender-balanced on FaceARG
**before** any calibration (gap 0.005-0.025). Calibration, fit on LFW's
own validation split, makes the gap substantially worse on this external
set in every case. This is the same calibration-transfer-failure pattern
found with FairFace's race calibration in Part 2, now confirmed
independently on a different model family and a different task. Taken
together, the two results say something general: **post-hoc calibration
fit on one dataset's validation split is not safe to assume will transfer
to a genuinely external test distribution, and should be re-validated on
any new distribution before being trusted.**

**M4's result is exactly as unremarkable as expected, and confirms rather
than adds to this project's understanding of LFW's M4 checkpoint.** At the
default threshold it predicts essentially one class for almost every
image (gender gap of 1.000 means one of the two recall terms is at or
near zero) — consistent with an epoch-0 checkpoint saved before the
adversarial mechanism ever ramped up, already documented elsewhere in
this file as not a genuine adversarial-debiasing result. Calibration
recovers some real signal (68.3% accuracy, gap 0.163), but this is still
far below M1/M2/M3's 90%+ default accuracy, and should not be read as
evidence of anything about adversarial debiasing itself — this checkpoint
never trained long enough for that question to be answerable on LFW. With
this result, every model family in this project (FairFace, UTKFace, LFW)
now has an M4 FaceARG data point, completing the set the earlier version
of this file left open.

## Part 5: M4 (adversarial debiasing) — what actually happened, per dataset

M4 was the one mitigation method not yet evaluated anywhere else in this
project. All three per-dataset runs were trained on the Wits cluster with
a gradient-reversal-layer adversarial head predicting ITA (skin tone) from
the shared backbone features, with `grl_lambda` ramped up over training to
gradually increase adversarial pressure.

### UTKFace M4 — trained, unstable late, but has a usable early checkpoint

Training ran to early stopping at epoch 15 (10 epochs with no
val-loss improvement). The loss trajectory destabilises badly from around
epoch 9 onward (train_loss climbing from ~1.75 to ~20.8, val_loss swinging
between 5 and 18 — not a clean convergence). However, the checkpoint
actually *saved* as "best" is from **epoch 5**, before the worst of the
instability, and it evaluates cleanly on UTKFace's own locked test set:

| Metric | Value |
|---|---|
| Race accuracy | 77.3% |
| Gender accuracy | 88.7% |
| Age-group accuracy | 75.1% |
| n (test) | 2,405 (2,404 valid) |

These are legitimate, non-degenerate numbers. **UTKFace's M4 checkpoint is
usable for the report**, with the caveat that it reflects a relatively
early point in training (epoch 5 of an eventual 15), not a fully converged
model, and that the training run as a whole was not stable.

### LFW M4 — trained, but the saved checkpoint is effectively untrained

Training ran to early stopping at epoch 10. The loss trajectory is
catastrophic: epoch 0's val_loss (0.89) was healthy, but the moment
`grl_lambda` started ramping up at epoch 1, val_loss exploded to 169.59
and continued swinging as high as 589.95 (epoch 9) without recovering.
Critically, the checkpoint saved as "best" is from **epoch 0** —
confirmed directly from the evaluation run's own output
(`checkpoint_epoch: 0`) — meaning it was saved *before* the adversarial
mechanism ever engaged at all. It evaluates as:

| Metric | Value |
|---|---|
| Male attribute accuracy | 75.0% |
| Mean attribute accuracy (all 73) | 76.9% |
| n (test) | 1,188 (8 of 1,196 excluded, no LFWA+ attribute match) |

**LFW's M4 is not a genuine adversarial-debiasing result.** It is an
early, essentially untrained snapshot that happens to load and evaluate
without error — the checkpoint is not corrupted (an earlier draft of this
project's evidence pack incorrectly described it that way), but it cannot
be reported as evidence that adversarial debiasing works or fails on LFW,
because the adversarial mechanism never ran long enough to have an effect
on the saved weights.

### FairFace M4 — stopped manually, checkpoint evaluated, no accuracy cost

FairFace's run was stopped manually at epoch 11, after roughly two days on
CPU, because its trajectory had settled into a flat, stable plateau with
no sign of further change — not because the training loop's own
early-stopping criterion fired. Adversarial loss stayed in a tight
0.39-0.50 band across all 12 logged epochs (0-11), adversary accuracy
drifted down only slightly (0.83 → ~0.79), and there was no sign of the
late-training instability seen on UTKFace or the single-epoch explosion
seen on LFW — the cleanest trajectory of the three datasets.

The checkpoint evaluated is from **epoch 9**, and on FairFace's own locked
test set (n=9,758) it scored race 63.0%, gender 90.3%, age-group 53.3% —
beating or tying the plain baseline (M1) on all three tasks, and beating
resampling (M2) and reweighting (M3) on race and gender. **This is the one
dataset where adversarial debiasing did not cost accuracy.** Full figures
and the per-method comparison are in `11_m4_adversarial_debiasing.md`.

An earlier pass at compiling this project's evidence pack reported that
"FairFace's M4 cluster re-attempt also diverged, epoch-0 loss hit
263,852" — that claim was checked directly against this run's live
history file while training was still in progress and did not match it;
now that the run has finished with ordinary, well-behaved numbers, that
claim is confirmed stale rather than merely unconfirmed. It most likely
reflects a stale log from one of the earlier, since-fixed attempts (this
project hit and fixed a genuine NaN bug in FairFace's training pipeline
earlier in the investigation, documented elsewhere in
`report_writing_supporting`). The affected evidence-pack file
(`materials for report/part3_results_evidence.md`) has been corrected.

### Bottom line on M4

Of three per-dataset adversarial-debiasing runs: one (FairFace) produced a
clean checkpoint with no accuracy cost; one (UTKFace) produced a usable,
if early, checkpoint that does cost some accuracy relative to baseline;
one (LFW) produced a checkpoint that is technically valid but represents
essentially no adversarial training at all. **M4 should not be reported
as a working or failed method in the aggregate** — FairFace and UTKFace
alone already disagree on whether the method costs accuracy. The honest
framing is per-dataset, with LFW's result excluded from any claim about
adversarial debiasing's effectiveness, and both FairFace's epoch-9 and
UTKFace's epoch-5 caveats stated plainly wherever their numbers are
quoted.

## Part 6: What ArcFace is, and why M0 has two versions

M0, this project's face-verification model, exists in two variants:
a softmax-loss version (`outputs/M0_LFW/`) and an ArcFace-loss version
(`outputs/M0_LFW_arcface/`). The difference is in how the model is trained
to produce its 512-dimensional embedding, not in its architecture.

A standard softmax classifier is trained to put a decision boundary
between classes, which works well for closed-set classification (the
model only ever has to choose among identities it saw during training),
but it does not explicitly push same-identity embeddings close together or
different-identity embeddings far apart in the embedding space. Face
verification is an open-set problem — the model must compare two
embeddings of people it has never seen during training and decide if they
are the same person — so the quality of the embedding space itself matters
more than where the training-time class boundaries sit.

ArcFace (Additive Angular Margin Loss) changes the loss function to work
directly in angular terms: it normalises both the embedding and the
per-class weight vector onto a hypersphere, then adds a fixed angular
margin penalty to the angle between an embedding and its own class's
weight vector before computing the softmax. In practice, this forces the
model to pull same-identity embeddings measurably closer together in angle
and push different-identity embeddings further apart than plain softmax
training would, because the margin makes the "easy" solution (correct
classification with a small angular gap) no longer sufficient — the model
has to close a bigger angular gap to get the same training signal. The
practical effect reported in the face-recognition literature, and the
reason this project trained a second M0 variant with it, is a more
discriminative, better-separated embedding space for verification
specifically, which should show up as a better FMR/FNMR/EER trade-off and
more reliable genuine-vs-impostor separation than the softmax version, at
the same embedding dimensionality and backbone.

Whether this project's own ArcFace variant actually achieves that
improvement over the softmax version, with real pair-count evidence, is
tracked separately in `02_m0_verification_model.md` and in the evidence
pack's methodology file (`materials for report/part2_methodology_evidence.md`,
item 2) — this section explains what ArcFace is and why it was tried, not
whether it won; that comparison should be read on its own terms, not
assumed from this explanation.

## Open items from this file

- The UTKFace M4 epoch-5 checkpoint's fairness gaps (not just accuracy) have not yet been computed — only the raw task accuracies are reported above.
- FairFace M4's fairness gaps (not just accuracy) have likewise not yet been computed.
- M0's ArcFace-vs-softmax comparison itself (which one actually wins) is not re-derived in this file — see `02_m0_verification_model.md` and the evidence pack for that result directly.
