# M4 — Adversarial Debiasing: Full Results, Per Dataset

M4 is the fourth mitigation method tried in this project, alongside M2
(resampling) and M3 (reweighting): a gradient-reversal-layer (GRL)
adversarial head is attached to the shared backbone, trained to predict
ITA (skin tone) from the same features the main task heads use, with its
gradient reversed before it reaches the backbone. As `grl_lambda` ramps up
over training, the backbone is pushed to produce features the adversary
finds progressively harder to use for guessing skin tone, while the main
task heads keep training normally — the idea being a backbone that can
still do race/gender/age classification well without leaning on skin-tone
information to do it.

One run was trained per dataset (FairFace, UTKFace, LFW), all seed 42,
same cluster, same architecture pattern as M1/M2/M3 with the extra
adversarial head. This file is the single place that holds the full,
per-dataset result — it was previously only summarised inline inside
`09_faceARG_external_benchmark.md`/`10_faceARG_aligned_results_and_m4_status.md`.

![M4 training trajectories, all three datasets](m4_training_trajectories.png)

The chart above is the one figure that explains everything that follows.
FairFace and UTKFace both start with a stable, low, flat adversarial loss.
LFW's loss explodes within a single epoch of `grl_lambda` starting to ramp
up and never recovers. The star markers show where each run's "best"
checkpoint was actually saved — UTKFace's lands at epoch 5, well before
its own later instability (visible as the slow climb in UTKFace's loss
curve from epoch ~9 onward); LFW's lands at epoch 0, before the
adversarial mechanism had done anything at all.

## FairFace — stopped manually, never destabilised, and the checkpoint is genuinely usable

FairFace's M4 run was never early-stopped by the training loop itself — it
was stopped manually at epoch 11 after running for roughly two days on
CPU, because the trajectory had settled into a flat, stable plateau with
no sign it would improve further: adversary loss stayed in a tight
0.39-0.50 band across all 12 logged epochs (0-11), with adversary accuracy
drifting down only slightly from 0.83 to around 0.79-0.79 and never
diverging. This is the cleanest trajectory of the three datasets — neither
UTKFace's late-training instability nor LFW's single-epoch loss explosion
shows up here at all.

The evaluated checkpoint is from **epoch 9** (`checkpoint_epoch` in
`test_metrics.json`), and it produces legitimate, non-degenerate numbers on
FairFace's own locked test set (n=9,758):

| Task | Accuracy |
|---|---|
| Race | 63.0% |
| Gender | 90.3% |
| Age group | 53.3% |

![FairFace: overall accuracy, all four methods](m4_fairface_overall_accuracy_comparison.png)

Against the other three FairFace methods, M4 is the **best performer on
two of three tasks**: race accuracy (63.0%) edges out the plain baseline M1
(62.6%) and reweighting M3 (61.1%), and comfortably beats resampling M2
(52.1%); gender accuracy (90.3%) is the best of all four methods by a clear
margin (M1 87.4%, M2 85.4%, M3 88.2%). Age-group accuracy (53.3%) is
essentially tied with M1 (53.4%), both ahead of M2 and M3. **This is the
one dataset where adversarial debiasing did not cost accuracy to get its
intended effect** — a materially different outcome from UTKFace (where M4
sits below the baseline) and LFW (where M4's checkpoint is unusable). This
should be stated plainly if M4 is discussed as a single cross-dataset
method, since FairFace alone does not support the "adversarial debiasing
trades accuracy for fairness" framing that UTKFace's result might suggest
in isolation.

![FairFace: race recall by class, all four methods](m4_fairface_race_recall_comparison.png)

![FairFace M4 race confusion matrix](m4_fairface_confusion_race.png)
![FairFace M4 gender confusion matrix](m4_fairface_confusion_gender.png)
![FairFace M4 age-group confusion matrix](m4_fairface_confusion_age_group.png)

**An earlier pass at this project's evidence pack
(`materials for report/part3_results_evidence.md`) reported that this run
"also diverged, epoch-0 loss hit 263,852"** — that claim was checked
directly against this run's own live history file while training was
still in progress and did not match it; now that the run has finished and
been evaluated with ordinary, well-behaved numbers, that claim is
confirmed stale rather than merely unconfirmed. It most likely reflects a
log from an earlier, aborted attempt at this training run, before a
genuine NaN bug in the training pipeline (a stale label-vocabulary config
file, unrelated to the adversarial mechanism) was found and fixed earlier
in this project. That evidence-pack file has been corrected accordingly.

## UTKFace — trained, destabilises late, but the saved checkpoint is usable

Early-stopped at epoch 15 (10 epochs with no validation improvement). The
loss trajectory is clean through roughly epoch 8, then destabilises:
train_loss climbs from ~1.75 to ~20.8 and val_loss swings between 5 and 18
rather than converging. The checkpoint saved as "best," however, is from
**epoch 5** — before the worst of this instability — and it produces
legitimate, non-degenerate numbers on UTKFace's own locked test set:

| Task | Accuracy | n (valid) |
|---|---|---|
| Race | 77.3% | 2,404 |
| Gender | 88.7% | 2,404 |
| Age group | 75.1% | 2,404 |

![UTKFace: overall accuracy, all four methods](m4_utkface_overall_accuracy_comparison.png)

Against the other three UTKFace methods: M4's race accuracy (77.3%) sits
in the middle of the pack — above M2 (76.1%) and essentially tied with M3
(77.3%, coincidentally identical to 13 decimal places; the two
checkpoints are confirmed to be different files by hash, so this is a
genuine coincidence in total correct-count, not a data error) — but below
the plain baseline M1 (80.2%). M4's gender accuracy (88.7%) and age
accuracy (75.1%) are both the lowest or near-lowest of the four methods.

![UTKFace: race recall by class, all four methods](m4_utkface_race_recall_comparison.png)

![UTKFace M4 race confusion matrix](m4_utkface_confusion_race.png)
![UTKFace M4 gender confusion matrix](m4_utkface_confusion_gender.png)
![UTKFace M4 age-group confusion matrix](m4_utkface_confusion_age_group.png)

**UTKFace's M4 checkpoint is usable for the report**, with two caveats
stated plainly wherever it is quoted: it reflects an early point in
training (epoch 5 of an eventual 15, not a fully converged run), and the
training run as a whole was not stable, so a different random seed or a
slightly different early-stopping patience could plausibly have landed on
a meaningfully different checkpoint.

## LFW — trained, but the saved checkpoint is effectively untrained

Early-stopped at epoch 10. The loss trajectory is the most severe
instability found anywhere in this project: val_loss was healthy at epoch
0 (0.89), exploded the moment `grl_lambda` began ramping at epoch 1
(169.59), and continued climbing as high as 589.95 by epoch 9, never
recovering. The checkpoint saved as "best" is from **epoch 0** — confirmed
directly from the evaluation run's own output (`checkpoint_epoch: 0`),
meaning the saved weights predate the adversarial mechanism engaging at
all.

| Metric | Value |
|---|---|
| Male attribute accuracy | 75.0% |
| Mean attribute accuracy (all 73) | 76.9% |
| n (test) | 1,188 (8 of 1,196 excluded, no LFWA+ attribute match) |

![LFW: gender accuracy, all four methods](m4_lfw_gender_accuracy_comparison.png)

The gap is stark: M1/M2/M3 all sit at 92.8-94.0% male-attribute accuracy;
M4 sits at 75.0%, nearly 18 points lower. **This is not evidence that
adversarial debiasing fails on LFW** — it is evidence that this particular
run's early-stopping mechanism saved a checkpoint from before the method
ever had a chance to run. The checkpoint itself is not corrupted and loads
and evaluates without error (an earlier pass at this project's evidence
pack incorrectly described it as corrupted); it simply does not represent
a trained adversarial-debiasing model, and should not be cited as a result
for or against the method on LFW.

## Bottom line

Three per-dataset M4 runs, three genuinely different outcomes: one clean
and usable with no accuracy cost (FairFace), one usable-with-caveats
(UTKFace), one uninterpretable because it never really trained (LFW).
**M4 cannot be reported as a single aggregate result across datasets** —
FairFace and UTKFace alone already disagree on whether adversarial
debiasing costs accuracy, and LFW's checkpoint cannot be cited at all. Any
claim about adversarial debiasing's effectiveness in this project has to
be stated per dataset, with FairFace's epoch-9 and UTKFace's epoch-5
caveats both named explicitly, and LFW excluded from any comparison rather
than quoted alongside the other two as if all three were comparable.

## Reproducing these figures

All figures in this file were generated by
`face_resnet34_training/src/evaluation/run_deep_dive_m4.py`, run from
`face_resnet34_training/` with the `face-demo-app/backend/venv` Python
environment. It reads each dataset's `adversarial_history.csv` for the
trajectory plot, and each method's `test_eval/` folder (`test_metrics.json`,
`metrics.json` where it exists, `test_predictions.parquet` as a fallback
for M4 specifically, since M4 was evaluated with the simpler `evaluate.py`
pass rather than the fuller pipeline M1/M2/M3 went through) for the
comparison bars and confusion matrices. FairFace's M4 checkpoint and test
evaluation were added once its training run was stopped (epoch 11) and
evaluated (epoch-9 checkpoint); all three datasets' comparison bars are
now present in this file.
