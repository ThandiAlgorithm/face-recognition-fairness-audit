# Track A — Demographic Attribute Classification Audit

## Evaluation design

FairFace ResNet-34 and DeepFace were evaluated on the same FairFace validation images and the same deterministically aligned UTKFace images. Race taxonomies were harmonised before direct comparison.

LFW contains no independent demographic ground truth in the canonical project table. LFW results therefore describe model-assigned demographic labels, cross-model agreement, and repeated-identity stability rather than demographic accuracy.

## Labelled-dataset results

- FairFace validation gender accuracy: FairFace ResNet-34 94.39%; DeepFace 70.17%.
- FairFace validation six-class race accuracy: FairFace ResNet-34 77.65%; DeepFace 65.75%.
- FairFace validation gender equalized-odds difference: FairFace ResNet-34 7.43 pp; DeepFace 34.03 pp.
- UTKFace gender accuracy: FairFace ResNet-34 94.35%; DeepFace 86.23%.
- UTKFace four-class race accuracy: FairFace ResNet-34 76.48%; DeepFace 75.16%.
- UTKFace gender equalized-odds difference: FairFace ResNet-34 6.18 pp; DeepFace 19.22 pp.

## LFW measurement sensitivity

- Image-level race-label agreement between the two demographic classifiers was 74.71%.
- Image-level gender-label agreement was 95.37%.
- Total-variation distance between the two model-assigned LFW race distributions was 0.0923; for gender it was 0.0360.
- Among repeated LFW identities, modal race assignments agreed between models for 75.00% of identities, while modal gender assignments agreed for 96.49%.

## Interpretation boundary

Differences between model outputs demonstrate measurement sensitivity and performance disparity under the evaluated datasets and preprocessing conditions. They do not establish a causal source of bias, and model-assigned LFW demographic labels must not be treated as verified demographic ground truth.
