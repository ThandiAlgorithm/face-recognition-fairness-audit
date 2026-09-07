# Algorithmic Fairness in Computer Vision

## Overview

Face analysis systems do not perform equally well for everyone. This project
audits a face attribute classification pipeline for performance disparities
across race, gender, age and skin tone, including intersectional subgroups,
and then tests whether four mitigation strategies actually close those gaps
and at what cost.

The audit is framed using a six-type bias taxonomy, which lets each measured
disparity be attributed to a specific point in the pipeline rather than
reported as a single undifferentiated accuracy gap.

## Objectives

1. Quantify subgroup performance disparities in a face attribute classifier
   evaluated on LFW.
2. Attribute each disparity to a source in the pipeline: representation,
   measurement, aggregation or evaluation.
3. Compare two skin tone measurement instruments and report their agreement.
4. Test four mitigation strategies at three intervention points and measure
   the accuracy cost of each.
5. Report cost per percentage point of fairness improvement.

## Datasets

| Dataset  | Role                  |
|----------|-----------------------|
| LFW      | Primary audit target  |
| FairFace | Training source       |
| UTKFace  | Independent test set  |

No images are stored in this repository. Only annotation index files and
numerical results are committed. Datasets must be obtained from their
original sources.

## Method

Four training conditions, each run with three seeds:

- **A** Skew-matched baseline
- **B** Balanced re-sampling
- **C** Adversarial de-biasing
- **D** Maximum mean discrepancy penalty

Equalised odds is the primary fairness criterion, with demographic parity and
equality of opportunity reported alongside it.

## Repository structure

    configs/       experiment configuration files
    src/           pipeline source code
    annotations/   label index files, no images
    results/       numerical results
    docs/          design decisions, model cards, meeting minutes
    data/          gitignored, never committed

Every experiment is driven by a configuration file. A run that cannot be
reproduced from a config file plus a seed does not count as a result.

## Status

Under active development.

## Authors

- Thandeka Primrose Malasa

- Ruvimbo Nicole Zimbodza


