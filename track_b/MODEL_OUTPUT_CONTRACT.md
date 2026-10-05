# Track B Model Output Contract

## Purpose

This document defines the required handoff between the model-training track and the common Track B verification/fairness evaluation pipeline.

The same interface applies to:

- M0 — unmitigated project baseline
- M1 — data-level mitigation
- M2 — loss/sample reweighting
- M3 — representation/adversarial mitigation

R0 ArcFace is a pretrained reference and is evaluated separately using the same verification protocol.

---

## 1. Common Evaluation Images

Every M0–M3 model must generate embeddings for the exact same 4,992 unique images required by LFW View 1.

The source verification pairs are defined by:

`results/track_b/arcface_reference/lfw_view1_pairs_resolved.csv`

The recognition input must be the frozen project-aligned image associated with each LFW image through:

`metadata/master_annotations.parquet`

using:

`aligned_path`

No model-specific redetection or realignment should be introduced during evaluation.

---

## 2. Required Model Output

Each model must provide one embedding for every required LFW View 1 image.

Preferred structure:

```text
results/track_b/<MODEL_ID>/
└── embeddings/
    ├── <MODEL_ID>_view1_embeddings.npz
    ├── <MODEL_ID>_embedding_manifest.csv
    └── <MODEL_ID>_embedding_metadata.json
```

Example:

```text
results/track_b/M0/
└── embeddings/
    ├── M0_view1_embeddings.npz
    ├── M0_embedding_manifest.csv
    └── M0_embedding_metadata.json
```

---

## 3. NPZ Requirements

The NPZ file must contain:

### `embeddings`

A two-dimensional numeric array:

```text
shape = (4992, embedding_dimension)
```

Each row represents one LFW image.

Embeddings should be finite numeric values.

The common evaluator will L2-normalize embeddings before cosine similarity so that the comparison rule remains consistent across models.

### `raw_image_paths`

A string array containing the original LFW path corresponding to each embedding.

Required length:

```text
4992
```

If available, the NPZ may additionally contain:

### `aligned_image_paths`

The frozen project-aligned input path used to generate each embedding.

---

## 4. Manifest Requirements

The embedding manifest must contain one row per embedding.

Required columns:

```text
embedding_index
raw_image_path
aligned_image_path
model_id
embedding_dimension
```

`embedding_index` must correspond exactly to the row index in the NPZ embedding matrix.

Every required View 1 image must appear exactly once.

There must be:

- no missing required images
- no duplicate images
- no extra images

---

## 5. Metadata Requirements

The metadata JSON should record at least:

```text
model_id
model_role
architecture
embedding_dimension
training_dataset
mitigation_method
checkpoint_identifier
random_seed
input_preprocessing
```

Recommended additional fields:

```text
identity_loss
optimizer
learning_rate
training_epochs
batch_size
training_split
model_selection_rule
```

This information is required for reproducibility and the final investigation report.

---

## 6. Model Roles

### M0

Ordinary project-trained facial recognition model.

No fairness-specific mitigation.

M0 is the primary baseline against which M1–M3 are compared.

### M1

Same core recognition system with data-level mitigation.

Examples may include:

- balanced sampling
- demographic resampling
- balanced mini-batches

The final implemented method must be documented precisely.

### M2

Same core recognition system with loss/sample weighting mitigation.

The final weighting rule must be documented precisely.

### M3

Same core recognition system with representation-level/adversarial mitigation.

The final adversarial objective and sensitive attribute used must be documented precisely.

---

## 7. Controlled Comparison Requirement

Where feasible, M0–M3 should retain the same:

- recognition architecture
- embedding dimension
- identity objective
- training/evaluation image preprocessing
- LFW evaluation protocol
- scoring method

The primary intended experimental change between models is the fairness mitigation strategy.

This supports attribution of observed performance/fairness changes to the mitigation rather than unrelated architecture changes.

---

## 8. Evaluation Ownership

The training track must not select the final LFW verification threshold.

For every model, the common evaluation pipeline will:

1. generate or load the 4,992 View 1 embeddings
2. L2-normalize embeddings
3. compute cosine similarity for all 3,200 View 1 pairs
4. use View 1 train only to select the model's global threshold
5. freeze that threshold
6. evaluate View 1 test
7. calculate demographic subgroup metrics
8. calculate bootstrap confidence intervals
9. compare M1/M2/M3 against M0 using paired tests

Each model may therefore obtain its own global development threshold, but no model may use demographic-specific thresholds.

---

## 9. View 2 Rule

LFW View 2 remains sealed during:

- model training
- hyperparameter tuning
- mitigation selection
- preprocessing selection
- threshold selection
- model comparison

View 2 is reserved for final reporting after the experimental pipeline is frozen.

---

## 10. Acceptance Checks

A model output is ready for evaluation only when:

- 4,992 embeddings are present
- every required LFW View 1 image resolves
- no required image is missing
- no image is duplicated
- embedding dimensions are consistent
- all embedding values are finite
- the aligned project input was used
- model metadata is present
- View 2 was not used

Only after these checks pass should verification scoring begin.
