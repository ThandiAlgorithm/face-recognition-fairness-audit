# Track A Qualitative Example Pack

Selection strategy: `audit`. Target sample size: 30 images per dataset.

The same selected image is shown with both FairFace ResNet-34 and DeepFace demographic predictions. FairFace and UTKFace panels also show dataset ground-truth labels.

LFW panels are prediction-only because the canonical project master table contains no independent LFW demographic ground truth. The displayed race and gender values are model-assigned labels, not verified attributes.

These examples are intended for qualitative inspection and error analysis. They do not replace the full-dataset quantitative metrics.

## Files

- Each dataset folder contains three contact sheets of ten examples.
- Each `individual/` folder contains one labelled PNG per selected image.
- Each dataset contains `sample_manifest.csv`.
- `all_samples_manifest.csv` combines all datasets.
