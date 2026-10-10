# Harmonized Diversity-Aware Face Dataset

Version: 1.0

## Overview

This dataset is a harmonized, diversity-aware combination of
FairFace, UTKFace and Labeled Faces in the Wild (LFW).

The final manifest contains **95,258 valid image records**.

The source images are not altered. The final dataset is represented
through a reproducible manifest that preserves the location and
provenance of every image.

## Final demographic taxonomy

The dataset uses seven explicit demographic categories:

1. White
2. Black
3. Indian
4. East Asian
5. Southeast Asian
6. Middle Eastern
7. Latino

## Final composition

- White: 14,981 (15.73%)
- Black: 14,986 (15.73%)
- Indian: 14,989 (15.74%)
- East Asian: 13,837 (14.53%)
- Southeast Asian: 12,210 (12.82%)
- Middle Eastern: 9,265 (9.73%)
- Latino: 14,990 (15.74%)


The largest-to-smallest demographic group ratio is
**1.62:1**.

## Construction principles

The dataset was created using the following principles:

- Source images were not relabelled through visual inspection.
- Original source labels and provenance are preserved.
- No minority demographic group was duplicated to create balance.
- Large groups were selectively reduced rather than minority
  groups being repeatedly oversampled.
- A 35/65 gender representation floor was applied where required.
- Skin tone was analysed separately using ITA rather than being
  treated as equivalent to demographic category.
- Image brightness, contrast and sharpness were audited separately.
- Records with invalid or missing image files were excluded from
  the final manifest.
- Known LFW identities were checked to prevent duplication across
  dataset partitions during development.

## Source-label harmonization

FairFace labels were retained at their available level of detail.

UTKFace and LFW records labelled only as broad `Asian` were not
forced into East Asian or Southeast Asian categories and therefore
were not included in this seven-class final product.

`Other` and `Unknown` records were also preserved during development
but excluded from this explicit-label final dataset.

## Important interpretation

These demographic labels are dataset annotations and should not be
treated as objective biological categories.

Race, ethnicity, geographic population and measured skin tone are
not interchangeable concepts.

The dataset is intended for research into representation, dataset
bias and computer-vision fairness.

## Files

- `harmonized_face_dataset_manifest.csv`
  - canonical final dataset manifest

- `labels/`
  - one CSV manifest for each demographic category

- `demographic_summary.csv`
  - final demographic counts and percentages

- `source_summary.csv`
  - contribution of each source dataset

- `gender_summary.csv`
  - gender composition by demographic category

- `ita_summary.csv`
  - ITA composition by demographic category

- `previews/`
  - visual sample sheets for inspection

- `label_schema.json`
  - machine-readable taxonomy definition

## Provenance

Source datasets:

- FairFace
- UTKFace
- Labeled Faces in the Wild (LFW)

The harmonized manifest preserves source dataset information for
every record.

## Limitations

Some final demographic categories originate predominantly or
entirely from FairFace because the other source datasets do not
provide equivalent fine-grained labels.

Gender labels use the binary categories available in the source
metadata and should not be interpreted as a complete representation
of gender identity.

ITA measurements have missing values for a small proportion of
records and are affected by imaging and illumination conditions.

The dataset should therefore be described as **harmonized,
diversity-aware and representation-balanced**, rather than as a
perfectly fair or universally representative dataset.
