# Skin tone: ITA used in place of Fitzpatrick

The project brief specifies skin tone using the **Fitzpatrick scale** (a
6-category clinical classification originally for sunburn/cancer risk,
typically assigned by visual inspection or self-report). This project's
data pipeline does not have Fitzpatrick annotations for any of the three
datasets, and none were available to add at the time of training.

Instead, every skin-tone measurement in this project uses **ITA
(Individual Typology Angle)**: a continuous value computed directly from
each image's own pixels (via CIELAB L*/b* values measured from a skin-ROI
in Phase 0 preprocessing), thresholded at the frozen project value of 28°
into a binary `ita_group` (`<28` / `≥28`). ITA and Fitzpatrick both
describe skin tone but are not the same measurement -- one is
image-derived and continuous, the other is categorical and typically
assigned by a human rater or self-report. They are not silently treated as
interchangeable anywhere in this codebase: every column, config key, and
report label says `ita_value` / `ita_group`, never "Fitzpatrick".

**Why ITA instead of Fitzpatrick here:**
- It's already computed, frozen, and validated from Phase 0 for all three
  datasets (100% consistent with the project's own frozen `skin_binary`
  threshold in `configs/categories.yaml`).
- It's objective and reproducible (derived from the image itself), with no
  annotator subjectivity.
- No Fitzpatrick-labeled resource was identified for LFW, FairFace, or
  UTKFace within this project's timeline.

**For brief compliance**: this file is the methodology note explaining the
substitution, per explicit instruction. If a Fitzpatrick-labeled resource
is identified later, the plan is to add it as its own column
(`fitzpatrick_category`) alongside -- not in place of -- `ita_value` /
`ita_group`, so both are reportable where valid, matching the brief's own
suggested table shape (Fitzpatrick category + ITA value + ITA group).
Nothing currently in this codebase needs to change to accommodate that
addition later; it would be a pure column addition to the existing
records/predictions tables.
