# Limitations draft

Running notes on methodology limitations, to be incorporated into the final report. Each entry records the limitation and which decision/section it traces back to.

## LFW: original vs. deep-funneled

LFW is processed from the original unaligned images so that the detection and alignment stage can be audited. Most published LFW results use the deep-funneled variant, so accuracy figures here are not directly comparable to those in the literature.

**Source:** Task 2 (data-preparation pipeline), 2026-09-15. Original archive downloaded via a Figshare mirror after the official `vis-www.cs.umass.edu` host failed to resolve from this network on both 2026-09-14 and 2026-09-15; checksum verified against the official MD5 documented in `torchvision.datasets.lfw` (see `metadata/acquisition_manifest.csv`).

## LFW: pre-filtered by Viola-Jones, in either variant

LFW was constructed by retaining only images in which the Viola-Jones detector found a face. Detection failure rates measured on LFW are therefore conditioned on prior successful detection and cannot be read as detector performance. Switching to the original unaligned variant removes a second, deep-funneling alignment pass but does not remove this filtering. The cross-dataset detection comparison is therefore reported on FairFace and UTKFace, which were not filtered this way; LFW detection results are reported separately with this caveat attached.

**Source:** Task 2 code review (data-preparation pipeline), 2026-09-15.

## Section 12 pilot briefly used SCRFD, not RetinaFace (superseded)

GOV-006 specifies `model_ref: retinaface`. For a short while the Section 12 pilot ran on InsightFace's `buffalo_sc` pack instead, whose detector is SCRFD, not RetinaFace. This was a practical substitution while the environment was on Python 3.14 (no TensorFlow/PyTorch wheels available at all). Once the venv was rebuilt on Python 3.11 and the user installed the real `retina-face` package by hand, the pilot switched over to actual RetinaFace. This entry is kept for the record; see `configs/GOV-008_models.yaml` checkpoint `detection_pilot_scrfd_superseded`.

## RetinaFace only runs on genuinely raw images

FairFace and UTKFace are both already publisher-cropped to tight face-only crops (confirmed in section 1). RetinaFace is trained on natural scene photos and barely detects anything on these crops even at very low confidence thresholds, while it detects LFW's original (raw) photos cleanly at 0.99+ confidence. Rather than force FairFace/UTKFace through the detector by dropping the threshold until something appears (which would just be a second, undocumented preprocessing stage stacked on the publisher's own crop), the pilot only runs RetinaFace on LFW. FairFace and UTKFace get a supplied-crop validation pass instead and go straight into feature engineering on the crop as given.

## RetinaFace model-build bug found and fixed

`retina-face` needs the legacy `tf.keras` behaviour, but TensorFlow 2.16+ defaults to Keras 3. If tensorflow gets imported anywhere before that's sorted out, model building fails with a `KerasTensor cannot be used as input to a TensorFlow function` error, on every single call, not just sometimes. First found this because `get_model_info()` did a plain `import tensorflow as tf` to log the version, before the first real detection call.

The proper fix: `os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")` at the very top of `src/detection.py`, before tensorflow can be imported by anything, plus a `_check_retinaface_runtime()` guard that raises loudly if tensorflow somehow got imported first anyway (only fix at that point is restarting the kernel). Package versions get read via `importlib.metadata` rather than a real import, since importing tensorflow just to check its version is the exact mistake that breaks everything. Worth remembering if this resurfaces on the Wits cluster with a different TensorFlow version.

**Source:** Section 12 pilot (data-preparation pipeline), 2026-09-17.

## image_id format and master table path drifted from GOV-005 (open, needs a decision)

`docs/GOV-005_data_dictionary.md` specifies `image_id = "<DATASET>-" + sha256(dataset_version + "|" + original_path)[:16]`, written to a single `metadata/master_annotations.parquet`. The implementation in `src/master_table.py` (`stable_image_id`) instead hashes `dataset::relative_path` with plain sha256, no dataset prefix, no truncation, and writes three files (`data/processed/master_annotations_base.parquet`, `_pilot.parquet`, `_features.parquet`) rather than one. `dataset_version` as named in GOV-005 does not exist anywhere in the codebase; it was never built as a real manifest field.

Neither GOV-005 nor GOV-006 has a filled-in `frozen_date`, so nothing has been broken that was actually locked, but the two disagree and one needs to change. Not resolved yet; flagged here rather than fixed silently in either direction, since 134,639 already-generated `image_id` values would change if the code moves to match the doc.

**Source:** Reconciliation check requested by user, 2026-09-18.
