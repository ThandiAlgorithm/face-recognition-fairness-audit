# Requested fixes implemented

1. **Standalone dependencies added**
   - `src/measurements.py`
   - `src/categories.py`
   - `src/config.py`
   - `src/image_integrity.py`
   - `src/paths.py`
   - `configs/preprocessing.yaml`
   - `configs/categories.yaml`
   - `requirements.txt`

2. **MEA eligibility separated from VIS**
   - Uses `eligible_measurement_audit` when present.
   - Otherwise processes every row with a non-empty aligned crop path.
   - Uses status `not_eligible_measurement_audit` for explicit exclusions.

3. **LFW made optional**
   - Default runner stages are now `mea vis shift`.
   - Use `--stages lfw` later, or `--include-lfw` for a full run.
   - Missing pair files create a registered skip-status JSON instead of crashing.

4. **Registry expanded to all module outputs**
   - Scans the complete module output tree after each stage.
   - Registers tables, Parquet copies/warnings, PNG figures, QA/contact sheets, JSON metadata/config snapshots, markdown drafts and checkpoints with SHA-256.
