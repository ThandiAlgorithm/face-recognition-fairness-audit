# Phase 1 audit — standalone close-out package

This package runs the reduced Phase 1 audit with **no manual image review**. It contains the supporting measurement/config/path/image-loading modules and YAML files it needs, so it can run as a standalone project once you provide your real `master_annotations` table and image files.

## Install

```bash
python -m pip install -r requirements.txt
```

## Normal workflow now: MEA → VIS → SHIFT

LFW is intentionally **optional** until the real LFW data and pair lists are available.

```bash
python run_phase1_audit.py --workers 4
```

The default command runs only:

```text
MEA → VIS → SHIFT
```

Run selected core stages if needed:

```bash
python run_phase1_audit.py --stages mea vis
python run_phase1_audit.py --stages shift
```

When LFW becomes available, run it separately:

```bash
python run_phase1_audit.py --stages lfw
```

or append it to a full run:

```bash
python run_phase1_audit.py --include-lfw --workers 4
```

If `--stages lfw` is run before any configured pair file exists, the stage **does not crash**. It writes `LFW_status.json`, records why it was skipped, registers those outputs, and exits cleanly.

## Real input required

Set this in `configs/phase1_audit.json`:

```json
"master_table": "metadata/master_annotations.parquet"
```

CSV also works. The master table must contain unique `image_id` values and a `dataset` column. MEA/VIS expect `aligned_path` for image measurement.

### MEA eligibility

MEA is now independent from VIS:

1. If `eligible_measurement_audit` exists, MEA uses it.
2. If the flag does not exist, MEA processes **every row with a non-empty `aligned_path`**.
3. Excluded MEA rows receive `status = not_eligible_measurement_audit`.

VIS still uses `eligible_visual_audit` when that column exists; otherwise it processes all master rows.

## Included standalone support files

The package now includes:

```text
src/measurements.py
src/categories.py
src/config.py
src/image_integrity.py
src/paths.py
configs/preprocessing.yaml
configs/categories.yaml
requirements.txt
```

`preprocessing.yaml` freezes the simple skin ROI/filter settings. `categories.yaml` contains ITA group definitions and the configured binary ITA threshold. Review these configuration values with your project protocol before the final confirmatory run; do not change them after looking at subgroup results without recording a protocol amendment.

## Outputs

Outputs go to:

```text
outputs/phase1/
├── measurement_audit/
│   ├── tables/
│   ├── figures/
│   ├── metadata/
│   └── qa/
├── visual_audit/
│   ├── tables/
│   ├── figures/
│   ├── metadata/
│   └── qa/
├── shift_audit/
│   ├── tables/
│   ├── figures/
│   ├── metadata/
│   └── qa/
├── lfw_audit/             # only when requested
└── result_registry.csv
```

The registry now scans the complete output directory for each completed stage. It therefore includes **CSV, Parquet, PNG figures, PNG QA/contact sheets, JSON metadata/config snapshots, markdown limitation drafts, checkpoint files and warning files**, each with SHA-256 and its generating module/script.

## Important measurement rules

- MEA uses the frozen relative ROI patches in `configs/preprocessing.yaml` and stores continuous `L*`, `a*`, `b*`, ITA, hue and chroma.
- The ITA/Fitzpatrick-like grouping is derived only after the continuous measurement. It is not treated as ground-truth skin type.
- Greyscale images are flagged rather than given chromatic skin measurements.
- VIS measures brightness, contrast, illumination asymmetry, sharpness, face size and approximate pose where the required fields exist.
- Alignment padding is excluded when `alignment_transform`, `width_px`, and `height_px` are available.
- Missing bbox/landmarks result in missing geometry/pose rather than fabricated values.
- No manual appearance labels, medical-status inference, or human image review are required.

## LFW rules

`configs/phase1_audit.json` may contain future LFW pair paths even while the files do not exist. The default run ignores the LFW stage entirely. When you later run LFW explicitly, absent pair files produce a clean `skipped` status rather than a failure.

View 2 remains sealed while:

```json
"allow_view2_metadata": false
```

Do not change that setting unless your project protocol/supervisor allows it.

## Tests

Run:

```bash
python -m pytest -q
```

The tests check statistics, ITA/hue, pose recovery, padding masks, duplicate handling, LFW parsing/yoking, end-to-end MEA/VIS/SHIFT/LFW behavior, separate MEA eligibility, optional-LFW skipping, and registry coverage of PNG/JSON/QA outputs.

## Main files

| File | Purpose |
|---|---|
| `src/measurement_audit.py` | MEA extraction, validity, reliability and subgroup outputs |
| `src/visual_audit.py` | VIS measurements, QC, plots and subgroup outputs |
| `src/shift_audit.py` | Cross-dataset representation/measurement shift |
| `src/lfw_audit.py` | Optional official-pair analysis when LFW is available |
| `src/stats.py` | CI, tests, effect sizes, BH correction |
| `src/phase1_common.py` | tables, checkpoints, masks, dedup, output registry |
| `configs/phase1_audit.json` | audit settings |
| `configs/preprocessing.yaml` | frozen MEA ROI/filter settings |
| `configs/categories.yaml` | ITA grouping definitions |

## First real run

Use your real files only. Do not create placeholder LFW data. Start with:

```bash
python run_phase1_audit.py --stages mea --workers 4
python run_phase1_audit.py --stages vis --workers 4
python run_phase1_audit.py --stages shift
```

Check each stage's `status_counts`, tables, figures and metadata before moving to the next stage.
