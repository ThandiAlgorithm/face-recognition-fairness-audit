# Software Environment

**Source issue:** GOV-008  
**Version:** 1.2.0  
**Status:** Approved; local base environment captured; HPC runtime and model-specific checkpoints must be validated before their first bulk use  
**Approval date:** 2026-09-17  
**Approved by:** RZ and TM  
**Freeze date:**   2026-09-17 
**Freeze commit:** PENDING_SHARED_REPO_FREEZE_COMMIT

## Why this exists

A random seed alone does not make a result reproducible. Outputs can change across library releases, CUDA versions, operating systems and hardware.

The specific risk for this project: `skimage.color.rgb2lab` governs every
CIELAB value, therefore every ITA value, therefore every skin tone group,
therefore every subgroup in H1, H2 and H3. A silent library upgrade would move subgroup boundaries with no error raised.

## Environment lock

The environment is captured in **one** exact lock file. Loose specifiers such
as `torch>=2` are not acceptable.

Choose one and record which:

- `environment.yml` with every version pinned, or
- `conda-lock.yml`, or
- `requirements-lock.txt` produced by `pip freeze` in a clean environment.

Regenerate with:

```bash
python -V > docs/env/python_version.txt
pip freeze > docs/env/requirements-lock.txt
python -c "import torch; print(torch.__version__, torch.version.cuda)" > docs/env/torch.txt
nvidia-smi > docs/env/gpu.txt 2>/dev/null || echo "no gpu" > docs/env/gpu.txt
```

Then copy the resulting versions into `configs/models.yaml` so they sit
alongside the checkpoints they govern.

## Python version

Starting recommendation: **Python 3.11.x**, subject to the Wits environment
and every required face library supporting it.

This issue does not close until the exact patch version is written. If the
available environment supports only another stable version, use it and record
it **before** bulk processing. That is an environment constraint, not a
post-hoc change.

## Seeds

Project seed `26041`, applied to Python `random`, NumPy, PyTorch CPU and
PyTorch CUDA. Any analysis requiring several seeds lists them explicitly rather
than deriving them at run time.

## Determinism

Where supported: cuDNN benchmark selection off for reproducibility runs, and
deterministic PyTorch algorithms enabled. Any operation that cannot run
deterministically is recorded in `configs/models.yaml` with its reason.

Bit-for-bit equality across different hardware is not claimed unless it has
been tested.

## Execution environments

### Local Windows development environment

The local Windows virtual environment is the project's development and
reproducibility baseline.

Python and all installed package versions are recorded in:

- `docs/env/python_version.txt`
- `docs/env/requirements-lock.txt`
- `docs/env/system.txt`
- `docs/env/gpu.txt`
- `docs/env/torch.txt`

The captured local PyTorch build is CPU-only. The installed NVIDIA GPU and
driver are retained as hardware provenance but are not treated as the
canonical accelerator for the local environment.

### Wits EIE compute cluster

The project has access to the Wits EIE compute cluster for heavier processing.

Infrastructure information supplied by the School reports:

- 20 CPU compute nodes;
- Intel Core i7-3770 @ 3.40 GHz on the CPU nodes;
- 32 GB RAM per CPU node;
- one GPU node;
- two Intel Xeon E5-2620 v3 @ 2.40 GHz CPUs on the GPU node;
- 128 GB RAM on the GPU node;
- NVIDIA Tesla K40c GPU with 12 GB memory;
- Ubuntu 20.04.3;
- Python is available.

These are infrastructure-reported specifications, not a measured software
environment.

Before the HPC cluster is used for a scientific bulk run, the exact Python
version, package lock, CUDA/runtime information, driver information and node identifier must be captured on the cluster. A smoke test must pass before that execution profile is accepted for scientific results.

Results produced locally and on the HPC must record which execution profile was used.

## Checkpoint provenance

Every checkpoint carries name, backbone, source, download date, file name,
SHA-256, claimed training dataset, licence, input size, input normalisation
and threshold where applicable.

**Never record a bare name.** "ArcFace" and "FaceNet" each denote many
checkpoints trained on different corpora. Since the project states LFW
identity overlap with those training corpora as an unresolved limitation, the
claim cannot even be investigated without knowing which checkpoint was used.

## Smoke test before freeze

Twenty images spanning LFW, FairFace and UTKFace, checking decode, EXIF
handling, colour conversion, detection, alignment, skin measurement and table write. Then rerun and diff the outputs. Only then is the environment frozen.

## Hardware log

Every substantial run records node name where permitted, GPU model and count,
CUDA runtime, CPU model, operating system, and peak GPU memory where practical.

## What this cannot prove

Pinned software does not make a model unbiased. It makes a result reproducible
and therefore checkable.


## Closure rule

The governance rule is complete, but GOV-008 is not marked Done until the exact runtime and every model/checkpoint actually used in Phase 1 are recorded and the 20-image smoke test passes.
