# Why this project trains on CPU, not the cluster's GPU

The Wits EIE compute cluster provides one GPU node (`dica10`, reached via
the `gpu-node` SSH alias, outside the Slurm-managed CPU cluster): an
NVIDIA Tesla K40c, 12GB, confirmed directly against the department's own
hardware announcement. This GPU cannot run any current PyTorch build.
Training in this project is CPU-only by necessity, not by choice.

## The finding

- `torch.cuda.is_available()` returns `True` once a CUDA-11.3-matched
  PyTorch build (`1.12.1+cu113`) is installed (the driver, 470.256.02, caps
  out at CUDA 11.4 -- PyPI's default `torch==2.3.1` wheel targets CUDA 12.1
  and fails to even initialize CUDA on this driver).
- But any real GPU operation -- a plain matrix multiply, not just model
  training -- fails:
  ```
  RuntimeError: CUDA error: no kernel image is available for execution on the device
  ```
- Root cause: the K40c is a Kepler-generation GPU, **compute capability
  3.5**. PyTorch's prebuilt binaries have not shipped compiled kernels for
  capability below 3.7 since around PyTorch 1.3. `is_available()` only
  confirms the driver initializes -- it does not confirm any kernel exists
  for that architecture. No PyTorch version pin fixes this; it is not a
  version-compatibility problem, it is a hardware-support cutoff.

## What would fix it, and why it isn't being attempted here

Compiling PyTorch from source with `TORCH_CUDA_ARCH_LIST=3.5` explicitly
included is the only way to get a working GPU build for this card. This
was deliberately not attempted:
- Multi-hour compile time, with no guarantee of success -- PyTorch's own
  C++/CUDA source may have dropped Kepler support at the code level by the
  versions this project otherwise wants to use, not only in prebuilt
  wheels, in which case the build fails outright regardless of time spent.
- The shared NFS home directory on this cluster is independently unreliable
  for large sustained writes (observed: a 2GB wheel extraction hung
  indefinitely in kernel `D`-state, uninterruptible even by `SIGKILL`,
  waiting on an NFS RPC that never returned) -- a multi-hour source build's
  I/O would hit the same failure mode.
- Under a two-day training budget, this is not a sensible risk/time trade.

## What this means for execution

Training runs on CPU across three machines in parallel instead:
- `dica10` (12 cores) -- FairFace, the largest/slowest dataset.
- jaguar cluster nodes (4 cores each) -- UTKFace and LFW, one per node.

AMP (`torch.cuda.amp.autocast`/`GradScaler`) is consequently a no-op in
every run here -- it's wired up and will activate automatically if this
code is ever run somewhere with a working GPU, but note that AMP's major
speedups come from Tensor Cores (Volta and newer); the K40c predates Tensor
Cores entirely, so even a working CUDA build of this GPU would not have
benefited much from AMP.

Confirmed empirically 2026-10-02/03, `train-baseline-classifiers` branch.
