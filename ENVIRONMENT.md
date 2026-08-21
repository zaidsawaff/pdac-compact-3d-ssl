# Computational Environment

This file documents the software environment audited while preparing
the expanded reproducibility release (`v1.1.0`).

## Audited Python environment

```text
Python   : 3.12.13
Compiler : GCC 11.4.0
Platform : Linux 6.6.122+, x86_64, glibc 2.35
```

## Direct Python dependencies used by the scientific code

```text
matplotlib==3.10.0
nibabel==5.4.2
numpy==2.0.2
pandas==2.2.3
requests==2.32.4
scipy==1.16.3
scikit-learn==1.6.1
torch==2.11.0
```

These dependencies were obtained by statically inspecting all 35
Python scripts and resolving the imported third-party modules.

No required third-party import was missing during the audit.

## PyTorch and GPU note

The repository-preparation audit session reported:

```text
PyTorch       : 2.11.0+cpu
CUDA available: False
```

The `+cpu` suffix describes the PyTorch build installed in the audit
session; it is not a requirement to reproduce full model training.

Full Stage 2A, R1A, and R1B training is computationally intensive and
should be performed with a CUDA-capable GPU and a PyTorch 2.11.0 build
compatible with the host CUDA environment.

Because CUDA wheels depend on operating system, driver, and CUDA runtime,
`requirements.txt` records `torch==2.11.0` rather than hard-coding the
CPU-only local version identifier `torch==2.11.0+cpu`.

## Installation

Create a clean Python 3.12 environment and install:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For GPU training, ensure that the installed PyTorch build recognizes
the intended CUDA device before beginning computationally expensive stages.

Example check:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

## Path configuration

The expanded release uses environment variables rather than requiring
one machine-specific Google Drive path:

```bash
export PDAC_PROJECT_ROOT="/absolute/path/to/PDAC_Public_Q1_Project"
export PDAC_RUNTIME_ROOT="/absolute/path/to/runtime"
```

`PDAC_PROJECT_ROOT` is persistent study storage.

`PDAC_RUNTIME_ROOT` is temporary/cache storage.

## Reproducibility scope

The environment audit confirms that all direct third-party imports used
by the released scientific scripts were importable in the audited session.

It does not imply that the complete model-training workflow was freshly
rerun in the CPU-only repository-preparation session.
