# Reproducibility Environment

This file records the environment observed when the public reproducibility
bundle was audited in Google Colab.

## Tested environment

- Python: 3.12.13
- OS: Linux x86_64, glibc 2.35
- NumPy: 2.0.2
- pandas: 2.2.2
- SciPy: 1.16.3
- scikit-learn: 1.6.1
- PyTorch: 2.11.0+cpu
- CUDA available in this audit session: No
- cuDNN available in this audit session: No

`requirements.txt` pins the public Python package versions. The local Colab
build reported PyTorch as `2.11.0+cpu`; the requirements file uses
`torch==2.11.0` for standard package resolution.

The CPU-only status above describes the audit session in which the environment
was recorded. It should not be interpreted as a claim about the hardware used
for every experiment in the study.
