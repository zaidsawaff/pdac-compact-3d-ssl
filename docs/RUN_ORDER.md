# PDAC Reproducibility Run Order

This document defines the computational workflow corresponding to the final revised study:

**Compact Self-Supervised 3D Deep Learning for CT-Based Pancreatic Lesion Detection and Segmentation**

## 1. Runtime configuration

Set two environment variables before running the pipeline:

```bash
export PDAC_PROJECT_ROOT="/absolute/path/to/PDAC_Public_Q1_Project"
export PDAC_RUNTIME_ROOT="/absolute/path/to/runtime"
```

`PDAC_PROJECT_ROOT` stores persistent study outputs.
`PDAC_RUNTIME_ROOT` stores temporary/cache data.

## 2. Initialize workspace

```bash
python scripts/init_workspace.py --project-root "$PDAC_PROJECT_ROOT" --repro-root .
```

This installs the frozen study-generated protocol and metadata files.

## 3. Retrieve PANORAMA labels

```bash
python scripts/bootstrap_panorama_labels.py --project-root "$PDAC_PROJECT_ROOT"
```

Frozen PANORAMA label commit: `bf1d6ba3`.

Expected counts:

- Manual labels: 482
- Automatic labels: 1756
- Total: 2238

PANORAMA CT volumes are not redistributed. Required CT members are retrieved
from the public Zenodo archives using the frozen remote ZIP inventory and
HTTP byte-range access.

# Part I — Frozen data-preparation workflow

## Stage 1A — Localizer preprocessing

```bash
python src/pipeline/stage1a_localizer_preprocessing.py
```

Resumable preprocessing of the development cohort only.

Optional one-case smoke test:

```bash
PDAC_MAX_CASES=1 python src/pipeline/stage1a_localizer_preprocessing.py
```

Do not use `PDAC_MAX_CASES` for the full reproduction run.

## Stage 1B — Freeze preprocessing

```bash
python src/pipeline/stage1b_localizer_preprocessing_freeze.py
```

## Stage 2A — Lightweight 3D localizer training

```bash
python src/pipeline/stage2a_lightweight_3d_localizer_training.py
```

## Stage 2B — Validation crop-coverage audit

```bash
python src/pipeline/stage2b_validation_crop_coverage_audit.py
```

## Stage 2C — Targeted R4 failure voxel audit

```bash
python src/pipeline/stage2c_targeted_r4_failure_voxel_audit.py
```

Stage 2D is a frozen geometry decision, not a separate training script:

`01_Metadata/stage2d_final_deployment_crop_geometry_protocol.json`

## Stage 2E — Development localizer inference

```bash
python src/pipeline/stage2e_all_development_localizer_inference.py
```

## Stage 3A — E5 crop preprocessing

```bash
python src/pipeline/stage3a_e5_resumable_crop_preprocessing.py
```

## Stage 3B — E5 QC and dataset freeze

```bash
python src/pipeline/stage3b_e5_full_qc_and_dataset_freeze.py
```

# Part II — Definitive Revision R1 workflow

## R1A — Train-only masked-context SSL

```bash
python src/revision_r1/stageR1a_train_only_masked_context_ssl.py
```

SSL pretraining is restricted to the training partition.

## R1B — Leakage-free dual-arm supervised training

```bash
python src/revision_r1/stageR1b_leakage_free_dual_arm_supervised_training.py
```

R1B compares the SSL-initialized and random-initialization arms.

The SSL arm is initialized from the R1A checkpoint:

`04_Models/Revision_R1/StageR1A_TrainOnlyMaskedContext3DCNN/stageR1a_ssl_final.pt`

Stage 5A is a historical frozen protocol dependency used by R1B for
locked optimization settings, including batch size 4 and gradient
accumulation 1. Stage 5A is not the source of the final R1 SSL weights.

## R1C — Full validation comparison

```bash
python src/revision_r1/stageR1c_leakage_free_full_validation_comparison.py
```

## R1D — Validation FROC calibration

```bash
python src/revision_r1/stageR1d_leakage_free_validation_froc_calibration.py
```

## R1E — Pre-freeze model/threshold evidence lock

```bash
python src/revision_r1/stageR1e_prefreeze_model_threshold_evidence_lock.py
```

Model and operating threshold are frozen before blind test evaluation.

## R1F — Blind internal-test inference

```bash
python src/revision_r1/stageR1f_i_resumable_blind_internal_test_inference.py
```

## R1G — Post-freeze internal-test evaluation

```bash
python src/revision_r1/stageR1g_postfreeze_internal_test_evaluation.py
```

## R1H — Blind external-test inference

```bash
python src/revision_r1/stageR1h_resumable_blind_external_test_inference.py
```

## R1I — Source-separated external evaluation

```bash
python src/revision_r1/stageR1i_postfreeze_source_separated_external_evaluation.py
```

## R1J — Final revision evidence freeze

```bash
python src/revision_r1/stageR1j_final_revision_evidence_freeze.py
```

# Part III — Post-freeze analyses

## R2A — Reviewer/post-freeze analyses

```bash
python src/revision_r2/stageR2a_postfreeze_reviewer_analyses.py
```

R2A does not retrain the model or modify the frozen operating point.

# Definitive workflow

```text
Stage1A -> Stage1B
        -> Stage2A -> Stage2B -> Stage2C
        -> Stage2D frozen geometry protocol
        -> Stage2E
        -> Stage3A -> Stage3B
        -> R1A -> R1B -> R1C -> R1D -> R1E
        -> R1F -> R1G
        -> R1H -> R1I
        -> R1J -> R2A
```

# Historical / legacy workflow

The repository also preserves the earlier Stage 4–7 development scripts
for provenance and auditability.

These scripts must not be interpreted as the definitive final R1
experimental sequence.

In particular:

- Stage 4A is the earlier SSL implementation.
- Stage 5A remains a frozen historical protocol dependency.
- Final R1B SSL initialization comes from R1A.
- Final blind evaluation is performed through R1F–R1I.
- R1J is the final revision evidence freeze.

# Reproducibility boundaries

Raw third-party medical imaging data are not redistributed.

The repository provides study-generated code, frozen splits, remote archive
inventory, protocol locks, QC evidence, bootstrap utilities, and the
training/evaluation workflow.

Third-party datasets remain governed by their original licenses and
distribution terms.
