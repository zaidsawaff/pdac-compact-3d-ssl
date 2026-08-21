# Compact Self-Supervised 3D Deep Learning for CT-Based Pancreatic Lesion Detection and Segmentation

Reproducibility repository for the study:

**Compact Self-Supervised 3D Deep Learning for CT-Based Pancreatic Lesion Detection and Segmentation**

This repository contains the study-generated source code, frozen protocol
metadata, reproducibility utilities, and documented execution order required
to reconstruct the computational workflow from the original public data sources.

---

## Archived release and DOI

The original archived software release is:

- Version: `v1.0.0`
- Version-specific DOI: `10.5281/zenodo.21998228`
- Concept / all-versions DOI: `10.5281/zenodo.21998227`

`v1.1.0` is the expanded reproducibility release. Its version-specific DOI
should be inserted here after the new Zenodo release is published.

---

## What changed in v1.1.0

The initial v1.0.0 archive emphasized auditability and selected study artifacts.
The expanded v1.1.0 release additionally provides:

- the complete 35-script scientific workflow;
- the frozen Stage 1–3 data-preparation pipeline;
- the definitive Revision R1 training and evaluation pipeline;
- the post-freeze R2 reviewer-analysis script;
- portable project/runtime path configuration;
- workspace initialization;
- automatic retrieval of the exact PANORAMA labels used in the study;
- frozen remote-ZIP metadata for direct CT retrieval from the public archives;
- explicit execution order and legacy-workflow clarification;
- reproducibility smoke-test support.

---

## Repository layout

```text
.
├── README.md
├── docs/
│   └── RUN_ORDER.md
├── scripts/
│   ├── init_workspace.py
│   └── bootstrap_panorama_labels.py
├── src/
│   ├── pipeline/
│   ├── revision_r1/
│   └── revision_r2/
└── protocols/
    ├── 01_Metadata/
    └── 02_Quality_Control/
```

`src/pipeline/` contains the Stage 1–7 development/provenance scripts.

`src/revision_r1/` contains the definitive final revision training and
evaluation workflow.

`src/revision_r2/` contains post-freeze supplementary/reviewer analyses.

`protocols/` contains small study-generated frozen metadata and protocol
artifacts required to reproduce the locked experimental workflow.

---

## Public data sources

### PANORAMA labels

The exact label repository state used by the study is frozen at commit:

```text
bf1d6ba3
```

The bootstrap utility retrieves:

- 482 manual label volumes;
- 1,756 automatic label volumes;
- 2,238 total label volumes.

The labels are retrieved from their original public repository rather than
redistributed in this software release.

### CT volumes

Raw CT data are not redistributed by this repository.

The frozen remote member inventory records the archive, ZIP member, byte
offset, compressed/uncompressed size, compression method, and CRC32 required
to recover each study from the original public Zenodo archives.

The pipeline uses HTTP byte-range requests, so the complete source archives
do not need to be downloaded before processing.

---

## Locked study cohort

The frozen study split contains 2,238 studies:

- Training: 1,376
- Validation: 295
- Internal test: 293
- External MSD partition: 194
- NIH negative stress-test partition: 80

The development cohort used before final test evaluation therefore contains
1,671 studies (`train + validation`).

The locked test partitions are not used during Stage 1A development preprocessing.

---


## Frozen reference artifacts

The `release_assets/` directory is retained in v1.1.0 as a frozen reference
package containing selected checkpoints, prediction manifests, protocol locks,
quality-control audits, and final publication-result summaries produced by the
completed study workflow.

Important examples include:

- the frozen Stage 2A localizer checkpoint;
- the final train-only R1A self-supervised checkpoint;
- the R1B SSL-initialized supervised checkpoint;
- the matched R1B randomly initialized supervised checkpoint;
- the pre-freeze model/threshold lock;
- blind internal-test and external-test prediction manifests;
- final R1 evidence manifests and audits; and
- the frozen publication metric summary.

These artifacts are provided so that researchers can inspect and verify the
reported frozen study state without having to regenerate every trained model
before examining the released evidence.

### Authoritative executable source

For v1.1.0, the authoritative executable scientific source code is located in:

```text
src/pipeline/
src/revision_r1/
src/revision_r2/
scripts/
```

The small historical code subset under:

```text
release_assets/05_Code/
```

is retained only as part of the earlier frozen reference package. It is **not**
the authoritative source tree for the expanded v1.1.0 reproducibility release.

The definitive execution sequence is documented in:

```text
docs/RUN_ORDER.md
```

### Data and licensing boundary

`release_assets/` does not contain raw CT volumes or raw medical segmentation
datasets. Third-party medical datasets remain subject to their original access,
licensing, and attribution terms and are not relicensed under this repository's
MIT license.

See `THIRD_PARTY_DATA_NOTICE.md` for the applicable data-source and licensing
boundary.

## Environment

Python 3.12 was used for the archived environment audit.

Install the repository dependencies before running the pipeline:

```bash
python -m pip install -r requirements.txt
```

GPU acceleration is required for the practical full training workflow.
CPU execution is suitable for metadata checks and limited smoke tests but is
not intended for full model training.

---

## Configure paths

The code no longer requires one fixed Google Drive location.

Set:

```bash
export PDAC_PROJECT_ROOT="/absolute/path/to/PDAC_Public_Q1_Project"
export PDAC_RUNTIME_ROOT="/absolute/path/to/runtime"
```

`PDAC_PROJECT_ROOT` stores persistent outputs.

`PDAC_RUNTIME_ROOT` stores temporary caches and scratch files.

For Google Colab, for example:

```bash
export PDAC_PROJECT_ROOT="/content/drive/MyDrive/PDAC_Public_Q1_Project"
export PDAC_RUNTIME_ROOT="/content"
```

---

## Step 1 — Initialize a clean workspace

From the repository root:

```bash
python scripts/init_workspace.py \
  --project-root "$PDAC_PROJECT_ROOT" \
  --repro-root .
```

This creates the required directory structure and installs the frozen
study-generated prerequisite metadata into the new workspace.

---

## Step 2 — Retrieve the exact labels

```bash
python scripts/bootstrap_panorama_labels.py \
  --project-root "$PDAC_PROJECT_ROOT"
```

Expected output:

```text
Manual labels      : 482
Automatic labels   : 1756
Total labels       : 2238
STATUS             : PASS
```

---

## Step 3 — Optional one-case reproducibility smoke test

Before processing the complete development cohort, Stage 1A can be tested
on one case:

```bash
PDAC_MAX_CASES=1 python src/pipeline/stage1a_localizer_preprocessing.py
```

A successful smoke test should:

- retrieve one CT member from the public remote archive;
- verify source integrity;
- match CT/annotation geometry;
- generate a `(96, 96, 160)` CT array;
- store CT values as `int16`;
- use the locked `[-200, 300] HU` range;
- create a durable preprocessing ledger;
- access zero locked-test cases.

`PDAC_MAX_CASES` is for testing only and must not be used for the complete
reproduction run.

---

## Step 4 — Full reproduction workflow

The complete execution order is documented in:

**`docs/RUN_ORDER.md`**

The definitive final manuscript workflow is:

```text
Stage 1A
  -> Stage 1B
  -> Stage 2A
  -> Stage 2B
  -> Stage 2C
  -> Stage 2D frozen geometry protocol
  -> Stage 2E
  -> Stage 3A
  -> Stage 3B
  -> R1A
  -> R1B
  -> R1C
  -> R1D
  -> R1E
  -> R1F / R1G
  -> R1H / R1I
  -> R1J
  -> R2A
```

---

## Important distinction: final vs historical workflow

The repository preserves the earlier Stage 4–7 scripts for provenance.

They must not be interpreted as the definitive final experimental sequence.

In the final revision:

- R1A performs train-only masked-context self-supervised pretraining;
- R1B performs the leakage-free SSL-vs-random supervised comparison;
- R1B initializes the SSL arm from `stageR1a_ssl_final.pt`;
- Stage 5A is retained only as a historical frozen protocol dependency for
  locked optimization parameters;
- R1E freezes model selection and the deployment threshold before blind tests;
- R1F–R1I perform the locked internal/external inference and evaluation;
- R1J creates the final revision evidence freeze;
- R2A performs post-freeze analyses without retraining or changing the
  deployment threshold.

---

## Reproducibility validation performed for v1.1.0 staging

The following checks were performed while preparing this expanded release:

- all 35 Python scripts passed syntax validation;
- no remaining machine-specific filesystem paths were detected;
- the clean workspace initializer reproduced all 21 frozen prerequisite files;
- PANORAMA label bootstrap returned exactly 482 manual and 1,756 automatic labels;
- no downloaded label was a Git-LFS pointer;
- all four remote CT archives returned valid HTTP 206 byte-range responses;
- a CT study was recovered directly from the remote ZIP inventory;
- recovered CT size matched the frozen uncompressed size;
- recovered CT CRC32 matched the frozen CRC32;
- the recovered NIfTI volume was readable;
- isolated Stage 1A processing successfully produced one valid
  `(96, 96, 160)` `int16` CT sample;
- the Stage 1A smoke test accessed zero locked-test cases.

These checks validate repository portability, public-source reconstruction,
and the initial preprocessing path. They do not claim that every full
training stage was freshly rerun during preparation of this software release.

---

## Resumability

Several computationally expensive stages were designed to be resumable.
When a resumable stage reports that the session is incomplete, rerun the
same command. Existing durable outputs and ledgers are checked before
remaining cases are processed.

---

## Reproducibility boundaries

This repository does not redistribute raw third-party medical imaging data.

It provides the study-generated software and frozen metadata needed to
reconstruct the experiment from the original public sources.

Third-party datasets remain governed by their original licenses, citations,
and distribution terms.

The MIT license in this repository applies to study-generated software and
does not supersede third-party dataset licenses.

---

## Citation

For the archived v1.0.0 software release:

```text
https://doi.org/10.5281/zenodo.21998228
```

For the expanded v1.1.0 release, use its version-specific DOI after the
new Zenodo archive has been published.

---

## Reproducibility support files

- `docs/RUN_ORDER.md` — authoritative stage-by-stage execution order
- `scripts/init_workspace.py` — clean workspace construction
- `scripts/bootstrap_panorama_labels.py` — exact public-label retrieval
- `protocols/` — frozen study-generated metadata and protocol locks
- `src/pipeline/` — Stage 1–7 provenance/development code
- `src/revision_r1/` — definitive R1 workflow
- `src/revision_r2/` — post-freeze analyses
