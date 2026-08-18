# Compact Self-Supervised 3D Deep Learning for CT-Based Pancreatic Lesion Detection and Segmentation

Reproducibility repository for the study **“Compact Self-Supervised 3D Deep
Learning for CT-Based Pancreatic Lesion Detection and Segmentation.”**

The project evaluates a compact 3D CT pipeline for pancreatic ductal
adenocarcinoma (PDAC) analysis. The pipeline combines a lightweight spatial
localizer, train-only masked-context self-supervised learning (SSL), and a
dual-head pancreas/lesion prediction network under patient-grouped development
and source-separated held-out evaluation.

## Archived release and DOI

The reproducibility materials corresponding to the frozen public release `v1.0.0` are archived on Zenodo:

**Version-specific DOI:** https://doi.org/10.5281/zenodo.21998228

**All-versions DOI:** https://doi.org/10.5281/zenodo.21998227

For exact reproducibility of the study, cite the version-specific DOI above.

## Repository scope

This repository is intended to expose study-generated reproducibility material,
including:

- analysis and training code used in the final supervised/SSL comparison;
- locked split and preprocessing protocols;
- quality-control and audit records;
- validation/model-selection locks;
- prediction-freeze manifests;
- held-out evaluation summaries;
- selected model checkpoints;
- SHA-256 checksums and release inventories.

Raw CT volumes and source segmentation masks are intentionally **not**
redistributed in this repository. They originate from third-party public
datasets and should be obtained from their original repositories under the
applicable terms.

## Study cohorts

The audited study cohort contains 2,238 CT studies from 2,224 unique patients:
676 PDAC and 1,562 non-PDAC studies.

The PANORAMA-local development cohort was patient-grouped into:

- 1,376 training studies;
- 295 validation studies;
- 293 held-out internal-test studies.

Source-separated held-out cohorts contain:

- 194 Medical Segmentation Decathlon (MSD) studies;
- 80 NIH negative-only studies.

The external/source-separated cohorts were not used for weight updates,
architecture selection, preprocessing tuning, or threshold calibration.

## Public data sources

The study used the official PANORAMA public training/development release,
including four Zenodo batch records:

- Batch 1 v3: https://doi.org/10.5281/zenodo.13715870
- Batch 2 v2: https://doi.org/10.5281/zenodo.13742336
- Batch 3: https://doi.org/10.5281/zenodo.11034011
- Batch 4 v1.0: https://doi.org/10.5281/zenodo.10999754

Annotations and clinical metadata were obtained from the official
`DIAGNijmegen/panorama_labels` repository at commit `bf1d6ba3`.

Additional source-separated data derive from:

- Medical Segmentation Decathlon;
- NIH Pancreas-CT collection, DOI: https://doi.org/10.7937/K9/TCIA.2016.tNB1kqBU

Users should obtain all source imaging data directly from the original
providers rather than from this repository.

## Environment

Tested audit environment:

- Python 3.12.13
- NumPy 2.0.2
- pandas 2.2.2
- SciPy 1.16.3
- scikit-learn 1.6.1
- PyTorch 2.11.0+cpu

Install the Python dependencies with:

```bash
python -m pip install -r requirements.txt
```

See `ENVIRONMENT.md` for the recorded audit environment.

## Reproducibility bundle

The release exporter builds a `release_assets/` directory and produces:

- `release_asset_manifest.json`
- `release_asset_inventory.csv`
- `SHA256SUMS.txt`
- `MISSING_ASSETS.txt`

The exporter verifies known checkpoint hashes and records missing or ambiguous
assets rather than silently ignoring them.

To build the release bundle from the original project tree:

```bash
python export_release_assets.py
```

The default project root expected by the exporter is:

```text
/content/drive/MyDrive/PDAC_Public_Q1_Project
```

## Integrity verification

From the root of the generated `release_assets/` directory:

```bash
sha256sum -c SHA256SUMS.txt
```

The audited bundle contained no known checkpoint hash mismatches.

## Repository structure

```text
.
├── README.md
├── ENVIRONMENT.md
├── requirements.txt
├── export_release_assets.py
└── release_assets/
    ├── 01_Metadata/
    ├── 02_Quality_Control/
    ├── 03_Results/
    ├── 04_Models/
    ├── 05_Code/
    ├── release_asset_manifest.json
    ├── release_asset_inventory.csv
    ├── SHA256SUMS.txt
    └── MISSING_ASSETS.txt
```

## Reproducibility boundaries

This repository supports auditability and partial computational
reproducibility of the reported workflow. It does not redistribute the
third-party CT volumes or source masks, and it should not be interpreted as a
stand-alone clinical product. Prospective and truly independent clinical
validation remain necessary before clinical use.

## Versioning and archival

For publication, the GitHub repository should be frozen as a versioned release
(e.g. `v1.0.0`) and archived in a DOI-minting repository such as Zenodo. The
resulting DOI should be inserted into the manuscript's Code Availability
statement and used to cite the exact release associated with the paper.

## Licence

Original study-generated software authored for this repository is released
under the MIT License; see `LICENSE`.

The MIT License does **not** apply to third-party CT data, source masks,
annotations, clinical metadata, or dataset-derived material whose reuse is
governed by the original data providers.

PANORAMA-derived material remains subject to the applicable PANORAMA terms,
including CC BY-NC 4.0 where applicable.

See `THIRD_PARTY_DATA_NOTICE.md` for details.
