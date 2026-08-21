from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os

import nibabel as nib
import numpy as np
import pandas as pd


# =============================================================================
# LOCKED INPUTS AND OUTPUTS
# =============================================================================

PROJECT_ROOT = Path(os.environ.get("PDAC_PROJECT_ROOT", "/content/drive/MyDrive/PDAC_Public_Q1_Project"))
META_DIR = PROJECT_ROOT / "01_Metadata"
QC_DIR = PROJECT_ROOT / "02_Quality_Control"
PROCESSED_DIR = (
    PROJECT_ROOT / "03_Processed" / "E5_240x192x128"
)

STAGE3A_LEDGER_PATH = (
    QC_DIR / "stage3a_e5_crop_preprocessing_ledger.csv"
)
STAGE3A_PROTOCOL_PATH = (
    META_DIR / "stage3a_e5_crop_preprocessing_protocol.json"
)
STAGE2E_PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "04_Models"
    / "Localizer"
    / "F1_96x96x160"
    / "stage2e_all_development_localizer_predictions.csv"
)
STAGE2D_CROP_PROTOCOL_PATH = (
    META_DIR / "stage2d_final_deployment_crop_geometry_protocol.json"
)

RESUME_QC_LEDGER_PATH = (
    QC_DIR / "stage3b_e5_full_qc_resume_ledger.csv"
)
CANONICAL_QC_PATH = (
    QC_DIR / "stage3b_e5_full_qc.csv"
)
FROZEN_MANIFEST_PATH = (
    META_DIR / "stage3b_e5_frozen_dataset_manifest.csv"
)
COVERAGE_SUMMARY_PATH = (
    QC_DIR / "stage3b_e5_crop_coverage_summary.csv"
)
FREEZE_PROTOCOL_PATH = (
    META_DIR / "stage3b_e5_dataset_freeze_protocol.json"
)
AUDIT_PATH = (
    QC_DIR / "stage3b_e5_dataset_freeze_audit.json"
)


# =============================================================================
# LOCKED ACCEPTANCE POLICY
# =============================================================================

EXPECTED_CASES = 1671
EXPECTED_TRAIN = 1376
EXPECTED_VALIDATION = 295
EXPECTED_PDAC = 491
EXPECTED_NON_PDAC = 1180
EXPECTED_GEOMETRY_COUNTS = {
    "EXACT_PHYSICAL_GEOMETRY_MATCH": 1659,
    "PHYSICAL_GEOMETRY_MATCH_WITHIN_NUMERICAL_HEADER_TOLERANCE": 10,
    "KNOWN_VISUALLY_CONFIRMED_INDEX_ALIGNED_HEADER_MISMATCH": 2,
}

TARGET_SHAPE = (240, 192, 128)
TARGET_SPACING_MM = np.asarray([1.25, 1.25, 2.0])
HU_MIN = -200
HU_MAX = 300
ALLOWED_LABELS = set(range(7))
MINIMUM_ACCEPTED_VOXEL_COVERAGE = 0.99

EXPECTED_KEYS = {
    "ct_hu",
    "mask_labels",
    "crop_affine",
    "crop_spacing_mm",
    "predicted_center_canonical",
    "predicted_center_world_mm",
    "study_id",
    "patient_id",
    "partition",
    "diagnostic_label",
    "annotation_type",
    "geometry_status",
}


# =============================================================================
# UTILITIES
# =============================================================================


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_write_csv(dataframe, path):
    temporary = Path(str(path) + ".tmp")
    dataframe.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_write_json(data, path):
    temporary = Path(str(path) + ".tmp")
    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def normalize_study_id(series):
    return series.astype(str).str.strip()


def truth_flags(series):
    return (
        series.fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes"})
    )


def scalar_text(array):
    return str(np.asarray(array).item()).strip()


def replace_row(frame, new_row):
    study_id = str(new_row["study_id"])
    if len(frame):
        frame = frame.loc[
            frame["study_id"].astype(str) != study_id
        ].copy()
    return pd.concat(
        [frame, pd.DataFrame([new_row])],
        ignore_index=True,
    )


def safe_fraction(numerator, denominator, empty_value):
    denominator = int(denominator)
    if denominator == 0:
        return float(empty_value)
    return float(int(numerator) / denominator)


def percentile(values, q):
    values = np.asarray(values, dtype=float)
    return float(np.percentile(values, q))


# =============================================================================
# INPUT AND INVENTORY LOCKS
# =============================================================================


print("=" * 116)
print("STAGE 3B — FULL E5 PREPROCESSING QC AND DATASET FREEZE")
print("=" * 116)

for required_path in [
    STAGE3A_LEDGER_PATH,
    STAGE3A_PROTOCOL_PATH,
    STAGE2E_PREDICTIONS_PATH,
    STAGE2D_CROP_PROTOCOL_PATH,
]:
    if not required_path.exists():
        raise FileNotFoundError(f"Required input missing:\n{required_path}")

with open(STAGE3A_PROTOCOL_PATH, "r", encoding="utf-8") as file:
    stage3a_protocol = json.load(file)
with open(
    STAGE2D_CROP_PROTOCOL_PATH,
    "r",
    encoding="utf-8",
) as file:
    stage2d_protocol = json.load(file)

if stage3a_protocol.get("crop_candidate") != "E5_240x192x128":
    raise RuntimeError("Stage 3A E5 protocol was not found.")
if stage2d_protocol.get("selected_candidate") != "E5_240x192x128":
    raise RuntimeError("Stage 2D E5 geometry is not locked.")
if stage3a_protocol.get("locked_test_cases_accessed") != 0:
    raise RuntimeError("Stage 3A reports locked-test access.")

stage3a = pd.read_csv(STAGE3A_LEDGER_PATH)
predictions = pd.read_csv(STAGE2E_PREDICTIONS_PATH)
stage3a["study_id"] = normalize_study_id(stage3a["study_id"])
predictions["study_id"] = normalize_study_id(
    predictions["study_id"]
)

required_stage3a_columns = {
    "study_id",
    "patient_id",
    "partition",
    "diagnostic_label",
    "annotation_type",
    "geometry_status",
    "derived_mask_reheader_applied",
    "output_path",
    "output_size_bytes",
    "output_sha256",
    "processing_complete",
    "full_lesion_voxels",
    "contained_source_lesion_voxels",
    "resampled_crop_lesion_voxels",
    "lesion_voxel_coverage",
    "full_pancreas_voxels",
    "contained_source_pancreas_voxels",
    "resampled_crop_pancreas_voxels",
    "pancreas_voxel_coverage",
    "full_union_voxels",
    "contained_source_union_voxels",
    "resampled_crop_union_voxels",
    "union_voxel_coverage",
}
missing_columns = sorted(
    required_stage3a_columns - set(stage3a.columns)
)
if missing_columns:
    raise RuntimeError(
        f"Stage 3A ledger missing columns: {missing_columns}"
    )

if len(stage3a) != EXPECTED_CASES:
    raise RuntimeError(
        f"Stage 3A ledger has {len(stage3a)} rows, expected 1671."
    )
if stage3a["study_id"].nunique() != EXPECTED_CASES:
    raise RuntimeError("Stage 3A study IDs are not unique.")
if not truth_flags(stage3a["processing_complete"]).all():
    raise RuntimeError("Stage 3A contains incomplete cases.")
if stage3a["partition"].value_counts().to_dict() != {
    "train": EXPECTED_TRAIN,
    "validation": EXPECTED_VALIDATION,
}:
    raise RuntimeError("Stage 3A partition counts changed.")
if set(stage3a["partition"]) != {"train", "validation"}:
    raise RuntimeError("A locked test case entered the E5 dataset.")
if len(predictions) != EXPECTED_CASES:
    raise RuntimeError("Stage 2E prediction count changed.")
if predictions["study_id"].nunique() != EXPECTED_CASES:
    raise RuntimeError("Stage 2E prediction IDs are not unique.")
if set(predictions["study_id"]) != set(stage3a["study_id"]):
    raise RuntimeError("Stage 2E and Stage 3A IDs differ.")

expected_files = {
    f"{study_id}.npz" for study_id in stage3a["study_id"]
}
observed_files = {
    path.name for path in PROCESSED_DIR.glob("*.npz")
}
partial_files = list(PROCESSED_DIR.glob("*.part"))

if observed_files != expected_files:
    missing = sorted(expected_files - observed_files)
    unexpected = sorted(observed_files - expected_files)
    raise RuntimeError(
        "Persistent E5 inventory mismatch: "
        f"missing={len(missing)}, unexpected={len(unexpected)}"
    )
if partial_files:
    raise RuntimeError(
        f"Partial E5 outputs remain: {len(partial_files)}"
    )

if RESUME_QC_LEDGER_PATH.exists():
    resume_qc = pd.read_csv(RESUME_QC_LEDGER_PATH)
    if len(resume_qc):
        resume_qc["study_id"] = normalize_study_id(
            resume_qc["study_id"]
        )
        resume_qc = resume_qc.drop_duplicates(
            subset=["study_id"],
            keep="last",
        )
else:
    resume_qc = pd.DataFrame()

stage3a_index = stage3a.set_index("study_id", drop=False)
prediction_index = predictions.set_index("study_id", drop=False)

completed_ids = set()
if len(resume_qc) and "qc_complete" in resume_qc.columns:
    completed_rows = resume_qc.loc[
        truth_flags(resume_qc["qc_complete"])
        & truth_flags(resume_qc["all_checks_pass"])
    ]
    for _, row in completed_rows.iterrows():
        study_id = str(row["study_id"])
        if study_id not in stage3a_index.index:
            continue
        source_row = stage3a_index.loc[study_id]
        path = Path(str(source_row["output_path"]))
        if (
            path.exists()
            and path.stat().st_size
            == int(source_row["output_size_bytes"])
            and str(row["output_mtime_ns"])
            == f"ns:{path.stat().st_mtime_ns}"
            and str(row["verified_sha256"])
            == str(source_row["output_sha256"])
        ):
            completed_ids.add(study_id)

pending = stage3a.loc[
    ~stage3a["study_id"].isin(completed_ids)
].sort_values(["partition", "study_id"]).reset_index(drop=True)

persistent_size = sum(
    path.stat().st_size for path in PROCESSED_DIR.glob("*.npz")
)

print()
print(f"Expected E5 cases: {EXPECTED_CASES}")
print(f"Previously QC-completed: {len(completed_ids)}")
print(f"Pending before this run: {len(pending)}")
print(f"Persistent NPZ files: {len(observed_files)}")
print(
    f"Persistent dataset size: "
    f"{persistent_size / (1024 ** 3):.3f} GiB"
)
print("Locked test cases accessed: 0")
print("Checkpoint frequency: after every case")


# =============================================================================
# RESUMABLE FULL FILE QC
# =============================================================================


qc_this_run = 0
errors = []

for order, (_, source_row) in enumerate(pending.iterrows(), start=1):
    study_id = str(source_row["study_id"])
    path = Path(str(source_row["output_path"]))
    prediction_row = prediction_index.loc[study_id]

    if order == 1 or order % 50 == 0 or order == len(pending):
        print(
            f"QC progress: {order}/{len(pending)} — {study_id} "
            f"— durable {len(completed_ids)}/{EXPECTED_CASES}"
        )

    try:
        if not path.exists():
            raise FileNotFoundError(f"E5 output missing: {path}")

        observed_size = path.stat().st_size
        expected_size = int(source_row["output_size_bytes"])
        size_matches = observed_size == expected_size
        if not size_matches:
            raise RuntimeError(
                f"Output size mismatch: {observed_size}/{expected_size}"
            )

        observed_hash = sha256_file(path)
        hash_matches = (
            observed_hash == str(source_row["output_sha256"])
        )
        if not hash_matches:
            raise RuntimeError("Output SHA-256 mismatch.")

        with np.load(path, allow_pickle=False) as data:
            key_set_exact = set(data.files) == EXPECTED_KEYS
            if not key_set_exact:
                raise RuntimeError(
                    "NPZ key set mismatch: "
                    f"{sorted(set(data.files) ^ EXPECTED_KEYS)}"
                )

            ct = np.asarray(data["ct_hu"])
            mask = np.asarray(data["mask_labels"])
            crop_affine = np.asarray(
                data["crop_affine"],
                dtype=float,
            )
            stored_spacing = np.asarray(
                data["crop_spacing_mm"],
                dtype=float,
            )
            stored_center = np.asarray(
                data["predicted_center_canonical"],
                dtype=float,
            )
            stored_world_center = np.asarray(
                data["predicted_center_world_mm"],
                dtype=float,
            )

            stored_study_id = scalar_text(data["study_id"])
            stored_partition = scalar_text(data["partition"])
            stored_label = scalar_text(data["diagnostic_label"])
            stored_annotation = scalar_text(data["annotation_type"])
            stored_geometry = scalar_text(data["geometry_status"])

        ct_shape_pass = tuple(ct.shape) == TARGET_SHAPE
        mask_shape_pass = tuple(mask.shape) == TARGET_SHAPE
        ct_dtype_pass = ct.dtype == np.int16
        mask_dtype_pass = mask.dtype == np.uint8
        ct_finite = bool(np.isfinite(ct).all())
        mask_finite = bool(np.isfinite(mask).all())
        hu_range_pass = (
            int(ct.min()) >= HU_MIN and int(ct.max()) <= HU_MAX
        )
        unique_labels = sorted(np.unique(mask).astype(int).tolist())
        labels_allowed = set(unique_labels).issubset(ALLOWED_LABELS)

        affine_shape_pass = crop_affine.shape == (4, 4)
        affine_finite = bool(np.isfinite(crop_affine).all())
        affine_nonsingular = (
            affine_shape_pass
            and affine_finite
            and abs(float(np.linalg.det(crop_affine[:3, :3])))
            > 1e-8
        )
        affine_spacing = (
            np.linalg.norm(crop_affine[:3, :3], axis=0)
            if affine_shape_pass
            else np.full(3, np.nan)
        )
        affine_spacing_pass = bool(
            np.allclose(
                affine_spacing,
                TARGET_SPACING_MM,
                rtol=1e-6,
                atol=1e-6,
            )
        )
        stored_spacing_pass = bool(
            stored_spacing.shape == (3,)
            and np.allclose(
                stored_spacing,
                TARGET_SPACING_MM,
                rtol=1e-6,
                atol=1e-6,
            )
        )
        orientation = (
            "".join(nib.aff2axcodes(crop_affine))
            if affine_nonsingular
            else "INVALID"
        )
        orientation_pass = orientation == "LPS"

        expected_center = np.asarray(
            [
                prediction_row["predicted_center_canonical_x"],
                prediction_row["predicted_center_canonical_y"],
                prediction_row["predicted_center_canonical_z"],
            ],
            dtype=float,
        )
        center_difference = float(
            np.max(np.abs(stored_center - expected_center))
        )
        center_matches = (
            stored_center.shape == (3,)
            and np.all(np.isfinite(stored_center))
            and center_difference <= 1e-5
        )
        world_center_finite = bool(
            stored_world_center.shape == (3,)
            and np.all(np.isfinite(stored_world_center))
        )

        metadata_matches = all(
            [
                stored_study_id == study_id,
                stored_partition == str(source_row["partition"]),
                stored_label == str(source_row["diagnostic_label"]),
                stored_annotation
                == str(source_row["annotation_type"]).lower(),
                stored_geometry == str(source_row["geometry_status"]),
            ]
        )

        label_counts = np.bincount(
            mask.reshape(-1),
            minlength=7,
        )[:7].astype(np.int64)
        resampled_lesion_matches = (
            int(label_counts[1])
            == int(source_row["resampled_crop_lesion_voxels"])
        )
        resampled_pancreas_matches = (
            int(label_counts[4])
            == int(source_row["resampled_crop_pancreas_voxels"])
        )
        resampled_union_matches = (
            int(label_counts[1] + label_counts[4])
            == int(source_row["resampled_crop_union_voxels"])
        )

        full_lesion = int(source_row["full_lesion_voxels"])
        contained_lesion = int(
            source_row["contained_source_lesion_voxels"]
        )
        full_pancreas = int(source_row["full_pancreas_voxels"])
        contained_pancreas = int(
            source_row["contained_source_pancreas_voxels"]
        )
        full_union = int(source_row["full_union_voxels"])
        contained_union = int(
            source_row["contained_source_union_voxels"]
        )

        recomputed_lesion_coverage = safe_fraction(
            contained_lesion,
            full_lesion,
            1.0,
        )
        recomputed_pancreas_coverage = safe_fraction(
            contained_pancreas,
            full_pancreas,
            0.0,
        )
        recomputed_union_coverage = safe_fraction(
            contained_union,
            full_union,
            0.0,
        )
        coverage_values_match = all(
            [
                np.isclose(
                    recomputed_lesion_coverage,
                    float(source_row["lesion_voxel_coverage"]),
                    atol=1e-12,
                ),
                np.isclose(
                    recomputed_pancreas_coverage,
                    float(source_row["pancreas_voxel_coverage"]),
                    atol=1e-12,
                ),
                np.isclose(
                    recomputed_union_coverage,
                    float(source_row["union_voxel_coverage"]),
                    atol=1e-12,
                ),
            ]
        )
        containment_counts_valid = all(
            [
                0 <= contained_lesion <= full_lesion,
                0 < contained_pancreas <= full_pancreas,
                0 < contained_union <= full_union,
                full_union == full_lesion + full_pancreas,
                contained_union
                == contained_lesion + contained_pancreas,
            ]
        )

        is_pdac = str(source_row["diagnostic_label"]).lower() == "pdac"
        lesion_consistency = (
            full_lesion > 0 and int(label_counts[1]) > 0
            if is_pdac
            else full_lesion == 0 and int(label_counts[1]) == 0
        )
        pancreas_present = (
            full_pancreas > 0 and int(label_counts[4]) > 0
        )

        case_checks = {
            "size_matches": size_matches,
            "hash_matches": hash_matches,
            "key_set_exact": key_set_exact,
            "ct_shape_pass": ct_shape_pass,
            "mask_shape_pass": mask_shape_pass,
            "ct_dtype_pass": ct_dtype_pass,
            "mask_dtype_pass": mask_dtype_pass,
            "ct_finite": ct_finite,
            "mask_finite": mask_finite,
            "hu_range_pass": hu_range_pass,
            "labels_allowed": labels_allowed,
            "affine_shape_pass": affine_shape_pass,
            "affine_finite": affine_finite,
            "affine_nonsingular": affine_nonsingular,
            "affine_spacing_pass": affine_spacing_pass,
            "stored_spacing_pass": stored_spacing_pass,
            "orientation_pass": orientation_pass,
            "center_matches_stage2e": center_matches,
            "world_center_finite": world_center_finite,
            "metadata_matches": metadata_matches,
            "resampled_lesion_count_matches": resampled_lesion_matches,
            "resampled_pancreas_count_matches": resampled_pancreas_matches,
            "resampled_union_count_matches": resampled_union_matches,
            "coverage_values_match": coverage_values_match,
            "containment_counts_valid": containment_counts_valid,
            "lesion_consistency": lesion_consistency,
            "pancreas_present": pancreas_present,
        }
        all_checks_pass = all(
            bool(value) for value in case_checks.values()
        )
        if not all_checks_pass:
            failed = [
                name
                for name, passed in case_checks.items()
                if not bool(passed)
            ]
            raise RuntimeError(f"Case QC gates failed: {failed}")

        qc_row = {
            "study_id": study_id,
            "patient_id": source_row["patient_id"],
            "partition": source_row["partition"],
            "diagnostic_label": source_row["diagnostic_label"],
            "annotation_type": source_row["annotation_type"],
            "geometry_status": source_row["geometry_status"],
            "output_path": str(path),
            "output_size_bytes": observed_size,
            "output_mtime_ns": f"ns:{path.stat().st_mtime_ns}",
            "verified_sha256": observed_hash,
            "ct_shape_json": json.dumps(list(ct.shape)),
            "mask_shape_json": json.dumps(list(mask.shape)),
            "ct_dtype": str(ct.dtype),
            "mask_dtype": str(mask.dtype),
            "ct_minimum_hu": int(ct.min()),
            "ct_maximum_hu": int(ct.max()),
            "unique_labels_json": json.dumps(unique_labels),
            "crop_orientation": orientation,
            "affine_spacing_mm_json": json.dumps(
                affine_spacing.tolist()
            ),
            "maximum_center_difference": center_difference,
            "crop_lesion_voxels": int(label_counts[1]),
            "crop_pancreas_voxels": int(label_counts[4]),
            "crop_union_voxels": int(
                label_counts[1] + label_counts[4]
            ),
            "lesion_voxel_coverage": recomputed_lesion_coverage,
            "pancreas_voxel_coverage": recomputed_pancreas_coverage,
            "union_voxel_coverage": recomputed_union_coverage,
            **{name: bool(value) for name, value in case_checks.items()},
            "all_checks_pass": True,
            "qc_complete": True,
            "error_type": "",
            "error_message": "",
            "checked_at_utc": utc_now(),
        }

        resume_qc = replace_row(resume_qc, qc_row)
        resume_qc = resume_qc.sort_values(
            ["partition", "study_id"],
            na_position="last",
        ).reset_index(drop=True)
        atomic_write_csv(resume_qc, RESUME_QC_LEDGER_PATH)
        completed_ids.add(study_id)
        qc_this_run += 1

        del ct, mask

    except Exception as error:
        error_row = {
            "study_id": study_id,
            "patient_id": source_row.get("patient_id", ""),
            "partition": source_row.get("partition", ""),
            "diagnostic_label": source_row.get(
                "diagnostic_label",
                "",
            ),
            "annotation_type": source_row.get(
                "annotation_type",
                "",
            ),
            "geometry_status": source_row.get(
                "geometry_status",
                "",
            ),
            "output_path": str(path),
            "all_checks_pass": False,
            "qc_complete": False,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "checked_at_utc": utc_now(),
        }
        resume_qc = replace_row(resume_qc, error_row)
        atomic_write_csv(resume_qc, RESUME_QC_LEDGER_PATH)
        errors.append(error_row)
        print()
        print(
            f"FAIL — {study_id}: {type(error).__name__}: {error}"
        )
        print(
            "QC stopped; previously verified cases remain "
            "durably checkpointed."
        )
        break


# =============================================================================
# FINAL AGGREGATION AND FREEZE
# =============================================================================


durable_complete = len(completed_ids)
remaining = EXPECTED_CASES - durable_complete

if errors or remaining:
    print()
    print("-" * 116)
    print("STAGE 3B RUN RESULT")
    print("-" * 116)
    print(f"QC-completed this run: {qc_this_run}")
    print(f"Errors this run: {len(errors)}")
    print(
        f"Durably QC-completed: "
        f"{durable_complete}/{EXPECTED_CASES}"
    )
    print(f"Remaining: {remaining}")
    print("Checkpoint:")
    print(RESUME_QC_LEDGER_PATH)
    print()
    print("=" * 116)
    print(
        "STAGE 3B STATUS: "
        + ("STOPPED_WITH_ERROR" if errors else "MORE_CASES_REQUIRED")
    )
    print("=" * 116)
    if errors:
        raise RuntimeError(
            "Stage 3B stopped on an error; verified cases remain "
            "checkpointed."
        )
    raise SystemExit(0)

final_qc = resume_qc.loc[
    resume_qc["study_id"].isin(stage3a["study_id"])
].copy()
final_qc = final_qc.sort_values(
    ["partition", "study_id"]
).reset_index(drop=True)

if len(final_qc) != EXPECTED_CASES:
    raise RuntimeError("Final QC row count is incomplete.")
if final_qc["study_id"].nunique() != EXPECTED_CASES:
    raise RuntimeError("Final QC IDs are not unique.")
if not truth_flags(final_qc["all_checks_pass"]).all():
    raise RuntimeError("One or more final case QC rows failed.")

pdac_qc = final_qc.loc[
    final_qc["diagnostic_label"].astype(str).str.lower() == "pdac"
]
non_pdac_qc = final_qc.loc[
    final_qc["diagnostic_label"].astype(str).str.lower()
    == "non-pdac"
]

coverage_summary_rows = []
for scope_name, group in [
    ("ALL_DEVELOPMENT", final_qc),
    ("TRAIN", final_qc.loc[final_qc["partition"] == "train"]),
    (
        "VALIDATION",
        final_qc.loc[final_qc["partition"] == "validation"],
    ),
    ("PDAC", pdac_qc),
    ("NON_PDAC", non_pdac_qc),
]:
    lesion_values = (
        group.loc[
            group["diagnostic_label"].astype(str).str.lower()
            == "pdac",
            "lesion_voxel_coverage",
        ].astype(float)
    )
    coverage_summary_rows.append(
        {
            "scope": scope_name,
            "cases": int(len(group)),
            "PDAC_cases": int(
                (
                    group["diagnostic_label"]
                    .astype(str)
                    .str.lower()
                    == "pdac"
                ).sum()
            ),
            "minimum_pancreas_coverage": float(
                group["pancreas_voxel_coverage"].min()
            ),
            "p01_pancreas_coverage": percentile(
                group["pancreas_voxel_coverage"],
                1,
            ),
            "minimum_union_coverage": float(
                group["union_voxel_coverage"].min()
            ),
            "p01_union_coverage": percentile(
                group["union_voxel_coverage"],
                1,
            ),
            "minimum_PDAC_lesion_coverage": (
                float(lesion_values.min())
                if len(lesion_values)
                else np.nan
            ),
            "PDAC_lesions_with_exact_coverage": int(
                np.isclose(lesion_values, 1.0, atol=1e-12).sum()
            ),
            "pancreas_exact_coverage_cases": int(
                np.isclose(
                    group["pancreas_voxel_coverage"],
                    1.0,
                    atol=1e-12,
                ).sum()
            ),
            "union_exact_coverage_cases": int(
                np.isclose(
                    group["union_voxel_coverage"],
                    1.0,
                    atol=1e-12,
                ).sum()
            ),
            "cases_below_99pct_union_coverage": int(
                (
                    group["union_voxel_coverage"].astype(float)
                    < MINIMUM_ACCEPTED_VOXEL_COVERAGE
                ).sum()
            ),
        }
    )

coverage_summary = pd.DataFrame(coverage_summary_rows)

geometry_counts = final_qc["geometry_status"].value_counts().to_dict()
reheader_count = int(
    truth_flags(stage3a["derived_mask_reheader_applied"]).sum()
)

readiness_checks = {
    "Exactly 1671 Stage 3A ledger rows": len(stage3a) == EXPECTED_CASES,
    "Exactly 1671 full-QC rows": len(final_qc) == EXPECTED_CASES,
    "Exactly 1671 unique full-QC IDs": (
        final_qc["study_id"].nunique() == EXPECTED_CASES
    ),
    "QC IDs exactly match Stage 3A": (
        set(final_qc["study_id"]) == set(stage3a["study_id"])
    ),
    "Persistent folder contains exactly 1671 NPZ files": (
        len(observed_files) == EXPECTED_CASES
    ),
    "Persistent inventory exactly matches Stage 3A": (
        observed_files == expected_files
    ),
    "No partial output files remain": len(partial_files) == 0,
    "All SHA-256 hashes match Stage 3A": truth_flags(
        final_qc["hash_matches"]
    ).all(),
    "All NPZ key sets are exact": truth_flags(
        final_qc["key_set_exact"]
    ).all(),
    "All CT arrays are 240x192x128 int16": (
        truth_flags(final_qc["ct_shape_pass"]).all()
        and truth_flags(final_qc["ct_dtype_pass"]).all()
    ),
    "All masks are 240x192x128 uint8": (
        truth_flags(final_qc["mask_shape_pass"]).all()
        and truth_flags(final_qc["mask_dtype_pass"]).all()
    ),
    "All arrays are finite": (
        truth_flags(final_qc["ct_finite"]).all()
        and truth_flags(final_qc["mask_finite"]).all()
    ),
    "All CT values are within -200 to 300 HU": truth_flags(
        final_qc["hu_range_pass"]
    ).all(),
    "All mask labels are restricted to 0-6": truth_flags(
        final_qc["labels_allowed"]
    ).all(),
    "All crop affines are finite and non-singular": (
        truth_flags(final_qc["affine_finite"]).all()
        and truth_flags(final_qc["affine_nonsingular"]).all()
    ),
    "All crop spacings are exactly 1.25x1.25x2 mm": (
        truth_flags(final_qc["affine_spacing_pass"]).all()
        and truth_flags(final_qc["stored_spacing_pass"]).all()
    ),
    "All crop orientations are LPS": truth_flags(
        final_qc["orientation_pass"]
    ).all(),
    "All saved centres match Stage 2E": truth_flags(
        final_qc["center_matches_stage2e"]
    ).all(),
    "All saved metadata matches Stage 3A": truth_flags(
        final_qc["metadata_matches"]
    ).all(),
    "All saved masks reproduce Stage 3A label counts": (
        truth_flags(
            final_qc["resampled_lesion_count_matches"]
        ).all()
        and truth_flags(
            final_qc["resampled_pancreas_count_matches"]
        ).all()
        and truth_flags(
            final_qc["resampled_union_count_matches"]
        ).all()
    ),
    "All containment coverage values reproduce Stage 3A": truth_flags(
        final_qc["coverage_values_match"]
    ).all(),
    "Every cropped mask contains pancreas": truth_flags(
        final_qc["pancreas_present"]
    ).all(),
    "All lesion labels remain diagnosis-consistent": truth_flags(
        final_qc["lesion_consistency"]
    ).all(),
    "PDAC count remains 491": len(pdac_qc) == EXPECTED_PDAC,
    "Non-PDAC count remains 1180": (
        len(non_pdac_qc) == EXPECTED_NON_PDAC
    ),
    "Train count remains 1376": int(
        (final_qc["partition"] == "train").sum()
    ) == EXPECTED_TRAIN,
    "Validation count remains 295": int(
        (final_qc["partition"] == "validation").sum()
    ) == EXPECTED_VALIDATION,
    "All pancreata retain at least 99% source voxels": float(
        final_qc["pancreas_voxel_coverage"].min()
    ) >= MINIMUM_ACCEPTED_VOXEL_COVERAGE,
    "All PDAC lesions retain at least 99% source voxels": float(
        pdac_qc["lesion_voxel_coverage"].min()
    ) >= MINIMUM_ACCEPTED_VOXEL_COVERAGE,
    "All pancreas-lesion unions retain at least 99% source voxels": float(
        final_qc["union_voxel_coverage"].min()
    ) >= MINIMUM_ACCEPTED_VOXEL_COVERAGE,
    "Geometry-status counts reproduce Stage 1B": (
        geometry_counts == EXPECTED_GEOMETRY_COUNTS
    ),
    "Exactly two approved derived reheaders were applied": (
        reheader_count == 2
    ),
    "No locked test case was accessed": (
        set(final_qc["partition"]) == {"train", "validation"}
    ),
    "Final E5 crop geometry remains locked": (
        stage2d_protocol.get("selected_candidate")
        == "E5_240x192x128"
    ),
}
all_checks_pass = all(
    bool(value) for value in readiness_checks.values()
)

atomic_write_csv(final_qc, CANONICAL_QC_PATH)
atomic_write_csv(coverage_summary, COVERAGE_SUMMARY_PATH)

frozen_manifest = stage3a.merge(
    final_qc[
        [
            "study_id",
            "verified_sha256",
            "output_mtime_ns",
            "ct_shape_json",
            "mask_shape_json",
            "ct_dtype",
            "mask_dtype",
            "unique_labels_json",
            "crop_orientation",
            "maximum_center_difference",
            "all_checks_pass",
        ]
    ],
    on="study_id",
    how="inner",
    validate="one_to_one",
    suffixes=("", "_qc"),
).sort_values(["partition", "study_id"]).reset_index(drop=True)
atomic_write_csv(frozen_manifest, FROZEN_MANIFEST_PATH)

manifest_hash = sha256_file(FROZEN_MANIFEST_PATH)
canonical_qc_hash = sha256_file(CANONICAL_QC_PATH)

freeze_protocol = {
    "stage": "3B",
    "created_at_utc": utc_now(),
    "dataset_name": "PANORAMA_LOCAL_DEVELOPMENT_E5_V1",
    "dataset_frozen": all_checks_pass,
    "cases": EXPECTED_CASES,
    "train_cases": EXPECTED_TRAIN,
    "validation_cases": EXPECTED_VALIDATION,
    "PDAC_cases": EXPECTED_PDAC,
    "non_PDAC_cases": EXPECTED_NON_PDAC,
    "crop_shape": list(TARGET_SHAPE),
    "crop_spacing_mm": TARGET_SPACING_MM.tolist(),
    "intensity_window_HU": [HU_MIN, HU_MAX],
    "mask_labels": list(range(7)),
    "minimum_accepted_source_voxel_coverage": (
        MINIMUM_ACCEPTED_VOXEL_COVERAGE
    ),
    "persistent_dataset_size_bytes": persistent_size,
    "frozen_manifest_path": str(FROZEN_MANIFEST_PATH),
    "frozen_manifest_sha256": manifest_hash,
    "canonical_qc_path": str(CANONICAL_QC_PATH),
    "canonical_qc_sha256": canonical_qc_hash,
    "locked_test_cases_accessed": 0,
    "raw_files_modified": False,
}
atomic_write_json(freeze_protocol, FREEZE_PROTOCOL_PATH)

audit = {
    "stage": "3B",
    "created_at_utc": utc_now(),
    "result": (
        "PASS_E5_DATASET_FROZEN" if all_checks_pass else "FAIL"
    ),
    "dataset_frozen": all_checks_pass,
    "persistent_files": len(observed_files),
    "persistent_size_bytes": persistent_size,
    "geometry_status_counts": {
        str(key): int(value)
        for key, value in geometry_counts.items()
    },
    "approved_derived_reheaders": reheader_count,
    "coverage_summary": json.loads(
        coverage_summary.to_json(orient="records")
    ),
    "readiness_checks": {
        key: bool(value)
        for key, value in readiness_checks.items()
    },
    "frozen_manifest_path": str(FROZEN_MANIFEST_PATH),
    "canonical_qc_path": str(CANONICAL_QC_PATH),
    "coverage_summary_path": str(COVERAGE_SUMMARY_PATH),
    "freeze_protocol_path": str(FREEZE_PROTOCOL_PATH),
    "locked_test_cases_accessed": 0,
}
atomic_write_json(audit, AUDIT_PATH)

print()
print("-" * 116)
print("CROP-COVERAGE SUMMARY")
print("-" * 116)
print(coverage_summary.to_string(index=False))

print()
print("-" * 116)
print("GEOMETRY STATUS COUNTS")
print("-" * 116)
print(final_qc["geometry_status"].value_counts().to_string())

print()
print("-" * 116)
print("READINESS CHECKS")
print("-" * 116)
for check, passed in readiness_checks.items():
    print(f"  {check}: {bool(passed)}")

print()
print("Frozen manifest:")
print(FROZEN_MANIFEST_PATH)
print()
print("Canonical full QC:")
print(CANONICAL_QC_PATH)
print()
print("Freeze protocol:")
print(FREEZE_PROTOCOL_PATH)
print()
print("Final audit:")
print(AUDIT_PATH)
print()
print("=" * 116)
print(
    "STAGE 3B RESULT: "
    + ("PASS_E5_DATASET_FROZEN" if all_checks_pass else "FAIL")
)
print("=" * 116)

if not all_checks_pass:
    failed = [
        check
        for check, passed in readiness_checks.items()
        if not bool(passed)
    ]
    raise RuntimeError(f"Stage 3B failed checks: {failed}")
