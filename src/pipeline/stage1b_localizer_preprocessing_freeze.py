from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(os.environ.get("PDAC_PROJECT_ROOT", "/content/drive/MyDrive/PDAC_Public_Q1_Project"))
META_DIR = PROJECT_ROOT / "01_Metadata"
QC_DIR = PROJECT_ROOT / "02_Quality_Control"
CASE_DIR = (
    PROJECT_ROOT
    / "03_Processed"
    / "Localizer"
    / "F1_96x96x160"
    / "Cases"
)

STAGE1A_LEDGER_PATH = (
    QC_DIR / "stage1a_localizer_preprocessing_ledger.csv"
)
CASE_GEOMETRY_PATH = (
    QC_DIR / "stage0m_b_r_localizer_case_geometry.csv"
)
PROTOCOL_PATH = (
    META_DIR / "stage0m_b_r_localizer_input_protocol.json"
)
STAGE1A_ERROR_PATH = (
    QC_DIR / "stage1a_localizer_preprocessing_errors.csv"
)

QC_CHECKPOINT_PATH = (
    QC_DIR / "stage1b_localizer_preprocessing_full_qc.csv"
)
MANIFEST_PATH = (
    META_DIR / "stage1b_localizer_preprocessed_manifest.csv"
)
SUMMARY_PATH = (
    QC_DIR / "stage1b_localizer_preprocessing_summary.csv"
)
ERROR_RESOLUTION_PATH = (
    QC_DIR / "stage1b_stage1a_error_resolution.csv"
)
FREEZE_PROTOCOL_PATH = (
    META_DIR / "stage1b_localizer_dataset_freeze.json"
)
AUDIT_PATH = (
    QC_DIR / "stage1b_localizer_preprocessing_audit.json"
)

EXPECTED_CASES = 1671
EXPECTED_PARTITIONS = {"train": 1376, "validation": 295}
EXPECTED_SHAPE = (96, 96, 160)
EXPECTED_KEYS = {
    "ct_hu",
    "pancreas_center_voxel",
    "pancreas_center_normalized",
    "pancreas_bbox_size_voxels",
}
ALLOWED_GEOMETRY_STATUSES = {
    "EXACT_PHYSICAL_GEOMETRY_MATCH",
    "PHYSICAL_GEOMETRY_MATCH_WITHIN_NUMERICAL_HEADER_TOLERANCE",
    "KNOWN_VISUALLY_CONFIRMED_INDEX_ALIGNED_HEADER_MISMATCH",
    "AUTOMATIC_INDEX_ALIGNED_HEADER_MISMATCH_USING_CT_GEOMETRY",
}
HU_LOWER = -200
HU_UPPER = 300


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def normalize_study_id(value):
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def as_bool(value):
    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
        "pass",
        "complete",
        "completed",
    }


def atomic_write_csv(dataframe, path):
    temporary_path = Path(str(path) + ".tmp")
    dataframe.to_csv(temporary_path, index=False)
    os.replace(temporary_path, path)


def atomic_write_json(data, path):
    temporary_path = Path(str(path) + ".tmp")
    with open(temporary_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
    os.replace(temporary_path, path)


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def parse_json_vector(value, dtype=float):
    vector = np.asarray(json.loads(str(value)), dtype=dtype)
    if vector.shape != (3,):
        raise RuntimeError(f"Expected a three-element vector: {value}")
    return vector


print("=" * 112)
print("STAGE 1B — RESUMABLE LOCALIZER PREPROCESSING QC AND DATASET FREEZE")
print("=" * 112)

required_paths = [
    STAGE1A_LEDGER_PATH,
    CASE_GEOMETRY_PATH,
    PROTOCOL_PATH,
    CASE_DIR,
]
for required_path in required_paths:
    if not required_path.exists():
        raise FileNotFoundError(f"Required input missing:\n{required_path}")

stage1a = pd.read_csv(STAGE1A_LEDGER_PATH)
case_geometry = pd.read_csv(CASE_GEOMETRY_PATH)

for dataframe in [stage1a, case_geometry]:
    dataframe["study_id"] = dataframe["study_id"].map(
        normalize_study_id
    )

with open(PROTOCOL_PATH, "r", encoding="utf-8") as file:
    input_protocol = json.load(file)

if len(stage1a) != EXPECTED_CASES:
    raise RuntimeError(
        f"Stage 1A ledger must contain {EXPECTED_CASES} rows; "
        f"observed {len(stage1a)}."
    )
if stage1a["study_id"].nunique() != EXPECTED_CASES:
    raise RuntimeError("Stage 1A ledger study IDs are not unique.")
if len(case_geometry) != EXPECTED_CASES:
    raise RuntimeError("Development geometry index is incomplete.")
if case_geometry["study_id"].nunique() != EXPECTED_CASES:
    raise RuntimeError("Development geometry IDs are not unique.")
if set(stage1a["study_id"]) != set(case_geometry["study_id"]):
    raise RuntimeError(
        "Stage 1A ledger does not exactly match the development cohort."
    )
if not stage1a["processing_complete"].map(as_bool).all():
    raise RuntimeError("At least one Stage 1A case is incomplete.")
if not stage1a["source_integrity_pass"].map(as_bool).all():
    raise RuntimeError("At least one Stage 1A source-integrity gate failed.")
if not stage1a["output_reopen_pass"].map(as_bool).all():
    raise RuntimeError("At least one Stage 1A output-reopen gate failed.")
if set(stage1a["partition"]) != set(EXPECTED_PARTITIONS):
    raise RuntimeError("Unexpected partition entered the localizer dataset.")
if (
    stage1a["partition"].value_counts().to_dict()
    != EXPECTED_PARTITIONS
):
    raise RuntimeError(
        "Train/validation counts differ from the locked split."
    )
if not set(stage1a["geometry_status"]).issubset(
    ALLOWED_GEOMETRY_STATUSES
):
    unexpected = sorted(
        set(stage1a["geometry_status"]) - ALLOWED_GEOMETRY_STATUSES
    )
    raise RuntimeError(
        f"Unexpected geometry status values: {unexpected}"
    )
if input_protocol.get("selected_candidate") != "F1_96x96x160":
    raise RuntimeError("The locked F1 localizer protocol was not found.")

expected_filenames = {
    f"{study_id}.npz" for study_id in stage1a["study_id"]
}
observed_files = sorted(CASE_DIR.glob("*.npz"))
observed_filenames = {path.name for path in observed_files}
partial_files = sorted(CASE_DIR.glob("*.part"))

if observed_filenames != expected_filenames:
    missing = sorted(expected_filenames - observed_filenames)
    unexpected = sorted(observed_filenames - expected_filenames)
    raise RuntimeError(
        "Persistent NPZ inventory mismatch. "
        f"Missing={missing[:5]}, unexpected={unexpected[:5]}"
    )
if partial_files:
    raise RuntimeError(
        f"Unresolved partial output files found: {partial_files[:5]}"
    )

if QC_CHECKPOINT_PATH.exists():
    qc = pd.read_csv(QC_CHECKPOINT_PATH)
    qc["study_id"] = qc["study_id"].map(normalize_study_id)
    if qc["study_id"].duplicated().any():
        raise RuntimeError("Duplicate IDs in the Stage 1B checkpoint.")
    if not set(qc["study_id"]).issubset(set(stage1a["study_id"])):
        raise RuntimeError("Unexpected ID in the Stage 1B checkpoint.")
    completed_ids = set(
        qc.loc[
            qc["qc_complete"].map(as_bool),
            "study_id",
        ]
    )
else:
    qc = pd.DataFrame()
    completed_ids = set()

pending = (
    stage1a.loc[~stage1a["study_id"].isin(completed_ids)]
    .sort_values(["partition", "study_id"])
    .reset_index(drop=True)
)

print()
print(f"Expected development cases: {EXPECTED_CASES}")
print(f"Previously QC-completed: {len(completed_ids)}")
print(f"Pending before this run: {len(pending)}")
print(f"Persistent NPZ files: {len(observed_files)}")
print(
    "Persistent size: "
    f"{sum(path.stat().st_size for path in observed_files) / (1024 ** 3):.3f} GiB"
)
print("Locked test cases accessed: 0")
print("Checkpoint frequency: after every case")

errors = []

for order, (_, source_row) in enumerate(pending.iterrows(), start=1):
    study_id = source_row["study_id"]
    output_path = Path(str(source_row["output_path"]))

    if order == 1 or order % 50 == 0 or order == len(pending):
        print(
            f"QC progress: {order}/{len(pending)} — "
            f"{study_id} — durable {len(completed_ids)}/{EXPECTED_CASES}"
        )

    try:
        if output_path.parent != CASE_DIR:
            raise RuntimeError("Output path lies outside the locked case folder.")
        if not output_path.exists():
            raise FileNotFoundError(f"Output missing: {output_path}")
        if output_path.name != f"{study_id}.npz":
            raise RuntimeError("Output filename does not match the study ID.")

        observed_size = int(output_path.stat().st_size)
        expected_size = int(source_row["output_size_bytes"])
        if observed_size != expected_size:
            raise RuntimeError(
                f"Output-size mismatch: {observed_size}/{expected_size}"
            )

        observed_hash = sha256_file(output_path)
        expected_hash = str(source_row["output_sha256"]).lower()
        if observed_hash.lower() != expected_hash:
            raise RuntimeError("Output SHA-256 mismatch.")

        with np.load(output_path, allow_pickle=False) as data:
            keys = set(data.files)
            if keys != EXPECTED_KEYS:
                raise RuntimeError(
                    f"Unexpected NPZ keys: {sorted(keys)}"
                )

            ct_hu = data["ct_hu"]
            center_voxel = np.asarray(
                data["pancreas_center_voxel"],
                dtype=float,
            )
            center_normalized = np.asarray(
                data["pancreas_center_normalized"],
                dtype=float,
            )
            bbox_size = np.asarray(
                data["pancreas_bbox_size_voxels"],
                dtype=float,
            )

            if ct_hu.shape != EXPECTED_SHAPE:
                raise RuntimeError(
                    f"CT shape mismatch: {ct_hu.shape}/{EXPECTED_SHAPE}"
                )
            if ct_hu.dtype != np.int16:
                raise RuntimeError(f"CT dtype mismatch: {ct_hu.dtype}")
            if not np.all(np.isfinite(ct_hu)):
                raise RuntimeError("CT contains non-finite values.")

            hu_min = int(np.min(ct_hu))
            hu_max = int(np.max(ct_hu))
            if hu_min < HU_LOWER or hu_max > HU_UPPER:
                raise RuntimeError(
                    f"CT HU bounds violated: [{hu_min}, {hu_max}]"
                )

            for name, vector in [
                ("center_voxel", center_voxel),
                ("center_normalized", center_normalized),
                ("bbox_size", bbox_size),
            ]:
                if vector.shape != (3,):
                    raise RuntimeError(
                        f"{name} shape mismatch: {vector.shape}"
                    )
                if not np.all(np.isfinite(vector)):
                    raise RuntimeError(f"{name} is non-finite.")

            shape_array = np.asarray(EXPECTED_SHAPE, dtype=float)
            if np.any(center_voxel < 0) or np.any(
                center_voxel > shape_array - 1
            ):
                raise RuntimeError("Pancreas centre lies outside the canvas.")
            if np.any(center_normalized < 0) or np.any(
                center_normalized > 1
            ):
                raise RuntimeError(
                    "Normalized pancreas centre lies outside [0, 1]."
                )
            if not np.allclose(
                center_normalized,
                center_voxel / (shape_array - 1),
                rtol=1e-5,
                atol=1e-5,
            ):
                raise RuntimeError(
                    "Voxel and normalized pancreas centres disagree."
                )
            if np.any(bbox_size <= 0) or np.any(
                bbox_size > shape_array + 1
            ):
                raise RuntimeError("Pancreas bounding-box size is invalid.")

            ledger_center = parse_json_vector(
                source_row["pancreas_center_voxel_json"]
            )
            ledger_center_normalized = parse_json_vector(
                source_row["pancreas_center_normalized_json"]
            )
            ledger_bbox_size = parse_json_vector(
                source_row["pancreas_bbox_size_voxels_json"]
            )
            if not np.allclose(
                center_voxel, ledger_center, rtol=1e-6, atol=1e-6
            ):
                raise RuntimeError(
                    "Stored centre differs from the Stage 1A ledger."
                )
            if not np.allclose(
                center_normalized,
                ledger_center_normalized,
                rtol=1e-6,
                atol=1e-6,
            ):
                raise RuntimeError(
                    "Stored normalized centre differs from the ledger."
                )
            if not np.allclose(
                bbox_size, ledger_bbox_size, rtol=1e-6, atol=1e-6
            ):
                raise RuntimeError(
                    "Stored bounding-box size differs from the ledger."
                )

        qc_row = {
            "study_id": study_id,
            "patient_id": source_row["patient_id"],
            "partition": source_row["partition"],
            "diagnostic_label": source_row["diagnostic_label"],
            "annotation_type": source_row["annotation_type"],
            "geometry_status": source_row["geometry_status"],
            "output_path": str(output_path),
            "output_size_bytes": observed_size,
            "expected_sha256": expected_hash,
            "observed_sha256": observed_hash,
            "hash_match": True,
            "npz_keys_exact": True,
            "ct_shape": json.dumps(list(EXPECTED_SHAPE)),
            "ct_dtype": "int16",
            "ct_finite": True,
            "hu_min": hu_min,
            "hu_max": hu_max,
            "hu_bounds_pass": True,
            "pancreas_center_voxel_json": json.dumps(
                [float(value) for value in center_voxel]
            ),
            "pancreas_center_normalized_json": json.dumps(
                [float(value) for value in center_normalized]
            ),
            "pancreas_bbox_size_voxels_json": json.dumps(
                [float(value) for value in bbox_size]
            ),
            "target_vectors_pass": True,
            "ledger_target_match": True,
            "qc_pass": True,
            "qc_complete": True,
            "qc_checked_at_utc": utc_now(),
        }

        qc = pd.concat(
            [qc, pd.DataFrame([qc_row])],
            ignore_index=True,
        )
        qc = qc.sort_values("study_id").reset_index(drop=True)
        atomic_write_csv(qc, QC_CHECKPOINT_PATH)
        completed_ids.add(study_id)

    except Exception as error:
        error_row = {
            "study_id": study_id,
            "error_type": type(error).__name__,
            "error_message": str(error),
        }
        errors.append(error_row)
        print(
            f"\nFAIL — {study_id}: "
            f"{type(error).__name__}: {error}"
        )
        print("QC stopped; completed QC rows remain checkpointed.")
        break

if errors:
    raise RuntimeError(
        "Stage 1B stopped on a QC error. "
        "Previously completed QC rows remain durable."
    )

if len(completed_ids) != EXPECTED_CASES:
    print()
    print(
        f"STAGE 1B INCOMPLETE: {len(completed_ids)}/{EXPECTED_CASES}. "
        "Rerun the same script."
    )
    raise SystemExit(0)

qc = pd.read_csv(QC_CHECKPOINT_PATH)
qc["study_id"] = qc["study_id"].map(normalize_study_id)

if len(qc) != EXPECTED_CASES or qc["study_id"].nunique() != EXPECTED_CASES:
    raise RuntimeError("Final Stage 1B QC ledger is incomplete.")
if not qc["qc_pass"].map(as_bool).all():
    raise RuntimeError("At least one final Stage 1B QC row failed.")
if set(qc["study_id"]) != set(stage1a["study_id"]):
    raise RuntimeError("Final QC identifiers do not match Stage 1A.")

manifest_columns = [
    "study_id",
    "patient_id",
    "partition",
    "diagnostic_label",
    "annotation_type",
    "geometry_status",
    "output_path",
    "output_size_bytes",
    "output_sha256",
    "ct_original_shape_json",
    "ct_original_spacing_mm_json",
    "ct_original_orientation",
    "canonical_shape_json",
    "canonical_spacing_mm_json",
    "effective_isotropic_spacing_mm",
    "resampled_core_shape_json",
    "pad_before_json",
    "pad_after_json",
    "output_canvas_shape_json",
    "pancreas_center_voxel_json",
    "pancreas_center_normalized_json",
    "pancreas_bbox_size_voxels_json",
]
manifest = (
    stage1a[manifest_columns]
    .sort_values(["partition", "study_id"])
    .reset_index(drop=True)
)
atomic_write_csv(manifest, MANIFEST_PATH)

partition_summary = (
    manifest.groupby(["partition", "diagnostic_label"])
    .size()
    .rename("cases")
    .reset_index()
)
geometry_summary = (
    manifest.groupby(["geometry_status"])
    .size()
    .rename("cases")
    .reset_index()
)
annotation_summary = (
    manifest.groupby(["partition", "annotation_type"])
    .size()
    .rename("cases")
    .reset_index()
)

summary = pd.concat(
    [
        partition_summary.assign(summary_type="PARTITION_LABEL"),
        annotation_summary.assign(summary_type="PARTITION_ANNOTATION"),
        geometry_summary.assign(summary_type="GEOMETRY"),
    ],
    ignore_index=True,
    sort=False,
)
atomic_write_csv(summary, SUMMARY_PATH)

if STAGE1A_ERROR_PATH.exists():
    historical_errors = pd.read_csv(STAGE1A_ERROR_PATH)
    historical_errors["study_id"] = historical_errors["study_id"].map(
        normalize_study_id
    )
    historical_errors["resolved_by_final_stage1b_qc"] = (
        historical_errors["study_id"].isin(set(qc["study_id"]))
    )
    historical_errors["resolution_status"] = np.where(
        historical_errors["resolved_by_final_stage1b_qc"],
        "RESOLVED_FINAL_OUTPUT_HASH_AND_CONTENT_VERIFIED",
        "UNRESOLVED",
    )
    historical_errors["resolved_at_utc"] = np.where(
        historical_errors["resolved_by_final_stage1b_qc"],
        utc_now(),
        "",
    )
else:
    historical_errors = pd.DataFrame(
        columns=[
            "study_id",
            "resolved_by_final_stage1b_qc",
            "resolution_status",
            "resolved_at_utc",
        ]
    )
atomic_write_csv(historical_errors, ERROR_RESOLUTION_PATH)

partition_counts = {
    str(key): int(value)
    for key, value in manifest["partition"].value_counts().items()
}
geometry_counts = {
    str(key): int(value)
    for key, value in manifest["geometry_status"].value_counts().items()
}
orientation_counts = {
    str(key): int(value)
    for key, value in manifest[
        "ct_original_orientation"
    ].value_counts().items()
}

readiness_checks = {
    "Exactly 1671 QC rows": len(qc) == EXPECTED_CASES,
    "Exactly 1671 unique study IDs":
        qc["study_id"].nunique() == EXPECTED_CASES,
    "QC IDs exactly match development cohort":
        set(qc["study_id"]) == set(case_geometry["study_id"]),
    "Persistent folder contains exactly 1671 NPZ files":
        len(observed_files) == EXPECTED_CASES,
    "Persistent inventory exactly matches Stage 1A":
        observed_filenames == expected_filenames,
    "No partial output files remain": len(partial_files) == 0,
    "All SHA-256 hashes match": qc["hash_match"].map(as_bool).all(),
    "All NPZ key sets are exact":
        qc["npz_keys_exact"].map(as_bool).all(),
    "All CT arrays have shape 96x96x160":
        (qc["ct_shape"] == json.dumps(list(EXPECTED_SHAPE))).all(),
    "All CT arrays are int16": (qc["ct_dtype"] == "int16").all(),
    "All CT arrays are finite": qc["ct_finite"].map(as_bool).all(),
    "All HU values remain within -200 to 300":
        qc["hu_bounds_pass"].map(as_bool).all(),
    "All pancreas targets pass":
        qc["target_vectors_pass"].map(as_bool).all(),
    "All stored targets match Stage 1A ledger":
        qc["ledger_target_match"].map(as_bool).all(),
    "Train count remains 1376": partition_counts.get("train") == 1376,
    "Validation count remains 295":
        partition_counts.get("validation") == 295,
    "No locked test case was accessed":
        set(partition_counts) == {"train", "validation"},
    "All geometry statuses are approved":
        set(geometry_counts).issubset(ALLOWED_GEOMETRY_STATUSES),
    "F1 input geometry remains locked":
        input_protocol.get("selected_candidate") == "F1_96x96x160",
    "All historical Stage 1A errors are resolved":
        (
            len(historical_errors) == 0
            or historical_errors[
                "resolved_by_final_stage1b_qc"
            ].map(as_bool).all()
        ),
}
all_checks_pass = all(bool(value) for value in readiness_checks.values())

freeze_protocol = {
    "stage": "1B",
    "created_at_utc": utc_now(),
    "dataset_role": "LOCALIZER_TRAINING_AND_VALIDATION_ONLY",
    "cases": EXPECTED_CASES,
    "partition_counts": partition_counts,
    "canvas_shape": list(EXPECTED_SHAPE),
    "ct_dtype": "int16",
    "hu_clip": [HU_LOWER, HU_UPPER],
    "model_normalization": "(CT_HU - 50) / 250",
    "target_fields": [
        "pancreas_center_voxel",
        "pancreas_center_normalized",
        "pancreas_bbox_size_voxels",
    ],
    "manifest_path": str(MANIFEST_PATH),
    "full_qc_path": str(QC_CHECKPOINT_PATH),
    "locked_test_cases_accessed": 0,
    "dataset_frozen": all_checks_pass,
}
atomic_write_json(freeze_protocol, FREEZE_PROTOCOL_PATH)

audit = {
    "stage": "1B",
    "created_at_utc": utc_now(),
    "result": "PASS" if all_checks_pass else "FAIL",
    "cases": EXPECTED_CASES,
    "partition_counts": partition_counts,
    "geometry_status_counts": geometry_counts,
    "original_orientation_counts": orientation_counts,
    "persistent_output_GiB": float(
        manifest["output_size_bytes"].sum() / (1024 ** 3)
    ),
    "historical_error_rows": int(len(historical_errors)),
    "readiness_checks": {
        str(key): bool(value)
        for key, value in readiness_checks.items()
    },
    "manifest_path": str(MANIFEST_PATH),
    "qc_path": str(QC_CHECKPOINT_PATH),
    "summary_path": str(SUMMARY_PATH),
    "error_resolution_path": str(ERROR_RESOLUTION_PATH),
    "freeze_protocol_path": str(FREEZE_PROTOCOL_PATH),
}
atomic_write_json(audit, AUDIT_PATH)

print()
print("-" * 112)
print("PARTITION COUNTS")
print("-" * 112)
print(manifest["partition"].value_counts().to_string())

print()
print("-" * 112)
print("GEOMETRY STATUS COUNTS")
print("-" * 112)
print(manifest["geometry_status"].value_counts().to_string())

print()
print("-" * 112)
print("READINESS CHECKS")
print("-" * 112)
for check, passed in readiness_checks.items():
    print(f"  {check}: {bool(passed)}")

print()
print("Canonical manifest:")
print(MANIFEST_PATH)
print()
print("Full QC ledger:")
print(QC_CHECKPOINT_PATH)
print()
print("Freeze protocol:")
print(FREEZE_PROTOCOL_PATH)
print()
print("Final audit:")
print(AUDIT_PATH)
print()
print("=" * 112)
print(
    "STAGE 1B RESULT: "
    + (
        "PASS — LOCALIZER DATASET FROZEN"
        if all_checks_pass
        else "FAIL"
    )
)
print("=" * 112)

if not all_checks_pass:
    failed_checks = [
        check
        for check, passed in readiness_checks.items()
        if not bool(passed)
    ]
    raise RuntimeError(
        f"Stage 1B failed readiness checks: {failed_checks}"
    )
