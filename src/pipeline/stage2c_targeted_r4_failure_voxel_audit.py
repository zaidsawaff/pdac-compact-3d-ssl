from pathlib import Path
from datetime import datetime, timezone
from itertools import product
import gzip
import json
import os
import shutil
import tempfile

import nibabel as nib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(os.environ.get("PDAC_PROJECT_ROOT", "/content/drive/MyDrive/PDAC_Public_Q1_Project"))
RUNTIME_ROOT = Path(os.environ.get("PDAC_RUNTIME_ROOT", "/content"))
RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
META_DIR = PROJECT_ROOT / "01_Metadata"
QC_DIR = PROJECT_ROOT / "02_Quality_Control"
MANUAL_MASK_DIR = (
    PROJECT_ROOT
    / "00_Raw"
    / "PANORAMA"
    / "Manual_Labels"
)

FAILURE_PATH = (
    QC_DIR / "stage2b_validation_crop_coverage_failures.csv"
)
CASE_LEDGER_PATH = (
    QC_DIR / "stage2b_validation_crop_coverage_case_ledger.csv"
)
MANIFEST_PATH = (
    META_DIR / "stage1b_localizer_preprocessed_manifest.csv"
)

OUTPUT_CSV_PATH = (
    QC_DIR / "stage2c_targeted_r4_failure_voxel_audit.csv"
)
DECISION_PATH = (
    META_DIR / "stage2c_r4_targeted_voxel_coverage_decision.json"
)
AUDIT_PATH = (
    QC_DIR / "stage2c_targeted_r4_failure_voxel_audit.json"
)

R4_FOV_MM = np.asarray([280.0, 240.0, 224.0], dtype=float)
EXPECTED_FAILURES = {"100684_00001", "100705_00001"}
SLICE_CHUNK = 8


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_write_csv(dataframe, path):
    temporary_path = Path(str(path) + ".tmp")
    dataframe.to_csv(temporary_path, index=False)
    os.replace(temporary_path, path)


def atomic_write_json(data, path):
    temporary_path = Path(str(path) + ".tmp")
    with open(temporary_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
    os.replace(temporary_path, path)


def normalize_study_id(value):
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def parse_vector(value, dtype=float):
    vector = np.asarray(json.loads(str(value)), dtype=dtype)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise RuntimeError(f"Invalid vector: {value}")
    return vector


def decompress_gzip(source, destination):
    bytes_written = 0
    with gzip.open(source, "rb") as input_file:
        with open(destination, "wb") as output_file:
            while True:
                chunk = input_file.read(8 * 1024 * 1024)
                if not chunk:
                    break
                output_file.write(chunk)
                bytes_written += len(chunk)
    return bytes_written


def canonical_crop_to_original_index_bounds(
    center_canonical,
    canonical_spacing,
    original_shape,
    original_orientation,
):
    half_fov = R4_FOV_MM / 2.0
    center_mm = center_canonical * canonical_spacing
    minimum_canonical = (
        center_mm - half_fov
    ) / canonical_spacing
    maximum_canonical = (
        center_mm + half_fov
    ) / canonical_spacing

    start_orientation = nib.orientations.axcodes2ornt(
        tuple(str(original_orientation))
    )
    target_orientation = nib.orientations.axcodes2ornt(
        ("L", "P", "S")
    )
    orientation_transform = nib.orientations.ornt_transform(
        start_orientation,
        target_orientation,
    )
    canonical_to_original = nib.orientations.inv_ornt_aff(
        orientation_transform,
        tuple(int(value) for value in original_shape),
    )

    canonical_corners = np.asarray(
        list(
            product(
                [minimum_canonical[0], maximum_canonical[0]],
                [minimum_canonical[1], maximum_canonical[1]],
                [minimum_canonical[2], maximum_canonical[2]],
            )
        ),
        dtype=float,
    )
    original_corners = nib.affines.apply_affine(
        canonical_to_original,
        canonical_corners,
    )
    minimum_original_float = np.min(original_corners, axis=0)
    maximum_original_float = np.max(original_corners, axis=0)

    minimum_index = np.ceil(
        minimum_original_float - 1e-6
    ).astype(int)
    maximum_index = np.floor(
        maximum_original_float + 1e-6
    ).astype(int)
    minimum_index = np.maximum(minimum_index, 0)
    maximum_index = np.minimum(
        maximum_index,
        original_shape.astype(int) - 1,
    )
    if np.any(maximum_index < minimum_index):
        raise RuntimeError("Predicted R4 crop does not intersect the mask.")

    return {
        "minimum_index": minimum_index,
        "maximum_index": maximum_index,
        "minimum_original_float": minimum_original_float,
        "maximum_original_float": maximum_original_float,
    }


def count_label_coverage(data_proxy, shape, minimum, maximum):
    totals = {1: 0, 4: 0}
    inside = {1: 0, 4: 0}

    for z_start in range(0, shape[2], SLICE_CHUNK):
        z_stop = min(z_start + SLICE_CHUNK, shape[2])
        block = np.asanyarray(
            data_proxy[:, :, z_start:z_stop]
        )

        totals[1] += int(np.count_nonzero(block == 1))
        totals[4] += int(np.count_nonzero(block == 4))

        inside_z_start = max(z_start, int(minimum[2]))
        inside_z_stop = min(
            z_stop,
            int(maximum[2]) + 1,
        )
        if inside_z_start < inside_z_stop:
            local_z_start = inside_z_start - z_start
            local_z_stop = inside_z_stop - z_start
            subvolume = block[
                int(minimum[0]):int(maximum[0]) + 1,
                int(minimum[1]):int(maximum[1]) + 1,
                local_z_start:local_z_stop,
            ]
            inside[1] += int(np.count_nonzero(subvolume == 1))
            inside[4] += int(np.count_nonzero(subvolume == 4))

        del block

    return totals, inside


print("=" * 112)
print("STAGE 2C — TARGETED R4 FAILURE VOXEL-COVERAGE AUDIT")
print("=" * 112)

for required_path in [
    FAILURE_PATH,
    CASE_LEDGER_PATH,
    MANIFEST_PATH,
    MANUAL_MASK_DIR,
]:
    if not required_path.exists():
        raise FileNotFoundError(f"Required input missing:\n{required_path}")

failures = pd.read_csv(FAILURE_PATH)
case_ledger = pd.read_csv(CASE_LEDGER_PATH)
manifest = pd.read_csv(MANIFEST_PATH)

for dataframe in [failures, case_ledger, manifest]:
    dataframe["study_id"] = dataframe["study_id"].map(
        normalize_study_id
    )

if set(failures["study_id"]) != EXPECTED_FAILURES:
    raise RuntimeError(
        "The Stage 2B failure set differs from the locked two cases. "
        f"Observed={sorted(set(failures['study_id']))}"
    )
if len(failures) != 2:
    raise RuntimeError("Exactly two Stage 2B failures were expected.")
if not (failures["diagnostic_label"] == "PDAC").all():
    raise RuntimeError("Both targeted cases must be PDAC.")
if not (failures["annotation_type"] == "manual").all():
    raise RuntimeError("Both targeted cases must have manual masks.")

target = (
    failures[
        [
            "study_id",
            "diagnostic_label",
            "annotation_type",
            "recorded_center_error_mm",
            "predicted_R4_bbox_volume_fraction",
            "predicted_R4_minimum_margin_mm",
        ]
    ]
    .merge(
        case_ledger[
            [
                "study_id",
                "predicted_center_canonical_json",
            ]
        ],
        on="study_id",
        how="inner",
        validate="one_to_one",
    )
    .merge(
        manifest[
            [
                "study_id",
                "partition",
                "ct_original_shape_json",
                "ct_original_orientation",
                "canonical_spacing_mm_json",
            ]
        ],
        on="study_id",
        how="inner",
        validate="one_to_one",
    )
    .sort_values("study_id")
    .reset_index(drop=True)
)
if len(target) != 2:
    raise RuntimeError("The targeted Stage 2C merge is incomplete.")
if set(target["partition"]) != {"validation"}:
    raise RuntimeError("A non-validation case entered Stage 2C.")

rows = []

for order, (_, row) in enumerate(target.iterrows(), start=1):
    study_id = row["study_id"]
    mask_path = MANUAL_MASK_DIR / f"{study_id}.nii.gz"
    if not mask_path.exists():
        raise FileNotFoundError(f"Manual mask missing: {mask_path}")

    original_shape = parse_vector(
        row["ct_original_shape_json"],
        dtype=int,
    )
    canonical_spacing = parse_vector(
        row["canonical_spacing_mm_json"],
        dtype=float,
    )
    predicted_center_canonical = parse_vector(
        row["predicted_center_canonical_json"],
        dtype=float,
    )
    crop_bounds = canonical_crop_to_original_index_bounds(
        predicted_center_canonical,
        canonical_spacing,
        original_shape,
        row["ct_original_orientation"],
    )

    print()
    print(f"[{order}/2] Processing {study_id}")

    with tempfile.TemporaryDirectory(
        prefix=f"stage2c_{study_id}_",
        dir=RUNTIME_ROOT,
    ) as temporary_directory:
        temporary_nii = (
            Path(temporary_directory) / f"{study_id}.nii"
        )
        decompressed_bytes = decompress_gzip(
            mask_path,
            temporary_nii,
        )
        image = nib.load(str(temporary_nii), mmap="r")
        mask_shape = np.asarray(image.shape, dtype=int)
        mask_orientation = "".join(
            nib.aff2axcodes(image.affine)
        )

        if not np.array_equal(mask_shape, original_shape):
            raise RuntimeError(
                f"{study_id}: mask/original shape mismatch "
                f"{mask_shape}/{original_shape}"
            )
        if mask_orientation != row["ct_original_orientation"]:
            raise RuntimeError(
                f"{study_id}: mask orientation mismatch "
                f"{mask_orientation}/"
                f"{row['ct_original_orientation']}"
            )

        proxy = image.dataobj
        unscaled = (
            proxy.get_unscaled()
            if hasattr(proxy, "get_unscaled")
            else proxy
        )
        totals, inside = count_label_coverage(
            unscaled,
            tuple(int(value) for value in mask_shape),
            crop_bounds["minimum_index"],
            crop_bounds["maximum_index"],
        )

    lesion_total = totals[1]
    pancreas_total = totals[4]
    lesion_inside = inside[1]
    pancreas_inside = inside[4]

    if lesion_total <= 0 or pancreas_total <= 0:
        raise RuntimeError(
            f"{study_id}: expected lesion and pancreas labels."
        )

    lesion_coverage = lesion_inside / lesion_total
    pancreas_coverage = pancreas_inside / pancreas_total
    union_total = lesion_total + pancreas_total
    union_inside = lesion_inside + pancreas_inside
    union_coverage = union_inside / union_total

    result = {
        "study_id": study_id,
        "partition": row["partition"],
        "diagnostic_label": row["diagnostic_label"],
        "annotation_type": row["annotation_type"],
        "recorded_center_error_mm": float(
            row["recorded_center_error_mm"]
        ),
        "bbox_volume_fraction": float(
            row["predicted_R4_bbox_volume_fraction"]
        ),
        "bbox_minimum_margin_mm": float(
            row["predicted_R4_minimum_margin_mm"]
        ),
        "crop_minimum_original_index_json": json.dumps(
            [
                int(value)
                for value in crop_bounds["minimum_index"]
            ]
        ),
        "crop_maximum_original_index_json": json.dumps(
            [
                int(value)
                for value in crop_bounds["maximum_index"]
            ]
        ),
        "lesion_total_voxels": lesion_total,
        "lesion_inside_voxels": lesion_inside,
        "lesion_outside_voxels": lesion_total - lesion_inside,
        "lesion_voxel_coverage": lesion_coverage,
        "pancreas_total_voxels": pancreas_total,
        "pancreas_inside_voxels": pancreas_inside,
        "pancreas_outside_voxels": (
            pancreas_total - pancreas_inside
        ),
        "pancreas_voxel_coverage": pancreas_coverage,
        "union_total_voxels": union_total,
        "union_inside_voxels": union_inside,
        "union_outside_voxels": union_total - union_inside,
        "union_voxel_coverage": union_coverage,
        "decompressed_mask_bytes": decompressed_bytes,
        "raw_mask_modified": False,
        "checked_at_utc": utc_now(),
    }
    rows.append(result)

    print(
        f"    Lesion coverage: {lesion_coverage:.8f} "
        f"({lesion_inside}/{lesion_total})"
    )
    print(
        f"    Pancreas coverage: {pancreas_coverage:.8f} "
        f"({pancreas_inside}/{pancreas_total})"
    )
    print(
        f"    Union coverage: {union_coverage:.8f} "
        f"({union_inside}/{union_total})"
    )

results = pd.DataFrame(rows)
atomic_write_csv(results, OUTPUT_CSV_PATH)

acceptance_checks = {
    "Both targeted lesions retain 100% of voxels":
        bool((results["lesion_voxel_coverage"] == 1.0).all()),
    "Both targeted pancreata retain at least 99% of voxels":
        bool((results["pancreas_voxel_coverage"] >= 0.99).all()),
    "Both targeted pancreas-lesion unions retain at least 99%":
        bool((results["union_voxel_coverage"] >= 0.99).all()),
    "Both cases remain validation-only":
        set(results["partition"]) == {"validation"},
    "No raw mask was modified":
        bool((~results["raw_mask_modified"].astype(bool)).all()),
}
r4_voxel_policy_accepted = all(
    bool(value) for value in acceptance_checks.values()
)

decision = {
    "stage": "2C",
    "created_at_utc": utc_now(),
    "targeted_cases": sorted(EXPECTED_FAILURES),
    "targeted_case_count": 2,
    "audit_role": (
        "EXACT_LABEL_VOXEL_COVERAGE_FOR_CONSERVATIVE_"
        "BBOX_CONTAINMENT_FAILURES"
    ),
    "acceptance_checks": acceptance_checks,
    "minimum_lesion_voxel_coverage": float(
        results["lesion_voxel_coverage"].min()
    ),
    "minimum_pancreas_voxel_coverage": float(
        results["pancreas_voxel_coverage"].min()
    ),
    "minimum_union_voxel_coverage": float(
        results["union_voxel_coverage"].min()
    ),
    "r4_voxel_policy_accepted": r4_voxel_policy_accepted,
    "decision": (
        "LOCK_R4_WITH_LEARNED_LOCALIZER"
        if r4_voxel_policy_accepted
        else "R4_REJECTED_USE_EXPANDED_CROP_POLICY"
    ),
    "locked_test_cases_accessed": 0,
    "output_csv_path": str(OUTPUT_CSV_PATH),
}
atomic_write_json(decision, DECISION_PATH)

readiness_checks = {
    "Exactly two targeted failures were audited":
        len(results) == 2,
    "The targeted IDs match Stage 2B failures":
        set(results["study_id"]) == EXPECTED_FAILURES,
    "Both masks were readable": True,
    "Both masks contained lesion and pancreas labels":
        bool(
            (results["lesion_total_voxels"] > 0).all()
            and (results["pancreas_total_voxels"] > 0).all()
        ),
    "All coverage values are finite":
        bool(
            np.isfinite(
                results[
                    [
                        "lesion_voxel_coverage",
                        "pancreas_voxel_coverage",
                        "union_voxel_coverage",
                    ]
                ].to_numpy()
            ).all()
        ),
    "No locked test case was accessed":
        set(results["partition"]) == {"validation"},
}
all_readiness_checks_pass = all(
    bool(value) for value in readiness_checks.values()
)

audit = {
    "stage": "2C",
    "created_at_utc": utc_now(),
    "result": (
        "PASS_LOCK_R4_WITH_LEARNED_LOCALIZER"
        if (
            all_readiness_checks_pass
            and r4_voxel_policy_accepted
        )
        else (
            "PASS_AUDIT_COMPLETE_EXPANDED_CROP_REQUIRED"
            if all_readiness_checks_pass
            else "FAIL"
        )
    ),
    "readiness_checks": readiness_checks,
    "acceptance_checks": acceptance_checks,
    "decision_path": str(DECISION_PATH),
    "locked_test_cases_accessed": 0,
}
atomic_write_json(audit, AUDIT_PATH)

print()
print("-" * 112)
print("TARGETED VOXEL-COVERAGE RESULTS")
print("-" * 112)
print(
    results[
        [
            "study_id",
            "lesion_voxel_coverage",
            "pancreas_voxel_coverage",
            "union_voxel_coverage",
            "lesion_outside_voxels",
            "pancreas_outside_voxels",
        ]
    ].to_string(index=False)
)

print()
print("-" * 112)
print("ACCEPTANCE CHECKS")
print("-" * 112)
for check, passed in acceptance_checks.items():
    print(f"  {check}: {bool(passed)}")

print()
print("Output CSV:")
print(OUTPUT_CSV_PATH)
print()
print("Decision:")
print(DECISION_PATH)
print()
print("Audit:")
print(AUDIT_PATH)
print()
print("=" * 112)
print(f"STAGE 2C RESULT: {audit['result']}")
print("=" * 112)

if not all_readiness_checks_pass:
    raise RuntimeError("Stage 2C readiness checks failed.")
