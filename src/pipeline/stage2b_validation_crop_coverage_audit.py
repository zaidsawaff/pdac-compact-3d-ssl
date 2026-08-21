from pathlib import Path
from datetime import datetime, timezone
from itertools import product
import json
import os

import nibabel as nib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(os.environ.get("PDAC_PROJECT_ROOT", "/content/drive/MyDrive/PDAC_Public_Q1_Project"))
META_DIR = PROJECT_ROOT / "01_Metadata"
QC_DIR = PROJECT_ROOT / "02_Quality_Control"
MODEL_DIR = (
    PROJECT_ROOT
    / "04_Models"
    / "Localizer"
    / "F1_96x96x160"
)

PREDICTIONS_PATH = (
    MODEL_DIR / "stage2a_localizer_validation_predictions.csv"
)
TRAINING_AUDIT_PATH = (
    MODEL_DIR / "stage2a_localizer_training_audit.json"
)
MANIFEST_PATH = (
    META_DIR / "stage1b_localizer_preprocessed_manifest.csv"
)
ROI_LEDGER_PATH = (
    QC_DIR / "stage0l_c_full_development_roi_geometry_ledger.csv"
)
CROP_PROTOCOL_PATH = (
    META_DIR / "stage0l_f_final_crop_geometry_protocol.json"
)

CASE_LEDGER_PATH = (
    QC_DIR / "stage2b_validation_crop_coverage_case_ledger.csv"
)
FAILURE_LEDGER_PATH = (
    QC_DIR / "stage2b_validation_crop_coverage_failures.csv"
)
SUMMARY_PATH = (
    QC_DIR / "stage2b_validation_crop_coverage_summary.csv"
)
DECISION_PATH = (
    META_DIR / "stage2b_localizer_crop_coverage_decision.json"
)
AUDIT_PATH = (
    QC_DIR / "stage2b_validation_crop_coverage_audit.json"
)

EXPECTED_VALIDATION = 295
LOCALIZER_CANVAS_SHAPE = np.asarray([96, 96, 160], dtype=float)

R4_SHAPE = np.asarray([224, 192, 112], dtype=float)
R4_SPACING_MM = np.asarray([1.25, 1.25, 2.0], dtype=float)
R4_FOV_MM = R4_SHAPE * R4_SPACING_MM

M1_SHAPE = np.asarray([256, 224, 128], dtype=float)
M1_SPACING_MM = np.asarray([1.25, 1.25, 2.0], dtype=float)
M1_FOV_MM = M1_SHAPE * M1_SPACING_MM

TRAIN_MEDIAN_NORMALIZED_CENTER = np.asarray(
    [0.566406, 0.465820, 0.569439],
    dtype=float,
)
LOCKED_DETERMINISTIC_VALIDATION_COVERAGE = 0.728814


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
        raise RuntimeError(f"Invalid three-element vector: {value}")
    return vector


def parse_bbox(value):
    parsed = json.loads(str(value))
    minimum = np.asarray(parsed["minimum"], dtype=float)
    maximum = np.asarray(parsed["maximum"], dtype=float)
    if (
        minimum.shape != (3,)
        or maximum.shape != (3,)
        or not np.all(np.isfinite(minimum))
        or not np.all(np.isfinite(maximum))
        or np.any(maximum < minimum)
    ):
        raise RuntimeError(f"Invalid ROI bounding box: {value}")
    return minimum, maximum


def original_bbox_edges_to_canonical(
    minimum,
    maximum,
    original_shape,
    original_orientation,
):
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
    original_to_canonical = np.linalg.inv(canonical_to_original)

    edge_corners = np.asarray(
        list(
            product(
                [minimum[0] - 0.5, maximum[0] + 0.5],
                [minimum[1] - 0.5, maximum[1] + 0.5],
                [minimum[2] - 0.5, maximum[2] + 0.5],
            )
        ),
        dtype=float,
    )
    canonical_corners = nib.affines.apply_affine(
        original_to_canonical,
        edge_corners,
    )
    return (
        np.min(canonical_corners, axis=0),
        np.max(canonical_corners, axis=0),
    )


def localizer_canvas_to_canonical(
    center_normalized,
    canonical_shape,
    canonical_spacing,
    effective_spacing,
    core_shape,
    pad_before,
):
    canvas_voxel = (
        center_normalized
        * (LOCALIZER_CANVAS_SHAPE - 1.0)
    )
    core_voxel = canvas_voxel - pad_before
    scale = effective_spacing / canonical_spacing
    offset = (
        (canonical_shape - 1.0) / 2.0
        - scale * ((core_shape - 1.0) / 2.0)
    )
    canonical_voxel = scale * core_voxel + offset
    return canonical_voxel


def evaluate_crop(
    center_canonical,
    bbox_edge_minimum_canonical,
    bbox_edge_maximum_canonical,
    canonical_spacing,
    crop_fov_mm,
):
    center_mm = center_canonical * canonical_spacing
    bbox_minimum_mm = (
        bbox_edge_minimum_canonical * canonical_spacing
    )
    bbox_maximum_mm = (
        bbox_edge_maximum_canonical * canonical_spacing
    )
    half_fov = crop_fov_mm / 2.0
    crop_minimum_mm = center_mm - half_fov
    crop_maximum_mm = center_mm + half_fov

    intersection_minimum = np.maximum(
        bbox_minimum_mm,
        crop_minimum_mm,
    )
    intersection_maximum = np.minimum(
        bbox_maximum_mm,
        crop_maximum_mm,
    )
    intersection_extent = np.maximum(
        intersection_maximum - intersection_minimum,
        0.0,
    )
    bbox_extent = bbox_maximum_mm - bbox_minimum_mm
    per_axis_fraction = np.clip(
        intersection_extent / bbox_extent,
        0.0,
        1.0,
    )
    volume_fraction = float(np.prod(per_axis_fraction))

    lower_margin = bbox_minimum_mm - crop_minimum_mm
    upper_margin = crop_maximum_mm - bbox_maximum_mm
    per_axis_margin = np.minimum(lower_margin, upper_margin)
    minimum_margin = float(np.min(per_axis_margin))
    fully_contained = bool(np.all(per_axis_margin >= -1e-6))

    return {
        "fully_contained": fully_contained,
        "volume_fraction": volume_fraction,
        "per_axis_fraction": per_axis_fraction,
        "per_axis_margin_mm": per_axis_margin,
        "minimum_margin_mm": minimum_margin,
    }


def summarize_policy(frame, prefix, scope, scope_value):
    if scope == "ALL":
        subset = frame
    else:
        subset = frame.loc[frame[scope] == scope_value]

    full = subset[f"{prefix}_fully_contained"].astype(bool)
    fraction = subset[f"{prefix}_bbox_volume_fraction"].astype(float)
    margin = subset[f"{prefix}_minimum_margin_mm"].astype(float)

    return {
        "policy": prefix,
        "scope": (
            "ALL_VALIDATION"
            if scope == "ALL"
            else f"{scope}={scope_value}"
        ),
        "cases": int(len(subset)),
        "full_union_bbox_containment": float(full.mean()),
        "mean_union_bbox_volume_fraction": float(fraction.mean()),
        "median_union_bbox_volume_fraction": float(
            fraction.median()
        ),
        "p05_union_bbox_volume_fraction": float(
            np.percentile(fraction, 5)
        ),
        "minimum_union_bbox_volume_fraction": float(
            fraction.min()
        ),
        "median_minimum_margin_mm": float(margin.median()),
        "p05_minimum_margin_mm": float(
            np.percentile(margin, 5)
        ),
        "minimum_margin_mm": float(margin.min()),
        "failed_full_containment": int((~full).sum()),
    }


print("=" * 116)
print("STAGE 2B — VALIDATION LOCALIZER-TO-CROP COVERAGE AUDIT")
print("=" * 116)

for required_path in [
    PREDICTIONS_PATH,
    TRAINING_AUDIT_PATH,
    MANIFEST_PATH,
    ROI_LEDGER_PATH,
    CROP_PROTOCOL_PATH,
]:
    if not required_path.exists():
        raise FileNotFoundError(f"Required input missing:\n{required_path}")

predictions = pd.read_csv(PREDICTIONS_PATH)
manifest = pd.read_csv(MANIFEST_PATH)
roi = pd.read_csv(ROI_LEDGER_PATH)

for dataframe in [predictions, manifest, roi]:
    dataframe["study_id"] = dataframe["study_id"].map(
        normalize_study_id
    )

with open(TRAINING_AUDIT_PATH, "r", encoding="utf-8") as file:
    training_audit = json.load(file)
with open(CROP_PROTOCOL_PATH, "r", encoding="utf-8") as file:
    crop_protocol = json.load(file)

if training_audit.get("training_complete") is not True:
    raise RuntimeError("Stage 2A training is not complete.")
if training_audit.get("locked_test_cases_accessed") != 0:
    raise RuntimeError("Stage 2A accessed a locked test case.")
if "R4_224x192x112" not in json.dumps(crop_protocol):
    raise RuntimeError("The locked R4 crop protocol was not reproduced.")
if len(predictions) != EXPECTED_VALIDATION:
    raise RuntimeError(
        f"Expected {EXPECTED_VALIDATION} predictions, "
        f"observed {len(predictions)}."
    )
if predictions["study_id"].nunique() != EXPECTED_VALIDATION:
    raise RuntimeError("Validation prediction IDs are not unique.")

validation_manifest = manifest.loc[
    manifest["partition"] == "validation"
].copy()
if len(validation_manifest) != EXPECTED_VALIDATION:
    raise RuntimeError("Validation manifest count changed.")
if set(predictions["study_id"]) != set(
    validation_manifest["study_id"]
):
    raise RuntimeError(
        "Prediction IDs do not exactly match locked validation IDs."
    )

roi_validation = roi.loc[
    roi["study_id"].isin(predictions["study_id"])
].copy()
if len(roi_validation) != EXPECTED_VALIDATION:
    raise RuntimeError("Validation ROI ledger is incomplete.")
if roi_validation["study_id"].nunique() != EXPECTED_VALIDATION:
    raise RuntimeError("Validation ROI IDs are not unique.")

required_manifest_columns = [
    "study_id",
    "partition",
    "diagnostic_label",
    "annotation_type",
    "ct_original_shape_json",
    "ct_original_orientation",
    "canonical_shape_json",
    "canonical_spacing_mm_json",
    "effective_isotropic_spacing_mm",
    "resampled_core_shape_json",
    "pad_before_json",
]
for column in required_manifest_columns:
    if column not in validation_manifest.columns:
        raise RuntimeError(f"Manifest column missing: {column}")
if "pancreas_lesion_union_bbox_json" not in roi_validation.columns:
    raise RuntimeError(
        "ROI union bounding-box column is missing. "
        f"Available columns: {list(roi_validation.columns)}"
    )

merged = (
    predictions
    .merge(
        validation_manifest[required_manifest_columns],
        on="study_id",
        how="inner",
        validate="one_to_one",
    )
    .merge(
        roi_validation[
            ["study_id", "pancreas_lesion_union_bbox_json"]
        ],
        on="study_id",
        how="inner",
        validate="one_to_one",
    )
    .sort_values("study_id")
    .reset_index(drop=True)
)
if len(merged) != EXPECTED_VALIDATION:
    raise RuntimeError("Stage 2B merge is incomplete.")

rows = []

for _, row in merged.iterrows():
    study_id = row["study_id"]
    original_shape = parse_vector(
        row["ct_original_shape_json"],
        dtype=float,
    )
    canonical_shape = parse_vector(
        row["canonical_shape_json"],
        dtype=float,
    )
    canonical_spacing = parse_vector(
        row["canonical_spacing_mm_json"],
        dtype=float,
    )
    core_shape = parse_vector(
        row["resampled_core_shape_json"],
        dtype=float,
    )
    pad_before = parse_vector(
        row["pad_before_json"],
        dtype=float,
    )
    effective_spacing = float(
        row["effective_isotropic_spacing_mm"]
    )

    predicted_normalized = np.asarray(
        [
            row["predicted_center_x"],
            row["predicted_center_y"],
            row["predicted_center_z"],
        ],
        dtype=float,
    )
    true_normalized = np.asarray(
        [
            row["true_center_x"],
            row["true_center_y"],
            row["true_center_z"],
        ],
        dtype=float,
    )

    predicted_canonical = localizer_canvas_to_canonical(
        predicted_normalized,
        canonical_shape,
        canonical_spacing,
        effective_spacing,
        core_shape,
        pad_before,
    )
    true_pancreas_canonical = localizer_canvas_to_canonical(
        true_normalized,
        canonical_shape,
        canonical_spacing,
        effective_spacing,
        core_shape,
        pad_before,
    )
    deterministic_canonical = (
        TRAIN_MEDIAN_NORMALIZED_CENTER
        * (canonical_shape - 1.0)
    )

    union_minimum, union_maximum = parse_bbox(
        row["pancreas_lesion_union_bbox_json"]
    )
    union_edge_minimum_canonical, union_edge_maximum_canonical = (
        original_bbox_edges_to_canonical(
            union_minimum,
            union_maximum,
            original_shape,
            row["ct_original_orientation"],
        )
    )
    oracle_union_canonical = (
        union_edge_minimum_canonical
        + union_edge_maximum_canonical
    ) / 2.0

    recomputed_error_mm = float(
        np.linalg.norm(
            (predicted_normalized - true_normalized)
            * (LOCALIZER_CANVAS_SHAPE - 1.0)
        )
        * effective_spacing
    )
    recorded_error_mm = float(row["center_error_mm"])
    error_reproduction_difference = abs(
        recomputed_error_mm - recorded_error_mm
    )

    policy_centers = {
        "predicted_R4": (predicted_canonical, R4_FOV_MM),
        "true_pancreas_R4": (
            true_pancreas_canonical,
            R4_FOV_MM,
        ),
        "deterministic_R4": (
            deterministic_canonical,
            R4_FOV_MM,
        ),
        "oracle_union_R4": (
            oracle_union_canonical,
            R4_FOV_MM,
        ),
        "predicted_M1": (predicted_canonical, M1_FOV_MM),
    }

    result_row = {
        "study_id": study_id,
        "partition": row["partition"],
        "diagnostic_label": row["diagnostic_label"],
        "annotation_type": row["annotation_type"],
        "original_orientation": row["ct_original_orientation"],
        "effective_isotropic_spacing_mm": effective_spacing,
        "recorded_center_error_mm": recorded_error_mm,
        "recomputed_center_error_mm": recomputed_error_mm,
        "center_error_reproduction_difference_mm":
            error_reproduction_difference,
        "predicted_center_canonical_json": json.dumps(
            [float(value) for value in predicted_canonical]
        ),
        "true_pancreas_center_canonical_json": json.dumps(
            [float(value) for value in true_pancreas_canonical]
        ),
        "union_edge_minimum_canonical_json": json.dumps(
            [
                float(value)
                for value in union_edge_minimum_canonical
            ]
        ),
        "union_edge_maximum_canonical_json": json.dumps(
            [
                float(value)
                for value in union_edge_maximum_canonical
            ]
        ),
    }

    for policy_name, (center, fov) in policy_centers.items():
        result = evaluate_crop(
            center,
            union_edge_minimum_canonical,
            union_edge_maximum_canonical,
            canonical_spacing,
            fov,
        )
        result_row[f"{policy_name}_fully_contained"] = (
            result["fully_contained"]
        )
        result_row[f"{policy_name}_bbox_volume_fraction"] = (
            result["volume_fraction"]
        )
        result_row[f"{policy_name}_minimum_margin_mm"] = (
            result["minimum_margin_mm"]
        )
        result_row[f"{policy_name}_axis_margin_mm_json"] = (
            json.dumps(
                [
                    float(value)
                    for value in result["per_axis_margin_mm"]
                ]
            )
        )

    rows.append(result_row)

case_ledger = pd.DataFrame(rows)
atomic_write_csv(case_ledger, CASE_LEDGER_PATH)

summary_rows = []
policies = [
    "predicted_R4",
    "true_pancreas_R4",
    "deterministic_R4",
    "oracle_union_R4",
    "predicted_M1",
]

for policy in policies:
    summary_rows.append(
        summarize_policy(case_ledger, policy, "ALL", None)
    )
    for label in sorted(case_ledger["diagnostic_label"].unique()):
        summary_rows.append(
            summarize_policy(
                case_ledger,
                policy,
                "diagnostic_label",
                label,
            )
        )
    for annotation in sorted(case_ledger["annotation_type"].unique()):
        summary_rows.append(
            summarize_policy(
                case_ledger,
                policy,
                "annotation_type",
                annotation,
            )
        )

summary = pd.DataFrame(summary_rows)
atomic_write_csv(summary, SUMMARY_PATH)

failures = (
    case_ledger.loc[
        ~case_ledger["predicted_R4_fully_contained"].astype(bool)
    ]
    .sort_values(
        [
            "predicted_R4_bbox_volume_fraction",
            "predicted_R4_minimum_margin_mm",
        ]
    )
    .reset_index(drop=True)
)
atomic_write_csv(failures, FAILURE_LEDGER_PATH)

def overall_metric(policy, column):
    row = summary.loc[
        (summary["policy"] == policy)
        & (summary["scope"] == "ALL_VALIDATION")
    ]
    if len(row) != 1:
        raise RuntimeError(f"Summary row missing for {policy}")
    return float(row.iloc[0][column])


predicted_containment = overall_metric(
    "predicted_R4",
    "full_union_bbox_containment",
)
predicted_mean_fraction = overall_metric(
    "predicted_R4",
    "mean_union_bbox_volume_fraction",
)
predicted_p05_fraction = overall_metric(
    "predicted_R4",
    "p05_union_bbox_volume_fraction",
)
true_pancreas_containment = overall_metric(
    "true_pancreas_R4",
    "full_union_bbox_containment",
)
deterministic_containment = overall_metric(
    "deterministic_R4",
    "full_union_bbox_containment",
)
oracle_containment = overall_metric(
    "oracle_union_R4",
    "full_union_bbox_containment",
)
m1_containment = overall_metric(
    "predicted_M1",
    "full_union_bbox_containment",
)

pdac_summary = summary.loc[
    (summary["policy"] == "predicted_R4")
    & (summary["scope"] == "diagnostic_label=PDAC")
]
if len(pdac_summary) != 1:
    raise RuntimeError("PDAC validation summary is missing.")
pdac_containment = float(
    pdac_summary.iloc[0]["full_union_bbox_containment"]
)

maximum_error_reproduction_difference = float(
    case_ledger[
        "center_error_reproduction_difference_mm"
    ].max()
)
stage2a_p95_error = float(
    training_audit["validation_metrics"]["p95_error_mm"]
)

acceptance_checks = {
    "Predicted R4 full union-bbox containment >= 95%":
        predicted_containment >= 0.95,
    "Predicted R4 PDAC full union-bbox containment >= 95%":
        pdac_containment >= 0.95,
    "Predicted R4 mean union-bbox volume coverage >= 99.5%":
        predicted_mean_fraction >= 0.995,
    "Predicted R4 p05 union-bbox volume coverage >= 95%":
        predicted_p05_fraction >= 0.95,
    "Predicted R4 improves deterministic coverage by >= 15 points":
        (
            predicted_containment
            - LOCKED_DETERMINISTIC_VALIDATION_COVERAGE
        ) >= 0.15,
    "Stage 2A validation p95 centre error <= 30 mm":
        stage2a_p95_error <= 30.0,
    "Predicted M1 sensitivity containment >= 99%":
        m1_containment >= 0.99,
}
localizer_crop_accepted = all(
    bool(value) for value in acceptance_checks.values()
)

readiness_checks = {
    "Exactly 295 locked validation predictions":
        len(predictions) == EXPECTED_VALIDATION,
    "Exactly 295 unique validation IDs":
        predictions["study_id"].nunique() == EXPECTED_VALIDATION,
    "Prediction IDs exactly match validation manifest":
        set(predictions["study_id"])
        == set(validation_manifest["study_id"]),
    "Every validation case has one union bounding box":
        len(case_ledger) == EXPECTED_VALIDATION,
    "All reconstructed crop values are finite":
        np.isfinite(
            case_ledger.select_dtypes(include=[np.number]).to_numpy()
        ).all(),
    "Stage 2A centre errors are reproduced within 0.001 mm":
        maximum_error_reproduction_difference <= 0.001,
    "Oracle R4 union-centred containment is 100%":
        oracle_containment == 1.0,
    "No locked test case was accessed":
        set(case_ledger["partition"]) == {"validation"},
}
all_readiness_checks_pass = all(
    bool(value) for value in readiness_checks.values()
)

decision = {
    "stage": "2B",
    "created_at_utc": utc_now(),
    "evaluation_role": "VALIDATION_ONLY_MODEL_AND_CROP_SELECTION",
    "validation_cases": EXPECTED_VALIDATION,
    "crop_R4": {
        "shape": [int(value) for value in R4_SHAPE],
        "spacing_mm": [
            float(value) for value in R4_SPACING_MM
        ],
        "fov_mm": [float(value) for value in R4_FOV_MM],
    },
    "sensitivity_crop_M1": {
        "shape": [int(value) for value in M1_SHAPE],
        "spacing_mm": [
            float(value) for value in M1_SPACING_MM
        ],
        "fov_mm": [float(value) for value in M1_FOV_MM],
    },
    "predicted_R4_full_union_bbox_containment":
        predicted_containment,
    "predicted_R4_PDAC_full_union_bbox_containment":
        pdac_containment,
    "predicted_R4_mean_union_bbox_volume_fraction":
        predicted_mean_fraction,
    "predicted_R4_p05_union_bbox_volume_fraction":
        predicted_p05_fraction,
    "true_pancreas_R4_full_union_bbox_containment":
        true_pancreas_containment,
    "deterministic_R4_independently_recomputed_containment":
        deterministic_containment,
    "locked_deterministic_validation_coverage":
        LOCKED_DETERMINISTIC_VALIDATION_COVERAGE,
    "oracle_union_R4_containment": oracle_containment,
    "predicted_M1_sensitivity_containment": m1_containment,
    "failed_predicted_R4_cases": int(len(failures)),
    "acceptance_checks": {
        key: bool(value)
        for key, value in acceptance_checks.items()
    },
    "localizer_crop_accepted": localizer_crop_accepted,
    "decision": (
        "ACCEPT_R4_WITH_LEARNED_LOCALIZER"
        if localizer_crop_accepted
        else "VALIDATION_CROP_POLICY_REVIEW_REQUIRED"
    ),
    "locked_test_cases_accessed": 0,
    "case_ledger_path": str(CASE_LEDGER_PATH),
    "failure_ledger_path": str(FAILURE_LEDGER_PATH),
    "summary_path": str(SUMMARY_PATH),
}
atomic_write_json(decision, DECISION_PATH)

audit = {
    "stage": "2B",
    "created_at_utc": utc_now(),
    "result": (
        "PASS_ACCEPT_R4_WITH_LEARNED_LOCALIZER"
        if (
            all_readiness_checks_pass
            and localizer_crop_accepted
        )
        else (
            "PASS_EVALUATION_COMPLETE_POLICY_REVIEW_REQUIRED"
            if all_readiness_checks_pass
            else "FAIL"
        )
    ),
    "readiness_checks": {
        key: bool(value)
        for key, value in readiness_checks.items()
    },
    "acceptance_checks": {
        key: bool(value)
        for key, value in acceptance_checks.items()
    },
    "maximum_center_error_reproduction_difference_mm":
        maximum_error_reproduction_difference,
    "locked_test_cases_accessed": 0,
    "decision_path": str(DECISION_PATH),
}
atomic_write_json(audit, AUDIT_PATH)

print()
print("-" * 116)
print("VALIDATION CROP-COVERAGE COMPARISON")
print("-" * 116)
display_columns = [
    "policy",
    "cases",
    "full_union_bbox_containment",
    "mean_union_bbox_volume_fraction",
    "p05_union_bbox_volume_fraction",
    "median_minimum_margin_mm",
    "p05_minimum_margin_mm",
    "failed_full_containment",
]
print(
    summary.loc[
        summary["scope"] == "ALL_VALIDATION",
        display_columns,
    ].to_string(index=False)
)

print()
print("-" * 116)
print("ACCEPTANCE CHECKS")
print("-" * 116)
for check, passed in acceptance_checks.items():
    print(f"  {check}: {bool(passed)}")

print()
print("-" * 116)
print("READINESS CHECKS")
print("-" * 116)
for check, passed in readiness_checks.items():
    print(f"  {check}: {bool(passed)}")

print()
print("Case ledger:")
print(CASE_LEDGER_PATH)
print()
print("Failure ledger:")
print(FAILURE_LEDGER_PATH)
print()
print("Summary:")
print(SUMMARY_PATH)
print()
print("Decision:")
print(DECISION_PATH)
print()
print("Audit:")
print(AUDIT_PATH)
print()
print("=" * 116)
print(f"STAGE 2B RESULT: {audit['result']}")
print("=" * 116)

if not all_readiness_checks_pass:
    failed = [
        check
        for check, passed in readiness_checks.items()
        if not bool(passed)
    ]
    raise RuntimeError(
        f"Stage 2B readiness failure: {failed}"
    )
