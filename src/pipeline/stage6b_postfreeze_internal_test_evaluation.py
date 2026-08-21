from pathlib import Path
from datetime import datetime, timezone
from tempfile import TemporaryDirectory
import gc
import gzip
import hashlib
import json
import os
import shutil

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt, map_coordinates
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    roc_curve,
)


# =============================================================================
# STAGE 6B — POST-FREEZE INTERNAL-TEST EVALUATION
# =============================================================================

PROJECT_ROOT = Path(os.environ.get("PDAC_PROJECT_ROOT", "/content/drive/MyDrive/PDAC_Public_Q1_Project"))
RUNTIME_ROOT = Path(os.environ.get("PDAC_RUNTIME_ROOT", "/content"))
RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
RAW_ROOT = PROJECT_ROOT / "00_Raw" / "PANORAMA"
META_DIR = PROJECT_ROOT / "01_Metadata"
QC_DIR = PROJECT_ROOT / "02_Quality_Control"

SPLIT_PATH = META_DIR / "stage0f_locked_study_split_index.csv"
PREDICTION_MANIFEST_PATH = (
    META_DIR / "stage6a_blind_internal_test_prediction_manifest.csv"
)
PREDICTION_FREEZE_PATH = (
    META_DIR / "stage6a_blind_internal_test_prediction_freeze.json"
)
STAGE6A_AUDIT_PATH = (
    QC_DIR / "stage6a_blind_internal_test_inference_audit.json"
)
STAGE6A_LEDGER_PATH = (
    QC_DIR / "stage6a_blind_internal_test_resume_ledger.csv"
)
FROC_PROTOCOL_PATH = META_DIR / "stage5d_detection_and_froc_protocol.json"

GEOMETRY_PATH = QC_DIR / "stage6b_internal_test_geometry_audit.csv"
CASE_METRICS_PATH = QC_DIR / "stage6b_internal_test_case_metrics.csv"
CANDIDATE_TRUTH_PATH = QC_DIR / "stage6b_internal_test_candidate_truth_ledger.csv"
FROC_CURVE_PATH = QC_DIR / "stage6b_internal_test_froc_curve.csv"
FROC_POINTS_PATH = QC_DIR / "stage6b_internal_test_froc_operating_points.csv"
ROC_CURVE_PATH = QC_DIR / "stage6b_internal_test_roc_curve.csv"
SUMMARY_PATH = QC_DIR / "stage6b_internal_test_metric_summary.csv"
BOOTSTRAP_PATH = QC_DIR / "stage6b_internal_test_bootstrap_95ci.csv"
SUBGROUP_PATH = QC_DIR / "stage6b_internal_test_subgroup_summary.csv"
AUDIT_PATH = QC_DIR / "stage6b_internal_test_evaluation_audit.json"
GEOMETRY_REVIEW_PATH = QC_DIR / "stage6b_geometry_review_required.json"
TARGETED_GEOMETRY_DIAGNOSTIC_PATH = (
    QC_DIR / "stage6b_100118_geometry_visual_qc.json"
)
TARGETED_GEOMETRY_OVERLAY_PATH = (
    QC_DIR / "stage6b_100118_index_alignment_overlay.png"
)
GEOMETRY_RESOLUTION_PATH = (
    QC_DIR / "stage6b_100118_geometry_resolution.json"
)
VISUALLY_CONFIRMED_INDEX_CASE = "100118_00001"

EXPECTED_CASES = 293
EXPECTED_PDAC = 87
EXPECTED_NON_PDAC = 206
EXPECTED_PARTITION = "internal_test"
EXPECTED_SOURCE = "PANORAMA_LOCAL"

CROP_SHAPE = np.asarray([240, 192, 128], dtype=int)
CROP_SPACING_MM = np.asarray([1.25, 1.25, 2.0], dtype=float)
SEGMENTATION_THRESHOLD = 0.50
REFERENCE_LOCALIZATION_TOLERANCE_MM = 5.0
FROC_TARGET_FP_PER_CASE = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
STANDARD_FROC_TARGETS = [0.25, 0.5, 1.0, 2.0, 4.0]
PROBABILITY_SCALE = 65535.0
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260728
MASK_RESAMPLE_Z_CHUNK = 8
IO_CHUNK_BYTES = 8 * 1024 * 1024
CANONICAL_AXCODES = ("L", "P", "S")

# Geometry differences below these tolerances are numerical header noise only.
AFFINE_RTOL = 1e-4
AFFINE_ATOL = 1e-3


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_write_csv(frame, path):
    temporary = Path(str(path) + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_write_json(payload, path):
    temporary = Path(str(path) + ".tmp")
    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def sha256_file(path, chunk_size=IO_CHUNK_BYTES):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def truth_flags(series):
    return (
        series.fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "pass", "complete", "completed"})
    )


def replace_case_rows(frame, replacement, key="study_id"):
    if len(frame):
        replacement_ids = set(replacement[key].astype(str))
        frame = frame.loc[~frame[key].astype(str).isin(replacement_ids)].copy()
    return pd.concat([frame, replacement], ignore_index=True)


def normalize_ids(series):
    return series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def parse_vector(value, dtype=float):
    vector = np.asarray(json.loads(str(value)), dtype=dtype)
    if vector.shape != (3,) or not np.all(np.isfinite(vector.astype(float))):
        raise RuntimeError(f"Invalid three-axis vector: {value}")
    return vector


def parse_matrix(value):
    matrix = np.asarray(json.loads(str(value)), dtype=float)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise RuntimeError("Invalid 4x4 affine matrix.")
    return matrix


def resolve_column(frame, candidates, description):
    for name in candidates:
        if name in frame.columns:
            return name
    raise RuntimeError(
        f"Could not resolve {description}. Tried {candidates}. "
        f"Available columns: {list(frame.columns)}"
    )


def normalize_diagnostic_label(value):
    text = str(value).strip().lower().replace("_", "-")
    if text in {"pdac", "1", "1.0", "true"}:
        return "PDAC", 1
    if text in {"non-pdac", "nonpdac", "0", "0.0", "false"}:
        return "non-PDAC", 0
    raise RuntimeError(f"Unexpected diagnostic label: {value!r}")


def resolve_mask_path(study_id):
    manual = RAW_ROOT / "Manual_Labels" / f"{study_id}.nii.gz"
    automatic = RAW_ROOT / "Automatic_Labels" / f"{study_id}.nii.gz"
    exists = [("manual", manual, manual.exists()), ("automatic", automatic, automatic.exists())]
    found = [(kind, path) for kind, path, present in exists if present]
    if len(found) != 1:
        raise RuntimeError(
            f"{study_id}: expected exactly one persistent mask; found {len(found)}."
        )
    return found[0]


def canonical_header_geometry(image):
    raw_shape = np.asarray(image.shape[:3], dtype=int)
    raw_affine = np.asarray(image.affine, dtype=float)
    source_orientation = nib.orientations.io_orientation(raw_affine)
    target_orientation = nib.orientations.axcodes2ornt(CANONICAL_AXCODES)
    transform = nib.orientations.ornt_transform(source_orientation, target_orientation)
    canonical_affine = raw_affine @ nib.orientations.inv_ornt_aff(transform, raw_shape)
    permutation = np.argsort(transform[:, 0].astype(int))
    canonical_shape = raw_shape[permutation]
    canonical_spacing = nib.affines.voxel_sizes(canonical_affine)
    orientation = "".join(nib.aff2axcodes(canonical_affine))
    if orientation != "LPS":
        raise RuntimeError(f"Canonical header orientation failed: {orientation}")
    return {
        "raw_shape": raw_shape,
        "raw_spacing": nib.affines.voxel_sizes(raw_affine),
        "raw_orientation": "".join(nib.aff2axcodes(raw_affine)),
        "canonical_shape": canonical_shape,
        "canonical_spacing": np.asarray(canonical_spacing, dtype=float),
        "canonical_affine": canonical_affine,
        "transform": transform,
    }


def reconstruct_ct_canonical_affine(stage6a_row):
    crop_affine = parse_matrix(stage6a_row["crop_affine_json"])
    canonical_spacing = parse_vector(
        stage6a_row["canonical_ct_spacing_mm_json"], dtype=float
    )
    predicted_center = parse_vector(
        stage6a_row["predicted_center_canonical_json"], dtype=float
    )
    center_world = parse_vector(
        stage6a_row["predicted_center_world_mm_json"], dtype=float
    )
    directions = crop_affine[:3, :3] / CROP_SPACING_MM[None, :]
    norms = np.linalg.norm(directions, axis=0)
    if not np.allclose(norms, 1.0, rtol=1e-5, atol=1e-5):
        raise RuntimeError("Stored crop directions are not unit length.")
    affine = np.eye(4, dtype=float)
    affine[:3, :3] = directions * canonical_spacing[None, :]
    affine[:3, 3] = center_world - affine[:3, :3] @ predicted_center
    if "".join(nib.aff2axcodes(affine)) != "LPS":
        raise RuntimeError("Reconstructed CT canonical affine is not LPS.")
    return affine


def decompress_gzip(source, destination):
    with gzip.open(source, "rb") as compressed, open(destination, "wb") as output:
        shutil.copyfileobj(compressed, output, length=IO_CHUNK_BYTES)
        output.flush()
        os.fsync(output.fileno())


def canonical_mask_geometry(image):
    source_orientation = nib.orientations.io_orientation(image.affine)
    target_orientation = nib.orientations.axcodes2ornt(CANONICAL_AXCODES)
    transform = nib.orientations.ornt_transform(source_orientation, target_orientation)
    canonical_affine = image.affine @ nib.orientations.inv_ornt_aff(
        transform, image.shape[:3]
    )
    raw_array = image.dataobj.get_unscaled()
    canonical_array = nib.orientations.apply_orientation(raw_array, transform)
    return {
        "array": canonical_array,
        "affine": np.asarray(canonical_affine, dtype=float),
        "shape": np.asarray(canonical_array.shape, dtype=int),
    }


def load_confirmed_geometry_resolution(prediction_freeze_sha256):
    """Validate the targeted evidence created after the Stage 6B pause.

    The review established that study 100118_00001 has voxel-index-aligned
    CT/mask arrays but discordant raw orientation metadata (CT=LPI, mask=LPS).
    This function does not alter either source file.  It freezes the visual
    review decision together with hashes of the evidence used to make it.
    """
    if not TARGETED_GEOMETRY_DIAGNOSTIC_PATH.exists():
        raise RuntimeError(
            "Targeted geometry diagnostic is missing; the reviewed exception "
            "cannot be accepted."
        )
    if not TARGETED_GEOMETRY_OVERLAY_PATH.exists():
        raise RuntimeError(
            "Targeted geometry overlay is missing; the reviewed exception "
            "cannot be accepted."
        )
    with open(TARGETED_GEOMETRY_DIAGNOSTIC_PATH, "r", encoding="utf-8") as file:
        diagnostic = json.load(file)

    checks = {
        "diagnostic_study_matches": str(diagnostic.get("study_id"))
        == VISUALLY_CONFIRMED_INDEX_CASE,
        "raw_shapes_match": diagnostic.get("shape_match") is True,
        "raw_CT_orientation_is_LPI": diagnostic.get("ct_orientation") == "LPI",
        "raw_mask_orientation_is_LPS": diagnostic.get("mask_orientation") == "LPS",
        "raw_orientations_differ": diagnostic.get("orientation_match") is False,
        "model_inference_was_not_repeated": diagnostic.get("model_inference_repeated")
        is False,
        "predictions_were_not_modified": diagnostic.get("prediction_files_modified")
        is False,
        "raw_inputs_were_not_modified": diagnostic.get("raw_ct_or_mask_modified")
        is False,
        "metrics_were_not_computed_during_review": diagnostic.get(
            "metric_computation_performed"
        )
        is False,
        "non_PDAC_mask_has_no_lesion_label": int(diagnostic.get("lesion_voxels", -1))
        == 0,
        "pancreas_label_is_present": int(diagnostic.get("pancreas_voxels", 0)) > 0,
    }
    pancreas_stats = diagnostic.get("pancreas_hu_statistics", {})
    checks["paired_pancreas_CT_values_are_finite_and_plausible"] = bool(
        int(pancreas_stats.get("count", 0)) > 0
        and np.isfinite(float(pancreas_stats.get("median", np.nan)))
        and float(pancreas_stats.get("fraction_minus150_to_300", 0.0)) >= 0.99
    )
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(
            "Targeted geometry-resolution evidence failed: " + str(failed)
        )

    record = {
        "stage": "6B-GEOMETRY-RESOLUTION",
        "created_at_utc": utc_now(),
        "study_id": VISUALLY_CONFIRMED_INDEX_CASE,
        "decision": "PASS_INDEX_ALIGNMENT_CONFIRMED",
        "geometry_status": (
            "VISUALLY_CONFIRMED_INDEX_ALIGNED_HEADER_ORIENTATION_MISMATCH"
        ),
        "interpretation": (
            "CT and mask arrays are aligned by voxel index; the CT LPI versus "
            "mask LPS header-orientation discrepancy is metadata-only for this "
            "case. During evaluation, the mask array is transformed with the "
            "CT raw-to-canonical discrete orientation transform and assigned "
            "the reconstructed frozen CT canonical affine."
        ),
        "visual_review_basis": (
            "Six-slice overlay visually confirmed pancreas and vascular contours "
            "on the corresponding CT anatomy."
        ),
        "prediction_freeze_sha256": prediction_freeze_sha256,
        "diagnostic_sha256": sha256_file(TARGETED_GEOMETRY_DIAGNOSTIC_PATH),
        "overlay_sha256": sha256_file(TARGETED_GEOMETRY_OVERLAY_PATH),
        "raw_CT_or_mask_modified": False,
        "frozen_predictions_modified": False,
        "model_or_threshold_adjusted": False,
        "readiness_checks": checks,
    }
    atomic_write_json(record, GEOMETRY_RESOLUTION_PATH)
    return record


def index_aligned_mask_using_ct_geometry(image, stage6a_row):
    """Canonicalize a reviewed index-aligned mask using frozen CT geometry."""
    raw_shape = np.asarray(image.shape[:3], dtype=int)
    expected_raw_shape = parse_vector(stage6a_row["raw_ct_shape_json"], dtype=int).astype(int)
    if not np.array_equal(raw_shape, expected_raw_shape):
        raise RuntimeError(
            f"{VISUALLY_CONFIRMED_INDEX_CASE}: reviewed raw CT/mask shapes changed."
        )
    ct_raw_orientation = str(stage6a_row["raw_ct_orientation"]).strip()
    if ct_raw_orientation != "LPI":
        raise RuntimeError(
            f"{VISUALLY_CONFIRMED_INDEX_CASE}: frozen raw CT orientation changed."
        )
    source_orientation = nib.orientations.axcodes2ornt(tuple(ct_raw_orientation))
    target_orientation = nib.orientations.axcodes2ornt(CANONICAL_AXCODES)
    transform = nib.orientations.ornt_transform(source_orientation, target_orientation)
    raw_array = image.dataobj.get_unscaled()
    canonical_array = nib.orientations.apply_orientation(raw_array, transform)
    ct_affine = reconstruct_ct_canonical_affine(stage6a_row)
    expected_canonical_shape = parse_vector(
        stage6a_row["canonical_ct_shape_json"], dtype=int
    ).astype(int)
    if not np.array_equal(np.asarray(canonical_array.shape, dtype=int), expected_canonical_shape):
        raise RuntimeError(
            f"{VISUALLY_CONFIRMED_INDEX_CASE}: reviewed canonical mask shape changed."
        )
    return {
        "array": canonical_array,
        "affine": np.asarray(ct_affine, dtype=float),
        "shape": np.asarray(canonical_array.shape, dtype=int),
    }


def resample_mask_to_crop(mask_geometry, crop_affine):
    inverse = np.linalg.inv(mask_geometry["affine"])
    output = np.empty(tuple(CROP_SHAPE), dtype=np.uint8)
    x_indices = np.arange(CROP_SHAPE[0], dtype=float)
    y_indices = np.arange(CROP_SHAPE[1], dtype=float)
    for z_start in range(0, CROP_SHAPE[2], MASK_RESAMPLE_Z_CHUNK):
        z_end = min(z_start + MASK_RESAMPLE_Z_CHUNK, CROP_SHAPE[2])
        z_indices = np.arange(z_start, z_end, dtype=float)
        grid = np.meshgrid(x_indices, y_indices, z_indices, indexing="ij")
        crop_voxels = np.stack([axis.reshape(-1) for axis in grid], axis=0)
        homogeneous = np.vstack(
            [crop_voxels, np.ones((1, crop_voxels.shape[1]), dtype=float)]
        )
        world = crop_affine @ homogeneous
        coordinates = (inverse @ world)[:3]
        values = map_coordinates(
            mask_geometry["array"],
            coordinates,
            order=0,
            mode="constant",
            cval=0,
            prefilter=False,
        )
        if not np.all(np.isfinite(values)):
            raise RuntimeError("Resampled mask contains non-finite values.")
        if not np.allclose(values, np.rint(values), rtol=0, atol=1e-6):
            raise RuntimeError("Resampled mask is not integer-valued.")
        values = np.rint(values).astype(np.int16)
        unique = set(np.unique(values).tolist())
        if not unique.issubset(set(range(7))):
            raise RuntimeError(f"Unexpected mask labels after resampling: {sorted(unique)}")
        output[:, :, z_start:z_end] = values.reshape(
            (CROP_SHAPE[0], CROP_SHAPE[1], z_end - z_start)
        ).astype(np.uint8)
    return output


def dice_binary(prediction, target, empty_value=0.0):
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    denominator = int(prediction.sum()) + int(target.sum())
    if denominator == 0:
        return float(empty_value)
    intersection = int(np.logical_and(prediction, target).sum())
    return float(2.0 * intersection / denominator)


def safe_auc(labels, scores):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(np.unique(labels)) < 2:
        return np.nan
    return float(roc_auc_score(labels, scores))


def safe_ap(labels, scores):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if labels.sum() == 0:
        return np.nan
    return float(average_precision_score(labels, scores))


def froc_from_candidate_truth(candidate_truth, total_cases, total_references):
    if len(candidate_truth) == 0:
        return pd.DataFrame(
            [{
                "probability_threshold": 1.0,
                "true_positive_lesions": 0,
                "false_positive_candidates": 0,
                "total_reference_lesions": total_references,
                "test_cases": total_cases,
                "sensitivity": 0.0,
                "false_positives_per_case": 0.0,
            }]
        )
    ranked = candidate_truth.sort_values(
        ["candidate_score", "study_id", "candidate_rank"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    ranked["cumulative_true_positives"] = truth_flags(
        ranked["is_true_positive_candidate"]
    ).astype(int).cumsum()
    ranked["cumulative_false_positives"] = truth_flags(
        ranked["is_false_positive_candidate"]
    ).astype(int).cumsum()
    curve_rows = ranked.groupby("candidate_score", sort=False).tail(1).copy()
    curve = pd.DataFrame(
        {
            "probability_threshold": curve_rows["candidate_score"].astype(float),
            "true_positive_lesions": curve_rows["cumulative_true_positives"].astype(int),
            "false_positive_candidates": curve_rows["cumulative_false_positives"].astype(int),
        }
    )
    curve["total_reference_lesions"] = int(total_references)
    curve["test_cases"] = int(total_cases)
    curve["sensitivity"] = curve["true_positive_lesions"] / float(total_references)
    curve["false_positives_per_case"] = curve["false_positive_candidates"] / float(total_cases)
    initial = pd.DataFrame(
        [{
            "probability_threshold": float(ranked["candidate_score"].max() + 1e-6),
            "true_positive_lesions": 0,
            "false_positive_candidates": 0,
            "total_reference_lesions": int(total_references),
            "test_cases": int(total_cases),
            "sensitivity": 0.0,
            "false_positives_per_case": 0.0,
        }]
    )
    return pd.concat([initial, curve], ignore_index=True).sort_values(
        ["false_positives_per_case", "sensitivity", "probability_threshold"],
        ascending=[True, True, False],
    ).reset_index(drop=True)


def froc_operating_points(curve):
    rows = []
    for target_fp in FROC_TARGET_FP_PER_CASE:
        eligible = curve[curve["false_positives_per_case"] <= target_fp + 1e-12]
        selected = (
            curve.iloc[0]
            if len(eligible) == 0
            else eligible.sort_values(
                ["sensitivity", "false_positives_per_case", "probability_threshold"],
                ascending=[False, False, False],
            ).iloc[0]
        )
        rows.append(
            {
                "target_false_positives_per_case": float(target_fp),
                "achieved_false_positives_per_case": float(selected["false_positives_per_case"]),
                "sensitivity": float(selected["sensitivity"]),
                "probability_threshold": float(selected["probability_threshold"]),
                "true_positive_lesions": int(selected["true_positive_lesions"]),
                "false_positive_candidates": int(selected["false_positive_candidates"]),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_ci(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


print("=" * 124)
print("STAGE 6B — POST-FREEZE INTERNAL-TEST GROUND-TRUTH EVALUATION")
print("=" * 124)

required_paths = [
    SPLIT_PATH,
    PREDICTION_MANIFEST_PATH,
    PREDICTION_FREEZE_PATH,
    STAGE6A_AUDIT_PATH,
    STAGE6A_LEDGER_PATH,
    FROC_PROTOCOL_PATH,
]
for required in required_paths:
    if not required.exists():
        raise FileNotFoundError(f"Required input missing:\n{required}")

with open(PREDICTION_FREEZE_PATH, "r", encoding="utf-8") as file:
    prediction_freeze = json.load(file)
with open(STAGE6A_AUDIT_PATH, "r", encoding="utf-8") as file:
    stage6a_audit = json.load(file)
with open(FROC_PROTOCOL_PATH, "r", encoding="utf-8") as file:
    froc_protocol = json.load(file)

if prediction_freeze.get("prediction_freeze_complete") is not True:
    raise RuntimeError("Stage 6A prediction freeze is incomplete.")
if stage6a_audit.get("all_checks_pass") is not True:
    raise RuntimeError("Stage 6A audit did not pass.")
if prediction_freeze.get("diagnostic_labels_accessed") != 0:
    raise RuntimeError("Stage 6A label blinding record changed.")
if prediction_freeze.get("test_masks_accessed") != 0:
    raise RuntimeError("Stage 6A mask blinding record changed.")
if froc_protocol.get("protocol_locked") is not True:
    raise RuntimeError("Stage 5D FROC protocol is not locked.")

deployment_threshold = float(froc_protocol["calibrated_probability_threshold"])
protocol_signature = str(froc_protocol["candidate_generation"]["protocol_signature"])
freeze_hash = sha256_file(PREDICTION_FREEZE_PATH)

manifest = pd.read_csv(PREDICTION_MANIFEST_PATH, dtype={"study_id": str, "patient_id": str})
manifest["study_id"] = normalize_ids(manifest["study_id"])
manifest["patient_id"] = normalize_ids(manifest["patient_id"])
stage6a = pd.read_csv(STAGE6A_LEDGER_PATH, dtype={"study_id": str, "patient_id": str})
stage6a["study_id"] = normalize_ids(stage6a["study_id"])
stage6a["patient_id"] = normalize_ids(stage6a["patient_id"])

if len(manifest) != EXPECTED_CASES or manifest["study_id"].nunique() != EXPECTED_CASES:
    raise RuntimeError("Frozen prediction manifest is not exactly 293 studies.")
if len(stage6a) != EXPECTED_CASES or stage6a["study_id"].nunique() != EXPECTED_CASES:
    raise RuntimeError("Stage 6A ledger is not exactly 293 studies.")
if set(manifest["study_id"]) != set(stage6a["study_id"]):
    raise RuntimeError("Stage 6A manifest and ledger identities differ.")
if set(manifest["candidate_protocol_signature"].astype(str)) != {protocol_signature}:
    raise RuntimeError("Frozen candidate protocol signature changed.")
if not np.allclose(
    manifest["deployment_threshold"].astype(float),
    deployment_threshold,
    rtol=0,
    atol=1e-8,
):
    raise RuntimeError("Frozen detection threshold changed.")

print("\nVerifying all 293 frozen prediction files before opening ground truth...")
for order, (_, row) in enumerate(manifest.sort_values("study_id").iterrows(), start=1):
    path = Path(str(row["output_path"]))
    if not path.exists():
        raise FileNotFoundError(f"Frozen prediction missing: {path}")
    if path.stat().st_size != int(row["output_size_bytes"]):
        raise RuntimeError(f"Frozen prediction size changed: {row['study_id']}")
    if sha256_file(path) != str(row["output_sha256"]):
        raise RuntimeError(f"Frozen prediction hash changed: {row['study_id']}")
    if order % 50 == 0 or order == EXPECTED_CASES:
        print(f"  Frozen prediction verification: {order}/{EXPECTED_CASES}")

print("Prediction freeze verified. Ground-truth access is now permitted for evaluation only.")

# The first Stage 6B run paused before metrics for exactly one geometry case.
# Its targeted post-freeze CT-mask overlay has since been visually reviewed.
# Validate and freeze that resolution before any metric calculation resumes.
geometry_resolution = load_confirmed_geometry_resolution(freeze_hash)
print(
    "Targeted geometry resolution verified: "
    f"{geometry_resolution['study_id']} — {geometry_resolution['decision']}"
)

split = pd.read_csv(SPLIT_PATH, dtype=str)
split["study_id"] = normalize_ids(split["study_id"])
partition_column = resolve_column(
    split, ["locked_partition", "partition"], "partition column"
)
patient_column = resolve_column(
    split, ["patient_id", "PANORAMA_patient_id"], "patient identifier"
)
label_column = resolve_column(
    split,
    ["diagnostic_label", "label_text", "label", "label_binary"],
    "diagnostic label",
)
source_column = resolve_column(
    split, ["source_group"], "source-group column"
)
internal = split.loc[
    split[partition_column].astype(str).str.strip() == EXPECTED_PARTITION
].copy()
internal["patient_id_resolved"] = normalize_ids(internal[patient_column])
labels = internal[label_column].map(normalize_diagnostic_label)
internal["diagnostic_label"] = labels.map(lambda value: value[0])
internal["label_binary"] = labels.map(lambda value: value[1]).astype(int)
if len(internal) != EXPECTED_CASES or internal["study_id"].nunique() != EXPECTED_CASES:
    raise RuntimeError("Locked internal-test cohort changed.")
if internal["label_binary"].value_counts().to_dict() != {0: EXPECTED_NON_PDAC, 1: EXPECTED_PDAC}:
    raise RuntimeError("Internal-test diagnostic counts changed.")
if set(internal[source_column].astype(str).str.strip()) != {EXPECTED_SOURCE}:
    raise RuntimeError("Internal-test source role changed.")
if set(internal["study_id"]) != set(manifest["study_id"]):
    raise RuntimeError("Ground-truth cohort IDs differ from frozen prediction IDs.")

evaluation_index = (
    manifest[["study_id", "patient_id", "output_path", "output_sha256"]]
    .merge(
        stage6a,
        on=["study_id", "patient_id"],
        how="inner",
        validate="one_to_one",
        suffixes=("_manifest", "_stage6a"),
    )
    .merge(
        internal[["study_id", "patient_id_resolved", "diagnostic_label", "label_binary"]],
        on="study_id",
        how="inner",
        validate="one_to_one",
    )
    .sort_values("study_id")
    .reset_index(drop=True)
)
if len(evaluation_index) != EXPECTED_CASES:
    raise RuntimeError("Evaluation-index merge is incomplete.")
if not (
    evaluation_index["patient_id"].astype(str)
    == evaluation_index["patient_id_resolved"].astype(str)
).all():
    raise RuntimeError("Patient identities changed between locks.")


# =============================================================================
# PRE-METRIC CT–MASK GEOMETRY AUDIT
# =============================================================================


print("\nAuditing post-freeze CT–mask geometry before computing any metric...")
geometry_rows = []
for order, (_, row) in enumerate(evaluation_index.iterrows(), start=1):
    study_id = str(row["study_id"])
    annotation_type, mask_path = resolve_mask_path(study_id)
    mask_image = nib.load(str(mask_path))
    mask_header = canonical_header_geometry(mask_image)
    ct_affine = reconstruct_ct_canonical_affine(row)
    ct_shape = parse_vector(row["canonical_ct_shape_json"], dtype=int).astype(int)
    ct_spacing = parse_vector(row["canonical_ct_spacing_mm_json"], dtype=float)
    shape_match = bool(np.array_equal(mask_header["canonical_shape"], ct_shape))
    spacing_match = bool(
        np.allclose(mask_header["canonical_spacing"], ct_spacing, rtol=AFFINE_RTOL, atol=AFFINE_ATOL)
    )
    affine_match = bool(
        np.allclose(mask_header["canonical_affine"], ct_affine, rtol=AFFINE_RTOL, atol=AFFINE_ATOL)
    )
    maximum_affine_difference = float(
        np.max(np.abs(mask_header["canonical_affine"] - ct_affine))
    )
    if shape_match and spacing_match and affine_match:
        geometry_status = "EXACT_OR_NUMERICAL_PHYSICAL_GEOMETRY_MATCH"
    elif (
        study_id == VISUALLY_CONFIRMED_INDEX_CASE
        and geometry_resolution["decision"] == "PASS_INDEX_ALIGNMENT_CONFIRMED"
        and str(row["raw_ct_orientation"]).strip() == "LPI"
        and mask_header["raw_orientation"] == "LPS"
        and np.array_equal(
            mask_header["raw_shape"],
            parse_vector(row["raw_ct_shape_json"], dtype=int).astype(int),
        )
        and np.allclose(
            mask_header["raw_spacing"],
            parse_vector(row["raw_ct_spacing_mm_json"], dtype=float),
            rtol=AFFINE_RTOL,
            atol=AFFINE_ATOL,
        )
    ):
        geometry_status = (
            "VISUALLY_CONFIRMED_INDEX_ALIGNED_HEADER_ORIENTATION_MISMATCH"
        )
    else:
        geometry_status = "POST_FREEZE_GEOMETRY_REVIEW_REQUIRED"
    geometry_rows.append(
        {
            "study_id": study_id,
            "patient_id": str(row["patient_id"]),
            "diagnostic_label": row["diagnostic_label"],
            "annotation_type": annotation_type,
            "mask_path": str(mask_path),
            "raw_mask_shape_json": json.dumps(mask_header["raw_shape"].astype(int).tolist()),
            "raw_mask_spacing_mm_json": json.dumps(mask_header["raw_spacing"].astype(float).tolist()),
            "raw_mask_orientation": mask_header["raw_orientation"],
            "canonical_mask_shape_json": json.dumps(mask_header["canonical_shape"].astype(int).tolist()),
            "canonical_mask_spacing_mm_json": json.dumps(mask_header["canonical_spacing"].astype(float).tolist()),
            "canonical_ct_shape_json": json.dumps(ct_shape.tolist()),
            "canonical_ct_spacing_mm_json": json.dumps(ct_spacing.tolist()),
            "shape_match": shape_match,
            "spacing_match": spacing_match,
            "affine_match": affine_match,
            "maximum_absolute_affine_difference": maximum_affine_difference,
            "geometry_status": geometry_status,
        }
    )
    del mask_image
    if order % 50 == 0 or order == EXPECTED_CASES:
        print(f"  Geometry audit: {order}/{EXPECTED_CASES}")

geometry = pd.DataFrame(geometry_rows)
atomic_write_csv(geometry, GEOMETRY_PATH)
review = geometry.loc[
    geometry["geometry_status"] == "POST_FREEZE_GEOMETRY_REVIEW_REQUIRED"
].copy()
exact_geometry_count = int(
    (geometry["geometry_status"] == "EXACT_OR_NUMERICAL_PHYSICAL_GEOMETRY_MATCH").sum()
)
reviewed_geometry_count = int(
    (
        geometry["geometry_status"]
        == "VISUALLY_CONFIRMED_INDEX_ALIGNED_HEADER_ORIENTATION_MISMATCH"
    ).sum()
)
print(f"Exact/numerical physical matches: {exact_geometry_count}/{EXPECTED_CASES}")
print(f"Visually confirmed index-aligned header mismatches: {reviewed_geometry_count}")
print(f"Geometry review required: {len(review)}")

if len(review):
    review_record = {
        "stage": "6B",
        "created_at_utc": utc_now(),
        "status": "PAUSED_BEFORE_METRICS_FOR_GEOMETRY_REVIEW",
        "prediction_freeze_sha256": freeze_hash,
        "prediction_freeze_remains_unchanged": True,
        "review_case_count": int(len(review)),
        "review_study_ids": review["study_id"].astype(str).tolist(),
        "metrics_computed": False,
        "model_or_threshold_adjusted": False,
        "geometry_audit_path": str(GEOMETRY_PATH),
    }
    atomic_write_json(review_record, GEOMETRY_REVIEW_PATH)
    print("\nReview cases:")
    print(
        review[[
            "study_id", "diagnostic_label", "annotation_type",
            "maximum_absolute_affine_difference",
        ]].to_string(index=False)
    )
    print("=" * 124)
    print("STAGE 6B STATUS: PAUSED — POST-FREEZE GEOMETRY REVIEW REQUIRED")
    print("=" * 124)
    raise SystemExit(0)


# =============================================================================
# MASK RESAMPLING + FROZEN-PREDICTION METRICS
# =============================================================================


print("\nGeometry gate passed. Computing internal-test metrics on frozen predictions...")
if CASE_METRICS_PATH.exists():
    case_metrics = pd.read_csv(
        CASE_METRICS_PATH, dtype={"study_id": str, "patient_id": str}
    )
    case_metrics["study_id"] = normalize_ids(case_metrics["study_id"])
    case_metrics = case_metrics.drop_duplicates("study_id", keep="last")
else:
    case_metrics = pd.DataFrame()

if CANDIDATE_TRUTH_PATH.exists():
    try:
        candidate_truth = pd.read_csv(
            CANDIDATE_TRUTH_PATH, dtype={"study_id": str, "patient_id": str}
        )
        candidate_truth["study_id"] = normalize_ids(candidate_truth["study_id"])
    except pd.errors.EmptyDataError:
        candidate_truth = pd.DataFrame()
else:
    candidate_truth = pd.DataFrame()

completed_ids = set()
if len(case_metrics) and "evaluation_complete" in case_metrics.columns:
    manifest_hashes = manifest.set_index("study_id")["output_sha256"].astype(str).to_dict()
    for _, completed_row in case_metrics.loc[
        truth_flags(case_metrics["evaluation_complete"])
    ].iterrows():
        study_id = str(completed_row["study_id"])
        expected_hash = manifest_hashes.get(study_id)
        if (
            expected_hash is not None
            and str(completed_row.get("prediction_sha256")) == expected_hash
            and str(completed_row.get("prediction_freeze_sha256")) == freeze_hash
            and str(completed_row.get("candidate_protocol_signature")) == protocol_signature
            and np.isclose(
                float(completed_row.get("locked_deployment_threshold")),
                deployment_threshold,
                rtol=0,
                atol=1e-8,
            )
            and np.isclose(
                float(completed_row.get("segmentation_threshold")),
                SEGMENTATION_THRESHOLD,
                rtol=0,
                atol=1e-8,
            )
        ):
            expected_candidates = int(completed_row.get("generated_candidates", -1))
            observed_candidates = int(
                (candidate_truth["study_id"].astype(str) == study_id).sum()
            ) if len(candidate_truth) else 0
            if observed_candidates == expected_candidates:
                completed_ids.add(study_id)

pending_evaluation = evaluation_index.loc[
    ~evaluation_index["study_id"].isin(completed_ids)
].copy().reset_index(drop=True)
print(f"Previously metric-completed: {len(completed_ids)}/{EXPECTED_CASES}")
print(f"Pending metric evaluation: {len(pending_evaluation)}")
print("Checkpoint frequency: after every case")

for order, (_, row) in enumerate(pending_evaluation.iterrows(), start=1):
    study_id = str(row["study_id"])
    annotation_type, mask_path = resolve_mask_path(study_id)
    prediction_path = Path(str(row["output_path_manifest"]))
    case_candidate_rows = []

    with np.load(prediction_path, allow_pickle=False) as prediction:
        probability_scale = float(np.asarray(prediction["probability_scale"]).item())
        if not np.isclose(probability_scale, PROBABILITY_SCALE, rtol=0, atol=1e-6):
            raise RuntimeError(f"{study_id}: probability scale changed.")
        pancreas_probability = (
            np.asarray(prediction["pancreas_probability_uint16"], dtype=np.float32)
            / probability_scale
        )
        lesion_probability = (
            np.asarray(prediction["lesion_probability_uint16"], dtype=np.float32)
            / probability_scale
        )
        crop_affine = np.asarray(prediction["crop_affine"], dtype=float)
        candidate_coordinates = np.asarray(
            prediction["candidate_coordinates_voxel"], dtype=int
        ).reshape(-1, 3)
        candidate_scores = np.asarray(prediction["candidate_scores"], dtype=float).reshape(-1)
        candidate_raw = np.asarray(
            prediction["candidate_raw_probabilities"], dtype=float
        ).reshape(-1)

    if pancreas_probability.shape != tuple(CROP_SHAPE) or lesion_probability.shape != tuple(CROP_SHAPE):
        raise RuntimeError(f"{study_id}: frozen probability geometry changed.")
    if len(candidate_coordinates) != len(candidate_scores) or len(candidate_scores) != len(candidate_raw):
        raise RuntimeError(f"{study_id}: frozen candidate arrays disagree.")
    if len(candidate_scores) and np.any(np.diff(candidate_scores) > 1e-7):
        raise RuntimeError(f"{study_id}: candidate ranking is not descending.")

    with TemporaryDirectory(prefix=f"stage6b_{study_id}_", dir=RUNTIME_ROOT) as temp_name:
        mask_nii_path = Path(temp_name) / "mask.nii"
        decompress_gzip(mask_path, mask_nii_path)
        mask_image = nib.load(str(mask_nii_path), mmap="r")
        if study_id == VISUALLY_CONFIRMED_INDEX_CASE:
            if geometry_resolution["decision"] != "PASS_INDEX_ALIGNMENT_CONFIRMED":
                raise RuntimeError(
                    f"{study_id}: reviewed geometry resolution is no longer valid."
                )
            mask_geometry = index_aligned_mask_using_ct_geometry(mask_image, row)
        else:
            mask_geometry = canonical_mask_geometry(mask_image)
        resampled_mask = resample_mask_to_crop(mask_geometry, crop_affine)

        pancreas_target = resampled_mask == 4
        lesion_target = resampled_mask == 1
        is_pdac = int(row["label_binary"]) == 1
        if is_pdac and not bool(lesion_target.any()):
            lesion_in_crop = False
        else:
            lesion_in_crop = bool(lesion_target.any())

        pancreas_prediction = pancreas_probability >= SEGMENTATION_THRESHOLD
        lesion_prediction = lesion_probability >= SEGMENTATION_THRESHOLD
        pancreas_dice = dice_binary(pancreas_prediction, pancreas_target, empty_value=0.0)
        lesion_dice = (
            dice_binary(lesion_prediction, lesion_target, empty_value=0.0)
            if is_pdac else np.nan
        )

        if is_pdac and lesion_in_crop:
            distance_to_lesion = distance_transform_edt(
                ~lesion_target, sampling=CROP_SPACING_MM
            )
        else:
            distance_to_lesion = None

        reference_already_matched = False
        localized_above_locked_threshold = False
        for candidate_index in range(len(candidate_scores)):
            coordinate = candidate_coordinates[candidate_index]
            if np.any(coordinate < 0) or np.any(coordinate >= CROP_SHAPE):
                raise RuntimeError(f"{study_id}: candidate lies outside frozen crop.")
            coordinate_tuple = tuple(int(v) for v in coordinate)
            reference_distance = (
                float(distance_to_lesion[coordinate_tuple])
                if distance_to_lesion is not None else np.nan
            )
            within_tolerance = bool(
                is_pdac
                and distance_to_lesion is not None
                and reference_distance <= REFERENCE_LOCALIZATION_TOLERANCE_MM
            )
            true_positive = bool(within_tolerance and not reference_already_matched)
            if true_positive:
                reference_already_matched = True
            false_positive = not true_positive
            above_threshold = bool(candidate_scores[candidate_index] >= deployment_threshold)
            if true_positive and above_threshold:
                localized_above_locked_threshold = True
            case_candidate_rows.append(
                {
                    "study_id": study_id,
                    "patient_id": str(row["patient_id"]),
                    "diagnostic_label": row["diagnostic_label"],
                    "label_binary": int(row["label_binary"]),
                    "annotation_type": annotation_type,
                    "candidate_rank": int(candidate_index + 1),
                    "candidate_score": float(candidate_scores[candidate_index]),
                    "candidate_raw_probability": float(candidate_raw[candidate_index]),
                    "candidate_x": int(coordinate[0]),
                    "candidate_y": int(coordinate[1]),
                    "candidate_z": int(coordinate[2]),
                    "reference_distance_mm": reference_distance,
                    "within_reference_tolerance": within_tolerance,
                    "is_true_positive_candidate": true_positive,
                    "is_false_positive_candidate": false_positive,
                    "above_locked_deployment_threshold": above_threshold,
                }
            )

        maximum_score = float(candidate_scores[0]) if len(candidate_scores) else 0.0
        case_positive = bool(maximum_score >= deployment_threshold)
        new_case_row = {
                "study_id": study_id,
                "patient_id": str(row["patient_id"]),
                "diagnostic_label": row["diagnostic_label"],
                "label_binary": int(row["label_binary"]),
                "annotation_type": annotation_type,
                "pancreas_target_voxels_in_crop": int(pancreas_target.sum()),
                "lesion_target_voxels_in_crop": int(lesion_target.sum()),
                "pancreas_target_present_in_crop": bool(pancreas_target.any()),
                "lesion_target_present_in_crop": lesion_in_crop,
                "pancreas_dice_at_0_5": pancreas_dice,
                "lesion_dice_at_0_5": lesion_dice,
                "generated_candidates": int(len(candidate_scores)),
                "maximum_candidate_score": maximum_score,
                "case_positive_at_locked_threshold": case_positive,
                "lesion_localized_at_locked_threshold": bool(localized_above_locked_threshold),
                "locked_deployment_threshold": deployment_threshold,
                "segmentation_threshold": SEGMENTATION_THRESHOLD,
                "prediction_sha256": str(row["output_sha256_manifest"]),
                "prediction_freeze_sha256": freeze_hash,
                "candidate_protocol_signature": protocol_signature,
                "evaluation_complete": True,
                "evaluated_at_utc": utc_now(),
            }

        new_candidate_frame = pd.DataFrame(
            case_candidate_rows
        )
        if len(candidate_truth):
            candidate_truth = candidate_truth.loc[
                candidate_truth["study_id"].astype(str) != study_id
            ].copy()
        if len(new_candidate_frame):
            candidate_truth = pd.concat(
                [candidate_truth, new_candidate_frame], ignore_index=True
            )
        if len(candidate_truth):
            candidate_truth = candidate_truth.sort_values(
                ["study_id", "candidate_rank"]
            ).reset_index(drop=True)
        else:
            candidate_truth = pd.DataFrame(
                columns=[
                    "study_id", "patient_id", "diagnostic_label", "label_binary",
                    "annotation_type", "candidate_rank", "candidate_score",
                    "candidate_raw_probability", "candidate_x", "candidate_y",
                    "candidate_z", "reference_distance_mm",
                    "within_reference_tolerance", "is_true_positive_candidate",
                    "is_false_positive_candidate", "above_locked_deployment_threshold",
                ]
            )
        case_metrics = replace_case_rows(
            case_metrics, pd.DataFrame([new_case_row])
        ).sort_values("study_id").reset_index(drop=True)
        atomic_write_csv(candidate_truth, CANDIDATE_TRUTH_PATH)
        atomic_write_csv(case_metrics, CASE_METRICS_PATH)

        del resampled_mask, pancreas_target, lesion_target
        del pancreas_probability, lesion_probability
        del pancreas_prediction, lesion_prediction, mask_geometry, mask_image
        if distance_to_lesion is not None:
            del distance_to_lesion
        gc.collect()

    durable_total = len(completed_ids) + order
    if order % 10 == 0 or order == len(pending_evaluation):
        print(f"  Evaluation: {durable_total}/{EXPECTED_CASES}")

if len(case_metrics) != EXPECTED_CASES or case_metrics["study_id"].nunique() != EXPECTED_CASES:
    raise RuntimeError("Case-metric ledger is incomplete.")
if case_metrics["label_binary"].value_counts().to_dict() != {0: EXPECTED_NON_PDAC, 1: EXPECTED_PDAC}:
    raise RuntimeError("Case-metric diagnostic counts changed.")


# =============================================================================
# ROC, FROC, LOCKED-THRESHOLD METRICS
# =============================================================================


labels_array = case_metrics["label_binary"].astype(int).to_numpy()
scores_array = case_metrics["maximum_candidate_score"].astype(float).to_numpy()
auc_value = float(roc_auc_score(labels_array, scores_array))
ap_value = float(average_precision_score(labels_array, scores_array))
fpr, tpr, roc_thresholds = roc_curve(labels_array, scores_array)
atomic_write_csv(
    pd.DataFrame(
        {
            "false_positive_rate": fpr,
            "true_positive_rate": tpr,
            "probability_threshold": roc_thresholds,
        }
    ),
    ROC_CURVE_PATH,
)

froc_curve = froc_from_candidate_truth(candidate_truth, EXPECTED_CASES, EXPECTED_PDAC)
froc_points = froc_operating_points(froc_curve)
atomic_write_csv(froc_curve, FROC_CURVE_PATH)
atomic_write_csv(froc_points, FROC_POINTS_PATH)

locked_candidates = candidate_truth.loc[
    candidate_truth["candidate_score"].astype(float) >= deployment_threshold
].copy()
locked_tp = int(truth_flags(locked_candidates["is_true_positive_candidate"]).sum())
locked_fp = int(truth_flags(locked_candidates["is_false_positive_candidate"]).sum())
locked_sensitivity = float(locked_tp / EXPECTED_PDAC)
locked_fp_per_case = float(locked_fp / EXPECTED_CASES)

positive_cases = case_metrics.loc[case_metrics["label_binary"] == 1].copy()
negative_cases = case_metrics.loc[case_metrics["label_binary"] == 0].copy()
negative_specificity = float(
    (~truth_flags(negative_cases["case_positive_at_locked_threshold"])).mean()
)
pdac_localization_sensitivity = float(
    truth_flags(positive_cases["lesion_localized_at_locked_threshold"]).mean()
)
pancreas_dice_mean = float(case_metrics["pancreas_dice_at_0_5"].mean())
pancreas_dice_median = float(case_metrics["pancreas_dice_at_0_5"].median())
pancreas_crop_coverage = float(
    truth_flags(case_metrics["pancreas_target_present_in_crop"]).mean()
)
lesion_dice_mean = float(positive_cases["lesion_dice_at_0_5"].mean())
lesion_dice_median = float(positive_cases["lesion_dice_at_0_5"].median())
lesion_crop_coverage = float(
    truth_flags(positive_cases["lesion_target_present_in_crop"]).mean()
)

standard_froc = froc_points.loc[
    froc_points["target_false_positives_per_case"].isin(STANDARD_FROC_TARGETS)
]
mean_froc_sensitivity = float(standard_froc["sensitivity"].mean())


# =============================================================================
# PATIENT-LEVEL STRATIFIED BOOTSTRAP 95% CONFIDENCE INTERVALS
# =============================================================================


print("\nComputing 2,000 stratified patient-level bootstrap replicates...")
rng = np.random.default_rng(BOOTSTRAP_SEED)
positive_indices = np.flatnonzero(labels_array == 1)
negative_indices = np.flatnonzero(labels_array == 0)
bootstrap_rows = []
for replicate in range(BOOTSTRAP_REPLICATES):
    sampled_positive = rng.choice(positive_indices, size=len(positive_indices), replace=True)
    sampled_negative = rng.choice(negative_indices, size=len(negative_indices), replace=True)
    sampled = np.concatenate([sampled_positive, sampled_negative])
    sampled_labels = labels_array[sampled]
    sampled_scores = scores_array[sampled]
    sampled_frame = case_metrics.iloc[sampled]
    sampled_positive_frame = case_metrics.iloc[sampled_positive]
    sampled_negative_frame = case_metrics.iloc[sampled_negative]
    bootstrap_rows.append(
        {
            "candidate_AUC": safe_auc(sampled_labels, sampled_scores),
            "candidate_average_precision": safe_ap(sampled_labels, sampled_scores),
            "mean_pancreas_Dice": float(sampled_frame["pancreas_dice_at_0_5"].mean()),
            "mean_PDAC_lesion_Dice": float(sampled_positive_frame["lesion_dice_at_0_5"].mean()),
            "PDAC_localization_sensitivity_locked_threshold": float(
                truth_flags(sampled_positive_frame["lesion_localized_at_locked_threshold"]).mean()
            ),
            "negative_specificity_locked_threshold": float(
                (~truth_flags(sampled_negative_frame["case_positive_at_locked_threshold"])).mean()
            ),
            "PDAC_lesion_crop_presence": float(
                truth_flags(sampled_positive_frame["lesion_target_present_in_crop"]).mean()
            ),
        }
    )

bootstrap = pd.DataFrame(bootstrap_rows)
ci_rows = []
point_estimates = {
    "candidate_AUC": auc_value,
    "candidate_average_precision": ap_value,
    "mean_pancreas_Dice": pancreas_dice_mean,
    "mean_PDAC_lesion_Dice": lesion_dice_mean,
    "PDAC_localization_sensitivity_locked_threshold": pdac_localization_sensitivity,
    "negative_specificity_locked_threshold": negative_specificity,
    "PDAC_lesion_crop_presence": lesion_crop_coverage,
}
for metric, point in point_estimates.items():
    lower, upper = bootstrap_ci(bootstrap[metric])
    ci_rows.append(
        {
            "metric": metric,
            "point_estimate": float(point),
            "bootstrap_95CI_lower": lower,
            "bootstrap_95CI_upper": upper,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_unit": "patient_stratified_by_diagnosis",
        }
    )
bootstrap_summary = pd.DataFrame(ci_rows)
atomic_write_csv(bootstrap_summary, BOOTSTRAP_PATH)


# =============================================================================
# SUMMARY + SUBGROUPS + FINAL AUDIT
# =============================================================================


summary_rows = [
    ("candidate_AUC", auc_value),
    ("candidate_average_precision", ap_value),
    ("mean_pancreas_Dice_at_0_5", pancreas_dice_mean),
    ("median_pancreas_Dice_at_0_5", pancreas_dice_median),
    ("pancreas_target_crop_presence", pancreas_crop_coverage),
    ("mean_PDAC_lesion_Dice_at_0_5", lesion_dice_mean),
    ("median_PDAC_lesion_Dice_at_0_5", lesion_dice_median),
    ("PDAC_lesion_crop_presence", lesion_crop_coverage),
    ("locked_detection_threshold", deployment_threshold),
    ("locked_threshold_lesion_sensitivity", locked_sensitivity),
    ("locked_threshold_PDAC_localization_sensitivity", pdac_localization_sensitivity),
    ("locked_threshold_false_positives_per_case", locked_fp_per_case),
    ("locked_threshold_negative_specificity", negative_specificity),
    ("mean_FROC_sensitivity_at_0.25_0.5_1_2_4_FP_per_case", mean_froc_sensitivity),
]
summary = pd.DataFrame(summary_rows, columns=["metric", "value"])
atomic_write_csv(summary, SUMMARY_PATH)

subgroup_rows = []
for annotation_type, group in case_metrics.groupby("annotation_type"):
    positive_group = group[group["label_binary"] == 1]
    negative_group = group[group["label_binary"] == 0]
    subgroup_rows.append(
        {
            "subgroup": f"annotation_type={annotation_type}",
            "cases": int(len(group)),
            "PDAC_cases": int((group["label_binary"] == 1).sum()),
            "non_PDAC_cases": int((group["label_binary"] == 0).sum()),
            "mean_pancreas_Dice": float(group["pancreas_dice_at_0_5"].mean()),
            "mean_PDAC_lesion_Dice": (
                float(positive_group["lesion_dice_at_0_5"].mean())
                if len(positive_group) else np.nan
            ),
            "PDAC_localization_sensitivity_locked_threshold": (
                float(truth_flags(positive_group["lesion_localized_at_locked_threshold"]).mean())
                if len(positive_group) else np.nan
            ),
            "negative_specificity_locked_threshold": (
                float((~truth_flags(negative_group["case_positive_at_locked_threshold"])).mean())
                if len(negative_group) else np.nan
            ),
        }
    )
subgroups = pd.DataFrame(subgroup_rows)
atomic_write_csv(subgroups, SUBGROUP_PATH)

readiness_checks = {
    "Stage 6A blind prediction freeze passed": prediction_freeze.get("prediction_freeze_complete") is True,
    "Exactly 293 frozen internal-test predictions": len(manifest) == EXPECTED_CASES,
    "All frozen prediction hashes were reverified before ground truth": True,
    "Ground-truth labels contain exactly 87 PDAC and 206 non-PDAC": case_metrics["label_binary"].value_counts().to_dict() == {0: EXPECTED_NON_PDAC, 1: EXPECTED_PDAC},
    "All 293 CT-mask geometries passed the post-freeze gate": len(review) == 0,
    "Exactly one reviewed header-orientation mismatch is explicitly resolved": reviewed_geometry_count == 1,
    "Every internal-test patient is represented exactly once": case_metrics["patient_id"].nunique() == EXPECTED_CASES,
    "Candidate AUC is finite": np.isfinite(auc_value),
    "Candidate average precision is finite": np.isfinite(ap_value),
    "Pancreas Dice is finite": np.isfinite(case_metrics["pancreas_dice_at_0_5"].astype(float)).all(),
    "All 87 PDAC lesion Dice values are finite": np.isfinite(positive_cases["lesion_dice_at_0_5"].astype(float)).all(),
    "FROC curve is non-empty": len(froc_curve) > 0,
    "Six prespecified FROC operating points exist": len(froc_points) == 6,
    "The deployment threshold exactly matches Stage 5D": np.allclose(case_metrics["locked_deployment_threshold"].astype(float), deployment_threshold, rtol=0, atol=1e-8),
    "No model fitting occurred in Stage 6B": True,
    "No threshold selection occurred in Stage 6B": True,
    "Bootstrap confidence intervals are finite": np.isfinite(bootstrap_summary[["point_estimate", "bootstrap_95CI_lower", "bootstrap_95CI_upper"]].to_numpy(dtype=float)).all(),
}
readiness_checks = {name: bool(value) for name, value in readiness_checks.items()}
all_checks_pass = bool(all(readiness_checks.values()))

audit = {
    "stage": "6B",
    "created_at_utc": utc_now(),
    "result": "PASS_INTERNAL_TEST_EVALUATION_COMPLETE" if all_checks_pass else "FAIL",
    "all_checks_pass": all_checks_pass,
    "readiness_checks": readiness_checks,
    "prediction_freeze_sha256": freeze_hash,
    "prediction_freeze_verified_before_ground_truth_access": True,
    "internal_test_cases": EXPECTED_CASES,
    "PDAC_cases": EXPECTED_PDAC,
    "non_PDAC_cases": EXPECTED_NON_PDAC,
    "model_training_performed": False,
    "threshold_selection_performed": False,
    "locked_detection_threshold": deployment_threshold,
    "segmentation_threshold": SEGMENTATION_THRESHOLD,
    "candidate_AUC": auc_value,
    "candidate_average_precision": ap_value,
    "mean_pancreas_Dice": pancreas_dice_mean,
    "pancreas_target_crop_presence": pancreas_crop_coverage,
    "mean_PDAC_lesion_Dice": lesion_dice_mean,
    "locked_threshold_lesion_sensitivity": locked_sensitivity,
    "locked_threshold_false_positives_per_case": locked_fp_per_case,
    "locked_threshold_negative_specificity": negative_specificity,
    "mean_FROC_sensitivity_standard_operating_points": mean_froc_sensitivity,
    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    "geometry_audit_path": str(GEOMETRY_PATH),
    "geometry_resolution_path": str(GEOMETRY_RESOLUTION_PATH),
    "visually_confirmed_index_aligned_header_mismatch_cases": reviewed_geometry_count,
    "case_metrics_path": str(CASE_METRICS_PATH),
    "candidate_truth_path": str(CANDIDATE_TRUTH_PATH),
    "froc_curve_path": str(FROC_CURVE_PATH),
    "froc_operating_points_path": str(FROC_POINTS_PATH),
    "roc_curve_path": str(ROC_CURVE_PATH),
    "summary_path": str(SUMMARY_PATH),
    "bootstrap_95ci_path": str(BOOTSTRAP_PATH),
    "subgroup_summary_path": str(SUBGROUP_PATH),
}
atomic_write_json(audit, AUDIT_PATH)

print("\n" + "-" * 124)
print("INTERNAL-TEST RESULTS")
print("-" * 124)
print(f"Candidate AUC: {auc_value:.6f}")
print(f"Candidate average precision: {ap_value:.6f}")
print(f"Mean pancreas Dice: {pancreas_dice_mean:.6f}")
print(f"Pancreas target present inside deployed crop: {pancreas_crop_coverage:.6f}")
print(f"Mean PDAC lesion Dice: {lesion_dice_mean:.6f}")
print(f"PDAC lesion present inside deployed crop: {lesion_crop_coverage:.6f}")
print(f"Locked threshold: {deployment_threshold:.6f}")
print(f"Lesion sensitivity at locked threshold: {locked_sensitivity:.6f}")
print(f"False positives/case at locked threshold: {locked_fp_per_case:.6f}")
print(f"Negative-case specificity at locked threshold: {negative_specificity:.6f}")
print(f"Mean FROC sensitivity (0.25,0.5,1,2,4 FP/case): {mean_froc_sensitivity:.6f}")

print("\nFROC OPERATING POINTS")
print("-" * 124)
print(froc_points.to_string(index=False))

print("\nBOOTSTRAP 95% CONFIDENCE INTERVALS")
print("-" * 124)
print(bootstrap_summary.to_string(index=False))

print("\nREADINESS CHECKS")
print("-" * 124)
for name, passed in readiness_checks.items():
    print(f"  {name}: {passed}")

print(f"\nCase metrics:\n{CASE_METRICS_PATH}")
print(f"FROC operating points:\n{FROC_POINTS_PATH}")
print(f"Bootstrap confidence intervals:\n{BOOTSTRAP_PATH}")
print(f"Final audit:\n{AUDIT_PATH}")
print("=" * 124)
print(
    "STAGE 6B RESULT: "
    + ("PASS — INTERNAL-TEST EVALUATION COMPLETE" if all_checks_pass else "FAIL")
)
print("=" * 124)
if not all_checks_pass:
    raise RuntimeError("Stage 6B failed one or more readiness checks.")
