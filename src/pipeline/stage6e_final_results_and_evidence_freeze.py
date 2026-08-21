from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os

import numpy as np
import pandas as pd


# =============================================================================
# STAGE 6E — FINAL RESULTS AND EVIDENCE FREEZE
# CPU only. No CT, mask, model inference, fitting, or threshold selection.
# =============================================================================

PROJECT_ROOT = Path(os.environ.get("PDAC_PROJECT_ROOT", "/content/drive/MyDrive/PDAC_Public_Q1_Project"))
META_DIR = PROJECT_ROOT / "01_Metadata"
QC_DIR = PROJECT_ROOT / "02_Quality_Control"

MODEL_SELECTION_PATH = META_DIR / "stage5c_supervised_model_selection.json"
FROC_PROTOCOL_PATH = META_DIR / "stage5d_detection_and_froc_protocol.json"

INTERNAL_PREDICTION_FREEZE_PATH = META_DIR / "stage6a_blind_internal_test_prediction_freeze.json"
INTERNAL_PREDICTION_MANIFEST_PATH = META_DIR / "stage6a_blind_internal_test_prediction_manifest.csv"
INTERNAL_INFERENCE_AUDIT_PATH = QC_DIR / "stage6a_blind_internal_test_inference_audit.json"
INTERNAL_GEOMETRY_RESOLUTION_PATH = QC_DIR / "stage6b_100118_geometry_resolution.json"
INTERNAL_GEOMETRY_PATH = QC_DIR / "stage6b_internal_test_geometry_audit.csv"
INTERNAL_AUDIT_PATH = QC_DIR / "stage6b_internal_test_evaluation_audit.json"
INTERNAL_SUMMARY_PATH = QC_DIR / "stage6b_internal_test_metric_summary.csv"
INTERNAL_FROC_PATH = QC_DIR / "stage6b_internal_test_froc_operating_points.csv"
INTERNAL_BOOTSTRAP_PATH = QC_DIR / "stage6b_internal_test_bootstrap_95ci.csv"

EXTERNAL_PREDICTION_FREEZE_PATH = META_DIR / "stage6c_blind_external_test_prediction_freeze.json"
EXTERNAL_PREDICTION_MANIFEST_PATH = META_DIR / "stage6c_blind_external_test_prediction_manifest.csv"
EXTERNAL_INFERENCE_AUDIT_PATH = QC_DIR / "stage6c_blind_external_test_inference_audit.json"
EXTERNAL_FREEZE_REPAIR_PATH = QC_DIR / "stage6c_r_external_freeze_metadata_repair.json"
EXTERNAL_GEOMETRY_PATH = QC_DIR / "stage6d_external_combined_geometry_audit.csv"
EXTERNAL_COMBINED_AUDIT_PATH = QC_DIR / "stage6d_external_combined_evaluation_audit.json"
EXTERNAL_COMBINED_SUMMARY_PATH = QC_DIR / "stage6d_external_combined_metric_summary.csv"
EXTERNAL_COMBINED_BOOTSTRAP_PATH = QC_DIR / "stage6d_external_combined_bootstrap_95ci.csv"
SOURCE_SUMMARY_PATH = QC_DIR / "stage6d_source_separated_external_summary.csv"
SOURCE_BOOTSTRAP_PATH = QC_DIR / "stage6d_source_separated_bootstrap_95ci.csv"
MSD_FROC_PATH = QC_DIR / "stage6d_msd_external_froc_operating_points.csv"
SOURCE_AUDIT_PATH = QC_DIR / "stage6d_source_separated_external_evaluation_audit.json"

EXTERNAL_LABEL_REPAIR_PATH = QC_DIR / "stage6e_external_combined_audit_label_repair.json"
FINAL_PERFORMANCE_PATH = QC_DIR / "stage6e_final_performance_summary.csv"
FINAL_CI_PATH = QC_DIR / "stage6e_final_bootstrap_95ci_summary.csv"
FINAL_FROC_COMPARISON_PATH = QC_DIR / "stage6e_internal_vs_msd_froc_comparison.csv"
EVIDENCE_HASH_MANIFEST_PATH = META_DIR / "stage6e_final_evidence_hash_manifest.csv"
FINAL_FREEZE_PATH = META_DIR / "stage6e_final_results_freeze.json"

EXPECTED_INTERNAL = {"cases": 293, "PDAC": 87, "non_PDAC": 206}
EXPECTED_MSD = {"cases": 194, "PDAC": 98, "non_PDAC": 96}
EXPECTED_NIH = {"cases": 80, "PDAC": 0, "non_PDAC": 80}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def atomic_write_json(payload, path):
    temporary = Path(str(path) + ".tmp")
    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False, allow_nan=False)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def atomic_write_csv(frame, path):
    temporary = Path(str(path) + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def metric_dict(path):
    frame = pd.read_csv(path)
    if not {"metric", "value"}.issubset(frame.columns):
        raise RuntimeError(f"Metric summary schema changed: {path}")
    return dict(zip(frame["metric"].astype(str), frame["value"].astype(float)))


def no_geometry_review(frame):
    if "geometry_status" not in frame.columns:
        return False
    return not frame["geometry_status"].astype(str).str.contains(
        "REVIEW_REQUIRED", regex=False
    ).any()


print("=" * 124)
print("STAGE 6E — FINAL RESULTS AND EVIDENCE FREEZE")
print("=" * 124)
print("GPU required: False")
print("CT volumes accessed: 0")
print("Segmentation masks accessed: 0")
print("Model inference performed: False")
print("Model fitting performed: False")
print("Threshold selection performed: False")

required_paths = [
    MODEL_SELECTION_PATH,
    FROC_PROTOCOL_PATH,
    INTERNAL_PREDICTION_FREEZE_PATH,
    INTERNAL_PREDICTION_MANIFEST_PATH,
    INTERNAL_INFERENCE_AUDIT_PATH,
    INTERNAL_GEOMETRY_RESOLUTION_PATH,
    INTERNAL_GEOMETRY_PATH,
    INTERNAL_AUDIT_PATH,
    INTERNAL_SUMMARY_PATH,
    INTERNAL_FROC_PATH,
    INTERNAL_BOOTSTRAP_PATH,
    EXTERNAL_PREDICTION_FREEZE_PATH,
    EXTERNAL_PREDICTION_MANIFEST_PATH,
    EXTERNAL_INFERENCE_AUDIT_PATH,
    EXTERNAL_FREEZE_REPAIR_PATH,
    EXTERNAL_GEOMETRY_PATH,
    EXTERNAL_COMBINED_AUDIT_PATH,
    EXTERNAL_COMBINED_SUMMARY_PATH,
    EXTERNAL_COMBINED_BOOTSTRAP_PATH,
    SOURCE_SUMMARY_PATH,
    SOURCE_BOOTSTRAP_PATH,
    MSD_FROC_PATH,
    SOURCE_AUDIT_PATH,
]
missing = [str(path) for path in required_paths if not path.exists()]
if missing:
    raise FileNotFoundError("Missing required final evidence:\n" + "\n".join(missing))

model_selection = read_json(MODEL_SELECTION_PATH)
froc_protocol = read_json(FROC_PROTOCOL_PATH)
internal_freeze = read_json(INTERNAL_PREDICTION_FREEZE_PATH)
internal_inference_audit = read_json(INTERNAL_INFERENCE_AUDIT_PATH)
internal_audit = read_json(INTERNAL_AUDIT_PATH)
external_freeze = read_json(EXTERNAL_PREDICTION_FREEZE_PATH)
external_inference_audit = read_json(EXTERNAL_INFERENCE_AUDIT_PATH)
external_combined_audit = read_json(EXTERNAL_COMBINED_AUDIT_PATH)
source_audit = read_json(SOURCE_AUDIT_PATH)


# =============================================================================
# CORRECT TWO DESCRIPTIVE READINESS LABELS FROM THE DERIVED COMBINED AUDIT
# =============================================================================

combined_hash_before = sha256_file(EXTERNAL_COMBINED_AUDIT_PATH)
checks = external_combined_audit.get("readiness_checks", {})
label_repairs = {
    "All 293 CT-mask geometries passed the post-freeze gate":
        "All 274 CT-mask geometries passed the post-freeze gate",
    "All 87 PDAC lesion Dice values are finite":
        "All 98 PDAC lesion Dice values are finite",
}
repaired_labels = []
for old_label, new_label in label_repairs.items():
    if old_label in checks:
        value = checks.pop(old_label)
        if new_label in checks and bool(checks[new_label]) != bool(value):
            raise RuntimeError(f"Conflicting readiness values for {new_label}")
        checks[new_label] = bool(value)
        repaired_labels.append({"old": old_label, "new": new_label})
    elif new_label not in checks:
        raise RuntimeError(f"Expected readiness label is missing: {old_label}")

external_combined_audit["readiness_checks"] = checks
external_combined_audit["all_checks_pass"] = bool(all(bool(v) for v in checks.values()))
external_combined_audit["metadata_label_repair"] = {
    "created_at_utc": utc_now(),
    "reason": "Correct inherited internal-test descriptive counts; numerical metrics unchanged",
    "repaired_labels": repaired_labels,
    "numerical_metrics_modified": False,
    "predictions_modified": False,
}
atomic_write_json(external_combined_audit, EXTERNAL_COMBINED_AUDIT_PATH)
combined_hash_after = sha256_file(EXTERNAL_COMBINED_AUDIT_PATH)

label_repair_record = {
    "stage": "6E-METADATA-REPAIR",
    "created_at_utc": utc_now(),
    "result": "PASS",
    "audit_path": str(EXTERNAL_COMBINED_AUDIT_PATH),
    "sha256_before": combined_hash_before,
    "sha256_after": combined_hash_after,
    "repaired_labels": repaired_labels,
    "numerical_metrics_modified": False,
    "prediction_files_modified": 0,
    "CT_or_masks_accessed": 0,
}
atomic_write_json(label_repair_record, EXTERNAL_LABEL_REPAIR_PATH)


# =============================================================================
# CANONICAL PERFORMANCE TABLES
# =============================================================================

internal_metrics = metric_dict(INTERNAL_SUMMARY_PATH)
external_combined_metrics = metric_dict(EXTERNAL_COMBINED_SUMMARY_PATH)
source_summary = pd.read_csv(SOURCE_SUMMARY_PATH)

msd_rows = source_summary.loc[source_summary["source_group"] == "MSD_EXTERNAL_MIXED"]
nih_rows = source_summary.loc[source_summary["source_group"] == "NIH_NEGATIVE_STRESS"]
if len(msd_rows) != 1 or len(nih_rows) != 1:
    raise RuntimeError("Source-separated summary must contain exactly one MSD and one NIH row.")
msd = msd_rows.iloc[0]
nih = nih_rows.iloc[0]

performance_rows = [
    {
        "cohort": "PANORAMA_INTERNAL_TEST",
        "analysis_role": "PRIMARY_INTERNAL_TEST",
        "cases": 293,
        "PDAC_cases": 87,
        "non_PDAC_cases": 206,
        "case_level_AUC": float(internal_audit["candidate_AUC"]),
        "average_precision": float(internal_audit["candidate_average_precision"]),
        "mean_pancreas_Dice": float(internal_audit["mean_pancreas_Dice"]),
        "mean_PDAC_lesion_Dice": float(internal_audit["mean_PDAC_lesion_Dice"]),
        "PDAC_lesion_crop_presence": float(internal_metrics["PDAC_lesion_crop_presence"]),
        "locked_threshold": float(internal_audit["locked_detection_threshold"]),
        "locked_threshold_lesion_sensitivity": float(internal_audit["locked_threshold_lesion_sensitivity"]),
        "locked_threshold_false_positives_per_case": float(internal_audit["locked_threshold_false_positives_per_case"]),
        "locked_threshold_negative_specificity": float(internal_audit["locked_threshold_negative_specificity"]),
        "mean_FROC_sensitivity": float(internal_audit["mean_FROC_sensitivity_standard_operating_points"]),
    },
    {
        "cohort": "EXTERNAL_COMBINED",
        "analysis_role": "SECONDARY_POOLED_EXTERNAL",
        "cases": 274,
        "PDAC_cases": 98,
        "non_PDAC_cases": 176,
        "case_level_AUC": float(external_combined_audit["candidate_AUC"]),
        "average_precision": float(external_combined_audit["candidate_average_precision"]),
        "mean_pancreas_Dice": float(external_combined_audit["mean_pancreas_Dice"]),
        "mean_PDAC_lesion_Dice": float(external_combined_audit["mean_PDAC_lesion_Dice"]),
        "PDAC_lesion_crop_presence": float(external_combined_metrics["PDAC_lesion_crop_presence"]),
        "locked_threshold": float(external_combined_audit["locked_detection_threshold"]),
        "locked_threshold_lesion_sensitivity": float(external_combined_audit["locked_threshold_lesion_sensitivity"]),
        "locked_threshold_false_positives_per_case": float(external_combined_audit["locked_threshold_false_positives_per_case"]),
        "locked_threshold_negative_specificity": float(external_combined_audit["locked_threshold_negative_specificity"]),
        "mean_FROC_sensitivity": float(external_combined_audit["mean_FROC_sensitivity_standard_operating_points"]),
    },
    {
        "cohort": "MSD_EXTERNAL_MIXED",
        "analysis_role": "PRIMARY_EXTERNAL_MIXED",
        "cases": int(msd["cases"]),
        "PDAC_cases": int(msd["PDAC_cases"]),
        "non_PDAC_cases": int(msd["non_PDAC_cases"]),
        "case_level_AUC": float(msd["case_level_AUC"]),
        "average_precision": float(msd["case_level_average_precision"]),
        "mean_pancreas_Dice": float(msd["mean_pancreas_Dice"]),
        "mean_PDAC_lesion_Dice": float(msd["mean_PDAC_lesion_Dice"]),
        "PDAC_lesion_crop_presence": float(msd["PDAC_lesion_crop_presence"]),
        "locked_threshold": float(msd["locked_threshold"]),
        "locked_threshold_lesion_sensitivity": float(msd["locked_threshold_lesion_sensitivity"]),
        "locked_threshold_false_positives_per_case": float(msd["locked_threshold_false_positives_per_case"]),
        "locked_threshold_negative_specificity": float(msd["locked_threshold_negative_specificity"]),
        "mean_FROC_sensitivity": float(msd["mean_FROC_sensitivity_0.25_0.5_1_2_4_FP_per_case"]),
    },
    {
        "cohort": "NIH_NEGATIVE_STRESS",
        "analysis_role": "NEGATIVE_ONLY_STRESS_TEST",
        "cases": int(nih["cases"]),
        "PDAC_cases": int(nih["PDAC_cases"]),
        "non_PDAC_cases": int(nih["non_PDAC_cases"]),
        "case_level_AUC": np.nan,
        "average_precision": np.nan,
        "mean_pancreas_Dice": float(nih["mean_pancreas_Dice"]),
        "mean_PDAC_lesion_Dice": np.nan,
        "PDAC_lesion_crop_presence": np.nan,
        "locked_threshold": float(nih["locked_threshold"]),
        "locked_threshold_lesion_sensitivity": np.nan,
        "locked_threshold_false_positives_per_case": float(nih["locked_threshold_false_positives_per_case"]),
        "locked_threshold_negative_specificity": float(nih["locked_threshold_negative_specificity"]),
        "mean_FROC_sensitivity": np.nan,
    },
]
performance = pd.DataFrame(performance_rows)
atomic_write_csv(performance, FINAL_PERFORMANCE_PATH)


# Merge canonical confidence intervals without recomputing any statistic.
internal_ci = pd.read_csv(INTERNAL_BOOTSTRAP_PATH)
internal_ci.insert(0, "cohort", "PANORAMA_INTERNAL_TEST")
external_combined_ci = pd.read_csv(EXTERNAL_COMBINED_BOOTSTRAP_PATH)
external_combined_ci.insert(0, "cohort", "EXTERNAL_COMBINED")
source_ci = pd.read_csv(SOURCE_BOOTSTRAP_PATH).rename(columns={"source_group": "cohort"})
final_ci = pd.concat([internal_ci, external_combined_ci, source_ci], ignore_index=True, sort=False)
atomic_write_csv(final_ci, FINAL_CI_PATH)

internal_froc = pd.read_csv(INTERNAL_FROC_PATH)
msd_froc = pd.read_csv(MSD_FROC_PATH)
froc_comparison = internal_froc.merge(
    msd_froc,
    on="target_false_positives_per_case",
    how="outer",
    validate="one_to_one",
    suffixes=("_internal", "_MSD_external"),
).sort_values("target_false_positives_per_case").reset_index(drop=True)
atomic_write_csv(froc_comparison, FINAL_FROC_COMPARISON_PATH)


# =============================================================================
# READINESS GATES BEFORE HASH FREEZE
# =============================================================================

internal_geometry = pd.read_csv(INTERNAL_GEOMETRY_PATH, dtype={"study_id": str})
external_geometry = pd.read_csv(EXTERNAL_GEOMETRY_PATH, dtype={"study_id": str})

locked_threshold = float(froc_protocol["calibrated_probability_threshold"])
threshold_values = performance["locked_threshold"].dropna().astype(float).to_numpy()

readiness = {
    "Stage 5C selected the SSL-initialized model": model_selection.get("selected_arm") == "SSL_INITIALIZED",
    "Stage 5D FROC protocol is locked": froc_protocol.get("protocol_locked") is True,
    "Internal blind prediction freeze is complete": internal_freeze.get("prediction_freeze_complete") is True,
    "Internal inference audit passed": internal_inference_audit.get("all_checks_pass") is True,
    "Internal evaluation audit passed": internal_audit.get("all_checks_pass") is True,
    "Internal cohort remains 293 cases": int(internal_audit.get("internal_test_cases", -1)) == 293,
    "Internal geometry evidence contains 293 cases": len(internal_geometry) == 293 and internal_geometry["study_id"].nunique() == 293,
    "No unresolved internal geometry review remains": no_geometry_review(internal_geometry),
    "External blind prediction freeze is complete": external_freeze.get("prediction_freeze_complete") is True,
    "External freeze is correctly labeled Stage 6C": external_freeze.get("stage") == "6C",
    "External inference audit passed": external_inference_audit.get("all_checks_pass") is True,
    "External combined evaluation audit passed": external_combined_audit.get("all_checks_pass") is True,
    "Source-separated external evaluation audit passed": source_audit.get("all_checks_pass") is True,
    "External geometry evidence contains 274 cases": len(external_geometry) == 274 and external_geometry["study_id"].nunique() == 274,
    "All 274 external geometries are exact/numerical matches": (
        external_geometry["geometry_status"].astype(str) == "EXACT_OR_NUMERICAL_PHYSICAL_GEOMETRY_MATCH"
    ).all(),
    "MSD cohort remains 194 with 98 PDAC and 96 non-PDAC": int(msd["cases"]) == 194 and int(msd["PDAC_cases"]) == 98 and int(msd["non_PDAC_cases"]) == 96,
    "NIH cohort remains 80 non-PDAC only": int(nih["cases"]) == 80 and int(nih["PDAC_cases"]) == 0 and int(nih["non_PDAC_cases"]) == 80,
    "All reported deployment thresholds match Stage 5D": np.allclose(threshold_values, locked_threshold, rtol=0, atol=1e-8),
    "All canonical finite point estimates are finite where applicable": np.isfinite(
        performance.loc[performance["cohort"] != "NIH_NEGATIVE_STRESS", [
            "case_level_AUC", "average_precision", "mean_pancreas_Dice",
            "locked_threshold_false_positives_per_case", "locked_threshold_negative_specificity",
        ]].to_numpy(dtype=float)
    ).all(),
    "All stored bootstrap confidence bounds are finite": np.isfinite(
        final_ci[["point_estimate", "bootstrap_95CI_lower", "bootstrap_95CI_upper"]].to_numpy(dtype=float)
    ).all(),
    "Internal and MSD FROC comparison contains six operating points": len(froc_comparison) == 6,
    "No CT or mask array was accessed in Stage 6E": True,
    "No model inference or fitting occurred in Stage 6E": True,
    "No threshold selection occurred in Stage 6E": True,
}
readiness = {name: bool(value) for name, value in readiness.items()}
failed = [name for name, passed in readiness.items() if not passed]
if failed:
    print("\nFAILED READINESS GATES:")
    for name in failed:
        print("  -", name)
    raise RuntimeError("Stage 6E stopped before final hashing because readiness gates failed.")


# =============================================================================
# HASH THE CANONICAL EVIDENCE SET
# =============================================================================

evidence_items = [
    (MODEL_SELECTION_PATH, "MODEL_SELECTION_LOCK"),
    (FROC_PROTOCOL_PATH, "FROC_THRESHOLD_PROTOCOL_LOCK"),
    (INTERNAL_PREDICTION_FREEZE_PATH, "INTERNAL_BLIND_PREDICTION_FREEZE"),
    (INTERNAL_PREDICTION_MANIFEST_PATH, "INTERNAL_PREDICTION_MANIFEST"),
    (INTERNAL_INFERENCE_AUDIT_PATH, "INTERNAL_BLIND_INFERENCE_AUDIT"),
    (INTERNAL_GEOMETRY_RESOLUTION_PATH, "INTERNAL_GEOMETRY_EXCEPTION_RESOLUTION"),
    (INTERNAL_GEOMETRY_PATH, "INTERNAL_GEOMETRY_AUDIT"),
    (INTERNAL_AUDIT_PATH, "INTERNAL_EVALUATION_AUDIT"),
    (INTERNAL_SUMMARY_PATH, "INTERNAL_METRIC_SUMMARY"),
    (INTERNAL_FROC_PATH, "INTERNAL_FROC_POINTS"),
    (INTERNAL_BOOTSTRAP_PATH, "INTERNAL_BOOTSTRAP_CI"),
    (EXTERNAL_PREDICTION_FREEZE_PATH, "EXTERNAL_BLIND_PREDICTION_FREEZE"),
    (EXTERNAL_PREDICTION_MANIFEST_PATH, "EXTERNAL_PREDICTION_MANIFEST"),
    (EXTERNAL_INFERENCE_AUDIT_PATH, "EXTERNAL_BLIND_INFERENCE_AUDIT"),
    (EXTERNAL_FREEZE_REPAIR_PATH, "EXTERNAL_FREEZE_METADATA_REPAIR"),
    (EXTERNAL_GEOMETRY_PATH, "EXTERNAL_GEOMETRY_AUDIT"),
    (EXTERNAL_COMBINED_AUDIT_PATH, "EXTERNAL_COMBINED_EVALUATION_AUDIT"),
    (EXTERNAL_LABEL_REPAIR_PATH, "EXTERNAL_COMBINED_LABEL_REPAIR"),
    (SOURCE_SUMMARY_PATH, "SOURCE_SEPARATED_EXTERNAL_SUMMARY"),
    (SOURCE_BOOTSTRAP_PATH, "SOURCE_SEPARATED_EXTERNAL_BOOTSTRAP_CI"),
    (MSD_FROC_PATH, "MSD_EXTERNAL_FROC_POINTS"),
    (SOURCE_AUDIT_PATH, "SOURCE_SEPARATED_EXTERNAL_AUDIT"),
    (FINAL_PERFORMANCE_PATH, "FINAL_CANONICAL_PERFORMANCE_TABLE"),
    (FINAL_CI_PATH, "FINAL_CANONICAL_CI_TABLE"),
    (FINAL_FROC_COMPARISON_PATH, "FINAL_FROC_COMPARISON"),
]

hash_rows = []
print("\nHashing canonical evidence...")
for order, (path, role) in enumerate(evidence_items, start=1):
    if not path.exists():
        raise FileNotFoundError(f"Evidence disappeared before hash freeze: {path}")
    hash_rows.append(
        {
            "evidence_role": role,
            "path": str(path),
            "size_bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
    )
    if order % 5 == 0 or order == len(evidence_items):
        print(f"  Hashed: {order}/{len(evidence_items)}")
hash_manifest = pd.DataFrame(hash_rows)
atomic_write_csv(hash_manifest, EVIDENCE_HASH_MANIFEST_PATH)
hash_manifest_sha256 = sha256_file(EVIDENCE_HASH_MANIFEST_PATH)


# =============================================================================
# FINAL RESULTS FREEZE
# =============================================================================

internal_row = performance.loc[performance["cohort"] == "PANORAMA_INTERNAL_TEST"].iloc[0]
msd_row = performance.loc[performance["cohort"] == "MSD_EXTERNAL_MIXED"].iloc[0]
nih_row = performance.loc[performance["cohort"] == "NIH_NEGATIVE_STRESS"].iloc[0]

final_freeze = {
    "stage": "6E",
    "created_at_utc": utc_now(),
    "result": "PASS_FINAL_RESULTS_AND_EVIDENCE_FROZEN",
    "final_results_frozen": True,
    "readiness_checks": readiness,
    "selected_model_arm": "SSL_INITIALIZED",
    "locked_detection_threshold": locked_threshold,
    "cohorts": {
        "PANORAMA_internal_test": EXPECTED_INTERNAL,
        "MSD_external_mixed": EXPECTED_MSD,
        "NIH_negative_stress": EXPECTED_NIH,
    },
    "headline_results": {
        "internal_case_level_AUC": float(internal_row["case_level_AUC"]),
        "internal_locked_sensitivity": float(internal_row["locked_threshold_lesion_sensitivity"]),
        "internal_locked_FP_per_case": float(internal_row["locked_threshold_false_positives_per_case"]),
        "internal_mean_FROC_sensitivity": float(internal_row["mean_FROC_sensitivity"]),
        "MSD_external_case_level_AUC": float(msd_row["case_level_AUC"]),
        "MSD_external_locked_sensitivity": float(msd_row["locked_threshold_lesion_sensitivity"]),
        "MSD_external_locked_FP_per_case": float(msd_row["locked_threshold_false_positives_per_case"]),
        "MSD_external_mean_FROC_sensitivity": float(msd_row["mean_FROC_sensitivity"]),
        "NIH_negative_stress_locked_FP_per_case": float(nih_row["locked_threshold_false_positives_per_case"]),
        "NIH_negative_stress_specificity": float(nih_row["locked_threshold_negative_specificity"]),
        "internal_to_MSD_AUC_change": float(msd_row["case_level_AUC"] - internal_row["case_level_AUC"]),
        "internal_to_MSD_sensitivity_change": float(msd_row["locked_threshold_lesion_sensitivity"] - internal_row["locked_threshold_lesion_sensitivity"]),
    },
    "post_test_model_fitting_performed": False,
    "post_test_threshold_selection_performed": False,
    "final_performance_table": str(FINAL_PERFORMANCE_PATH),
    "final_bootstrap_CI_table": str(FINAL_CI_PATH),
    "final_FROC_comparison": str(FINAL_FROC_COMPARISON_PATH),
    "evidence_hash_manifest": str(EVIDENCE_HASH_MANIFEST_PATH),
    "evidence_hash_manifest_sha256": hash_manifest_sha256,
    "hashed_evidence_items": int(len(hash_manifest)),
}
atomic_write_json(final_freeze, FINAL_FREEZE_PATH)

print("\n" + "-" * 124)
print("FINAL CANONICAL PERFORMANCE SUMMARY")
print("-" * 124)
display_columns = [
    "cohort", "cases", "PDAC_cases", "non_PDAC_cases", "case_level_AUC",
    "average_precision", "mean_pancreas_Dice", "mean_PDAC_lesion_Dice",
    "locked_threshold_lesion_sensitivity", "locked_threshold_false_positives_per_case",
    "locked_threshold_negative_specificity", "mean_FROC_sensitivity",
]
print(performance[display_columns].to_string(index=False))

print("\n" + "-" * 124)
print("READINESS CHECKS")
print("-" * 124)
for name, passed in readiness.items():
    print(f"  {name}: {passed}")

print(f"\nFinal performance table:\n{FINAL_PERFORMANCE_PATH}")
print(f"Final bootstrap CI table:\n{FINAL_CI_PATH}")
print(f"Internal-vs-MSD FROC comparison:\n{FINAL_FROC_COMPARISON_PATH}")
print(f"Evidence hash manifest:\n{EVIDENCE_HASH_MANIFEST_PATH}")
print(f"Final results freeze:\n{FINAL_FREEZE_PATH}")
print("\n" + "=" * 124)
print("STAGE 6E RESULT: PASS — FINAL RESULTS AND EVIDENCE FROZEN")
print("=" * 124)
