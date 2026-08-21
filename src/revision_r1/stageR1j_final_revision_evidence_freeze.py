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
RESULTS_DIR = PROJECT_ROOT / "03_Results" / "Revision_R1_Final"
MODEL_DIR = PROJECT_ROOT / "04_Models" / "Revision_R1"

INTERNAL_MANIFEST = META_DIR / "stageR1f_i_blind_internal_test_prediction_manifest.csv"
EXTERNAL_MANIFEST = META_DIR / "stageR1h_blind_external_test_prediction_manifest.csv"
R1E_LOCK = META_DIR / "stageR1e_prefreeze_model_threshold_lock.json"
R1E_AUDIT = QC_DIR / "stageR1e_prefreeze_model_threshold_audit.json"
R1G_AUDIT = QC_DIR / "stageR1g_internal_test_evaluation_audit.json"
R1I_AUDIT = QC_DIR / "stageR1i_source_separated_external_evaluation_audit.json"

EVIDENCE_MANIFEST_PATH = META_DIR / "stageR1j_revision_r1_final_evidence_manifest.csv"
RESULT_SUMMARY_PATH = RESULTS_DIR / "stageR1j_revision_r1_publication_metric_summary.csv"
FREEZE_PATH = META_DIR / "stageR1j_revision_r1_final_evidence_freeze.json"
AUDIT_PATH = QC_DIR / "stageR1j_revision_r1_final_evidence_audit.json"


EVIDENCE_PATHS = [
    META_DIR / "stageR1a_train_only_ssl_protocol.json",
    QC_DIR / "stageR1a_train_only_ssl_training_audit.json",
    MODEL_DIR / "StageR1A_TrainOnlyMaskedContext3DCNN" / "stageR1a_ssl_final.pt",
    META_DIR / "stageR1b_dual_arm_training_protocol.json",
    QC_DIR / "stageR1b_dual_arm_training_audit.json",
    MODEL_DIR / "StageR1B_LeakageFreeDualArm" / "SSL_INITIALIZED" / "stageR1b_best.pt",
    MODEL_DIR / "StageR1B_LeakageFreeDualArm" / "RANDOM_INITIALIZED" / "stageR1b_best.pt",
    META_DIR / "stageR1c_supervised_model_selection.json",
    QC_DIR / "stageR1c_full_validation_comparison_audit.json",
    META_DIR / "stageR1d_detection_and_froc_protocol.json",
    QC_DIR / "stageR1d_validation_froc_calibration_audit.json",
    QC_DIR / "stageR1d_validation_froc_operating_points.csv",
    META_DIR / "stageR1e_prefreeze_evidence_manifest.csv",
    R1E_LOCK,
    R1E_AUDIT,
    META_DIR / "stageR1f_i_blind_internal_test_prediction_freeze.json",
    INTERNAL_MANIFEST,
    QC_DIR / "stageR1f_i_blind_internal_test_inference_audit.json",
    R1G_AUDIT,
    QC_DIR / "stageR1g_internal_test_bootstrap_95ci.csv",
    QC_DIR / "stageR1g_internal_test_froc_operating_points.csv",
    META_DIR / "stageR1h_blind_external_test_prediction_freeze.json",
    EXTERNAL_MANIFEST,
    QC_DIR / "stageR1h_blind_external_test_inference_audit.json",
    R1I_AUDIT,
    QC_DIR / "stageR1i_source_separated_external_summary.csv",
    QC_DIR / "stageR1i_source_separated_bootstrap_95ci.csv",
    QC_DIR / "stageR1i_msd_external_froc_operating_points.csv",
]


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


def atomic_write_csv(frame, path):
    temporary = Path(str(path) + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def atomic_write_json(payload, path):
    temporary = Path(str(path) + ".tmp")
    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(json_safe(payload), file, indent=2, ensure_ascii=False)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def read_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def first_present(mapping, keys, label):
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    raise RuntimeError(
        f"Could not resolve {label}. Tried keys: {keys}. "
        f"Available keys: {sorted(mapping.keys())}"
    )


def audit_passed(path):
    payload = read_json(path)
    if "all_checks_pass" in payload:
        return bool(payload["all_checks_pass"] is True)
    if "training_complete" in payload:
        return bool(payload["training_complete"] is True)
    result = str(payload.get("result", "")).strip().upper()
    return result.startswith("PASS")


def verify_prediction_manifest(path, expected_cases, label):
    manifest = pd.read_csv(path, dtype={"study_id": str})
    if len(manifest) != expected_cases or manifest["study_id"].nunique() != expected_cases:
        raise RuntimeError(f"{label} prediction manifest count changed.")
    required = {"study_id", "output_path", "output_size_bytes", "output_sha256"}
    if not required.issubset(manifest.columns):
        raise RuntimeError(f"{label} prediction manifest schema changed.")
    for order, (_, row) in enumerate(manifest.sort_values("study_id").iterrows(), start=1):
        prediction_path = Path(str(row["output_path"]))
        if not prediction_path.exists():
            raise FileNotFoundError(f"Missing frozen prediction: {prediction_path}")
        if prediction_path.stat().st_size != int(row["output_size_bytes"]):
            raise RuntimeError(f"{label} prediction size changed: {row['study_id']}")
        if sha256_file(prediction_path) != str(row["output_sha256"]):
            raise RuntimeError(f"{label} prediction hash changed: {row['study_id']}")
        if order % 50 == 0 or order == expected_cases:
            print(f"  {label} prediction verification: {order}/{expected_cases}")
    return manifest


print("=" * 124)
print("STAGE R1J — FINAL REVIEWER-REVISION EVIDENCE FREEZE")
print("=" * 124)
print("GPU required: False")
print("CT volumes accessed: 0")
print("Segmentation masks accessed: 0")
print("Model training/inference: False")
print("Threshold selection: False")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
for path in EVIDENCE_PATHS:
    if not path.exists():
        raise FileNotFoundError(f"Required R1 evidence is missing:\n{path}")

r1a_audit = QC_DIR / "stageR1a_train_only_ssl_training_audit.json"
r1b_audit = QC_DIR / "stageR1b_dual_arm_training_audit.json"
r1c_audit = QC_DIR / "stageR1c_full_validation_comparison_audit.json"
r1d_audit = QC_DIR / "stageR1d_validation_froc_calibration_audit.json"
r1f_audit = QC_DIR / "stageR1f_i_blind_internal_test_inference_audit.json"
r1h_audit = QC_DIR / "stageR1h_blind_external_test_inference_audit.json"

print("\nVerifying 293 frozen internal predictions...")
internal_manifest = verify_prediction_manifest(INTERNAL_MANIFEST, 293, "Internal")
print("\nVerifying 274 frozen external predictions...")
external_manifest = verify_prediction_manifest(EXTERNAL_MANIFEST, 274, "External")

prefreeze = read_json(R1E_LOCK)
r1d_payload = read_json(r1d_audit)
internal_audit = read_json(R1G_AUDIT)
external_audit = read_json(R1I_AUDIT)
locked_threshold = float(
    first_present(
        prefreeze,
        [
            "calibrated_probability_threshold",
            "calibrated_threshold",
            "locked_deployment_threshold",
        ],
        "the frozen R1E deployment threshold",
    )
)
if not np.isfinite(locked_threshold) or not 0.0 <= locked_threshold <= 1.0:
    raise RuntimeError(f"Invalid frozen R1E deployment threshold: {locked_threshold}")
selected_checkpoint = Path(str(prefreeze["selected_checkpoint_path"]))
if not selected_checkpoint.exists():
    raise FileNotFoundError(f"Frozen selected checkpoint is missing: {selected_checkpoint}")
selected_checkpoint_hash = sha256_file(selected_checkpoint)

evidence_rows = []
for order, path in enumerate(EVIDENCE_PATHS, start=1):
    evidence_rows.append(
        {
            "evidence_order": order,
            "file_name": path.name,
            "path": str(path),
            "size_bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
            "frozen_at_utc": utc_now(),
        }
    )
    if order % 5 == 0 or order == len(EVIDENCE_PATHS):
        print(f"  Canonical evidence: {order}/{len(EVIDENCE_PATHS)}")
evidence_manifest = pd.DataFrame(evidence_rows)
atomic_write_csv(evidence_manifest, EVIDENCE_MANIFEST_PATH)

source_summary = pd.read_csv(
    QC_DIR / "stageR1i_source_separated_external_summary.csv"
).set_index("source_group")
msd = source_summary.loc["MSD_EXTERNAL_MIXED"]
nih = source_summary.loc["NIH_NEGATIVE_STRESS"]

publication_rows = [
    {"cohort": "validation", "cases": 295, "metric": "candidate_AUC", "value": float(r1d_payload["candidate_AUC"])},
    {"cohort": "validation", "cases": 295, "metric": "candidate_average_precision", "value": float(r1d_payload["candidate_average_precision"])},
    {"cohort": "validation", "cases": 295, "metric": "locked_threshold", "value": locked_threshold},
    {"cohort": "internal_test", "cases": 293, "metric": "candidate_AUC", "value": float(internal_audit["candidate_AUC"])},
    {"cohort": "internal_test", "cases": 293, "metric": "candidate_average_precision", "value": float(internal_audit["candidate_average_precision"])},
    {"cohort": "internal_test", "cases": 293, "metric": "mean_pancreas_Dice", "value": float(internal_audit["mean_pancreas_Dice"])},
    {"cohort": "internal_test", "cases": 293, "metric": "mean_PDAC_lesion_Dice", "value": float(internal_audit["mean_PDAC_lesion_Dice"])},
    {"cohort": "internal_test", "cases": 293, "metric": "locked_threshold_lesion_sensitivity", "value": float(internal_audit["locked_threshold_lesion_sensitivity"])},
    {"cohort": "internal_test", "cases": 293, "metric": "locked_threshold_false_positives_per_case", "value": float(internal_audit["locked_threshold_false_positives_per_case"])},
    {"cohort": "MSD_EXTERNAL_MIXED", "cases": 194, "metric": "case_level_AUC", "value": float(msd["case_level_AUC"])},
    {"cohort": "MSD_EXTERNAL_MIXED", "cases": 194, "metric": "case_level_average_precision", "value": float(msd["case_level_average_precision"])},
    {"cohort": "MSD_EXTERNAL_MIXED", "cases": 194, "metric": "mean_pancreas_Dice", "value": float(msd["mean_pancreas_Dice"])},
    {"cohort": "MSD_EXTERNAL_MIXED", "cases": 194, "metric": "mean_PDAC_lesion_Dice", "value": float(msd["mean_PDAC_lesion_Dice"])},
    {"cohort": "MSD_EXTERNAL_MIXED", "cases": 194, "metric": "locked_threshold_lesion_sensitivity", "value": float(msd["locked_threshold_lesion_sensitivity"])},
    {"cohort": "MSD_EXTERNAL_MIXED", "cases": 194, "metric": "locked_threshold_false_positives_per_case", "value": float(msd["locked_threshold_false_positives_per_case"])},
    {"cohort": "NIH_NEGATIVE_STRESS", "cases": 80, "metric": "mean_pancreas_Dice", "value": float(nih["mean_pancreas_Dice"])},
    {"cohort": "NIH_NEGATIVE_STRESS", "cases": 80, "metric": "locked_threshold_false_positives_per_case", "value": float(nih["locked_threshold_false_positives_per_case"])},
    {"cohort": "NIH_NEGATIVE_STRESS", "cases": 80, "metric": "locked_threshold_negative_specificity", "value": float(nih["locked_threshold_negative_specificity"])},
]
publication_summary = pd.DataFrame(publication_rows)
atomic_write_csv(publication_summary, RESULT_SUMMARY_PATH)

readiness_checks = {
    "R1A train-only SSL audit passed": audit_passed(r1a_audit),
    "R1B leakage-free dual-arm audit passed": audit_passed(r1b_audit),
    "R1C full-validation comparison audit passed": audit_passed(r1c_audit),
    "R1D validation-only threshold calibration audit passed": audit_passed(r1d_audit),
    "R1E pre-test freeze audit passed": audit_passed(R1E_AUDIT),
    "R1F-I blind internal inference audit passed": audit_passed(r1f_audit),
    "R1G internal evaluation audit passed": audit_passed(R1G_AUDIT),
    "R1H blind external inference audit passed": audit_passed(r1h_audit),
    "R1I source-separated external evaluation audit passed": audit_passed(R1I_AUDIT),
    "Selected checkpoint hash still matches R1E": selected_checkpoint_hash == str(prefreeze["selected_checkpoint_sha256"]),
    "Exactly 293 internal predictions were reverified": len(internal_manifest) == 293,
    "Exactly 274 external predictions were reverified": len(external_manifest) == 274,
    "Exactly 28 canonical evidence items were hashed": len(evidence_manifest) == 28,
    "Locked threshold remains 0.462060 within serialized precision": np.isclose(locked_threshold, 0.4620599448680877, rtol=0, atol=1e-8),
    "No CT or segmentation mask was accessed": True,
    "No training or inference occurred": True,
    "No threshold selection occurred": True,
}
readiness_checks = {name: bool(value) for name, value in readiness_checks.items()}
all_checks_pass = bool(all(readiness_checks.values()))

freeze = {
    "stage": "R1J",
    "created_at_utc": utc_now(),
    "result": "PASS_FINAL_REVIEWER_REVISION_EVIDENCE_FROZEN" if all_checks_pass else "FAIL",
    "all_checks_pass": all_checks_pass,
    "selected_arm": "SSL_INITIALIZED",
    "selected_checkpoint_path": str(selected_checkpoint),
    "selected_checkpoint_sha256": selected_checkpoint_hash,
    "locked_detection_threshold": locked_threshold,
    "internal_test_cases": 293,
    "MSD_external_cases": 194,
    "NIH_negative_stress_cases": 80,
    "prediction_files_reverified": 567,
    "canonical_evidence_items": len(evidence_manifest),
    "readiness_checks": readiness_checks,
    "evidence_manifest_path": str(EVIDENCE_MANIFEST_PATH),
    "publication_metric_summary_path": str(RESULT_SUMMARY_PATH),
    "model_or_threshold_adjustment_after_test": False,
}
atomic_write_json(freeze, FREEZE_PATH)

audit = {
    **freeze,
    "CT_volumes_accessed": 0,
    "segmentation_masks_accessed": 0,
    "model_training_performed": False,
    "model_inference_performed": False,
    "threshold_selection_performed": False,
}
atomic_write_json(audit, AUDIT_PATH)

print("\nREADINESS CHECKS")
print("-" * 124)
for name, passed in readiness_checks.items():
    print(f"  {name}: {passed}")
print(f"\nEvidence manifest:\n{EVIDENCE_MANIFEST_PATH}")
print(f"Publication metric summary:\n{RESULT_SUMMARY_PATH}")
print(f"Final freeze:\n{FREEZE_PATH}")
print(f"Final audit:\n{AUDIT_PATH}")
print("=" * 124)
print(
    "STAGE R1J RESULT: "
    + ("PASS — FINAL REVIEWER-REVISION EVIDENCE FROZEN" if all_checks_pass else "FAIL")
)
print("=" * 124)
if not all_checks_pass:
    raise RuntimeError("Stage R1J failed one or more final readiness checks.")
