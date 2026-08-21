from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os

import pandas as pd


PROJECT_ROOT = Path(os.environ.get("PDAC_PROJECT_ROOT", "/content/drive/MyDrive/PDAC_Public_Q1_Project"))
META_DIR = PROJECT_ROOT / "01_Metadata"
QC_DIR = PROJECT_ROOT / "02_Quality_Control"

R1A_AUDIT = QC_DIR / "stageR1a_train_only_ssl_training_audit.json"
R1B_AUDIT = QC_DIR / "stageR1b_dual_arm_training_audit.json"
R1C_SELECTION = META_DIR / "stageR1c_supervised_model_selection.json"
R1C_AUDIT = QC_DIR / "stageR1c_full_validation_comparison_audit.json"
R1D_PROTOCOL = META_DIR / "stageR1d_detection_and_froc_protocol.json"
R1D_AUDIT = QC_DIR / "stageR1d_validation_froc_calibration_audit.json"
R1D_CASE_LEDGER = QC_DIR / "stageR1d_validation_candidate_case_ledger.csv"
R1D_CANDIDATE_LEDGER = QC_DIR / "stageR1d_validation_candidate_ledger.csv"
R1D_FROC_CURVE = QC_DIR / "stageR1d_validation_froc_curve.csv"
R1D_OPERATING_POINTS = QC_DIR / "stageR1d_validation_froc_operating_points.csv"

FREEZE_RECORD = META_DIR / "stageR1e_prefreeze_model_threshold_lock.json"
EVIDENCE_MANIFEST = META_DIR / "stageR1e_prefreeze_evidence_manifest.csv"
AUDIT_PATH = QC_DIR / "stageR1e_prefreeze_model_threshold_audit.json"

EXPECTED_VALIDATION = 295
EXPECTED_PDAC = 88
EXPECTED_NON_PDAC = 207
EXPECTED_SELECTED_ARM = "SSL_INITIALIZED"
EXPECTED_SSL_TRAIN_CASES = 1376
EXPECTED_FP_TARGET = 1.0


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


def atomic_write_json(data, path):
    temporary = Path(str(path) + ".tmp")
    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def atomic_write_csv(frame, path):
    temporary = Path(str(path) + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def read_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def first_present(mapping, keys, default=None):
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


print("=" * 120)
print("STAGE R1E — PRE-TEST MODEL, THRESHOLD, AND EVIDENCE FREEZE")
print("=" * 120)
print("GPU required: False")
print("CT volumes accessed: 0")
print("Segmentation masks accessed: 0")
print("Model inference performed: False")
print("Locked test cases accessed: 0")

required_inputs = [
    R1A_AUDIT,
    R1B_AUDIT,
    R1C_SELECTION,
    R1C_AUDIT,
    R1D_PROTOCOL,
    R1D_AUDIT,
    R1D_CASE_LEDGER,
    R1D_CANDIDATE_LEDGER,
    R1D_FROC_CURVE,
    R1D_OPERATING_POINTS,
]
missing = [str(path) for path in required_inputs if not path.exists()]
if missing:
    raise FileNotFoundError("Required R1 evidence is missing:\n" + "\n".join(missing))

r1a = read_json(R1A_AUDIT)
r1b = read_json(R1B_AUDIT)
r1c_selection = read_json(R1C_SELECTION)
r1c = read_json(R1C_AUDIT)
r1d_protocol = read_json(R1D_PROTOCOL)
r1d = read_json(R1D_AUDIT)

selected_checkpoint = Path(r1c_selection["selected_checkpoint_path"])
if not selected_checkpoint.exists():
    raise FileNotFoundError(f"Selected checkpoint is missing: {selected_checkpoint}")
selected_checkpoint_hash = sha256_file(selected_checkpoint)

case_ledger = pd.read_csv(R1D_CASE_LEDGER, dtype={"study_id": str, "patient_id": str})
candidate_ledger = pd.read_csv(
    R1D_CANDIDATE_LEDGER, dtype={"study_id": str, "patient_id": str}
)
froc_curve = pd.read_csv(R1D_FROC_CURVE)
operating_points = pd.read_csv(R1D_OPERATING_POINTS)

deployment_rows = operating_points[
    operating_points["target_false_positives_per_case"].astype(float).sub(
        EXPECTED_FP_TARGET
    ).abs() < 1e-12
]
if len(deployment_rows) != 1:
    raise RuntimeError("Exactly one 1.0 FP/case deployment row is required.")
deployment_row = deployment_rows.iloc[0]

protocol_threshold = float(r1d_protocol["calibrated_probability_threshold"])
operating_threshold = float(deployment_row["probability_threshold"])
protocol_checkpoint_hash = str(r1d_protocol["selected_checkpoint_sha256"])

ssl_arm_records = [
    arm for arm in r1b.get("arms", [])
    if str(arm.get("arm")) == EXPECTED_SELECTED_ARM
]
if len(ssl_arm_records) != 1:
    raise RuntimeError("Exactly one completed SSL_INITIALIZED R1B arm is required.")
selected_epoch = int(ssl_arm_records[0]["best_epoch"])

readiness_checks = {
    "R1A train-only SSL completed": bool(r1a.get("training_complete")),
    "R1A used exactly 1376 SSL training cases": int(
        first_present(
            r1a,
            ["training_cases", "ssl_training_cases", "train_cases"],
            -1,
        )
    ) == EXPECTED_SSL_TRAIN_CASES,
    "R1A validation CT arrays were never accessed": int(
        r1a.get("main_validation_CT_arrays_accessed_during_SSL", -1)
    ) == 0,
    "R1B dual-arm training completed": bool(r1b.get("training_complete")),
    "R1B selected SSL arm checkpoint completed": bool(
        ssl_arm_records[0].get("training_complete")
    ),
    "R1B accessed no locked test cases": int(
        r1b.get("locked_test_cases_accessed", -1)
    ) == 0,
    "R1C full-validation comparison passed": bool(r1c.get("all_checks_pass")),
    "R1C selected SSL_INITIALIZED": str(
        r1c_selection.get("selected_arm")
    ) == EXPECTED_SELECTED_ARM,
    "R1C accessed no locked test cases": int(
        r1c_selection.get("locked_test_cases_accessed", -1)
    ) == 0,
    "Selected checkpoint exists": selected_checkpoint.exists(),
    "Selected checkpoint hash matches R1C": selected_checkpoint_hash == str(
        r1c_selection.get("selected_checkpoint_sha256")
    ),
    "R1D calibration passed": bool(r1d.get("all_checks_pass")),
    "R1D selected checkpoint hash matches R1C": (
        protocol_checkpoint_hash == selected_checkpoint_hash
    ),
    "R1D threshold was selected on validation only": str(
        r1d_protocol.get("threshold_selection_data")
    ) == "validation_only",
    "R1D prohibits internal-test threshold adjustment": str(
        r1d_protocol.get("internal_test_threshold_adjustment")
    ) == "prohibited",
    "R1D prohibits external-test threshold adjustment": str(
        r1d_protocol.get("external_test_threshold_adjustment")
    ) == "prohibited",
    "R1D accessed no locked test cases": int(
        r1d_protocol.get("locked_test_cases_accessed", -1)
    ) == 0,
    "R1D contains exactly 295 validation cases": (
        len(case_ledger) == EXPECTED_VALIDATION
        and case_ledger["study_id"].nunique() == EXPECTED_VALIDATION
    ),
    "R1D contains exactly 88 PDAC and 207 non-PDAC cases": (
        int((case_ledger["label_binary"].astype(int) == 1).sum()) == EXPECTED_PDAC
        and int((case_ledger["label_binary"].astype(int) == 0).sum())
        == EXPECTED_NON_PDAC
    ),
    "Candidate ledger is non-empty": len(candidate_ledger) > 0,
    "FROC curve is non-empty and finite": (
        len(froc_curve) > 0
        and froc_curve[
            ["probability_threshold", "sensitivity", "false_positives_per_case"]
        ].apply(pd.to_numeric, errors="coerce").notna().all().all()
    ),
    "Six FROC operating points exist": len(operating_points) == 6,
    "Deployment target is exactly 1.0 FP/case": float(
        deployment_row["target_false_positives_per_case"]
    ) == EXPECTED_FP_TARGET,
    "Protocol and operating-point thresholds match": abs(
        protocol_threshold - operating_threshold
    ) < 1e-12,
    "Calibrated threshold is within [0, 1]": 0.0 <= protocol_threshold <= 1.0,
}
all_checks_pass = all(bool(value) for value in readiness_checks.values())

evidence_paths = required_inputs + [selected_checkpoint]
evidence_rows = []
for path in evidence_paths:
    evidence_rows.append(
        {
            "evidence_role": (
                "selected_model_checkpoint"
                if path == selected_checkpoint
                else "prefreeze_input_evidence"
            ),
            "path": str(path),
            "size_bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
            "verified_at_utc": utc_now(),
        }
    )
evidence_manifest = pd.DataFrame(evidence_rows)
atomic_write_csv(evidence_manifest, EVIDENCE_MANIFEST)

freeze_record = {
    "stage": "R1E",
    "reviewer_correction": "TRAIN_ONLY_SSL_PRETEST_MODEL_AND_THRESHOLD_FREEZE",
    "created_at_utc": utc_now(),
    "result": "PASS_PRETEST_FREEZE_LOCKED" if all_checks_pass else "FAIL",
    "prefreeze_complete": all_checks_pass,
    "selected_arm": EXPECTED_SELECTED_ARM,
    "selected_checkpoint_path": str(selected_checkpoint),
    "selected_checkpoint_sha256": selected_checkpoint_hash,
    "selected_epoch": selected_epoch,
    "calibrated_probability_threshold": protocol_threshold,
    "calibration_target_false_positives_per_case": EXPECTED_FP_TARGET,
    "achieved_validation_false_positives_per_case": float(
        deployment_row["achieved_false_positives_per_case"]
    ),
    "achieved_validation_lesion_sensitivity": float(
        deployment_row["sensitivity"]
    ),
    "candidate_generation_protocol_signature": str(
        r1d_protocol["candidate_generation"]["protocol_signature"]
    ),
    "threshold_selection_data": "validation_only",
    "model_weight_updates_after_freeze": "prohibited",
    "threshold_adjustment_after_freeze": "prohibited",
    "candidate_protocol_adjustment_after_freeze": "prohibited",
    "locked_test_prediction_generation_now_permitted": all_checks_pass,
    "ground_truth_metric_computation_before_prediction_freeze": "prohibited",
    "ct_volumes_accessed_in_R1E": 0,
    "segmentation_masks_accessed_in_R1E": 0,
    "locked_test_cases_accessed_in_R1E": 0,
    "evidence_manifest_path": str(EVIDENCE_MANIFEST),
    "evidence_manifest_sha256": sha256_file(EVIDENCE_MANIFEST),
}
atomic_write_json(freeze_record, FREEZE_RECORD)

audit = {
    "stage": "R1E",
    "created_at_utc": utc_now(),
    "result": freeze_record["result"],
    "all_checks_pass": all_checks_pass,
    "readiness_checks": {name: bool(value) for name, value in readiness_checks.items()},
    "evidence_items_hashed": len(evidence_manifest),
    "selected_checkpoint_sha256": selected_checkpoint_hash,
    "calibrated_probability_threshold": protocol_threshold,
    "locked_test_cases_accessed": 0,
    "freeze_record_path": str(FREEZE_RECORD),
    "evidence_manifest_path": str(EVIDENCE_MANIFEST),
}
atomic_write_json(audit, AUDIT_PATH)

print("\nLOCKED DEPLOYMENT CONFIGURATION")
print("-" * 120)
print(f"Selected arm: {EXPECTED_SELECTED_ARM}")
print(f"Selected checkpoint SHA-256: {selected_checkpoint_hash}")
print(f"Calibrated threshold: {protocol_threshold:.6f}")
print(f"Target FP/case: {EXPECTED_FP_TARGET:.2f}")
print(
    "Achieved validation FP/case: "
    f"{float(deployment_row['achieved_false_positives_per_case']):.6f}"
)
print(
    "Validation lesion sensitivity: "
    f"{float(deployment_row['sensitivity']):.6f}"
)

print("\nREADINESS CHECKS")
print("-" * 120)
for name, passed in readiness_checks.items():
    print(f"  {name}: {bool(passed)}")

print("\nEvidence manifest:")
print(EVIDENCE_MANIFEST)
print("\nFreeze record:")
print(FREEZE_RECORD)
print("\nAudit:")
print(AUDIT_PATH)
print("\n" + "=" * 120)
print(
    "STAGE R1E RESULT: "
    + (
        "PASS — MODEL, THRESHOLD, AND EVIDENCE FROZEN BEFORE TEST ACCESS"
        if all_checks_pass
        else "FAIL"
    )
)
print("=" * 120)

if not all_checks_pass:
    failed = [name for name, passed in readiness_checks.items() if not passed]
    raise RuntimeError(f"Stage R1E failed readiness checks: {failed}")
