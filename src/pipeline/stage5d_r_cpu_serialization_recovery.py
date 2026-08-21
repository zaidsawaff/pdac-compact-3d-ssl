from pathlib import Path
from datetime import datetime, timezone
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


PROJECT_ROOT = Path(os.environ.get("PDAC_PROJECT_ROOT", "/content/drive/MyDrive/PDAC_Public_Q1_Project"))
META_DIR = PROJECT_ROOT / "01_Metadata"
QC_DIR = PROJECT_ROOT / "02_Quality_Control"

FROZEN_MANIFEST_PATH = META_DIR / "stage3b_e5_frozen_dataset_manifest.csv"
MODEL_SELECTION_PATH = META_DIR / "stage5c_supervised_model_selection.json"
CASE_LEDGER_PATH = QC_DIR / "stage5d_validation_candidate_case_ledger.csv"
CANDIDATE_LEDGER_PATH = QC_DIR / "stage5d_validation_candidate_ledger.csv"
FROC_CURVE_PATH = QC_DIR / "stage5d_validation_froc_curve.csv"
OPERATING_POINTS_PATH = QC_DIR / "stage5d_validation_froc_operating_points.csv"
PROTOCOL_PATH = META_DIR / "stage5d_detection_and_froc_protocol.json"
AUDIT_PATH = QC_DIR / "stage5d_validation_froc_calibration_audit.json"
RESOLUTION_PATH = QC_DIR / "stage5d_r_serialization_recovery_record.json"

EXPECTED_VALIDATION = 295
EXPECTED_PDAC = 88
EXPECTED_NON_PDAC = 207
EXPECTED_OPERATING_POINTS = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def atomic_write_json(data, path):
    temporary = Path(str(path) + ".tmp")
    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
            default=json_default,
        )
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def truth_flags(series):
    return (
        series.fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes"})
    )


print("=" * 120)
print("STAGE 5D-R — CPU-ONLY FROC SERIALIZATION RECOVERY")
print("=" * 120)
print("GPU required: False")
print("Model inference repeated: False")
print("Locked test cases accessed: 0")

for required in [
    FROZEN_MANIFEST_PATH,
    MODEL_SELECTION_PATH,
    CASE_LEDGER_PATH,
    CANDIDATE_LEDGER_PATH,
    FROC_CURVE_PATH,
    OPERATING_POINTS_PATH,
    PROTOCOL_PATH,
]:
    if not required.exists():
        raise FileNotFoundError(f"Required completed Stage 5D output missing:\n{required}")

with open(MODEL_SELECTION_PATH, "r", encoding="utf-8") as file:
    model_selection = json.load(file)
with open(PROTOCOL_PATH, "r", encoding="utf-8") as file:
    protocol = json.load(file)

manifest = pd.read_csv(FROZEN_MANIFEST_PATH, dtype={"study_id": str})
manifest["study_id"] = manifest["study_id"].astype(str).str.strip()
manifest["partition"] = manifest["partition"].astype(str).str.strip().str.lower()
validation = manifest[manifest["partition"] == "validation"].copy()

case_ledger = pd.read_csv(
    CASE_LEDGER_PATH,
    dtype={"study_id": str, "patient_id": str},
)
candidate_ledger = pd.read_csv(
    CANDIDATE_LEDGER_PATH,
    dtype={"study_id": str, "patient_id": str},
)
curve = pd.read_csv(FROC_CURVE_PATH)
operating_points = pd.read_csv(OPERATING_POINTS_PATH)

case_ledger["study_id"] = case_ledger["study_id"].astype(str).str.strip()
candidate_ledger["study_id"] = candidate_ledger["study_id"].astype(str).str.strip()
case_ledger = case_ledger.drop_duplicates("study_id", keep="last")
candidate_ledger = candidate_ledger.drop_duplicates(
    ["study_id", "candidate_rank"], keep="last"
)

selected_arm = str(model_selection["selected_arm"])
selected_checkpoint_hash = str(model_selection["selected_checkpoint_sha256"])
protocol_signature = str(protocol["candidate_generation"]["protocol_signature"])

case_ledger = case_ledger[
    (case_ledger["protocol_signature"].astype(str) == protocol_signature)
    & (case_ledger["checkpoint_sha256"].astype(str) == selected_checkpoint_hash)
].copy()
candidate_ledger = candidate_ledger[
    (candidate_ledger["protocol_signature"].astype(str) == protocol_signature)
    & (candidate_ledger["checkpoint_sha256"].astype(str) == selected_checkpoint_hash)
].copy()

pdac_cases = case_ledger[case_ledger["label_binary"].astype(int) == 1]
negative_cases = case_ledger[case_ledger["label_binary"].astype(int) == 0]

labels = case_ledger["label_binary"].astype(int).to_numpy()
scores = case_ledger["maximum_candidate_score"].astype(float).to_numpy()
candidate_auc = float(roc_auc_score(labels, scores))
candidate_ap = float(average_precision_score(labels, scores))

deployment_threshold = float(protocol["calibrated_probability_threshold"])
deployment_candidates = candidate_ledger[
    candidate_ledger["candidate_score"].astype(float) >= deployment_threshold
].copy()
deployment_candidates["true_positive_bool"] = truth_flags(
    deployment_candidates["is_true_positive_candidate"]
)
case_ids_with_candidate = set(deployment_candidates["study_id"])

negative_specificity = float(
    np.mean(
        [study_id not in case_ids_with_candidate for study_id in negative_cases["study_id"]]
    )
)
pdac_sensitivity = float(
    np.mean(
        [
            bool(
                deployment_candidates.loc[
                    deployment_candidates["study_id"] == study_id,
                    "true_positive_bool",
                ].any()
            )
            for study_id in pdac_cases["study_id"]
        ]
    )
)

standard_points = operating_points[
    operating_points["target_false_positives_per_case"].isin(
        [0.25, 0.5, 1.0, 2.0, 4.0]
    )
]
froc_mean_sensitivity = float(standard_points["sensitivity"].mean())

readiness_checks = {
    "Selected arm remains SSL_INITIALIZED": selected_arm == "SSL_INITIALIZED",
    "Exactly 295 validation rows were recovered": len(case_ledger) == EXPECTED_VALIDATION,
    "Exactly 295 unique validation IDs were recovered": case_ledger["study_id"].nunique() == EXPECTED_VALIDATION,
    "Recovered IDs exactly match validation lock": set(case_ledger["study_id"]) == set(validation["study_id"]),
    "Exactly 88 PDAC validation cases were recovered": len(pdac_cases) == EXPECTED_PDAC,
    "Exactly 207 non-PDAC validation cases were recovered": len(negative_cases) == EXPECTED_NON_PDAC,
    "All case processing flags pass": truth_flags(case_ledger["processing_complete"]).all(),
    "All probability outputs were finite": truth_flags(case_ledger["probabilities_finite"]).all(),
    "Pancreas gate seed exists in every case": truth_flags(case_ledger["pancreas_seed_present"]).all(),
    "All 88 PDAC references are represented": int(case_ledger["reference_lesions"].sum()) == EXPECTED_PDAC,
    "Candidate ledger is non-empty": len(candidate_ledger) > 0,
    "FROC curve is non-empty": len(curve) > 0,
    "FROC curve values are finite": bool(
        np.isfinite(
            curve[
                [
                    "probability_threshold",
                    "sensitivity",
                    "false_positives_per_case",
                ]
            ]
        ).all().all()
    ),
    "Six expected operating points exist": set(
        np.round(
            operating_points["target_false_positives_per_case"].astype(float),
            8,
        )
    ) == set(EXPECTED_OPERATING_POINTS),
    "Calibrated threshold is finite": bool(np.isfinite(deployment_threshold)),
    "Candidate AUC is finite": bool(np.isfinite(candidate_auc)),
    "Candidate average precision is finite": bool(np.isfinite(candidate_ap)),
    "No locked test case was accessed": set(manifest["partition"]) == {"train", "validation"},
}
readiness_checks = {
    str(name): bool(value) for name, value in readiness_checks.items()
}
all_checks_pass = all(readiness_checks.values())

deployment_point = operating_points[
    np.isclose(
        operating_points["target_false_positives_per_case"].astype(float),
        1.0,
    )
].iloc[0]

audit = {
    "stage": "5D",
    "created_at_utc": utc_now(),
    "result": "PASS_VALIDATION_FROC_PROTOCOL_LOCKED" if all_checks_pass else "FAIL",
    "all_checks_pass": all_checks_pass,
    "readiness_checks": readiness_checks,
    "selected_arm": selected_arm,
    "selected_checkpoint_sha256": selected_checkpoint_hash,
    "protocol_signature": protocol_signature,
    "validation_cases": int(len(case_ledger)),
    "PDAC_references": int(case_ledger["reference_lesions"].sum()),
    "generated_candidates": int(len(candidate_ledger)),
    "mean_candidates_per_case": float(len(candidate_ledger) / EXPECTED_VALIDATION),
    "PDAC_references_localized_at_any_score": int(
        truth_flags(pdac_cases["reference_localized_at_any_score"]).sum()
    ),
    "candidate_AUC": candidate_auc,
    "candidate_average_precision": candidate_ap,
    "FROC_mean_sensitivity": froc_mean_sensitivity,
    "deployment_threshold": deployment_threshold,
    "deployment_achieved_FP_per_case": float(
        deployment_point["achieved_false_positives_per_case"]
    ),
    "deployment_sensitivity": float(deployment_point["sensitivity"]),
    "deployment_negative_specificity": negative_specificity,
    "case_ledger_path": str(CASE_LEDGER_PATH),
    "candidate_ledger_path": str(CANDIDATE_LEDGER_PATH),
    "froc_curve_path": str(FROC_CURVE_PATH),
    "operating_points_path": str(OPERATING_POINTS_PATH),
    "protocol_path": str(PROTOCOL_PATH),
    "locked_test_cases_accessed": 0,
}
atomic_write_json(audit, AUDIT_PATH)

resolution = {
    "stage": "5D-R",
    "created_at_utc": utc_now(),
    "result": "PASS_SERIALIZATION_RECOVERED" if all_checks_pass else "FAIL",
    "original_failure": "NumPy boolean was not JSON serializable",
    "scientific_outputs_affected": False,
    "model_inference_repeated": False,
    "GPU_used": False,
    "recovered_case_rows": int(len(case_ledger)),
    "recovered_candidate_rows": int(len(candidate_ledger)),
    "canonical_audit_path": str(AUDIT_PATH),
    "all_checks_pass": all_checks_pass,
    "locked_test_cases_accessed": 0,
}
atomic_write_json(resolution, RESOLUTION_PATH)

print("\nFROC OPERATING POINTS")
print("-" * 120)
print(operating_points.to_string(index=False))

print("\nRECOVERED CALIBRATION SUMMARY")
print("-" * 120)
print(f"Generated candidates: {len(candidate_ledger)}")
print(f"Mean candidates/case: {len(candidate_ledger) / EXPECTED_VALIDATION:.3f}")
print(f"Candidate AUC: {candidate_auc:.6f}")
print(f"Candidate average precision: {candidate_ap:.6f}")
print(f"Mean FROC sensitivity: {froc_mean_sensitivity:.6f}")
print(f"Calibrated threshold: {deployment_threshold:.6f}")
print(
    "Achieved FP/case: "
    f"{float(deployment_point['achieved_false_positives_per_case']):.6f}"
)
print(f"Lesion sensitivity: {float(deployment_point['sensitivity']):.6f}")
print(f"Negative-case specificity: {negative_specificity:.6f}")
print(f"PDAC case sensitivity: {pdac_sensitivity:.6f}")

print("\nREADINESS CHECKS")
print("-" * 120)
for name, passed in readiness_checks.items():
    print(f"  {name}: {passed}")

print("\nCanonical audit:")
print(AUDIT_PATH)
print("\nRecovery record:")
print(RESOLUTION_PATH)
print("\n" + "=" * 120)
print(
    "STAGE 5D-R RESULT: "
    + ("PASS — SERIALIZATION RECOVERED; FROC PROTOCOL LOCKED" if all_checks_pass else "FAIL")
)
print("=" * 120)

if not all_checks_pass:
    failed = [name for name, passed in readiness_checks.items() if not passed]
    raise RuntimeError(f"Stage 5D-R failed checks: {failed}")
