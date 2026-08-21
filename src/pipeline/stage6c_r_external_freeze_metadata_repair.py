from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os

import numpy as np
import pandas as pd


# =============================================================================
# STAGE 6C-R — CPU-ONLY EXTERNAL FREEZE METADATA REPAIR
#
# Stage 6C blind inference itself completed correctly for all 274 external
# cases. The derived script inherited two descriptive JSON values from Stage
# 6A (stage="6A" and an internal-test result label). This recovery verifies
# the frozen prediction set again and corrects metadata only. No prediction,
# CT, mask, diagnostic label, model weight, or threshold is changed/accessed.
# =============================================================================

PROJECT_ROOT = Path(os.environ.get("PDAC_PROJECT_ROOT", "/content/drive/MyDrive/PDAC_Public_Q1_Project"))
META_DIR = PROJECT_ROOT / "01_Metadata"
QC_DIR = PROJECT_ROOT / "02_Quality_Control"

MANIFEST_PATH = META_DIR / "stage6c_blind_external_test_prediction_manifest.csv"
FREEZE_PATH = META_DIR / "stage6c_blind_external_test_prediction_freeze.json"
AUDIT_PATH = QC_DIR / "stage6c_blind_external_test_inference_audit.json"
LEDGER_PATH = QC_DIR / "stage6c_blind_external_test_resume_ledger.csv"
CANDIDATE_PATH = QC_DIR / "stage6c_blind_external_test_candidate_ledger.csv"
REPAIR_PATH = QC_DIR / "stage6c_r_external_freeze_metadata_repair.json"

EXPECTED_CASES = 274
EXPECTED_PARTITIONS = {"external_msd_test": 194, "nih_negative_stress_test": 80}
EXPECTED_SOURCES = {"MSD_EXTERNAL_MIXED": 194, "NIH_NEGATIVE_STRESS": 80}
EXPECTED_THRESHOLD = 0.41886454820632935


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


def atomic_write_json(payload, path):
    temporary = Path(str(path) + ".tmp")
    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def truth_flags(series):
    return (
        series.fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "pass", "complete", "completed"})
    )


print("=" * 124)
print("STAGE 6C-R — CPU-ONLY EXTERNAL FREEZE METADATA REPAIR")
print("=" * 124)
print("GPU required: False")
print("Model inference repeated: False")
print("Diagnostic labels accessed: 0")
print("Test masks accessed: 0")

for path in [MANIFEST_PATH, FREEZE_PATH, AUDIT_PATH, LEDGER_PATH, CANDIDATE_PATH]:
    if not path.exists():
        raise FileNotFoundError(f"Required Stage 6C evidence is missing:\n{path}")

manifest = pd.read_csv(MANIFEST_PATH, dtype={"study_id": str, "patient_id": str})
ledger = pd.read_csv(LEDGER_PATH, dtype={"study_id": str, "patient_id": str})
with open(FREEZE_PATH, "r", encoding="utf-8") as file:
    freeze = json.load(file)
with open(AUDIT_PATH, "r", encoding="utf-8") as file:
    audit = json.load(file)

freeze_hash_before = sha256_file(FREEZE_PATH)
audit_hash_before = sha256_file(AUDIT_PATH)

manifest["study_id"] = manifest["study_id"].astype(str).str.strip()
ledger["study_id"] = ledger["study_id"].astype(str).str.strip()
ledger = ledger.drop_duplicates("study_id", keep="last")

partition_counts = manifest["locked_partition"].value_counts().to_dict()
source_counts = manifest["source_group"].value_counts().to_dict()

print("\nReverifying 274 frozen prediction files...")
hash_matches = 0
for order, (_, row) in enumerate(manifest.iterrows(), start=1):
    path = Path(str(row["output_path"]))
    if not path.exists():
        raise FileNotFoundError(f"Frozen prediction missing: {path}")
    if path.stat().st_size != int(row["output_size_bytes"]):
        raise RuntimeError(f"Frozen prediction size changed: {row['study_id']}")
    if sha256_file(path) != str(row["output_sha256"]):
        raise RuntimeError(f"Frozen prediction hash changed: {row['study_id']}")
    hash_matches += 1
    if order % 50 == 0 or order == EXPECTED_CASES:
        print(f"  Hash verification: {order}/{EXPECTED_CASES}")

ledger_complete = truth_flags(ledger["processing_complete"])
checks = {
    "Exactly 274 manifest rows": len(manifest) == EXPECTED_CASES,
    "Exactly 274 unique manifest IDs": manifest["study_id"].nunique() == EXPECTED_CASES,
    "Exactly 274 complete ledger IDs": int(ledger_complete.sum()) == EXPECTED_CASES,
    "Manifest IDs exactly match complete ledger IDs": set(manifest["study_id"])
    == set(ledger.loc[ledger_complete, "study_id"]),
    "External partition counts are 194 MSD plus 80 NIH": partition_counts
    == EXPECTED_PARTITIONS,
    "External source counts are 194 MSD plus 80 NIH": source_counts
    == EXPECTED_SOURCES,
    "All 274 frozen prediction hashes reverified": hash_matches == EXPECTED_CASES,
    "Original freeze was complete": freeze.get("prediction_freeze_complete") is True,
    "Original inference scope is external blind inference": freeze.get("inference_scope")
    == "source_separated_external_CT_only_blind_inference",
    "Original freeze records zero diagnostic-label access": int(
        freeze.get("diagnostic_labels_accessed", -1)
    )
    == 0,
    "Original freeze records zero mask access": int(freeze.get("test_masks_accessed", -1))
    == 0,
    "Locked threshold remains unchanged": np.isclose(
        float(freeze.get("locked_deployment_threshold", np.nan)),
        EXPECTED_THRESHOLD,
        rtol=0,
        atol=1e-8,
    ),
    "Original audit passed all inference gates": audit.get("all_checks_pass") is True,
}
checks = {name: bool(value) for name, value in checks.items()}
if not all(checks.values()):
    failed = [name for name, passed in checks.items() if not passed]
    raise RuntimeError(f"Stage 6C freeze repair blocked by failed checks: {failed}")

# Correct descriptive metadata only after the entire frozen prediction set has
# independently re-passed its integrity gates.
freeze["stage"] = "6C"
freeze["created_at_utc_metadata_repaired"] = utc_now()
freeze["metadata_repair"] = {
    "reason": "Inherited Stage 6A descriptive stage label corrected after external freeze",
    "prediction_arrays_modified": False,
    "model_inference_repeated": False,
    "diagnostic_labels_accessed": 0,
    "test_masks_accessed": 0,
}

audit["stage"] = "6C"
audit["result"] = "PASS_BLIND_EXTERNAL_TEST_PREDICTIONS_FROZEN"
audit["created_at_utc_metadata_repaired"] = utc_now()
audit["metadata_repair"] = {
    "reason": "Inherited Stage 6A/internal-test descriptive labels corrected",
    "prediction_arrays_modified": False,
    "model_inference_repeated": False,
    "diagnostic_labels_accessed": 0,
    "test_masks_accessed": 0,
}

atomic_write_json(freeze, FREEZE_PATH)
atomic_write_json(audit, AUDIT_PATH)

freeze_hash_after = sha256_file(FREEZE_PATH)
audit_hash_after = sha256_file(AUDIT_PATH)
repair = {
    "stage": "6C-R",
    "created_at_utc": utc_now(),
    "result": "PASS_EXTERNAL_FREEZE_METADATA_REPAIRED",
    "readiness_checks": checks,
    "prediction_files_reverified": EXPECTED_CASES,
    "prediction_files_modified": 0,
    "model_inference_repeated": False,
    "diagnostic_labels_accessed": 0,
    "test_masks_accessed": 0,
    "threshold_adjusted": False,
    "freeze_sha256_before": freeze_hash_before,
    "freeze_sha256_after": freeze_hash_after,
    "audit_sha256_before": audit_hash_before,
    "audit_sha256_after": audit_hash_after,
    "corrected_fields": {
        "freeze.stage": "6C",
        "audit.stage": "6C",
        "audit.result": "PASS_BLIND_EXTERNAL_TEST_PREDICTIONS_FROZEN",
    },
}
atomic_write_json(repair, REPAIR_PATH)

print("\n" + "-" * 124)
print("READINESS CHECKS")
print("-" * 124)
for name, passed in checks.items():
    print(f"  {name}: {passed}")
print("\nCorrected freeze:")
print(FREEZE_PATH)
print("Corrected audit:")
print(AUDIT_PATH)
print("Repair record:")
print(REPAIR_PATH)
print("\n" + "=" * 124)
print("STAGE 6C-R RESULT: PASS — EXTERNAL FREEZE METADATA REPAIRED")
print("=" * 124)
