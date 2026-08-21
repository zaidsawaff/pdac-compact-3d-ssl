from pathlib import Path
import os
from datetime import datetime, timezone
import hashlib
import json
import runpy


# =============================================================================
# STAGE 6C — BUILD + RUN SOURCE-SEPARATED BLIND EXTERNAL INFERENCE
#
# This launcher deliberately derives the external-inference implementation
# from the already frozen Stage 6A blind internal-test implementation.  Only
# cohort identity, expected counts, stage/output names, and per-row source role
# fields are changed.  Model, preprocessing, crop geometry, candidate
# generation, threshold, and all numerical inference code remain identical.
# =============================================================================

PROJECT_ROOT = Path(os.environ.get("PDAC_PROJECT_ROOT", "/content/drive/MyDrive/PDAC_Public_Q1_Project"))
META_DIR = PROJECT_ROOT / "01_Metadata"
QC_DIR = PROJECT_ROOT / "02_Quality_Control"

BASE_SCRIPT = META_DIR / "stage6a_resumable_blind_internal_test_inference.py"
EXPANDED_SCRIPT = META_DIR / "stage6c_blind_external_test_inference_expanded.py"
BUILD_RECORD = QC_DIR / "stage6c_external_inference_build_record.json"

EXPECTED_EXTERNAL_CASES = 274
EXPECTED_EXTERNAL_PARTITIONS = {
    "external_msd_test": 194,
    "nih_negative_stress_test": 80,
}
EXPECTED_EXTERNAL_SOURCES = {
    "MSD_EXTERNAL_MIXED": 194,
    "NIH_NEGATIVE_STRESS": 80,
}


def sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def replace_exact(text, old, new, expected_count=1):
    observed = text.count(old)
    if observed != expected_count:
        raise RuntimeError(
            f"Stage 6A source drift detected. Expected {expected_count} occurrence(s) "
            f"of {old!r}; observed {observed}."
        )
    return text.replace(old, new)


if not BASE_SCRIPT.exists():
    raise FileNotFoundError(
        "The frozen Stage 6A inference script is missing:\n"
        f"{BASE_SCRIPT}\n"
        "Do not substitute a different implementation."
    )

base_bytes = BASE_SCRIPT.read_bytes()
base_hash = sha256_bytes(base_bytes)
source = base_bytes.decode("utf-8")

# First make stage/output terminology external while preserving all numerical
# code and inference functions.
source = source.replace("STAGE 6A", "STAGE 6C")
source = source.replace("Stage 6A", "Stage 6C")
source = source.replace("Stage6A", "Stage6C")
source = source.replace("stage6a", "stage6c")
source = source.replace("INTERNAL-TEST", "EXTERNAL-TEST")
source = source.replace("Internal-test", "External-test")
source = source.replace("internal-test", "external-test")
source = source.replace("Internal_Test", "External_Test")
source = source.replace("internal_test", "external_test")
source = source.replace("EXPECTED_INTERNAL_TEST_CASES", "EXPECTED_EXTERNAL_TEST_CASES")
source = source.replace("EXPECTED_SOURCE_GROUP", "EXPECTED_SOURCE_GROUPS")

# The locked external cohort is the union of two source-separated partitions.
source = replace_exact(
    source,
    'EXPECTED_EXTERNAL_TEST_CASES = 293',
    f'EXPECTED_EXTERNAL_TEST_CASES = {EXPECTED_EXTERNAL_CASES}',
)
source = replace_exact(
    source,
    'EXPECTED_SOURCE_GROUPS = "PANORAMA_LOCAL"',
    'EXPECTED_SOURCE_GROUPS = {"MSD_EXTERNAL_MIXED", "NIH_NEGATIVE_STRESS"}',
)
source = replace_exact(
    source,
    'split["locked_partition"].astype(str).str.strip() == "external_test"',
    'split["locked_partition"].astype(str).str.strip().isin('
    '["external_msd_test", "nih_negative_stress_test"]'
    ')',
)
source = replace_exact(
    source,
    'if set(external_test["source_group"]) != {EXPECTED_SOURCE_GROUPS}:',
    'if set(external_test["source_group"]) != EXPECTED_SOURCE_GROUPS:',
)

# Preserve the actual source/partition role in every frozen prediction row.
source = replace_exact(
    source,
    '"locked_partition": "external_test",',
    '"locked_partition": str(identity_row["locked_partition"]),',
)
source = replace_exact(
    source,
    '"source_group": EXPECTED_SOURCE_GROUPS,',
    '"source_group": str(identity_row["source_group"]),',
)

# Make Stage 6C outputs fully independent of the already-frozen Stage 6A data.
# The earlier terminology transformations already changed filenames containing
# stage6a/internal_test; these explicit assertions ensure none were missed.
source = replace_exact(
    source,
    'PROJECT_ROOT / "05_Predictions" / "Stage6C_External_Test_Blind"',
    'PROJECT_ROOT / "05_Predictions" / "Stage6C_External_SourceSeparated_Blind"',
)

# Readiness labels are descriptive text only; update the hard-coded displayed
# count without changing any numerical condition (which uses the constant).
source = source.replace("Exactly 293 blind prediction rows", "Exactly 274 blind prediction rows")
source = source.replace("Exactly 293 unique blind study IDs", "Exactly 274 unique blind study IDs")
source = source.replace("exactly 293 NPZ files", "exactly 274 NPZ files")

# External source separation is itself a freeze gate.  Insert it immediately
# after the source-group identity gate; no labels or masks are needed.
anchor = 'if set(external_test["source_group"]) != EXPECTED_SOURCE_GROUPS:\n    raise RuntimeError("Unexpected source group entered external testing.")'
if anchor not in source:
    # The inherited wording may retain "internal testing" after identifier
    # replacement because it is plain prose. Accept that exact frozen variant.
    anchor = 'if set(external_test["source_group"]) != EXPECTED_SOURCE_GROUPS:\n    raise RuntimeError("Unexpected source group entered internal testing.")'
if anchor not in source:
    raise RuntimeError("Could not locate the external source-group gate in derived source.")
separation_gate = anchor + '''
partition_counts = external_test["locked_partition"].value_counts().to_dict()
source_counts = external_test["source_group"].value_counts().to_dict()
if partition_counts != {"external_msd_test": 194, "nih_negative_stress_test": 80}:
    raise RuntimeError(f"External partition counts changed: {partition_counts}")
if source_counts != {"MSD_EXTERNAL_MIXED": 194, "NIH_NEGATIVE_STRESS": 80}:
    raise RuntimeError(f"External source counts changed: {source_counts}")
'''
source = source.replace(anchor, separation_gate, 1)

# Final freeze metadata should explicitly say source-separated external blind
# inference. This is metadata only, not a model behavior change.
source = source.replace(
    '"inference_scope": "external_test_CT_only_blind_inference"',
    '"inference_scope": "source_separated_external_CT_only_blind_inference"',
)
source = source.replace(
    '"external_test_cases": EXPECTED_EXTERNAL_TEST_CASES,',
    '"external_test_cases": EXPECTED_EXTERNAL_TEST_CASES,\n'
    '    "external_partition_counts": {"external_msd_test": 194, "nih_negative_stress_test": 80},\n'
    '    "external_source_counts": {"MSD_EXTERNAL_MIXED": 194, "NIH_NEGATIVE_STRESS": 80},',
    1,
)

# Sanity locks: no internal-test output path or cohort selector may survive.
for prohibited in [
    "stage6a_blind_internal_test",
    "Stage6A_Internal_Test_Blind",
    '== "internal_test"',
    'EXPECTED_INTERNAL_TEST_CASES',
]:
    if prohibited in source:
        raise RuntimeError(f"Derived Stage 6C still contains prohibited token: {prohibited}")

required_tokens = [
    'EXPECTED_EXTERNAL_TEST_CASES = 274',
    '"external_msd_test"',
    '"nih_negative_stress_test"',
    '"MSD_EXTERNAL_MIXED"',
    '"NIH_NEGATIVE_STRESS"',
    'diagnostic_label_accessed": False',
    'test_mask_accessed": False',
]
for token in required_tokens:
    if token not in source:
        raise RuntimeError(f"Derived Stage 6C is missing required safety token: {token}")

expanded_bytes = source.encode("utf-8")
EXPANDED_SCRIPT.write_bytes(expanded_bytes)

build_record = {
    "stage": "6C-BUILD",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "purpose": "DERIVE_EXTERNAL_BLIND_INFERENCE_FROM_FROZEN_STAGE6A",
    "base_script": str(BASE_SCRIPT),
    "base_script_sha256": base_hash,
    "expanded_script": str(EXPANDED_SCRIPT),
    "expanded_script_sha256": sha256_bytes(expanded_bytes),
    "expected_external_cases": EXPECTED_EXTERNAL_CASES,
    "expected_external_partitions": EXPECTED_EXTERNAL_PARTITIONS,
    "expected_external_sources": EXPECTED_EXTERNAL_SOURCES,
    "diagnostic_labels_permitted_during_inference": False,
    "test_masks_permitted_during_inference": False,
    "model_change_permitted": False,
    "threshold_change_permitted": False,
    "derivation_status": "PASS",
}
QC_DIR.mkdir(parents=True, exist_ok=True)
BUILD_RECORD.write_text(json.dumps(build_record, indent=2), encoding="utf-8")

print("=" * 124)
print("STAGE 6C — EXTERNAL BLIND-INFERENCE DERIVATION")
print("=" * 124)
print(f"Base Stage 6A SHA-256: {base_hash}")
print(f"Expanded Stage 6C SHA-256: {build_record['expanded_script_sha256']}")
print(f"External cases: {EXPECTED_EXTERNAL_CASES} (MSD=194, NIH=80)")
print("Labels accessed by derivation: 0")
print("Masks accessed by derivation: 0")
print(f"Expanded script saved as:\n{EXPANDED_SCRIPT}")
print("Starting blind external inference...\n")

runpy.run_path(str(EXPANDED_SCRIPT), run_name="__main__")
