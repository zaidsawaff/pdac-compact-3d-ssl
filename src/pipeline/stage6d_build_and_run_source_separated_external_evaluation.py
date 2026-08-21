from pathlib import Path
import os
from datetime import datetime, timezone
import hashlib
import json
import re
import runpy

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


# =============================================================================
# STAGE 6D — POST-FREEZE SOURCE-SEPARATED EXTERNAL EVALUATION
# =============================================================================

PROJECT_ROOT = Path(os.environ.get("PDAC_PROJECT_ROOT", "/content/drive/MyDrive/PDAC_Public_Q1_Project"))
META_DIR = PROJECT_ROOT / "01_Metadata"
QC_DIR = PROJECT_ROOT / "02_Quality_Control"

BASE_EVALUATOR = META_DIR / "stage6b_postfreeze_internal_test_evaluation.py"
EXPANDED_EVALUATOR = META_DIR / "stage6d_external_evaluation_expanded.py"
BUILD_RECORD = QC_DIR / "stage6d_external_evaluation_build_record.json"

SPLIT_PATH = META_DIR / "stage0f_locked_study_split_index.csv"
FREEZE_PATH = META_DIR / "stage6c_blind_external_test_prediction_freeze.json"
COMBINED_CASE_PATH = QC_DIR / "stage6d_external_combined_case_metrics.csv"
COMBINED_CANDIDATE_PATH = QC_DIR / "stage6d_external_combined_candidate_truth_ledger.csv"

SOURCE_SUMMARY_PATH = QC_DIR / "stage6d_source_separated_external_summary.csv"
MSD_FROC_CURVE_PATH = QC_DIR / "stage6d_msd_external_froc_curve.csv"
MSD_FROC_POINTS_PATH = QC_DIR / "stage6d_msd_external_froc_operating_points.csv"
SOURCE_BOOTSTRAP_PATH = QC_DIR / "stage6d_source_separated_bootstrap_95ci.csv"
FINAL_AUDIT_PATH = QC_DIR / "stage6d_source_separated_external_evaluation_audit.json"

EXPECTED_CASES = 274
EXPECTED_PDAC = 98
EXPECTED_NON_PDAC = 176
EXPECTED_PARTITIONS = {"external_msd_test": 194, "nih_negative_stress_test": 80}
EXPECTED_SOURCES = {"MSD_EXTERNAL_MIXED": 194, "NIH_NEGATIVE_STRESS": 80}
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260728


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


def replace_exact(text, old, new, expected_count=1):
    observed = text.count(old)
    if observed != expected_count:
        raise RuntimeError(
            f"Stage 6B source drift: expected {expected_count} occurrence(s) "
            f"of {old!r}; observed {observed}."
        )
    return text.replace(old, new)


def truth_flags(series):
    return (
        series.fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "pass", "complete", "completed"})
    )


def percentile_ci(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


if not BASE_EVALUATOR.exists():
    raise FileNotFoundError(
        "The completed Stage 6B evaluator is missing:\n"
        f"{BASE_EVALUATOR}"
    )
if not FREEZE_PATH.exists():
    raise FileNotFoundError(f"Corrected Stage 6C freeze is missing:\n{FREEZE_PATH}")

with open(FREEZE_PATH, "r", encoding="utf-8") as file:
    external_freeze = json.load(file)
if external_freeze.get("stage") != "6C":
    raise RuntimeError("Run Stage 6C-R before external ground-truth evaluation.")
if external_freeze.get("prediction_freeze_complete") is not True:
    raise RuntimeError("Stage 6C external prediction freeze is incomplete.")
if int(external_freeze.get("diagnostic_labels_accessed", -1)) != 0:
    raise RuntimeError("External label-blinding record changed before evaluation.")
if int(external_freeze.get("test_masks_accessed", -1)) != 0:
    raise RuntimeError("External mask-blinding record changed before evaluation.")

base_bytes = BASE_EVALUATOR.read_bytes()
source = base_bytes.decode("utf-8")
base_hash = hashlib.sha256(base_bytes).hexdigest()

# -----------------------------------------------------------------------------
# Derive the external evaluator from the already validated Stage 6B evaluator.
# Numerical metric, mask-resampling, candidate-matching and bootstrap code is
# inherited. Only stage/cohort paths, expected identities and output names are
# changed.
# -----------------------------------------------------------------------------

source = source.replace("STAGE 6B", "STAGE 6D")
source = source.replace("Stage 6B", "Stage 6D")
source = source.replace('"stage": "6B"', '"stage": "6D"')
source = source.replace("PASS_INTERNAL_TEST_EVALUATION_COMPLETE", "PASS_EXTERNAL_TEST_EVALUATION_COMPLETE")

source = source.replace("STAGE6A_AUDIT_PATH", "STAGE6C_AUDIT_PATH")
source = source.replace("STAGE6A_LEDGER_PATH", "STAGE6C_LEDGER_PATH")
source = source.replace("stage6a_audit", "stage6c_audit")
source = source.replace("stage6a", "stage6c")
source = source.replace("Stage 6A", "Stage 6C")

source = source.replace("stage6b_internal_test", "stage6d_external_combined")
source = source.replace(
    "stage6b_geometry_review_required.json",
    "stage6d_external_geometry_review_required.json",
)

source = replace_exact(source, "EXPECTED_CASES = 293", "EXPECTED_CASES = 274")
source = replace_exact(source, "EXPECTED_PDAC = 87", "EXPECTED_PDAC = 98")
source = replace_exact(source, "EXPECTED_NON_PDAC = 206", "EXPECTED_NON_PDAC = 176")
source = replace_exact(
    source,
    'EXPECTED_PARTITION = "internal_test"\nEXPECTED_SOURCE = "PANORAMA_LOCAL"',
    'EXPECTED_PARTITIONS = {"external_msd_test", "nih_negative_stress_test"}\n'
    'EXPECTED_SOURCES = {"MSD_EXTERNAL_MIXED", "NIH_NEGATIVE_STRESS"}',
)

source = replace_exact(
    source,
    '    META_DIR / "stage6c_blind_internal_test_prediction_manifest.csv"',
    '    META_DIR / "stage6c_blind_external_test_prediction_manifest.csv"',
)
source = replace_exact(
    source,
    '    META_DIR / "stage6c_blind_internal_test_prediction_freeze.json"',
    '    META_DIR / "stage6c_blind_external_test_prediction_freeze.json"',
)
source = replace_exact(
    source,
    '    QC_DIR / "stage6c_blind_internal_test_inference_audit.json"',
    '    QC_DIR / "stage6c_blind_external_test_inference_audit.json"',
)
source = replace_exact(
    source,
    '    QC_DIR / "stage6c_blind_internal_test_resume_ledger.csv"',
    '    QC_DIR / "stage6c_blind_external_test_resume_ledger.csv"',
)

# The internal-only visual exception is not pre-authorized for any external
# case. Every new external geometry mismatch must pause before metrics.
resolution_block = '''# The first Stage 6D run paused before metrics for exactly one geometry case.
# Its targeted post-freeze CT-mask overlay has since been visually reviewed.
# Validate and freeze that resolution before any metric calculation resumes.
geometry_resolution = load_confirmed_geometry_resolution(freeze_hash)
print(
    "Targeted geometry resolution verified: "
    f"{geometry_resolution['study_id']} — {geometry_resolution['decision']}"
)
'''
if resolution_block not in source:
    # Comments inherited from Stage 6B are transformed only in the stage name;
    # tolerate the exact original comment text while keeping the executable
    # block strict.
    start = source.find("# The first Stage 6D run paused before metrics for exactly one geometry case.")
    end_token = "f\"{geometry_resolution['study_id']} — {geometry_resolution['decision']}\"\n)\n"
    end = source.find(end_token, start)
    if start < 0 or end < 0:
        raise RuntimeError("Could not isolate the internal-only geometry-resolution block.")
    end += len(end_token)
    source = source[:start] + (
        "# No external geometry exception is pre-authorized.\n"
        "geometry_resolution = None\n"
        "print(\"No pre-authorized external geometry exceptions.\")\n"
    ) + source[end:]
else:
    source = source.replace(
        resolution_block,
        "# No external geometry exception is pre-authorized.\n"
        "geometry_resolution = None\n"
        "print(\"No pre-authorized external geometry exceptions.\")\n",
        1,
    )

# External ground-truth cohort is the union of the locked MSD and NIH roles.
old_selection = '''internal = split.loc[
    split[partition_column].astype(str).str.strip() == EXPECTED_PARTITION
].copy()
'''
new_selection = '''external = split.loc[
    split[partition_column].astype(str).str.strip().isin(EXPECTED_PARTITIONS)
].copy()
'''
source = replace_exact(source, old_selection, new_selection)
source = re.sub(r"\binternal\b", "external", source)
source = source.replace("INTERNAL-TEST", "EXTERNAL-TEST")
source = source.replace("Internal-test", "External-test")
source = source.replace("internal-test", "external-test")

source = replace_exact(
    source,
    'if set(external[source_column].astype(str).str.strip()) != {EXPECTED_SOURCE}:',
    'if set(external[source_column].astype(str).str.strip()) != EXPECTED_SOURCES:',
)

# Require the exact source-separated counts after ground truth is opened.
cohort_anchor = '''if set(external["study_id"]) != set(manifest["study_id"]):
    raise RuntimeError("Ground-truth cohort IDs differ from frozen prediction IDs.")
'''
cohort_gate = cohort_anchor + '''external_partition_counts = external[partition_column].astype(str).str.strip().value_counts().to_dict()
external_source_counts = external[source_column].astype(str).str.strip().value_counts().to_dict()
if external_partition_counts != {"external_msd_test": 194, "nih_negative_stress_test": 80}:
    raise RuntimeError(f"External partition counts changed: {external_partition_counts}")
if external_source_counts != {"MSD_EXTERNAL_MIXED": 194, "NIH_NEGATIVE_STRESS": 80}:
    raise RuntimeError(f"External source counts changed: {external_source_counts}")
'''
source = replace_exact(source, cohort_anchor, cohort_gate)

# No reviewed exception is expected in external data. Any mismatch remains in
# the review set and stops execution before metric calculation.
source = replace_exact(
    source,
    '"Exactly one reviewed header-orientation mismatch is explicitly resolved": reviewed_geometry_count == 1,',
    '"No external geometry exception was silently accepted": reviewed_geometry_count == 0,',
)
source = source.replace(
    '    "geometry_resolution_path": str(GEOMETRY_RESOLUTION_PATH),\n', ""
)
source = source.replace(
    '"visually_confirmed_index_aligned_header_mismatch_cases": reviewed_geometry_count,',
    '"visually_confirmed_index_aligned_header_mismatch_cases": reviewed_geometry_count,',
)

# Update descriptive hard-coded counts and labels.
source = source.replace("exactly 293 studies", "exactly 274 studies")
source = source.replace("all 293 frozen prediction files", "all 274 frozen prediction files")
source = source.replace("Exactly 293 frozen external-test predictions", "Exactly 274 frozen external-test predictions")
source = source.replace("exactly 87 PDAC and 206 non-PDAC", "exactly 98 PDAC and 176 non-PDAC")
source = source.replace("Every external-test patient", "Every external patient")

# Safety: the external pipeline may not silently refer back to the Stage 6A
# prediction freeze or select the single local/internal review case.
for prohibited in [
    "stage6a_blind_internal_test",
    'EXPECTED_PARTITION = "internal_test"',
    'EXPECTED_SOURCE = "PANORAMA_LOCAL"',
]:
    if prohibited in source:
        raise RuntimeError(f"Derived Stage 6D retains prohibited token: {prohibited}")

compile(source, str(EXPANDED_EVALUATOR), "exec")
expanded_bytes = source.encode("utf-8")
EXPANDED_EVALUATOR.write_bytes(expanded_bytes)

build_record = {
    "stage": "6D-BUILD",
    "created_at_utc": utc_now(),
    "base_stage6b_sha256": base_hash,
    "expanded_stage6d_sha256": hashlib.sha256(expanded_bytes).hexdigest(),
    "expected_cases": EXPECTED_CASES,
    "expected_PDAC_cases": EXPECTED_PDAC,
    "expected_non_PDAC_cases": EXPECTED_NON_PDAC,
    "expected_partitions": {"external_msd_test": 194, "nih_negative_stress_test": 80},
    "expected_sources": {"MSD_EXTERNAL_MIXED": 194, "NIH_NEGATIVE_STRESS": 80},
    "external_geometry_exceptions_pre_authorized": 0,
    "threshold_selection_permitted": False,
    "model_fitting_permitted": False,
    "build_status": "PASS",
}
QC_DIR.mkdir(parents=True, exist_ok=True)
BUILD_RECORD.write_text(json.dumps(build_record, indent=2), encoding="utf-8")

print("=" * 124)
print("STAGE 6D — EXTERNAL EVALUATION DERIVATION")
print("=" * 124)
print(f"Frozen external cases: {EXPECTED_CASES} (MSD=194, NIH=80)")
print("Pre-authorized geometry exceptions: 0")
print("Threshold adjustment permitted: False")
print("Model fitting permitted: False")
print(f"Expanded evaluator:\n{EXPANDED_EVALUATOR}\n")

# If a geometry mismatch is found, the inherited evaluator raises SystemExit(0)
# before any metric and this launcher stops here. On a complete geometry pass,
# the namespace is returned for the source-separated analysis below.
namespace = runpy.run_path(str(EXPANDED_EVALUATOR), run_name="__main__")


# =============================================================================
# SOURCE-SEPARATED PRIMARY EXTERNAL ANALYSIS
# =============================================================================

case_metrics = pd.read_csv(COMBINED_CASE_PATH, dtype={"study_id": str, "patient_id": str})
candidate_truth = pd.read_csv(COMBINED_CANDIDATE_PATH, dtype={"study_id": str, "patient_id": str})
split = pd.read_csv(SPLIT_PATH, dtype=str)
split["study_id"] = split["study_id"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)

external_roles = split.loc[
    split["locked_partition"].astype(str).str.strip().isin(EXPECTED_PARTITIONS)
][["study_id", "locked_partition", "source_group"]].copy()
if len(external_roles) != EXPECTED_CASES or external_roles["study_id"].nunique() != EXPECTED_CASES:
    raise RuntimeError("External role index changed during source-separated analysis.")

case_metrics = case_metrics.merge(external_roles, on="study_id", how="inner", validate="one_to_one")
if len(case_metrics) != EXPECTED_CASES:
    raise RuntimeError("Source-separated case merge is incomplete.")

deployment_threshold = float(namespace["deployment_threshold"])
froc_from_candidate_truth = namespace["froc_from_candidate_truth"]
froc_operating_points = namespace["froc_operating_points"]
standard_froc_targets = namespace["STANDARD_FROC_TARGETS"]

msd = case_metrics.loc[case_metrics["source_group"] == "MSD_EXTERNAL_MIXED"].copy()
nih = case_metrics.loc[case_metrics["source_group"] == "NIH_NEGATIVE_STRESS"].copy()
if len(msd) != 194 or msd["label_binary"].astype(int).value_counts().to_dict() != {1: 98, 0: 96}:
    raise RuntimeError("MSD external diagnostic lock changed.")
if len(nih) != 80 or nih["label_binary"].astype(int).value_counts().to_dict() != {0: 80}:
    raise RuntimeError("NIH negative-stress diagnostic lock changed.")

msd_ids = set(msd["study_id"].astype(str))
nih_ids = set(nih["study_id"].astype(str))
msd_candidates = candidate_truth.loc[candidate_truth["study_id"].astype(str).isin(msd_ids)].copy()
nih_candidates = candidate_truth.loc[candidate_truth["study_id"].astype(str).isin(nih_ids)].copy()

msd_labels = msd["label_binary"].astype(int).to_numpy()
msd_scores = msd["maximum_candidate_score"].astype(float).to_numpy()
msd_auc = float(roc_auc_score(msd_labels, msd_scores))
msd_ap = float(average_precision_score(msd_labels, msd_scores))
msd_froc_curve = froc_from_candidate_truth(msd_candidates, 194, 98)
msd_froc_points = froc_operating_points(msd_froc_curve)
msd_froc_curve.to_csv(MSD_FROC_CURVE_PATH, index=False)
msd_froc_points.to_csv(MSD_FROC_POINTS_PATH, index=False)
msd_standard_froc = msd_froc_points.loc[
    msd_froc_points["target_false_positives_per_case"].isin(standard_froc_targets)
]
msd_mean_froc = float(msd_standard_froc["sensitivity"].mean())

msd_positive = msd.loc[msd["label_binary"].astype(int) == 1]
msd_negative = msd.loc[msd["label_binary"].astype(int) == 0]
msd_locked = msd_candidates.loc[
    msd_candidates["candidate_score"].astype(float) >= deployment_threshold
]
msd_locked_tp = int(truth_flags(msd_locked["is_true_positive_candidate"]).sum())
msd_locked_fp = int(truth_flags(msd_locked["is_false_positive_candidate"]).sum())
msd_sensitivity = float(msd_locked_tp / 98.0)
msd_fp_per_case = float(msd_locked_fp / 194.0)
msd_specificity = float((~truth_flags(msd_negative["case_positive_at_locked_threshold"])).mean())

nih_locked = nih_candidates.loc[
    nih_candidates["candidate_score"].astype(float) >= deployment_threshold
]
nih_locked_fp = int(truth_flags(nih_locked["is_false_positive_candidate"]).sum())
nih_fp_per_case = float(nih_locked_fp / 80.0)
nih_specificity = float((~truth_flags(nih["case_positive_at_locked_threshold"])).mean())
nih_case_positive_fraction = float(truth_flags(nih["case_positive_at_locked_threshold"]).mean())

source_summary = pd.DataFrame(
    [
        {
            "source_group": "MSD_EXTERNAL_MIXED",
            "cases": 194,
            "PDAC_cases": 98,
            "non_PDAC_cases": 96,
            "case_level_AUC": msd_auc,
            "case_level_average_precision": msd_ap,
            "mean_pancreas_Dice": float(msd["pancreas_dice_at_0_5"].mean()),
            "mean_PDAC_lesion_Dice": float(msd_positive["lesion_dice_at_0_5"].mean()),
            "PDAC_lesion_crop_presence": float(truth_flags(msd_positive["lesion_target_present_in_crop"]).mean()),
            "locked_threshold": deployment_threshold,
            "locked_threshold_lesion_sensitivity": msd_sensitivity,
            "locked_threshold_false_positives_per_case": msd_fp_per_case,
            "locked_threshold_negative_specificity": msd_specificity,
            "mean_FROC_sensitivity_0.25_0.5_1_2_4_FP_per_case": msd_mean_froc,
        },
        {
            "source_group": "NIH_NEGATIVE_STRESS",
            "cases": 80,
            "PDAC_cases": 0,
            "non_PDAC_cases": 80,
            "case_level_AUC": np.nan,
            "case_level_average_precision": np.nan,
            "mean_pancreas_Dice": float(nih["pancreas_dice_at_0_5"].mean()),
            "mean_PDAC_lesion_Dice": np.nan,
            "PDAC_lesion_crop_presence": np.nan,
            "locked_threshold": deployment_threshold,
            "locked_threshold_lesion_sensitivity": np.nan,
            "locked_threshold_false_positives_per_case": nih_fp_per_case,
            "locked_threshold_negative_specificity": nih_specificity,
            "mean_FROC_sensitivity_0.25_0.5_1_2_4_FP_per_case": np.nan,
        },
    ]
)
source_summary.to_csv(SOURCE_SUMMARY_PATH, index=False)


# =============================================================================
# SOURCE-SPECIFIC PATIENT BOOTSTRAP 95% CIs
# =============================================================================

rng = np.random.default_rng(BOOTSTRAP_SEED)
msd_positive_index = np.flatnonzero(msd_labels == 1)
msd_negative_index = np.flatnonzero(msd_labels == 0)
msd_bootstrap = []
for _ in range(BOOTSTRAP_REPLICATES):
    pos = rng.choice(msd_positive_index, size=len(msd_positive_index), replace=True)
    neg = rng.choice(msd_negative_index, size=len(msd_negative_index), replace=True)
    chosen = np.concatenate([pos, neg])
    sample = msd.iloc[chosen]
    sample_positive = msd.iloc[pos]
    sample_negative = msd.iloc[neg]
    labels = msd_labels[chosen]
    scores = msd_scores[chosen]
    msd_bootstrap.append(
        {
            "case_level_AUC": float(roc_auc_score(labels, scores)),
            "case_level_average_precision": float(average_precision_score(labels, scores)),
            "mean_pancreas_Dice": float(sample["pancreas_dice_at_0_5"].mean()),
            "mean_PDAC_lesion_Dice": float(sample_positive["lesion_dice_at_0_5"].mean()),
            "PDAC_localization_sensitivity_locked_threshold": float(
                truth_flags(sample_positive["lesion_localized_at_locked_threshold"]).mean()
            ),
            "negative_specificity_locked_threshold": float(
                (~truth_flags(sample_negative["case_positive_at_locked_threshold"])).mean()
            ),
        }
    )
msd_bootstrap = pd.DataFrame(msd_bootstrap)

nih_locked_counts = (
    nih_locked.groupby("study_id").size().reindex(nih["study_id"].astype(str), fill_value=0).to_numpy(dtype=float)
)
nih_bootstrap = []
for _ in range(BOOTSTRAP_REPLICATES):
    chosen = rng.choice(np.arange(len(nih)), size=len(nih), replace=True)
    sample = nih.iloc[chosen]
    nih_bootstrap.append(
        {
            "mean_pancreas_Dice": float(sample["pancreas_dice_at_0_5"].mean()),
            "negative_specificity_locked_threshold": float(
                (~truth_flags(sample["case_positive_at_locked_threshold"])).mean()
            ),
            "false_positive_candidates_per_case_locked_threshold": float(
                nih_locked_counts[chosen].mean()
            ),
        }
    )
nih_bootstrap = pd.DataFrame(nih_bootstrap)

ci_rows = []
msd_points = {
    "case_level_AUC": msd_auc,
    "case_level_average_precision": msd_ap,
    "mean_pancreas_Dice": float(msd["pancreas_dice_at_0_5"].mean()),
    "mean_PDAC_lesion_Dice": float(msd_positive["lesion_dice_at_0_5"].mean()),
    "PDAC_localization_sensitivity_locked_threshold": float(
        truth_flags(msd_positive["lesion_localized_at_locked_threshold"]).mean()
    ),
    "negative_specificity_locked_threshold": msd_specificity,
}
for metric, point in msd_points.items():
    lower, upper = percentile_ci(msd_bootstrap[metric])
    ci_rows.append(
        {
            "source_group": "MSD_EXTERNAL_MIXED",
            "metric": metric,
            "point_estimate": point,
            "bootstrap_95CI_lower": lower,
            "bootstrap_95CI_upper": upper,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_unit": "patient_stratified_by_diagnosis",
        }
    )
nih_points = {
    "mean_pancreas_Dice": float(nih["pancreas_dice_at_0_5"].mean()),
    "negative_specificity_locked_threshold": nih_specificity,
    "false_positive_candidates_per_case_locked_threshold": nih_fp_per_case,
}
for metric, point in nih_points.items():
    lower, upper = percentile_ci(nih_bootstrap[metric])
    ci_rows.append(
        {
            "source_group": "NIH_NEGATIVE_STRESS",
            "metric": metric,
            "point_estimate": point,
            "bootstrap_95CI_lower": lower,
            "bootstrap_95CI_upper": upper,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_unit": "patient_case_resampling",
        }
    )
source_bootstrap = pd.DataFrame(ci_rows)
source_bootstrap.to_csv(SOURCE_BOOTSTRAP_PATH, index=False)


readiness = {
    "Corrected Stage 6C external freeze passed before ground truth": external_freeze.get("stage") == "6C"
    and external_freeze.get("prediction_freeze_complete") is True,
    "Exactly 274 external cases evaluated": len(case_metrics) == EXPECTED_CASES,
    "MSD external cohort is exactly 98 PDAC plus 96 non-PDAC": len(msd) == 194
    and msd["label_binary"].astype(int).value_counts().to_dict() == {1: 98, 0: 96},
    "NIH stress cohort is exactly 80 non-PDAC": len(nih) == 80
    and nih["label_binary"].astype(int).value_counts().to_dict() == {0: 80},
    "MSD case-level AUC is finite": np.isfinite(msd_auc),
    "MSD average precision is finite": np.isfinite(msd_ap),
    "MSD FROC has six prespecified operating points": len(msd_froc_points) == 6,
    "NIH AUC/FROC are not computed because there are no positive cases": True,
    "Locked deployment threshold remains unchanged": np.isclose(
        deployment_threshold,
        float(external_freeze["locked_deployment_threshold"]),
        rtol=0,
        atol=1e-8,
    ),
    "No model fitting occurred in Stage 6D": True,
    "No threshold selection occurred in Stage 6D": True,
    "All source-specific bootstrap confidence intervals are finite": np.isfinite(
        source_bootstrap[["point_estimate", "bootstrap_95CI_lower", "bootstrap_95CI_upper"]].to_numpy(dtype=float)
    ).all(),
}
readiness = {name: bool(value) for name, value in readiness.items()}
all_checks_pass = bool(all(readiness.values()))

final_audit = {
    "stage": "6D",
    "created_at_utc": utc_now(),
    "result": "PASS_SOURCE_SEPARATED_EXTERNAL_EVALUATION_COMPLETE" if all_checks_pass else "FAIL",
    "all_checks_pass": all_checks_pass,
    "readiness_checks": readiness,
    "stage6c_freeze_sha256": sha256_file(FREEZE_PATH),
    "external_cases": EXPECTED_CASES,
    "MSD_external_cases": 194,
    "NIH_negative_stress_cases": 80,
    "locked_threshold": deployment_threshold,
    "MSD_case_level_AUC": msd_auc,
    "MSD_case_level_average_precision": msd_ap,
    "MSD_locked_threshold_lesion_sensitivity": msd_sensitivity,
    "MSD_locked_threshold_false_positives_per_case": msd_fp_per_case,
    "MSD_locked_threshold_negative_specificity": msd_specificity,
    "MSD_mean_FROC_sensitivity": msd_mean_froc,
    "NIH_locked_threshold_false_positives_per_case": nih_fp_per_case,
    "NIH_locked_threshold_negative_specificity": nih_specificity,
    "NIH_case_positive_fraction_locked_threshold": nih_case_positive_fraction,
    "model_fitting_performed": False,
    "threshold_selection_performed": False,
    "source_summary_path": str(SOURCE_SUMMARY_PATH),
    "MSD_froc_curve_path": str(MSD_FROC_CURVE_PATH),
    "MSD_froc_operating_points_path": str(MSD_FROC_POINTS_PATH),
    "source_bootstrap_95ci_path": str(SOURCE_BOOTSTRAP_PATH),
}
with open(FINAL_AUDIT_PATH, "w", encoding="utf-8") as file:
    json.dump(final_audit, file, indent=2, ensure_ascii=False)

print("\n" + "=" * 124)
print("STAGE 6D — SOURCE-SEPARATED EXTERNAL RESULTS")
print("=" * 124)
print("\nMSD_EXTERNAL_MIXED (N=194; PDAC=98, non-PDAC=96)")
print(f"  Case-level AUC: {msd_auc:.6f}")
print(f"  Average precision: {msd_ap:.6f}")
print(f"  Locked-threshold lesion sensitivity: {msd_sensitivity:.6f}")
print(f"  Locked-threshold FP/case: {msd_fp_per_case:.6f}")
print(f"  Negative specificity: {msd_specificity:.6f}")
print(f"  Mean FROC sensitivity: {msd_mean_froc:.6f}")
print("\nNIH_NEGATIVE_STRESS (N=80; all non-PDAC)")
print("  AUC/FROC: not applicable (no positive cases)")
print(f"  Locked-threshold FP/case: {nih_fp_per_case:.6f}")
print(f"  Negative specificity: {nih_specificity:.6f}")
print("\nREADINESS CHECKS")
for name, passed in readiness.items():
    print(f"  {name}: {passed}")
print(f"\nSource-separated summary:\n{SOURCE_SUMMARY_PATH}")
print(f"MSD FROC points:\n{MSD_FROC_POINTS_PATH}")
print(f"Source-specific bootstrap CIs:\n{SOURCE_BOOTSTRAP_PATH}")
print(f"Final audit:\n{FINAL_AUDIT_PATH}")
print("\n" + "=" * 124)
print(
    "STAGE 6D RESULT: "
    + ("PASS — SOURCE-SEPARATED EXTERNAL EVALUATION COMPLETE" if all_checks_pass else "FAIL")
)
print("=" * 124)
if not all_checks_pass:
    raise RuntimeError("Stage 6D failed one or more final readiness checks.")
