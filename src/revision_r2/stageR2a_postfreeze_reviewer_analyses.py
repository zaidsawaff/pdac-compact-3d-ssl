from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import os
import hashlib
import json
import math
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# STAGE R2A — POST-FREEZE REVIEWER-REQUESTED SECONDARY ANALYSES
# =============================================================================

PROJECT_ROOT = Path(os.environ.get("PDAC_PROJECT_ROOT", "/content/drive/MyDrive/PDAC_Public_Q1_Project"))
RUNTIME_ROOT = Path(os.environ.get("PDAC_RUNTIME_ROOT", "/content"))
RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
QC_DIR = PROJECT_ROOT / "02_Quality_Control"
RESULT_DIR = (
    PROJECT_ROOT / "03_Results" / "Revision_R2_Reviewer_Analyses"
)

INTERNAL_CASE_PATH = QC_DIR / "stageR1g_internal_test_case_metrics.csv"
INTERNAL_CANDIDATE_PATH = (
    QC_DIR / "stageR1g_internal_test_candidate_truth_ledger.csv"
)
INTERNAL_FROC_PATH = QC_DIR / "stageR1g_internal_test_froc_operating_points.csv"
EXTERNAL_CASE_PATH = QC_DIR / "stageR1i_external_combined_case_metrics.csv"
EXTERNAL_CANDIDATE_PATH = (
    QC_DIR / "stageR1i_external_combined_candidate_truth_ledger.csv"
)
MSD_FROC_PATH = QC_DIR / "stageR1i_msd_external_froc_operating_points.csv"
SOURCE_CI_PATH = QC_DIR / "stageR1i_source_separated_bootstrap_95ci.csv"
INTERNAL_AUDIT_PATH = QC_DIR / "stageR1g_internal_test_evaluation_audit.json"
EXTERNAL_AUDIT_PATH = (
    QC_DIR / "stageR1i_source_separated_external_evaluation_audit.json"
)

OUTPUT_MSD_CI = RESULT_DIR / "stageR2a_msd_complete_bootstrap_95ci.csv"
OUTPUT_FROC = RESULT_DIR / "stageR2a_internal_vs_msd_froc_operating_points.csv"
OUTPUT_MANUAL = RESULT_DIR / "stageR2a_manual_reference_only_analysis.csv"
OUTPUT_SIZE_CASE = RESULT_DIR / "stageR2a_lesion_size_case_ledger.csv"
OUTPUT_SIZE_SUMMARY = RESULT_DIR / "stageR2a_lesion_size_stratified_analysis.csv"
OUTPUT_FIGURE_PNG = RESULT_DIR / "stageR2a_internal_vs_msd_froc.png"
OUTPUT_FIGURE_PDF = RESULT_DIR / "stageR2a_internal_vs_msd_froc.pdf"
OUTPUT_AUDIT = QC_DIR / "stageR2a_postfreeze_reviewer_analyses_audit.json"
OUTPUT_AUDIT_COPY = RESULT_DIR / "stageR2a_postfreeze_reviewer_analyses_audit.json"
OUTPUT_SUMMARY = RESULT_DIR / "stageR2a_reviewer_analysis_summary.md"
OUTPUT_ZIP_BASE = (RUNTIME_ROOT / "stageR2A_reviewer_analyses")

EXPECTED_INTERNAL = 293
EXPECTED_INTERNAL_PDAC = 87
EXPECTED_INTERNAL_NON_PDAC = 206
EXPECTED_EXTERNAL = 274
EXPECTED_MSD = 194
EXPECTED_MSD_PDAC = 98
EXPECTED_MSD_NON_PDAC = 96

BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260728
VOXEL_VOLUME_MM3 = 1.25 * 1.25 * 2.0
STANDARD_FROC_TARGETS = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while True:
            chunk = file.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def truth_flags(series: pd.Series) -> pd.Series:
    return (
        series.fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes"})
    )


def normalize_text(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().str.replace("_", "-", regex=False)


def percentile_ci(values: list[float] | np.ndarray) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all() or len(array) != BOOTSTRAP_REPLICATES:
        raise RuntimeError("Bootstrap distribution is incomplete or non-finite.")
    lower, upper = np.quantile(array, [0.025, 0.975])
    return float(lower), float(upper)


def mean_ci(values: np.ndarray, seed: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0 or not np.isfinite(values).all():
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    replicates = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    indices = np.arange(len(values))
    for index in range(BOOTSTRAP_REPLICATES):
        chosen = rng.choice(indices, size=len(indices), replace=True)
        replicates[index] = float(values[chosen].mean())
    lower, upper = percentile_ci(replicates)
    return float(values.mean()), lower, upper


def proportion_ci(flags: np.ndarray, seed: int) -> tuple[float, float, float]:
    flags = np.asarray(flags, dtype=float)
    return mean_ci(flags, seed)


def size_category(diameter_mm: float) -> str:
    if diameter_mm <= 20.0:
        return "small_<=20mm"
    if diameter_mm <= 40.0:
        return "medium_>20-40mm"
    return "large_>40mm"


def format_estimate(point: float, lower: float, upper: float, digits: int = 3) -> str:
    if not np.isfinite(point):
        return "NA"
    return f"{point:.{digits}f} ({lower:.{digits}f}-{upper:.{digits}f})"


def json_safe(value):
    """Recursively convert NumPy/Pandas scalar values to JSON-native types."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


required_paths = [
    INTERNAL_CASE_PATH,
    INTERNAL_CANDIDATE_PATH,
    INTERNAL_FROC_PATH,
    EXTERNAL_CASE_PATH,
    EXTERNAL_CANDIDATE_PATH,
    MSD_FROC_PATH,
    SOURCE_CI_PATH,
    INTERNAL_AUDIT_PATH,
    EXTERNAL_AUDIT_PATH,
]
missing = [str(path) for path in required_paths if not path.exists()]
if missing:
    raise FileNotFoundError("Missing required frozen evidence:\n" + "\n".join(missing))

RESULT_DIR.mkdir(parents=True, exist_ok=True)

with open(INTERNAL_AUDIT_PATH, "r", encoding="utf-8") as file:
    internal_audit = json.load(file)
with open(EXTERNAL_AUDIT_PATH, "r", encoding="utf-8") as file:
    external_audit = json.load(file)
if internal_audit.get("all_checks_pass") is not True:
    raise RuntimeError("Frozen internal-test evaluation audit is not PASS.")
if external_audit.get("all_checks_pass") is not True:
    raise RuntimeError("Frozen source-separated external evaluation audit is not PASS.")

internal = pd.read_csv(INTERNAL_CASE_PATH, dtype={"study_id": str, "patient_id": str})
external = pd.read_csv(EXTERNAL_CASE_PATH, dtype={"study_id": str, "patient_id": str})
internal_candidates = pd.read_csv(
    INTERNAL_CANDIDATE_PATH, dtype={"study_id": str, "patient_id": str}
)
external_candidates = pd.read_csv(
    EXTERNAL_CANDIDATE_PATH, dtype={"study_id": str, "patient_id": str}
)
internal_froc = pd.read_csv(INTERNAL_FROC_PATH)
msd_froc = pd.read_csv(MSD_FROC_PATH)
source_ci = pd.read_csv(SOURCE_CI_PATH)

for frame in [internal, external, internal_candidates, external_candidates]:
    frame["study_id"] = frame["study_id"].astype(str).str.strip()
if "source_group" not in external.columns:
    raise RuntimeError("External case ledger lacks source_group.")

internal["annotation_type_normalized"] = normalize_text(internal["annotation_type"])
external["annotation_type_normalized"] = normalize_text(external["annotation_type"])
external["source_group_normalized"] = external["source_group"].astype(str).str.strip()

msd = external.loc[
    external["source_group_normalized"] == "MSD_EXTERNAL_MIXED"
].copy()
msd_ids = set(msd["study_id"])
msd_candidates = external_candidates.loc[
    external_candidates["study_id"].isin(msd_ids)
].copy()

if len(internal) != EXPECTED_INTERNAL or internal["study_id"].nunique() != EXPECTED_INTERNAL:
    raise RuntimeError("Internal case lock changed.")
if internal["label_binary"].astype(int).value_counts().to_dict() != {
    0: EXPECTED_INTERNAL_NON_PDAC,
    1: EXPECTED_INTERNAL_PDAC,
}:
    raise RuntimeError("Internal diagnostic lock changed.")
if len(external) != EXPECTED_EXTERNAL or external["study_id"].nunique() != EXPECTED_EXTERNAL:
    raise RuntimeError("External case lock changed.")
if len(msd) != EXPECTED_MSD or msd["label_binary"].astype(int).value_counts().to_dict() != {
    0: EXPECTED_MSD_NON_PDAC,
    1: EXPECTED_MSD_PDAC,
}:
    raise RuntimeError("MSD source lock changed.")

threshold_values = pd.concat(
    [internal["locked_deployment_threshold"], external["locked_deployment_threshold"]]
).astype(float)
if threshold_values.nunique() != 1:
    raise RuntimeError("Deployment threshold is not identical across frozen ledgers.")
locked_threshold = float(threshold_values.iloc[0])

print("=" * 120)
print("STAGE R2A — POST-FREEZE REVIEWER-REQUESTED SECONDARY ANALYSES")
print("=" * 120)
print("GPU required: False")
print("Model training/inference: False")
print("Threshold selection/adjustment: False")
print(f"Frozen threshold: {locked_threshold:.6f}")
print(f"Internal cases: {len(internal)}")
print(f"MSD cases: {len(msd)}")


# =============================================================================
# 1. COMPLETE MSD CONFIDENCE INTERVALS, INCLUDING FP/CASE
# =============================================================================

msd_ci = source_ci.loc[
    source_ci["source_group"].astype(str) == "MSD_EXTERNAL_MIXED"
].copy()
expected_msd_metrics = {
    "case_level_AUC",
    "case_level_average_precision",
    "mean_pancreas_Dice",
    "mean_PDAC_lesion_Dice",
    "PDAC_localization_sensitivity_locked_threshold",
    "negative_specificity_locked_threshold",
}
if set(msd_ci["metric"].astype(str)) != expected_msd_metrics:
    raise RuntimeError("Existing MSD bootstrap metric set changed.")

msd_locked_fp = (
    msd_candidates.loc[
        (msd_candidates["candidate_score"].astype(float) >= locked_threshold)
        & truth_flags(msd_candidates["is_false_positive_candidate"])
    ]
    .groupby("study_id")
    .size()
    .reindex(msd["study_id"], fill_value=0)
    .to_numpy(dtype=float)
)
msd_labels = msd["label_binary"].astype(int).to_numpy()
positive_index = np.flatnonzero(msd_labels == 1)
negative_index = np.flatnonzero(msd_labels == 0)
rng = np.random.default_rng(BOOTSTRAP_SEED)
fp_replicates = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
for bootstrap_index in range(BOOTSTRAP_REPLICATES):
    sampled_positive = rng.choice(positive_index, size=len(positive_index), replace=True)
    sampled_negative = rng.choice(negative_index, size=len(negative_index), replace=True)
    sampled = np.concatenate([sampled_positive, sampled_negative])
    fp_replicates[bootstrap_index] = float(msd_locked_fp[sampled].mean())
fp_lower, fp_upper = percentile_ci(fp_replicates)
fp_point = float(msd_locked_fp.mean())

msd_ci = pd.concat(
    [
        msd_ci,
        pd.DataFrame(
            [
                {
                    "source_group": "MSD_EXTERNAL_MIXED",
                    "metric": "false_positive_candidates_per_case_locked_threshold",
                    "point_estimate": fp_point,
                    "bootstrap_95CI_lower": fp_lower,
                    "bootstrap_95CI_upper": fp_upper,
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                    "bootstrap_unit": "patient_stratified_by_diagnosis",
                }
            ]
        ),
    ],
    ignore_index=True,
)
msd_ci.to_csv(OUTPUT_MSD_CI, index=False)


# =============================================================================
# 2. INTERNAL-TO-MSD FROC TABLE AND PRIMARY EXTERNAL FIGURE
# =============================================================================

def froc_side(frame: pd.DataFrame, cohort: str) -> pd.DataFrame:
    needed = {
        "target_false_positives_per_case",
        "achieved_false_positives_per_case",
        "sensitivity",
        "probability_threshold",
        "true_positive_lesions",
        "false_positive_candidates",
    }
    if not needed.issubset(frame.columns):
        raise RuntimeError(f"{cohort} FROC table is missing columns.")
    result = frame.loc[
        frame["target_false_positives_per_case"].astype(float).isin(STANDARD_FROC_TARGETS),
        list(needed),
    ].copy()
    result["cohort"] = cohort
    return result


froc_long = pd.concat(
    [froc_side(internal_froc, "PANORAMA_INTERNAL_TEST"), froc_side(msd_froc, "MSD_EXTERNAL_MIXED")],
    ignore_index=True,
)
froc_long = froc_long.sort_values(["cohort", "target_false_positives_per_case"])
froc_long.to_csv(OUTPUT_FROC, index=False)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
styles = {
    "PANORAMA_INTERNAL_TEST": {
        "label": "Internal test (n=293)",
        "color": "#2F6690",
        "marker": "o",
    },
    "MSD_EXTERNAL_MIXED": {
        "label": "MSD source-separated external (n=194)",
        "color": "#E07A5F",
        "marker": "s",
    },
}
for cohort, style in styles.items():
    group = froc_long.loc[froc_long["cohort"] == cohort].sort_values(
        "achieved_false_positives_per_case"
    )
    ax.plot(
        group["achieved_false_positives_per_case"],
        group["sensitivity"],
        color=style["color"],
        marker=style["marker"],
        linewidth=2.2,
        markersize=6,
        label=style["label"],
    )
ax.set_xlabel("Achieved false positives per case")
ax.set_ylabel("Lesion-localization sensitivity")
ax.set_title("FROC using achieved false-positive coordinates", weight="bold")
ax.set_ylim(0.50, 1.00)
ax.grid(True, alpha=0.25)
ax.legend(loc="lower right", frameon=True)
fig.savefig(OUTPUT_FIGURE_PNG, dpi=400, bbox_inches="tight")
fig.savefig(OUTPUT_FIGURE_PDF, bbox_inches="tight")
plt.close(fig)


# =============================================================================
# 3. MANUAL-REFERENCE-ONLY ANALYSIS (PDAC CASES ONLY)
# =============================================================================

manual_rows = []
manual_sources = [
    ("PANORAMA_INTERNAL_TEST", internal),
    ("MSD_EXTERNAL_MIXED", msd),
    ("INTERNAL_PLUS_MSD", pd.concat([internal, msd], ignore_index=True)),
]
for cohort_index, (cohort, frame) in enumerate(manual_sources):
    group = frame.loc[
        (frame["label_binary"].astype(int) == 1)
        & (frame["annotation_type_normalized"] == "manual")
    ].copy()
    lesion_point, lesion_low, lesion_high = mean_ci(
        group["lesion_dice_at_0_5"].astype(float).to_numpy(),
        BOOTSTRAP_SEED + 100 + cohort_index,
    )
    pancreas_point, pancreas_low, pancreas_high = mean_ci(
        group["pancreas_dice_at_0_5"].astype(float).to_numpy(),
        BOOTSTRAP_SEED + 200 + cohort_index,
    )
    sensitivity_point, sensitivity_low, sensitivity_high = proportion_ci(
        truth_flags(group["lesion_localized_at_locked_threshold"]).astype(float).to_numpy(),
        BOOTSTRAP_SEED + 300 + cohort_index,
    )
    manual_rows.append(
        {
            "cohort": cohort,
            "analysis_role": "post_hoc_manual_reference_only",
            "manual_PDAC_cases": int(len(group)),
            "mean_pancreas_Dice": pancreas_point,
            "mean_pancreas_Dice_95CI_lower": pancreas_low,
            "mean_pancreas_Dice_95CI_upper": pancreas_high,
            "mean_lesion_Dice": lesion_point,
            "mean_lesion_Dice_95CI_lower": lesion_low,
            "mean_lesion_Dice_95CI_upper": lesion_high,
            "lesion_localization_sensitivity_locked_threshold": sensitivity_point,
            "lesion_localization_sensitivity_95CI_lower": sensitivity_low,
            "lesion_localization_sensitivity_95CI_upper": sensitivity_high,
            "locked_threshold": locked_threshold,
            "classification_metrics_applicable": False,
            "classification_metrics_not_applicable_reason": (
                "manual-reference subset contains PDAC reference masks only"
            ),
        }
    )
manual_summary = pd.DataFrame(manual_rows)
manual_summary.to_csv(OUTPUT_MANUAL, index=False)


# =============================================================================
# 4. LESION-SIZE-STRATIFIED ANALYSIS
# =============================================================================

positive_frames = []
for cohort, frame in [
    ("PANORAMA_INTERNAL_TEST", internal),
    ("MSD_EXTERNAL_MIXED", msd),
]:
    positive = frame.loc[frame["label_binary"].astype(int) == 1].copy()
    positive["cohort"] = cohort
    positive_frames.append(positive)
size_cases = pd.concat(positive_frames, ignore_index=True)
size_cases["lesion_volume_mm3"] = (
    size_cases["lesion_target_voxels_in_crop"].astype(float) * VOXEL_VOLUME_MM3
)
size_cases["equivalent_spherical_diameter_mm"] = (
    6.0 * size_cases["lesion_volume_mm3"] / math.pi
) ** (1.0 / 3.0)
size_cases["lesion_size_group"] = size_cases[
    "equivalent_spherical_diameter_mm"
].map(size_category)
size_cases[
    [
        "cohort",
        "study_id",
        "patient_id",
        "annotation_type",
        "lesion_target_voxels_in_crop",
        "lesion_volume_mm3",
        "equivalent_spherical_diameter_mm",
        "lesion_size_group",
        "lesion_dice_at_0_5",
        "lesion_localized_at_locked_threshold",
        "locked_deployment_threshold",
    ]
].to_csv(OUTPUT_SIZE_CASE, index=False)

size_summary_rows = []
size_source_frames = [
    ("PANORAMA_INTERNAL_TEST", size_cases.loc[size_cases["cohort"] == "PANORAMA_INTERNAL_TEST"]),
    ("MSD_EXTERNAL_MIXED", size_cases.loc[size_cases["cohort"] == "MSD_EXTERNAL_MIXED"]),
    ("INTERNAL_PLUS_MSD", size_cases),
]
ordered_groups = ["small_<=20mm", "medium_>20-40mm", "large_>40mm"]
for cohort_index, (cohort, frame) in enumerate(size_source_frames):
    for group_index, group_name in enumerate(ordered_groups):
        group = frame.loc[frame["lesion_size_group"] == group_name].copy()
        lesion_point, lesion_low, lesion_high = mean_ci(
            group["lesion_dice_at_0_5"].astype(float).to_numpy(),
            BOOTSTRAP_SEED + 1000 + 10 * cohort_index + group_index,
        )
        sensitivity_point, sensitivity_low, sensitivity_high = proportion_ci(
            truth_flags(group["lesion_localized_at_locked_threshold"]).astype(float).to_numpy(),
            BOOTSTRAP_SEED + 2000 + 10 * cohort_index + group_index,
        )
        size_summary_rows.append(
            {
                "cohort": cohort,
                "analysis_role": "post_hoc_lesion_size_stratification",
                "size_definition": "equivalent_spherical_diameter_from_resampled_reference_volume",
                "lesion_size_group": group_name,
                "PDAC_cases": int(len(group)),
                "manual_cases": int((group["annotation_type_normalized"] == "manual").sum()),
                "automatic_cases": int((group["annotation_type_normalized"] == "automatic").sum()),
                "minimum_equivalent_diameter_mm": (
                    float(group["equivalent_spherical_diameter_mm"].min()) if len(group) else np.nan
                ),
                "median_equivalent_diameter_mm": (
                    float(group["equivalent_spherical_diameter_mm"].median()) if len(group) else np.nan
                ),
                "maximum_equivalent_diameter_mm": (
                    float(group["equivalent_spherical_diameter_mm"].max()) if len(group) else np.nan
                ),
                "mean_lesion_Dice": lesion_point,
                "mean_lesion_Dice_95CI_lower": lesion_low,
                "mean_lesion_Dice_95CI_upper": lesion_high,
                "lesion_localization_sensitivity_locked_threshold": sensitivity_point,
                "lesion_localization_sensitivity_95CI_lower": sensitivity_low,
                "lesion_localization_sensitivity_95CI_upper": sensitivity_high,
                "locked_threshold": locked_threshold,
            }
        )
size_summary = pd.DataFrame(size_summary_rows)
size_summary.to_csv(OUTPUT_SIZE_SUMMARY, index=False)


# =============================================================================
# SUMMARY AND AUDIT
# =============================================================================

msd_ci_map = msd_ci.set_index("metric")
summary_lines = [
    "# Stage R2A reviewer-requested analyses",
    "",
    "All analyses were performed after prediction freeze, without model training, model inference, or threshold adjustment.",
    "",
    "## Complete MSD 95% confidence intervals",
    "",
]
for metric in [
    "case_level_AUC",
    "case_level_average_precision",
    "mean_pancreas_Dice",
    "mean_PDAC_lesion_Dice",
    "PDAC_localization_sensitivity_locked_threshold",
    "negative_specificity_locked_threshold",
    "false_positive_candidates_per_case_locked_threshold",
]:
    row = msd_ci_map.loc[metric]
    summary_lines.append(
        f"- {metric}: "
        + format_estimate(
            float(row["point_estimate"]),
            float(row["bootstrap_95CI_lower"]),
            float(row["bootstrap_95CI_upper"]),
        )
    )
summary_lines.extend(["", "## Manual-reference-only analysis", ""])
for _, row in manual_summary.iterrows():
    summary_lines.append(
        f"- {row['cohort']} (n={int(row['manual_PDAC_cases'])}): lesion Dice "
        + format_estimate(
            row["mean_lesion_Dice"],
            row["mean_lesion_Dice_95CI_lower"],
            row["mean_lesion_Dice_95CI_upper"],
        )
        + "; sensitivity "
        + format_estimate(
            row["lesion_localization_sensitivity_locked_threshold"],
            row["lesion_localization_sensitivity_95CI_lower"],
            row["lesion_localization_sensitivity_95CI_upper"],
        )
    )
summary_lines.extend(["", "## Lesion-size-stratified analysis", ""])
pooled_size = size_summary.loc[size_summary["cohort"] == "INTERNAL_PLUS_MSD"]
for _, row in pooled_size.iterrows():
    summary_lines.append(
        f"- {row['lesion_size_group']} (n={int(row['PDAC_cases'])}): lesion Dice "
        + format_estimate(
            row["mean_lesion_Dice"],
            row["mean_lesion_Dice_95CI_lower"],
            row["mean_lesion_Dice_95CI_upper"],
        )
        + "; sensitivity "
        + format_estimate(
            row["lesion_localization_sensitivity_locked_threshold"],
            row["lesion_localization_sensitivity_95CI_lower"],
            row["lesion_localization_sensitivity_95CI_upper"],
        )
    )
OUTPUT_SUMMARY.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

input_hashes = {str(path): sha256(path) for path in required_paths}
readiness = {
    "Frozen internal evaluation audit passed": internal_audit.get("all_checks_pass") is True,
    "Frozen external evaluation audit passed": external_audit.get("all_checks_pass") is True,
    "Exactly 293 internal cases": len(internal) == EXPECTED_INTERNAL,
    "Exactly 194 MSD cases": len(msd) == EXPECTED_MSD,
    "MSD contains 98 PDAC and 96 non-PDAC cases": (
        msd["label_binary"].astype(int).value_counts().to_dict()
        == {1: EXPECTED_MSD_PDAC, 0: EXPECTED_MSD_NON_PDAC}
    ),
    "Seven complete MSD confidence intervals": len(msd_ci) == 7,
    "MSD FP/case matches frozen point estimate": np.isclose(
        fp_point,
        float(external_audit["MSD_locked_threshold_false_positives_per_case"]),
        rtol=0,
        atol=1e-12,
    ),
    "Internal and MSD FROC contain six target points each": (
        froc_long.groupby("cohort").size().to_dict()
        == {"MSD_EXTERNAL_MIXED": 6, "PANORAMA_INTERNAL_TEST": 6}
    ),
    "Manual-reference analysis is non-empty": bool(
        (manual_summary["manual_PDAC_cases"] > 0).all()
    ),
    "All 185 held-out PDAC cases entered lesion-size analysis": len(size_cases) == 185,
    "All lesion sizes are positive and finite": bool(
        np.isfinite(size_cases["equivalent_spherical_diameter_mm"]).all()
        and (size_cases["equivalent_spherical_diameter_mm"] > 0).all()
    ),
    "Locked threshold remains unchanged": np.isclose(locked_threshold, 0.462060, atol=5e-7),
    "No model training occurred": True,
    "No model inference occurred": True,
    "No threshold selection or adjustment occurred": True,
}
all_checks_pass = bool(all(readiness.values()))
audit = {
    "stage": "R2A",
    "created_at_utc": utc_now(),
    "result": "PASS_POSTFREEZE_REVIEWER_ANALYSES_COMPLETE" if all_checks_pass else "FAIL",
    "all_checks_pass": all_checks_pass,
    "analysis_role": "post_hoc_reviewer_requested_secondary_analysis",
    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    "bootstrap_seed": BOOTSTRAP_SEED,
    "locked_threshold": locked_threshold,
    "training_performed": False,
    "model_inference_performed": False,
    "threshold_selection_or_adjustment_performed": False,
    "raw_CT_volumes_accessed": 0,
    "segmentation_mask_arrays_accessed": 0,
    "input_sha256": input_hashes,
    "readiness_checks": readiness,
    "outputs": [
        str(OUTPUT_MSD_CI),
        str(OUTPUT_FROC),
        str(OUTPUT_MANUAL),
        str(OUTPUT_SIZE_CASE),
        str(OUTPUT_SIZE_SUMMARY),
        str(OUTPUT_FIGURE_PNG),
        str(OUTPUT_FIGURE_PDF),
        str(OUTPUT_SUMMARY),
    ],
}
for audit_path in [OUTPUT_AUDIT, OUTPUT_AUDIT_COPY]:
    with open(audit_path, "w", encoding="utf-8") as file:
        json.dump(json_safe(audit), file, indent=2, allow_nan=False)

zip_path = shutil.make_archive(
    str(OUTPUT_ZIP_BASE), "zip", root_dir=RESULT_DIR
)

print("\nMSD COMPLETE CONFIDENCE INTERVALS")
print("-" * 120)
print(msd_ci.to_string(index=False))
print("\nMANUAL-REFERENCE-ONLY ANALYSIS")
print("-" * 120)
print(manual_summary.to_string(index=False))
print("\nLESION-SIZE-STRATIFIED ANALYSIS")
print("-" * 120)
print(size_summary.to_string(index=False))
print("\nREADINESS CHECKS")
print("-" * 120)
for name, passed in readiness.items():
    print(f"  {name}: {passed}")
print(f"\nResult folder:\n{RESULT_DIR}")
print(f"Audit:\n{OUTPUT_AUDIT}")
print(f"Upload this ZIP back to ChatGPT:\n{zip_path}")
print("=" * 120)
print(
    "STAGE R2A RESULT: "
    + ("PASS — POST-FREEZE REVIEWER ANALYSES COMPLETE" if all_checks_pass else "FAIL")
)
print("=" * 120)
if not all_checks_pass:
    raise RuntimeError("Stage R2A failed one or more readiness checks.")
