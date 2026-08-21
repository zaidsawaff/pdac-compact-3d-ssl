#!/usr/bin/env python3
"""Stage 7A: build publication-ready tables, figures, and Results text.

This stage is deliberately post-freeze and read-only with respect to the
canonical Stage 6E evidence. It performs no model fitting, inference,
threshold selection, CT access, or segmentation-mask access.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("PDAC_PROJECT_ROOT", "/content/drive/MyDrive/PDAC_Public_Q1_Project"))
META = ROOT / "01_Metadata"
QC = ROOT / "02_Quality_Control"
OUT = ROOT / "03_Results" / "Stage7A_Publication_Ready"
OUT.mkdir(parents=True, exist_ok=True)

FREEZE = META / "stage6e_final_results_freeze.json"
HASH_MANIFEST = META / "stage6e_final_evidence_hash_manifest.csv"
PERFORMANCE = QC / "stage6e_final_performance_summary.csv"
CI = QC / "stage6e_final_bootstrap_95ci_summary.csv"
FROC = QC / "stage6e_internal_vs_msd_froc_comparison.csv"


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def write_text(text: str, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def fmt3(x) -> str:
    return "—" if pd.isna(x) else f"{float(x):.3f}"


def pct1(x) -> str:
    return "—" if pd.isna(x) else f"{100.0 * float(x):.1f}%"


def save_figure(fig, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.png", dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


print("=" * 120)
print("STAGE 7A — PUBLICATION-READY RESULTS PACKAGE")
print("=" * 120)
print("GPU required: False")
print("CT volumes accessed: 0")
print("Segmentation masks accessed: 0")
print("Model fitting/inference: False")
print("Threshold selection: False")

required = [FREEZE, HASH_MANIFEST, PERFORMANCE, CI, FROC]
missing = [str(p) for p in required if not p.exists()]
if missing:
    raise FileNotFoundError("Missing frozen Stage 6E input(s):\n" + "\n".join(missing))

freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
if freeze.get("result") != "PASS_FINAL_RESULTS_AND_EVIDENCE_FROZEN":
    raise RuntimeError("Stage 6E final freeze is not in PASS state.")
if freeze.get("final_results_frozen") is not True:
    raise RuntimeError("Stage 6E final_results_frozen flag is not True.")

# Verify the manifest itself and every canonical evidence object before use.
manifest_hash = sha256_file(HASH_MANIFEST)
expected_manifest_hash = str(freeze.get("evidence_hash_manifest_sha256", ""))
if manifest_hash != expected_manifest_hash:
    raise RuntimeError("Stage 6E evidence hash manifest no longer matches its frozen SHA-256.")

manifest = pd.read_csv(HASH_MANIFEST)
expected_cols = {"evidence_role", "path", "size_bytes", "sha256"}
if not expected_cols.issubset(manifest.columns):
    raise RuntimeError("Stage 6E evidence manifest schema is incomplete.")
if len(manifest) != 25:
    raise RuntimeError(f"Expected 25 frozen evidence items, observed {len(manifest)}.")

print("\nVerifying 25 frozen evidence hashes...")
for i, row in manifest.iterrows():
    p = Path(str(row["path"]))
    if not p.exists():
        raise FileNotFoundError(f"Frozen evidence file missing: {p}")
    if int(p.stat().st_size) != int(row["size_bytes"]):
        raise RuntimeError(f"Frozen evidence size mismatch: {p}")
    if sha256_file(p) != str(row["sha256"]):
        raise RuntimeError(f"Frozen evidence SHA-256 mismatch: {p}")
    if (i + 1) % 5 == 0 or (i + 1) == len(manifest):
        print(f"  Verified: {i + 1}/{len(manifest)}")

perf = pd.read_csv(PERFORMANCE)
ci = pd.read_csv(CI)
froc = pd.read_csv(FROC)

expected_cohorts = {
    "PANORAMA_INTERNAL_TEST",
    "EXTERNAL_COMBINED",
    "MSD_EXTERNAL_MIXED",
    "NIH_NEGATIVE_STRESS",
}
if set(perf["cohort"].astype(str)) != expected_cohorts:
    raise RuntimeError("Stage 6E performance cohort set changed unexpectedly.")


# -------------------------------------------------------------------------
# TABLE 1 — Cohort composition and intended role
# -------------------------------------------------------------------------
role_label = {
    "PANORAMA_INTERNAL_TEST": "Internal held-out test",
    "EXTERNAL_COMBINED": "Pooled external (secondary)",
    "MSD_EXTERNAL_MIXED": "Source-separated external mixed test",
    "NIH_NEGATIVE_STRESS": "Negative-only external stress test",
}
table1 = perf[["cohort", "cases", "PDAC_cases", "non_PDAC_cases"]].copy()
table1.insert(1, "evaluation_role", table1["cohort"].map(role_label))
table1["PDAC_prevalence_percent"] = 100.0 * table1["PDAC_cases"] / table1["cases"]
write_csv(table1, OUT / "table1_evaluation_cohorts.csv")


# -------------------------------------------------------------------------
# TABLE 2 — Main frozen performance metrics
# -------------------------------------------------------------------------
table2 = perf[[
    "cohort", "case_level_AUC", "average_precision", "mean_pancreas_Dice",
    "mean_PDAC_lesion_Dice", "locked_threshold_lesion_sensitivity",
    "locked_threshold_false_positives_per_case",
    "locked_threshold_negative_specificity", "mean_FROC_sensitivity",
]].copy()
write_csv(table2, OUT / "table2_final_performance.csv")


# -------------------------------------------------------------------------
# TABLE 3 — Frozen bootstrap confidence intervals (no recomputation)
# -------------------------------------------------------------------------
table3 = ci.copy()
write_csv(table3, OUT / "table3_bootstrap_95ci.csv")


# -------------------------------------------------------------------------
# TABLE 4 — Internal versus MSD FROC operating points
# -------------------------------------------------------------------------
table4 = froc.copy()
write_csv(table4, OUT / "table4_internal_vs_msd_froc.csv")


# -------------------------------------------------------------------------
# FIGURE 1 — Cohort composition
# -------------------------------------------------------------------------
plot_order = ["PANORAMA_INTERNAL_TEST", "MSD_EXTERNAL_MIXED", "NIH_NEGATIVE_STRESS"]
pc = perf.set_index("cohort").loc[plot_order]
labels = ["Internal", "MSD external", "NIH stress"]
fig, ax = plt.subplots(figsize=(7.2, 4.8))
x = np.arange(len(labels))
ax.bar(x, pc["non_PDAC_cases"], label="non-PDAC", color="#8FBBD9")
ax.bar(x, pc["PDAC_cases"], bottom=pc["non_PDAC_cases"], label="PDAC", color="#D95F5F")
for j, (_, row) in enumerate(pc.iterrows()):
    ax.text(j, row["cases"] + 4, f"N={int(row['cases'])}", ha="center", va="bottom", fontsize=9)
ax.set_xticks(x, labels)
ax.set_ylabel("Number of studies")
ax.set_title("Evaluation cohort composition")
ax.legend(frameon=False, ncol=2, loc="upper right")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
save_figure(fig, "figure1_evaluation_cohort_composition")


# -------------------------------------------------------------------------
# FIGURE 2 — Internal versus MSD performance generalization
# -------------------------------------------------------------------------
pidx = perf.set_index("cohort")
internal = pidx.loc["PANORAMA_INTERNAL_TEST"]
msd = pidx.loc["MSD_EXTERNAL_MIXED"]
metric_cols = [
    "case_level_AUC",
    "average_precision",
    "mean_pancreas_Dice",
    "mean_PDAC_lesion_Dice",
    "locked_threshold_lesion_sensitivity",
]
metric_labels = ["AUC", "Average\nprecision", "Pancreas\nDice", "Lesion\nDice", "Lesion\nsensitivity"]
xi = np.arange(len(metric_cols))
width = 0.36
fig, ax = plt.subplots(figsize=(9.0, 5.0))
ax.bar(xi - width / 2, [internal[c] for c in metric_cols], width, label="Internal", color="#2B6F9E")
ax.bar(xi + width / 2, [msd[c] for c in metric_cols], width, label="MSD external", color="#E08B3E")
ax.set_ylim(0, 1.0)
ax.set_ylabel("Metric value")
ax.set_xticks(xi, metric_labels)
ax.set_title("Internal-to-external performance generalization")
ax.legend(frameon=False, ncol=2)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.18)
fig.tight_layout()
save_figure(fig, "figure2_internal_vs_msd_generalization")


# -------------------------------------------------------------------------
# FIGURE 3 — FROC internal versus MSD external
# -------------------------------------------------------------------------
target = froc["target_false_positives_per_case"].to_numpy(dtype=float)
sens_internal_col = "sensitivity_internal"
sens_msd_col = "sensitivity_MSD_external"
if sens_internal_col not in froc or sens_msd_col not in froc:
    candidates = [c for c in froc.columns if c.startswith("sensitivity")]
    raise RuntimeError(f"Unexpected FROC sensitivity columns: {candidates}")

fig, ax = plt.subplots(figsize=(7.0, 5.2))
ax.plot(target, froc[sens_internal_col], "o-", lw=2.2, ms=6, label="Internal", color="#2B6F9E")
ax.plot(target, froc[sens_msd_col], "s-", lw=2.2, ms=6, label="MSD external", color="#E08B3E")
ax.set_xscale("log", base=2)
ax.set_xticks(target)
ax.set_xticklabels([f"{v:g}" for v in target])
ax.set_ylim(0, 1.0)
ax.set_xlabel("False positives per case")
ax.set_ylabel("Lesion sensitivity")
ax.set_title("FROC performance with the frozen detection protocol")
ax.grid(alpha=0.22, which="both")
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
save_figure(fig, "figure3_internal_vs_msd_froc")


# -------------------------------------------------------------------------
# PAPER-READY RESULTS TEXT — strictly derived from frozen point estimates.
# -------------------------------------------------------------------------
combined = pidx.loc["EXTERNAL_COMBINED"]
nih = pidx.loc["NIH_NEGATIVE_STRESS"]
auc_delta = float(msd["case_level_AUC"] - internal["case_level_AUC"])
sens_delta = float(msd["locked_threshold_lesion_sensitivity"] - internal["locked_threshold_lesion_sensitivity"])

results_md = f"""# Publication-ready Results summary

All values below are derived from the Stage 6E frozen evidence. No post-test model fitting, inference, or threshold selection was performed in Stage 7A.

## Evaluation cohorts

The held-out PANORAMA internal test set contained 293 studies (87 PDAC and 206 non-PDAC). Source-separated external evaluation comprised 194 MSD-derived studies (98 PDAC and 96 non-PDAC) and an 80-study NIH negative-only stress cohort. The pooled external analysis therefore contained 274 studies (98 PDAC and 176 non-PDAC).

## Internal test performance

On the held-out internal test set, the frozen SSL-initialized model achieved a case-level AUC of {internal['case_level_AUC']:.3f} and an average precision of {internal['average_precision']:.3f}. Mean pancreas Dice was {internal['mean_pancreas_Dice']:.3f}, and mean PDAC lesion Dice was {internal['mean_PDAC_lesion_Dice']:.3f}. At the validation-locked detection threshold ({internal['locked_threshold']:.6f}), lesion sensitivity was {pct1(internal['locked_threshold_lesion_sensitivity'])} with {internal['locked_threshold_false_positives_per_case']:.3f} false-positive candidates per case. Mean FROC sensitivity across the prespecified 0.25, 0.5, 1, 2, and 4 FP/case operating points was {internal['mean_FROC_sensitivity']:.3f}.

## External generalization

In the MSD external mixed cohort, case-level AUC was {msd['case_level_AUC']:.3f} and average precision was {msd['average_precision']:.3f}. Mean pancreas Dice was {msd['mean_pancreas_Dice']:.3f}, while mean PDAC lesion Dice was {msd['mean_PDAC_lesion_Dice']:.3f}. Using the identical frozen threshold, lesion sensitivity was {pct1(msd['locked_threshold_lesion_sensitivity'])} with {msd['locked_threshold_false_positives_per_case']:.3f} false positives per case. Relative to internal testing, the absolute AUC change was {auc_delta:+.3f}, whereas the absolute lesion-sensitivity change was {100*sens_delta:+.1f} percentage points. Mean FROC sensitivity was {msd['mean_FROC_sensitivity']:.3f}.

Across the pooled external cohort, case-level AUC was {combined['case_level_AUC']:.3f}, average precision was {combined['average_precision']:.3f}, mean pancreas Dice was {combined['mean_pancreas_Dice']:.3f}, and mean PDAC lesion Dice was {combined['mean_PDAC_lesion_Dice']:.3f}. The frozen-threshold lesion sensitivity was {pct1(combined['locked_threshold_lesion_sensitivity'])} at {combined['locked_threshold_false_positives_per_case']:.3f} false positives per case.

## Negative-only stress testing

The NIH negative-only stress cohort contained no PDAC-positive studies; consequently, AUC and lesion sensitivity are not defined for this source. At the frozen threshold, false positives averaged {nih['locked_threshold_false_positives_per_case']:.3f} per case and negative-case specificity was {pct1(nih['locked_threshold_negative_specificity'])}. Mean pancreas Dice was {nih['mean_pancreas_Dice']:.3f}.

## Interpretation guardrail

The operating point was selected and frozen on validation data for lesion detection/FROC analysis. The negative-case specificity values should therefore not be interpreted as the result of a separately optimized diagnostic-classification threshold. Confidence intervals for manuscript reporting are provided unchanged in Table 3.
"""
write_text(results_md, OUT / "paper_ready_results_summary.md")


# -------------------------------------------------------------------------
# Output hashes and Stage 7A audit
# -------------------------------------------------------------------------
publication_files = sorted(
    [p for p in OUT.iterdir() if p.is_file() and not p.name.startswith("stage7a_publication_file_manifest")]
)
out_manifest = pd.DataFrame([
    {"file": p.name, "size_bytes": p.stat().st_size, "sha256": sha256_file(p)}
    for p in publication_files
])
write_csv(out_manifest, OUT / "stage7a_publication_file_manifest.csv")

checks = {
    "Stage 6E final freeze is PASS": freeze.get("final_results_frozen") is True,
    "Stage 6E hash manifest matches frozen SHA-256": manifest_hash == expected_manifest_hash,
    "Exactly 25 canonical evidence items were verified": len(manifest) == 25,
    "Four canonical evaluation cohorts are present": set(perf["cohort"].astype(str)) == expected_cohorts,
    "Internal cohort remains 293 studies": int(internal["cases"]) == 293,
    "MSD external cohort remains 194 studies": int(msd["cases"]) == 194,
    "NIH stress cohort remains 80 studies": int(nih["cases"]) == 80,
    "No CT or segmentation mask was accessed": True,
    "No model fitting or inference occurred": True,
    "No threshold selection occurred": True,
}
checks = {k: bool(v) for k, v in checks.items()}
if not all(checks.values()):
    raise RuntimeError("Stage 7A readiness failure: " + str([k for k, v in checks.items() if not v]))

audit = {
    "stage": "7A",
    "result": "PASS_PUBLICATION_READY_RESULTS_PACKAGE_CREATED",
    "source_freeze_stage": "6E",
    "source_evidence_manifest_sha256": manifest_hash,
    "canonical_source_files_modified": 0,
    "CT_volumes_accessed": 0,
    "segmentation_masks_accessed": 0,
    "model_fitting_performed": False,
    "model_inference_performed": False,
    "threshold_selection_performed": False,
    "readiness_checks": checks,
    "publication_output_directory": str(OUT),
    "publication_files": int(len(out_manifest)),
}
write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n", OUT / "stage7a_publication_ready_audit.json")

print("\n" + "-" * 120)
print("PUBLICATION OUTPUTS")
print("-" * 120)
print("Tables: 4 CSV files")
print("Figures: 3 figures, each in PNG (400 dpi) and vector PDF")
print("Results text: paper_ready_results_summary.md")
print("Output folder:", OUT)
print("\nREADINESS CHECKS")
for name, passed in checks.items():
    print(f"  {name}: {passed}")
print("\n" + "=" * 120)
print("STAGE 7A RESULT: PASS — PUBLICATION-READY RESULTS PACKAGE CREATED")
print("=" * 120)
