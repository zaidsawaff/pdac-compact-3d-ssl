from pathlib import Path
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
import random
import shutil
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score


# =============================================================================
# PATHS AND LOCKS
# =============================================================================

PROJECT_ROOT = Path("/content/drive/MyDrive/PDAC_Public_Q1_Project")
META_DIR = PROJECT_ROOT / "01_Metadata"
QC_DIR = PROJECT_ROOT / "02_Quality_Control"
MODEL_ROOT = (
    PROJECT_ROOT
    / "04_Models"
    / "Supervised"
    / "Stage5_LightweightLesionSegmentation"
)

FROZEN_MANIFEST_PATH = META_DIR / "stage3b_e5_frozen_dataset_manifest.csv"
STAGE5A_PROTOCOL_PATH = META_DIR / "stage5a_supervised_training_protocol.json"
STAGE5B_AUDIT_PATH = QC_DIR / "stage5b_dual_arm_training_audit.json"
LOCAL_CACHE_DIR = Path("/content/pdac_e5_ssl_cache")

SUMMARY_PATH = QC_DIR / "stage5c_full_validation_arm_summary.csv"
PAIRED_COMPARISON_PATH = QC_DIR / "stage5c_paired_arm_comparison.csv"
SELECTION_PATH = META_DIR / "stage5c_supervised_model_selection.json"
AUDIT_PATH = QC_DIR / "stage5c_full_validation_comparison_audit.json"

ARMS = ["SSL_INITIALIZED", "RANDOM_INITIALIZED"]
SEED = 20260728
EXPECTED_CASES = 1671
EXPECTED_VALIDATION = 295
EXPECTED_VALIDATION_PDAC = 88
EXPECTED_VALIDATION_NON_PDAC = 207
VOLUME_SHAPE = np.asarray([240, 192, 128], dtype=int)
PATCH_SHAPE = np.asarray([128, 128, 64], dtype=int)
SLIDING_OVERLAP = 0.50
PROBABILITY_THRESHOLD = 0.50
HU_CENTER = 50.0
HU_HALF_WIDTH = 250.0
BATCH_SIZE = 4
BOOTSTRAP_REPLICATES = 2000
USE_AMP = True


# =============================================================================
# UTILITIES
# =============================================================================


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_write_csv(dataframe, path):
    temporary = Path(str(path) + ".tmp")
    dataframe.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_write_json(data, path):
    temporary = Path(str(path) + ".tmp")
    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def sha256_file(path, chunk_size=8 * 1024 * 1024):
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
        .isin({"true", "1", "yes"})
    )


def disk_free_gib(path):
    return shutil.disk_usage(path).free / (1024 ** 3)


def group_count(channels):
    for groups in [8, 4, 2, 1]:
        if channels % groups == 0:
            return groups
    return 1


def dice_from_binary(prediction, target, empty_value=1.0):
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    denominator = int(prediction.sum()) + int(target.sum())
    if denominator == 0:
        return float(empty_value)
    intersection = int(np.logical_and(prediction, target).sum())
    return float(2.0 * intersection / denominator)


def make_starts(volume_size, patch_size, overlap):
    stride = max(1, int(round(patch_size * (1.0 - overlap))))
    if volume_size <= patch_size:
        return [0]
    starts = list(range(0, volume_size - patch_size + 1, stride))
    final_start = volume_size - patch_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


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


# =============================================================================
# MODEL — IDENTICAL TO STAGE 5B
# =============================================================================


class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(group_count(out_channels), out_channels),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv3d(in_channels, out_channels, 1, bias=False)
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.activation(self.main(x) + self.skip(x))


class DownBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv3d(
                in_channels, out_channels, 3, stride=2, padding=1, bias=False
            ),
            nn.GroupNorm(group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )
        self.residual = ResidualBlock3D(out_channels, out_channels)

    def forward(self, x):
        return self.residual(self.down(x))


class CompactEncoder3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(1, 16, 3, padding=1, bias=False),
            nn.GroupNorm(group_count(16), 16),
            nn.SiLU(inplace=True),
            ResidualBlock3D(16, 16),
        )
        self.down1 = DownBlock3D(16, 24)
        self.down2 = DownBlock3D(24, 48)
        self.down3 = DownBlock3D(48, 96)
        self.bottleneck = ResidualBlock3D(96, 96)

    def forward(self, x):
        skip0 = self.stem(x)
        skip1 = self.down1(skip0)
        skip2 = self.down2(skip1)
        bottleneck = self.bottleneck(self.down3(skip2))
        return skip0, skip1, skip2, bottleneck


class UpBlock3D(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, 2, stride=2)
        self.fuse = ResidualBlock3D(
            out_channels + skip_channels, out_channels
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            raise RuntimeError("Decoder and skip shapes differ.")
        return self.fuse(torch.cat([x, skip], dim=1))


class LightweightDualHead3DSegmenter(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = CompactEncoder3D()
        self.up2 = UpBlock3D(96, 48, 48)
        self.up1 = UpBlock3D(48, 24, 24)
        self.up0 = UpBlock3D(24, 16, 16)
        self.segmentation_head = nn.Conv3d(16, 2, 1)

    def forward(self, x):
        skip0, skip1, skip2, bottleneck = self.encoder(x)
        x = self.up2(bottleneck, skip2)
        x = self.up1(x, skip1)
        x = self.up0(x, skip0)
        return self.segmentation_head(x)


# =============================================================================
# INPUTS, CACHE, AND INFERENCE GRID
# =============================================================================


print("=" * 120)
print("STAGE 5C — RESUMABLE FULL-VALIDATION SSL-VERSUS-RANDOM COMPARISON")
print("=" * 120)

for required in [
    FROZEN_MANIFEST_PATH,
    STAGE5A_PROTOCOL_PATH,
    STAGE5B_AUDIT_PATH,
]:
    if not required.exists():
        raise FileNotFoundError(f"Required input missing:\n{required}")

best_paths = {
    arm: MODEL_ROOT / arm / "stage5b_best.pt" for arm in ARMS
}
arm_audit_paths = {
    arm: QC_DIR / f"stage5b_{arm.lower()}_training_audit.json"
    for arm in ARMS
}
for arm in ARMS:
    if not best_paths[arm].exists():
        raise FileNotFoundError(f"{arm} best checkpoint missing: {best_paths[arm]}")
    if not arm_audit_paths[arm].exists():
        raise FileNotFoundError(f"{arm} training audit missing.")
    with open(arm_audit_paths[arm], "r", encoding="utf-8") as file:
        arm_audit = json.load(file)
    if arm_audit.get("training_complete") is not True:
        raise RuntimeError(f"{arm} training is not complete.")

with open(STAGE5A_PROTOCOL_PATH, "r", encoding="utf-8") as file:
    stage5a_protocol = json.load(file)
with open(STAGE5B_AUDIT_PATH, "r", encoding="utf-8") as file:
    stage5b_audit = json.load(file)
if stage5a_protocol.get("protocol_locked") is not True:
    raise RuntimeError("Stage 5A protocol is not locked.")
if stage5b_audit.get("training_complete") is not True:
    raise RuntimeError("Stage 5B dual-arm training did not complete.")

if not torch.cuda.is_available():
    raise RuntimeError("Stage 5C requires a CUDA GPU.")

manifest = pd.read_csv(FROZEN_MANIFEST_PATH, dtype={"study_id": str})
manifest["study_id"] = manifest["study_id"].astype(str).str.strip()
manifest["patient_id"] = manifest["patient_id"].astype(str).str.strip()
manifest["partition"] = manifest["partition"].astype(str).str.strip().str.lower()
manifest["diagnostic_label_normalized"] = (
    manifest["diagnostic_label"].astype(str).str.strip().str.lower()
)
manifest["annotation_type_normalized"] = (
    manifest["annotation_type"].astype(str).str.strip().str.lower()
)

required_columns = {
    "study_id", "patient_id", "partition", "diagnostic_label",
    "annotation_type", "output_path", "output_size_bytes",
    "verified_sha256", "all_checks_pass",
}
missing = sorted(required_columns - set(manifest.columns))
if missing:
    raise RuntimeError(f"Frozen manifest missing columns: {missing}")
if len(manifest) != EXPECTED_CASES or manifest["study_id"].nunique() != EXPECTED_CASES:
    raise RuntimeError("Frozen development cohort changed.")
if not truth_flags(manifest["all_checks_pass"]).all():
    raise RuntimeError("Frozen manifest contains a failed case.")
if set(manifest["partition"]) != {"train", "validation"}:
    raise RuntimeError("A locked test case entered Stage 5C.")

validation = (
    manifest[manifest["partition"] == "validation"]
    .sort_values("study_id")
    .reset_index(drop=True)
)
if len(validation) != EXPECTED_VALIDATION:
    raise RuntimeError("Validation count changed.")
if (validation["diagnostic_label_normalized"] == "pdac").sum() != EXPECTED_VALIDATION_PDAC:
    raise RuntimeError("Validation PDAC count changed.")
if (validation["diagnostic_label_normalized"] == "non-pdac").sum() != EXPECTED_VALIDATION_NON_PDAC:
    raise RuntimeError("Validation non-PDAC count changed.")

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda")
print(f"PyTorch: {torch.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Validation cases: {len(validation)}")
print("Locked test cases accessed: 0")

LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
additional_bytes = 0
for _, row in manifest.iterrows():
    destination = LOCAL_CACHE_DIR / f"{row['study_id']}.npz"
    expected_size = int(row["output_size_bytes"])
    if not (destination.exists() and destination.stat().st_size == expected_size):
        additional_bytes += expected_size
if disk_free_gib("/content") * (1024 ** 3) < additional_bytes * 1.10 + 1024 ** 3:
    raise RuntimeError("Insufficient local storage for the E5 cache.")

print("\nPreparing or verifying the local E5 cache...")
copied = 0
reused = 0
for order, (_, row) in enumerate(manifest.iterrows(), start=1):
    study_id = str(row["study_id"])
    source = Path(str(row["output_path"]))
    destination = LOCAL_CACHE_DIR / f"{study_id}.npz"
    expected_size = int(row["output_size_bytes"])
    expected_hash = str(row["verified_sha256"])
    valid = False
    if destination.exists() and destination.stat().st_size == expected_size:
        valid = sha256_file(destination) == expected_hash
    if valid:
        reused += 1
    else:
        if not source.exists():
            raise FileNotFoundError(f"Frozen file missing: {source}")
        temporary = Path(str(destination) + ".part")
        if temporary.exists():
            temporary.unlink()
        shutil.copy2(source, temporary)
        if temporary.stat().st_size != expected_size:
            temporary.unlink()
            raise RuntimeError(f"Cache size mismatch: {study_id}")
        if sha256_file(temporary) != expected_hash:
            temporary.unlink()
            raise RuntimeError(f"Cache hash mismatch: {study_id}")
        os.replace(temporary, destination)
        copied += 1
    if order % 200 == 0 or order == len(manifest):
        print(f"  Cache {order}/{len(manifest)} — reused={reused}, copied={copied}")

starts_by_axis = [
    make_starts(int(v), int(p), SLIDING_OVERLAP)
    for v, p in zip(VOLUME_SHAPE, PATCH_SHAPE)
]
window_starts = [
    (x, y, z)
    for x in starts_by_axis[0]
    for y in starts_by_axis[1]
    for z in starts_by_axis[2]
]


@torch.no_grad()
def infer_full_crop(model, ct_hu):
    normalized = np.clip(
        (ct_hu.astype(np.float32) - HU_CENTER) / HU_HALF_WIDTH,
        -1.0,
        1.0,
    )
    probability_sum = np.zeros((2, *VOLUME_SHAPE), dtype=np.float32)
    count = np.zeros(tuple(VOLUME_SHAPE), dtype=np.float32)
    for offset in range(0, len(window_starts), BATCH_SIZE):
        chunk = window_starts[offset:offset + BATCH_SIZE]
        patches = []
        for start in chunk:
            end = np.asarray(start) + PATCH_SHAPE
            patch = normalized[
                start[0]:end[0], start[1]:end[1], start[2]:end[2]
            ]
            patches.append(np.ascontiguousarray(patch, dtype=np.float32))
        batch = torch.from_numpy(np.stack(patches)).unsqueeze(1).to(
            device, non_blocking=True
        )
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=USE_AMP):
            logits = model(batch)
            probabilities = torch.sigmoid(logits).float().cpu().numpy()
        for local_index, start in enumerate(chunk):
            end = np.asarray(start) + PATCH_SHAPE
            slices = (
                slice(start[0], int(end[0])),
                slice(start[1], int(end[1])),
                slice(start[2], int(end[2])),
            )
            probability_sum[(slice(None),) + slices] += probabilities[local_index]
            count[slices] += 1.0
        del batch, logits, probabilities
    if np.any(count <= 0):
        raise RuntimeError("Sliding-window inference left uncovered voxels.")
    return probability_sum / count[None, ...]


# =============================================================================
# RESUMABLE FULL-VALIDATION INFERENCE
# =============================================================================


def evaluate_arm(arm):
    checkpoint_path = best_paths[arm]
    checkpoint_hash = sha256_file(checkpoint_path)
    ledger_path = QC_DIR / f"stage5c_{arm.lower()}_full_validation_ledger.csv"

    if ledger_path.exists():
        ledger = pd.read_csv(ledger_path, dtype={"study_id": str, "patient_id": str})
        if "checkpoint_sha256" not in ledger.columns:
            ledger = pd.DataFrame()
        else:
            ledger = ledger[
                ledger["checkpoint_sha256"].astype(str) == checkpoint_hash
            ].copy()
            ledger = ledger.drop_duplicates("study_id", keep="last")
    else:
        ledger = pd.DataFrame()

    completed_ids = set(ledger.get("study_id", pd.Series(dtype=str)).astype(str))
    pending = validation[~validation["study_id"].isin(completed_ids)]

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("arm") != arm:
        raise RuntimeError(f"{arm}: checkpoint identity mismatch.")
    model = LightweightDualHead3DSegmenter().to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()

    print(f"\n{arm}")
    print("-" * 120)
    print(f"Best epoch: {int(checkpoint['best_epoch'])}")
    print(f"Previously completed: {len(completed_ids)}/{len(validation)}")
    print(f"Pending: {len(pending)}")

    for order, (_, row) in enumerate(pending.iterrows(), start=1):
        study_id = str(row["study_id"])
        started = time.time()
        with np.load(LOCAL_CACHE_DIR / f"{study_id}.npz", allow_pickle=False) as data:
            ct_hu = np.asarray(data["ct_hu"], dtype=np.int16)
            mask = np.asarray(data["mask_labels"], dtype=np.uint8)
        if tuple(ct_hu.shape) != tuple(VOLUME_SHAPE) or mask.shape != ct_hu.shape:
            raise RuntimeError(f"{study_id}: invalid frozen E5 geometry.")

        probabilities = infer_full_crop(model, ct_hu)
        finite = bool(np.isfinite(probabilities).all())
        pancreas_prediction = probabilities[0] >= PROBABILITY_THRESHOLD
        lesion_prediction = probabilities[1] >= PROBABILITY_THRESHOLD
        pancreas_target = mask == 4
        lesion_target = mask == 1
        is_pdac = row["diagnostic_label_normalized"] == "pdac"

        pancreas_dice = dice_from_binary(pancreas_prediction, pancreas_target)
        lesion_dice = (
            dice_from_binary(lesion_prediction, lesion_target, empty_value=0.0)
            if is_pdac else np.nan
        )
        target_scores = probabilities[1][lesion_target]
        result_row = {
            "arm": arm,
            "checkpoint_sha256": checkpoint_hash,
            "best_epoch": int(checkpoint["best_epoch"]),
            "study_id": study_id,
            "patient_id": str(row["patient_id"]),
            "diagnostic_label": row["diagnostic_label_normalized"],
            "label_binary": int(is_pdac),
            "annotation_type": row["annotation_type_normalized"],
            "pancreas_dice_at_0_5": pancreas_dice,
            "lesion_dice_at_0_5": lesion_dice,
            "lesion_detection_overlap_at_0_5": bool(
                is_pdac and np.logical_and(lesion_prediction, lesion_target).any()
            ),
            "negative_case_specific_at_0_5": bool(
                (not is_pdac) and not lesion_prediction.any()
            ),
            "patient_score_max_lesion_probability": float(probabilities[1].max()),
            "target_lesion_max_probability": (
                float(target_scores.max()) if len(target_scores) else np.nan
            ),
            "target_lesion_mean_probability": (
                float(target_scores.mean()) if len(target_scores) else np.nan
            ),
            "predicted_lesion_voxels_at_0_5": int(lesion_prediction.sum()),
            "target_lesion_voxels": int(lesion_target.sum()),
            "probabilities_finite": finite,
            "inference_seconds": float(time.time() - started),
            "processing_complete": finite,
            "completed_at_utc": utc_now(),
        }
        ledger = pd.concat([ledger, pd.DataFrame([result_row])], ignore_index=True)
        ledger = ledger.drop_duplicates("study_id", keep="last").sort_values("study_id")
        atomic_write_csv(ledger, ledger_path)
        del ct_hu, mask, probabilities
        if order % 10 == 0 or order == len(pending):
            print(
                f"  Completed this run {order}/{len(pending)} — "
                f"durable total {len(ledger)}/{len(validation)}"
            )

    del model, checkpoint
    gc.collect()
    torch.cuda.empty_cache()
    if len(ledger) != EXPECTED_VALIDATION or ledger["study_id"].nunique() != EXPECTED_VALIDATION:
        raise RuntimeError(f"{arm}: full validation ledger is incomplete.")
    if set(ledger["study_id"].astype(str)) != set(validation["study_id"]):
        raise RuntimeError(f"{arm}: validation IDs do not match the lock.")
    if not truth_flags(ledger["processing_complete"]).all():
        raise RuntimeError(f"{arm}: one or more inference rows failed.")
    return ledger.sort_values("study_id").reset_index(drop=True), ledger_path


arm_ledgers = {}
arm_ledger_paths = {}
for arm in ARMS:
    arm_ledgers[arm], arm_ledger_paths[arm] = evaluate_arm(arm)


# =============================================================================
# ARM SUMMARIES AND PATIENT-GROUPED PAIRED BOOTSTRAP
# =============================================================================


def summarize_arm(arm, ledger):
    pdac = ledger[ledger["label_binary"] == 1]
    negative = ledger[ledger["label_binary"] == 0]
    pancreas_dice = float(ledger["pancreas_dice_at_0_5"].mean())
    lesion_dice = float(pdac["lesion_dice_at_0_5"].mean())
    selection_score = 0.70 * lesion_dice + 0.30 * pancreas_dice
    return {
        "arm": arm,
        "cases": int(len(ledger)),
        "PDAC_cases": int(len(pdac)),
        "non_PDAC_cases": int(len(negative)),
        "best_epoch": int(ledger["best_epoch"].iloc[0]),
        "mean_pancreas_dice_at_0_5": pancreas_dice,
        "mean_PDAC_lesion_dice_at_0_5": lesion_dice,
        "median_PDAC_lesion_dice_at_0_5": float(pdac["lesion_dice_at_0_5"].median()),
        "PDAC_lesion_detection_sensitivity_at_0_5": float(
            truth_flags(pdac["lesion_detection_overlap_at_0_5"]).mean()
        ),
        "non_PDAC_case_specificity_at_0_5": float(
            truth_flags(negative["negative_case_specific_at_0_5"]).mean()
        ),
        "patient_level_AUC_max_probability": safe_auc(
            ledger["label_binary"], ledger["patient_score_max_lesion_probability"]
        ),
        "patient_level_average_precision": safe_ap(
            ledger["label_binary"], ledger["patient_score_max_lesion_probability"]
        ),
        "selection_score": selection_score,
        "mean_inference_seconds_per_case": float(ledger["inference_seconds"].mean()),
        "checkpoint_sha256": str(ledger["checkpoint_sha256"].iloc[0]),
    }


summary = pd.DataFrame(
    [summarize_arm(arm, arm_ledgers[arm]) for arm in ARMS]
).sort_values("selection_score", ascending=False).reset_index(drop=True)
atomic_write_csv(summary, SUMMARY_PATH)

ssl = arm_ledgers["SSL_INITIALIZED"].copy()
random_arm = arm_ledgers["RANDOM_INITIALIZED"].copy()
paired = ssl.merge(
    random_arm,
    on=["study_id", "patient_id", "diagnostic_label", "label_binary", "annotation_type"],
    suffixes=("_ssl", "_random"),
    validate="one_to_one",
)
paired["pancreas_dice_difference_ssl_minus_random"] = (
    paired["pancreas_dice_at_0_5_ssl"] - paired["pancreas_dice_at_0_5_random"]
)
paired["lesion_dice_difference_ssl_minus_random"] = (
    paired["lesion_dice_at_0_5_ssl"] - paired["lesion_dice_at_0_5_random"]
)
atomic_write_csv(paired, PAIRED_COMPARISON_PATH)


def selection_score_from_frame(frame, suffix):
    pancreas = float(frame[f"pancreas_dice_at_0_5_{suffix}"].mean())
    lesion = float(
        frame.loc[frame["label_binary"] == 1, f"lesion_dice_at_0_5_{suffix}"].mean()
    )
    return 0.70 * lesion + 0.30 * pancreas


rng = np.random.RandomState(SEED)
unique_patients = np.asarray(sorted(paired["patient_id"].unique()), dtype=object)
bootstrap_score_differences = []
bootstrap_auc_differences = []
for _ in range(BOOTSTRAP_REPLICATES):
    sampled_patients = rng.choice(
        unique_patients, size=len(unique_patients), replace=True
    )
    pieces = [paired[paired["patient_id"] == patient] for patient in sampled_patients]
    sample = pd.concat(pieces, ignore_index=True)
    ssl_score = selection_score_from_frame(sample, "ssl")
    random_score = selection_score_from_frame(sample, "random")
    bootstrap_score_differences.append(ssl_score - random_score)
    ssl_auc = safe_auc(
        sample["label_binary"], sample["patient_score_max_lesion_probability_ssl"]
    )
    random_auc = safe_auc(
        sample["label_binary"], sample["patient_score_max_lesion_probability_random"]
    )
    if np.isfinite(ssl_auc) and np.isfinite(random_auc):
        bootstrap_auc_differences.append(ssl_auc - random_auc)

score_difference = float(
    summary.loc[summary["arm"] == "SSL_INITIALIZED", "selection_score"].iloc[0]
    - summary.loc[summary["arm"] == "RANDOM_INITIALIZED", "selection_score"].iloc[0]
)
score_ci = np.percentile(bootstrap_score_differences, [2.5, 97.5])
auc_difference = float(
    summary.loc[summary["arm"] == "SSL_INITIALIZED", "patient_level_AUC_max_probability"].iloc[0]
    - summary.loc[summary["arm"] == "RANDOM_INITIALIZED", "patient_level_AUC_max_probability"].iloc[0]
)
auc_ci = np.percentile(bootstrap_auc_differences, [2.5, 97.5])

selected_arm = str(summary.iloc[0]["arm"])
selected_checkpoint = best_paths[selected_arm]
selection = {
    "stage": "5C",
    "created_at_utc": utc_now(),
    "selected_arm": selected_arm,
    "selected_checkpoint_path": str(selected_checkpoint),
    "selected_checkpoint_sha256": sha256_file(selected_checkpoint),
    "selection_metric": "0.70*mean_PDAC_lesion_Dice + 0.30*mean_pancreas_Dice",
    "selection_score": float(summary.iloc[0]["selection_score"]),
    "SSL_minus_random_selection_score_difference": score_difference,
    "SSL_minus_random_selection_score_difference_patient_bootstrap_95_CI": [
        float(score_ci[0]), float(score_ci[1])
    ],
    "SSL_minus_random_AUC_difference": auc_difference,
    "SSL_minus_random_AUC_difference_patient_bootstrap_95_CI": [
        float(auc_ci[0]), float(auc_ci[1])
    ],
    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    "full_validation_cases": EXPECTED_VALIDATION,
    "locked_test_cases_accessed": 0,
    "next_stage": "validation_only_component_and_FROC_calibration",
}
atomic_write_json(selection, SELECTION_PATH)

readiness_checks = {
    "Both Stage 5B arms completed": stage5b_audit.get("training_complete") is True,
    "Exactly 295 validation cases per arm": all(
        len(arm_ledgers[arm]) == EXPECTED_VALIDATION for arm in ARMS
    ),
    "Exactly 295 unique validation IDs per arm": all(
        arm_ledgers[arm]["study_id"].nunique() == EXPECTED_VALIDATION for arm in ARMS
    ),
    "Validation IDs exactly match the frozen lock": all(
        set(arm_ledgers[arm]["study_id"]) == set(validation["study_id"]) for arm in ARMS
    ),
    "All probability outputs were finite": all(
        truth_flags(arm_ledgers[arm]["probabilities_finite"]).all() for arm in ARMS
    ),
    "Paired comparison contains 295 cases": len(paired) == EXPECTED_VALIDATION,
    "Both patient-level AUC values are finite": bool(
        np.isfinite(summary["patient_level_AUC_max_probability"]).all()
    ),
    "Both selection scores are finite": bool(np.isfinite(summary["selection_score"]).all()),
    "One best arm was selected": selected_arm in ARMS,
    "No locked test case was accessed": set(manifest["partition"]) == {"train", "validation"},
}
all_checks_pass = all(bool(value) for value in readiness_checks.values())

audit = {
    "stage": "5C",
    "created_at_utc": utc_now(),
    "result": "PASS_FULL_VALIDATION_MODEL_SELECTED" if all_checks_pass else "FAIL",
    "all_checks_pass": all_checks_pass,
    "readiness_checks": readiness_checks,
    "selected_arm": selected_arm,
    "arm_summaries": summary.to_dict(orient="records"),
    "selection_score_difference_SSL_minus_random": score_difference,
    "selection_score_difference_95_CI": [float(score_ci[0]), float(score_ci[1])],
    "AUC_difference_SSL_minus_random": auc_difference,
    "AUC_difference_95_CI": [float(auc_ci[0]), float(auc_ci[1])],
    "arm_ledger_paths": {arm: str(path) for arm, path in arm_ledger_paths.items()},
    "summary_path": str(SUMMARY_PATH),
    "paired_comparison_path": str(PAIRED_COMPARISON_PATH),
    "selection_path": str(SELECTION_PATH),
    "locked_test_cases_accessed": 0,
}
atomic_write_json(audit, AUDIT_PATH)

print("\nFULL-VALIDATION ARM SUMMARY")
print("-" * 120)
print(summary.to_string(index=False))
print("\nPAIRED DIFFERENCES")
print("-" * 120)
print(
    f"SSL - random selection score: {score_difference:.6f} "
    f"(95% CI {score_ci[0]:.6f} to {score_ci[1]:.6f})"
)
print(
    f"SSL - random patient-level AUC: {auc_difference:.6f} "
    f"(95% CI {auc_ci[0]:.6f} to {auc_ci[1]:.6f})"
)
print(f"Selected arm: {selected_arm}")

print("\nREADINESS CHECKS")
print("-" * 120)
for name, passed in readiness_checks.items():
    print(f"  {name}: {bool(passed)}")

print("\nSelection record:")
print(SELECTION_PATH)
print("\nAudit:")
print(AUDIT_PATH)
print("\n" + "=" * 120)
print(
    "STAGE 5C RESULT: "
    + ("PASS — FULL VALIDATION COMPLETE AND MODEL SELECTED" if all_checks_pass else "FAIL")
)
print("=" * 120)

if not all_checks_pass:
    failed = [name for name, passed in readiness_checks.items() if not passed]
    raise RuntimeError(f"Stage 5C failed readiness checks: {failed}")
