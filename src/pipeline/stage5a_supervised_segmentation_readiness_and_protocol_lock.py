from pathlib import Path
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# =============================================================================
# LOCKED INPUTS AND OUTPUTS
# =============================================================================

PROJECT_ROOT = Path(os.environ.get("PDAC_PROJECT_ROOT", "/content/drive/MyDrive/PDAC_Public_Q1_Project"))
META_DIR = PROJECT_ROOT / "01_Metadata"
QC_DIR = PROJECT_ROOT / "02_Quality_Control"
MODEL_DIR = (
    PROJECT_ROOT
    / "04_Models"
    / "Supervised"
    / "Stage5_LightweightLesionSegmentation"
)

FROZEN_MANIFEST_PATH = META_DIR / "stage3b_e5_frozen_dataset_manifest.csv"
STAGE3B_PROTOCOL_PATH = META_DIR / "stage3b_e5_dataset_freeze_protocol.json"
STAGE3B_AUDIT_PATH = QC_DIR / "stage3b_e5_dataset_freeze_audit.json"
SSL_CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "04_Models"
    / "SSL"
    / "Stage4A_MaskedContext3DCNN"
    / "stage4a_ssl_best.pt"
)
SSL_AUDIT_PATH = QC_DIR / "stage4a_ssl_training_audit.json"

LABEL_AUDIT_PATH = QC_DIR / "stage5a_supervised_label_audit.csv"
MEMORY_PILOT_PATH = QC_DIR / "stage5a_supervised_gpu_memory_pilot.json"
PROTOCOL_PATH = META_DIR / "stage5a_supervised_training_protocol.json"
AUDIT_PATH = QC_DIR / "stage5a_supervised_readiness_audit.json"


# =============================================================================
# SCIENTIFIC LOCKS
# =============================================================================

SEED = 20260728
EXPECTED_CASES = 1671
EXPECTED_TRAIN = 1376
EXPECTED_VALIDATION = 295
EXPECTED_TRAIN_PDAC = 403
EXPECTED_TRAIN_NON_PDAC = 973
EXPECTED_VALIDATION_PDAC = 88
EXPECTED_VALIDATION_NON_PDAC = 207

VOLUME_SHAPE = [240, 192, 128]
VOLUME_SPACING_MM = [1.25, 1.25, 2.0]
PATCH_SHAPE = [128, 128, 64]
OUTPUT_CHANNELS = ["pancreas_label_4", "lesion_label_1"]

HU_CENTER = 50.0
HU_HALF_WIDTH = 250.0
NORMALIZED_RANGE = [-1.0, 1.0]

TRAINING_ARMS = ["SSL_INITIALIZED", "RANDOM_INITIALIZED"]
MAX_EPOCHS = 30
MIN_EPOCHS = 10
EARLY_STOPPING_PATIENCE = 6
LEARNING_RATE = 2e-4
MINIMUM_LEARNING_RATE = 1e-6
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP_NORM = 1.0
TARGET_EFFECTIVE_BATCH_SIZE = 4
BATCH_SIZE_CANDIDATES = [4, 2, 1]

PANCREAS_LOSS_WEIGHT = 0.35
LESION_LOSS_WEIGHT = 0.65
MANUAL_PDAC_LESION_RELIABILITY_WEIGHT = 1.0
AUTOMATIC_PDAC_LESION_RELIABILITY_WEIGHT = 0.5
NON_PDAC_NEGATIVE_RELIABILITY_WEIGHT = 1.0

VALIDATION_MONITOR_CASES = 96
VALIDATION_CADENCE_EPOCHS = 2
SLIDING_WINDOW_OVERLAP = 0.50
PRELIMINARY_PROBABILITY_THRESHOLD = 0.50


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


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def group_count(channels):
    for groups in [8, 4, 2, 1]:
        if channels % groups == 0:
            return groups
    return 1


def normalize_text(series):
    return series.astype(str).str.strip().str.lower()


def scalar_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "pass"}


# =============================================================================
# ARCHITECTURE IDENTICAL TO THE STAGE 4A SSL ENCODER
# =============================================================================


class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(group_count(out_channels), out_channels),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.activation(self.main(x) + self.skip(x))


class DownBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
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
            nn.Conv3d(1, 16, kernel_size=3, padding=1, bias=False),
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
        self.up = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )
        self.fuse = ResidualBlock3D(
            out_channels + skip_channels,
            out_channels,
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
        self.segmentation_head = nn.Conv3d(16, 2, kernel_size=1)

    def forward(self, x):
        skip0, skip1, skip2, bottleneck = self.encoder(x)
        x = self.up2(bottleneck, skip2)
        x = self.up1(x, skip1)
        x = self.up0(x, skip0)
        return self.segmentation_head(x)


def soft_dice_loss(logits, target, epsilon=1e-6):
    probability = torch.sigmoid(logits)
    reduce_axes = tuple(range(2, probability.ndim))
    intersection = (probability * target).sum(dim=reduce_axes)
    denominator = probability.sum(dim=reduce_axes) + target.sum(dim=reduce_axes)
    dice = (2.0 * intersection + epsilon) / (denominator + epsilon)
    return 1.0 - dice.mean()


def pilot_loss(logits, target):
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target)
    dice = soft_dice_loss(logits, target)
    return 0.5 * bce + 0.5 * dice


# =============================================================================
# INPUT LOCKS
# =============================================================================


print("=" * 116)
print("STAGE 5A — SUPERVISED LESION-SEGMENTATION READINESS AND PROTOCOL LOCK")
print("=" * 116)

for required_path in [
    FROZEN_MANIFEST_PATH,
    STAGE3B_PROTOCOL_PATH,
    STAGE3B_AUDIT_PATH,
    SSL_CHECKPOINT_PATH,
    SSL_AUDIT_PATH,
]:
    if not required_path.exists():
        raise FileNotFoundError(f"Required input missing:\n{required_path}")

META_DIR.mkdir(parents=True, exist_ok=True)
QC_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

with open(STAGE3B_PROTOCOL_PATH, "r", encoding="utf-8") as file:
    stage3b_protocol = json.load(file)
with open(STAGE3B_AUDIT_PATH, "r", encoding="utf-8") as file:
    stage3b_audit = json.load(file)
with open(SSL_AUDIT_PATH, "r", encoding="utf-8") as file:
    ssl_audit = json.load(file)

manifest = pd.read_csv(FROZEN_MANIFEST_PATH, dtype={"study_id": str})
required_columns = {
    "study_id",
    "patient_id",
    "partition",
    "diagnostic_label",
    "annotation_type",
    "output_path",
    "verified_sha256",
    "all_checks_pass",
}
missing_columns = sorted(required_columns - set(manifest.columns))
if missing_columns:
    raise RuntimeError(f"Frozen manifest missing columns: {missing_columns}")

manifest["study_id"] = manifest["study_id"].astype(str)
manifest["partition"] = normalize_text(manifest["partition"])
manifest["diagnostic_label_normalized"] = normalize_text(
    manifest["diagnostic_label"]
).replace({"non-pdac": "non-pdac", "pdac": "pdac"})
manifest["annotation_type_normalized"] = normalize_text(
    manifest["annotation_type"]
)

if manifest["study_id"].duplicated().any():
    raise RuntimeError("Frozen manifest contains duplicate study IDs.")
if set(manifest["partition"]) != {"train", "validation"}:
    raise RuntimeError("A locked test partition appeared in Stage 5A inputs.")
if not set(manifest["diagnostic_label_normalized"]).issubset(
    {"pdac", "non-pdac"}
):
    raise RuntimeError("Unresolved diagnostic labels were found.")
if not set(manifest["annotation_type_normalized"]).issubset(
    {"manual", "automatic"}
):
    raise RuntimeError("Unresolved annotation types were found.")

manifest["persistent_file_exists"] = manifest["output_path"].map(
    lambda value: Path(str(value)).exists()
)
manifest["stage3b_qc_pass"] = manifest["all_checks_pass"].map(scalar_bool)

label_audit = (
    manifest.groupby(
        [
            "partition",
            "diagnostic_label_normalized",
            "annotation_type_normalized",
        ],
        dropna=False,
    )
    .agg(
        cases=("study_id", "size"),
        patients=("patient_id", "nunique"),
        persistent_files=("persistent_file_exists", "sum"),
        stage3b_qc_passed=("stage3b_qc_pass", "sum"),
    )
    .reset_index()
    .rename(
        columns={
            "diagnostic_label_normalized": "diagnostic_label",
            "annotation_type_normalized": "annotation_type",
        }
    )
)
atomic_write_csv(label_audit, LABEL_AUDIT_PATH)

partition_label_counts = pd.crosstab(
    manifest["partition"],
    manifest["diagnostic_label_normalized"],
)

print("\nSUPERVISED COHORT")
print("-" * 116)
print(label_audit.to_string(index=False))


# =============================================================================
# SSL ENCODER COMPATIBILITY
# =============================================================================


set_global_seed(SEED)
checkpoint = torch.load(SSL_CHECKPOINT_PATH, map_location="cpu", weights_only=False)
if "encoder_state" not in checkpoint:
    raise RuntimeError("Stage 4A checkpoint does not contain encoder_state.")

reference_model = LightweightDualHead3DSegmenter()
encoder_load_result = reference_model.encoder.load_state_dict(
    checkpoint["encoder_state"],
    strict=True,
)
ssl_encoder_compatible = (
    len(encoder_load_result.missing_keys) == 0
    and len(encoder_load_result.unexpected_keys) == 0
)
model_parameters = sum(p.numel() for p in reference_model.parameters())
encoder_parameters = sum(p.numel() for p in reference_model.encoder.parameters())
del reference_model
gc.collect()


# =============================================================================
# GPU MEMORY PILOT
# =============================================================================


if not torch.cuda.is_available():
    raise RuntimeError(
        "A CUDA GPU is required for the Stage 5A memory pilot. "
        "Select a GPU runtime, reconnect, and rerun this same file."
    )

device = torch.device("cuda")
gpu_name = torch.cuda.get_device_name(0)
gpu_memory_gib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)


def run_memory_pilot(batch_size):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = None
    optimizer = None
    try:
        set_global_seed(SEED + int(batch_size))
        model = LightweightDualHead3DSegmenter().to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )
        x = torch.zeros(
            (batch_size, 1, *PATCH_SHAPE),
            dtype=torch.float32,
            device=device,
        )
        target = torch.zeros(
            (batch_size, 2, *PATCH_SHAPE),
            dtype=torch.float32,
            device=device,
        )
        # Non-empty synthetic targets exercise both supervised heads.
        target[:, 0, 24:104, 28:100, 12:52] = 1.0
        target[:, 1, 50:78, 52:80, 24:44] = 1.0

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(x)
            loss = pilot_loss(logits, target)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
        result = {
            "batch_size": int(batch_size),
            "status": "PASS",
            "peak_allocated_GiB": float(peak),
            "pilot_loss": float(loss.detach().cpu()),
        }
    except torch.cuda.OutOfMemoryError as error:
        result = {
            "batch_size": int(batch_size),
            "status": "CUDA_OUT_OF_MEMORY",
            "error": str(error),
        }
    finally:
        del model, optimizer
        for name in ["x", "target", "logits", "loss"]:
            if name in locals():
                del locals()[name]
        gc.collect()
        torch.cuda.empty_cache()
    return result


memory_results = []
selected_batch_size = None
for candidate in BATCH_SIZE_CANDIDATES:
    result = run_memory_pilot(candidate)
    memory_results.append(result)
    if result["status"] == "PASS":
        selected_batch_size = int(candidate)
        break

if selected_batch_size is None:
    raise RuntimeError("No supervised patch batch size passed the GPU memory pilot.")

gradient_accumulation_steps = int(
    math.ceil(TARGET_EFFECTIVE_BATCH_SIZE / selected_batch_size)
)
effective_batch_size = selected_batch_size * gradient_accumulation_steps

memory_pilot = {
    "stage": "5A",
    "created_at_utc": utc_now(),
    "gpu": gpu_name,
    "gpu_memory_GiB": gpu_memory_gib,
    "patch_shape": PATCH_SHAPE,
    "candidate_results": memory_results,
    "selected_batch_size": selected_batch_size,
    "gradient_accumulation_steps": gradient_accumulation_steps,
    "effective_batch_size": effective_batch_size,
}
atomic_write_json(memory_pilot, MEMORY_PILOT_PATH)

print("\nGPU MEMORY PILOT")
print("-" * 116)
for result in memory_results:
    print(result)
print(f"Selected batch size: {selected_batch_size}")
print(f"Gradient accumulation: {gradient_accumulation_steps}")
print(f"Effective batch size: {effective_batch_size}")


# =============================================================================
# LOCK PROTOCOL AND AUDIT
# =============================================================================


train = manifest[manifest["partition"] == "train"]
validation = manifest[manifest["partition"] == "validation"]

observed = {
    "cases": int(len(manifest)),
    "train": int(len(train)),
    "validation": int(len(validation)),
    "train_PDAC": int(
        (train["diagnostic_label_normalized"] == "pdac").sum()
    ),
    "train_non_PDAC": int(
        (train["diagnostic_label_normalized"] == "non-pdac").sum()
    ),
    "validation_PDAC": int(
        (validation["diagnostic_label_normalized"] == "pdac").sum()
    ),
    "validation_non_PDAC": int(
        (validation["diagnostic_label_normalized"] == "non-pdac").sum()
    ),
}

readiness_checks = {
    "Stage 3B dataset is frozen": bool(stage3b_protocol.get("dataset_frozen")),
    "Stage 3B audit passed": str(stage3b_audit.get("result", "")).startswith("PASS"),
    "Exactly 1671 development cases": observed["cases"] == EXPECTED_CASES,
    "Training count remains 1376": observed["train"] == EXPECTED_TRAIN,
    "Validation count remains 295": observed["validation"] == EXPECTED_VALIDATION,
    "Training PDAC count remains 403": observed["train_PDAC"] == EXPECTED_TRAIN_PDAC,
    "Training non-PDAC count remains 973": observed["train_non_PDAC"] == EXPECTED_TRAIN_NON_PDAC,
    "Validation PDAC count remains 88": observed["validation_PDAC"] == EXPECTED_VALIDATION_PDAC,
    "Validation non-PDAC count remains 207": observed["validation_non_PDAC"] == EXPECTED_VALIDATION_NON_PDAC,
    "All persistent E5 files exist": bool(manifest["persistent_file_exists"].all()),
    "All Stage 3B case QC flags pass": bool(manifest["stage3b_qc_pass"].all()),
    "No duplicate study IDs": not bool(manifest["study_id"].duplicated().any()),
    "No locked test case was accessed": set(manifest["partition"]) == {"train", "validation"},
    "Stage 4A SSL audit passed": str(ssl_audit.get("result", "")).startswith("PASS"),
    "SSL encoder checkpoint is compatible": bool(ssl_encoder_compatible),
    "GPU memory pilot passed": selected_batch_size is not None,
    "Model remains below three million parameters": model_parameters < 3_000_000,
}
all_checks_pass = all(bool(value) for value in readiness_checks.values())

protocol = {
    "stage": "5A",
    "created_at_utc": utc_now(),
    "protocol_locked": all_checks_pass,
    "task": "dual_head_3D_pancreas_and_PDAC_lesion_segmentation",
    "deployment_output": "lesion_probability_map_for_component_detection_and_FROC",
    "training_arms": TRAINING_ARMS,
    "fair_ablation_rule": (
        "The two arms use identical architecture, data order, augmentations, "
        "losses, optimizer, scheduler, validation cases, and stopping rule. "
        "Only encoder initialization differs."
    ),
    "ssl_arm_initialization": {
        "encoder": "stage4a_best_encoder_state",
        "decoder_and_head": "deterministic_random_initialization",
        "checkpoint_path": str(SSL_CHECKPOINT_PATH),
        "checkpoint_sha256": sha256_file(SSL_CHECKPOINT_PATH),
        "best_ssl_epoch": int(checkpoint.get("best_epoch", -1)),
    },
    "random_arm_initialization": "deterministic_random_initialization_of_entire_model",
    "model": {
        "name": "LightweightDualHead3DSegmenter",
        "total_parameters": int(model_parameters),
        "encoder_parameters": int(encoder_parameters),
        "output_channels": OUTPUT_CHANNELS,
    },
    "inputs": {
        "dataset": "PANORAMA_LOCAL_DEVELOPMENT_E5_V1",
        "volume_shape": VOLUME_SHAPE,
        "spacing_mm": VOLUME_SPACING_MM,
        "patch_shape": PATCH_SHAPE,
        "normalization": "clip((HU - 50) / 250, -1, 1)",
        "normalized_range": NORMALIZED_RANGE,
    },
    "supervision": {
        "pancreas_target": "mask_labels == 4",
        "lesion_target": "mask_labels == 1",
        "pancreas_loss_weight": PANCREAS_LOSS_WEIGHT,
        "lesion_loss_weight": LESION_LOSS_WEIGHT,
        "per_head_loss": "0.5 * BCEWithLogits + 0.5 * soft_Dice",
        "manual_PDAC_lesion_reliability_weight": MANUAL_PDAC_LESION_RELIABILITY_WEIGHT,
        "automatic_PDAC_lesion_reliability_weight": AUTOMATIC_PDAC_LESION_RELIABILITY_WEIGHT,
        "non_PDAC_negative_reliability_weight": NON_PDAC_NEGATIVE_RELIABILITY_WEIGHT,
    },
    "sampling": {
        "case_balance": "50_percent_PDAC_50_percent_non_PDAC",
        "PDAC_patch_center": "random_lesion_voxel_with_bounded_jitter",
        "non_PDAC_patch_center": "random_pancreas_voxel_with_bounded_jitter",
        "augmentations": [
            "left_right_flip",
            "anterior_posterior_flip",
            "small_intensity_scale",
            "small_intensity_shift",
        ],
    },
    "optimization": {
        "maximum_epochs_per_arm": MAX_EPOCHS,
        "minimum_epochs_per_arm": MIN_EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "learning_rate": LEARNING_RATE,
        "minimum_learning_rate": MINIMUM_LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "gradient_clip_norm": GRADIENT_CLIP_NORM,
        "selected_batch_size": selected_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_batch_size": effective_batch_size,
        "mixed_precision": True,
    },
    "validation": {
        "monitor_subset_cases": VALIDATION_MONITOR_CASES,
        "monitor_subset_lock": "patient_stratified_fixed_before_training",
        "cadence_epochs": VALIDATION_CADENCE_EPOCHS,
        "inference": "sliding_window_over_full_E5_crop",
        "sliding_window_overlap": SLIDING_WINDOW_OVERLAP,
        "selection_metric": "0.70_lesion_Dice_plus_0.30_pancreas_Dice",
        "full_validation_evaluation": "once_for_each_selected_best_arm_checkpoint",
        "preliminary_probability_threshold": PRELIMINARY_PROBABILITY_THRESHOLD,
        "final_detection_threshold": "calibrated_later_on_validation_only",
    },
    "data_governance": {
        "weights_updated_from": ["train"],
        "model_selected_from": ["validation"],
        "internal_test_access": "prohibited",
        "external_test_access": "prohibited",
        "test_threshold_calibration": "prohibited",
    },
    "cohort_counts": observed,
    "frozen_manifest_path": str(FROZEN_MANIFEST_PATH),
    "label_audit_path": str(LABEL_AUDIT_PATH),
    "gpu_memory_pilot_path": str(MEMORY_PILOT_PATH),
}
atomic_write_json(protocol, PROTOCOL_PATH)

audit = {
    "stage": "5A",
    "created_at_utc": utc_now(),
    "result": (
        "PASS_SUPERVISED_PROTOCOL_LOCKED" if all_checks_pass else "FAIL"
    ),
    "readiness_checks": readiness_checks,
    "all_checks_pass": all_checks_pass,
    "observed_counts": observed,
    "model_parameters": int(model_parameters),
    "encoder_parameters": int(encoder_parameters),
    "ssl_checkpoint_path": str(SSL_CHECKPOINT_PATH),
    "ssl_checkpoint_sha256": sha256_file(SSL_CHECKPOINT_PATH),
    "selected_batch_size": selected_batch_size,
    "gradient_accumulation_steps": gradient_accumulation_steps,
    "effective_batch_size": effective_batch_size,
    "locked_test_cases_accessed": 0,
    "raw_files_modified": False,
    "protocol_path": str(PROTOCOL_PATH),
}
atomic_write_json(audit, AUDIT_PATH)

print("\nREADINESS CHECKS")
print("-" * 116)
for name, passed in readiness_checks.items():
    print(f"  {name}: {bool(passed)}")

print("\nLOCKED SUPERVISED DESIGN")
print("-" * 116)
print(f"Training arms: {TRAINING_ARMS}")
print(f"Model parameters: {model_parameters:,}")
print(f"Encoder parameters: {encoder_parameters:,}")
print(f"Patch shape: {PATCH_SHAPE}")
print(f"Output channels: {OUTPUT_CHANNELS}")
print(f"Selected GPU batch size: {selected_batch_size}")
print(f"Effective batch size: {effective_batch_size}")
print("Locked test cases accessed: 0")

print("\nProtocol:")
print(PROTOCOL_PATH)
print("\nAudit:")
print(AUDIT_PATH)
print("\n" + "=" * 116)
print(
    "STAGE 5A RESULT: "
    + (
        "PASS — SUPERVISED PROTOCOL LOCKED"
        if all_checks_pass
        else "FAIL"
    )
)
print("=" * 116)

if not all_checks_pass:
    failed = [name for name, passed in readiness_checks.items() if not passed]
    raise RuntimeError(f"Stage 5A failed readiness checks: {failed}")
