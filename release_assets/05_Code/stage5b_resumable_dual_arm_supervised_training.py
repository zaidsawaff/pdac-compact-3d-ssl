from pathlib import Path
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
import random
import shutil
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# =============================================================================
# PATHS
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
STAGE3B_PROTOCOL_PATH = META_DIR / "stage3b_e5_dataset_freeze_protocol.json"
STAGE5A_PROTOCOL_PATH = META_DIR / "stage5a_supervised_training_protocol.json"
STAGE5A_AUDIT_PATH = QC_DIR / "stage5a_supervised_readiness_audit.json"
SSL_CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "04_Models"
    / "SSL"
    / "Stage4A_MaskedContext3DCNN"
    / "stage4a_ssl_best.pt"
)

VALIDATION_SUBSET_PATH = META_DIR / "stage5b_validation_monitor_subset.csv"
TRAINING_PROTOCOL_PATH = META_DIR / "stage5b_dual_arm_training_protocol.json"
FINAL_AUDIT_PATH = QC_DIR / "stage5b_dual_arm_training_audit.json"
LOCAL_CACHE_DIR = Path("/content/pdac_e5_ssl_cache")


# =============================================================================
# LOCKED DESIGN
# =============================================================================

SEED = 20260728
EXPECTED_CASES = 1671
EXPECTED_TRAIN = 1376
EXPECTED_VALIDATION = 295
EXPECTED_TRAIN_PDAC = 403
EXPECTED_TRAIN_NON_PDAC = 973

VOLUME_SHAPE = np.asarray([240, 192, 128], dtype=int)
PATCH_SHAPE = np.asarray([128, 128, 64], dtype=int)
HU_CENTER = 50.0
HU_HALF_WIDTH = 250.0

ARMS = ["SSL_INITIALIZED", "RANDOM_INITIALIZED"]
SAMPLES_PER_EPOCH = 1376
PDAC_SAMPLES_PER_EPOCH = 688
NON_PDAC_SAMPLES_PER_EPOCH = 688
MAX_EPOCHS = 30
MIN_EPOCHS = 10
EARLY_STOPPING_PATIENCE = 6
LEARNING_RATE = 2e-4
MINIMUM_LEARNING_RATE = 1e-6
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP_NORM = 1.0
PANCREAS_LOSS_WEIGHT = 0.35
LESION_LOSS_WEIGHT = 0.65
VALIDATION_MONITOR_CASES = 96
VALIDATION_PDAC_CASES = 48
VALIDATION_NON_PDAC_CASES = 48
VALIDATION_CADENCE_EPOCHS = 2
SLIDING_OVERLAP = 0.50
PROBABILITY_THRESHOLD = 0.50
NUM_WORKERS = 2
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


def atomic_torch_save(payload, path):
    temporary = Path(str(path) + ".tmp")
    torch.save(payload, temporary)
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


def stable_integer(*parts):
    text = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2 ** 32)


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


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


# =============================================================================
# MODEL — SAME ENCODER AS STAGE 4A
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


def supervised_loss(logits, target, reliability):
    probability = torch.sigmoid(logits)
    reduce_axes = tuple(range(2, logits.ndim))

    bce_voxel = nn.functional.binary_cross_entropy_with_logits(
        logits, target, reduction="none"
    )
    bce_per_sample_head = bce_voxel.mean(dim=reduce_axes)
    intersection = (probability * target).sum(dim=reduce_axes)
    denominator = probability.sum(dim=reduce_axes) + target.sum(dim=reduce_axes)
    dice_per_sample_head = 1.0 - (
        (2.0 * intersection + 1e-6) / (denominator + 1e-6)
    )
    combined = 0.5 * bce_per_sample_head + 0.5 * dice_per_sample_head

    pancreas_loss = combined[:, 0].mean()
    reliability = torch.clamp(reliability, min=1e-6)
    lesion_loss = (combined[:, 1] * reliability).sum() / reliability.sum()
    total = (
        PANCREAS_LOSS_WEIGHT * pancreas_loss
        + LESION_LOSS_WEIGHT * lesion_loss
    )
    return total, pancreas_loss, lesion_loss


# =============================================================================
# BALANCED, DETERMINISTIC SUPERVISED PATCH DATASET
# =============================================================================


class BalancedSupervisedPatchDataset(Dataset):
    def __init__(self, train_frame, cache_dir, epoch):
        self.cache_dir = Path(cache_dir)
        self.epoch = int(epoch)
        frame = train_frame.reset_index(drop=True).copy()
        pdac = frame[frame["diagnostic_label_normalized"] == "pdac"]
        non_pdac = frame[frame["diagnostic_label_normalized"] == "non-pdac"]
        rng = np.random.RandomState(stable_integer(SEED, "samples", epoch))

        pdac_choice = rng.choice(
            len(pdac), size=PDAC_SAMPLES_PER_EPOCH, replace=True
        )
        non_choice = rng.choice(
            len(non_pdac), size=NON_PDAC_SAMPLES_PER_EPOCH, replace=False
        )
        sampled = pd.concat(
            [pdac.iloc[pdac_choice], non_pdac.iloc[non_choice]],
            ignore_index=True,
        )
        permutation = rng.permutation(len(sampled))
        self.rows = sampled.iloc[permutation].reset_index(drop=True)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        study_id = str(row["study_id"])
        rng = np.random.RandomState(
            stable_integer(SEED, "patch", self.epoch, index, study_id)
        )
        path = self.cache_dir / f"{study_id}.npz"
        with np.load(path, allow_pickle=False) as data:
            ct_hu = np.asarray(data["ct_hu"], dtype=np.int16)
            mask = np.asarray(data["mask_labels"], dtype=np.uint8)

        if tuple(ct_hu.shape) != tuple(VOLUME_SHAPE):
            raise RuntimeError(f"{study_id}: unexpected CT shape {ct_hu.shape}.")
        if mask.shape != ct_hu.shape:
            raise RuntimeError(f"{study_id}: CT-mask shape mismatch.")

        is_pdac = row["diagnostic_label_normalized"] == "pdac"
        anchor_label = 1 if is_pdac else 4
        candidates = np.flatnonzero(mask.reshape(-1) == anchor_label)
        if len(candidates) == 0:
            raise RuntimeError(
                f"{study_id}: required anchor label {anchor_label} is absent."
            )
        flat_index = int(candidates[rng.randint(0, len(candidates))])
        anchor = np.asarray(np.unravel_index(flat_index, mask.shape), dtype=int)
        jitter = np.asarray(
            [rng.randint(-24, 25), rng.randint(-24, 25), rng.randint(-12, 13)],
            dtype=int,
        )
        center = anchor + jitter
        maximum_start = VOLUME_SHAPE - PATCH_SHAPE
        start = np.clip(center - PATCH_SHAPE // 2, 0, maximum_start)
        end = start + PATCH_SHAPE
        slices = tuple(slice(int(a), int(b)) for a, b in zip(start, end))

        image = ct_hu[slices].astype(np.float32, copy=False)
        patch_mask = mask[slices]
        image = np.clip(
            (image - HU_CENTER) / HU_HALF_WIDTH, -1.0, 1.0
        ).astype(np.float32, copy=False)
        target = np.stack(
            [patch_mask == 4, patch_mask == 1], axis=0
        ).astype(np.float32, copy=False)

        for axis in [0, 1]:
            if rng.rand() < 0.5:
                image = np.flip(image, axis=axis)
                target = np.flip(target, axis=axis + 1)
        scale = float(rng.uniform(0.92, 1.08))
        shift = float(rng.uniform(-0.06, 0.06))
        image = np.clip(image * scale + shift, -1.0, 1.0)

        annotation = str(row["annotation_type_normalized"])
        if is_pdac and annotation == "automatic":
            reliability = 0.5
        else:
            reliability = 1.0

        return {
            "image": torch.from_numpy(
                np.ascontiguousarray(image, dtype=np.float32)
            ).unsqueeze(0),
            "target": torch.from_numpy(
                np.ascontiguousarray(target, dtype=np.float32)
            ),
            "reliability": torch.tensor(reliability, dtype=torch.float32),
            "is_pdac": torch.tensor(is_pdac, dtype=torch.bool),
            "study_id": study_id,
        }


# =============================================================================
# DATA AND PROTOCOL LOCKS
# =============================================================================


print("=" * 120)
print("STAGE 5B — RESUMABLE DUAL-ARM SUPERVISED 3D LESION SEGMENTATION")
print("=" * 120)

for required in [
    FROZEN_MANIFEST_PATH,
    STAGE3B_PROTOCOL_PATH,
    STAGE5A_PROTOCOL_PATH,
    STAGE5A_AUDIT_PATH,
    SSL_CHECKPOINT_PATH,
]:
    if not required.exists():
        raise FileNotFoundError(f"Required input missing:\n{required}")

if not torch.cuda.is_available():
    raise RuntimeError("Stage 5B requires a CUDA GPU.")

with open(STAGE3B_PROTOCOL_PATH, "r", encoding="utf-8") as file:
    stage3b_protocol = json.load(file)
with open(STAGE5A_PROTOCOL_PATH, "r", encoding="utf-8") as file:
    stage5a_protocol = json.load(file)
with open(STAGE5A_AUDIT_PATH, "r", encoding="utf-8") as file:
    stage5a_audit = json.load(file)

if stage3b_protocol.get("dataset_frozen") is not True:
    raise RuntimeError("Stage 3B frozen dataset lock is not valid.")
if stage5a_protocol.get("protocol_locked") is not True:
    raise RuntimeError("Stage 5A supervised protocol is not locked.")
if stage5a_audit.get("all_checks_pass") is not True:
    raise RuntimeError("Stage 5A readiness audit did not pass.")

selected_batch_size = int(
    stage5a_protocol["optimization"]["selected_batch_size"]
)
accumulation_steps = int(
    stage5a_protocol["optimization"]["gradient_accumulation_steps"]
)
if selected_batch_size != 4 or accumulation_steps != 1:
    raise RuntimeError("Unexpected Stage 5A batch configuration.")

MODEL_ROOT.mkdir(parents=True, exist_ok=True)
QC_DIR.mkdir(parents=True, exist_ok=True)
META_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

manifest = pd.read_csv(FROZEN_MANIFEST_PATH, dtype={"study_id": str})
manifest["study_id"] = manifest["study_id"].astype(str).str.strip()
manifest["partition"] = manifest["partition"].astype(str).str.strip().str.lower()
manifest["diagnostic_label_normalized"] = (
    manifest["diagnostic_label"].astype(str).str.strip().str.lower()
)
manifest["annotation_type_normalized"] = (
    manifest["annotation_type"].astype(str).str.strip().str.lower()
)

required_columns = {
    "study_id", "partition", "diagnostic_label", "annotation_type",
    "output_path", "output_size_bytes", "verified_sha256", "all_checks_pass",
}
missing_columns = sorted(required_columns - set(manifest.columns))
if missing_columns:
    raise RuntimeError(f"Frozen manifest missing columns: {missing_columns}")
if len(manifest) != EXPECTED_CASES or manifest["study_id"].nunique() != EXPECTED_CASES:
    raise RuntimeError("Frozen development cohort changed.")
if not truth_flags(manifest["all_checks_pass"]).all():
    raise RuntimeError("Frozen manifest contains a failed case.")
if set(manifest["partition"]) != {"train", "validation"}:
    raise RuntimeError("A locked test case entered Stage 5B.")

train_frame = manifest[manifest["partition"] == "train"].copy()
validation_frame = manifest[manifest["partition"] == "validation"].copy()
if len(train_frame) != EXPECTED_TRAIN or len(validation_frame) != EXPECTED_VALIDATION:
    raise RuntimeError("Train-validation counts changed.")
if (train_frame["diagnostic_label_normalized"] == "pdac").sum() != EXPECTED_TRAIN_PDAC:
    raise RuntimeError("Training PDAC count changed.")
if (train_frame["diagnostic_label_normalized"] == "non-pdac").sum() != EXPECTED_TRAIN_NON_PDAC:
    raise RuntimeError("Training non-PDAC count changed.")

set_global_seed(SEED)
torch.set_float32_matmul_precision("high")
device = torch.device("cuda")
gpu_name = torch.cuda.get_device_name(0)
gpu_memory_gib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

print(f"PyTorch: {torch.__version__}")
print(f"GPU: {gpu_name}")
print(f"GPU memory: {gpu_memory_gib:.2f} GiB")
print(f"Training cases: {len(train_frame)}")
print(f"Validation cases: {len(validation_frame)}")
print(f"Batch size: {selected_batch_size}")
print("Locked test cases accessed: 0")


# =============================================================================
# LOCAL HIGH-SPEED CACHE — REUSES THE STAGE 4A CACHE WHEN AVAILABLE
# =============================================================================


print("\nPreparing or verifying the local high-speed E5 cache...")
additional_bytes = 0
for _, row in manifest.iterrows():
    destination = LOCAL_CACHE_DIR / f"{row['study_id']}.npz"
    expected_size = int(row["output_size_bytes"])
    if not (destination.exists() and destination.stat().st_size == expected_size):
        additional_bytes += expected_size

if disk_free_gib("/content") * (1024 ** 3) < additional_bytes * 1.10 + 1024 ** 3:
    raise RuntimeError("Insufficient local storage for the E5 cache.")

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
            raise FileNotFoundError(f"Frozen E5 file missing: {source}")
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

expected_cache = {f"{study_id}.npz" for study_id in manifest["study_id"]}
observed_cache = {path.name for path in LOCAL_CACHE_DIR.glob("*.npz")}
if observed_cache != expected_cache:
    raise RuntimeError("Local E5 cache inventory is not exact.")


# =============================================================================
# FIXED VALIDATION MONITOR SUBSET
# =============================================================================


def stable_rank(study_id):
    return hashlib.sha256(f"{SEED}|validation|{study_id}".encode()).hexdigest()


def build_validation_subset(frame):
    frame = frame.copy()
    frame["stable_rank"] = frame["study_id"].map(stable_rank)
    pdac = (
        frame[frame["diagnostic_label_normalized"] == "pdac"]
        .sort_values("stable_rank")
        .head(VALIDATION_PDAC_CASES)
    )
    non_pdac = (
        frame[frame["diagnostic_label_normalized"] == "non-pdac"]
        .sort_values("stable_rank")
        .head(VALIDATION_NON_PDAC_CASES)
    )
    subset = pd.concat([pdac, non_pdac], ignore_index=True)
    subset = subset.sort_values("study_id").reset_index(drop=True)
    subset["monitor_subset_locked"] = True
    return subset


if VALIDATION_SUBSET_PATH.exists():
    saved_subset = pd.read_csv(VALIDATION_SUBSET_PATH, dtype={"study_id": str})
    expected_subset = build_validation_subset(validation_frame)
    if set(saved_subset["study_id"].astype(str)) != set(expected_subset["study_id"]):
        raise RuntimeError("Saved validation monitor subset does not match the lock.")
    validation_subset = expected_subset
else:
    validation_subset = build_validation_subset(validation_frame)
    atomic_write_csv(validation_subset, VALIDATION_SUBSET_PATH)

if len(validation_subset) != VALIDATION_MONITOR_CASES:
    raise RuntimeError("Validation monitor subset count is incorrect.")
if (validation_subset["diagnostic_label_normalized"] == "pdac").sum() != VALIDATION_PDAC_CASES:
    raise RuntimeError("Validation monitor PDAC count is incorrect.")

training_protocol = {
    "stage": "5B",
    "created_at_utc": utc_now(),
    "arms": ARMS,
    "maximum_epochs_per_arm": MAX_EPOCHS,
    "samples_per_epoch": SAMPLES_PER_EPOCH,
    "PDAC_samples_per_epoch": PDAC_SAMPLES_PER_EPOCH,
    "non_PDAC_samples_per_epoch": NON_PDAC_SAMPLES_PER_EPOCH,
    "patch_shape": PATCH_SHAPE.tolist(),
    "selected_batch_size": selected_batch_size,
    "gradient_accumulation_steps": accumulation_steps,
    "validation_monitor_cases": VALIDATION_MONITOR_CASES,
    "validation_monitor_PDAC": VALIDATION_PDAC_CASES,
    "validation_monitor_non_PDAC": VALIDATION_NON_PDAC_CASES,
    "validation_cadence_epochs": VALIDATION_CADENCE_EPOCHS,
    "selection_score": "0.70*PDAC_lesion_Dice + 0.30*all_case_pancreas_Dice",
    "sliding_window_overlap": SLIDING_OVERLAP,
    "probability_threshold": PROBABILITY_THRESHOLD,
    "locked_test_cases_accessed": 0,
}
atomic_write_json(training_protocol, TRAINING_PROTOCOL_PATH)


# =============================================================================
# LOADERS, TRAINING, AND FULL-CROP MONITOR VALIDATION
# =============================================================================


def make_train_loader(epoch):
    dataset = BalancedSupervisedPatchDataset(
        train_frame, LOCAL_CACHE_DIR, epoch=epoch
    )
    generator = torch.Generator()
    generator.manual_seed(stable_integer(SEED, "loader", epoch))
    return DataLoader(
        dataset,
        batch_size=selected_batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=False,
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=True,
    )


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
    model.eval()
    normalized = np.clip(
        (ct_hu.astype(np.float32) - HU_CENTER) / HU_HALF_WIDTH,
        -1.0,
        1.0,
    )
    probability_sum = np.zeros((2, *VOLUME_SHAPE), dtype=np.float32)
    count = np.zeros(tuple(VOLUME_SHAPE), dtype=np.float32)

    for offset in range(0, len(window_starts), selected_batch_size):
        chunk = window_starts[offset:offset + selected_batch_size]
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


@torch.no_grad()
def evaluate_monitor_subset(model, arm, epoch):
    case_rows = []
    for order, (_, row) in enumerate(validation_subset.iterrows(), start=1):
        study_id = str(row["study_id"])
        with np.load(LOCAL_CACHE_DIR / f"{study_id}.npz", allow_pickle=False) as data:
            ct_hu = np.asarray(data["ct_hu"], dtype=np.int16)
            mask = np.asarray(data["mask_labels"], dtype=np.uint8)
        probabilities = infer_full_crop(model, ct_hu)
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
        case_rows.append(
            {
                "arm": arm,
                "epoch": int(epoch),
                "study_id": study_id,
                "diagnostic_label": row["diagnostic_label_normalized"],
                "annotation_type": row["annotation_type_normalized"],
                "pancreas_dice": pancreas_dice,
                "lesion_dice": lesion_dice,
                "predicted_lesion_voxels": int(lesion_prediction.sum()),
                "target_lesion_voxels": int(lesion_target.sum()),
                "negative_case_false_positive": bool(
                    (not is_pdac) and lesion_prediction.any()
                ),
                "evaluated_at_utc": utc_now(),
            }
        )
        del ct_hu, mask, probabilities
        if order % 16 == 0 or order == len(validation_subset):
            print(f"    Validation {order}/{len(validation_subset)}")

    cases = pd.DataFrame(case_rows)
    pdac_cases = cases[cases["diagnostic_label"] == "pdac"]
    negative_cases = cases[cases["diagnostic_label"] == "non-pdac"]
    pancreas_dice = float(cases["pancreas_dice"].mean())
    lesion_dice = float(pdac_cases["lesion_dice"].mean())
    negative_specificity = float(
        1.0 - negative_cases["negative_case_false_positive"].mean()
    )
    selection_score = 0.70 * lesion_dice + 0.30 * pancreas_dice
    summary = {
        "validation_pancreas_dice": pancreas_dice,
        "validation_PDAC_lesion_dice": lesion_dice,
        "validation_negative_specificity_at_0_5": negative_specificity,
        "validation_selection_score": selection_score,
    }
    return summary, cases


def initialize_model(arm):
    set_global_seed(SEED)
    model = LightweightDualHead3DSegmenter()
    initialization = {
        "arm": arm,
        "encoder_source": "deterministic_random",
        "missing_keys": [],
        "unexpected_keys": [],
    }
    if arm == "SSL_INITIALIZED":
        ssl_checkpoint = torch.load(
            SSL_CHECKPOINT_PATH, map_location="cpu", weights_only=False
        )
        result = model.encoder.load_state_dict(
            ssl_checkpoint["encoder_state"], strict=True
        )
        initialization.update(
            {
                "encoder_source": "stage4a_ssl_best_encoder",
                "ssl_best_epoch": int(ssl_checkpoint.get("best_epoch", -1)),
                "missing_keys": list(result.missing_keys),
                "unexpected_keys": list(result.unexpected_keys),
            }
        )
    return model, initialization


def train_one_arm(arm):
    arm_slug = arm.lower()
    arm_dir = MODEL_ROOT / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    last_path = arm_dir / "stage5b_last.pt"
    best_path = arm_dir / "stage5b_best.pt"
    history_path = arm_dir / "stage5b_training_history.csv"
    validation_case_path = QC_DIR / f"stage5b_{arm_slug}_validation_monitor_cases.csv"
    arm_audit_path = QC_DIR / f"stage5b_{arm_slug}_training_audit.json"

    if arm_audit_path.exists():
        with open(arm_audit_path, "r", encoding="utf-8") as file:
            existing_audit = json.load(file)
        if existing_audit.get("training_complete") is True:
            print(f"\n{arm}: already complete — skipping.")
            return existing_audit

    model, initialization = initialize_model(arm)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS, eta_min=MINIMUM_LEARNING_RATE
    )
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    start_epoch = 1
    best_epoch = -1
    best_score = -math.inf
    validations_without_improvement = 0
    history = pd.DataFrame()
    validation_cases_all = pd.DataFrame()

    if history_path.exists():
        history = pd.read_csv(history_path)
    if validation_case_path.exists():
        validation_cases_all = pd.read_csv(validation_case_path)

    if last_path.exists():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        if checkpoint.get("arm") != arm:
            raise RuntimeError(f"{arm}: checkpoint arm identity mismatch.")
        model.load_state_dict(checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_epoch = int(checkpoint["best_epoch"])
        best_score = float(checkpoint["best_score"])
        validations_without_improvement = int(
            checkpoint["validations_without_improvement"]
        )
        print(
            f"\n{arm}: resuming after epoch {start_epoch - 1}; "
            f"best epoch={best_epoch}, best score={best_score:.6f}"
        )
    else:
        print(f"\n{arm}: starting from epoch 1")

    stopped_early = False
    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        epoch_start = time.time()
        train_loader = make_train_loader(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss_sum = 0.0
        pancreas_loss_sum = 0.0
        lesion_loss_sum = 0.0
        sample_count = 0

        for step, batch in enumerate(train_loader, start=1):
            image = batch["image"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            reliability = batch["reliability"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=USE_AMP):
                logits = model(image)
                total_loss, pancreas_loss, lesion_loss = supervised_loss(
                    logits, target, reliability
                )
                scaled_loss = total_loss / accumulation_steps
            scaler.scale(scaled_loss).backward()

            if step % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), GRADIENT_CLIP_NORM
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            batch_size_now = int(image.shape[0])
            total_loss_sum += float(total_loss.detach().cpu()) * batch_size_now
            pancreas_loss_sum += float(pancreas_loss.detach().cpu()) * batch_size_now
            lesion_loss_sum += float(lesion_loss.detach().cpu()) * batch_size_now
            sample_count += batch_size_now
            if step % 50 == 0 or step == len(train_loader):
                print(
                    f"  {arm} epoch {epoch}/{MAX_EPOCHS} — "
                    f"step {step}/{len(train_loader)} — "
                    f"loss {total_loss_sum / sample_count:.5f}"
                )
            del image, target, reliability, logits, total_loss, pancreas_loss, lesion_loss

        should_validate = (
            epoch == 1
            or epoch % VALIDATION_CADENCE_EPOCHS == 0
            or epoch == MAX_EPOCHS
        )
        validation_summary = {
            "validation_pancreas_dice": np.nan,
            "validation_PDAC_lesion_dice": np.nan,
            "validation_negative_specificity_at_0_5": np.nan,
            "validation_selection_score": np.nan,
        }
        improved = False
        if should_validate:
            print(f"  {arm} epoch {epoch}: full-crop monitor validation...")
            validation_summary, validation_cases = evaluate_monitor_subset(
                model, arm, epoch
            )
            validation_cases_all = pd.concat(
                [validation_cases_all, validation_cases], ignore_index=True
            )
            validation_cases_all = validation_cases_all.drop_duplicates(
                subset=["arm", "epoch", "study_id"], keep="last"
            ).reset_index(drop=True)
            atomic_write_csv(validation_cases_all, validation_case_path)
            score = float(validation_summary["validation_selection_score"])
            improved = score > best_score + 1e-6
            if improved:
                best_score = score
                best_epoch = epoch
                validations_without_improvement = 0
            else:
                validations_without_improvement += 1

        scheduler.step()
        epoch_seconds = float(time.time() - epoch_start)
        history_row = {
            "arm": arm,
            "epoch": epoch,
            "train_cases_seen": sample_count,
            "train_total_loss": total_loss_sum / sample_count,
            "train_pancreas_loss": pancreas_loss_sum / sample_count,
            "train_lesion_loss": lesion_loss_sum / sample_count,
            **validation_summary,
            "validation_performed": should_validate,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "best_epoch_so_far": best_epoch,
            "best_selection_score_so_far": best_score,
            "validations_without_improvement": validations_without_improvement,
            "epoch_seconds": epoch_seconds,
            "completed_at_utc": utc_now(),
        }
        history = pd.concat([history, pd.DataFrame([history_row])], ignore_index=True)
        history = history.drop_duplicates(subset=["arm", "epoch"], keep="last")
        atomic_write_csv(history, history_path)

        checkpoint_payload = {
            "stage": "5B",
            "arm": arm,
            "epoch": epoch,
            "model_state": model.state_dict(),
            "encoder_state": model.encoder.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "best_epoch": best_epoch,
            "best_score": best_score,
            "validations_without_improvement": validations_without_improvement,
            "initialization": initialization,
            "selected_batch_size": selected_batch_size,
            "patch_shape": PATCH_SHAPE.tolist(),
            "saved_at_utc": utc_now(),
        }
        atomic_torch_save(checkpoint_payload, last_path)
        if improved:
            atomic_torch_save(checkpoint_payload, best_path)

        print(
            f"  Epoch {epoch} complete — train loss={total_loss_sum/sample_count:.6f} — "
            f"best epoch={best_epoch} — best score={best_score:.6f} — "
            f"{epoch_seconds/60:.1f} min"
        )
        if should_validate:
            print(
                f"    lesion Dice={validation_summary['validation_PDAC_lesion_dice']:.5f}, "
                f"pancreas Dice={validation_summary['validation_pancreas_dice']:.5f}, "
                f"negative specificity={validation_summary['validation_negative_specificity_at_0_5']:.5f}"
            )

        if (
            epoch >= MIN_EPOCHS
            and validations_without_improvement >= EARLY_STOPPING_PATIENCE
        ):
            stopped_early = True
            print(f"  {arm}: early stopping after epoch {epoch}.")
            break

        del train_loader
        gc.collect()
        torch.cuda.empty_cache()

    if not best_path.exists():
        raise RuntimeError(f"{arm}: no best checkpoint was created.")

    completed_epoch = int(history["epoch"].max())
    training_complete = bool(
        completed_epoch >= MAX_EPOCHS or stopped_early
        or validations_without_improvement >= EARLY_STOPPING_PATIENCE
    )
    audit = {
        "stage": "5B",
        "arm": arm,
        "created_at_utc": utc_now(),
        "result": "PASS_ARM_TRAINING_COMPLETE" if training_complete else "INCOMPLETE",
        "training_complete": training_complete,
        "completed_epoch": completed_epoch,
        "best_epoch": int(best_epoch),
        "best_selection_score": float(best_score),
        "stopped_early": bool(stopped_early),
        "model_parameters": int(sum(p.numel() for p in model.parameters())),
        "best_checkpoint_path": str(best_path),
        "best_checkpoint_sha256": sha256_file(best_path),
        "last_checkpoint_path": str(last_path),
        "history_path": str(history_path),
        "validation_case_ledger_path": str(validation_case_path),
        "locked_test_cases_accessed": 0,
        "initialization": initialization,
    }
    atomic_write_json(audit, arm_audit_path)
    del model, optimizer, scheduler, scaler
    gc.collect()
    torch.cuda.empty_cache()
    return audit


# =============================================================================
# RUN BOTH ARMS SEQUENTIALLY; EACH ARM RESUMES INDEPENDENTLY
# =============================================================================


arm_audits = []
for arm in ARMS:
    arm_audits.append(train_one_arm(arm))

both_complete = all(audit.get("training_complete") is True for audit in arm_audits)
final_audit = {
    "stage": "5B",
    "created_at_utc": utc_now(),
    "result": "PASS_DUAL_ARM_TRAINING_COMPLETE" if both_complete else "INCOMPLETE",
    "training_complete": both_complete,
    "arms": arm_audits,
    "validation_monitor_subset_path": str(VALIDATION_SUBSET_PATH),
    "training_protocol_path": str(TRAINING_PROTOCOL_PATH),
    "locked_test_cases_accessed": 0,
}
atomic_write_json(final_audit, FINAL_AUDIT_PATH)

print("\n" + "=" * 120)
if both_complete:
    print("STAGE 5B RESULT: PASS — BOTH SUPERVISED TRAINING ARMS COMPLETE")
else:
    print("STAGE 5B STATUS: INCOMPLETE — RERUN THE SAME FILE TO RESUME")
print("=" * 120)
print("Final audit:")
print(FINAL_AUDIT_PATH)

if not both_complete:
    raise RuntimeError("Stage 5B is incomplete; rerun the same file to resume.")
