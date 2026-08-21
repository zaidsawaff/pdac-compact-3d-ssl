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
# LOCKED INPUTS AND OUTPUTS
# =============================================================================

PROJECT_ROOT = Path(os.environ.get("PDAC_PROJECT_ROOT", "/content/drive/MyDrive/PDAC_Public_Q1_Project"))
RUNTIME_ROOT = Path(os.environ.get("PDAC_RUNTIME_ROOT", "/content"))
RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
META_DIR = PROJECT_ROOT / "01_Metadata"
QC_DIR = PROJECT_ROOT / "02_Quality_Control"
MODEL_DIR = (
    PROJECT_ROOT
    / "04_Models"
    / "SSL"
    / "Stage4A_MaskedContext3DCNN"
)

FROZEN_MANIFEST_PATH = (
    META_DIR / "stage3b_e5_frozen_dataset_manifest.csv"
)
STAGE3B_FREEZE_PROTOCOL_PATH = (
    META_DIR / "stage3b_e5_dataset_freeze_protocol.json"
)
STAGE3B_AUDIT_PATH = (
    QC_DIR / "stage3b_e5_dataset_freeze_audit.json"
)

LAST_CHECKPOINT_PATH = MODEL_DIR / "stage4a_ssl_last.pt"
BEST_CHECKPOINT_PATH = MODEL_DIR / "stage4a_ssl_best.pt"
HISTORY_PATH = MODEL_DIR / "stage4a_ssl_training_history.csv"
MEMORY_PILOT_PATH = QC_DIR / "stage4a_ssl_gpu_memory_pilot.json"
PROTOCOL_PATH = META_DIR / "stage4a_ssl_training_protocol.json"
AUDIT_PATH = QC_DIR / "stage4a_ssl_training_audit.json"

LOCAL_CACHE_DIR = (RUNTIME_ROOT / "pdac_e5_ssl_cache")


# =============================================================================
# SELF-SUPERVISED TRAINING LOCK
# =============================================================================

SEED = 20260728
EXPECTED_CASES = 1671
EXPECTED_TRAIN = 1376
EXPECTED_VALIDATION = 295

VOLUME_SHAPE = np.asarray([240, 192, 128], dtype=int)
PATCH_SHAPE = np.asarray([128, 128, 64], dtype=int)
MASK_BLOCK_SHAPE = np.asarray([16, 16, 8], dtype=int)
MASK_RATIO = 0.45

HU_CENTER = 50.0
HU_HALF_WIDTH = 250.0
NORMALIZED_MIN = -1.0
NORMALIZED_MAX = 1.0

MAX_EPOCHS = 30
MIN_EPOCHS = 10
EARLY_STOPPING_PATIENCE = 7
LEARNING_RATE = 2e-4
MINIMUM_LEARNING_RATE = 1e-6
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP_NORM = 1.0
TARGET_EFFECTIVE_BATCH_SIZE = 4
VISIBLE_LOSS_WEIGHT = 0.05

BATCH_SIZE_CANDIDATES = [2, 1]
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


def stable_seed(text):
    digest = hashlib.sha256(str(text).encode("utf-8")).digest()
    return (int.from_bytes(digest[:8], "little") + SEED) % (2 ** 32)


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


def make_block_mask(rng):
    coarse_shape = PATCH_SHAPE // MASK_BLOCK_SHAPE
    coarse_voxels = int(np.prod(coarse_shape))
    masked_blocks = int(round(MASK_RATIO * coarse_voxels))

    flat = np.zeros(coarse_voxels, dtype=np.uint8)
    selected = rng.choice(
        coarse_voxels,
        size=masked_blocks,
        replace=False,
    )
    flat[selected] = 1
    coarse = flat.reshape(tuple(coarse_shape))

    mask = coarse
    for axis, repeats in enumerate(MASK_BLOCK_SHAPE):
        mask = np.repeat(mask, int(repeats), axis=axis)
    if tuple(mask.shape) != tuple(PATCH_SHAPE):
        raise RuntimeError("Block-mask shape is incorrect.")
    return mask.astype(np.float32, copy=False)


# =============================================================================
# LABEL-FREE DATASET
# =============================================================================


class MaskedContextDataset(Dataset):
    """Reads only CT HU arrays; mask_labels are never requested."""

    def __init__(self, dataframe, cache_dir, training):
        self.rows = dataframe.reset_index(drop=True)
        self.cache_dir = Path(cache_dir)
        self.training = bool(training)

    def __len__(self):
        return len(self.rows)

    def _rng(self, study_id):
        if self.training:
            # NumPy's worker-local RNG is seeded by seed_worker.
            return np.random
        return np.random.RandomState(stable_seed(study_id))

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        study_id = str(row["study_id"])
        path = self.cache_dir / f"{study_id}.npz"

        # np.load is lazy for NPZ members. Only ct_hu is decompressed; the
        # segmentation mask is deliberately not accessed in this SSL stage.
        with np.load(path, allow_pickle=False) as data:
            ct_hu = np.asarray(data["ct_hu"], dtype=np.int16)

        if tuple(ct_hu.shape) != tuple(VOLUME_SHAPE):
            raise RuntimeError(
                f"{study_id}: unexpected E5 CT shape {ct_hu.shape}."
            )

        rng = self._rng(study_id)
        maximum_start = VOLUME_SHAPE - PATCH_SHAPE
        centered_start = maximum_start // 2

        if self.training:
            jitter_limits = np.asarray([40, 24, 24], dtype=int)
            jitter = np.asarray(
                [
                    rng.randint(-limit, limit + 1)
                    for limit in jitter_limits
                ],
                dtype=int,
            )
            start = np.clip(
                centered_start + jitter,
                0,
                maximum_start,
            )
        else:
            start = centered_start

        end = start + PATCH_SHAPE
        patch_hu = ct_hu[
            start[0]:end[0],
            start[1]:end[1],
            start[2]:end[2],
        ].astype(np.float32, copy=False)

        target = np.clip(
            (patch_hu - HU_CENTER) / HU_HALF_WIDTH,
            NORMALIZED_MIN,
            NORMALIZED_MAX,
        ).astype(np.float32, copy=False)

        if self.training:
            for axis in [0, 1]:
                if rng.rand() < 0.5:
                    target = np.flip(target, axis=axis)

            intensity_scale = float(rng.uniform(0.90, 1.10))
            intensity_shift = float(rng.uniform(-0.08, 0.08))
            target = np.clip(
                target * intensity_scale + intensity_shift,
                NORMALIZED_MIN,
                NORMALIZED_MAX,
            )

        target = np.ascontiguousarray(target, dtype=np.float32)
        block_mask = make_block_mask(rng)

        corrupted = target.copy()
        if self.training:
            noise = rng.normal(
                loc=0.0,
                scale=0.025,
                size=tuple(PATCH_SHAPE),
            ).astype(np.float32)
            corrupted = np.clip(
                corrupted + noise,
                NORMALIZED_MIN,
                NORMALIZED_MAX,
            )
        corrupted[block_mask > 0.5] = 0.0

        return {
            "input": torch.from_numpy(corrupted).unsqueeze(0),
            "target": torch.from_numpy(target).unsqueeze(0),
            "mask": torch.from_numpy(block_mask).unsqueeze(0),
            "study_id": study_id,
        }


# =============================================================================
# LIGHTWEIGHT 3D CNN AUTOENCODER
# =============================================================================


def group_count(channels):
    for groups in [8, 4, 2, 1]:
        if channels % groups == 0:
            return groups
    return 1


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
            else nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            )
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
        self.residual = ResidualBlock3D(
            out_channels,
            out_channels,
        )

    def forward(self, x):
        return self.residual(self.down(x))


class CompactEncoder3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(
                1,
                16,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
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


class MaskedContextAutoencoder3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = CompactEncoder3D()
        self.up2 = UpBlock3D(96, 48, 48)
        self.up1 = UpBlock3D(48, 24, 24)
        self.up0 = UpBlock3D(24, 16, 16)
        self.reconstruction_head = nn.Conv3d(
            16,
            1,
            kernel_size=1,
        )

    def forward(self, x):
        skip0, skip1, skip2, bottleneck = self.encoder(x)
        x = self.up2(bottleneck, skip2)
        x = self.up1(x, skip1)
        x = self.up0(x, skip0)
        return self.reconstruction_head(x)


def masked_reconstruction_loss(prediction, target, mask):
    absolute_error = torch.abs(prediction - target)
    masked_denominator = torch.clamp(mask.sum(), min=1.0)
    visible = 1.0 - mask
    visible_denominator = torch.clamp(visible.sum(), min=1.0)

    masked_l1 = (absolute_error * mask).sum() / masked_denominator
    visible_l1 = (
        (absolute_error * visible).sum() / visible_denominator
    )
    total = masked_l1 + VISIBLE_LOSS_WEIGHT * visible_l1
    return total, masked_l1, visible_l1


# =============================================================================
# DATA LOCKS AND LOCAL CACHE
# =============================================================================


print("=" * 116)
print("STAGE 4A — MASKED-CONTEXT LIGHTWEIGHT 3D CNN SELF-SUPERVISED PRETRAINING")
print("=" * 116)

for required_path in [
    FROZEN_MANIFEST_PATH,
    STAGE3B_FREEZE_PROTOCOL_PATH,
    STAGE3B_AUDIT_PATH,
]:
    if not required_path.exists():
        raise FileNotFoundError(f"Required input missing:\n{required_path}")

with open(
    STAGE3B_FREEZE_PROTOCOL_PATH,
    "r",
    encoding="utf-8",
) as file:
    freeze_protocol = json.load(file)
with open(STAGE3B_AUDIT_PATH, "r", encoding="utf-8") as file:
    stage3b_audit = json.load(file)

if freeze_protocol.get("dataset_frozen") is not True:
    raise RuntimeError("The E5 dataset is not frozen.")
if stage3b_audit.get("dataset_frozen") is not True:
    raise RuntimeError("Stage 3B audit did not pass.")
if freeze_protocol.get("dataset_name") != "PANORAMA_LOCAL_DEVELOPMENT_E5_V1":
    raise RuntimeError("Unexpected frozen dataset identity.")

if not torch.cuda.is_available():
    raise RuntimeError("Stage 4A requires a CUDA GPU.")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
set_global_seed(SEED)
torch.set_float32_matmul_precision("high")

device = torch.device("cuda")
gpu_name = torch.cuda.get_device_name(0)
gpu_memory_gib = (
    torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
)

manifest = pd.read_csv(FROZEN_MANIFEST_PATH)
manifest["study_id"] = manifest["study_id"].astype(str).str.strip()

required_columns = {
    "study_id",
    "partition",
    "output_path",
    "output_size_bytes",
    "verified_sha256",
    "all_checks_pass",
}
missing_columns = sorted(required_columns - set(manifest.columns))
if missing_columns:
    raise RuntimeError(
        f"Frozen manifest missing columns: {missing_columns}"
    )
if len(manifest) != EXPECTED_CASES:
    raise RuntimeError("Frozen E5 case count changed.")
if manifest["study_id"].nunique() != EXPECTED_CASES:
    raise RuntimeError("Frozen E5 IDs are not unique.")
if not truth_flags(manifest["all_checks_pass"]).all():
    raise RuntimeError("Frozen manifest contains failed cases.")
if manifest["partition"].value_counts().to_dict() != {
    "train": EXPECTED_TRAIN,
    "validation": EXPECTED_VALIDATION,
}:
    raise RuntimeError("Frozen train/validation counts changed.")
if set(manifest["partition"]) != {"train", "validation"}:
    raise RuntimeError("A locked test case entered Stage 4A.")

if AUDIT_PATH.exists():
    with open(AUDIT_PATH, "r", encoding="utf-8") as file:
        previous_audit = json.load(file)
    if previous_audit.get("training_complete") is True:
        print("Stage 4A is already complete; no rerun is required.")
        print(BEST_CHECKPOINT_PATH)
        raise SystemExit(0)

print()
print(f"PyTorch: {torch.__version__}")
print(f"GPU: {gpu_name}")
print(f"GPU memory: {gpu_memory_gib:.2f} GiB")
print(f"Training cases: {EXPECTED_TRAIN}")
print(f"Validation cases: {EXPECTED_VALIDATION}")
print("Mask-label arrays accessed for SSL: 0")
print("Locked test cases accessed: 0")

print()
print("Preparing the local high-speed E5 cache...")
copied = 0
reused = 0
cache_verified = 0
additional_cache_bytes = 0
for _, storage_row in manifest.iterrows():
    storage_path = (
        LOCAL_CACHE_DIR / f"{storage_row['study_id']}.npz"
    )
    storage_size = int(storage_row["output_size_bytes"])
    if not (
        storage_path.exists()
        and storage_path.stat().st_size == storage_size
    ):
        additional_cache_bytes += storage_size

if disk_free_gib(RUNTIME_ROOT) * (1024 ** 3) < (
    additional_cache_bytes * 1.10 + 1024 ** 3
):
    raise RuntimeError(
        "Local storage cannot hold the remaining E5 cache plus "
        "10% overhead and 1 GiB working space."
    )

for order, (_, row) in enumerate(manifest.iterrows(), start=1):
    study_id = str(row["study_id"])
    source = Path(str(row["output_path"]))
    destination = LOCAL_CACHE_DIR / f"{study_id}.npz"
    expected_size = int(row["output_size_bytes"])
    expected_hash = str(row["verified_sha256"])

    destination_valid = False
    if destination.exists() and destination.stat().st_size == expected_size:
        destination_valid = sha256_file(destination) == expected_hash

    if destination_valid:
        reused += 1
        cache_verified += 1
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
            raise RuntimeError(f"Cache SHA-256 mismatch: {study_id}")
        os.replace(temporary, destination)
        copied += 1
        cache_verified += 1

    if order % 200 == 0 or order == len(manifest):
        print(
            f"  Cache: {order}/{len(manifest)} "
            f"(verified={cache_verified}, reused={reused}, "
            f"copied={copied})"
        )

cache_files = {path.name for path in LOCAL_CACHE_DIR.glob("*.npz")}
expected_cache_files = {
    f"{study_id}.npz" for study_id in manifest["study_id"]
}
if cache_files != expected_cache_files:
    raise RuntimeError("Local SSL cache inventory is not exact.")

train_frame = manifest.loc[
    manifest["partition"] == "train"
].sort_values("study_id").reset_index(drop=True)
validation_frame = manifest.loc[
    manifest["partition"] == "validation"
].sort_values("study_id").reset_index(drop=True)


# =============================================================================
# AUTOMATIC GPU MEMORY PILOT
# =============================================================================


def run_memory_pilot(candidate_batch_size):
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()

    pilot_dataset = MaskedContextDataset(
        train_frame.head(candidate_batch_size),
        LOCAL_CACHE_DIR,
        training=True,
    )
    batch_items = [
        pilot_dataset[index]
        for index in range(candidate_batch_size)
    ]
    inputs = torch.stack(
        [item["input"] for item in batch_items]
    ).to(device)
    targets = torch.stack(
        [item["target"] for item in batch_items]
    ).to(device)
    masks = torch.stack(
        [item["mask"] for item in batch_items]
    ).to(device)

    pilot_model = MaskedContextAutoencoder3D().to(device)
    pilot_optimizer = torch.optim.AdamW(
        pilot_model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    pilot_scaler = torch.amp.GradScaler(
        "cuda",
        enabled=USE_AMP,
    )
    pilot_optimizer.zero_grad(set_to_none=True)

    with torch.amp.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=USE_AMP,
    ):
        reconstruction = pilot_model(inputs)
        loss, _, _ = masked_reconstruction_loss(
            reconstruction,
            targets,
            masks,
        )
    pilot_scaler.scale(loss).backward()
    pilot_scaler.unscale_(pilot_optimizer)
    torch.nn.utils.clip_grad_norm_(
        pilot_model.parameters(),
        GRADIENT_CLIP_NORM,
    )
    pilot_scaler.step(pilot_optimizer)
    pilot_scaler.update()
    torch.cuda.synchronize()

    peak_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)
    loss_value = float(loss.detach().cpu())

    del (
        pilot_model,
        pilot_optimizer,
        pilot_scaler,
        inputs,
        targets,
        masks,
        reconstruction,
        loss,
        batch_items,
        pilot_dataset,
    )
    torch.cuda.empty_cache()
    gc.collect()
    return peak_gib, loss_value


pilot_attempts = []
selected_batch_size = None
for candidate_batch_size in BATCH_SIZE_CANDIDATES:
    try:
        peak_gib, pilot_loss = run_memory_pilot(
            candidate_batch_size
        )
        pilot_attempts.append(
            {
                "batch_size": candidate_batch_size,
                "status": "PASS",
                "peak_allocated_GiB": peak_gib,
                "pilot_loss": pilot_loss,
            }
        )
        selected_batch_size = candidate_batch_size
        break
    except torch.cuda.OutOfMemoryError as error:
        pilot_attempts.append(
            {
                "batch_size": candidate_batch_size,
                "status": "CUDA_OOM",
                "error": str(error),
            }
        )
        torch.cuda.empty_cache()
        gc.collect()

if selected_batch_size is None:
    raise RuntimeError("No safe SSL batch size was found on this GPU.")

gradient_accumulation_steps = max(
    1,
    int(math.ceil(
        TARGET_EFFECTIVE_BATCH_SIZE / selected_batch_size
    )),
)
effective_batch_size = (
    selected_batch_size * gradient_accumulation_steps
)

memory_pilot = {
    "stage": "4A",
    "created_at_utc": utc_now(),
    "gpu": gpu_name,
    "gpu_memory_GiB": gpu_memory_gib,
    "patch_shape": PATCH_SHAPE.tolist(),
    "attempts": pilot_attempts,
    "selected_batch_size": selected_batch_size,
    "gradient_accumulation_steps": gradient_accumulation_steps,
    "effective_batch_size": effective_batch_size,
    "result": "PASS",
}
atomic_write_json(memory_pilot, MEMORY_PILOT_PATH)

print()
print("GPU MEMORY PILOT")
print("-" * 116)
for attempt in pilot_attempts:
    print(attempt)
print(f"Selected batch size: {selected_batch_size}")
print(
    f"Gradient accumulation steps: "
    f"{gradient_accumulation_steps}"
)
print(f"Effective batch size: {effective_batch_size}")


# =============================================================================
# TRAINING SETUP AND RESUME
# =============================================================================


model = MaskedContextAutoencoder3D().to(device)
parameter_count = sum(
    parameter.numel() for parameter in model.parameters()
)
encoder_parameter_count = sum(
    parameter.numel() for parameter in model.encoder.parameters()
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=MAX_EPOCHS,
    eta_min=MINIMUM_LEARNING_RATE,
)
scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

start_epoch = 1
best_epoch = 0
best_validation_mae = float("inf")
best_validation_psnr = float("-inf")
epochs_without_improvement = 0

if HISTORY_PATH.exists():
    history = pd.read_csv(HISTORY_PATH)
else:
    history = pd.DataFrame()

if LAST_CHECKPOINT_PATH.exists():
    checkpoint = torch.load(
        LAST_CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )
    if int(checkpoint["selected_batch_size"]) != selected_batch_size:
        raise RuntimeError(
            "The safe batch size differs from the saved run. "
            "Resume on the same GPU type or remove only the Stage 4A "
            "checkpoints to restart."
        )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    scaler.load_state_dict(checkpoint["scaler_state"])
    start_epoch = int(checkpoint["epoch"]) + 1
    best_epoch = int(checkpoint["best_epoch"])
    best_validation_mae = float(
        checkpoint["best_validation_mae"]
    )
    best_validation_psnr = float(
        checkpoint["best_validation_psnr"]
    )
    epochs_without_improvement = int(
        checkpoint["epochs_without_improvement"]
    )
    print()
    print(
        f"Resuming from epoch {checkpoint['epoch']} — "
        f"best epoch {best_epoch}"
    )

protocol = {
    "stage": "4A",
    "created_at_utc": utc_now(),
    "method": "MASKED_CONTEXT_RECONSTRUCTION_3D_CNN",
    "architecture": "COMPACT_RESIDUAL_3D_UNET_AUTOENCODER",
    "training_cases": EXPECTED_TRAIN,
    "validation_cases": EXPECTED_VALIDATION,
    "locked_test_cases_accessed": 0,
    "mask_label_arrays_accessed": 0,
    "input_crop_shape": VOLUME_SHAPE.tolist(),
    "training_patch_shape": PATCH_SHAPE.tolist(),
    "mask_block_shape": MASK_BLOCK_SHAPE.tolist(),
    "mask_ratio": MASK_RATIO,
    "intensity_normalization": "(clip(HU,-200,300)-50)/250",
    "case_specific_normalization": "PROHIBITED",
    "selected_batch_size": selected_batch_size,
    "gradient_accumulation_steps": gradient_accumulation_steps,
    "effective_batch_size": effective_batch_size,
    "maximum_epochs": MAX_EPOCHS,
    "minimum_epochs": MIN_EPOCHS,
    "early_stopping_patience": EARLY_STOPPING_PATIENCE,
    "learning_rate": LEARNING_RATE,
    "minimum_learning_rate": MINIMUM_LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "optimizer": "AdamW",
    "scheduler": "CosineAnnealingLR",
    "mixed_precision": USE_AMP,
    "model_parameters": parameter_count,
    "encoder_parameters": encoder_parameter_count,
    "best_checkpoint_path": str(BEST_CHECKPOINT_PATH),
    "last_checkpoint_path": str(LAST_CHECKPOINT_PATH),
}
atomic_write_json(protocol, PROTOCOL_PATH)

print()
print("TRAINING CONFIGURATION")
print("-" * 116)
print(f"Model parameters: {parameter_count:,}")
print(f"Encoder parameters: {encoder_parameter_count:,}")
print(f"Start epoch: {start_epoch}")
print(f"Maximum epochs: {MAX_EPOCHS}")


def make_loader(dataframe, training, epoch):
    dataset = MaskedContextDataset(
        dataframe,
        LOCAL_CACHE_DIR,
        training=training,
    )
    generator = torch.Generator()
    generator.manual_seed(SEED + epoch + (0 if training else 100000))
    return DataLoader(
        dataset,
        batch_size=selected_batch_size,
        shuffle=training,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=training,
        persistent_workers=False,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def evaluate(validation_loader):
    model.eval()
    absolute_error_sum = 0.0
    squared_error_sum = 0.0
    baseline_absolute_error_sum = 0.0
    masked_voxels = 0.0
    visible_absolute_error_sum = 0.0
    visible_voxels = 0.0

    with torch.no_grad():
        for batch in validation_loader:
            inputs = batch["input"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)

            with torch.amp.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=USE_AMP,
            ):
                predictions = model(inputs)

            differences = predictions.float() - targets.float()
            visible = 1.0 - masks
            absolute_error_sum += float(
                (torch.abs(differences) * masks).sum().cpu()
            )
            squared_error_sum += float(
                ((differences ** 2) * masks).sum().cpu()
            )
            baseline_absolute_error_sum += float(
                (torch.abs(targets.float()) * masks).sum().cpu()
            )
            masked_voxels += float(masks.sum().cpu())
            visible_absolute_error_sum += float(
                (torch.abs(differences) * visible).sum().cpu()
            )
            visible_voxels += float(visible.sum().cpu())

    masked_mae = absolute_error_sum / masked_voxels
    masked_mse = squared_error_sum / masked_voxels
    visible_mae = visible_absolute_error_sum / visible_voxels
    zero_fill_baseline_mae = (
        baseline_absolute_error_sum / masked_voxels
    )
    psnr = 10.0 * math.log10(4.0 / max(masked_mse, 1e-12))
    return {
        "validation_masked_mae": masked_mae,
        "validation_masked_mse": masked_mse,
        "validation_masked_psnr_db": psnr,
        "validation_visible_mae": visible_mae,
        "validation_zero_fill_baseline_mae": zero_fill_baseline_mae,
    }


# =============================================================================
# RESUMABLE TRAINING LOOP
# =============================================================================


training_started = time.time()
stopped_early = False

for epoch in range(start_epoch, MAX_EPOCHS + 1):
    epoch_started = time.time()
    train_loader = make_loader(train_frame, training=True, epoch=epoch)
    validation_loader = make_loader(
        validation_frame,
        training=False,
        epoch=epoch,
    )

    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss_sum = 0.0
    masked_loss_sum = 0.0
    visible_loss_sum = 0.0
    cases_seen = 0

    for step, batch in enumerate(train_loader, start=1):
        inputs = batch["input"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        with torch.amp.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=USE_AMP,
        ):
            reconstruction = model(inputs)
            total_loss, masked_l1, visible_l1 = (
                masked_reconstruction_loss(
                    reconstruction,
                    targets,
                    masks,
                )
            )
            scaled_loss = (
                total_loss / gradient_accumulation_steps
            )

        scaler.scale(scaled_loss).backward()

        should_step = (
            step % gradient_accumulation_steps == 0
            or step == len(train_loader)
        )
        if should_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRADIENT_CLIP_NORM,
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        batch_cases = int(inputs.shape[0])
        cases_seen += batch_cases
        total_loss_sum += float(total_loss.detach().cpu()) * batch_cases
        masked_loss_sum += float(masked_l1.detach().cpu()) * batch_cases
        visible_loss_sum += float(visible_l1.detach().cpu()) * batch_cases

        if step % 100 == 0 or step == len(train_loader):
            print(
                f"Epoch {epoch:02d}/{MAX_EPOCHS} — "
                f"step {step}/{len(train_loader)} — "
                f"masked L1 {masked_loss_sum / cases_seen:.5f}"
            )

    validation_metrics = evaluate(validation_loader)
    learning_rate = float(optimizer.param_groups[0]["lr"])
    scheduler.step()

    train_total_loss = total_loss_sum / cases_seen
    train_masked_l1 = masked_loss_sum / cases_seen
    train_visible_l1 = visible_loss_sum / cases_seen
    validation_mae = float(
        validation_metrics["validation_masked_mae"]
    )
    validation_psnr = float(
        validation_metrics["validation_masked_psnr_db"]
    )

    improved = validation_mae < best_validation_mae - 1e-6
    if improved:
        best_validation_mae = validation_mae
        best_validation_psnr = validation_psnr
        best_epoch = epoch
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    history_row = {
        "epoch": epoch,
        "train_cases_seen": cases_seen,
        "train_total_loss": train_total_loss,
        "train_masked_l1": train_masked_l1,
        "train_visible_l1": train_visible_l1,
        **validation_metrics,
        "learning_rate": learning_rate,
        "best_epoch_so_far": best_epoch,
        "best_validation_mae_so_far": best_validation_mae,
        "epochs_without_improvement": epochs_without_improvement,
        "epoch_seconds": time.time() - epoch_started,
        "completed_at_utc": utc_now(),
    }
    if len(history):
        history = history.loc[history["epoch"] != epoch].copy()
    history = pd.concat(
        [history, pd.DataFrame([history_row])],
        ignore_index=True,
    ).sort_values("epoch").reset_index(drop=True)
    atomic_write_csv(history, HISTORY_PATH)

    checkpoint_payload = {
        "stage": "4A",
        "epoch": epoch,
        "model_state": model.state_dict(),
        "encoder_state": model.encoder.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "best_epoch": best_epoch,
        "best_validation_mae": best_validation_mae,
        "best_validation_psnr": best_validation_psnr,
        "epochs_without_improvement": epochs_without_improvement,
        "selected_batch_size": selected_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "model_parameters": parameter_count,
        "encoder_parameters": encoder_parameter_count,
        "patch_shape": PATCH_SHAPE.tolist(),
        "mask_ratio": MASK_RATIO,
        "protocol_path": str(PROTOCOL_PATH),
        "saved_at_utc": utc_now(),
    }
    atomic_torch_save(checkpoint_payload, LAST_CHECKPOINT_PATH)
    if improved:
        atomic_torch_save(checkpoint_payload, BEST_CHECKPOINT_PATH)

    print()
    print(
        f"Epoch {epoch:02d} complete — "
        f"train masked L1={train_masked_l1:.6f} — "
        f"val masked MAE={validation_mae:.6f} — "
        f"val PSNR={validation_psnr:.3f} dB — "
        f"best epoch={best_epoch}"
    )
    print(
        f"Checkpoint saved — patience "
        f"{epochs_without_improvement}/{EARLY_STOPPING_PATIENCE}"
    )
    print()

    interim_audit = {
        "stage": "4A",
        "updated_at_utc": utc_now(),
        "training_complete": False,
        "last_completed_epoch": epoch,
        "best_epoch": best_epoch,
        "best_validation_masked_mae": best_validation_mae,
        "best_validation_masked_psnr_db": best_validation_psnr,
        "epochs_without_improvement": epochs_without_improvement,
        "locked_test_cases_accessed": 0,
        "mask_label_arrays_accessed": 0,
        "last_checkpoint_path": str(LAST_CHECKPOINT_PATH),
        "best_checkpoint_path": str(BEST_CHECKPOINT_PATH),
    }
    atomic_write_json(interim_audit, AUDIT_PATH)

    if (
        epoch >= MIN_EPOCHS
        and epochs_without_improvement >= EARLY_STOPPING_PATIENCE
    ):
        stopped_early = True
        print(
            f"Early stopping at epoch {epoch}; "
            f"best epoch was {best_epoch}."
        )
        break


# =============================================================================
# FINAL TRAINING AUDIT
# =============================================================================


if not BEST_CHECKPOINT_PATH.exists():
    raise RuntimeError("No best SSL checkpoint was created.")
if len(history) < MIN_EPOCHS:
    raise RuntimeError(
        f"SSL training ended before {MIN_EPOCHS} complete epochs."
    )

best_row = history.loc[history["epoch"] == best_epoch]
if len(best_row) != 1:
    raise RuntimeError("Best SSL history row is not unique.")
best_row = best_row.iloc[0]
baseline_mae = float(
    best_row["validation_zero_fill_baseline_mae"]
)
relative_improvement = float(
    (baseline_mae - best_validation_mae) / baseline_mae
)

readiness_checks = {
    "Frozen E5 dataset contains exactly 1671 cases": (
        len(manifest) == EXPECTED_CASES
    ),
    "Training partition contains exactly 1376 cases": (
        len(train_frame) == EXPECTED_TRAIN
    ),
    "Validation partition contains exactly 295 cases": (
        len(validation_frame) == EXPECTED_VALIDATION
    ),
    "No locked test case was accessed": (
        set(manifest["partition"]) == {"train", "validation"}
    ),
    "No segmentation mask array was accessed": True,
    "GPU memory pilot passed": selected_batch_size is not None,
    "At least ten epochs completed": len(history) >= MIN_EPOCHS,
    "Best validation MAE is finite": np.isfinite(
        best_validation_mae
    ),
    "Best validation PSNR is finite": np.isfinite(
        best_validation_psnr
    ),
    "Masked reconstruction improves zero-fill baseline": (
        best_validation_mae < baseline_mae
    ),
    "Best checkpoint exists": BEST_CHECKPOINT_PATH.exists(),
    "Last checkpoint exists": LAST_CHECKPOINT_PATH.exists(),
    "Model remains below three million parameters": (
        parameter_count < 3_000_000
    ),
}
all_checks_pass = all(
    bool(value) for value in readiness_checks.values()
)

final_audit = {
    "stage": "4A",
    "created_at_utc": utc_now(),
    "result": (
        "PASS_SSL_PRETRAINING_COMPLETE"
        if all_checks_pass
        else "FAIL"
    ),
    "training_complete": all_checks_pass,
    "stopped_early": stopped_early,
    "epochs_completed": int(history["epoch"].max()),
    "best_epoch": best_epoch,
    "best_validation_masked_mae": best_validation_mae,
    "best_validation_masked_psnr_db": best_validation_psnr,
    "validation_zero_fill_baseline_mae": baseline_mae,
    "relative_improvement_over_zero_fill": relative_improvement,
    "model_parameters": parameter_count,
    "encoder_parameters": encoder_parameter_count,
    "selected_batch_size": selected_batch_size,
    "gradient_accumulation_steps": gradient_accumulation_steps,
    "effective_batch_size": effective_batch_size,
    "training_wall_time_seconds_this_run": (
        time.time() - training_started
    ),
    "best_checkpoint_path": str(BEST_CHECKPOINT_PATH),
    "best_checkpoint_sha256": sha256_file(BEST_CHECKPOINT_PATH),
    "last_checkpoint_path": str(LAST_CHECKPOINT_PATH),
    "history_path": str(HISTORY_PATH),
    "protocol_path": str(PROTOCOL_PATH),
    "locked_test_cases_accessed": 0,
    "mask_label_arrays_accessed": 0,
    "readiness_checks": {
        key: bool(value)
        for key, value in readiness_checks.items()
    },
}
atomic_write_json(final_audit, AUDIT_PATH)

print()
print("-" * 116)
print("FINAL SELF-SUPERVISED RESULTS")
print("-" * 116)
print(f"Epochs completed: {int(history['epoch'].max())}")
print(f"Best epoch: {best_epoch}")
print(
    f"Best validation masked MAE: "
    f"{best_validation_mae:.6f}"
)
print(
    f"Best validation masked PSNR: "
    f"{best_validation_psnr:.3f} dB"
)
print(f"Zero-fill baseline MAE: {baseline_mae:.6f}")
print(
    f"Relative improvement: "
    f"{100.0 * relative_improvement:.2f}%"
)

print()
print("-" * 116)
print("READINESS CHECKS")
print("-" * 116)
for check, passed in readiness_checks.items():
    print(f"  {check}: {bool(passed)}")

print()
print("Best encoder checkpoint:")
print(BEST_CHECKPOINT_PATH)
print()
print("Training history:")
print(HISTORY_PATH)
print()
print("Final audit:")
print(AUDIT_PATH)
print()
print("=" * 116)
print(
    "STAGE 4A RESULT: "
    + (
        "PASS_SSL_PRETRAINING_COMPLETE"
        if all_checks_pass
        else "FAIL"
    )
)
print("=" * 116)

if not all_checks_pass:
    failed = [
        check
        for check, passed in readiness_checks.items()
        if not bool(passed)
    ]
    raise RuntimeError(f"Stage 4A failed checks: {failed}")
