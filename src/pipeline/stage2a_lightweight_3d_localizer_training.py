from pathlib import Path
from datetime import datetime, timezone
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# =============================================================================
# LOCKED CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(os.environ.get("PDAC_PROJECT_ROOT", "/content/drive/MyDrive/PDAC_Public_Q1_Project"))
RUNTIME_ROOT = Path(os.environ.get("PDAC_RUNTIME_ROOT", "/content"))
RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
META_DIR = PROJECT_ROOT / "01_Metadata"
QC_DIR = PROJECT_ROOT / "02_Quality_Control"
MODEL_DIR = (
    PROJECT_ROOT
    / "04_Models"
    / "Localizer"
    / "F1_96x96x160"
)

MANIFEST_PATH = (
    META_DIR / "stage1b_localizer_preprocessed_manifest.csv"
)
FREEZE_PATH = (
    META_DIR / "stage1b_localizer_dataset_freeze.json"
)
STAGE1B_AUDIT_PATH = (
    QC_DIR / "stage1b_localizer_preprocessing_audit.json"
)

LOCAL_CACHE_DIR = (RUNTIME_ROOT / "pdac_stage2a_localizer_cache")

LATEST_CHECKPOINT_PATH = MODEL_DIR / "stage2a_localizer_latest.pt"
BEST_CHECKPOINT_PATH = MODEL_DIR / "stage2a_localizer_best.pt"
HISTORY_PATH = MODEL_DIR / "stage2a_localizer_training_history.csv"
PREDICTIONS_PATH = (
    MODEL_DIR / "stage2a_localizer_validation_predictions.csv"
)
PROTOCOL_PATH = MODEL_DIR / "stage2a_localizer_training_protocol.json"
AUDIT_PATH = MODEL_DIR / "stage2a_localizer_training_audit.json"

SEED = 20260728
EXPECTED_TRAIN = 1376
EXPECTED_VALIDATION = 295
INPUT_SHAPE = (96, 96, 160)
HEATMAP_SHAPE = (12, 12, 20)
HU_CENTER = 50.0
HU_HALF_WIDTH = 250.0

BATCH_SIZE = 8
NUM_WORKERS = 2
MAX_EPOCHS = 60
EARLY_STOPPING_PATIENCE = 12
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP_NORM = 5.0
HEATMAP_SIGMA_CELLS = 1.5

CENTER_LOSS_WEIGHT = 1.0
HEATMAP_LOSS_WEIGHT = 0.25
BBOX_LOSS_WEIGHT = 0.10


# =============================================================================
# UTILITIES
# =============================================================================

def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_write_csv(dataframe, path):
    temporary_path = Path(str(path) + ".tmp")
    dataframe.to_csv(temporary_path, index=False)
    os.replace(temporary_path, path)


def atomic_write_json(data, path):
    temporary_path = Path(str(path) + ".tmp")
    with open(temporary_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
    os.replace(temporary_path, path)


def atomic_torch_save(payload, path):
    temporary_path = Path(str(path) + ".tmp")
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)


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


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def parse_vector(value):
    vector = np.asarray(json.loads(str(value)), dtype=np.float32)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise RuntimeError(f"Invalid target vector: {value}")
    return vector


def capture_rng_state(train_generator):
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "train_generator": train_generator.get_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state, train_generator):
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    train_generator.set_state(state["train_generator"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


# =============================================================================
# DATASET
# =============================================================================

class LocalizerDataset(Dataset):
    def __init__(self, dataframe, cache_dir, training):
        self.rows = dataframe.reset_index(drop=True).copy()
        self.cache_dir = Path(cache_dir)
        self.training = bool(training)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        study_id = str(row["study_id"])
        path = self.cache_dir / f"{study_id}.npz"

        with np.load(path, allow_pickle=False) as data:
            ct_hu = np.asarray(data["ct_hu"], dtype=np.float32)
            center = np.asarray(
                data["pancreas_center_normalized"],
                dtype=np.float32,
            )
            bbox_size = np.asarray(
                data["pancreas_bbox_size_voxels"],
                dtype=np.float32,
            )

        if ct_hu.shape != INPUT_SHAPE:
            raise RuntimeError(
                f"{study_id}: invalid CT shape {ct_hu.shape}"
            )

        image = torch.from_numpy(
            (ct_hu - HU_CENTER) / HU_HALF_WIDTH
        ).unsqueeze(0)
        image = torch.clamp(image, -1.0, 1.0)

        if self.training:
            intensity_scale = torch.empty(1).uniform_(0.95, 1.05)
            intensity_bias = torch.empty(1).uniform_(-0.03, 0.03)
            image = image * intensity_scale + intensity_bias
            image = image + torch.randn_like(image) * 0.005
            image = torch.clamp(image, -1.0, 1.0)

        center_tensor = torch.from_numpy(center)
        bbox_tensor = torch.from_numpy(
            np.clip(
                bbox_size / np.asarray(INPUT_SHAPE, dtype=np.float32),
                0.0,
                1.0,
            )
        )

        return {
            "image": image,
            "center": center_tensor,
            "bbox_size": bbox_tensor,
            "effective_spacing_mm": torch.tensor(
                float(row["effective_isotropic_spacing_mm"]),
                dtype=torch.float32,
            ),
            "study_id": study_id,
        }


# =============================================================================
# LIGHTWEIGHT SPATIAL 3D LOCALIZER
# =============================================================================

def group_count(channels):
    for groups in [8, 4, 2, 1]:
        if channels % groups == 0:
            return groups
    return 1


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.main = nn.Sequential(
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
            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(group_count(out_channels), out_channels),
        )
        self.skip = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=2,
            bias=False,
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.activation(self.main(x) + self.skip(x))


class DilatedResidualBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv3d(
                channels,
                channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.GroupNorm(group_count(channels), channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(
                channels,
                channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.GroupNorm(group_count(channels), channels),
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.activation(x + self.main(x))


class LightweightSpatialLocalizer3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            DownBlock(1, 8),
            DownBlock(8, 16),
            DownBlock(16, 32),
            nn.Conv3d(32, 48, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(group_count(48), 48),
            nn.SiLU(inplace=True),
            DilatedResidualBlock(48, dilation=1),
            DilatedResidualBlock(48, dilation=2),
            DilatedResidualBlock(48, dilation=4),
        )
        self.heatmap_head = nn.Conv3d(48, 1, kernel_size=1)
        self.bbox_head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(48, 32),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.10),
            nn.Linear(32, 3),
            nn.Sigmoid(),
        )

        axes = [
            torch.linspace(0.0, 1.0, steps=size)
            for size in HEATMAP_SHAPE
        ]
        grid = torch.stack(
            torch.meshgrid(*axes, indexing="ij"),
            dim=0,
        )
        self.register_buffer(
            "coordinate_grid",
            grid.unsqueeze(0),
            persistent=False,
        )

    def forward(self, image):
        features = self.encoder(image)
        if tuple(features.shape[2:]) != HEATMAP_SHAPE:
            raise RuntimeError(
                f"Unexpected heatmap shape: {features.shape[2:]}"
            )

        logits = self.heatmap_head(features)
        probabilities = torch.softmax(
            logits.flatten(start_dim=2),
            dim=-1,
        ).reshape_as(logits)
        center = torch.sum(
            probabilities * self.coordinate_grid,
            dim=(2, 3, 4),
        )
        bbox_size = self.bbox_head(features)
        return center, bbox_size, logits


def build_target_heatmap(center, coordinate_grid):
    heatmap_shape = torch.tensor(
        HEATMAP_SHAPE,
        dtype=center.dtype,
        device=center.device,
    )
    sigma_normalized = (
        HEATMAP_SIGMA_CELLS
        / torch.clamp(heatmap_shape - 1.0, min=1.0)
    )
    difference = (
        coordinate_grid
        - center[:, :, None, None, None]
    ) / sigma_normalized[None, :, None, None, None]
    target = torch.exp(
        -0.5 * torch.sum(difference ** 2, dim=1, keepdim=True)
    )
    target = target / torch.clamp(
        target.sum(dim=(2, 3, 4), keepdim=True),
        min=1e-8,
    )
    return target


def compute_loss(
    predicted_center,
    predicted_bbox,
    heatmap_logits,
    target_center,
    target_bbox,
    coordinate_grid,
):
    center_loss = F.smooth_l1_loss(
        predicted_center,
        target_center,
        beta=0.03,
    )
    bbox_loss = F.smooth_l1_loss(
        predicted_bbox,
        target_bbox,
        beta=0.05,
    )
    target_heatmap = build_target_heatmap(
        target_center,
        coordinate_grid,
    )
    log_probabilities = F.log_softmax(
        heatmap_logits.flatten(start_dim=2),
        dim=-1,
    ).reshape_as(heatmap_logits)
    heatmap_loss = -torch.mean(
        torch.sum(
            target_heatmap * log_probabilities,
            dim=(2, 3, 4),
        )
    )
    total_loss = (
        CENTER_LOSS_WEIGHT * center_loss
        + HEATMAP_LOSS_WEIGHT * heatmap_loss
        + BBOX_LOSS_WEIGHT * bbox_loss
    )
    return total_loss, center_loss, heatmap_loss, bbox_loss


# =============================================================================
# TRAINING AND EVALUATION
# =============================================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    amp_enabled,
):
    model.train()
    totals = {
        "loss": 0.0,
        "center": 0.0,
        "heatmap": 0.0,
        "bbox": 0.0,
        "cases": 0,
    }

    for batch_index, batch in enumerate(loader, start=1):
        image = batch["image"].to(
            device,
            non_blocking=True,
        )
        target_center = batch["center"].to(
            device,
            non_blocking=True,
        )
        target_bbox = batch["bbox_size"].to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(
            device_type="cuda",
            enabled=amp_enabled,
        ):
            predicted_center, predicted_bbox, heatmap_logits = model(
                image
            )
            loss, center_loss, heatmap_loss, bbox_loss = compute_loss(
                predicted_center,
                predicted_bbox,
                heatmap_logits,
                target_center,
                target_bbox,
                model.coordinate_grid,
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRADIENT_CLIP_NORM,
        )
        scaler.step(optimizer)
        scaler.update()

        batch_cases = int(image.shape[0])
        totals["loss"] += float(loss.detach()) * batch_cases
        totals["center"] += float(center_loss.detach()) * batch_cases
        totals["heatmap"] += float(heatmap_loss.detach()) * batch_cases
        totals["bbox"] += float(bbox_loss.detach()) * batch_cases
        totals["cases"] += batch_cases

        if batch_index % 50 == 0:
            print(
                f"      Training batch {batch_index}/{len(loader)}"
            )

    return {
        key: totals[key] / totals["cases"]
        for key in ["loss", "center", "heatmap", "bbox"]
    }


@torch.no_grad()
def evaluate(model, loader, device, amp_enabled):
    model.eval()
    rows = []
    loss_totals = {
        "loss": 0.0,
        "center": 0.0,
        "heatmap": 0.0,
        "bbox": 0.0,
        "cases": 0,
    }
    shape_minus_one = np.asarray(INPUT_SHAPE, dtype=np.float32) - 1.0

    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        target_center = batch["center"].to(
            device,
            non_blocking=True,
        )
        target_bbox = batch["bbox_size"].to(
            device,
            non_blocking=True,
        )

        with torch.amp.autocast(
            device_type="cuda",
            enabled=amp_enabled,
        ):
            predicted_center, predicted_bbox, heatmap_logits = model(
                image
            )
            loss, center_loss, heatmap_loss, bbox_loss = compute_loss(
                predicted_center,
                predicted_bbox,
                heatmap_logits,
                target_center,
                target_bbox,
                model.coordinate_grid,
            )

        batch_cases = int(image.shape[0])
        loss_totals["loss"] += float(loss) * batch_cases
        loss_totals["center"] += float(center_loss) * batch_cases
        loss_totals["heatmap"] += float(heatmap_loss) * batch_cases
        loss_totals["bbox"] += float(bbox_loss) * batch_cases
        loss_totals["cases"] += batch_cases

        predicted = predicted_center.float().cpu().numpy()
        target = target_center.float().cpu().numpy()
        spacing = (
            batch["effective_spacing_mm"].float().cpu().numpy()
        )
        predicted_voxel = predicted * shape_minus_one[None, :]
        target_voxel = target * shape_minus_one[None, :]
        error_voxel = predicted_voxel - target_voxel
        error_mm_axes = np.abs(error_voxel) * spacing[:, None]
        error_mm_euclidean = (
            np.linalg.norm(error_voxel, axis=1) * spacing
        )

        bbox_predicted = predicted_bbox.float().cpu().numpy()
        bbox_target = target_bbox.float().cpu().numpy()

        for index, study_id in enumerate(batch["study_id"]):
            rows.append(
                {
                    "study_id": str(study_id),
                    "effective_spacing_mm": float(spacing[index]),
                    "true_center_x": float(target[index, 0]),
                    "true_center_y": float(target[index, 1]),
                    "true_center_z": float(target[index, 2]),
                    "predicted_center_x": float(predicted[index, 0]),
                    "predicted_center_y": float(predicted[index, 1]),
                    "predicted_center_z": float(predicted[index, 2]),
                    "absolute_error_x_mm": float(
                        error_mm_axes[index, 0]
                    ),
                    "absolute_error_y_mm": float(
                        error_mm_axes[index, 1]
                    ),
                    "absolute_error_z_mm": float(
                        error_mm_axes[index, 2]
                    ),
                    "center_error_mm": float(
                        error_mm_euclidean[index]
                    ),
                    "true_bbox_x": float(bbox_target[index, 0]),
                    "true_bbox_y": float(bbox_target[index, 1]),
                    "true_bbox_z": float(bbox_target[index, 2]),
                    "predicted_bbox_x": float(
                        bbox_predicted[index, 0]
                    ),
                    "predicted_bbox_y": float(
                        bbox_predicted[index, 1]
                    ),
                    "predicted_bbox_z": float(
                        bbox_predicted[index, 2]
                    ),
                }
            )

    predictions = pd.DataFrame(rows)
    errors_mm = predictions["center_error_mm"].to_numpy()
    metrics = {
        "loss": loss_totals["loss"] / loss_totals["cases"],
        "center_loss": (
            loss_totals["center"] / loss_totals["cases"]
        ),
        "heatmap_loss": (
            loss_totals["heatmap"] / loss_totals["cases"]
        ),
        "bbox_loss": loss_totals["bbox"] / loss_totals["cases"],
        "median_error_mm": float(np.median(errors_mm)),
        "mean_error_mm": float(np.mean(errors_mm)),
        "p90_error_mm": float(np.percentile(errors_mm, 90)),
        "p95_error_mm": float(np.percentile(errors_mm, 95)),
        "maximum_error_mm": float(np.max(errors_mm)),
        "within_10mm": float(np.mean(errors_mm <= 10.0)),
        "within_20mm": float(np.mean(errors_mm <= 20.0)),
        "within_30mm": float(np.mean(errors_mm <= 30.0)),
        "within_50mm": float(np.mean(errors_mm <= 50.0)),
    }
    return metrics, predictions


# =============================================================================
# INITIALIZATION AND LOCKS
# =============================================================================

print("=" * 112)
print("STAGE 2A — RESUMABLE LIGHTWEIGHT 3D PANCREAS LOCALIZER TRAINING")
print("=" * 112)

for required_path in [
    MANIFEST_PATH,
    FREEZE_PATH,
    STAGE1B_AUDIT_PATH,
]:
    if not required_path.exists():
        raise FileNotFoundError(f"Required input missing:\n{required_path}")

if not torch.cuda.is_available():
    raise RuntimeError(
        "A CUDA GPU is required. Enable a T4 GPU in Colab."
    )

device = torch.device("cuda")
gpu_name = torch.cuda.get_device_name(0)
gpu_memory_gib = (
    torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
)
amp_enabled = True

MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
set_global_seed(SEED)

manifest_hash = sha256_file(MANIFEST_PATH)
manifest = pd.read_csv(MANIFEST_PATH)
manifest["study_id"] = manifest["study_id"].astype(str).str.strip()

with open(FREEZE_PATH, "r", encoding="utf-8") as file:
    freeze = json.load(file)
with open(STAGE1B_AUDIT_PATH, "r", encoding="utf-8") as file:
    stage1b_audit = json.load(file)

if freeze.get("dataset_frozen") is not True:
    raise RuntimeError("The Stage 1B dataset freeze is not valid.")
if stage1b_audit.get("result") != "PASS":
    raise RuntimeError("Stage 1B audit did not pass.")
if len(manifest) != EXPECTED_TRAIN + EXPECTED_VALIDATION:
    raise RuntimeError("The localizer manifest is incomplete.")
if manifest["study_id"].nunique() != len(manifest):
    raise RuntimeError("The localizer manifest contains duplicate IDs.")
if (
    manifest["partition"].value_counts().to_dict()
    != {"train": EXPECTED_TRAIN, "validation": EXPECTED_VALIDATION}
):
    raise RuntimeError("The locked train/validation counts changed.")
if set(manifest["partition"]) != {"train", "validation"}:
    raise RuntimeError("A locked test case entered Stage 2A.")

for column in [
    "pancreas_center_normalized_json",
    "pancreas_bbox_size_voxels_json",
    "effective_isotropic_spacing_mm",
    "output_path",
    "output_size_bytes",
    "output_sha256",
]:
    if column not in manifest.columns:
        raise RuntimeError(f"Manifest column missing: {column}")

if AUDIT_PATH.exists():
    with open(AUDIT_PATH, "r", encoding="utf-8") as file:
        previous_audit = json.load(file)
    if previous_audit.get("training_complete") is True:
        print()
        print("Training is already complete. No rerun is required.")
        print("Best checkpoint:")
        print(BEST_CHECKPOINT_PATH)
        print("Validation predictions:")
        print(PREDICTIONS_PATH)
        raise SystemExit(0)

print()
print(f"GPU: {gpu_name}")
print(f"GPU memory: {gpu_memory_gib:.2f} GiB")
print(f"PyTorch: {torch.__version__}")
print(f"Training cases: {EXPECTED_TRAIN}")
print(f"Validation cases: {EXPECTED_VALIDATION}")
print("Locked test cases accessed: 0")


# =============================================================================
# LOCAL HIGH-SPEED CACHE
# =============================================================================

local_free_gib = shutil.disk_usage(RUNTIME_ROOT).free / (1024 ** 3)
if local_free_gib < 3.0:
    raise RuntimeError(
        f"Insufficient local storage: {local_free_gib:.2f} GiB"
    )

print()
print("Preparing the local high-speed cache...")
copied_now = 0
reused = 0

for order, (_, row) in enumerate(manifest.iterrows(), start=1):
    study_id = str(row["study_id"])
    source = Path(str(row["output_path"]))
    destination = LOCAL_CACHE_DIR / f"{study_id}.npz"
    expected_size = int(row["output_size_bytes"])

    if not source.exists():
        raise FileNotFoundError(f"Frozen source missing: {source}")

    if destination.exists() and destination.stat().st_size == expected_size:
        reused += 1
    else:
        temporary = Path(str(destination) + ".part")
        if temporary.exists():
            temporary.unlink()
        shutil.copy2(source, temporary)
        if temporary.stat().st_size != expected_size:
            temporary.unlink()
            raise RuntimeError(
                f"Cache size mismatch for {study_id}"
            )
        os.replace(temporary, destination)
        copied_now += 1

    if order % 100 == 0 or order == len(manifest):
        print(
            f"  Cache progress: {order}/{len(manifest)} "
            f"(copied={copied_now}, reused={reused})"
        )

cache_files = list(LOCAL_CACHE_DIR.glob("*.npz"))
expected_cache_names = {
    f"{study_id}.npz" for study_id in manifest["study_id"]
}
observed_cache_names = {path.name for path in cache_files}
if observed_cache_names != expected_cache_names:
    raise RuntimeError(
        "Local cache inventory does not exactly match the frozen manifest."
    )


# =============================================================================
# LOADERS, MODEL, AND RESUME
# =============================================================================

train_frame = (
    manifest.loc[manifest["partition"] == "train"]
    .sort_values("study_id")
    .reset_index(drop=True)
)
validation_frame = (
    manifest.loc[manifest["partition"] == "validation"]
    .sort_values("study_id")
    .reset_index(drop=True)
)

train_dataset = LocalizerDataset(
    train_frame,
    LOCAL_CACHE_DIR,
    training=True,
)
validation_dataset = LocalizerDataset(
    validation_frame,
    LOCAL_CACHE_DIR,
    training=False,
)

train_generator = torch.Generator()
train_generator.manual_seed(SEED)

loader_arguments = {
    "batch_size": BATCH_SIZE,
    "num_workers": NUM_WORKERS,
    "pin_memory": True,
    "worker_init_fn": seed_worker,
    "persistent_workers": NUM_WORKERS > 0,
}
train_loader = DataLoader(
    train_dataset,
    shuffle=True,
    generator=train_generator,
    drop_last=False,
    **loader_arguments,
)
validation_loader = DataLoader(
    validation_dataset,
    shuffle=False,
    drop_last=False,
    **loader_arguments,
)

model = LightweightSpatialLocalizer3D().to(device)
parameter_count = sum(
    parameter.numel() for parameter in model.parameters()
)
trainable_parameter_count = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=3,
    min_lr=1e-6,
)
scaler = torch.amp.GradScaler(
    "cuda",
    enabled=amp_enabled,
)

start_epoch = 1
best_median_error = float("inf")
bad_epochs = 0

if HISTORY_PATH.exists():
    history = pd.read_csv(HISTORY_PATH)
else:
    history = pd.DataFrame()

if LATEST_CHECKPOINT_PATH.exists():
    checkpoint = torch.load(
        LATEST_CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )
    if checkpoint.get("manifest_sha256") != manifest_hash:
        raise RuntimeError(
            "Checkpoint manifest hash differs from the frozen dataset."
        )
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    scaler.load_state_dict(checkpoint["scaler_state"])
    start_epoch = int(checkpoint["epoch"]) + 1
    best_median_error = float(checkpoint["best_median_error_mm"])
    bad_epochs = int(checkpoint["bad_epochs"])
    restore_rng_state(
        checkpoint.get("rng_state"),
        train_generator,
    )
    print()
    print(
        f"Resuming after epoch {start_epoch - 1}; "
        f"best median error={best_median_error:.3f} mm"
    )

protocol = {
    "stage": "2A",
    "created_at_utc": utc_now(),
    "seed": SEED,
    "architecture": "LightweightSpatialLocalizer3D",
    "trainable_parameters": int(trainable_parameter_count),
    "input_shape": list(INPUT_SHAPE),
    "heatmap_shape": list(HEATMAP_SHAPE),
    "training_cases": EXPECTED_TRAIN,
    "validation_cases": EXPECTED_VALIDATION,
    "batch_size": BATCH_SIZE,
    "maximum_epochs": MAX_EPOCHS,
    "early_stopping_patience": EARLY_STOPPING_PATIENCE,
    "optimizer": "AdamW",
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "loss": {
        "center_smooth_l1_weight": CENTER_LOSS_WEIGHT,
        "heatmap_cross_entropy_weight": HEATMAP_LOSS_WEIGHT,
        "bbox_smooth_l1_weight": BBOX_LOSS_WEIGHT,
    },
    "model_selection_metric": "validation_median_center_error_mm",
    "hu_normalization": "(CT_HU - 50) / 250",
    "training_only_intensity_augmentation": {
        "scale": [0.95, 1.05],
        "bias": [-0.03, 0.03],
        "gaussian_noise_std": 0.005,
    },
    "mixed_precision": True,
    "gpu": gpu_name,
    "manifest_path": str(MANIFEST_PATH),
    "manifest_sha256": manifest_hash,
    "locked_test_cases_accessed": 0,
}
atomic_write_json(protocol, PROTOCOL_PATH)

print()
print(f"Model parameters: {parameter_count:,}")
print(f"Starting epoch: {start_epoch}/{MAX_EPOCHS}")

# Shape-only smoke test before training.
smoke_batch = next(iter(validation_loader))
with torch.no_grad():
    smoke_image = smoke_batch["image"][:1].to(device)
    with torch.amp.autocast(
        device_type="cuda",
        enabled=amp_enabled,
    ):
        smoke_center, smoke_bbox, smoke_heatmap = model(smoke_image)
if smoke_center.shape != (1, 3):
    raise RuntimeError("Model centre-output smoke test failed.")
if smoke_bbox.shape != (1, 3):
    raise RuntimeError("Model bbox-output smoke test failed.")
if tuple(smoke_heatmap.shape) != (1, 1, *HEATMAP_SHAPE):
    raise RuntimeError("Model heatmap-output smoke test failed.")
if not torch.all(torch.isfinite(smoke_center)):
    raise RuntimeError("Model smoke output is non-finite.")
del smoke_image, smoke_center, smoke_bbox, smoke_heatmap, smoke_batch
torch.cuda.empty_cache()
print("GPU model smoke test: PASS")


# =============================================================================
# TRAINING LOOP
# =============================================================================

training_started = time.time()
stopped_early = False

for epoch in range(start_epoch, MAX_EPOCHS + 1):
    epoch_started = time.time()
    print()
    print("-" * 112)
    print(f"EPOCH {epoch}/{MAX_EPOCHS}")
    print("-" * 112)

    train_metrics = train_one_epoch(
        model,
        train_loader,
        optimizer,
        scaler,
        device,
        amp_enabled,
    )
    validation_metrics, validation_predictions = evaluate(
        model,
        validation_loader,
        device,
        amp_enabled,
    )

    selection_metric = validation_metrics["median_error_mm"]
    scheduler.step(selection_metric)
    learning_rate = float(optimizer.param_groups[0]["lr"])
    improved = selection_metric < best_median_error - 1e-6

    if improved:
        best_median_error = selection_metric
        bad_epochs = 0
        best_payload = {
            "stage": "2A",
            "epoch": epoch,
            "model_state": model.state_dict(),
            "manifest_sha256": manifest_hash,
            "architecture": "LightweightSpatialLocalizer3D",
            "input_shape": INPUT_SHAPE,
            "heatmap_shape": HEATMAP_SHAPE,
            "best_validation_metrics": validation_metrics,
            "trainable_parameters": trainable_parameter_count,
            "saved_at_utc": utc_now(),
        }
        atomic_torch_save(best_payload, BEST_CHECKPOINT_PATH)
        atomic_write_csv(
            validation_predictions.sort_values("study_id"),
            PREDICTIONS_PATH,
        )
    else:
        bad_epochs += 1

    epoch_seconds = time.time() - epoch_started
    history_row = {
        "epoch": epoch,
        "learning_rate": learning_rate,
        "train_loss": train_metrics["loss"],
        "train_center_loss": train_metrics["center"],
        "train_heatmap_loss": train_metrics["heatmap"],
        "train_bbox_loss": train_metrics["bbox"],
        "validation_loss": validation_metrics["loss"],
        "validation_center_loss": validation_metrics["center_loss"],
        "validation_heatmap_loss": validation_metrics["heatmap_loss"],
        "validation_bbox_loss": validation_metrics["bbox_loss"],
        "validation_median_error_mm": (
            validation_metrics["median_error_mm"]
        ),
        "validation_mean_error_mm": (
            validation_metrics["mean_error_mm"]
        ),
        "validation_p90_error_mm": (
            validation_metrics["p90_error_mm"]
        ),
        "validation_p95_error_mm": (
            validation_metrics["p95_error_mm"]
        ),
        "validation_within_20mm": (
            validation_metrics["within_20mm"]
        ),
        "validation_within_30mm": (
            validation_metrics["within_30mm"]
        ),
        "validation_within_50mm": (
            validation_metrics["within_50mm"]
        ),
        "improved_best": improved,
        "bad_epochs": bad_epochs,
        "epoch_seconds": epoch_seconds,
        "completed_at_utc": utc_now(),
    }
    history = pd.concat(
        [history, pd.DataFrame([history_row])],
        ignore_index=True,
    )
    history = (
        history.drop_duplicates("epoch", keep="last")
        .sort_values("epoch")
        .reset_index(drop=True)
    )
    atomic_write_csv(history, HISTORY_PATH)

    latest_payload = {
        "stage": "2A",
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "best_median_error_mm": best_median_error,
        "bad_epochs": bad_epochs,
        "manifest_sha256": manifest_hash,
        "rng_state": capture_rng_state(train_generator),
        "saved_at_utc": utc_now(),
    }
    atomic_torch_save(latest_payload, LATEST_CHECKPOINT_PATH)

    print(
        f"Train loss: {train_metrics['loss']:.5f} | "
        f"Validation loss: {validation_metrics['loss']:.5f}"
    )
    print(
        f"Validation centre error: "
        f"median={validation_metrics['median_error_mm']:.2f} mm | "
        f"p90={validation_metrics['p90_error_mm']:.2f} mm | "
        f"within 30 mm={validation_metrics['within_30mm']:.3f}"
    )
    print(
        f"Best median: {best_median_error:.2f} mm | "
        f"bad epochs: {bad_epochs}/{EARLY_STOPPING_PATIENCE} | "
        f"epoch time: {epoch_seconds / 60:.1f} min"
    )

    if bad_epochs >= EARLY_STOPPING_PATIENCE:
        stopped_early = True
        print("Early stopping activated.")
        break


# =============================================================================
# FINAL BEST-MODEL EVALUATION AND AUDIT
# =============================================================================

if not BEST_CHECKPOINT_PATH.exists():
    raise RuntimeError("No best checkpoint was created.")

best_checkpoint = torch.load(
    BEST_CHECKPOINT_PATH,
    map_location=device,
    weights_only=False,
)
model.load_state_dict(best_checkpoint["model_state"])
best_epoch = int(best_checkpoint["epoch"])

final_metrics, final_predictions = evaluate(
    model,
    validation_loader,
    device,
    amp_enabled,
)
final_predictions = final_predictions.sort_values(
    "study_id"
).reset_index(drop=True)
atomic_write_csv(final_predictions, PREDICTIONS_PATH)

performance_screen = {
    "median_error_le_30mm":
        final_metrics["median_error_mm"] <= 30.0,
    "p90_error_le_60mm":
        final_metrics["p90_error_mm"] <= 60.0,
    "at_least_90_percent_within_50mm":
        final_metrics["within_50mm"] >= 0.90,
}
provisional_localizer_candidate = all(
    performance_screen.values()
)

readiness_checks = {
    "Stage 1B freeze passed": stage1b_audit.get("result") == "PASS",
    "Exactly 1376 training cases": len(train_frame) == EXPECTED_TRAIN,
    "Exactly 295 validation cases":
        len(validation_frame) == EXPECTED_VALIDATION,
    "No locked test case was accessed":
        set(manifest["partition"]) == {"train", "validation"},
    "All 1671 cached inputs were available":
        len(cache_files) == len(manifest),
    "GPU smoke test passed": True,
    "Best checkpoint exists": BEST_CHECKPOINT_PATH.exists(),
    "Validation predictions contain 295 unique cases":
        (
            len(final_predictions) == EXPECTED_VALIDATION
            and final_predictions["study_id"].nunique()
            == EXPECTED_VALIDATION
        ),
    "All validation errors are finite":
        np.isfinite(
            final_predictions["center_error_mm"].to_numpy()
        ).all(),
}
all_readiness_checks_pass = all(
    bool(value) for value in readiness_checks.values()
)

audit = {
    "stage": "2A",
    "created_at_utc": utc_now(),
    "result": (
        "PASS_PROVISIONAL_LOCALIZER_CANDIDATE"
        if provisional_localizer_candidate
        else "PASS_TRAINING_COMPLETE_PERFORMANCE_REVIEW_REQUIRED"
    ),
    "training_complete": True,
    "stopped_early": stopped_early,
    "epochs_completed": int(history["epoch"].max()),
    "best_epoch": best_epoch,
    "trainable_parameters": int(trainable_parameter_count),
    "gpu": gpu_name,
    "training_duration_hours_this_run": (
        time.time() - training_started
    ) / 3600.0,
    "validation_metrics": {
        key: float(value)
        for key, value in final_metrics.items()
    },
    "performance_screen": {
        key: bool(value)
        for key, value in performance_screen.items()
    },
    "provisional_localizer_candidate":
        provisional_localizer_candidate,
    "readiness_checks": {
        key: bool(value)
        for key, value in readiness_checks.items()
    },
    "best_checkpoint_path": str(BEST_CHECKPOINT_PATH),
    "latest_checkpoint_path": str(LATEST_CHECKPOINT_PATH),
    "history_path": str(HISTORY_PATH),
    "validation_predictions_path": str(PREDICTIONS_PATH),
    "protocol_path": str(PROTOCOL_PATH),
    "locked_test_cases_accessed": 0,
}
atomic_write_json(audit, AUDIT_PATH)

print()
print("=" * 112)
print("STAGE 2A FINAL RESULT")
print("=" * 112)
print(f"Best epoch: {best_epoch}")
print(
    f"Validation median centre error: "
    f"{final_metrics['median_error_mm']:.3f} mm"
)
print(
    f"Validation p90 centre error: "
    f"{final_metrics['p90_error_mm']:.3f} mm"
)
print(
    f"Validation p95 centre error: "
    f"{final_metrics['p95_error_mm']:.3f} mm"
)
print(
    f"Within 20/30/50 mm: "
    f"{final_metrics['within_20mm']:.3f} / "
    f"{final_metrics['within_30mm']:.3f} / "
    f"{final_metrics['within_50mm']:.3f}"
)
print(f"Provisional localizer candidate: {provisional_localizer_candidate}")
print()
print("Best checkpoint:")
print(BEST_CHECKPOINT_PATH)
print()
print("Training history:")
print(HISTORY_PATH)
print()
print("Validation predictions:")
print(PREDICTIONS_PATH)
print()
print("Audit:")
print(AUDIT_PATH)
print()
print(
    "STAGE 2A RESULT: "
    + audit["result"]
)
print("=" * 112)

if not all_readiness_checks_pass:
    raise RuntimeError("Stage 2A final readiness checks failed.")
