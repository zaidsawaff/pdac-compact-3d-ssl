from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import random
import shutil

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


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
BEST_CHECKPOINT_PATH = (
    MODEL_DIR / "stage2a_localizer_best.pt"
)
STAGE2A_PREDICTIONS_PATH = (
    MODEL_DIR / "stage2a_localizer_validation_predictions.csv"
)
STAGE2A_AUDIT_PATH = (
    MODEL_DIR / "stage2a_localizer_training_audit.json"
)
FINAL_CROP_PROTOCOL_PATH = (
    META_DIR / "stage2d_final_deployment_crop_geometry_protocol.json"
)

OUTPUT_PATH = (
    MODEL_DIR / "stage2e_all_development_localizer_predictions.csv"
)
INFERENCE_PROTOCOL_PATH = (
    META_DIR / "stage2e_development_localizer_inference_protocol.json"
)
AUDIT_PATH = (
    QC_DIR / "stage2e_all_development_localizer_inference_audit.json"
)

LOCAL_CACHE_DIR = (RUNTIME_ROOT / "pdac_stage2a_localizer_cache")

SEED = 20260728
EXPECTED_CASES = 1671
EXPECTED_TRAIN = 1376
EXPECTED_VALIDATION = 295
INPUT_SHAPE = np.asarray([96, 96, 160], dtype=float)
HEATMAP_SHAPE = (12, 12, 20)
HU_CENTER = 50.0
HU_HALF_WIDTH = 250.0
BATCH_SIZE = 8
NUM_WORKERS = 2


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


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def parse_vector(value, dtype=float):
    vector = np.asarray(json.loads(str(value)), dtype=dtype)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise RuntimeError(f"Invalid vector: {value}")
    return vector


def set_seed(seed):
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


class InferenceDataset(Dataset):
    def __init__(self, dataframe, cache_dir):
        self.rows = dataframe.reset_index(drop=True)
        self.cache_dir = Path(cache_dir)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        study_id = str(row["study_id"])
        path = self.cache_dir / f"{study_id}.npz"

        with np.load(path, allow_pickle=False) as data:
            ct_hu = np.asarray(data["ct_hu"], dtype=np.float32)
            true_center = np.asarray(
                data["pancreas_center_normalized"],
                dtype=np.float32,
            )

        if tuple(ct_hu.shape) != tuple(INPUT_SHAPE.astype(int)):
            raise RuntimeError(
                f"{study_id}: unexpected localizer input shape."
            )

        image = torch.from_numpy(
            (ct_hu - HU_CENTER) / HU_HALF_WIDTH
        ).unsqueeze(0)
        image = torch.clamp(image, -1.0, 1.0)

        return {
            "image": image,
            "true_center": torch.from_numpy(true_center),
            "study_id": study_id,
        }


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
        return center, bbox_size, logits, probabilities


def canvas_to_canonical(
    center_normalized,
    canonical_shape,
    canonical_spacing,
    effective_spacing,
    core_shape,
    pad_before,
):
    canvas_voxel = center_normalized * (INPUT_SHAPE - 1.0)
    core_voxel = canvas_voxel - pad_before
    scale = effective_spacing / canonical_spacing
    offset = (
        (canonical_shape - 1.0) / 2.0
        - scale * ((core_shape - 1.0) / 2.0)
    )
    return scale * core_voxel + offset


print("=" * 112)
print("STAGE 2E — ALL-DEVELOPMENT LOCALIZER INFERENCE AND CENTER FREEZE")
print("=" * 112)

for required_path in [
    MANIFEST_PATH,
    BEST_CHECKPOINT_PATH,
    STAGE2A_PREDICTIONS_PATH,
    STAGE2A_AUDIT_PATH,
    FINAL_CROP_PROTOCOL_PATH,
]:
    if not required_path.exists():
        raise FileNotFoundError(f"Required input missing:\n{required_path}")

if not torch.cuda.is_available():
    raise RuntimeError("Stage 2E requires a CUDA GPU.")

if AUDIT_PATH.exists():
    with open(AUDIT_PATH, "r", encoding="utf-8") as file:
        previous_audit = json.load(file)
    if previous_audit.get("centers_frozen") is True:
        print("Stage 2E is already complete; no rerun is required.")
        print(OUTPUT_PATH)
        raise SystemExit(0)

set_seed(SEED)
device = torch.device("cuda")
gpu_name = torch.cuda.get_device_name(0)

manifest = pd.read_csv(MANIFEST_PATH)
manifest["study_id"] = manifest["study_id"].astype(str).str.strip()
stage2a_predictions = pd.read_csv(STAGE2A_PREDICTIONS_PATH)
stage2a_predictions["study_id"] = (
    stage2a_predictions["study_id"].astype(str).str.strip()
)

with open(STAGE2A_AUDIT_PATH, "r", encoding="utf-8") as file:
    stage2a_audit = json.load(file)
with open(FINAL_CROP_PROTOCOL_PATH, "r", encoding="utf-8") as file:
    crop_protocol = json.load(file)

if stage2a_audit.get("training_complete") is not True:
    raise RuntimeError("Stage 2A training is incomplete.")
if crop_protocol.get("selected_candidate") != "E5_240x192x128":
    raise RuntimeError("The locked E5 deployment crop was not found.")
if len(manifest) != EXPECTED_CASES:
    raise RuntimeError("The frozen development manifest is incomplete.")
if manifest["study_id"].nunique() != EXPECTED_CASES:
    raise RuntimeError("Development manifest IDs are not unique.")
if manifest["partition"].value_counts().to_dict() != {
    "train": EXPECTED_TRAIN,
    "validation": EXPECTED_VALIDATION,
}:
    raise RuntimeError("The locked development partitions changed.")
if set(manifest["partition"]) != {"train", "validation"}:
    raise RuntimeError("A locked test case entered Stage 2E.")

print()
print(f"GPU: {gpu_name}")
print(f"Development cases: {EXPECTED_CASES}")
print(f"Train: {EXPECTED_TRAIN}")
print(f"Validation: {EXPECTED_VALIDATION}")
print("Locked test cases accessed: 0")


# Local high-speed cache: normally reused from Stage 2A.
LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
print()
print("Checking the local inference cache...")
copied = 0
reused = 0

for order, (_, row) in enumerate(manifest.iterrows(), start=1):
    study_id = str(row["study_id"])
    source = Path(str(row["output_path"]))
    destination = LOCAL_CACHE_DIR / f"{study_id}.npz"
    expected_size = int(row["output_size_bytes"])

    if destination.exists() and destination.stat().st_size == expected_size:
        reused += 1
    else:
        if not source.exists():
            raise FileNotFoundError(f"Frozen input missing: {source}")
        temporary = Path(str(destination) + ".part")
        if temporary.exists():
            temporary.unlink()
        shutil.copy2(source, temporary)
        if temporary.stat().st_size != expected_size:
            temporary.unlink()
            raise RuntimeError(f"Cache size mismatch: {study_id}")
        os.replace(temporary, destination)
        copied += 1

    if order % 200 == 0 or order == len(manifest):
        print(
            f"  Cache: {order}/{len(manifest)} "
            f"(reused={reused}, copied={copied})"
        )

cache_names = {
    path.name for path in LOCAL_CACHE_DIR.glob("*.npz")
}
expected_names = {
    f"{study_id}.npz" for study_id in manifest["study_id"]
}
if cache_names != expected_names:
    raise RuntimeError("The local cache inventory is not exact.")

inference_frame = manifest.sort_values(
    ["partition", "study_id"]
).reset_index(drop=True)
dataset = InferenceDataset(inference_frame, LOCAL_CACHE_DIR)
generator = torch.Generator()
generator.manual_seed(SEED)
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=NUM_WORKERS > 0,
    worker_init_fn=seed_worker,
    generator=generator,
)

model = LightweightSpatialLocalizer3D().to(device)
checkpoint = torch.load(
    BEST_CHECKPOINT_PATH,
    map_location=device,
    weights_only=False,
)
model.load_state_dict(checkpoint["model_state"], strict=True)
model.eval()

parameter_count = sum(
    parameter.numel() for parameter in model.parameters()
)
if parameter_count != 471764:
    raise RuntimeError(
        f"Model parameter-count mismatch: {parameter_count}"
    )

rows = []
heatmap_voxels = int(np.prod(HEATMAP_SHAPE))
log_heatmap_voxels = float(np.log(heatmap_voxels))

with torch.no_grad():
    for batch_index, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        with torch.amp.autocast(
            device_type="cuda",
            enabled=True,
        ):
            predicted_center, predicted_bbox, logits, probabilities = (
                model(images)
            )

        predicted_center_np = (
            predicted_center.float().cpu().numpy()
        )
        predicted_bbox_np = (
            predicted_bbox.float().cpu().numpy()
        )
        true_center_np = (
            batch["true_center"].float().cpu().numpy()
        )
        flat_probabilities = probabilities.float().flatten(
            start_dim=1
        )
        maximum_probability = (
            flat_probabilities.max(dim=1).values.cpu().numpy()
        )
        normalized_entropy = (
            -torch.sum(
                flat_probabilities
                * torch.log(
                    torch.clamp(flat_probabilities, min=1e-12)
                ),
                dim=1,
            )
            / log_heatmap_voxels
        ).cpu().numpy()

        for index, study_id in enumerate(batch["study_id"]):
            source_row = inference_frame.loc[
                inference_frame["study_id"] == study_id
            ].iloc[0]

            canonical_shape = parse_vector(
                source_row["canonical_shape_json"]
            )
            canonical_spacing = parse_vector(
                source_row["canonical_spacing_mm_json"]
            )
            core_shape = parse_vector(
                source_row["resampled_core_shape_json"]
            )
            pad_before = parse_vector(
                source_row["pad_before_json"]
            )
            effective_spacing = float(
                source_row["effective_isotropic_spacing_mm"]
            )
            predicted_canonical = canvas_to_canonical(
                predicted_center_np[index],
                canonical_shape,
                canonical_spacing,
                effective_spacing,
                core_shape,
                pad_before,
            )

            error_voxel = (
                predicted_center_np[index]
                - true_center_np[index]
            ) * (INPUT_SHAPE - 1.0)
            center_error_mm = float(
                np.linalg.norm(error_voxel) * effective_spacing
            )

            rows.append(
                {
                    "study_id": str(study_id),
                    "patient_id": source_row["patient_id"],
                    "partition": source_row["partition"],
                    "diagnostic_label": (
                        source_row["diagnostic_label"]
                    ),
                    "annotation_type": source_row["annotation_type"],
                    "predicted_center_x": float(
                        predicted_center_np[index, 0]
                    ),
                    "predicted_center_y": float(
                        predicted_center_np[index, 1]
                    ),
                    "predicted_center_z": float(
                        predicted_center_np[index, 2]
                    ),
                    "predicted_center_canonical_x": float(
                        predicted_canonical[0]
                    ),
                    "predicted_center_canonical_y": float(
                        predicted_canonical[1]
                    ),
                    "predicted_center_canonical_z": float(
                        predicted_canonical[2]
                    ),
                    "true_center_x": float(
                        true_center_np[index, 0]
                    ),
                    "true_center_y": float(
                        true_center_np[index, 1]
                    ),
                    "true_center_z": float(
                        true_center_np[index, 2]
                    ),
                    "center_error_mm": center_error_mm,
                    "predicted_bbox_x": float(
                        predicted_bbox_np[index, 0]
                    ),
                    "predicted_bbox_y": float(
                        predicted_bbox_np[index, 1]
                    ),
                    "predicted_bbox_z": float(
                        predicted_bbox_np[index, 2]
                    ),
                    "heatmap_max_probability": float(
                        maximum_probability[index]
                    ),
                    "heatmap_normalized_entropy": float(
                        normalized_entropy[index]
                    ),
                    "effective_isotropic_spacing_mm":
                        effective_spacing,
                    "canonical_shape_json": (
                        source_row["canonical_shape_json"]
                    ),
                    "canonical_spacing_mm_json": (
                        source_row["canonical_spacing_mm_json"]
                    ),
                    "localizer_checkpoint_epoch": int(
                        checkpoint["epoch"]
                    ),
                }
            )

        if (
            batch_index == 1
            or batch_index % 25 == 0
            or batch_index == len(loader)
        ):
            print(
                f"Inference progress: "
                f"{min(batch_index * BATCH_SIZE, len(dataset))}/"
                f"{len(dataset)}"
            )

predictions = pd.DataFrame(rows).sort_values(
    ["partition", "study_id"]
).reset_index(drop=True)
atomic_write_csv(predictions, OUTPUT_PATH)

if len(predictions) != EXPECTED_CASES:
    raise RuntimeError("All-development inference row count failed.")
if predictions["study_id"].nunique() != EXPECTED_CASES:
    raise RuntimeError("All-development prediction IDs are not unique.")
if predictions["partition"].value_counts().to_dict() != {
    "train": EXPECTED_TRAIN,
    "validation": EXPECTED_VALIDATION,
}:
    raise RuntimeError("Prediction partition counts changed.")

numeric = predictions.select_dtypes(include=[np.number]).to_numpy()
if not np.isfinite(numeric).all():
    raise RuntimeError("Inference output contains non-finite values.")

for axis in ["x", "y", "z"]:
    values = predictions[f"predicted_center_{axis}"]
    if not values.between(0.0, 1.0, inclusive="both").all():
        raise RuntimeError(
            f"Predicted normalized {axis} centre lies outside [0, 1]."
        )

# Independent reproduction of the validation predictions saved at Stage 2A.
validation_new = predictions.loc[
    predictions["partition"] == "validation"
].copy()
comparison = validation_new.merge(
    stage2a_predictions[
        [
            "study_id",
            "predicted_center_x",
            "predicted_center_y",
            "predicted_center_z",
            "center_error_mm",
        ]
    ],
    on="study_id",
    how="inner",
    validate="one_to_one",
    suffixes=("_new", "_stage2a"),
)
if len(comparison) != EXPECTED_VALIDATION:
    raise RuntimeError("Validation reproduction merge is incomplete.")

center_differences = []
for axis in ["x", "y", "z"]:
    center_differences.append(
        np.abs(
            comparison[f"predicted_center_{axis}_new"]
            - comparison[f"predicted_center_{axis}_stage2a"]
        ).to_numpy()
    )
maximum_center_difference = float(
    np.max(np.stack(center_differences))
)
maximum_error_difference_mm = float(
    np.max(
        np.abs(
            comparison["center_error_mm_new"]
            - comparison["center_error_mm_stage2a"]
        )
    )
)

partition_metrics = []
for partition, group in predictions.groupby("partition"):
    errors = group["center_error_mm"].to_numpy()
    partition_metrics.append(
        {
            "partition": partition,
            "cases": int(len(group)),
            "median_center_error_mm": float(np.median(errors)),
            "p90_center_error_mm": float(
                np.percentile(errors, 90)
            ),
            "p95_center_error_mm": float(
                np.percentile(errors, 95)
            ),
            "within_30mm": float(np.mean(errors <= 30.0)),
            "within_50mm": float(np.mean(errors <= 50.0)),
        }
    )

readiness_checks = {
    "Exactly 1671 development predictions":
        len(predictions) == EXPECTED_CASES,
    "Exactly 1671 unique study IDs":
        predictions["study_id"].nunique() == EXPECTED_CASES,
    "Prediction IDs exactly match frozen manifest":
        set(predictions["study_id"]) == set(manifest["study_id"]),
    "Train count remains 1376":
        int((predictions["partition"] == "train").sum())
        == EXPECTED_TRAIN,
    "Validation count remains 295":
        int((predictions["partition"] == "validation").sum())
        == EXPECTED_VALIDATION,
    "All normalized centres lie in [0, 1]": all(
        predictions[f"predicted_center_{axis}"].between(
            0.0,
            1.0,
            inclusive="both",
        ).all()
        for axis in ["x", "y", "z"]
    ),
    "All prediction and confidence values are finite":
        np.isfinite(numeric).all(),
    "Validation centres reproduce Stage 2A within 0.0001":
        maximum_center_difference <= 1e-4,
    "Validation errors reproduce Stage 2A within 0.05 mm":
        maximum_error_difference_mm <= 0.05,
    "Best checkpoint has 471764 parameters":
        parameter_count == 471764,
    "Final E5 crop protocol is locked":
        crop_protocol.get("selected_candidate")
        == "E5_240x192x128",
    "No locked test case was accessed":
        set(predictions["partition"]) == {"train", "validation"},
}
all_checks_pass = all(
    bool(value) for value in readiness_checks.values()
)

output_hash = sha256_file(OUTPUT_PATH)
protocol = {
    "stage": "2E",
    "created_at_utc": utc_now(),
    "model_checkpoint": str(BEST_CHECKPOINT_PATH),
    "model_checkpoint_sha256": sha256_file(
        BEST_CHECKPOINT_PATH
    ),
    "best_epoch": int(checkpoint["epoch"]),
    "model_parameters": parameter_count,
    "development_cases": EXPECTED_CASES,
    "train_cases": EXPECTED_TRAIN,
    "validation_cases": EXPECTED_VALIDATION,
    "center_coordinate_system": (
        "CANONICAL_LPS_VOXEL_COORDINATES_WITH_PER_CASE_SPACING"
    ),
    "deployment_crop": crop_protocol,
    "predictions_path": str(OUTPUT_PATH),
    "predictions_sha256": output_hash,
    "locked_test_cases_accessed": 0,
    "centers_frozen": all_checks_pass,
}
atomic_write_json(protocol, INFERENCE_PROTOCOL_PATH)

audit = {
    "stage": "2E",
    "created_at_utc": utc_now(),
    "result": (
        "PASS_ALL_DEVELOPMENT_CENTERS_FROZEN"
        if all_checks_pass
        else "FAIL"
    ),
    "centers_frozen": all_checks_pass,
    "partition_metrics": partition_metrics,
    "maximum_validation_center_reproduction_difference":
        maximum_center_difference,
    "maximum_validation_error_reproduction_difference_mm":
        maximum_error_difference_mm,
    "heatmap_max_probability_summary": {
        "minimum": float(
            predictions["heatmap_max_probability"].min()
        ),
        "median": float(
            predictions["heatmap_max_probability"].median()
        ),
        "maximum": float(
            predictions["heatmap_max_probability"].max()
        ),
    },
    "heatmap_normalized_entropy_summary": {
        "minimum": float(
            predictions["heatmap_normalized_entropy"].min()
        ),
        "median": float(
            predictions["heatmap_normalized_entropy"].median()
        ),
        "maximum": float(
            predictions["heatmap_normalized_entropy"].max()
        ),
    },
    "readiness_checks": {
        key: bool(value)
        for key, value in readiness_checks.items()
    },
    "predictions_path": str(OUTPUT_PATH),
    "protocol_path": str(INFERENCE_PROTOCOL_PATH),
    "locked_test_cases_accessed": 0,
}
atomic_write_json(audit, AUDIT_PATH)

print()
print("-" * 112)
print("PARTITION METRICS")
print("-" * 112)
print(pd.DataFrame(partition_metrics).to_string(index=False))

print()
print("-" * 112)
print("READINESS CHECKS")
print("-" * 112)
for check, passed in readiness_checks.items():
    print(f"  {check}: {bool(passed)}")

print()
print("Frozen predictions:")
print(OUTPUT_PATH)
print()
print("Inference protocol:")
print(INFERENCE_PROTOCOL_PATH)
print()
print("Audit:")
print(AUDIT_PATH)
print()
print("=" * 112)
print(
    "STAGE 2E RESULT: "
    + (
        "PASS_ALL_DEVELOPMENT_CENTERS_FROZEN"
        if all_checks_pass
        else "FAIL"
    )
)
print("=" * 112)

if not all_checks_pass:
    failed = [
        check
        for check, passed in readiness_checks.items()
        if not bool(passed)
    ]
    raise RuntimeError(f"Stage 2E failed checks: {failed}")
