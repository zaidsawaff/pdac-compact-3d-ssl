from pathlib import Path
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
import shutil
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.ndimage import (
    distance_transform_edt,
    find_objects,
    gaussian_filter,
    label as connected_components,
    maximum_filter,
)
from sklearn.metrics import average_precision_score, roc_auc_score


# =============================================================================
# PATHS AND LOCKED CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(os.environ.get("PDAC_PROJECT_ROOT", "/content/drive/MyDrive/PDAC_Public_Q1_Project"))
RUNTIME_ROOT = Path(os.environ.get("PDAC_RUNTIME_ROOT", "/content"))
RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
META_DIR = PROJECT_ROOT / "01_Metadata"
QC_DIR = PROJECT_ROOT / "02_Quality_Control"

FROZEN_MANIFEST_PATH = META_DIR / "stage3b_e5_frozen_dataset_manifest.csv"
MODEL_SELECTION_PATH = META_DIR / "stage5c_supervised_model_selection.json"
STAGE5C_AUDIT_PATH = QC_DIR / "stage5c_full_validation_comparison_audit.json"

CASE_LEDGER_PATH = QC_DIR / "stage5d_validation_candidate_case_ledger.csv"
CANDIDATE_LEDGER_PATH = QC_DIR / "stage5d_validation_candidate_ledger.csv"
FROC_CURVE_PATH = QC_DIR / "stage5d_validation_froc_curve.csv"
FROC_OPERATING_POINTS_PATH = QC_DIR / "stage5d_validation_froc_operating_points.csv"
PROTOCOL_PATH = META_DIR / "stage5d_detection_and_froc_protocol.json"
AUDIT_PATH = QC_DIR / "stage5d_validation_froc_calibration_audit.json"

LOCAL_CACHE_DIR = (RUNTIME_ROOT / "pdac_e5_ssl_cache")

SEED = 20260728
EXPECTED_CASES = 1671
EXPECTED_VALIDATION = 295
EXPECTED_VALIDATION_PDAC = 88
EXPECTED_VALIDATION_NON_PDAC = 207
VOLUME_SHAPE = np.asarray([240, 192, 128], dtype=int)
PATCH_SHAPE = np.asarray([128, 128, 64], dtype=int)
SPACING_MM = np.asarray([1.25, 1.25, 2.0], dtype=float)
HU_CENTER = 50.0
HU_HALF_WIDTH = 250.0
BATCH_SIZE = 4
SLIDING_OVERLAP = 0.50
USE_AMP = True

# Deployment-compatible candidate generation. These values are fixed before
# test access; probability threshold alone is calibrated on validation.
PANCREAS_GATE_PROBABILITY = 0.10
PANCREAS_GATE_MARGIN_MM = 20.0
LESION_SMOOTHING_SIGMA_MM = 2.5
NMS_RADIUS_MM = 10.0
MINIMUM_CANDIDATE_SCORE = 0.01
MAXIMUM_CANDIDATES_PER_CASE = 256
REFERENCE_LOCALIZATION_TOLERANCE_MM = 5.0
FROC_TARGET_FP_PER_CASE = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
DEPLOYMENT_TARGET_FP_PER_CASE = 1.0


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


def make_starts(volume_size, patch_size, overlap):
    stride = max(1, int(round(patch_size * (1.0 - overlap))))
    if volume_size <= patch_size:
        return [0]
    starts = list(range(0, volume_size - patch_size + 1, stride))
    final_start = volume_size - patch_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def odd_filter_size(radius_mm, spacing_mm):
    radii = np.ceil(float(radius_mm) / np.asarray(spacing_mm)).astype(int)
    return tuple(int(2 * radius + 1) for radius in radii)


def protocol_signature():
    payload = {
        "pancreas_gate_probability": PANCREAS_GATE_PROBABILITY,
        "pancreas_gate_margin_mm": PANCREAS_GATE_MARGIN_MM,
        "lesion_smoothing_sigma_mm": LESION_SMOOTHING_SIGMA_MM,
        "nms_radius_mm": NMS_RADIUS_MM,
        "minimum_candidate_score": MINIMUM_CANDIDATE_SCORE,
        "maximum_candidates_per_case": MAXIMUM_CANDIDATES_PER_CASE,
        "reference_localization_tolerance_mm": REFERENCE_LOCALIZATION_TOLERANCE_MM,
        "spacing_mm": SPACING_MM.tolist(),
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# =============================================================================
# MODEL — IDENTICAL TO STAGES 5B/5C
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
# INPUT AUDIT, CACHE, AND SELECTED MODEL
# =============================================================================


print("=" * 120)
print("STAGE 5D — RESUMABLE VALIDATION-ONLY CANDIDATE AND FROC CALIBRATION")
print("=" * 120)

for required in [FROZEN_MANIFEST_PATH, MODEL_SELECTION_PATH, STAGE5C_AUDIT_PATH]:
    if not required.exists():
        raise FileNotFoundError(f"Required input missing:\n{required}")

with open(MODEL_SELECTION_PATH, "r", encoding="utf-8") as file:
    model_selection = json.load(file)
with open(STAGE5C_AUDIT_PATH, "r", encoding="utf-8") as file:
    stage5c_audit = json.load(file)
if stage5c_audit.get("all_checks_pass") is not True:
    raise RuntimeError("Stage 5C full-validation comparison did not pass.")

selected_arm = str(model_selection["selected_arm"])
selected_checkpoint_path = Path(model_selection["selected_checkpoint_path"])
if selected_arm != "SSL_INITIALIZED":
    raise RuntimeError("Unexpected selected arm; Stage 5C lock changed.")
if not selected_checkpoint_path.exists():
    raise FileNotFoundError(f"Selected checkpoint missing: {selected_checkpoint_path}")
selected_checkpoint_hash = sha256_file(selected_checkpoint_path)
if selected_checkpoint_hash != str(model_selection["selected_checkpoint_sha256"]):
    raise RuntimeError("Selected checkpoint SHA-256 does not match Stage 5C.")

if not torch.cuda.is_available():
    raise RuntimeError("Stage 5D requires a CUDA GPU.")

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
    raise RuntimeError("A locked test case entered Stage 5D.")

validation = (
    manifest[manifest["partition"] == "validation"]
    .sort_values("study_id")
    .reset_index(drop=True)
)
if len(validation) != EXPECTED_VALIDATION:
    raise RuntimeError("Validation case count changed.")
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

checkpoint = torch.load(
    selected_checkpoint_path, map_location=device, weights_only=False
)
if checkpoint.get("arm") != selected_arm:
    raise RuntimeError("Selected checkpoint arm identity mismatch.")
model = LightweightDualHead3DSegmenter().to(device)
model.load_state_dict(checkpoint["model_state"], strict=True)
model.eval()

print(f"PyTorch: {torch.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Selected arm: {selected_arm}")
print(f"Selected epoch: {int(checkpoint['best_epoch'])}")
print(f"Validation cases: {len(validation)}")
print("Locked test cases accessed: 0")

LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
additional_bytes = 0
for _, row in manifest.iterrows():
    destination = LOCAL_CACHE_DIR / f"{row['study_id']}.npz"
    expected_size = int(row["output_size_bytes"])
    if not (destination.exists() and destination.stat().st_size == expected_size):
        additional_bytes += expected_size
if disk_free_gib(RUNTIME_ROOT) * (1024 ** 3) < additional_bytes * 1.10 + 1024 ** 3:
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
def infer_full_crop(ct_hu):
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
# DEPLOYMENT-COMPATIBLE CANDIDATE GENERATION
# =============================================================================


sigma_voxels = tuple(
    float(LESION_SMOOTHING_SIGMA_MM / spacing) for spacing in SPACING_MM
)
nms_filter_size = odd_filter_size(NMS_RADIUS_MM, SPACING_MM)
connectivity = np.ones((3, 3, 3), dtype=np.uint8)
signature = protocol_signature()


def generate_candidates(probabilities, lesion_target):
    pancreas_seed = probabilities[0] >= PANCREAS_GATE_PROBABILITY
    pancreas_seed_present = bool(pancreas_seed.any())
    if pancreas_seed_present:
        gate_distance = distance_transform_edt(
            ~pancreas_seed, sampling=SPACING_MM
        )
        anatomical_gate = gate_distance <= PANCREAS_GATE_MARGIN_MM
    else:
        anatomical_gate = np.ones(tuple(VOLUME_SHAPE), dtype=bool)

    smoothed = gaussian_filter(
        probabilities[1], sigma=sigma_voxels, mode="nearest"
    ).astype(np.float32, copy=False)
    local_maximum = maximum_filter(
        smoothed, size=nms_filter_size, mode="nearest"
    )
    peak_mask = (
        anatomical_gate
        & (smoothed >= MINIMUM_CANDIDATE_SCORE)
        & np.isclose(smoothed, local_maximum, rtol=0.0, atol=1e-7)
    )

    # Collapse any flat local-maximum plateau to one peak.
    plateau_labels, plateau_count = connected_components(
        peak_mask, structure=connectivity
    )
    candidates = []
    plateau_objects = find_objects(plateau_labels)
    for plateau_id, object_slices in enumerate(plateau_objects, start=1):
        if object_slices is None:
            continue
        local_labels = plateau_labels[object_slices]
        local_component = local_labels == plateau_id
        local_scores = np.where(
            local_component,
            smoothed[object_slices],
            -np.inf,
        )
        local_coordinate = np.asarray(
            np.unravel_index(int(np.argmax(local_scores)), local_scores.shape),
            dtype=int,
        )
        coordinate = np.asarray(
            [
                int(local_coordinate[axis] + object_slices[axis].start)
                for axis in range(3)
            ],
            dtype=int,
        )
        score = float(smoothed[tuple(coordinate)])
        candidates.append(
            {
                "coordinate": coordinate.astype(int),
                "score": score,
                "raw_probability": float(
                    probabilities[1][tuple(coordinate)]
                ),
            }
        )
    candidates.sort(key=lambda item: item["score"], reverse=True)
    candidates = candidates[:MAXIMUM_CANDIDATES_PER_CASE]

    is_pdac = bool(lesion_target.any())
    if is_pdac:
        distance_to_reference = distance_transform_edt(
            ~lesion_target, sampling=SPACING_MM
        )
    else:
        distance_to_reference = None

    reference_already_matched = False
    for rank, candidate in enumerate(candidates, start=1):
        coordinate = tuple(candidate["coordinate"])
        reference_distance = (
            float(distance_to_reference[coordinate])
            if distance_to_reference is not None else np.nan
        )
        localized = bool(
            is_pdac
            and reference_distance <= REFERENCE_LOCALIZATION_TOLERANCE_MM
        )
        true_positive = bool(localized and not reference_already_matched)
        if true_positive:
            reference_already_matched = True
        candidate.update(
            {
                "rank": int(rank),
                "reference_distance_mm": reference_distance,
                "within_reference_tolerance": localized,
                "is_true_positive_candidate": true_positive,
                "is_false_positive_candidate": not true_positive,
            }
        )

    del smoothed, local_maximum, peak_mask, plateau_labels
    if pancreas_seed_present:
        del gate_distance
    if distance_to_reference is not None:
        del distance_to_reference
    return candidates, pancreas_seed_present, reference_already_matched


# =============================================================================
# RESUMABLE CASE-BY-CASE CALIBRATION INFERENCE
# =============================================================================


if CASE_LEDGER_PATH.exists():
    case_ledger = pd.read_csv(
        CASE_LEDGER_PATH, dtype={"study_id": str, "patient_id": str}
    )
    if "protocol_signature" in case_ledger.columns:
        case_ledger = case_ledger[
            (case_ledger["protocol_signature"].astype(str) == signature)
            & (
                case_ledger["checkpoint_sha256"].astype(str)
                == selected_checkpoint_hash
            )
        ].copy()
        case_ledger = case_ledger.drop_duplicates("study_id", keep="last")
    else:
        case_ledger = pd.DataFrame()
else:
    case_ledger = pd.DataFrame()

if CANDIDATE_LEDGER_PATH.exists():
    candidate_ledger = pd.read_csv(
        CANDIDATE_LEDGER_PATH, dtype={"study_id": str, "patient_id": str}
    )
    if "protocol_signature" in candidate_ledger.columns:
        candidate_ledger = candidate_ledger[
            (candidate_ledger["protocol_signature"].astype(str) == signature)
            & (
                candidate_ledger["checkpoint_sha256"].astype(str)
                == selected_checkpoint_hash
            )
        ].copy()
        candidate_ledger = candidate_ledger.drop_duplicates(
            ["study_id", "candidate_rank"], keep="last"
        )
    else:
        candidate_ledger = pd.DataFrame()
else:
    candidate_ledger = pd.DataFrame()

completed_ids = set(
    case_ledger.get("study_id", pd.Series(dtype=str)).astype(str)
)
pending = validation[~validation["study_id"].isin(completed_ids)]

print("\nCANDIDATE CALIBRATION INFERENCE")
print("-" * 120)
print(f"Previously completed: {len(completed_ids)}/{len(validation)}")
print(f"Pending: {len(pending)}")
print(f"NMS filter size: {nms_filter_size}")
print(f"Protocol signature: {signature}")

for order, (_, row) in enumerate(pending.iterrows(), start=1):
    study_id = str(row["study_id"])
    started = time.time()
    with np.load(LOCAL_CACHE_DIR / f"{study_id}.npz", allow_pickle=False) as data:
        ct_hu = np.asarray(data["ct_hu"], dtype=np.int16)
        mask = np.asarray(data["mask_labels"], dtype=np.uint8)
    if tuple(ct_hu.shape) != tuple(VOLUME_SHAPE) or mask.shape != ct_hu.shape:
        raise RuntimeError(f"{study_id}: frozen E5 geometry mismatch.")

    probabilities = infer_full_crop(ct_hu)
    finite = bool(np.isfinite(probabilities).all())
    if not finite:
        raise RuntimeError(f"{study_id}: non-finite probability output.")
    lesion_target = mask == 1
    is_pdac = row["diagnostic_label_normalized"] == "pdac"
    if bool(lesion_target.any()) != bool(is_pdac):
        raise RuntimeError(f"{study_id}: lesion label conflicts with diagnosis.")

    candidates, pancreas_seed_present, lesion_localized = generate_candidates(
        probabilities, lesion_target
    )
    candidate_rows = []
    for candidate in candidates:
        coordinate = candidate["coordinate"]
        candidate_rows.append(
            {
                "protocol_signature": signature,
                "checkpoint_sha256": selected_checkpoint_hash,
                "selected_arm": selected_arm,
                "study_id": study_id,
                "patient_id": str(row["patient_id"]),
                "diagnostic_label": row["diagnostic_label_normalized"],
                "label_binary": int(is_pdac),
                "annotation_type": row["annotation_type_normalized"],
                "candidate_rank": int(candidate["rank"]),
                "candidate_score": float(candidate["score"]),
                "candidate_raw_probability": float(candidate["raw_probability"]),
                "candidate_x": int(coordinate[0]),
                "candidate_y": int(coordinate[1]),
                "candidate_z": int(coordinate[2]),
                "candidate_world_x_mm": float(coordinate[0] * SPACING_MM[0]),
                "candidate_world_y_mm": float(coordinate[1] * SPACING_MM[1]),
                "candidate_world_z_mm": float(coordinate[2] * SPACING_MM[2]),
                "reference_distance_mm": candidate["reference_distance_mm"],
                "within_reference_tolerance": bool(
                    candidate["within_reference_tolerance"]
                ),
                "is_true_positive_candidate": bool(
                    candidate["is_true_positive_candidate"]
                ),
                "is_false_positive_candidate": bool(
                    candidate["is_false_positive_candidate"]
                ),
                "completed_at_utc": utc_now(),
            }
        )
    if candidate_rows:
        candidate_ledger = pd.concat(
            [candidate_ledger, pd.DataFrame(candidate_rows)], ignore_index=True
        )
        candidate_ledger = candidate_ledger.drop_duplicates(
            ["study_id", "candidate_rank"], keep="last"
        ).sort_values(["study_id", "candidate_rank"])
        atomic_write_csv(candidate_ledger, CANDIDATE_LEDGER_PATH)

    case_row = {
        "protocol_signature": signature,
        "checkpoint_sha256": selected_checkpoint_hash,
        "selected_arm": selected_arm,
        "study_id": study_id,
        "patient_id": str(row["patient_id"]),
        "diagnostic_label": row["diagnostic_label_normalized"],
        "label_binary": int(is_pdac),
        "annotation_type": row["annotation_type_normalized"],
        "reference_lesions": int(is_pdac),
        "generated_candidates": int(len(candidates)),
        "pancreas_seed_present": bool(pancreas_seed_present),
        "reference_localized_at_any_score": bool(lesion_localized),
        "maximum_candidate_score": (
            float(candidates[0]["score"]) if candidates else 0.0
        ),
        "probabilities_finite": finite,
        "inference_and_candidate_seconds": float(time.time() - started),
        "processing_complete": True,
        "completed_at_utc": utc_now(),
    }
    case_ledger = pd.concat(
        [case_ledger, pd.DataFrame([case_row])], ignore_index=True
    )
    case_ledger = case_ledger.drop_duplicates("study_id", keep="last").sort_values(
        "study_id"
    )
    atomic_write_csv(case_ledger, CASE_LEDGER_PATH)

    del ct_hu, mask, probabilities, lesion_target, candidates
    if order % 10 == 0 or order == len(pending):
        print(
            f"  Completed this run {order}/{len(pending)} — "
            f"durable total {len(case_ledger)}/{len(validation)}"
        )

if len(case_ledger) != EXPECTED_VALIDATION:
    raise RuntimeError("Candidate case ledger is incomplete.")
if case_ledger["study_id"].nunique() != EXPECTED_VALIDATION:
    raise RuntimeError("Candidate case ledger IDs are not unique.")
if set(case_ledger["study_id"].astype(str)) != set(validation["study_id"]):
    raise RuntimeError("Candidate case IDs do not match validation lock.")
if not truth_flags(case_ledger["processing_complete"]).all():
    raise RuntimeError("One or more candidate cases failed.")
if len(candidate_ledger) == 0:
    raise RuntimeError("No candidates were generated.")


# =============================================================================
# FROC CURVE, OPERATING POINTS, AND THRESHOLD LOCK
# =============================================================================


candidate_ledger["is_true_positive_candidate_bool"] = truth_flags(
    candidate_ledger["is_true_positive_candidate"]
)
candidate_ledger["is_false_positive_candidate_bool"] = truth_flags(
    candidate_ledger["is_false_positive_candidate"]
)
ranked = candidate_ledger.sort_values(
    ["candidate_score", "study_id", "candidate_rank"],
    ascending=[False, True, True],
).reset_index(drop=True)
ranked["cumulative_true_positives"] = ranked[
    "is_true_positive_candidate_bool"
].cumsum()
ranked["cumulative_false_positives"] = ranked[
    "is_false_positive_candidate_bool"
].cumsum()

curve = (
    ranked.groupby("candidate_score", sort=False)
    .tail(1)
    .copy()
)
curve = pd.DataFrame(
    {
        "probability_threshold": curve["candidate_score"].astype(float),
        "true_positive_lesions": curve["cumulative_true_positives"].astype(int),
        "false_positive_candidates": curve["cumulative_false_positives"].astype(int),
    }
)
curve["total_reference_lesions"] = EXPECTED_VALIDATION_PDAC
curve["validation_cases"] = EXPECTED_VALIDATION
curve["sensitivity"] = (
    curve["true_positive_lesions"] / EXPECTED_VALIDATION_PDAC
)
curve["false_positives_per_case"] = (
    curve["false_positive_candidates"] / EXPECTED_VALIDATION
)
initial = pd.DataFrame(
    [
        {
            "probability_threshold": float(ranked["candidate_score"].max() + 1e-6),
            "true_positive_lesions": 0,
            "false_positive_candidates": 0,
            "total_reference_lesions": EXPECTED_VALIDATION_PDAC,
            "validation_cases": EXPECTED_VALIDATION,
            "sensitivity": 0.0,
            "false_positives_per_case": 0.0,
        }
    ]
)
curve = pd.concat([initial, curve], ignore_index=True)
curve = curve.sort_values(
    ["false_positives_per_case", "sensitivity", "probability_threshold"],
    ascending=[True, True, False],
).reset_index(drop=True)
atomic_write_csv(curve, FROC_CURVE_PATH)

operating_rows = []
for target_fp in FROC_TARGET_FP_PER_CASE:
    eligible = curve[curve["false_positives_per_case"] <= target_fp + 1e-12]
    if len(eligible) == 0:
        selected = curve.iloc[0]
    else:
        selected = eligible.sort_values(
            ["sensitivity", "false_positives_per_case", "probability_threshold"],
            ascending=[False, False, False],
        ).iloc[0]
    operating_rows.append(
        {
            "target_false_positives_per_case": float(target_fp),
            "achieved_false_positives_per_case": float(
                selected["false_positives_per_case"]
            ),
            "sensitivity": float(selected["sensitivity"]),
            "probability_threshold": float(selected["probability_threshold"]),
            "true_positive_lesions": int(selected["true_positive_lesions"]),
            "false_positive_candidates": int(selected["false_positive_candidates"]),
        }
    )
operating_points = pd.DataFrame(operating_rows)
atomic_write_csv(operating_points, FROC_OPERATING_POINTS_PATH)

deployment_row = operating_points[
    np.isclose(
        operating_points["target_false_positives_per_case"],
        DEPLOYMENT_TARGET_FP_PER_CASE,
    )
].iloc[0]
deployment_threshold = float(deployment_row["probability_threshold"])

deployment_candidates = candidate_ledger[
    candidate_ledger["candidate_score"] >= deployment_threshold
].copy()
case_has_candidate = set(deployment_candidates["study_id"].astype(str))
negative_cases = case_ledger[case_ledger["label_binary"] == 0]
pdac_cases = case_ledger[case_ledger["label_binary"] == 1]
negative_specificity = float(
    np.mean(
        [str(study_id) not in case_has_candidate for study_id in negative_cases["study_id"]]
    )
)
pdac_case_sensitivity = float(
    np.mean(
        [
            bool(
                (
                    deployment_candidates[
                        deployment_candidates["study_id"].astype(str) == str(study_id)
                    ]["is_true_positive_candidate_bool"]
                ).any()
            )
            for study_id in pdac_cases["study_id"]
        ]
    )
)

labels = case_ledger["label_binary"].astype(int).to_numpy()
scores = case_ledger["maximum_candidate_score"].astype(float).to_numpy()
candidate_auc = float(roc_auc_score(labels, scores))
candidate_ap = float(average_precision_score(labels, scores))
standard_froc_targets = operating_points[
    operating_points["target_false_positives_per_case"].isin(
        [0.25, 0.5, 1.0, 2.0, 4.0]
    )
]
froc_mean_sensitivity = float(standard_froc_targets["sensitivity"].mean())

protocol = {
    "stage": "5D",
    "created_at_utc": utc_now(),
    "protocol_locked": True,
    "selected_arm": selected_arm,
    "selected_checkpoint_path": str(selected_checkpoint_path),
    "selected_checkpoint_sha256": selected_checkpoint_hash,
    "candidate_generation": {
        "lesion_probability_smoothing_sigma_mm": LESION_SMOOTHING_SIGMA_MM,
        "pancreas_probability_gate": PANCREAS_GATE_PROBABILITY,
        "pancreas_gate_expansion_mm": PANCREAS_GATE_MARGIN_MM,
        "non_maximum_suppression_radius_mm": NMS_RADIUS_MM,
        "minimum_candidate_score": MINIMUM_CANDIDATE_SCORE,
        "maximum_candidates_per_case": MAXIMUM_CANDIDATES_PER_CASE,
        "candidate_score": "smoothed_lesion_probability_at_local_maximum",
        "reference_localization_rule": (
            "highest-scoring unmatched candidate within 5 mm of label-1 lesion"
        ),
        "reference_localization_tolerance_mm": REFERENCE_LOCALIZATION_TOLERANCE_MM,
        "one_reference_primary_PDAC_lesion_per_positive_study": True,
        "protocol_signature": signature,
    },
    "calibrated_probability_threshold": deployment_threshold,
    "calibration_target_false_positives_per_case": DEPLOYMENT_TARGET_FP_PER_CASE,
    "achieved_validation_false_positives_per_case": float(
        deployment_row["achieved_false_positives_per_case"]
    ),
    "achieved_validation_lesion_sensitivity": float(
        deployment_row["sensitivity"]
    ),
    "validation_negative_case_specificity": negative_specificity,
    "validation_PDAC_case_sensitivity": pdac_case_sensitivity,
    "validation_candidate_AUC": candidate_auc,
    "validation_candidate_average_precision": candidate_ap,
    "validation_FROC_mean_sensitivity_at_0_25_0_5_1_2_4_FP_per_case": (
        froc_mean_sensitivity
    ),
    "threshold_selection_data": "validation_only",
    "internal_test_threshold_adjustment": "prohibited",
    "external_test_threshold_adjustment": "prohibited",
    "locked_test_cases_accessed": 0,
    "froc_curve_path": str(FROC_CURVE_PATH),
    "froc_operating_points_path": str(FROC_OPERATING_POINTS_PATH),
}
atomic_write_json(protocol, PROTOCOL_PATH)

readiness_checks = {
    "Stage 5C selected SSL model": selected_arm == "SSL_INITIALIZED",
    "Exactly 295 validation cases completed": len(case_ledger) == EXPECTED_VALIDATION,
    "Exactly 295 unique validation IDs": case_ledger["study_id"].nunique() == EXPECTED_VALIDATION,
    "Validation IDs exactly match frozen lock": set(case_ledger["study_id"].astype(str)) == set(validation["study_id"]),
    "All predicted probability arrays were finite": truth_flags(case_ledger["probabilities_finite"]).all(),
    "Pancreas gate seed exists in every case": truth_flags(case_ledger["pancreas_seed_present"]).all(),
    "All 88 PDAC references are represented": int(case_ledger["reference_lesions"].sum()) == EXPECTED_VALIDATION_PDAC,
    "At least one candidate was generated": len(candidate_ledger) > 0,
    "FROC curve contains finite values": bool(np.isfinite(curve[["probability_threshold", "sensitivity", "false_positives_per_case"]]).all().all()),
    "Six FROC operating points were computed": len(operating_points) == len(FROC_TARGET_FP_PER_CASE),
    "Deployment threshold is finite": bool(np.isfinite(deployment_threshold)),
    "Candidate-level AUC is finite": bool(np.isfinite(candidate_auc)),
    "Candidate-level average precision is finite": bool(np.isfinite(candidate_ap)),
    "No locked test case was accessed": set(manifest["partition"]) == {"train", "validation"},
}
all_checks_pass = all(bool(value) for value in readiness_checks.values())

audit = {
    "stage": "5D",
    "created_at_utc": utc_now(),
    "result": "PASS_VALIDATION_FROC_PROTOCOL_LOCKED" if all_checks_pass else "FAIL",
    "all_checks_pass": all_checks_pass,
    "readiness_checks": readiness_checks,
    "selected_arm": selected_arm,
    "selected_checkpoint_sha256": selected_checkpoint_hash,
    "protocol_signature": signature,
    "validation_cases": EXPECTED_VALIDATION,
    "PDAC_references": EXPECTED_VALIDATION_PDAC,
    "generated_candidates": int(len(candidate_ledger)),
    "mean_candidates_per_case": float(len(candidate_ledger) / EXPECTED_VALIDATION),
    "PDAC_references_localized_at_any_score": int(
        truth_flags(
            case_ledger[case_ledger["label_binary"] == 1][
                "reference_localized_at_any_score"
            ]
        ).sum()
    ),
    "candidate_AUC": candidate_auc,
    "candidate_average_precision": candidate_ap,
    "FROC_mean_sensitivity": froc_mean_sensitivity,
    "deployment_threshold": deployment_threshold,
    "deployment_achieved_FP_per_case": float(
        deployment_row["achieved_false_positives_per_case"]
    ),
    "deployment_sensitivity": float(deployment_row["sensitivity"]),
    "deployment_negative_specificity": negative_specificity,
    "case_ledger_path": str(CASE_LEDGER_PATH),
    "candidate_ledger_path": str(CANDIDATE_LEDGER_PATH),
    "froc_curve_path": str(FROC_CURVE_PATH),
    "operating_points_path": str(FROC_OPERATING_POINTS_PATH),
    "protocol_path": str(PROTOCOL_PATH),
    "locked_test_cases_accessed": 0,
}
atomic_write_json(audit, AUDIT_PATH)

print("\nFROC OPERATING POINTS")
print("-" * 120)
print(operating_points.to_string(index=False))
print("\nCALIBRATED DEPLOYMENT POINT")
print("-" * 120)
print(f"Probability threshold: {deployment_threshold:.6f}")
print(
    f"Achieved FP/case: "
    f"{float(deployment_row['achieved_false_positives_per_case']):.6f}"
)
print(f"Lesion sensitivity: {float(deployment_row['sensitivity']):.6f}")
print(f"Negative-case specificity: {negative_specificity:.6f}")
print(f"Candidate-level AUC: {candidate_auc:.6f}")
print(f"Candidate-level AP: {candidate_ap:.6f}")
print(f"Mean FROC sensitivity: {froc_mean_sensitivity:.6f}")

print("\nREADINESS CHECKS")
print("-" * 120)
for name, passed in readiness_checks.items():
    print(f"  {name}: {bool(passed)}")

print("\nLocked detection protocol:")
print(PROTOCOL_PATH)
print("\nAudit:")
print(AUDIT_PATH)
print("\n" + "=" * 120)
print(
    "STAGE 5D RESULT: "
    + ("PASS — VALIDATION FROC AND DETECTION PROTOCOL LOCKED" if all_checks_pass else "FAIL")
)
print("=" * 120)

if not all_checks_pass:
    failed = [name for name, passed in readiness_checks.items() if not passed]
    raise RuntimeError(f"Stage 5D failed readiness checks: {failed}")
