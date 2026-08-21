from pathlib import Path
from datetime import datetime, timezone
from tempfile import TemporaryDirectory
import binascii
import gc
import gzip
import hashlib
import json
import os
import random
import shutil
import struct
import time
import zlib

import nibabel as nib
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from scipy import ndimage
from scipy.ndimage import (
    distance_transform_edt,
    find_objects,
    gaussian_filter,
    label as connected_components,
    map_coordinates,
    maximum_filter,
)


# =============================================================================
# STAGE R1H — LOCKED PATHS
# =============================================================================

PROJECT_ROOT = Path(os.environ.get("PDAC_PROJECT_ROOT", "/content/drive/MyDrive/PDAC_Public_Q1_Project"))
RUNTIME_ROOT = Path(os.environ.get("PDAC_RUNTIME_ROOT", "/content"))
RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
META_DIR = PROJECT_ROOT / "01_Metadata"
QC_DIR = PROJECT_ROOT / "02_Quality_Control"
PREDICTION_DIR = (
    PROJECT_ROOT / "05_Predictions" / "StageR1H_External_Test_Blind"
)

LOCKED_SPLIT_PATH = META_DIR / "stage0f_locked_study_split_index.csv"
REMOTE_MEMBER_PATH = META_DIR / "stage0i_panorama_remote_zip_member_inventory.csv"
REMOTE_SOURCE_PATH = META_DIR / "stage0i_panorama_remote_archive_source_audit.csv"
LOCALIZER_PROTOCOL_PATH = META_DIR / "stage0m_b_r_localizer_input_protocol.json"
CROP_PROTOCOL_PATH = META_DIR / "stage2d_final_deployment_crop_geometry_protocol.json"
PREFREEZE_PATH = META_DIR / "stageR1e_prefreeze_model_threshold_lock.json"
PREFREEZE_MANIFEST_PATH = META_DIR / "stageR1e_prefreeze_evidence_manifest.csv"
PREFREEZE_AUDIT_PATH = QC_DIR / "stageR1e_prefreeze_model_threshold_audit.json"
FROC_PROTOCOL_PATH = META_DIR / "stageR1d_detection_and_froc_protocol.json"
R1D_AUDIT_PATH = QC_DIR / "stageR1d_validation_froc_calibration_audit.json"

LOCALIZER_CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "04_Models"
    / "Localizer"
    / "F1_96x96x160"
    / "stage2a_localizer_best.pt"
)

RESUME_LEDGER_PATH = QC_DIR / "stageR1h_blind_external_test_resume_ledger.csv"
CANDIDATE_LEDGER_PATH = QC_DIR / "stageR1h_blind_external_test_candidate_ledger.csv"
MANIFEST_PATH = META_DIR / "stageR1h_blind_external_test_prediction_manifest.csv"
FREEZE_PATH = META_DIR / "stageR1h_blind_external_test_prediction_freeze.json"
AUDIT_PATH = QC_DIR / "stageR1h_blind_external_test_inference_audit.json"
ERROR_PATH = QC_DIR / "stageR1h_blind_external_test_errors.csv"


# =============================================================================
# LOCKED BLIND-INFERENCE CONFIGURATION
# =============================================================================

SEED = 20260728
EXPECTED_EXTERNAL_TEST_CASES = 274
EXPECTED_PARTITION_COUNTS = {
    "external_msd_test": 194,
    "nih_negative_stress_test": 80,
}
EXPECTED_SOURCE_COUNTS = {
    "MSD_EXTERNAL_MIXED": 194,
    "NIH_NEGATIVE_STRESS": 80,
}

LOCALIZER_SHAPE = np.asarray([96, 96, 160], dtype=int)
LOCALIZER_PADDING_PER_SIDE = 4
LOCALIZER_HEATMAP_SHAPE = (12, 12, 20)

CROP_SHAPE = np.asarray([240, 192, 128], dtype=int)
CROP_SPACING_MM = np.asarray([1.25, 1.25, 2.0], dtype=float)
PATCH_SHAPE = np.asarray([128, 128, 64], dtype=int)
SLIDING_OVERLAP = 0.50
SEGMENTATION_BATCH_SIZE = 4

HU_MIN = -200.0
HU_MAX = 300.0
HU_CENTER = 50.0
HU_HALF_WIDTH = 250.0

PANCREAS_GATE_PROBABILITY = 0.10
PANCREAS_GATE_MARGIN_MM = 20.0
LESION_SMOOTHING_SIGMA_MM = 2.5
NMS_RADIUS_MM = 10.0
MINIMUM_CANDIDATE_SCORE = 0.01
MAXIMUM_CANDIDATES_PER_CASE = 256

# None processes every unfinished case. Set a positive integer only for a
# deliberate short run; every accepted case is checkpointed durably.
MAX_CASES_PER_RUN = None

RESAMPLE_Z_CHUNK = 8
HTTP_CHUNK_BYTES = 8 * 1024 * 1024
HTTP_CONNECT_TIMEOUT = 30
HTTP_READ_TIMEOUT = 180
HTTP_MAX_ATTEMPTS = 12
PROBABILITY_SCALE = 65535.0
CANONICAL_AXCODES = ("L", "P", "S")


# =============================================================================
# GENERAL UTILITIES
# =============================================================================


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_write_csv(frame, path):
    temporary = Path(str(path) + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_write_json(payload, path):
    temporary = Path(str(path) + ".tmp")
    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def atomic_save_npz(path, **arrays):
    temporary = Path(str(path) + ".part")
    if temporary.exists():
        temporary.unlink()
    with open(temporary, "wb") as file:
        np.savez_compressed(file, **arrays)
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


def json_vector(values):
    return json.dumps([float(v) for v in np.asarray(values).reshape(-1)])


def json_matrix(values):
    return json.dumps(np.asarray(values, dtype=float).tolist())


def normalize_id_series(series):
    return series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def truth_flags(series):
    return (
        series.fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "pass", "complete", "completed"})
    )


def replace_rows(frame, replacement, key):
    if len(frame):
        values = set(replacement[key].astype(str))
        frame = frame.loc[~frame[key].astype(str).isin(values)].copy()
    return pd.concat([frame, replacement], ignore_index=True)


def validate_prediction_file(path, expected_size, expected_hash):
    path = Path(path)
    return bool(
        path.exists()
        and path.stat().st_size == int(expected_size)
        and sha256_file(path) == str(expected_hash)
    )


def set_determinism():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_float32_matmul_precision("high")


def group_count(channels):
    for groups in [8, 4, 2, 1]:
        if channels % groups == 0:
            return groups
    return 1


# =============================================================================
# SAFE REMOTE ZIP-MEMBER ACQUISITION
# =============================================================================


def parse_crc32(value):
    text = str(value).strip().lower()
    if text.startswith("0x"):
        text = text[2:]
    return int(text, 16)


def candidate_endpoints(source_row):
    endpoints = []
    selected = str(source_row.get("selected_range_url", "")).strip()
    if selected and selected.lower() != "nan":
        endpoints.append(selected)
    record_id = int(source_row["record_id"])
    archive_name = str(source_row["archive_name"])
    endpoints.extend(
        [
            f"https://zenodo.org/api/records/{record_id}/files/{archive_name}/content",
            f"https://zenodo.org/records/{record_id}/files/{archive_name}?download=1",
        ]
    )
    return list(dict.fromkeys(endpoints))


def parse_content_range(value):
    if not value or not str(value).lower().startswith("bytes "):
        return None
    try:
        interval = str(value).split(" ", 1)[1].split("/", 1)[0]
        start, end = interval.split("-", 1)
        return int(start), int(end)
    except Exception:
        return None


def fetch_small_range(session, endpoints, start, end):
    errors = []
    for attempt in range(HTTP_MAX_ATTEMPTS):
        endpoint = endpoints[attempt % len(endpoints)]
        try:
            response = session.get(
                endpoint,
                headers={
                    "Range": f"bytes={start}-{end}",
                    "Accept-Encoding": "identity",
                },
                timeout=(HTTP_CONNECT_TIMEOUT, HTTP_READ_TIMEOUT),
                allow_redirects=True,
            )
            if response.status_code != 206:
                response.close()
                raise RuntimeError(
                    f"Unsafe full/non-range response rejected: HTTP {response.status_code}"
                )
            observed = parse_content_range(response.headers.get("Content-Range"))
            if observed is None or observed[0] != start or observed[1] != end:
                response.close()
                raise RuntimeError("Invalid Content-Range response.")
            payload = response.content
            response.close()
            if len(payload) != end - start + 1:
                raise RuntimeError("Short byte-range response.")
            return payload, endpoint
        except Exception as error:
            errors.append(f"{type(error).__name__}: {error}")
            time.sleep(min(2 ** min(attempt, 4), 20))
    raise RuntimeError("All safe range endpoints failed: " + " | ".join(errors[-5:]))


def download_exact_range(session, endpoints, start, size, destination):
    end = int(start + size - 1)
    temporary = Path(str(destination) + ".part")
    if temporary.exists() and temporary.stat().st_size > size:
        temporary.unlink()
    completed = temporary.stat().st_size if temporary.exists() else 0
    attempts = 0
    selected_endpoint = None
    with open(temporary, "ab") as output:
        while completed < size:
            request_start = start + completed
            endpoint = endpoints[attempts % len(endpoints)]
            attempts += 1
            try:
                with session.get(
                    endpoint,
                    headers={
                        "Range": f"bytes={request_start}-{end}",
                        "Accept-Encoding": "identity",
                    },
                    timeout=(HTTP_CONNECT_TIMEOUT, HTTP_READ_TIMEOUT),
                    stream=True,
                    allow_redirects=True,
                ) as response:
                    if response.status_code != 206:
                        raise RuntimeError(
                            f"Unsafe full/non-range response rejected: HTTP {response.status_code}"
                        )
                    observed = parse_content_range(response.headers.get("Content-Range"))
                    if observed is None or observed[0] != request_start or observed[1] > end:
                        raise RuntimeError("Invalid Content-Range response.")
                    before = completed
                    for chunk in response.iter_content(chunk_size=HTTP_CHUNK_BYTES):
                        if not chunk:
                            continue
                        if len(chunk) > size - completed:
                            raise RuntimeError("Range response exceeded expected size.")
                        output.write(chunk)
                        completed += len(chunk)
                    output.flush()
                    os.fsync(output.fileno())
                    selected_endpoint = endpoint
                    if completed == before:
                        raise RuntimeError("Range response returned no payload.")
                attempts = 0
            except Exception:
                if attempts >= HTTP_MAX_ATTEMPTS:
                    raise
                time.sleep(min(2 ** min(attempts, 4), 20))
    if temporary.stat().st_size != size:
        raise RuntimeError("Compressed payload size mismatch.")
    os.replace(temporary, destination)
    return selected_endpoint


def recover_zip_member(session, member_row, source_row, work_dir):
    endpoints = candidate_endpoints(source_row)
    local_header_offset = int(member_row["local_header_offset"])
    header, header_endpoint = fetch_small_range(
        session, endpoints, local_header_offset, local_header_offset + 29
    )
    fields = struct.unpack("<IHHHHHIIIHH", header)
    (
        signature,
        _version,
        flag_bits,
        compression_type,
        _mod_time,
        _mod_date,
        _local_crc,
        _local_compressed,
        _local_uncompressed,
        filename_length,
        extra_length,
    ) = fields
    if signature != 0x04034B50:
        raise RuntimeError("Invalid ZIP local-header signature.")
    if flag_bits & 0x1:
        raise RuntimeError("Encrypted ZIP member prohibited.")
    filename_bytes, _ = fetch_small_range(
        session,
        endpoints,
        local_header_offset + 30,
        local_header_offset + 30 + filename_length - 1,
    )
    local_filename = filename_bytes.decode("utf-8")
    if local_filename != str(member_row["member_name"]):
        raise RuntimeError("ZIP local filename differs from central inventory.")
    if int(compression_type) != int(member_row["compression_type"]):
        raise RuntimeError("ZIP compression type mismatch.")

    compressed_size = int(member_row["compressed_size_bytes"])
    expected_size = int(member_row["uncompressed_size_bytes"])
    expected_crc = parse_crc32(member_row["crc32_hex"])
    data_start = local_header_offset + 30 + filename_length + extra_length
    compressed_path = work_dir / "member_payload.bin"
    inner_path = work_dir / "ct.nii.gz"
    download_endpoint = download_exact_range(
        session, endpoints, data_start, compressed_size, compressed_path
    )

    decompressor = zlib.decompressobj(-zlib.MAX_WBITS) if compression_type == 8 else None
    if compression_type not in {0, 8}:
        raise RuntimeError(f"Unsupported ZIP compression type: {compression_type}")
    recovered_size = 0
    crc_value = 0
    with open(compressed_path, "rb") as source, open(inner_path, "wb") as destination:
        while True:
            chunk = source.read(HTTP_CHUNK_BYTES)
            if not chunk:
                break
            recovered = decompressor.decompress(chunk) if decompressor else chunk
            if recovered:
                destination.write(recovered)
                recovered_size += len(recovered)
                crc_value = binascii.crc32(recovered, crc_value)
        if decompressor:
            recovered = decompressor.flush()
            if recovered:
                destination.write(recovered)
                recovered_size += len(recovered)
                crc_value = binascii.crc32(recovered, crc_value)
        destination.flush()
        os.fsync(destination.fileno())
    crc_value &= 0xFFFFFFFF
    compressed_path.unlink()
    if recovered_size != expected_size or crc_value != expected_crc:
        raise RuntimeError("Recovered ZIP member failed size/CRC verification.")
    with open(inner_path, "rb") as file:
        if file.read(2) != b"\x1f\x8b":
            raise RuntimeError("Recovered member is not gzip NIfTI.")
    return {
        "inner_path": inner_path,
        "compressed_transfer_bytes": compressed_size,
        "recovered_size_bytes": recovered_size,
        "crc32_hex": f"{crc_value:08x}",
        "header_endpoint": header_endpoint,
        "download_endpoint": download_endpoint,
    }


def decompress_gzip_to_nii(source, destination):
    with gzip.open(source, "rb") as compressed, open(destination, "wb") as output:
        shutil.copyfileobj(compressed, output, length=HTTP_CHUNK_BYTES)
        output.flush()
        os.fsync(output.fileno())


# =============================================================================
# CT-ONLY GEOMETRY AND PREPROCESSING
# =============================================================================


def canonical_image_geometry(image):
    source_orientation = nib.orientations.io_orientation(image.affine)
    target_orientation = nib.orientations.axcodes2ornt(CANONICAL_AXCODES)
    transform = nib.orientations.ornt_transform(source_orientation, target_orientation)
    canonical_affine = image.affine @ nib.orientations.inv_ornt_aff(
        transform, image.shape[:3]
    )
    raw_array = image.dataobj.get_unscaled()
    canonical_array = nib.orientations.apply_orientation(raw_array, transform)
    orientation = "".join(nib.aff2axcodes(canonical_affine))
    if orientation != "LPS":
        raise RuntimeError(f"Canonical orientation failed: {orientation}")
    spacing = np.linalg.norm(canonical_affine[:3, :3], axis=0)
    return {
        "array": canonical_array,
        "affine": np.asarray(canonical_affine, dtype=float),
        "shape": np.asarray(canonical_array.shape, dtype=int),
        "spacing": np.asarray(spacing, dtype=float),
        "raw_orientation": "".join(nib.aff2axcodes(image.affine)),
    }


def proxy_scaling(image):
    slope = getattr(image.dataobj, "slope", None)
    intercept = getattr(image.dataobj, "inter", None)
    slope = 1.0 if slope is None else float(slope)
    intercept = 0.0 if intercept is None else float(intercept)
    if not np.isfinite(slope) or slope == 0 or not np.isfinite(intercept):
        raise RuntimeError("Invalid CT scaling parameters.")
    return slope, intercept


def ct_to_localizer_canvas(ct_geometry, slope, intercept):
    canonical_shape = ct_geometry["shape"]
    canonical_spacing = ct_geometry["spacing"]
    volume_extent = canonical_shape.astype(float) * canonical_spacing
    usable_shape = LOCALIZER_SHAPE - 2 * LOCALIZER_PADDING_PER_SIDE
    effective_spacing = float(np.max(volume_extent / usable_shape))
    core_shape = np.ceil(volume_extent / effective_spacing - 1e-6).astype(int)
    if np.any(core_shape > usable_shape):
        raise RuntimeError("Localizer core exceeds usable canvas.")
    transform_matrix = np.diag(
        np.full(3, effective_spacing, dtype=float) / canonical_spacing
    )
    transform_offset = (
        (canonical_shape.astype(float) - 1.0) / 2.0
        - transform_matrix @ ((core_shape.astype(float) - 1.0) / 2.0)
    )
    core = ndimage.affine_transform(
        ct_geometry["array"],
        matrix=transform_matrix,
        offset=transform_offset,
        output_shape=tuple(int(v) for v in core_shape),
        output=np.float32,
        order=1,
        mode="constant",
        cval=float((HU_MIN - intercept) / slope),
        prefilter=False,
    )
    core = core * slope + intercept
    if not np.isfinite(core).all():
        raise RuntimeError("Non-finite localizer CT values.")
    core = np.rint(np.clip(core, HU_MIN, HU_MAX)).astype(np.int16)
    total_padding = LOCALIZER_SHAPE - core_shape
    pad_before = total_padding // 2
    pad_after = total_padding - pad_before
    canvas = np.full(tuple(LOCALIZER_SHAPE), int(HU_MIN), dtype=np.int16)
    insertion = tuple(
        slice(int(pad_before[a]), int(pad_before[a] + core_shape[a]))
        for a in range(3)
    )
    canvas[insertion] = core
    return {
        "canvas": canvas,
        "effective_spacing": effective_spacing,
        "core_shape": core_shape,
        "pad_before": pad_before,
        "pad_after": pad_after,
    }


def canvas_to_canonical(
    center_normalized,
    canonical_shape,
    canonical_spacing,
    effective_spacing,
    core_shape,
    pad_before,
):
    canvas_voxel = center_normalized * (LOCALIZER_SHAPE.astype(float) - 1.0)
    core_voxel = canvas_voxel - pad_before
    scale = effective_spacing / canonical_spacing
    offset = (
        (canonical_shape - 1.0) / 2.0
        - scale * ((core_shape - 1.0) / 2.0)
    )
    return scale * core_voxel + offset


def create_centered_crop_affine(canonical_affine, center_voxel):
    center_voxel = np.asarray(center_voxel, dtype=float)
    center_world = nib.affines.apply_affine(canonical_affine, center_voxel)
    directions = canonical_affine[:3, :3].copy()
    norms = np.linalg.norm(directions, axis=0)
    if not np.isfinite(norms).all() or np.any(norms <= 0):
        raise RuntimeError("Invalid canonical affine directions.")
    directions /= norms
    crop_affine = np.eye(4, dtype=float)
    crop_affine[:3, :3] = directions * CROP_SPACING_MM
    crop_center = (CROP_SHAPE.astype(float) - 1.0) / 2.0
    crop_affine[:3, 3] = center_world - crop_affine[:3, :3] @ crop_center
    return crop_affine, center_world


def resample_ct_crop(ct_geometry, crop_affine, slope, intercept):
    ct_inverse = np.linalg.inv(ct_geometry["affine"])
    output = np.empty(tuple(CROP_SHAPE), dtype=np.int16)
    x_indices = np.arange(CROP_SHAPE[0], dtype=float)
    y_indices = np.arange(CROP_SHAPE[1], dtype=float)
    for z_start in range(0, CROP_SHAPE[2], RESAMPLE_Z_CHUNK):
        z_end = min(z_start + RESAMPLE_Z_CHUNK, CROP_SHAPE[2])
        z_indices = np.arange(z_start, z_end, dtype=float)
        grid = np.meshgrid(x_indices, y_indices, z_indices, indexing="ij")
        output_voxels = np.stack([axis.reshape(-1) for axis in grid], axis=0)
        homogeneous = np.vstack(
            [output_voxels, np.ones((1, output_voxels.shape[1]), dtype=float)]
        )
        world = crop_affine @ homogeneous
        coordinates = (ct_inverse @ world)[:3]
        values = map_coordinates(
            ct_geometry["array"],
            coordinates,
            order=1,
            mode="constant",
            cval=(HU_MIN - intercept) / slope,
            prefilter=False,
        )
        values = values * slope + intercept
        if not np.isfinite(values).all():
            raise RuntimeError("Non-finite resampled CT values.")
        values = np.clip(np.rint(values), HU_MIN, HU_MAX).astype(np.int16)
        output[:, :, z_start:z_end] = values.reshape(
            (CROP_SHAPE[0], CROP_SHAPE[1], z_end - z_start)
        )
    return output


# =============================================================================
# MODEL DEFINITIONS — IDENTICAL TO LOCKED TRAINING/VALIDATION STAGES
# =============================================================================


class LocalizerDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(group_count(out_channels), out_channels),
        )
        self.skip = nn.Conv3d(in_channels, out_channels, 1, stride=2, bias=False)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.activation(self.main(x) + self.skip(x))


class DilatedResidualBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.GroupNorm(group_count(channels), channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(channels, channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.GroupNorm(group_count(channels), channels),
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.activation(x + self.main(x))


class LightweightSpatialLocalizer3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            LocalizerDownBlock(1, 8),
            LocalizerDownBlock(8, 16),
            LocalizerDownBlock(16, 32),
            nn.Conv3d(32, 48, 3, padding=1, bias=False),
            nn.GroupNorm(group_count(48), 48),
            nn.SiLU(inplace=True),
            DilatedResidualBlock(48, 1),
            DilatedResidualBlock(48, 2),
            DilatedResidualBlock(48, 4),
        )
        self.heatmap_head = nn.Conv3d(48, 1, 1)
        self.bbox_head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(48, 32),
            nn.SiLU(inplace=True),
            nn.Dropout(0.10),
            nn.Linear(32, 3),
            nn.Sigmoid(),
        )
        axes = [torch.linspace(0.0, 1.0, steps=n) for n in LOCALIZER_HEATMAP_SHAPE]
        grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=0)
        self.register_buffer("coordinate_grid", grid.unsqueeze(0), persistent=False)

    def forward(self, image):
        features = self.encoder(image)
        logits = self.heatmap_head(features)
        probabilities = torch.softmax(logits.flatten(start_dim=2), dim=-1).reshape_as(logits)
        center = torch.sum(probabilities * self.coordinate_grid, dim=(2, 3, 4))
        bbox_size = self.bbox_head(features)
        return center, bbox_size, logits, probabilities


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
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv3d(
            in_channels, out_channels, 1, bias=False
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.activation(self.main(x) + self.skip(x))


class DownBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, stride=2, padding=1, bias=False),
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
        self.fuse = ResidualBlock3D(out_channels + skip_channels, out_channels)

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
# LOCKED INFERENCE AND CANDIDATE GENERATION
# =============================================================================


def make_starts(volume_size, patch_size, overlap):
    stride = max(1, int(round(patch_size * (1.0 - overlap))))
    if volume_size <= patch_size:
        return [0]
    starts = list(range(0, volume_size - patch_size + 1, stride))
    final_start = volume_size - patch_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


WINDOW_STARTS = [
    (x, y, z)
    for x in make_starts(CROP_SHAPE[0], PATCH_SHAPE[0], SLIDING_OVERLAP)
    for y in make_starts(CROP_SHAPE[1], PATCH_SHAPE[1], SLIDING_OVERLAP)
    for z in make_starts(CROP_SHAPE[2], PATCH_SHAPE[2], SLIDING_OVERLAP)
]


@torch.no_grad()
def infer_localizer(model, canvas, device):
    image = torch.from_numpy(
        np.clip((canvas.astype(np.float32) - HU_CENTER) / HU_HALF_WIDTH, -1.0, 1.0)
    )[None, None].to(device)
    with torch.amp.autocast("cuda", dtype=torch.float16, enabled=True):
        center, bbox, _logits, probabilities = model(image)
    flat = probabilities.float().flatten(start_dim=1)
    maximum = float(flat.max().item())
    entropy = float(
        (-torch.sum(flat * torch.log(torch.clamp(flat, min=1e-12)), dim=1)
         / np.log(float(np.prod(LOCALIZER_HEATMAP_SHAPE))))
        .item()
    )
    return (
        center.float().cpu().numpy()[0],
        bbox.float().cpu().numpy()[0],
        maximum,
        entropy,
    )


@torch.no_grad()
def infer_segmentation(model, ct_hu, device):
    normalized = np.clip(
        (ct_hu.astype(np.float32) - HU_CENTER) / HU_HALF_WIDTH, -1.0, 1.0
    )
    probability_sum = np.zeros((2, *CROP_SHAPE), dtype=np.float32)
    count = np.zeros(tuple(CROP_SHAPE), dtype=np.float32)
    for offset in range(0, len(WINDOW_STARTS), SEGMENTATION_BATCH_SIZE):
        chunk = WINDOW_STARTS[offset : offset + SEGMENTATION_BATCH_SIZE]
        patches = []
        for start in chunk:
            end = np.asarray(start) + PATCH_SHAPE
            patch = normalized[
                start[0] : end[0], start[1] : end[1], start[2] : end[2]
            ]
            patches.append(np.ascontiguousarray(patch, dtype=np.float32))
        batch = torch.from_numpy(np.stack(patches)).unsqueeze(1).to(device)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=True):
            logits = model(batch)
            predicted = torch.sigmoid(logits).float().cpu().numpy()
        for local_index, start in enumerate(chunk):
            end = np.asarray(start) + PATCH_SHAPE
            slices = tuple(slice(int(start[a]), int(end[a])) for a in range(3))
            probability_sum[(slice(None),) + slices] += predicted[local_index]
            count[slices] += 1.0
        del batch, logits, predicted
    if np.any(count <= 0):
        raise RuntimeError("Sliding-window inference left uncovered voxels.")
    result = probability_sum / count[None]
    if result.shape != (2, *CROP_SHAPE) or not np.isfinite(result).all():
        raise RuntimeError("Invalid segmentation probability output.")
    return result


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
        "reference_localization_tolerance_mm": 5.0,
        "spacing_mm": CROP_SPACING_MM.tolist(),
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def generate_blind_candidates(probabilities, crop_affine):
    pancreas_seed = probabilities[0] >= PANCREAS_GATE_PROBABILITY
    pancreas_seed_present = bool(pancreas_seed.any())
    if pancreas_seed_present:
        gate_distance = distance_transform_edt(~pancreas_seed, sampling=CROP_SPACING_MM)
        anatomical_gate = gate_distance <= PANCREAS_GATE_MARGIN_MM
    else:
        anatomical_gate = np.ones(tuple(CROP_SHAPE), dtype=bool)
    sigma_voxels = tuple(
        float(LESION_SMOOTHING_SIGMA_MM / spacing) for spacing in CROP_SPACING_MM
    )
    smoothed = gaussian_filter(
        probabilities[1], sigma=sigma_voxels, mode="nearest"
    ).astype(np.float32, copy=False)
    local_maximum = maximum_filter(
        smoothed, size=odd_filter_size(NMS_RADIUS_MM, CROP_SPACING_MM), mode="nearest"
    )
    peak_mask = (
        anatomical_gate
        & (smoothed >= MINIMUM_CANDIDATE_SCORE)
        & np.isclose(smoothed, local_maximum, rtol=0.0, atol=1e-7)
    )
    plateau_labels, _ = connected_components(
        peak_mask, structure=np.ones((3, 3, 3), dtype=np.uint8)
    )
    candidates = []
    for plateau_id, object_slices in enumerate(find_objects(plateau_labels), start=1):
        if object_slices is None:
            continue
        component = plateau_labels[object_slices] == plateau_id
        local_scores = np.where(component, smoothed[object_slices], -np.inf)
        local_coordinate = np.asarray(
            np.unravel_index(int(np.argmax(local_scores)), local_scores.shape), dtype=int
        )
        coordinate = np.asarray(
            [local_coordinate[a] + object_slices[a].start for a in range(3)], dtype=int
        )
        candidates.append(
            {
                "coordinate": coordinate,
                "score": float(smoothed[tuple(coordinate)]),
                "raw_probability": float(probabilities[1][tuple(coordinate)]),
                "world": nib.affines.apply_affine(crop_affine, coordinate),
            }
        )
    candidates.sort(key=lambda item: item["score"], reverse=True)
    candidates = candidates[:MAXIMUM_CANDIDATES_PER_CASE]
    for rank, item in enumerate(candidates, start=1):
        item["rank"] = rank
    del smoothed, local_maximum, peak_mask, plateau_labels
    if pancreas_seed_present:
        del gate_distance
    return candidates, pancreas_seed_present


# =============================================================================
# INPUT LOCKS AND BLINDING GATES
# =============================================================================


print("=" * 124)
print("STAGE R1H — RESUMABLE BLIND EXTERNAL-TEST INFERENCE AND PREDICTION FREEZE")
print("=" * 124)

required_paths = [
    LOCKED_SPLIT_PATH,
    REMOTE_MEMBER_PATH,
    REMOTE_SOURCE_PATH,
    LOCALIZER_PROTOCOL_PATH,
    CROP_PROTOCOL_PATH,
    PREFREEZE_PATH,
    PREFREEZE_MANIFEST_PATH,
    PREFREEZE_AUDIT_PATH,
    FROC_PROTOCOL_PATH,
    R1D_AUDIT_PATH,
    LOCALIZER_CHECKPOINT_PATH,
]
for required in required_paths:
    if not required.exists():
        raise FileNotFoundError(f"Required locked input missing:\n{required}")
if not torch.cuda.is_available():
    raise RuntimeError("Stage R1H requires a CUDA GPU.")

PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
QC_DIR.mkdir(parents=True, exist_ok=True)
META_DIR.mkdir(parents=True, exist_ok=True)
set_determinism()
device = torch.device("cuda")

with open(LOCALIZER_PROTOCOL_PATH, "r", encoding="utf-8") as file:
    localizer_protocol = json.load(file)
with open(CROP_PROTOCOL_PATH, "r", encoding="utf-8") as file:
    crop_protocol = json.load(file)
with open(PREFREEZE_PATH, "r", encoding="utf-8") as file:
    prefreeze = json.load(file)
with open(PREFREEZE_AUDIT_PATH, "r", encoding="utf-8") as file:
    prefreeze_audit = json.load(file)
with open(FROC_PROTOCOL_PATH, "r", encoding="utf-8") as file:
    froc_protocol = json.load(file)
with open(R1D_AUDIT_PATH, "r", encoding="utf-8") as file:
    r1d_audit = json.load(file)

if localizer_protocol.get("selected_candidate") != "F1_96x96x160":
    raise RuntimeError("F1 localizer geometry is not locked.")
if crop_protocol.get("selected_candidate") != "E5_240x192x128":
    raise RuntimeError("E5 deployment crop is not locked.")
if prefreeze.get("selected_arm") != "SSL_INITIALIZED":
    raise RuntimeError("Stage R1E did not freeze the SSL-initialized arm.")
if prefreeze_audit.get("all_checks_pass") is not True:
    raise RuntimeError("Stage R1E pre-test freeze audit did not pass.")
if froc_protocol.get("protocol_locked") is not True:
    raise RuntimeError("Stage R1D FROC protocol is not locked.")
if r1d_audit.get("all_checks_pass") is not True:
    raise RuntimeError("Stage R1D canonical audit did not pass.")

candidate_policy = froc_protocol["candidate_generation"]
locked_values = {
    "pancreas_probability_gate": PANCREAS_GATE_PROBABILITY,
    "pancreas_gate_expansion_mm": PANCREAS_GATE_MARGIN_MM,
    "lesion_probability_smoothing_sigma_mm": LESION_SMOOTHING_SIGMA_MM,
    "non_maximum_suppression_radius_mm": NMS_RADIUS_MM,
    "minimum_candidate_score": MINIMUM_CANDIDATE_SCORE,
    "maximum_candidates_per_case": MAXIMUM_CANDIDATES_PER_CASE,
}
for name, expected in locked_values.items():
    observed = candidate_policy.get(name)
    if isinstance(expected, int):
        match = int(observed) == expected
    else:
        match = np.isclose(float(observed), float(expected), rtol=0, atol=1e-8)
    if not match:
        raise RuntimeError(f"Locked candidate policy changed: {name}")
if candidate_policy.get("protocol_signature") != protocol_signature():
    raise RuntimeError("Stage R1D candidate protocol signature changed.")

deployment_threshold = float(froc_protocol["calibrated_probability_threshold"])
if not np.isfinite(deployment_threshold) or not 0 <= deployment_threshold <= 1:
    raise RuntimeError("Invalid locked deployment threshold.")
prefreeze_threshold = prefreeze.get(
    "calibrated_threshold",
    prefreeze.get(
        "calibrated_probability_threshold",
        prefreeze.get("locked_deployment_threshold"),
    ),
)
if prefreeze_threshold is None or not np.isclose(
    float(prefreeze_threshold), deployment_threshold, rtol=0, atol=1e-8
):
    raise RuntimeError("R1D and R1E deployment thresholds do not match.")

# Read only identity/partition fields. Diagnostic labels are intentionally not
# loaded into memory during Stage R1H.
split_header = pd.read_csv(LOCKED_SPLIT_PATH, nrows=0).columns.tolist()
split_required = {"study_id", "patient_id", "locked_partition", "source_group"}
if not split_required.issubset(split_header):
    raise RuntimeError(f"Locked split missing columns: {sorted(split_required - set(split_header))}")
split = pd.read_csv(LOCKED_SPLIT_PATH, usecols=sorted(split_required), dtype=str)
split["study_id"] = normalize_id_series(split["study_id"])
split["patient_id"] = normalize_id_series(split["patient_id"])
split["locked_partition"] = split["locked_partition"].astype(str).str.strip()
split["source_group"] = split["source_group"].astype(str).str.strip()
external_test = split.loc[
    split["locked_partition"].astype(str).str.strip().isin(
        EXPECTED_PARTITION_COUNTS
    )
].copy()
external_test = external_test.sort_values("study_id").reset_index(drop=True)
if len(external_test) != EXPECTED_EXTERNAL_TEST_CASES:
    raise RuntimeError(
        f"Expected {EXPECTED_EXTERNAL_TEST_CASES} external-test studies; observed {len(external_test)}."
    )
if external_test["study_id"].nunique() != EXPECTED_EXTERNAL_TEST_CASES:
    raise RuntimeError("External-test study IDs are not unique.")
partition_counts = external_test["locked_partition"].value_counts().to_dict()
source_counts = external_test["source_group"].value_counts().to_dict()
if partition_counts != EXPECTED_PARTITION_COUNTS:
    raise RuntimeError(
        f"External partition counts changed: {partition_counts}"
    )
if source_counts != EXPECTED_SOURCE_COUNTS:
    raise RuntimeError(f"External source counts changed: {source_counts}")

members = pd.read_csv(REMOTE_MEMBER_PATH, dtype={"study_id": str})
members["study_id"] = normalize_id_series(members["study_id"])
member_required = {
    "study_id", "batch_number", "member_name", "compressed_size_bytes",
    "uncompressed_size_bytes", "compression_type", "crc32_hex", "local_header_offset",
}
if not member_required.issubset(members.columns):
    raise RuntimeError("Remote member inventory schema changed.")
members = members.loc[members["study_id"].isin(external_test["study_id"])].copy()
if len(members) != EXPECTED_EXTERNAL_TEST_CASES or members["study_id"].nunique() != EXPECTED_EXTERNAL_TEST_CASES:
    raise RuntimeError("Remote inventory does not map every external-test study exactly once.")

sources = pd.read_csv(REMOTE_SOURCE_PATH)
if not {"batch_number", "record_id", "archive_name"}.issubset(sources.columns):
    raise RuntimeError("Remote archive-source schema changed.")
sources["batch_number"] = sources["batch_number"].astype(int)
if sources["batch_number"].nunique() != 4:
    raise RuntimeError("All four PANORAMA source batches are required.")

selected_checkpoint_path = Path(prefreeze["selected_checkpoint_path"])
if not selected_checkpoint_path.exists():
    raise FileNotFoundError(f"Selected segmentation checkpoint missing: {selected_checkpoint_path}")
selected_checkpoint_hash = sha256_file(selected_checkpoint_path)
if selected_checkpoint_hash != str(prefreeze["selected_checkpoint_sha256"]):
    raise RuntimeError("Selected segmentation checkpoint changed after the R1E freeze.")
localizer_checkpoint_hash = sha256_file(LOCALIZER_CHECKPOINT_PATH)

localizer_checkpoint = torch.load(
    LOCALIZER_CHECKPOINT_PATH, map_location=device, weights_only=False
)
localizer = LightweightSpatialLocalizer3D().to(device)
localizer.load_state_dict(localizer_checkpoint["model_state"], strict=True)
localizer.eval()
if sum(p.numel() for p in localizer.parameters()) != 471764:
    raise RuntimeError("Localizer parameter count changed.")

segmentation_checkpoint = torch.load(
    selected_checkpoint_path, map_location=device, weights_only=False
)
if segmentation_checkpoint.get("arm") != "SSL_INITIALIZED":
    raise RuntimeError("Selected segmentation checkpoint arm changed.")
segmenter = LightweightDualHead3DSegmenter().to(device)
segmenter.load_state_dict(segmentation_checkpoint["model_state"], strict=True)
segmenter.eval()


# =============================================================================
# RESUME STATE
# =============================================================================


if RESUME_LEDGER_PATH.exists():
    ledger = pd.read_csv(RESUME_LEDGER_PATH, dtype={"study_id": str, "patient_id": str})
    ledger["study_id"] = normalize_id_series(ledger["study_id"])
    ledger = ledger.drop_duplicates("study_id", keep="last")
else:
    ledger = pd.DataFrame()

if CANDIDATE_LEDGER_PATH.exists():
    try:
        candidate_ledger = pd.read_csv(
            CANDIDATE_LEDGER_PATH, dtype={"study_id": str}
        )
        candidate_ledger["study_id"] = normalize_id_series(
            candidate_ledger["study_id"]
        )
    except pd.errors.EmptyDataError:
        candidate_ledger = pd.DataFrame()
else:
    candidate_ledger = pd.DataFrame()

candidate_columns = [
    "study_id",
    "patient_id",
    "candidate_rank",
    "candidate_score",
    "candidate_raw_probability",
    "candidate_x",
    "candidate_y",
    "candidate_z",
    "candidate_world_x_mm",
    "candidate_world_y_mm",
    "candidate_world_z_mm",
    "above_locked_deployment_threshold",
    "candidate_protocol_signature",
    "selected_checkpoint_sha256",
]
if not len(candidate_ledger):
    candidate_ledger = pd.DataFrame(columns=candidate_columns)

completed_ids = set()
if len(ledger) and "processing_complete" in ledger.columns:
    for _, row in ledger.loc[truth_flags(ledger["processing_complete"])].iterrows():
        if (
            str(row.get("selected_checkpoint_sha256")) == selected_checkpoint_hash
            and str(row.get("localizer_checkpoint_sha256")) == localizer_checkpoint_hash
            and str(row.get("candidate_protocol_signature")) == protocol_signature()
            and np.isclose(float(row.get("deployment_threshold")), deployment_threshold, atol=1e-8)
            and validate_prediction_file(
                row["output_path"], row["output_size_bytes"], row["output_sha256"]
            )
        ):
            completed_ids.add(str(row["study_id"]))

pending = external_test.loc[~external_test["study_id"].isin(completed_ids)].copy()
if MAX_CASES_PER_RUN is not None:
    pending = pending.head(int(MAX_CASES_PER_RUN))

expected_transfer = int(
    members.set_index("study_id").loc[pending["study_id"], "compressed_size_bytes"].sum()
) if len(pending) else 0

print()
print(f"PyTorch: {torch.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Locked external-test cases: {EXPECTED_EXTERNAL_TEST_CASES}")
print(f"Previously completed: {len(completed_ids)}")
print(f"Selected for this run: {len(pending)}")
print(f"Expected selected CT transfer: {expected_transfer / (1024 ** 3):.2f} GiB")
print("Diagnostic labels loaded: 0")
print("Test masks accessed: 0")
print(f"Locked candidate threshold: {deployment_threshold:.6f}")
print("Checkpoint frequency: after every case")


# =============================================================================
# BLIND INFERENCE LOOP
# =============================================================================


session = requests.Session()
session.headers.update({"User-Agent": "PDAC-Public-Q1-StageR1H/1.0"})
errors = []
completed_this_run = 0
transferred_this_run = 0

for run_order, (_, identity_row) in enumerate(pending.iterrows(), start=1):
    study_id = str(identity_row["study_id"])
    patient_id = str(identity_row["patient_id"])
    output_path = PREDICTION_DIR / f"{study_id}.npz"
    started = time.time()
    print(f"\n[{run_order}/{len(pending)}] Processing {study_id}")
    try:
        member_rows = members.loc[members["study_id"] == study_id]
        if len(member_rows) != 1:
            raise RuntimeError("Remote CT member is not unique.")
        member_row = member_rows.iloc[0]
        source_rows = sources.loc[
            sources["batch_number"] == int(member_row["batch_number"])
        ]
        if len(source_rows) != 1:
            raise RuntimeError("Remote archive source is not unique.")
        source_row = source_rows.iloc[0]

        with TemporaryDirectory(prefix=f"stageR1h_{study_id}_", dir=RUNTIME_ROOT) as temp_name:
            work_dir = Path(temp_name)
            acquisition = recover_zip_member(
                session, member_row, source_row, work_dir
            )
            transferred_this_run += int(acquisition["compressed_transfer_bytes"])
            ct_nii_path = work_dir / "ct.nii"
            decompress_gzip_to_nii(acquisition["inner_path"], ct_nii_path)
            ct_image = nib.load(str(ct_nii_path), mmap="r")
            if len(ct_image.shape) != 3:
                raise RuntimeError("CT must be three-dimensional.")
            if not np.isfinite(ct_image.affine).all() or abs(np.linalg.det(ct_image.affine[:3, :3])) <= 0:
                raise RuntimeError("CT affine is invalid.")
            ct_geometry = canonical_image_geometry(ct_image)
            slope, intercept = proxy_scaling(ct_image)

            localizer_data = ct_to_localizer_canvas(ct_geometry, slope, intercept)
            center_normalized, predicted_bbox, heatmap_max, heatmap_entropy = infer_localizer(
                localizer, localizer_data["canvas"], device
            )
            if not np.isfinite(center_normalized).all() or np.any(center_normalized < 0) or np.any(center_normalized > 1):
                raise RuntimeError("Localizer predicted a centre outside [0,1].")
            predicted_center = canvas_to_canonical(
                center_normalized,
                ct_geometry["shape"].astype(float),
                ct_geometry["spacing"],
                localizer_data["effective_spacing"],
                localizer_data["core_shape"].astype(float),
                localizer_data["pad_before"].astype(float),
            )
            crop_affine, center_world = create_centered_crop_affine(
                ct_geometry["affine"], predicted_center
            )
            ct_crop = resample_ct_crop(ct_geometry, crop_affine, slope, intercept)
            if ct_crop.shape != tuple(CROP_SHAPE) or ct_crop.dtype != np.int16:
                raise RuntimeError("E5 CT crop geometry or dtype changed.")
            if ct_crop.min() < HU_MIN or ct_crop.max() > HU_MAX:
                raise RuntimeError("E5 CT crop lies outside locked HU window.")

            probabilities = infer_segmentation(segmenter, ct_crop, device)
            candidates, pancreas_seed_present = generate_blind_candidates(
                probabilities, crop_affine
            )
            coordinates = np.asarray(
                [item["coordinate"] for item in candidates], dtype=np.int16
            ).reshape(-1, 3)
            scores = np.asarray(
                [item["score"] for item in candidates], dtype=np.float32
            )
            raw_scores = np.asarray(
                [item["raw_probability"] for item in candidates], dtype=np.float32
            )
            worlds = np.asarray(
                [item["world"] for item in candidates], dtype=np.float32
            ).reshape(-1, 3)
            quantized = np.rint(
                np.clip(probabilities, 0.0, 1.0) * PROBABILITY_SCALE
            ).astype(np.uint16)
            maximum_quantization_error = float(
                np.max(np.abs(quantized.astype(np.float32) / PROBABILITY_SCALE - probabilities))
            )
            if maximum_quantization_error > 1.0 / PROBABILITY_SCALE + 1e-7:
                raise RuntimeError("Probability quantization error exceeded its lock.")

            atomic_save_npz(
                output_path,
                study_id=np.asarray(study_id),
                pancreas_probability_uint16=quantized[0],
                lesion_probability_uint16=quantized[1],
                probability_scale=np.asarray(PROBABILITY_SCALE, dtype=np.float32),
                crop_shape=CROP_SHAPE.astype(np.int16),
                crop_spacing_mm=CROP_SPACING_MM.astype(np.float32),
                crop_affine=crop_affine.astype(np.float64),
                predicted_center_normalized=center_normalized.astype(np.float32),
                predicted_center_canonical=predicted_center.astype(np.float32),
                predicted_center_world_mm=center_world.astype(np.float64),
                predicted_bbox_normalized=predicted_bbox.astype(np.float32),
                candidate_coordinates_voxel=coordinates,
                candidate_scores=scores,
                candidate_raw_probabilities=raw_scores,
                candidate_world_mm=worlds,
                deployment_threshold=np.asarray(deployment_threshold, dtype=np.float32),
                candidate_protocol_signature=np.asarray(protocol_signature()),
                segmentation_checkpoint_sha256=np.asarray(selected_checkpoint_hash),
                localizer_checkpoint_sha256=np.asarray(localizer_checkpoint_hash),
            )

            # Immediate independent structural verification before the case is
            # admitted to the durable blind-prediction ledger.
            with np.load(output_path, allow_pickle=False) as frozen:
                required_keys = {
                    "study_id", "pancreas_probability_uint16", "lesion_probability_uint16",
                    "probability_scale", "crop_shape", "crop_spacing_mm", "crop_affine",
                    "predicted_center_normalized", "predicted_center_canonical",
                    "predicted_center_world_mm", "predicted_bbox_normalized",
                    "candidate_coordinates_voxel", "candidate_scores",
                    "candidate_raw_probabilities", "candidate_world_mm",
                    "deployment_threshold", "candidate_protocol_signature",
                    "segmentation_checkpoint_sha256", "localizer_checkpoint_sha256",
                }
                if set(frozen.files) != required_keys:
                    raise RuntimeError("Frozen prediction key set is not exact.")
                if frozen["pancreas_probability_uint16"].shape != tuple(CROP_SHAPE):
                    raise RuntimeError("Frozen pancreas probability shape changed.")
                if frozen["lesion_probability_uint16"].shape != tuple(CROP_SHAPE):
                    raise RuntimeError("Frozen lesion probability shape changed.")
                if frozen["candidate_scores"].shape[0] != len(candidates):
                    raise RuntimeError("Frozen candidate count changed.")

            output_size = output_path.stat().st_size
            output_hash = sha256_file(output_path)
            case_candidate_rows = []
            for item in candidates:
                coordinate = item["coordinate"]
                world = item["world"]
                case_candidate_rows.append(
                    {
                        "study_id": study_id,
                        "patient_id": patient_id,
                        "candidate_rank": int(item["rank"]),
                        "candidate_score": float(item["score"]),
                        "candidate_raw_probability": float(item["raw_probability"]),
                        "candidate_x": int(coordinate[0]),
                        "candidate_y": int(coordinate[1]),
                        "candidate_z": int(coordinate[2]),
                        "candidate_world_x_mm": float(world[0]),
                        "candidate_world_y_mm": float(world[1]),
                        "candidate_world_z_mm": float(world[2]),
                        "above_locked_deployment_threshold": bool(
                            item["score"] >= deployment_threshold
                        ),
                        "candidate_protocol_signature": protocol_signature(),
                        "selected_checkpoint_sha256": selected_checkpoint_hash,
                    }
                )
            if len(candidate_ledger):
                candidate_ledger = candidate_ledger.loc[
                    candidate_ledger["study_id"].astype(str) != study_id
                ].copy()
            if case_candidate_rows:
                candidate_ledger = pd.concat(
                    [candidate_ledger, pd.DataFrame(case_candidate_rows)], ignore_index=True
                )
            if len(candidate_ledger):
                candidate_ledger = candidate_ledger.sort_values(
                    ["study_id", "candidate_rank"]
                ).reset_index(drop=True)
            atomic_write_csv(candidate_ledger, CANDIDATE_LEDGER_PATH)

            maximum_score = float(scores[0]) if len(scores) else 0.0
            ledger_row = {
                "study_id": study_id,
                "patient_id": patient_id,
                "locked_partition": str(identity_row["locked_partition"]),
                "source_group": str(identity_row["source_group"]),
                "batch_number": int(member_row["batch_number"]),
                "member_name": str(member_row["member_name"]),
                "compressed_transfer_bytes": int(acquisition["compressed_transfer_bytes"]),
                "recovered_size_bytes": int(acquisition["recovered_size_bytes"]),
                "crc32_hex": acquisition["crc32_hex"],
                "raw_ct_shape_json": json.dumps([int(v) for v in ct_image.shape]),
                "raw_ct_spacing_mm_json": json_vector(nib.affines.voxel_sizes(ct_image.affine)),
                "raw_ct_orientation": ct_geometry["raw_orientation"],
                "canonical_ct_shape_json": json.dumps(ct_geometry["shape"].astype(int).tolist()),
                "canonical_ct_spacing_mm_json": json_vector(ct_geometry["spacing"]),
                "effective_localizer_spacing_mm": float(localizer_data["effective_spacing"]),
                "localizer_core_shape_json": json.dumps(localizer_data["core_shape"].astype(int).tolist()),
                "localizer_pad_before_json": json.dumps(localizer_data["pad_before"].astype(int).tolist()),
                "predicted_center_normalized_json": json_vector(center_normalized),
                "predicted_center_canonical_json": json_vector(predicted_center),
                "predicted_center_world_mm_json": json_vector(center_world),
                "predicted_bbox_normalized_json": json_vector(predicted_bbox),
                "localizer_heatmap_max_probability": heatmap_max,
                "localizer_heatmap_normalized_entropy": heatmap_entropy,
                "crop_shape_json": json.dumps(CROP_SHAPE.tolist()),
                "crop_spacing_mm_json": json_vector(CROP_SPACING_MM),
                "crop_affine_json": json_matrix(crop_affine),
                "pancreas_seed_present": pancreas_seed_present,
                "generated_candidates": int(len(candidates)),
                "maximum_candidate_score": maximum_score,
                "candidates_above_deployment_threshold": int(np.sum(scores >= deployment_threshold)),
                "case_positive_at_locked_threshold": bool(maximum_score >= deployment_threshold),
                "deployment_threshold": deployment_threshold,
                "maximum_probability_quantization_error": maximum_quantization_error,
                "candidate_protocol_signature": protocol_signature(),
                "selected_checkpoint_sha256": selected_checkpoint_hash,
                "localizer_checkpoint_sha256": localizer_checkpoint_hash,
                "output_path": str(output_path),
                "output_size_bytes": int(output_size),
                "output_sha256": output_hash,
                "test_ct_accessed_for_blind_inference": True,
                "test_mask_accessed": False,
                "diagnostic_label_accessed": False,
                "processing_complete": True,
                "processing_seconds": float(time.time() - started),
                "completed_at_utc": utc_now(),
            }
            ledger = replace_rows(ledger, pd.DataFrame([ledger_row]), "study_id")
            ledger = ledger.sort_values("study_id").reset_index(drop=True)
            atomic_write_csv(ledger, RESUME_LEDGER_PATH)

            del probabilities, quantized, ct_crop, candidates, ct_geometry
            del localizer_data, ct_image
            gc.collect()
            torch.cuda.empty_cache()

        completed_this_run += 1
        print(
            f"    PASS — candidates={ledger_row['generated_candidates']} — "
            f"maximum score={ledger_row['maximum_candidate_score']:.6f} — "
            f"durable {len(completed_ids) + completed_this_run}/{EXPECTED_EXTERNAL_TEST_CASES}"
        )
    except Exception as error:
        errors.append(
            {
                "study_id": study_id,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "failed_at_utc": utc_now(),
            }
        )
        atomic_write_csv(pd.DataFrame(errors), ERROR_PATH)
        gc.collect()
        torch.cuda.empty_cache()
        print(f"    FAIL — {type(error).__name__}: {error}")
        print("Processing stopped; all previously accepted cases remain durable.")
        break


# =============================================================================
# FINAL FREEZE OR RESUMABLE HANDOFF
# =============================================================================


completion_flags = truth_flags(ledger["processing_complete"]) if len(ledger) else pd.Series(dtype=bool)
valid_rows = []
for _, row in ledger.loc[completion_flags].iterrows() if len(ledger) else []:
    if (
        str(row.get("selected_checkpoint_sha256")) == selected_checkpoint_hash
        and str(row.get("localizer_checkpoint_sha256")) == localizer_checkpoint_hash
        and str(row.get("candidate_protocol_signature")) == protocol_signature()
        and validate_prediction_file(row["output_path"], row["output_size_bytes"], row["output_sha256"])
    ):
        valid_rows.append(row)
valid_ledger = pd.DataFrame(valid_rows)
if len(valid_ledger):
    valid_ledger["study_id"] = normalize_id_series(valid_ledger["study_id"])
    valid_ledger = valid_ledger.drop_duplicates("study_id", keep="last")

completed_total = len(valid_ledger)
remaining = EXPECTED_EXTERNAL_TEST_CASES - completed_total
print("\n" + "-" * 124)
print("STAGE R1H RUN RESULT")
print("-" * 124)
print(f"Completed this run: {completed_this_run}")
print(f"Durably completed: {completed_total}/{EXPECTED_EXTERNAL_TEST_CASES}")
print(f"Remaining: {remaining}")
print(f"Transferred this run: {transferred_this_run / (1024 ** 3):.3f} GiB")
print("Diagnostic labels accessed: 0")
print("Test masks accessed: 0")
print(f"Resume checkpoint:\n{RESUME_LEDGER_PATH}")

if errors:
    print("=" * 124)
    print("STAGE R1H STATUS: STOPPED_WITH_ERROR")
    print("=" * 124)
    raise RuntimeError("Stage R1H stopped on an error; accepted cases remain checkpointed.")

if remaining > 0:
    run_audit = {
        "stage": "R1H",
        "created_at_utc": utc_now(),
        "status": "MORE_CASES_REQUIRED",
        "durably_completed": completed_total,
        "remaining": remaining,
        "diagnostic_labels_accessed": 0,
        "test_masks_accessed": 0,
        "prediction_freeze_complete": False,
    }
    atomic_write_json(run_audit, AUDIT_PATH)
    print("=" * 124)
    print("STAGE R1H STATUS: MORE_CASES_REQUIRED")
    print("=" * 124)
    raise SystemExit(0)

expected_ids = set(external_test["study_id"])
observed_ids = set(valid_ledger["study_id"])
candidate_ids = set(candidate_ledger["study_id"].astype(str)) if len(candidate_ledger) else set()
files = list(PREDICTION_DIR.glob("*.npz"))
observed_partition_counts = valid_ledger["locked_partition"].value_counts().to_dict()
observed_source_counts = valid_ledger["source_group"].value_counts().to_dict()
readiness_checks = {
    "Exactly 274 blind prediction rows": completed_total == EXPECTED_EXTERNAL_TEST_CASES,
    "Exactly 274 unique blind study IDs": valid_ledger["study_id"].nunique() == EXPECTED_EXTERNAL_TEST_CASES,
    "Exactly 194 MSD and 80 NIH predictions": observed_partition_counts == EXPECTED_PARTITION_COUNTS,
    "Source-separated counts remain locked": observed_source_counts == EXPECTED_SOURCE_COUNTS,
    "Prediction IDs exactly match the external-test lock": observed_ids == expected_ids,
    "Persistent folder contains exactly 274 NPZ files": len(files) == EXPECTED_EXTERNAL_TEST_CASES,
    "Persistent filenames exactly match the external-test lock": {p.stem for p in files} == expected_ids,
    "Every prediction file passed hash verification": all(
        validate_prediction_file(row["output_path"], row["output_size_bytes"], row["output_sha256"])
        for _, row in valid_ledger.iterrows()
    ),
    "All cases used the locked SSL checkpoint": set(valid_ledger["selected_checkpoint_sha256"]) == {selected_checkpoint_hash},
    "All cases used the locked localizer checkpoint": set(valid_ledger["localizer_checkpoint_sha256"]) == {localizer_checkpoint_hash},
    "All cases used the locked candidate protocol": set(valid_ledger["candidate_protocol_signature"]) == {protocol_signature()},
    "Every candidate row belongs to a locked test study": candidate_ids.issubset(expected_ids),
    "All probability quantization errors are bounded": (valid_ledger["maximum_probability_quantization_error"].astype(float) <= 1.0 / PROBABILITY_SCALE + 1e-7).all(),
    "Every case produced a pancreas gate seed": truth_flags(valid_ledger["pancreas_seed_present"]).all(),
    "No diagnostic label was accessed": not truth_flags(valid_ledger["diagnostic_label_accessed"]).any(),
    "No test mask was accessed": not truth_flags(valid_ledger["test_mask_accessed"]).any(),
    "No threshold was changed on test": np.allclose(valid_ledger["deployment_threshold"].astype(float), deployment_threshold, rtol=0, atol=1e-8),
}
readiness_checks = {
    name: bool(passed) for name, passed in readiness_checks.items()
}
all_checks_pass = bool(all(readiness_checks.values()))

manifest_columns = [
    "study_id", "patient_id", "locked_partition", "source_group", "batch_number",
    "output_path", "output_size_bytes", "output_sha256", "generated_candidates",
    "maximum_candidate_score", "candidates_above_deployment_threshold",
    "case_positive_at_locked_threshold", "deployment_threshold",
    "candidate_protocol_signature", "selected_checkpoint_sha256",
    "localizer_checkpoint_sha256", "completed_at_utc",
]
manifest = valid_ledger[manifest_columns].sort_values("study_id").reset_index(drop=True)
atomic_write_csv(manifest, MANIFEST_PATH)

freeze = {
    "stage": "R1H",
    "created_at_utc": utc_now(),
    "prediction_freeze_complete": all_checks_pass,
    "inference_scope": "external_test_CT_only_blind_inference",
    "external_test_cases": EXPECTED_EXTERNAL_TEST_CASES,
    "partition_counts": observed_partition_counts,
    "source_group_counts": observed_source_counts,
    "diagnostic_labels_accessed": 0,
    "test_masks_accessed": 0,
    "selected_arm": "SSL_INITIALIZED",
    "selected_checkpoint_path": str(selected_checkpoint_path),
    "selected_checkpoint_sha256": selected_checkpoint_hash,
    "localizer_checkpoint_path": str(LOCALIZER_CHECKPOINT_PATH),
    "localizer_checkpoint_sha256": localizer_checkpoint_hash,
    "crop_shape": CROP_SHAPE.tolist(),
    "crop_spacing_mm": CROP_SPACING_MM.tolist(),
    "candidate_protocol_signature": protocol_signature(),
    "locked_deployment_threshold": deployment_threshold,
    "test_specific_model_adjustment": "prohibited_and_not_performed",
    "test_specific_threshold_adjustment": "prohibited_and_not_performed",
    "probability_storage": {
        "dtype": "uint16",
        "scale": PROBABILITY_SCALE,
        "decoding": "probability_uint16 / 65535",
    },
    "prediction_manifest_path": str(MANIFEST_PATH),
    "candidate_ledger_path": str(CANDIDATE_LEDGER_PATH),
    "resume_ledger_path": str(RESUME_LEDGER_PATH),
}
atomic_write_json(freeze, FREEZE_PATH)

audit = {
    "stage": "R1H",
    "created_at_utc": utc_now(),
    "result": "PASS_BLIND_EXTERNAL_TEST_PREDICTIONS_FROZEN" if all_checks_pass else "FAIL",
    "all_checks_pass": all_checks_pass,
    "readiness_checks": readiness_checks,
    "durably_completed": completed_total,
    "candidate_rows": int(len(candidate_ledger)),
    "mean_candidates_per_case": float(len(candidate_ledger) / EXPECTED_EXTERNAL_TEST_CASES),
    "case_positive_fraction_at_locked_threshold": float(
        truth_flags(valid_ledger["case_positive_at_locked_threshold"]).mean()
    ),
    "diagnostic_labels_accessed": 0,
    "test_masks_accessed": 0,
    "prediction_manifest_path": str(MANIFEST_PATH),
    "prediction_freeze_path": str(FREEZE_PATH),
}
atomic_write_json(audit, AUDIT_PATH)

print("\n" + "-" * 124)
print("READINESS CHECKS")
print("-" * 124)
for name, passed in readiness_checks.items():
    print(f"  {name}: {passed}")
print(f"\nPrediction manifest:\n{MANIFEST_PATH}")
print(f"Prediction freeze:\n{FREEZE_PATH}")
print(f"Candidate ledger:\n{CANDIDATE_LEDGER_PATH}")
print(f"Audit:\n{AUDIT_PATH}")
print("=" * 124)
print(
    "STAGE R1H RESULT: "
    + ("PASS — BLIND EXTERNAL-TEST PREDICTIONS FROZEN" if all_checks_pass else "FAIL")
)
print("=" * 124)
if not all_checks_pass:
    raise RuntimeError("Stage R1H failed one or more final readiness checks.")
