from pathlib import Path
from datetime import datetime, timezone
from tempfile import TemporaryDirectory
import binascii
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
from scipy.ndimage import map_coordinates


# =============================================================================
# LOCKED PROJECT INPUTS
# =============================================================================

PROJECT_ROOT = Path(os.environ.get("PDAC_PROJECT_ROOT", "/content/drive/MyDrive/PDAC_Public_Q1_Project"))
RUNTIME_ROOT = Path(os.environ.get("PDAC_RUNTIME_ROOT", "/content"))
RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
RAW_ROOT = PROJECT_ROOT / "00_Raw" / "PANORAMA"
META_DIR = PROJECT_ROOT / "01_Metadata"
QC_DIR = PROJECT_ROOT / "02_Quality_Control"
PROCESSED_DIR = (
    PROJECT_ROOT
    / "03_Processed"
    / "E5_240x192x128"
)

DEVELOPMENT_MANIFEST_PATH = (
    META_DIR / "stage1b_localizer_preprocessed_manifest.csv"
)
STAGE1A_LEDGER_PATH = (
    QC_DIR / "stage1a_localizer_preprocessing_ledger.csv"
)
PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "04_Models"
    / "Localizer"
    / "F1_96x96x160"
    / "stage2e_all_development_localizer_predictions.csv"
)
STAGE2E_PROTOCOL_PATH = (
    META_DIR / "stage2e_development_localizer_inference_protocol.json"
)
FINAL_CROP_PROTOCOL_PATH = (
    META_DIR / "stage2d_final_deployment_crop_geometry_protocol.json"
)
REMOTE_MEMBER_INVENTORY_PATH = (
    META_DIR / "stage0i_panorama_remote_zip_member_inventory.csv"
)
REMOTE_SOURCE_AUDIT_PATH = (
    META_DIR / "stage0i_panorama_remote_archive_source_audit.csv"
)

LEDGER_PATH = (
    QC_DIR / "stage3a_e5_crop_preprocessing_ledger.csv"
)
RUN_AUDIT_PATH = (
    QC_DIR / "stage3a_e5_crop_preprocessing_run_audit.json"
)
DATASET_PROTOCOL_PATH = (
    META_DIR / "stage3a_e5_crop_preprocessing_protocol.json"
)


# =============================================================================
# LOCKED PREPROCESSING POLICY
# =============================================================================

EXPECTED_CASES = 1671
EXPECTED_TRAIN = 1376
EXPECTED_VALIDATION = 295

TARGET_SHAPE = np.asarray([240, 192, 128], dtype=int)
TARGET_SPACING_MM = np.asarray([1.25, 1.25, 2.0], dtype=float)
EXPECTED_FOV_MM = TARGET_SHAPE * TARGET_SPACING_MM

HU_MIN = -200.0
HU_MAX = 300.0
ALLOWED_LABELS = set(range(7))
CANONICAL_AXCODES = ("L", "P", "S")

# None means process every pending case. To perform a short diagnostic run,
# temporarily set this to an integer; the durable ledger makes later resumption
# automatic.
MAX_CASES_PER_RUN = None

RESAMPLE_Z_CHUNK = 8
MASK_SCAN_Z_CHUNK = 8
HTTP_CHUNK_BYTES = 8 * 1024 * 1024
HTTP_CONNECT_TIMEOUT = 30
HTTP_READ_TIMEOUT = 180
HTTP_MAX_ATTEMPTS = 12
SEED = 20260728

APPROVED_GEOMETRY_STATUSES = {
    "EXACT_PHYSICAL_GEOMETRY_MATCH",
    "PHYSICAL_GEOMETRY_MATCH_WITHIN_NUMERICAL_HEADER_TOLERANCE",
    "KNOWN_VISUALLY_CONFIRMED_INDEX_ALIGNED_HEADER_MISMATCH",
}
KNOWN_REHEADER_STATUS = (
    "KNOWN_VISUALLY_CONFIRMED_INDEX_ALIGNED_HEADER_MISMATCH"
)


# =============================================================================
# GENERAL UTILITIES
# =============================================================================


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def json_vector(values):
    return json.dumps(
        [float(value) for value in np.asarray(values).reshape(-1)]
    )


def json_matrix(values):
    return json.dumps(
        np.asarray(values, dtype=float).tolist()
    )


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


def disk_free_gib(path):
    return shutil.disk_usage(path).free / (1024 ** 3)


def normalize_study_id(series):
    return series.astype(str).str.strip()


def parse_crc32(value):
    text = str(value).strip().lower()
    if text.startswith("0x"):
        text = text[2:]
    return int(text, 16)


def exact_one_row(frame, study_id, name):
    rows = frame.loc[frame["study_id"] == study_id]
    if len(rows) != 1:
        raise RuntimeError(
            f"{study_id}: expected one {name} row; observed {len(rows)}."
        )
    return rows.iloc[0]


def replace_ledger_row(ledger, new_row):
    study_id = str(new_row["study_id"])
    if len(ledger):
        ledger = ledger.loc[
            ledger["study_id"].astype(str) != study_id
        ].copy()
    return pd.concat(
        [ledger, pd.DataFrame([new_row])],
        ignore_index=True,
    )


def validate_output_file(path, expected_size, expected_hash):
    if not path.exists():
        return False
    if path.stat().st_size != int(expected_size):
        return False
    return sha256_file(path) == str(expected_hash)


# =============================================================================
# SAFE REMOTE ZIP-MEMBER ACQUISITION
# =============================================================================


def candidate_endpoints(source_row):
    endpoints = []
    selected = str(source_row.get("selected_range_url", "")).strip()
    if selected and selected.lower() != "nan":
        endpoints.append(selected)

    record_id = int(source_row["record_id"])
    archive_name = str(source_row["archive_name"])
    endpoints.extend(
        [
            (
                f"https://zenodo.org/api/records/{record_id}/files/"
                f"{archive_name}/content"
            ),
            (
                f"https://zenodo.org/records/{record_id}/files/"
                f"{archive_name}?download=1"
            ),
        ]
    )

    unique = []
    for endpoint in endpoints:
        if endpoint not in unique:
            unique.append(endpoint)
    return unique


def parse_content_range(header_value):
    # Expected form: bytes START-END/TOTAL
    if not header_value:
        return None
    text = str(header_value).strip()
    if not text.lower().startswith("bytes "):
        return None
    try:
        interval = text.split(" ", 1)[1].split("/", 1)[0]
        start_text, end_text = interval.split("-", 1)
        return int(start_text), int(end_text)
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
                timeout=(
                    HTTP_CONNECT_TIMEOUT,
                    HTTP_READ_TIMEOUT,
                ),
                allow_redirects=True,
            )
            if response.status_code != 206:
                raise RuntimeError(
                    "Unsafe response rejected: "
                    f"HTTP {response.status_code}"
                )
            observed = parse_content_range(
                response.headers.get("Content-Range")
            )
            if observed is None or observed[0] != start:
                raise RuntimeError(
                    "Invalid Content-Range response."
                )
            data = response.content
            expected = end - start + 1
            if len(data) != expected:
                raise RuntimeError(
                    f"Short range: {len(data)}/{expected} bytes."
                )
            return data, endpoint
        except Exception as error:
            errors.append(f"{type(error).__name__}: {error}")
            time.sleep(min(2 ** min(attempt, 4), 20))
    raise RuntimeError(
        "All safe range endpoints failed: " + " | ".join(errors[-5:])
    )


def download_exact_range(
    session,
    endpoints,
    start,
    size,
    destination,
):
    end = start + size - 1
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
                    timeout=(
                        HTTP_CONNECT_TIMEOUT,
                        HTTP_READ_TIMEOUT,
                    ),
                    stream=True,
                    allow_redirects=True,
                ) as response:
                    if response.status_code != 206:
                        raise RuntimeError(
                            "Unsafe response rejected: "
                            f"HTTP {response.status_code}"
                        )
                    observed = parse_content_range(
                        response.headers.get("Content-Range")
                    )
                    if (
                        observed is None
                        or observed[0] != request_start
                        or observed[1] > end
                    ):
                        raise RuntimeError(
                            "Invalid Content-Range response."
                        )

                    bytes_before = completed
                    for chunk in response.iter_content(
                        chunk_size=HTTP_CHUNK_BYTES
                    ):
                        if not chunk:
                            continue
                        remaining = size - completed
                        if len(chunk) > remaining:
                            raise RuntimeError(
                                "Range response exceeded expected size."
                            )
                        output.write(chunk)
                        completed += len(chunk)
                    output.flush()
                    os.fsync(output.fileno())
                    selected_endpoint = endpoint

                    if completed == bytes_before:
                        raise RuntimeError(
                            "Range response returned no payload."
                        )

                attempts = 0
            except Exception:
                if attempts >= HTTP_MAX_ATTEMPTS:
                    raise
                time.sleep(min(2 ** min(attempts, 4), 20))

    if temporary.stat().st_size != size:
        raise RuntimeError(
            f"Compressed payload size mismatch: "
            f"{temporary.stat().st_size}/{size}"
        )
    os.replace(temporary, destination)
    return selected_endpoint


def recover_zip_member(
    session,
    member_row,
    source_row,
    work_dir,
):
    endpoints = candidate_endpoints(source_row)
    local_header_offset = int(member_row["local_header_offset"])

    header, header_endpoint = fetch_small_range(
        session,
        endpoints,
        local_header_offset,
        local_header_offset + 29,
    )
    fields = struct.unpack("<IHHHHHIIIHH", header)
    (
        signature,
        _version,
        flag_bits,
        compression_type,
        _mod_time,
        _mod_date,
        _local_crc32,
        _local_compressed_size,
        _local_uncompressed_size,
        filename_length,
        extra_length,
    ) = fields

    if signature != 0x04034B50:
        raise RuntimeError("Invalid ZIP local-header signature.")
    if flag_bits & 0x1:
        raise RuntimeError("Encrypted ZIP members are prohibited.")

    filename_bytes, _ = fetch_small_range(
        session,
        endpoints,
        local_header_offset + 30,
        local_header_offset + 30 + filename_length - 1,
    )
    local_filename = filename_bytes.decode("utf-8")
    expected_member_name = str(member_row["member_name"])
    if local_filename != expected_member_name:
        raise RuntimeError(
            "ZIP local filename differs from central inventory: "
            f"{local_filename} != {expected_member_name}"
        )

    expected_compression = int(member_row["compression_type"])
    if compression_type != expected_compression:
        raise RuntimeError("ZIP compression type mismatch.")

    compressed_size = int(member_row["compressed_size_bytes"])
    expected_uncompressed_size = int(
        member_row["uncompressed_size_bytes"]
    )
    expected_crc32 = parse_crc32(member_row["crc32_hex"])
    data_start = (
        local_header_offset
        + 30
        + filename_length
        + extra_length
    )

    compressed_path = work_dir / "member_payload.bin"
    inner_gzip_path = work_dir / "ct.nii.gz"
    download_endpoint = download_exact_range(
        session,
        endpoints,
        data_start,
        compressed_size,
        compressed_path,
    )

    crc32_value = 0
    recovered_size = 0
    decompressor = (
        zlib.decompressobj(-zlib.MAX_WBITS)
        if compression_type == 8
        else None
    )
    if compression_type not in {0, 8}:
        raise RuntimeError(
            f"Unsupported ZIP compression type: {compression_type}"
        )

    with open(compressed_path, "rb") as source, open(
        inner_gzip_path,
        "wb",
    ) as destination:
        while True:
            chunk = source.read(HTTP_CHUNK_BYTES)
            if not chunk:
                break
            recovered = (
                decompressor.decompress(chunk)
                if decompressor is not None
                else chunk
            )
            if recovered:
                destination.write(recovered)
                recovered_size += len(recovered)
                crc32_value = binascii.crc32(
                    recovered,
                    crc32_value,
                )
        if decompressor is not None:
            recovered = decompressor.flush()
            if recovered:
                destination.write(recovered)
                recovered_size += len(recovered)
                crc32_value = binascii.crc32(
                    recovered,
                    crc32_value,
                )
        destination.flush()
        os.fsync(destination.fileno())

    crc32_value &= 0xFFFFFFFF
    if recovered_size != expected_uncompressed_size:
        raise RuntimeError(
            "Recovered member size mismatch: "
            f"{recovered_size}/{expected_uncompressed_size}"
        )
    if crc32_value != expected_crc32:
        raise RuntimeError(
            f"CRC-32 mismatch: {crc32_value:08x}/"
            f"{expected_crc32:08x}"
        )

    with open(inner_gzip_path, "rb") as file:
        if file.read(2) != b"\x1f\x8b":
            raise RuntimeError("Recovered member is not gzip NIfTI.")

    compressed_path.unlink()
    return {
        "inner_gzip_path": inner_gzip_path,
        "compressed_transfer_bytes": compressed_size,
        "recovered_size_bytes": recovered_size,
        "crc32_hex": f"{crc32_value:08x}",
        "header_endpoint": header_endpoint,
        "download_endpoint": download_endpoint,
        "local_header_filename_matches": True,
    }


def decompress_gzip_to_nii(source, destination):
    with gzip.open(source, "rb") as compressed, open(
        destination,
        "wb",
    ) as uncompressed:
        shutil.copyfileobj(
            compressed,
            uncompressed,
            length=HTTP_CHUNK_BYTES,
        )
        uncompressed.flush()
        os.fsync(uncompressed.fileno())


# =============================================================================
# CANONICAL GEOMETRY AND MEMORY-SAFE RESAMPLING
# =============================================================================


def canonical_image_geometry(image):
    source_orientation = nib.orientations.io_orientation(
        image.affine
    )
    target_orientation = nib.orientations.axcodes2ornt(
        CANONICAL_AXCODES
    )
    transform = nib.orientations.ornt_transform(
        source_orientation,
        target_orientation,
    )
    canonical_affine = (
        image.affine
        @ nib.orientations.inv_ornt_aff(
            transform,
            image.shape[:3],
        )
    )
    raw_array = image.dataobj.get_unscaled()
    canonical_array = nib.orientations.apply_orientation(
        raw_array,
        transform,
    )
    orientation = "".join(
        nib.aff2axcodes(canonical_affine)
    )
    if orientation != "LPS":
        raise RuntimeError(
            f"Canonical orientation failed: {orientation}"
        )
    spacing = np.linalg.norm(
        canonical_affine[:3, :3],
        axis=0,
    )
    return {
        "array": canonical_array,
        "affine": canonical_affine,
        "shape": np.asarray(canonical_array.shape, dtype=int),
        "spacing": spacing,
        "raw_orientation": "".join(
            nib.aff2axcodes(image.affine)
        ),
        "transform": transform,
    }


def proxy_scaling(image):
    slope = getattr(image.dataobj, "slope", None)
    intercept = getattr(image.dataobj, "inter", None)
    slope = 1.0 if slope is None else float(slope)
    intercept = 0.0 if intercept is None else float(intercept)
    if not np.isfinite(slope) or slope == 0:
        slope = 1.0
    if not np.isfinite(intercept):
        intercept = 0.0
    return slope, intercept


def create_centered_crop_affine(
    canonical_affine,
    center_voxel,
):
    center_voxel = np.asarray(center_voxel, dtype=float)
    if center_voxel.shape != (3,):
        raise RuntimeError("Predicted centre must have three axes.")

    center_world = nib.affines.apply_affine(
        canonical_affine,
        center_voxel,
    )
    directions = canonical_affine[:3, :3].copy()
    norms = np.linalg.norm(directions, axis=0)
    if not np.all(np.isfinite(norms)) or np.any(norms <= 0):
        raise RuntimeError("Invalid canonical affine directions.")
    directions /= norms

    crop_affine = np.eye(4, dtype=float)
    crop_affine[:3, :3] = directions * TARGET_SPACING_MM
    crop_center = (TARGET_SHAPE.astype(float) - 1.0) / 2.0
    crop_affine[:3, 3] = (
        center_world
        - crop_affine[:3, :3] @ crop_center
    )
    return crop_affine, center_world


def resample_crop(
    ct_geometry,
    mask_geometry,
    crop_affine,
    ct_slope,
    ct_intercept,
    reheader_mask_to_ct,
):
    ct_inverse = np.linalg.inv(ct_geometry["affine"])
    mask_affine = (
        ct_geometry["affine"]
        if reheader_mask_to_ct
        else mask_geometry["affine"]
    )
    mask_inverse = np.linalg.inv(mask_affine)

    output_ct = np.empty(tuple(TARGET_SHAPE), dtype=np.int16)
    output_mask = np.empty(tuple(TARGET_SHAPE), dtype=np.uint8)

    x_indices = np.arange(TARGET_SHAPE[0], dtype=float)
    y_indices = np.arange(TARGET_SHAPE[1], dtype=float)

    for z_start in range(0, TARGET_SHAPE[2], RESAMPLE_Z_CHUNK):
        z_end = min(
            z_start + RESAMPLE_Z_CHUNK,
            TARGET_SHAPE[2],
        )
        z_indices = np.arange(z_start, z_end, dtype=float)
        grid = np.meshgrid(
            x_indices,
            y_indices,
            z_indices,
            indexing="ij",
        )
        output_voxels = np.stack(
            [axis.reshape(-1) for axis in grid],
            axis=0,
        )
        homogeneous = np.vstack(
            [
                output_voxels,
                np.ones(
                    (1, output_voxels.shape[1]),
                    dtype=float,
                ),
            ]
        )
        world = crop_affine @ homogeneous
        ct_coordinates = (ct_inverse @ world)[:3]
        mask_coordinates = (mask_inverse @ world)[:3]

        ct_values = map_coordinates(
            ct_geometry["array"],
            ct_coordinates,
            order=1,
            mode="constant",
            cval=(HU_MIN - ct_intercept) / ct_slope,
            prefilter=False,
        )
        ct_values = ct_values * ct_slope + ct_intercept
        ct_values = np.clip(
            np.rint(ct_values),
            HU_MIN,
            HU_MAX,
        ).astype(np.int16)

        mask_values = map_coordinates(
            mask_geometry["array"],
            mask_coordinates,
            order=0,
            mode="constant",
            cval=0,
            prefilter=False,
        )
        if not np.all(np.isfinite(mask_values)):
            raise RuntimeError(
                "Resampled mask contains non-finite values."
            )
        if not np.allclose(
            mask_values,
            np.rint(mask_values),
            atol=1e-6,
        ):
            raise RuntimeError(
                "Resampled mask is not integer-valued."
            )
        mask_values = np.rint(mask_values).astype(np.int16)
        unique = set(np.unique(mask_values).tolist())
        if not unique.issubset(ALLOWED_LABELS):
            raise RuntimeError(
                f"Unexpected resampled labels: {sorted(unique)}"
            )

        chunk_shape = (
            TARGET_SHAPE[0],
            TARGET_SHAPE[1],
            z_end - z_start,
        )
        output_ct[:, :, z_start:z_end] = ct_values.reshape(
            chunk_shape
        )
        output_mask[:, :, z_start:z_end] = mask_values.reshape(
            chunk_shape
        ).astype(np.uint8)

    return output_ct, output_mask


def scan_full_mask(
    mask_geometry,
    spatial_affine,
    crop_affine,
):
    label_counts = np.zeros(7, dtype=np.int64)
    contained_counts = np.zeros(7, dtype=np.int64)
    z_size = int(mask_geometry["shape"][2])
    source_to_crop = np.linalg.inv(crop_affine) @ spatial_affine
    lower = np.full(3, -0.5, dtype=float)
    upper = TARGET_SHAPE.astype(float) - 0.5

    for z_start in range(0, z_size, MASK_SCAN_Z_CHUNK):
        z_end = min(z_start + MASK_SCAN_Z_CHUNK, z_size)
        values = np.asarray(
            mask_geometry["array"][:, :, z_start:z_end]
        )
        if not np.all(np.isfinite(values)):
            raise RuntimeError(
                "Full source mask contains non-finite values."
            )
        rounded = np.rint(values)
        if not np.allclose(values, rounded, atol=1e-6):
            raise RuntimeError(
                "Full source mask is not integer-valued."
            )
        rounded = rounded.astype(np.int16, copy=False)
        unique = set(np.unique(rounded).tolist())
        if not unique.issubset(ALLOWED_LABELS):
            raise RuntimeError(
                f"Unexpected full-mask labels: {sorted(unique)}"
            )
        label_counts += np.bincount(
            rounded.reshape(-1),
            minlength=7,
        )[:7]

        foreground = (rounded == 1) | (rounded == 4)
        positions = np.argwhere(foreground)
        if positions.size:
            positions[:, 2] += z_start
            homogeneous = np.column_stack(
                [
                    positions.astype(float),
                    np.ones(len(positions), dtype=float),
                ]
            )
            crop_coordinates = (
                source_to_crop @ homogeneous.T
            )[:3].T
            inside = np.all(
                (crop_coordinates >= lower)
                & (crop_coordinates < upper),
                axis=1,
            )
            inside_labels = rounded[
                foreground
            ][inside].astype(np.int16, copy=False)
            contained_counts += np.bincount(
                inside_labels,
                minlength=7,
            )[:7]

    return label_counts, contained_counts


def save_case_npz(
    destination,
    ct_hu,
    mask_labels,
    crop_affine,
    metadata,
):
    temporary = Path(str(destination) + ".part")
    if temporary.exists():
        temporary.unlink()

    with open(temporary, "wb") as file:
        np.savez_compressed(
            file,
            ct_hu=ct_hu,
            mask_labels=mask_labels,
            crop_affine=np.asarray(crop_affine, dtype=np.float64),
            crop_spacing_mm=TARGET_SPACING_MM.astype(np.float32),
            predicted_center_canonical=np.asarray(
                metadata["predicted_center_canonical"],
                dtype=np.float32,
            ),
            predicted_center_world_mm=np.asarray(
                metadata["predicted_center_world_mm"],
                dtype=np.float64,
            ),
            study_id=np.asarray(str(metadata["study_id"])),
            patient_id=np.asarray(str(metadata["patient_id"])),
            partition=np.asarray(str(metadata["partition"])),
            diagnostic_label=np.asarray(
                str(metadata["diagnostic_label"])
            ),
            annotation_type=np.asarray(
                str(metadata["annotation_type"])
            ),
            geometry_status=np.asarray(
                str(metadata["geometry_status"])
            ),
        )
        file.flush()
        os.fsync(file.fileno())

    with np.load(temporary, allow_pickle=False) as data:
        if set(data.files) != {
            "ct_hu",
            "mask_labels",
            "crop_affine",
            "crop_spacing_mm",
            "predicted_center_canonical",
            "predicted_center_world_mm",
            "study_id",
            "patient_id",
            "partition",
            "diagnostic_label",
            "annotation_type",
            "geometry_status",
        }:
            raise RuntimeError("Saved NPZ key set is not exact.")
        if tuple(data["ct_hu"].shape) != tuple(TARGET_SHAPE):
            raise RuntimeError("Saved CT shape is incorrect.")
        if tuple(data["mask_labels"].shape) != tuple(TARGET_SHAPE):
            raise RuntimeError("Saved mask shape is incorrect.")
        if data["ct_hu"].dtype != np.int16:
            raise RuntimeError("Saved CT dtype is not int16.")
        if data["mask_labels"].dtype != np.uint8:
            raise RuntimeError("Saved mask dtype is not uint8.")

    os.replace(temporary, destination)
    return destination.stat().st_size, sha256_file(destination)


# =============================================================================
# INPUT LOCKS
# =============================================================================


print("=" * 120)
print("STAGE 3A — RESUMABLE E5 HIGH-RESOLUTION CT–MASK CROP PREPROCESSING")
print("=" * 120)

required_paths = [
    DEVELOPMENT_MANIFEST_PATH,
    STAGE1A_LEDGER_PATH,
    PREDICTIONS_PATH,
    STAGE2E_PROTOCOL_PATH,
    FINAL_CROP_PROTOCOL_PATH,
    REMOTE_MEMBER_INVENTORY_PATH,
    REMOTE_SOURCE_AUDIT_PATH,
]
for required_path in required_paths:
    if not required_path.exists():
        raise FileNotFoundError(f"Required input missing:\n{required_path}")

with open(STAGE2E_PROTOCOL_PATH, "r", encoding="utf-8") as file:
    stage2e_protocol = json.load(file)
with open(FINAL_CROP_PROTOCOL_PATH, "r", encoding="utf-8") as file:
    crop_protocol = json.load(file)

if stage2e_protocol.get("centers_frozen") is not True:
    raise RuntimeError("Stage 2E centres are not frozen.")
if crop_protocol.get("selected_candidate") != "E5_240x192x128":
    raise RuntimeError("The locked E5 crop protocol was not found.")

manifest = pd.read_csv(DEVELOPMENT_MANIFEST_PATH)
stage1a_ledger = pd.read_csv(STAGE1A_LEDGER_PATH)
predictions = pd.read_csv(PREDICTIONS_PATH)
members = pd.read_csv(REMOTE_MEMBER_INVENTORY_PATH)
sources = pd.read_csv(REMOTE_SOURCE_AUDIT_PATH)

for frame in [manifest, stage1a_ledger, predictions, members]:
    frame["study_id"] = normalize_study_id(frame["study_id"])

if "geometry_status" not in manifest.columns:
    if "geometry_status" not in stage1a_ledger.columns:
        raise RuntimeError(
            "Geometry status is absent from both the Stage 1B "
            "manifest and Stage 1A ledger."
        )
    geometry_rows = stage1a_ledger[
        ["study_id", "geometry_status"]
    ].drop_duplicates(subset=["study_id"], keep="last")
    manifest = manifest.merge(
        geometry_rows,
        on="study_id",
        how="left",
        validate="one_to_one",
    )

manifest_required = {
    "study_id",
    "patient_id",
    "partition",
    "diagnostic_label",
    "annotation_type",
    "geometry_status",
    "canonical_shape_json",
}
prediction_required = {
    "study_id",
    "predicted_center_canonical_x",
    "predicted_center_canonical_y",
    "predicted_center_canonical_z",
}
member_required = {
    "study_id",
    "batch_number",
    "member_name",
    "compressed_size_bytes",
    "uncompressed_size_bytes",
    "compression_type",
    "crc32_hex",
    "local_header_offset",
}
source_required = {
    "batch_number",
    "record_id",
    "archive_name",
}

for name, frame, required in [
    ("development manifest", manifest, manifest_required),
    ("Stage 2E predictions", predictions, prediction_required),
    ("remote member inventory", members, member_required),
    ("remote source audit", sources, source_required),
]:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"{name} missing columns: {missing}")

if len(manifest) != EXPECTED_CASES:
    raise RuntimeError("Development manifest count changed.")
if manifest["study_id"].nunique() != EXPECTED_CASES:
    raise RuntimeError("Development manifest IDs are not unique.")
if len(predictions) != EXPECTED_CASES:
    raise RuntimeError("Stage 2E prediction count changed.")
if predictions["study_id"].nunique() != EXPECTED_CASES:
    raise RuntimeError("Stage 2E prediction IDs are not unique.")
if set(manifest["study_id"]) != set(predictions["study_id"]):
    raise RuntimeError("Prediction and development IDs differ.")
if set(manifest["partition"]) != {"train", "validation"}:
    raise RuntimeError("A locked test partition entered Stage 3A.")
if manifest["partition"].value_counts().to_dict() != {
    "train": EXPECTED_TRAIN,
    "validation": EXPECTED_VALIDATION,
}:
    raise RuntimeError("Development partition counts changed.")
if not set(manifest["geometry_status"]).issubset(
    APPROVED_GEOMETRY_STATUSES
):
    unexpected = sorted(
        set(manifest["geometry_status"])
        - APPROVED_GEOMETRY_STATUSES
    )
    raise RuntimeError(
        f"Unapproved geometry statuses: {unexpected}"
    )

members = members.loc[
    members["study_id"].isin(manifest["study_id"])
].copy()
if len(members) != EXPECTED_CASES:
    raise RuntimeError(
        "Remote inventory does not contain exactly one member per "
        "development study."
    )
if members["study_id"].nunique() != EXPECTED_CASES:
    raise RuntimeError("Remote development members are not unique.")

sources["batch_number"] = sources["batch_number"].astype(int)
if sources["batch_number"].nunique() != 4:
    raise RuntimeError("All four remote archive sources are required.")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
QC_DIR.mkdir(parents=True, exist_ok=True)
META_DIR.mkdir(parents=True, exist_ok=True)

if LEDGER_PATH.exists():
    ledger = pd.read_csv(LEDGER_PATH)
    if len(ledger):
        ledger["study_id"] = normalize_study_id(
            ledger["study_id"]
        )
        ledger = ledger.drop_duplicates(
            subset=["study_id"],
            keep="last",
        )
else:
    ledger = pd.DataFrame()

completed_ids = set()
if len(ledger) and "processing_complete" in ledger.columns:
    completion_flags = (
        ledger["processing_complete"]
        .fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes"})
    )
    completed_rows = ledger.loc[
        completion_flags
    ]
    for _, completed_row in completed_rows.iterrows():
        output_path = Path(str(completed_row["output_path"]))
        if validate_output_file(
            output_path,
            completed_row["output_size_bytes"],
            completed_row["output_sha256"],
        ):
            completed_ids.add(str(completed_row["study_id"]))

pending = manifest.loc[
    ~manifest["study_id"].isin(completed_ids)
].copy()
pending = pending.sort_values(
    ["partition", "study_id"]
).reset_index(drop=True)

if MAX_CASES_PER_RUN is None:
    selected = pending
else:
    selected = pending.head(int(MAX_CASES_PER_RUN))

selected_member_rows = members.set_index("study_id").loc[
    selected["study_id"]
] if len(selected) else pd.DataFrame()
expected_transfer = (
    int(selected_member_rows["compressed_size_bytes"].sum())
    if len(selected)
    else 0
)
uncompressed_case_bytes = int(
    np.prod(TARGET_SHAPE)
    * (
        np.dtype(np.int16).itemsize
        + np.dtype(np.uint8).itemsize
    )
)
worst_case_dataset_gib = (
    uncompressed_case_bytes * EXPECTED_CASES / (1024 ** 3)
)

print()
print(f"Development cases: {EXPECTED_CASES}")
print(f"Previously completed and reverified: {len(completed_ids)}")
print(f"Pending before this run: {len(pending)}")
print(f"Selected for this run: {len(selected)}")
print(
    f"Expected selected CT transfer: "
    f"{expected_transfer / (1024 ** 3):.2f} GiB"
)
print(
    f"Worst-case uncompressed E5 dataset: "
    f"{worst_case_dataset_gib:.2f} GiB"
)
print(f"Drive free space: {disk_free_gib(PROJECT_ROOT):.2f} GiB")
print(f"Local free space: {disk_free_gib(RUNTIME_ROOT):.2f} GiB")
print("Locked test cases accessed: 0")
print("Checkpoint frequency: after every case")

if disk_free_gib(PROJECT_ROOT) < worst_case_dataset_gib * 1.10:
    raise RuntimeError(
        "Drive space is insufficient for the conservative "
        "uncompressed E5 storage bound plus 10% overhead."
    )

protocol = {
    "stage": "3A",
    "created_at_utc": utc_now(),
    "development_cases": EXPECTED_CASES,
    "train_cases": EXPECTED_TRAIN,
    "validation_cases": EXPECTED_VALIDATION,
    "crop_candidate": "E5_240x192x128",
    "crop_shape": TARGET_SHAPE.tolist(),
    "crop_spacing_mm": TARGET_SPACING_MM.tolist(),
    "crop_fov_mm": EXPECTED_FOV_MM.tolist(),
    "centering_source": (
        "STAGE2E_FROZEN_LEARNED_LOCALIZER_PREDICTIONS"
    ),
    "canonical_orientation": "LPS",
    "ct_interpolation": "TRILINEAR",
    "mask_interpolation": "NEAREST_NEIGHBOR",
    "ct_storage": "INT16_HU_CLIPPED_MINUS200_TO_300",
    "mask_storage": "UINT8_LABELS_0_TO_6",
    "case_specific_normalization": "PROHIBITED",
    "ground_truth_center_adjustment": "PROHIBITED",
    "locked_test_cases_accessed": 0,
    "raw_files_modified": False,
    "output_directory": str(PROCESSED_DIR),
    "ledger_path": str(LEDGER_PATH),
}
atomic_write_json(protocol, DATASET_PROTOCOL_PATH)


# =============================================================================
# RESUMABLE CASE-BY-CASE PROCESSING
# =============================================================================


random.seed(SEED)
np.random.seed(SEED)
session = requests.Session()
session.headers.update(
    {
        "User-Agent": (
            "PDAC-Public-Q1-Stage3A/1.0 "
            "(research reproducibility audit)"
        )
    }
)

completed_this_run = 0
transferred_this_run = 0
errors = []

for run_order, (_, manifest_row) in enumerate(
    selected.iterrows(),
    start=1,
):
    study_id = str(manifest_row["study_id"])
    print()
    print(
        f"[{run_order}/{len(selected)}] Processing {study_id} "
        f"— durable {len(completed_ids)}/{EXPECTED_CASES}"
    )

    output_path = PROCESSED_DIR / f"{study_id}.npz"
    started_at = time.time()

    try:
        prediction_row = exact_one_row(
            predictions,
            study_id,
            "Stage 2E prediction",
        )
        member_row = exact_one_row(
            members,
            study_id,
            "remote member",
        )
        source_rows = sources.loc[
            sources["batch_number"].astype(int)
            == int(member_row["batch_number"])
        ]
        if len(source_rows) != 1:
            raise RuntimeError(
                f"{study_id}: archive source row is not unique."
            )
        source_row = source_rows.iloc[0]

        annotation_type = str(
            manifest_row["annotation_type"]
        ).strip().lower()
        if annotation_type == "manual":
            mask_path = (
                RAW_ROOT
                / "Manual_Labels"
                / f"{study_id}.nii.gz"
            )
        elif annotation_type == "automatic":
            mask_path = (
                RAW_ROOT
                / "Automatic_Labels"
                / f"{study_id}.nii.gz"
            )
        else:
            raise RuntimeError(
                f"Unknown annotation type: {annotation_type}"
            )
        if not mask_path.exists():
            raise FileNotFoundError(
                f"Persistent mask missing: {mask_path}"
            )

        with TemporaryDirectory(
            prefix=f"stage3a_{study_id}_",
            dir=RUNTIME_ROOT,
        ) as temporary_directory:
            work_dir = Path(temporary_directory)
            acquisition = recover_zip_member(
                session,
                member_row,
                source_row,
                work_dir,
            )
            transferred_this_run += int(
                acquisition["compressed_transfer_bytes"]
            )

            ct_nii_path = work_dir / "ct.nii"
            mask_nii_path = work_dir / "mask.nii"
            decompress_gzip_to_nii(
                acquisition["inner_gzip_path"],
                ct_nii_path,
            )
            decompress_gzip_to_nii(mask_path, mask_nii_path)

            ct_image = nib.load(str(ct_nii_path), mmap="r")
            mask_image = nib.load(str(mask_nii_path), mmap="r")
            if len(ct_image.shape) != 3 or len(mask_image.shape) != 3:
                raise RuntimeError("CT and mask must be 3D.")

            ct_geometry = canonical_image_geometry(ct_image)
            mask_geometry = canonical_image_geometry(mask_image)

            expected_canonical_shape = np.asarray(
                json.loads(
                    str(manifest_row["canonical_shape_json"])
                ),
                dtype=int,
            )
            if not np.array_equal(
                ct_geometry["shape"],
                expected_canonical_shape,
            ):
                raise RuntimeError(
                    "CT canonical shape differs from Stage 1B: "
                    f"{ct_geometry['shape'].tolist()} vs "
                    f"{expected_canonical_shape.tolist()}"
                )
            if not np.array_equal(
                ct_geometry["shape"],
                mask_geometry["shape"],
            ):
                raise RuntimeError(
                    "Canonical CT and mask shapes differ."
                )

            geometry_status = str(
                manifest_row["geometry_status"]
            )
            reheader_mask_to_ct = (
                geometry_status == KNOWN_REHEADER_STATUS
            )

            if not reheader_mask_to_ct:
                if not np.allclose(
                    ct_geometry["affine"],
                    mask_geometry["affine"],
                    rtol=1e-4,
                    atol=1e-3,
                ):
                    raise RuntimeError(
                        "Unapproved CT–mask affine mismatch."
                    )

            predicted_center = np.asarray(
                [
                    prediction_row[
                        "predicted_center_canonical_x"
                    ],
                    prediction_row[
                        "predicted_center_canonical_y"
                    ],
                    prediction_row[
                        "predicted_center_canonical_z"
                    ],
                ],
                dtype=float,
            )
            if (
                predicted_center.shape != (3,)
                or not np.all(np.isfinite(predicted_center))
            ):
                raise RuntimeError(
                    "Invalid frozen predicted centre."
                )

            crop_affine, center_world = (
                create_centered_crop_affine(
                    ct_geometry["affine"],
                    predicted_center,
                )
            )
            ct_slope, ct_intercept = proxy_scaling(ct_image)
            mask_slope, mask_intercept = proxy_scaling(mask_image)
            if (
                not np.isclose(mask_slope, 1.0, atol=1e-8)
                or not np.isclose(mask_intercept, 0.0, atol=1e-8)
            ):
                raise RuntimeError(
                    "Mask scaling is not identity; raw labels cannot "
                    "be interpreted safely."
                )

            spatial_mask_affine = (
                ct_geometry["affine"]
                if reheader_mask_to_ct
                else mask_geometry["affine"]
            )
            (
                full_label_counts,
                contained_label_counts,
            ) = scan_full_mask(
                mask_geometry,
                spatial_mask_affine,
                crop_affine,
            )
            ct_crop, mask_crop = resample_crop(
                ct_geometry,
                mask_geometry,
                crop_affine,
                ct_slope,
                ct_intercept,
                reheader_mask_to_ct,
            )

            crop_label_counts = np.bincount(
                mask_crop.reshape(-1),
                minlength=7,
            )[:7].astype(np.int64)

            lesion_total = int(full_label_counts[1])
            pancreas_total = int(full_label_counts[4])
            union_total = lesion_total + pancreas_total
            lesion_contained = int(contained_label_counts[1])
            pancreas_contained = int(contained_label_counts[4])
            union_contained = lesion_contained + pancreas_contained

            lesion_coverage = (
                lesion_contained / lesion_total
                if lesion_total > 0
                else 1.0
            )
            pancreas_coverage = (
                pancreas_contained / pancreas_total
                if pancreas_total > 0
                else 0.0
            )
            union_coverage = (
                union_contained / union_total
                if union_total > 0
                else 0.0
            )

            diagnostic_label = str(
                manifest_row["diagnostic_label"]
            ).strip()
            is_pdac = diagnostic_label.lower() == "pdac"
            if is_pdac and lesion_total <= 0:
                raise RuntimeError(
                    "PDAC source mask lacks lesion label 1."
                )
            if not is_pdac and lesion_total != 0:
                raise RuntimeError(
                    "Non-PDAC source mask contains lesion label 1."
                )
            if pancreas_total <= 0:
                raise RuntimeError(
                    "Source mask lacks pancreas label 4."
                )
            if int(crop_label_counts[4]) <= 0:
                raise RuntimeError(
                    "Predicted E5 crop contains no pancreas voxels."
                )
            if (
                ct_crop.shape != tuple(TARGET_SHAPE)
                or mask_crop.shape != tuple(TARGET_SHAPE)
            ):
                raise RuntimeError("E5 output shape failed.")
            if ct_crop.dtype != np.int16:
                raise RuntimeError("E5 CT dtype failed.")
            if mask_crop.dtype != np.uint8:
                raise RuntimeError("E5 mask dtype failed.")
            if (
                int(ct_crop.min()) < int(HU_MIN)
                or int(ct_crop.max()) > int(HU_MAX)
            ):
                raise RuntimeError("E5 CT HU bounds failed.")

            output_size, output_hash = save_case_npz(
                output_path,
                ct_crop,
                mask_crop,
                crop_affine,
                {
                    "study_id": study_id,
                    "patient_id": manifest_row["patient_id"],
                    "partition": manifest_row["partition"],
                    "diagnostic_label": diagnostic_label,
                    "annotation_type": annotation_type,
                    "geometry_status": geometry_status,
                    "predicted_center_canonical":
                        predicted_center,
                    "predicted_center_world_mm": center_world,
                },
            )

            row = {
                "study_id": study_id,
                "patient_id": manifest_row["patient_id"],
                "partition": manifest_row["partition"],
                "diagnostic_label": diagnostic_label,
                "annotation_type": annotation_type,
                "batch_number": int(
                    member_row["batch_number"]
                ),
                "member_name": member_row["member_name"],
                "compressed_transfer_bytes": int(
                    acquisition["compressed_transfer_bytes"]
                ),
                "recovered_size_bytes": int(
                    acquisition["recovered_size_bytes"]
                ),
                "crc32_hex": acquisition["crc32_hex"],
                "local_header_filename_matches":
                    acquisition[
                        "local_header_filename_matches"
                    ],
                "header_endpoint":
                    acquisition["header_endpoint"],
                "download_endpoint":
                    acquisition["download_endpoint"],
                "raw_ct_orientation":
                    ct_geometry["raw_orientation"],
                "raw_mask_orientation":
                    mask_geometry["raw_orientation"],
                "canonical_orientation": "LPS",
                "canonical_shape_json": json.dumps(
                    ct_geometry["shape"].tolist()
                ),
                "canonical_spacing_mm_json": json_vector(
                    ct_geometry["spacing"]
                ),
                "geometry_status": geometry_status,
                "derived_mask_reheader_applied":
                    reheader_mask_to_ct,
                "predicted_center_canonical_json":
                    json_vector(predicted_center),
                "predicted_center_world_mm_json":
                    json_vector(center_world),
                "crop_shape_json": json.dumps(
                    TARGET_SHAPE.tolist()
                ),
                "crop_spacing_mm_json": json_vector(
                    TARGET_SPACING_MM
                ),
                "crop_affine_json": json_matrix(crop_affine),
                "ct_dtype": str(ct_crop.dtype),
                "mask_dtype": str(mask_crop.dtype),
                "ct_minimum_hu": int(ct_crop.min()),
                "ct_maximum_hu": int(ct_crop.max()),
                "full_lesion_voxels": lesion_total,
                "contained_source_lesion_voxels":
                    lesion_contained,
                "resampled_crop_lesion_voxels": int(
                    crop_label_counts[1]
                ),
                "lesion_voxel_coverage": lesion_coverage,
                "full_pancreas_voxels": pancreas_total,
                "contained_source_pancreas_voxels":
                    pancreas_contained,
                "resampled_crop_pancreas_voxels": int(
                    crop_label_counts[4]
                ),
                "pancreas_voxel_coverage": pancreas_coverage,
                "full_union_voxels": union_total,
                "contained_source_union_voxels":
                    union_contained,
                "resampled_crop_union_voxels": int(
                    crop_label_counts[1]
                    + crop_label_counts[4]
                ),
                "union_voxel_coverage": union_coverage,
                "output_path": str(output_path),
                "output_size_bytes": output_size,
                "output_sha256": output_hash,
                "processing_seconds": float(
                    time.time() - started_at
                ),
                "processing_complete": True,
                "error_type": "",
                "error_message": "",
                "completed_at_utc": utc_now(),
            }

        ledger = replace_ledger_row(ledger, row)
        ledger = ledger.sort_values(
            ["partition", "study_id"],
            na_position="last",
        ).reset_index(drop=True)
        atomic_write_csv(ledger, LEDGER_PATH)
        completed_ids.add(study_id)
        completed_this_run += 1

        print(
            "    PASS — "
            f"pancreas={pancreas_coverage:.6f}, "
            f"lesion={lesion_coverage:.6f}, "
            f"union={union_coverage:.6f} — "
            f"durable {len(completed_ids)}/{EXPECTED_CASES}"
        )

    except Exception as error:
        error_row = {
            "study_id": study_id,
            "patient_id": manifest_row.get("patient_id", ""),
            "partition": manifest_row.get("partition", ""),
            "diagnostic_label": manifest_row.get(
                "diagnostic_label",
                "",
            ),
            "annotation_type": manifest_row.get(
                "annotation_type",
                "",
            ),
            "output_path": str(output_path),
            "processing_seconds": float(time.time() - started_at),
            "processing_complete": False,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "completed_at_utc": utc_now(),
        }
        ledger = replace_ledger_row(ledger, error_row)
        atomic_write_csv(ledger, LEDGER_PATH)
        errors.append(error_row)
        print()
        print(
            f"FAIL — {study_id}: {type(error).__name__}: {error}"
        )
        print(
            "Processing stopped; completed cases remain durably "
            "saved."
        )
        break


# =============================================================================
# RUN SUMMARY
# =============================================================================


remaining = EXPECTED_CASES - len(completed_ids)
persistent_size = sum(
    path.stat().st_size
    for path in PROCESSED_DIR.glob("*.npz")
)

run_audit = {
    "stage": "3A",
    "created_at_utc": utc_now(),
    "development_cases": EXPECTED_CASES,
    "previously_completed": (
        len(completed_ids) - completed_this_run
    ),
    "completed_this_run": completed_this_run,
    "durably_completed": len(completed_ids),
    "remaining": remaining,
    "errors_this_run": len(errors),
    "transferred_this_run_bytes": transferred_this_run,
    "persistent_output_bytes": persistent_size,
    "locked_test_cases_accessed": 0,
    "raw_files_modified": False,
    "ledger_path": str(LEDGER_PATH),
    "output_directory": str(PROCESSED_DIR),
    "status": (
        "COMPLETE"
        if remaining == 0 and not errors
        else (
            "STOPPED_WITH_ERROR"
            if errors
            else "MORE_CASES_REQUIRED"
        )
    ),
}
atomic_write_json(run_audit, RUN_AUDIT_PATH)

print()
print("-" * 120)
print("STAGE 3A RUN RESULT")
print("-" * 120)
print(f"Completed this run: {completed_this_run}")
print(f"Errors this run: {len(errors)}")
print(
    f"Durably completed: "
    f"{len(completed_ids)}/{EXPECTED_CASES}"
)
print(f"Remaining: {remaining}")
print(
    f"Transferred this run: "
    f"{transferred_this_run / (1024 ** 3):.3f} GiB"
)
print(
    f"Persistent processed data: "
    f"{persistent_size / (1024 ** 3):.3f} GiB"
)
print("Locked test cases accessed: 0")
print("Checkpoint:")
print(LEDGER_PATH)
print()
print("=" * 120)
if errors:
    print("STAGE 3A STATUS: STOPPED_WITH_ERROR")
elif remaining:
    print("STAGE 3A STATUS: MORE_CASES_REQUIRED")
else:
    print("STAGE 3A STATUS: COMPLETE")
print("=" * 120)

if errors:
    raise RuntimeError(
        "Stage 3A stopped on an error. Completed cases remain "
        "durably saved."
    )
