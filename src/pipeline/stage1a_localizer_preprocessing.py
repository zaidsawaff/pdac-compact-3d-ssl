from pathlib import Path
from datetime import datetime, timezone
from itertools import product
import gc
import gzip
import hashlib
import json
import os
import shutil
import struct
import tempfile
import time
import zlib

import nibabel as nib
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from scipy import ndimage


PROJECT_ROOT = Path(os.environ.get("PDAC_PROJECT_ROOT", "/content/drive/MyDrive/PDAC_Public_Q1_Project"))
RUNTIME_ROOT = Path(os.environ.get("PDAC_RUNTIME_ROOT", "/content"))
RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
META_DIR = PROJECT_ROOT / "01_Metadata"
QC_DIR = PROJECT_ROOT / "02_Quality_Control"
OUTPUT_DIR = (
    PROJECT_ROOT
    / "03_Processed"
    / "Localizer"
    / "F1_96x96x160"
    / "Cases"
)

CASE_GEOMETRY_PATH = (
    QC_DIR / "stage0m_b_r_localizer_case_geometry.csv"
)
FULL_LEDGER_PATH = (
    QC_DIR / "stage0l_c_full_development_roi_geometry_ledger.csv"
)
REMOTE_INVENTORY_PATH = (
    META_DIR / "stage0i_panorama_remote_zip_member_inventory.csv"
)
SOURCE_AUDIT_PATH = (
    META_DIR / "stage0i_panorama_remote_archive_source_audit.csv"
)
PROTOCOL_PATH = (
    META_DIR / "stage0m_b_r_localizer_input_protocol.json"
)

OUTPUT_LEDGER_PATH = (
    QC_DIR / "stage1a_localizer_preprocessing_ledger.csv"
)
ERROR_PATH = (
    QC_DIR / "stage1a_localizer_preprocessing_errors.csv"
)
RUN_AUDIT_PATH = (
    QC_DIR / "stage1a_localizer_preprocessing_run_audit.json"
)

CANVAS_SHAPE = np.asarray([96, 96, 160], dtype=int)
PADDING_VOXELS_PER_SIDE = 4
HU_LOWER = -200.0
HU_UPPER = 300.0

# Process every remaining case in the current Colab session.
# A checkpoint is committed after each case.
MAX_CASES_THIS_RUN = int(os.environ.get("PDAC_MAX_CASES", "1671"))

REMOTE_CHUNK_BYTES = 16 * 1024 * 1024
MAX_RETRIES = 6


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


def normalize_study_id(value):
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def parse_list(value):
    array = np.asarray(json.loads(value), dtype=float)
    if array.shape != (3,):
        raise RuntimeError(f"Expected three values, observed {array}.")
    return array


def parse_bbox(value):
    parsed = json.loads(value)
    required = {
        "minimum",
        "maximum",
        "voxel_count",
        "size_voxels",
        "center_voxel",
    }
    if not required.issubset(parsed):
        raise RuntimeError("Incomplete pancreas bounding-box record.")
    return parsed


def as_bool_series(series):
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes", "pass", "complete", "completed"])
    )


def safe_range_get(session, url, start, end):
    expected_length = end - start + 1
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        response = None
        try:
            response = session.get(
                url,
                headers={
                    "Range": f"bytes={start}-{end}",
                    "Accept-Encoding": "identity",
                },
                stream=True,
                timeout=(30, 180),
                allow_redirects=True,
            )

            if response.status_code == 200:
                response.close()
                raise RuntimeError(
                    "HTTP 200 full-archive response rejected unread."
                )

            if response.status_code != 206:
                status = response.status_code
                response.close()
                raise RuntimeError(
                    f"Expected HTTP 206, received HTTP {status}."
                )

            content_range = response.headers.get("Content-Range", "")
            expected_prefix = f"bytes {start}-{end}/"
            if not content_range.startswith(expected_prefix):
                response.close()
                raise RuntimeError(
                    f"Unexpected Content-Range: {content_range!r}"
                )

            payload = response.raw.read(expected_length + 1)
            response.close()

            if len(payload) != expected_length:
                raise RuntimeError(
                    f"Range length mismatch: "
                    f"{len(payload)}/{expected_length}"
                )

            return payload

        except Exception as error:
            last_error = error
            if response is not None:
                try:
                    response.close()
                except Exception:
                    pass

            if attempt < MAX_RETRIES:
                wait_seconds = min(2 ** (attempt - 1), 16)
                print(
                    f"      Retry {attempt}/{MAX_RETRIES}: "
                    f"{type(error).__name__}: {error}"
                )
                time.sleep(wait_seconds)

    raise RuntimeError(
        f"Range request failed after {MAX_RETRIES} attempts: "
        f"{last_error}"
    )


def resolve_member_data_start(
    session,
    endpoint,
    local_header_offset,
    expected_member_name,
    expected_compression_type,
):
    fixed = safe_range_get(
        session,
        endpoint,
        local_header_offset,
        local_header_offset + 29,
    )

    fields = struct.unpack("<IHHHHHIIIHH", fixed)
    signature = fields[0]
    flag_bits = fields[2]
    compression_type = fields[3]
    filename_length = fields[9]
    extra_length = fields[10]

    if signature != 0x04034B50:
        raise RuntimeError("Invalid ZIP local-header signature.")
    if flag_bits & 0x0001:
        raise RuntimeError("Encrypted ZIP member is prohibited.")
    if compression_type != expected_compression_type:
        raise RuntimeError(
            f"Compression mismatch: "
            f"{compression_type}/{expected_compression_type}"
        )

    variable_length = filename_length + extra_length
    variable = safe_range_get(
        session,
        endpoint,
        local_header_offset + 30,
        local_header_offset + 30 + variable_length - 1,
    )

    filename_bytes = variable[:filename_length]
    encoding = "utf-8" if flag_bits & 0x0800 else "cp437"
    filename = filename_bytes.decode(encoding)

    if filename != expected_member_name:
        raise RuntimeError(
            f"Local filename mismatch: {filename!r} "
            f"!= {expected_member_name!r}"
        )

    return {
        "data_start": (
            local_header_offset + 30 + variable_length
        ),
        "header_transfer_bytes": 30 + variable_length,
    }


def recover_zip_member(
    session,
    endpoint,
    data_start,
    compressed_size,
    compression_type,
    output_path,
):
    if compression_type == 8:
        decompressor = zlib.decompressobj(-15)
    elif compression_type == 0:
        decompressor = None
    else:
        raise RuntimeError(
            f"Unsupported ZIP compression type: {compression_type}"
        )

    crc = 0
    recovered_size = 0
    downloaded_size = 0
    digest = hashlib.sha256()

    with open(output_path, "wb") as output:
        while downloaded_size < compressed_size:
            request_length = min(
                REMOTE_CHUNK_BYTES,
                compressed_size - downloaded_size,
            )
            start = data_start + downloaded_size
            end = start + request_length - 1

            payload = safe_range_get(
                session,
                endpoint,
                start,
                end,
            )
            downloaded_size += len(payload)

            if decompressor is None:
                recovered = payload
            else:
                recovered = decompressor.decompress(payload)

            if recovered:
                output.write(recovered)
                crc = zlib.crc32(recovered, crc)
                digest.update(recovered)
                recovered_size += len(recovered)

        if decompressor is not None:
            tail = decompressor.flush()
            if tail:
                output.write(tail)
                crc = zlib.crc32(tail, crc)
                digest.update(tail)
                recovered_size += len(tail)

    return {
        "compressed_bytes_downloaded": int(downloaded_size),
        "recovered_size_bytes": int(recovered_size),
        "crc32_hex": f"{crc & 0xffffffff:08x}",
        "recovered_sha256": digest.hexdigest(),
    }


def decompress_nifti_gzip(source_path, destination_path):
    with open(source_path, "rb") as file:
        magic = file.read(2)
    if magic != b"\x1f\x8b":
        raise RuntimeError("Recovered member is not gzip.")

    decompressed_size = 0
    with gzip.open(source_path, "rb") as source:
        with open(destination_path, "wb") as destination:
            while True:
                chunk = source.read(8 * 1024 * 1024)
                if not chunk:
                    break
                destination.write(chunk)
                decompressed_size += len(chunk)
    return int(decompressed_size)


def process_ct_to_canvas(
    ct_nii_path,
    mask_path,
    pancreas_bbox,
    annotation_type,
    study_id,
):
    ct_image = nib.load(str(ct_nii_path), mmap="r")
    mask_image = nib.load(str(mask_path))

    ct_shape = tuple(int(value) for value in ct_image.shape)
    mask_shape = tuple(int(value) for value in mask_image.shape)
    if len(ct_shape) != 3 or ct_shape != mask_shape:
        raise RuntimeError(
            f"CT-mask shape mismatch: {ct_shape}/{mask_shape}"
        )

    ct_affine = np.asarray(ct_image.affine, dtype=float)
    mask_affine = np.asarray(mask_image.affine, dtype=float)
    ct_orientation = "".join(nib.aff2axcodes(ct_affine))
    mask_orientation = "".join(nib.aff2axcodes(mask_affine))

    if ct_orientation != mask_orientation:
        raise RuntimeError(
            f"CT-mask orientation mismatch: "
            f"{ct_orientation}/{mask_orientation}"
        )

    ct_spacing = nib.affines.voxel_sizes(ct_affine)
    mask_spacing = nib.affines.voxel_sizes(mask_affine)
    strict_spacing_match = bool(
        np.allclose(
            ct_spacing,
            mask_spacing,
            rtol=1e-5,
            atol=1e-5,
        )
    )
    strict_affine_match = bool(
        np.allclose(
            ct_affine,
            mask_affine,
            rtol=1e-5,
            atol=1e-5,
        )
    )

    spacing_match = bool(
        np.allclose(
            ct_spacing,
            mask_spacing,
            rtol=1e-4,
            atol=1e-4,
        )
    )
    affine_match = bool(
        np.allclose(
            ct_affine,
            mask_affine,
            rtol=1e-5,
            atol=1e-4,
        )
    )

    if strict_spacing_match and strict_affine_match:
        geometry_status = "EXACT_PHYSICAL_GEOMETRY_MATCH"
    elif spacing_match and affine_match:
        geometry_status = (
            "PHYSICAL_GEOMETRY_MATCH_WITHIN_NUMERICAL_HEADER_TOLERANCE"
        )
    elif study_id in {"100936_00001", "100028_00001"}:
        geometry_status = (
            "KNOWN_VISUALLY_CONFIRMED_INDEX_ALIGNED_HEADER_MISMATCH"
        )
    elif annotation_type == "automatic":
        geometry_status = (
            "AUTOMATIC_INDEX_ALIGNED_HEADER_MISMATCH_USING_CT_GEOMETRY"
        )
    else:
        geometry_diagnostic = {
            "study_id": study_id,
            "annotation_type": annotation_type,
            "ct_shape": [int(v) for v in ct_shape],
            "mask_shape": [int(v) for v in mask_shape],
            "ct_orientation": ct_orientation,
            "mask_orientation": mask_orientation,
            "ct_spacing_mm": [float(v) for v in ct_spacing],
            "mask_spacing_mm": [float(v) for v in mask_spacing],
            "spacing_match": spacing_match,
            "affine_match": affine_match,
            "maximum_absolute_affine_difference": float(
                np.max(np.abs(ct_affine - mask_affine))
            ),
            "ct_affine": [
                [float(v) for v in row] for row in ct_affine
            ],
            "mask_affine": [
                [float(v) for v in row] for row in mask_affine
            ],
            "shape_and_orientation_index_aligned": True,
        }

        diagnostic_path = (
            QC_DIR
            / f"stage1a_{study_id}_manual_geometry_mismatch.json"
        )
        mask_review = np.asarray(
            mask_image.dataobj,
            dtype=np.uint8,
        )

        roi_counts = np.sum(
            np.isin(mask_review, [1, 4]),
            axis=(0, 1),
        )
        roi_slices = np.flatnonzero(roi_counts > 0)

        if len(roi_slices) == 0:
            raise RuntimeError(
                "Manual mask contains no pancreas or lesion voxels."
            )

        selected_indices = np.linspace(
            0,
            len(roi_slices) - 1,
            min(6, len(roi_slices)),
            dtype=int,
        )
        selected_slices = [
            int(roi_slices[index])
            for index in selected_indices
        ]

        overlay_path = (
            QC_DIR
            / f"stage1a_{study_id}_manual_index_alignment_overlay.png"
        )

        figure, axes = plt.subplots(
            2,
            3,
            figsize=(18, 12),
        )
        axes = np.asarray(axes).reshape(-1)

        for panel_index, axis in enumerate(axes):
            if panel_index >= len(selected_slices):
                axis.axis("off")
                continue

            z_index = selected_slices[panel_index]

            ct_slice = np.asarray(
                ct_image.dataobj[:, :, z_index],
                dtype=np.float32,
            )
            mask_slice = mask_review[:, :, z_index]

            axis.imshow(
                ct_slice,
                cmap="gray",
                vmin=-200,
                vmax=300,
                origin="lower",
            )

            if np.any(mask_slice == 4):
                axis.contour(
                    mask_slice == 4,
                    levels=[0.5],
                    colors=["cyan"],
                    linewidths=1.5,
                )

            if np.any(mask_slice == 1):
                axis.contour(
                    mask_slice == 1,
                    levels=[0.5],
                    colors=["red"],
                    linewidths=1.5,
                )

            axis.set_title(
                f"z={z_index} | "
                f"pancreas={int(np.sum(mask_slice == 4)):,} | "
                f"lesion={int(np.sum(mask_slice == 1)):,}"
            )
            axis.axis("off")

        figure.suptitle(
            f"Study {study_id} — voxel-index overlay\n"
            "Cyan: pancreas label 4 | Red: lesion label 1",
            fontsize=16,
        )
        figure.tight_layout()
        figure.savefig(
            overlay_path,
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(figure)

        geometry_diagnostic["selected_overlay_slices"] = (
            selected_slices
        )
        geometry_diagnostic["pancreas_voxels"] = int(
            np.sum(mask_review == 4)
        )
        geometry_diagnostic["lesion_voxels"] = int(
            np.sum(mask_review == 1)
        )
        geometry_diagnostic["overlay_path"] = str(overlay_path)

        del mask_review
        gc.collect()

        atomic_write_json(geometry_diagnostic, diagnostic_path)

        print()
        print("MANUAL GEOMETRY MISMATCH DIAGNOSTIC")
        print(json.dumps(geometry_diagnostic, indent=2))
        print(f"Diagnostic saved: {diagnostic_path}")

        raise RuntimeError(
            "NEW_MANUAL_CT_MASK_HEADER_MISMATCH_REVIEW_REQUIRED"
        )

    start_orientation = nib.orientations.io_orientation(ct_affine)
    target_orientation = nib.orientations.axcodes2ornt(
        ("L", "P", "S")
    )
    orientation_transform = nib.orientations.ornt_transform(
        start_orientation,
        target_orientation,
    )
    transformed_to_original = nib.orientations.inv_ornt_aff(
        orientation_transform,
        ct_shape,
    )
    canonical_affine = ct_affine @ transformed_to_original
    canonical_orientation = "".join(
        nib.aff2axcodes(canonical_affine)
    )
    if canonical_orientation != "LPS":
        raise RuntimeError(
            f"Canonical orientation failed: {canonical_orientation}"
        )

    proxy = ct_image.dataobj
    slope = 1.0 if proxy.slope is None else float(proxy.slope)
    intercept = 0.0 if proxy.inter is None else float(proxy.inter)
    if not np.isfinite(slope) or slope == 0 or not np.isfinite(intercept):
        raise RuntimeError("Invalid CT scaling parameters.")

    raw_array = np.asanyarray(proxy.get_unscaled())
    canonical_array = nib.orientations.apply_orientation(
        raw_array,
        orientation_transform,
    )
    canonical_shape = np.asarray(
        canonical_array.shape,
        dtype=int,
    )
    canonical_spacing = nib.affines.voxel_sizes(
        canonical_affine
    )
    volume_extent = (
        canonical_shape.astype(float) * canonical_spacing
    )

    usable_shape = (
        CANVAS_SHAPE - 2 * PADDING_VOXELS_PER_SIDE
    )
    effective_spacing = float(
        np.max(volume_extent / usable_shape)
    )
    scaled_shape = volume_extent / effective_spacing
    core_shape = np.ceil(
        scaled_shape - 1e-6
    ).astype(int)
    if np.any(core_shape > usable_shape):
        raise RuntimeError(
            f"Core shape exceeds usable canvas: "
            f"{core_shape}/{usable_shape}"
        )

    transform_matrix = np.diag(
        np.full(3, effective_spacing)
        / canonical_spacing
    )
    transform_offset = (
        (canonical_shape.astype(float) - 1.0) / 2.0
        - transform_matrix
        @ ((core_shape.astype(float) - 1.0) / 2.0)
    )

    raw_padding_value = (
        (HU_LOWER - intercept) / slope
    )
    core_ct = ndimage.affine_transform(
        canonical_array,
        matrix=transform_matrix,
        offset=transform_offset,
        output_shape=tuple(int(v) for v in core_shape),
        output=np.float32,
        order=1,
        mode="constant",
        cval=float(raw_padding_value),
        prefilter=False,
    )
    core_ct = core_ct * slope + intercept
    if not np.all(np.isfinite(core_ct)):
        raise RuntimeError("Non-finite resampled CT values.")
    core_ct = np.rint(
        np.clip(core_ct, HU_LOWER, HU_UPPER)
    ).astype(np.int16)

    total_padding = CANVAS_SHAPE - core_shape
    pad_before = total_padding // 2
    pad_after = total_padding - pad_before
    canvas = np.full(
        tuple(int(v) for v in CANVAS_SHAPE),
        int(HU_LOWER),
        dtype=np.int16,
    )
    insertion = tuple(
        slice(
            int(pad_before[axis]),
            int(pad_before[axis] + core_shape[axis]),
        )
        for axis in range(3)
    )
    canvas[insertion] = core_ct

    old_to_canonical = (
        np.linalg.inv(canonical_affine) @ ct_affine
    )
    original_center = np.asarray(
        pancreas_bbox["center_voxel"],
        dtype=float,
    )
    canonical_center = nib.affines.apply_affine(
        old_to_canonical,
        original_center,
    )
    core_center = np.linalg.solve(
        transform_matrix,
        canonical_center - transform_offset,
    )
    canvas_center = (
        core_center + pad_before.astype(float)
    )

    bbox_minimum = np.asarray(
        pancreas_bbox["minimum"],
        dtype=float,
    )
    bbox_maximum = np.asarray(
        pancreas_bbox["maximum"],
        dtype=float,
    )
    corners = np.asarray(
        list(
            product(
                [bbox_minimum[0], bbox_maximum[0]],
                [bbox_minimum[1], bbox_maximum[1]],
                [bbox_minimum[2], bbox_maximum[2]],
            )
        ),
        dtype=float,
    )
    canonical_corners = nib.affines.apply_affine(
        old_to_canonical,
        corners,
    )
    core_corners = (
        canonical_corners - transform_offset[None, :]
    ) @ np.linalg.inv(transform_matrix).T
    canvas_corners = (
        core_corners + pad_before[None, :]
    )
    bbox_size_canvas = (
        np.max(canvas_corners, axis=0)
        - np.min(canvas_corners, axis=0)
        + 1.0
    )

    if (
        not np.all(np.isfinite(canvas_center))
        or np.any(canvas_center < 0)
        or np.any(canvas_center > CANVAS_SHAPE - 1)
    ):
        raise RuntimeError(
            f"Pancreas centre outside canvas: {canvas_center}"
        )

    center_normalized = (
        canvas_center
        / (CANVAS_SHAPE.astype(float) - 1.0)
    )

    result = {
        "ct_hu": canvas,
        "pancreas_center_voxel": canvas_center.astype(np.float32),
        "pancreas_center_normalized": center_normalized.astype(np.float32),
        "pancreas_bbox_size_voxels": bbox_size_canvas.astype(np.float32),
        "ct_shape": ct_shape,
        "mask_shape": mask_shape,
        "ct_spacing": ct_spacing,
        "mask_spacing": mask_spacing,
        "ct_orientation": ct_orientation,
        "mask_orientation": mask_orientation,
        "canonical_shape": canonical_shape,
        "canonical_spacing": canonical_spacing,
        "effective_spacing": effective_spacing,
        "core_shape": core_shape,
        "pad_before": pad_before,
        "pad_after": pad_after,
        "geometry_status": geometry_status,
        "slope": slope,
        "intercept": intercept,
    }

    del core_ct, canonical_array, raw_array, ct_image, mask_image
    gc.collect()
    return result


print("=" * 112)
print("STAGE 1A — RESUMABLE LOCALIZER CT PREPROCESSING")
print("=" * 112)

required_paths = [
    CASE_GEOMETRY_PATH,
    FULL_LEDGER_PATH,
    REMOTE_INVENTORY_PATH,
    SOURCE_AUDIT_PATH,
    PROTOCOL_PATH,
]
for required_path in required_paths:
    if not required_path.exists():
        raise FileNotFoundError(f"Required file missing:\n{required_path}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

case_geometry = pd.read_csv(CASE_GEOMETRY_PATH)
full_ledger = pd.read_csv(FULL_LEDGER_PATH)
inventory = pd.read_csv(REMOTE_INVENTORY_PATH)
sources = pd.read_csv(SOURCE_AUDIT_PATH)

for dataframe in [case_geometry, full_ledger, inventory]:
    dataframe["study_id"] = (
        dataframe["study_id"].map(normalize_study_id)
    )

with open(PROTOCOL_PATH, "r", encoding="utf-8") as file:
    protocol = json.load(file)

if protocol.get("selected_candidate") != "F1_96x96x160":
    raise RuntimeError("F1 localizer protocol is not locked.")
if len(case_geometry) != 1671:
    raise RuntimeError(
        f"Expected 1671 development cases, observed {len(case_geometry)}."
    )
if not set(case_geometry["partition"]).issubset(
    {"train", "validation"}
):
    raise RuntimeError("A locked test case entered Stage 1A.")

source_columns = sources[
    [
        "batch_number",
        "selected_range_url",
        "safe_range_supported",
    ]
]
remote = inventory.merge(
    source_columns,
    on="batch_number",
    how="left",
    validate="many_to_one",
)

target = (
    case_geometry
    .merge(
        full_ledger[
            [
                "study_id",
                "mask_path",
                "mask_shape_json",
                "mask_spacing_mm_json",
                "pancreas_bbox_json",
                "annotation_type",
            ]
        ],
        on=["study_id", "annotation_type"],
        how="inner",
        validate="one_to_one",
    )
    .merge(
        remote,
        on="study_id",
        how="inner",
        validate="one_to_one",
        suffixes=("", "_remote"),
    )
)

if len(target) != 1671 or target["study_id"].nunique() != 1671:
    raise RuntimeError("Stage 1A target merge is incomplete.")
if not as_bool_series(target["safe_range_supported"]).all():
    raise RuntimeError("A target lacks a safe HTTP range endpoint.")

if OUTPUT_LEDGER_PATH.exists():
    output_ledger = pd.read_csv(OUTPUT_LEDGER_PATH)
    output_ledger["study_id"] = (
        output_ledger["study_id"].map(normalize_study_id)
    )
    if output_ledger["study_id"].duplicated().any():
        raise RuntimeError("Duplicate IDs in Stage 1A checkpoint.")
else:
    output_ledger = pd.DataFrame()

completed_ids = (
    set(output_ledger["study_id"])
    if len(output_ledger) > 0
    else set()
)
target_ids = set(target["study_id"])
if completed_ids - target_ids:
    raise RuntimeError("Checkpoint contains unexpected study IDs.")

if len(output_ledger) > 0:
    missing_outputs = [
        path
        for path in output_ledger["output_path"]
        if not Path(str(path)).exists()
    ]
    if missing_outputs:
        raise RuntimeError(
            f"Checkpoint output files missing: {missing_outputs[:5]}"
        )

pending = target.loc[
    ~target["study_id"].isin(completed_ids)
].copy()

# Process manual labels first to expose any previously unknown manual-header issue early.
pending["_manual_first"] = (
    pending["annotation_type"].astype(str).str.lower() != "manual"
)
pending = pending.sort_values(
    ["_manual_first", "compressed_size_bytes", "study_id"]
).reset_index(drop=True)
selected = pending.head(MAX_CASES_THIS_RUN).copy()

drive_free_gib = (
    shutil.disk_usage(PROJECT_ROOT).free / (1024 ** 3)
)
local_free_gib = (
    shutil.disk_usage(RUNTIME_ROOT).free / (1024 ** 3)
)
pending_transfer_gib = (
    selected["compressed_size_bytes"].sum() / (1024 ** 3)
)

if drive_free_gib < 10:
    raise RuntimeError(
        f"Insufficient Drive free space: {drive_free_gib:.2f} GiB"
    )
if local_free_gib < 12:
    raise RuntimeError(
        f"Insufficient local free space: {local_free_gib:.2f} GiB"
    )

print()
print(f"Development cases: {len(target)}")
print(f"Previously completed: {len(completed_ids)}")
print(f"Pending before this run: {len(pending)}")
print(f"Selected for this run: {len(selected)}")
print(f"Expected selected CT transfer: {pending_transfer_gib:.2f} GiB")
print(f"Drive free space: {drive_free_gib:.2f} GiB")
print(f"Local free space: {local_free_gib:.2f} GiB")
print("Locked test cases accessed: 0")
print("Checkpoint frequency: after every case")

session = requests.Session()
session.headers.update(
    {
        "User-Agent":
            "PDAC-Public-Q1-Project/Stage-1A-Localizer-Preprocessing"
    }
)

errors = []
completed_this_run = 0
transferred_this_run = 0

for order, (_, case) in enumerate(selected.iterrows(), start=1):
    study_id = case["study_id"]
    destination = OUTPUT_DIR / f"{study_id}.npz"
    destination_part = Path(str(destination) + ".part")

    if order == 1 or order % 5 == 0 or order == len(selected):
        print(
            f"\n[{order}/{len(selected)}] Processing {study_id} "
            f"— durable {len(completed_ids)}/1671"
        )

    try:
        if destination_part.exists():
            destination_part.unlink()

        mask_path = Path(str(case["mask_path"]))
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask missing: {mask_path}")

        pancreas_bbox = parse_bbox(case["pancreas_bbox_json"])
        endpoint = str(case["selected_range_url"])
        local_header_offset = int(case["local_header_offset"])
        compressed_size = int(case["compressed_size_bytes"])
        expected_recovered_size = int(case["uncompressed_size_bytes"])
        compression_type = int(case["compression_type"])
        member_name = str(case["member_name"])
        expected_crc = str(case["crc32_hex"]).lower().zfill(8)

        with tempfile.TemporaryDirectory(
            prefix=f"stage1a_{study_id}_",
            dir=RUNTIME_ROOT,
        ) as temporary_directory:
            temporary_directory = Path(temporary_directory)
            inner_gzip_path = temporary_directory / Path(member_name).name
            ct_nii_path = temporary_directory / f"{study_id}.nii"
            local_npz_path = temporary_directory / f"{study_id}.npz"

            member_header = resolve_member_data_start(
                session=session,
                endpoint=endpoint,
                local_header_offset=local_header_offset,
                expected_member_name=member_name,
                expected_compression_type=compression_type,
            )

            recovery = recover_zip_member(
                session=session,
                endpoint=endpoint,
                data_start=member_header["data_start"],
                compressed_size=compressed_size,
                compression_type=compression_type,
                output_path=inner_gzip_path,
            )

            if recovery["recovered_size_bytes"] != expected_recovered_size:
                raise RuntimeError(
                    f"Recovered-size mismatch: "
                    f"{recovery['recovered_size_bytes']}/"
                    f"{expected_recovered_size}"
                )
            if recovery["crc32_hex"].lower() != expected_crc:
                raise RuntimeError(
                    f"CRC mismatch: "
                    f"{recovery['crc32_hex']}/{expected_crc}"
                )

            decompressed_nifti_size = decompress_nifti_gzip(
                inner_gzip_path,
                ct_nii_path,
            )

            processed = process_ct_to_canvas(
                ct_nii_path=ct_nii_path,
                mask_path=mask_path,
                pancreas_bbox=pancreas_bbox,
                annotation_type=str(case["annotation_type"]).lower(),
                study_id=study_id,
            )

            np.savez_compressed(
                local_npz_path,
                ct_hu=processed["ct_hu"],
                pancreas_center_voxel=processed[
                    "pancreas_center_voxel"
                ],
                pancreas_center_normalized=processed[
                    "pancreas_center_normalized"
                ],
                pancreas_bbox_size_voxels=processed[
                    "pancreas_bbox_size_voxels"
                ],
            )

            local_hash = sha256_file(local_npz_path)
            shutil.copy2(local_npz_path, destination_part)
            os.replace(destination_part, destination)

            destination_hash = sha256_file(destination)
            if destination_hash != local_hash:
                raise RuntimeError("Output NPZ SHA-256 mismatch.")

            with np.load(destination, allow_pickle=False) as check:
                if check["ct_hu"].shape != tuple(CANVAS_SHAPE):
                    raise RuntimeError("Stored CT canvas shape mismatch.")
                if check["ct_hu"].dtype != np.int16:
                    raise RuntimeError("Stored CT dtype mismatch.")
                if not np.all(np.isfinite(check["ct_hu"])):
                    raise RuntimeError("Stored CT contains non-finite values.")
                if (
                    np.min(check["ct_hu"]) < HU_LOWER
                    or np.max(check["ct_hu"]) > HU_UPPER
                ):
                    raise RuntimeError("Stored CT lies outside HU lock.")
                if not np.all(
                    np.isfinite(check["pancreas_center_normalized"])
                ):
                    raise RuntimeError("Stored target centre is non-finite.")

            row = {
                "study_id": study_id,
                "patient_id": case["patient_id"],
                "partition": case["partition"],
                "diagnostic_label": case["diagnostic_label"],
                "annotation_type": case["annotation_type"],
                "batch_number": int(case["batch_number"]),
                "member_name": member_name,
                "compressed_size_bytes": compressed_size,
                "remote_bytes_transferred": (
                    recovery["compressed_bytes_downloaded"]
                    + member_header["header_transfer_bytes"]
                ),
                "recovered_inner_gzip_size_bytes": (
                    recovery["recovered_size_bytes"]
                ),
                "decompressed_nifti_size_bytes": decompressed_nifti_size,
                "source_crc32_hex": expected_crc,
                "recovered_crc32_hex": recovery["crc32_hex"],
                "source_integrity_pass": True,
                "ct_original_shape_json": json.dumps(
                    [int(v) for v in processed["ct_shape"]]
                ),
                "ct_original_spacing_mm_json": json.dumps(
                    [float(v) for v in processed["ct_spacing"]]
                ),
                "ct_original_orientation": processed["ct_orientation"],
                "canonical_shape_json": json.dumps(
                    [int(v) for v in processed["canonical_shape"]]
                ),
                "canonical_spacing_mm_json": json.dumps(
                    [float(v) for v in processed["canonical_spacing"]]
                ),
                "effective_isotropic_spacing_mm": (
                    processed["effective_spacing"]
                ),
                "resampled_core_shape_json": json.dumps(
                    [int(v) for v in processed["core_shape"]]
                ),
                "pad_before_json": json.dumps(
                    [int(v) for v in processed["pad_before"]]
                ),
                "pad_after_json": json.dumps(
                    [int(v) for v in processed["pad_after"]]
                ),
                "output_canvas_shape_json": json.dumps(
                    [int(v) for v in CANVAS_SHAPE]
                ),
                "output_ct_dtype": "int16",
                "output_HU_min": int(np.min(processed["ct_hu"])),
                "output_HU_max": int(np.max(processed["ct_hu"])),
                "pancreas_center_voxel_json": json.dumps(
                    [
                        float(v)
                        for v in processed["pancreas_center_voxel"]
                    ]
                ),
                "pancreas_center_normalized_json": json.dumps(
                    [
                        float(v)
                        for v in processed["pancreas_center_normalized"]
                    ]
                ),
                "pancreas_bbox_size_voxels_json": json.dumps(
                    [
                        float(v)
                        for v in processed["pancreas_bbox_size_voxels"]
                    ]
                ),
                "geometry_status": processed["geometry_status"],
                "output_path": str(destination),
                "output_size_bytes": destination.stat().st_size,
                "output_sha256": destination_hash,
                "output_reopen_pass": True,
                "processing_complete": True,
                "processed_at_utc": utc_now(),
            }

            output_ledger = pd.concat(
                [output_ledger, pd.DataFrame([row])],
                ignore_index=True,
            )
            output_ledger = output_ledger.sort_values(
                "study_id"
            ).reset_index(drop=True)
            atomic_write_csv(output_ledger, OUTPUT_LEDGER_PATH)

            completed_ids.add(study_id)
            completed_this_run += 1
            transferred_this_run += int(
                row["remote_bytes_transferred"]
            )

            del processed
            gc.collect()

    except Exception as error:
        if destination_part.exists():
            destination_part.unlink()

        error_row = {
            "study_id": study_id,
            "partition": case["partition"],
            "annotation_type": case["annotation_type"],
            "batch_number": int(case["batch_number"]),
            "member_name": case["member_name"],
            "error_type": type(error).__name__,
            "error_message": str(error),
            "failed_at_utc": utc_now(),
        }
        errors.append(error_row)

        if ERROR_PATH.exists():
            previous_errors = pd.read_csv(ERROR_PATH)
        else:
            previous_errors = pd.DataFrame()
        updated_errors = pd.concat(
            [previous_errors, pd.DataFrame([error_row])],
            ignore_index=True,
        )
        atomic_write_csv(updated_errors, ERROR_PATH)

        print(
            f"\nFAIL — {study_id}: "
            f"{type(error).__name__}: {error}"
        )
        print("Processing stopped; completed cases remain checkpointed.")
        break

session.close()

durably_completed = len(completed_ids)
remaining = 1671 - durably_completed

if errors:
    run_status = "STOPPED_WITH_ERROR"
elif remaining == 0:
    run_status = "COMPLETE"
else:
    run_status = "SESSION_INCOMPLETE_RERUN_SAME_CELL"

if len(output_ledger) > 0:
    persistent_size = int(output_ledger["output_size_bytes"].sum())
    geometry_counts = (
        output_ledger["geometry_status"].value_counts().to_dict()
    )
else:
    persistent_size = 0
    geometry_counts = {}

run_audit = {
    "stage": "1A",
    "created_at_utc": utc_now(),
    "expected_development_cases": 1671,
    "durably_completed": durably_completed,
    "remaining": remaining,
    "completed_this_run": completed_this_run,
    "errors_this_run": len(errors),
    "transferred_this_run_GiB": (
        transferred_this_run / (1024 ** 3)
    ),
    "persistent_output_GiB": (
        persistent_size / (1024 ** 3)
    ),
    "locked_test_cases_accessed": 0,
    "geometry_status_counts": {
        str(key): int(value)
        for key, value in geometry_counts.items()
    },
    "checkpoint_path": str(OUTPUT_LEDGER_PATH),
    "output_directory": str(OUTPUT_DIR),
    "run_status": run_status,
}
atomic_write_json(run_audit, RUN_AUDIT_PATH)

print()
print("-" * 112)
print("STAGE 1A RUN RESULT")
print("-" * 112)
print(f"Completed this run: {completed_this_run}")
print(f"Errors this run: {len(errors)}")
print(f"Durably completed: {durably_completed}/1671")
print(f"Remaining: {remaining}")
print(
    "Transferred this run:",
    f"{transferred_this_run / (1024 ** 3):.3f} GiB",
)
print(
    "Persistent processed data:",
    f"{persistent_size / (1024 ** 3):.3f} GiB",
)
print("Locked test cases accessed: 0")
print("Checkpoint:")
print(OUTPUT_LEDGER_PATH)
print()
print("=" * 112)
print(f"STAGE 1A STATUS: {run_status}")
print("=" * 112)

if errors:
    raise RuntimeError(
        "Stage 1A stopped on an error. "
        "Completed cases remain durably saved."
    )
