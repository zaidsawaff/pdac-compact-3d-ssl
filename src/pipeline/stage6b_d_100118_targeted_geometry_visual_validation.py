from pathlib import Path
from tempfile import TemporaryDirectory
from datetime import datetime, timezone
import binascii
import gzip
import json
import os
import shutil
import struct
import time
import zlib

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import requests


# =============================================================================
# STAGE 6B-D — TARGETED POST-FREEZE GEOMETRY VISUAL VALIDATION
# One review case only: 100118_00001. No model inference is repeated.
# =============================================================================

PROJECT_ROOT = Path(os.environ.get("PDAC_PROJECT_ROOT", "/content/drive/MyDrive/PDAC_Public_Q1_Project"))
META_DIR = PROJECT_ROOT / "01_Metadata"
QC_DIR = PROJECT_ROOT / "02_Quality_Control"

TARGET_STUDY_ID = "100118_00001"
MASK_PATH = (
    PROJECT_ROOT
    / "00_Raw"
    / "PANORAMA"
    / "Automatic_Labels"
    / f"{TARGET_STUDY_ID}.nii.gz"
)

REMOTE_MEMBER_PATH = META_DIR / "stage0i_panorama_remote_zip_member_inventory.csv"
REMOTE_SOURCE_PATH = META_DIR / "stage0i_panorama_remote_archive_source_audit.csv"
STAGE6B_GEOMETRY_PATH = QC_DIR / "stage6b_internal_test_geometry_audit.csv"

OVERLAY_PATH = QC_DIR / "stage6b_100118_index_alignment_overlay.png"
OUTPUT_CSV = QC_DIR / "stage6b_100118_geometry_visual_qc.csv"
OUTPUT_JSON = QC_DIR / "stage6b_100118_geometry_visual_qc.json"

HTTP_CHUNK_BYTES = 8 * 1024 * 1024
HTTP_CONNECT_TIMEOUT = 30
HTTP_READ_TIMEOUT = 180
HTTP_MAX_ATTEMPTS = 12
CANONICAL_AXCODES = ("L", "P", "S")
HU_WINDOW = (-200.0, 300.0)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def py_scalar(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def json_ready(value):
    if isinstance(value, np.ndarray):
        return [json_ready(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    return py_scalar(value)


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
                headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"},
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
    if temporary.exists():
        temporary.unlink()
    completed = 0
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
    header, _ = fetch_small_range(
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
    compressed_path.unlink()
    crc_value &= 0xFFFFFFFF
    if recovered_size != expected_size or crc_value != expected_crc:
        raise RuntimeError("Recovered ZIP member failed size/CRC verification.")
    with open(inner_path, "rb") as file:
        if file.read(2) != b"\x1f\x8b":
            raise RuntimeError("Recovered member is not gzip NIfTI.")
    return inner_path, compressed_size, download_endpoint


def decompress_gzip_to_nii(source, destination):
    with gzip.open(source, "rb") as compressed, open(destination, "wb") as output:
        shutil.copyfileobj(compressed, output, length=HTTP_CHUNK_BYTES)
        output.flush()
        os.fsync(output.fileno())


def orientation_string(image):
    return "".join(nib.aff2axcodes(image.affine))


def canonical_geometry(image):
    source_orientation = nib.orientations.io_orientation(image.affine)
    target_orientation = nib.orientations.axcodes2ornt(CANONICAL_AXCODES)
    transform = nib.orientations.ornt_transform(source_orientation, target_orientation)
    canonical_affine = image.affine @ nib.orientations.inv_ornt_aff(
        transform, image.shape[:3]
    )
    canonical_shape = np.asarray(image.shape[:3], dtype=int)[
        transform[:, 0].astype(int)
    ]
    return {
        "affine": np.asarray(canonical_affine, dtype=float),
        "shape": canonical_shape,
        "spacing": np.linalg.norm(canonical_affine[:3, :3], axis=0),
        "orientation": "".join(nib.aff2axcodes(canonical_affine)),
        "transform": transform,
    }


def proxy_scaling(image):
    slope = getattr(image.dataobj, "slope", None)
    intercept = getattr(image.dataobj, "inter", None)
    slope = 1.0 if slope is None else float(slope)
    intercept = 0.0 if intercept is None else float(intercept)
    if not np.isfinite(slope) or slope == 0 or not np.isfinite(intercept):
        raise RuntimeError("Invalid CT intensity scaling.")
    return slope, intercept


def choose_slices(counts, number=6):
    positive = np.flatnonzero(np.asarray(counts) > 0)
    if len(positive) == 0:
        raise RuntimeError("Pancreas label 4 is absent; visual alignment cannot be audited.")
    quantile_positions = np.linspace(0, len(positive) - 1, number)
    chosen = [int(positive[int(round(v))]) for v in quantile_positions]
    peak = int(np.argmax(counts))
    if peak not in chosen:
        # Replace the interior slice closest to the peak so the highest-content
        # pancreas section is always represented.
        interior = list(range(1, max(1, len(chosen) - 1)))
        if interior:
            replace_index = min(interior, key=lambda i: abs(chosen[i] - peak))
            chosen[replace_index] = peak
    chosen = sorted(set(chosen))
    while len(chosen) < min(number, len(positive)):
        for z in positive:
            if int(z) not in chosen:
                chosen.append(int(z))
                if len(chosen) >= min(number, len(positive)):
                    break
    return sorted(chosen)


def describe(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"count": 0}
    return {
        "count": int(values.size),
        "minimum": float(np.min(values)),
        "p01": float(np.percentile(values, 1)),
        "median": float(np.median(values)),
        "p99": float(np.percentile(values, 99)),
        "maximum": float(np.max(values)),
        "fraction_minus150_to_300": float(np.mean((values >= -150) & (values <= 300))),
    }


print("=" * 112)
print("STAGE 6B-D — TARGETED POST-FREEZE GEOMETRY VISUAL VALIDATION")
print("=" * 112)
print(f"Target study: {TARGET_STUDY_ID}")
print("Purpose: geometry review only; no model inference and no metric computation.")

required = [REMOTE_MEMBER_PATH, REMOTE_SOURCE_PATH, MASK_PATH]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise FileNotFoundError("Missing required input(s): " + " | ".join(missing))

member_inventory = pd.read_csv(REMOTE_MEMBER_PATH)
source_inventory = pd.read_csv(REMOTE_SOURCE_PATH)
member_inventory["study_id"] = member_inventory["study_id"].astype(str).str.strip()
target_members = member_inventory.loc[member_inventory["study_id"] == TARGET_STUDY_ID]
if len(target_members) != 1:
    raise RuntimeError(f"Expected one remote member for {TARGET_STUDY_ID}; found {len(target_members)}")
member_row = target_members.iloc[0]
batch_number = int(member_row["batch_number"])
source_rows = source_inventory.loc[
    pd.to_numeric(source_inventory["batch_number"], errors="coerce") == batch_number
]
if len(source_rows) != 1:
    raise RuntimeError(f"Expected one remote source row for batch {batch_number}.")
source_row = source_rows.iloc[0]

existing_geometry = None
if STAGE6B_GEOMETRY_PATH.exists():
    geometry_frame = pd.read_csv(STAGE6B_GEOMETRY_PATH)
    if "study_id" in geometry_frame.columns:
        geometry_frame["study_id"] = geometry_frame["study_id"].astype(str).str.strip()
        rows = geometry_frame.loc[geometry_frame["study_id"] == TARGET_STUDY_ID]
        if len(rows) == 1:
            existing_geometry = {
                key: py_scalar(value)
                for key, value in rows.iloc[0].to_dict().items()
                if not (isinstance(value, float) and np.isnan(value))
            }

QC_DIR.mkdir(parents=True, exist_ok=True)

with TemporaryDirectory(prefix="stage6b_100118_") as temporary_directory:
    work_dir = Path(temporary_directory)
    session = requests.Session()
    print(f"Downloading only the target ZIP member from batch {batch_number}...")
    inner_ct_gz, transferred_bytes, selected_endpoint = recover_zip_member(
        session, member_row, source_row, work_dir
    )

    ct_nii = work_dir / "ct.nii"
    mask_nii = work_dir / "mask.nii"
    decompress_gzip_to_nii(inner_ct_gz, ct_nii)
    decompress_gzip_to_nii(MASK_PATH, mask_nii)

    ct_image = nib.load(str(ct_nii), mmap=True)
    mask_image = nib.load(str(mask_nii), mmap=True)
    ct_shape = tuple(int(v) for v in ct_image.shape[:3])
    mask_shape = tuple(int(v) for v in mask_image.shape[:3])
    ct_spacing = tuple(float(v) for v in ct_image.header.get_zooms()[:3])
    mask_spacing = tuple(float(v) for v in mask_image.header.get_zooms()[:3])
    ct_orientation = orientation_string(ct_image)
    mask_orientation = orientation_string(mask_image)
    shape_match = ct_shape == mask_shape
    orientation_match = ct_orientation == mask_orientation
    raw_affine_match = bool(np.allclose(ct_image.affine, mask_image.affine, atol=1e-3, rtol=0))
    max_affine_difference = float(np.max(np.abs(ct_image.affine - mask_image.affine)))
    ct_canonical = canonical_geometry(ct_image)
    mask_canonical = canonical_geometry(mask_image)
    canonical_shape_match = bool(np.array_equal(ct_canonical["shape"], mask_canonical["shape"]))
    canonical_affine_match = bool(
        np.allclose(ct_canonical["affine"], mask_canonical["affine"], atol=1e-3, rtol=0)
    )

    print("\nRAW CT–MASK GEOMETRY")
    print("-" * 112)
    print(f"CT shape: {ct_shape}")
    print(f"Mask shape: {mask_shape}")
    print(f"CT spacing: {ct_spacing}")
    print(f"Mask spacing: {mask_spacing}")
    print(f"CT orientation: {ct_orientation}")
    print(f"Mask orientation: {mask_orientation}")
    print(f"Shape match: {shape_match}")
    print(f"Orientation match: {orientation_match}")
    print(f"Raw affine match: {raw_affine_match}")
    print(f"Maximum absolute affine difference: {max_affine_difference:.6f} mm")

    if not shape_match:
        raise RuntimeError("Raw CT and mask shapes differ; voxel-index overlay is not valid.")

    ct_raw = ct_image.dataobj.get_unscaled()
    mask_raw = mask_image.dataobj.get_unscaled()
    slope, intercept = proxy_scaling(ct_image)

    pancreas_counts = np.zeros(ct_shape[2], dtype=np.int64)
    vessel_counts = np.zeros(ct_shape[2], dtype=np.int64)
    lesion_counts = np.zeros(ct_shape[2], dtype=np.int64)
    pancreas_hu_chunks = []

    print("\nScanning labels and paired CT intensities slice by slice...")
    for z in range(ct_shape[2]):
        mask_slice = np.asarray(mask_raw[:, :, z])
        pancreas = mask_slice == 4
        pancreas_counts[z] = int(np.count_nonzero(pancreas))
        vessel_counts[z] = int(np.count_nonzero((mask_slice == 2) | (mask_slice == 3)))
        lesion_counts[z] = int(np.count_nonzero(mask_slice == 1))
        if pancreas_counts[z] > 0:
            raw_values = np.asarray(ct_raw[:, :, z])[pancreas]
            hu_values = raw_values.astype(np.float64) * slope + intercept
            if not np.all(np.isfinite(hu_values)):
                raise RuntimeError("Non-finite CT intensity found inside pancreas label.")
            pancreas_hu_chunks.append(hu_values)
        if (z + 1) % 50 == 0 or z + 1 == ct_shape[2]:
            print(f"  Slice scan: {z + 1}/{ct_shape[2]}")

    if int(lesion_counts.sum()) != 0:
        raise RuntimeError(
            "Target is expected to be non-PDAC but automatic mask contains lesion label 1."
        )
    selected_slices = choose_slices(pancreas_counts, number=6)
    pancreas_hu = np.concatenate(pancreas_hu_chunks) if pancreas_hu_chunks else np.asarray([])
    pancreas_statistics = describe(pancreas_hu)

    figure, axes = plt.subplots(2, 3, figsize=(18, 12), constrained_layout=True)
    axes = axes.reshape(-1)
    for axis_index, axis in enumerate(axes):
        if axis_index >= len(selected_slices):
            axis.axis("off")
            continue
        z = selected_slices[axis_index]
        raw_slice = np.asarray(ct_raw[:, :, z])
        ct_slice = raw_slice.astype(np.float32) * slope + intercept
        mask_slice = np.asarray(mask_raw[:, :, z])

        # Identical transpose/origin is applied to CT and all contours, so this
        # is a direct voxel-index alignment visualization.
        axis.imshow(
            ct_slice.T,
            cmap="gray",
            vmin=HU_WINDOW[0],
            vmax=HU_WINDOW[1],
            origin="lower",
            interpolation="nearest",
        )
        if np.any(mask_slice == 4):
            axis.contour(
                (mask_slice == 4).T,
                levels=[0.5],
                colors=["cyan"],
                linewidths=1.8,
                origin="lower",
            )
        if np.any(mask_slice == 2):
            axis.contour(
                (mask_slice == 2).T,
                levels=[0.5],
                colors=["magenta"],
                linewidths=1.25,
                origin="lower",
            )
        if np.any(mask_slice == 3):
            axis.contour(
                (mask_slice == 3).T,
                levels=[0.5],
                colors=["yellow"],
                linewidths=1.25,
                origin="lower",
            )
        axis.set_title(
            f"z={z} | pancreas={pancreas_counts[z]:,} | vessels={vessel_counts[z]:,}",
            fontsize=11,
        )
        axis.axis("off")

    figure.suptitle(
        "Study 100118_00001 — voxel-index geometry review\n"
        "Cyan: pancreas label 4 | Magenta: vein label 2 | Yellow: artery label 3",
        fontsize=16,
    )
    figure.savefig(OVERLAY_PATH, dpi=170, bbox_inches="tight")
    plt.close(figure)

    record = {
        "stage": "6B-D",
        "study_id": TARGET_STUDY_ID,
        "purpose": "TARGETED_POSTFREEZE_CT_MASK_GEOMETRY_VISUAL_VALIDATION",
        "model_inference_repeated": False,
        "prediction_files_modified": False,
        "raw_ct_or_mask_modified": False,
        "metric_computation_performed": False,
        "batch_number": batch_number,
        "member_name": str(member_row["member_name"]),
        "compressed_transfer_bytes": int(transferred_bytes),
        "selected_range_endpoint": str(selected_endpoint),
        "ct_shape": list(ct_shape),
        "mask_shape": list(mask_shape),
        "ct_spacing_mm": list(ct_spacing),
        "mask_spacing_mm": list(mask_spacing),
        "ct_orientation": ct_orientation,
        "mask_orientation": mask_orientation,
        "shape_match": shape_match,
        "orientation_match": orientation_match,
        "raw_affine_match": raw_affine_match,
        "maximum_absolute_affine_difference_mm": max_affine_difference,
        "ct_affine": np.asarray(ct_image.affine, dtype=float),
        "mask_affine": np.asarray(mask_image.affine, dtype=float),
        "canonical_ct_shape": ct_canonical["shape"],
        "canonical_mask_shape": mask_canonical["shape"],
        "canonical_shape_match": canonical_shape_match,
        "canonical_affine_match": canonical_affine_match,
        "canonical_ct_affine": ct_canonical["affine"],
        "canonical_mask_affine": mask_canonical["affine"],
        "selected_overlay_slices": selected_slices,
        "pancreas_voxels": int(pancreas_counts.sum()),
        "vessel_voxels": int(vessel_counts.sum()),
        "lesion_voxels": int(lesion_counts.sum()),
        "pancreas_hu_statistics": pancreas_statistics,
        "existing_stage6b_geometry_record": existing_geometry,
        "overlay_path": str(OVERLAY_PATH),
        "visual_review_status": "PENDING_HUMAN_VISUAL_REVIEW",
        "checked_at_utc": utc_now(),
    }

record = json_ready(record)
with open(OUTPUT_JSON, "w", encoding="utf-8") as file:
    json.dump(record, file, indent=2, ensure_ascii=False)

csv_record = {
    "study_id": TARGET_STUDY_ID,
    "batch_number": batch_number,
    "ct_shape": json.dumps(record["ct_shape"]),
    "mask_shape": json.dumps(record["mask_shape"]),
    "ct_spacing_mm": json.dumps(record["ct_spacing_mm"]),
    "mask_spacing_mm": json.dumps(record["mask_spacing_mm"]),
    "ct_orientation": record["ct_orientation"],
    "mask_orientation": record["mask_orientation"],
    "shape_match": record["shape_match"],
    "orientation_match": record["orientation_match"],
    "raw_affine_match": record["raw_affine_match"],
    "maximum_absolute_affine_difference_mm": record[
        "maximum_absolute_affine_difference_mm"
    ],
    "selected_overlay_slices": json.dumps(record["selected_overlay_slices"]),
    "pancreas_voxels": record["pancreas_voxels"],
    "vessel_voxels": record["vessel_voxels"],
    "lesion_voxels": record["lesion_voxels"],
    "pancreas_hu_statistics": json.dumps(record["pancreas_hu_statistics"]),
    "visual_review_status": record["visual_review_status"],
    "overlay_path": record["overlay_path"],
    "checked_at_utc": record["checked_at_utc"],
}
pd.DataFrame([csv_record]).to_csv(OUTPUT_CSV, index=False)

print("\n" + "-" * 112)
print("RECOVERED SUMMARY")
print("-" * 112)
print(f"Selected overlay slices: {record['selected_overlay_slices']}")
print(f"Pancreas statistics: {record['pancreas_hu_statistics']}")
print(f"Overlay image: {OVERLAY_PATH}")
print(f"Visual-QC CSV: {OUTPUT_CSV}")
print(f"Visual-QC JSON: {OUTPUT_JSON}")
print("Raw CT/mask modified: False")
print("Model inference repeated: False")
print("Metrics computed: False")
print("\n" + "=" * 112)
print("STAGE 6B-D RESULT: PASS_VISUAL_REVIEW_PENDING")
print("=" * 112)
