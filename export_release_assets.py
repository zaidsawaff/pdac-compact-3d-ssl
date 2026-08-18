#!/usr/bin/env python3
"""Build a reproducibility release for the PDAC project.

The exporter intentionally excludes raw CT volumes and source masks. It copies
study-generated metadata, audits, selected results, reproducibility code,
selected checkpoints, and prediction manifests, then writes SHA-256 checksums
and machine-readable inventories.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path

PROJECT_ROOT = Path('/content/drive/MyDrive/PDAC_Public_Q1_Project')
DEST = Path.cwd() / 'release_assets'

KNOWN_HASHES = {
    '04_Models/Localizer/F1_96x96x160/stage2a_localizer_best.pt':
        'bbf2825174f2a93fe07d37fb3bbbc51b0e7767427007bd187931bc1534ca4978',
    '04_Models/Revision_R1/StageR1B_LeakageFreeDualArm/SSL_INITIALIZED/stageR1b_best.pt':
        'f571a611c4125ef92b80c51e3276fb03a02d4073317aa2460dca1649ae655f81',
    '04_Models/Revision_R1/StageR1B_LeakageFreeDualArm/RANDOM_INITIALIZED/stageR1b_best.pt':
        '9ca78bb185ed3e4a4c6173daabdd91a6f0e83e25f3d2b713d42df7622e91d4b3',
}

CORE_ARTIFACTS = [
    '01_Metadata/stage0f_locked_study_split_index.csv',
    '01_Metadata/stage0i_panorama_remote_zip_member_inventory.csv',
    '01_Metadata/stage0i_panorama_remote_archive_source_audit.csv',
    '01_Metadata/stage0m_b_r_localizer_input_protocol.json',
    '01_Metadata/stage2d_final_deployment_crop_geometry_protocol.json',
    '01_Metadata/stage3b_e5_frozen_dataset_manifest.csv',
    '01_Metadata/stage3b_e5_dataset_freeze_protocol.json',
    '02_Quality_Control/stage3b_e5_dataset_freeze_audit.json',
    '01_Metadata/stageR1a_train_only_ssl_protocol.json',
    '02_Quality_Control/stageR1a_train_only_ssl_training_audit.json',
    '01_Metadata/stageR1b_validation_monitor_subset.csv',
    '01_Metadata/stageR1b_dual_arm_training_protocol.json',
    '02_Quality_Control/stageR1b_dual_arm_training_audit.json',
    '01_Metadata/stageR1c_supervised_model_selection.json',
    '02_Quality_Control/stageR1c_full_validation_comparison_audit.json',
    '01_Metadata/stageR1d_detection_and_froc_protocol.json',
    '02_Quality_Control/stageR1d_validation_froc_calibration_audit.json',
    '01_Metadata/stageR1e_prefreeze_model_threshold_lock.json',
    '01_Metadata/stageR1e_prefreeze_evidence_manifest.csv',
    '02_Quality_Control/stageR1e_prefreeze_model_threshold_audit.json',
    '01_Metadata/stageR1f_i_blind_internal_test_prediction_manifest.csv',
    '01_Metadata/stageR1f_i_blind_internal_test_prediction_freeze.json',
    '02_Quality_Control/stageR1f_i_blind_internal_test_inference_audit.json',
    '02_Quality_Control/stageR1g_internal_test_evaluation_audit.json',
    '01_Metadata/stageR1h_blind_external_test_prediction_manifest.csv',
    '01_Metadata/stageR1h_blind_external_test_prediction_freeze.json',
    '02_Quality_Control/stageR1h_blind_external_test_inference_audit.json',
    '02_Quality_Control/stageR1i_source_separated_external_evaluation_audit.json',
    '04_Models/Localizer/F1_96x96x160/stage2a_localizer_best.pt',
    '04_Models/Revision_R1/StageR1A_TrainOnlyMaskedContext3DCNN/stageR1a_ssl_final.pt',
    '04_Models/Revision_R1/StageR1B_LeakageFreeDualArm/SSL_INITIALIZED/stageR1b_best.pt',
    '04_Models/Revision_R1/StageR1B_LeakageFreeDualArm/RANDOM_INITIALIZED/stageR1b_best.pt',
]

R1J_ALTERNATIVES = [
    (
        '01_Metadata/stageR1j_revision_r1_final_evidence_manifest.csv',
        '01_Metadata/stageR1j_final_revision_evidence_manifest.csv',
    ),
    (
        '01_Metadata/stageR1j_revision_r1_final_evidence_freeze.json',
        '01_Metadata/stageR1j_final_revision_evidence_freeze.json',
    ),
    (
        '02_Quality_Control/stageR1j_revision_r1_final_evidence_audit.json',
        '02_Quality_Control/stageR1j_final_revision_evidence_audit.json',
    ),
    (
        '03_Results/Revision_R1_Final/stageR1j_revision_r1_publication_metric_summary.csv',
    ),
]

OPTIONAL_SOURCE_BASENAMES = [
    'stage5a_supervised_segmentation_readiness_and_protocol_lock.py',
    'stage5b_resumable_dual_arm_supervised_training.py',
    'stage5c_resumable_full_validation_comparison.py',
    'stage5d_resumable_validation_froc_calibration.py',
]

MANUSCRIPT_BASENAMES = []


@dataclass
class Record:
    category: str
    status: str
    source_path: str
    release_path: str
    size_bytes: int | None = None
    sha256: str | None = None
    note: str = ''


def sha256(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def copy_one(rel: str, category: str, records: list[Record], mismatches: list[dict]) -> bool:
    src = PROJECT_ROOT / rel
    if not src.is_file():
        records.append(Record(category, 'MISSING', rel, ''))
        return False

    dst = DEST / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    digest = sha256(dst)
    expected = KNOWN_HASHES.get(rel)
    note = ''

    if expected and digest != expected:
        note = 'KNOWN_HASH_MISMATCH'
        mismatches.append({'path': rel, 'expected': expected, 'observed': digest})

    records.append(
        Record(category, 'COPIED', rel, rel, dst.stat().st_size, digest, note)
    )
    return True


def find_by_basename(name: str) -> list[Path]:
    return sorted(p for p in PROJECT_ROOT.rglob(name) if p.is_file())


def copy_optional_source(name: str, records: list[Record]) -> None:
    matches = find_by_basename(name)

    if not matches:
        records.append(Record('source_code', 'MISSING_OPTIONAL', name, ''))
        return

    if len(matches) > 1:
        records.append(
            Record(
                'source_code',
                'AMBIGUOUS_NOT_COPIED',
                '; '.join(str(p.relative_to(PROJECT_ROOT)) for p in matches),
                '',
                note='Multiple files share this basename; choose manually.',
            )
        )
        return

    src = matches[0]
    rel_src = src.relative_to(PROJECT_ROOT)
    dst_rel = Path('05_Code') / src.name
    dst = DEST / dst_rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    digest = sha256(dst)

    records.append(
        Record(
            'source_code',
            'COPIED',
            str(rel_src),
            str(dst_rel),
            dst.stat().st_size,
            digest,
        )
    )


def inventory_manuscript(name: str, records: list[Record]) -> None:
    matches = find_by_basename(name)

    if not matches:
        records.append(
            Record(
                'manuscript',
                'NOT_FOUND',
                name,
                '',
                note='Not part of public bundle.',
            )
        )
        return

    for p in matches:
        records.append(
            Record(
                'manuscript',
                'FOUND_NOT_COPIED',
                str(p.relative_to(PROJECT_ROOT)),
                '',
                p.stat().st_size,
                sha256(p),
                'Intentionally excluded from the public reproducibility bundle.',
            )
        )


def write_outputs(records: list[Record], mismatches: list[dict]) -> None:
    DEST.mkdir(parents=True, exist_ok=True)

    with (DEST / 'release_asset_inventory.csv').open(
        'w', newline='', encoding='utf-8'
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        for r in records:
            writer.writerow(asdict(r))

    copied = [asdict(r) for r in records if r.status == 'COPIED']
    missing = [
        asdict(r)
        for r in records
        if 'MISSING' in r.status or r.status == 'NOT_FOUND'
    ]
    ambiguous = [
        asdict(r) for r in records if r.status == 'AMBIGUOUS_NOT_COPIED'
    ]

    manifest = {
        'project_root': str(PROJECT_ROOT),
        'release_root': str(DEST),
        'policy': {
            'raw_ct_volumes_copied': False,
            'source_masks_copied': False,
            'manuscript_copied': False,
            'selected_checkpoints_only': True,
        },
        'summary': {
            'copied': len(copied),
            'missing_or_not_found': len(missing),
            'ambiguous_not_copied': len(ambiguous),
            'hash_mismatches': len(mismatches),
        },
        'records': [asdict(r) for r in records],
        'hash_mismatches': mismatches,
    }

    (DEST / 'release_asset_manifest.json').write_text(
        json.dumps(manifest, indent=2),
        encoding='utf-8',
    )

    with (DEST / 'SHA256SUMS.txt').open('w', encoding='utf-8') as f:
        for r in records:
            if r.status == 'COPIED' and r.sha256:
                f.write(f'{r.sha256}  {r.release_path}\n')

    with (DEST / 'MISSING_ASSETS.txt').open('w', encoding='utf-8') as f:
        for r in records:
            if 'MISSING' in r.status or r.status in {
                'NOT_FOUND',
                'AMBIGUOUS_NOT_COPIED',
            }:
                f.write(
                    f'{r.status}\t{r.category}\t'
                    f'{r.source_path}\t{r.note}\n'
                )


def main() -> None:
    if not PROJECT_ROOT.exists():
        raise SystemExit(f'Project root not found: {PROJECT_ROOT}')

    records: list[Record] = []
    mismatches: list[dict] = []

    for rel in CORE_ARTIFACTS:
        copy_one(rel, 'core_evidence', records, mismatches)

    for alternatives in R1J_ALTERNATIVES:
        for rel in alternatives:
            if (PROJECT_ROOT / rel).is_file():
                copy_one(rel, 'final_R1J_evidence', records, mismatches)
                break
        else:
            records.append(
                Record(
                    'final_R1J_evidence',
                    'MISSING',
                    ' OR '.join(alternatives),
                    '',
                    note='No recognized R1J filename variant found.',
                )
            )

    for name in OPTIONAL_SOURCE_BASENAMES:
        copy_optional_source(name, records)

    for name in MANUSCRIPT_BASENAMES:
        inventory_manuscript(name, records)

    write_outputs(records, mismatches)

    print('=' * 88)
    print('PDAC REPRODUCIBILITY RELEASE EXPORT')
    print('=' * 88)
    print(f'Project root : {PROJECT_ROOT}')
    print(f'Destination  : {DEST}')
    print(f'Copied       : {sum(r.status == "COPIED" for r in records)}')
    print(
        'Missing      : '
        f'{sum("MISSING" in r.status or r.status == "NOT_FOUND" for r in records)}'
    )
    print(
        'Ambiguous    : '
        f'{sum(r.status == "AMBIGUOUS_NOT_COPIED" for r in records)}'
    )
    print(f'Hash mismatch: {len(mismatches)}')
    print('Raw CT/source masks: NOT COPIED')
    print('Manifest     : release_asset_manifest.json')
    print('Inventory    : release_asset_inventory.csv')
    print('Checksums    : SHA256SUMS.txt')
    print('Missing log  : MISSING_ASSETS.txt')

    if mismatches:
        raise SystemExit(2)

    print('\nKnown checkpoint hashes: PASS (for all known hashes encountered)')


if __name__ == '__main__':
    main()
