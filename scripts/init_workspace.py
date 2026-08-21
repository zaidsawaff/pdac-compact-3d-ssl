#!/usr/bin/env python3

from pathlib import Path
import argparse
import shutil


TOP_LEVEL_DIRS = [
    "00_Raw",
    "01_Metadata",
    "02_Quality_Control",
    "03_Preprocessed",
    "03_Processed",
    "03_Results",
    "04_Models",
    "04_SSL_Pretraining",
    "05_Models",
    "05_Predictions",
    "06_Predictions",
    "07_Results",
    "08_Figures",
    "09_Manuscript",
    "10_Logs",
    "11_Evidence_Packets",
    "12_Final_Freeze",
]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Initialize the directory structure and frozen protocol "
            "inputs required by the PDAC reproducibility pipeline."
        )
    )
    parser.add_argument(
        "--project-root",
        required=True,
        help="Destination project workspace.",
    )
    parser.add_argument(
        "--repro-root",
        required=True,
        help="Root of the reproducibility repository.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing frozen prerequisite files.",
    )

    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    repro_root = Path(args.repro_root).resolve()

    protocols_root = repro_root / "protocols"

    if not protocols_root.is_dir():
        raise RuntimeError(
            f"Protocol directory not found: {protocols_root}"
        )

    project_root.mkdir(parents=True, exist_ok=True)

    for rel in TOP_LEVEL_DIRS:
        (project_root / rel).mkdir(
            parents=True,
            exist_ok=True,
        )

    protocol_files = sorted(
        p for p in protocols_root.rglob("*")
        if p.is_file()
    )

    copied = 0
    preserved = 0

    for src in protocol_files:
        rel = src.relative_to(protocols_root)
        dst = project_root / rel

        dst.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if dst.exists() and not args.overwrite:
            preserved += 1
            continue

        shutil.copy2(src, dst)

        # Runtime relocation:
        # Preserve the archival protocol files in the repository unchanged,
        # but replace historical absolute project-root strings in workspace
        # copies so that mask/output paths point to this new workspace.
        if dst.suffix.lower() in {".csv", ".json", ".txt", ".md"}:
            try:
                content = dst.read_text(encoding="utf-8")
                historical_root = "/content/drive/MyDrive/PDAC_Public_Q1_Project"

                if historical_root in content:
                    content = content.replace(
                        historical_root,
                        str(project_root),
                    )
                    dst.write_text(content, encoding="utf-8")
            except UnicodeDecodeError:
                pass

        copied += 1

    print("=" * 80)
    print("PDAC REPRODUCIBILITY WORKSPACE INITIALIZER")
    print("=" * 80)
    print("Project root        :", project_root)
    print("Protocol files      :", len(protocol_files))
    print("Copied              :", copied)
    print("Already present     :", preserved)
    print()
    print("Set this environment variable before running stages:")
    print()
    print(f'export PDAC_PROJECT_ROOT="{project_root}"')
    print()
    print("STATUS: PASS")


if __name__ == "__main__":
    main()
