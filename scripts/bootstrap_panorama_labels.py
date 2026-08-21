#!/usr/bin/env python3

from pathlib import Path
import argparse
import shutil
import subprocess


REPOSITORY = "https://github.com/DIAGNijmegen/panorama_labels.git"
COMMIT = "bf1d6ba3"


def run(cmd):
    print("+", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Retrieve the exact PANORAMA labels used by the study."
    )
    parser.add_argument(
        "--project-root",
        default="./PDAC_Public_Q1_Project",
        help="Local project root."
    )
    args = parser.parse_args()

    root = Path(args.project_root).resolve()

    raw = root / "00_Raw" / "PANORAMA"
    manual_dst = raw / "Manual_Labels"
    automatic_dst = raw / "Automatic_Labels"

    work = root / "_source_downloads" / "panorama_labels"

    if work.exists():
        shutil.rmtree(work)

    work.parent.mkdir(parents=True, exist_ok=True)

    run([
        "git", "clone",
        "--filter=blob:none",
        REPOSITORY,
        str(work)
    ])

    run(["git", "-C", str(work), "checkout", COMMIT])

    manual_src = work / "manual_labels"
    automatic_src = work / "automatic_labels"

    manual_files = sorted(manual_src.glob("*.nii.gz"))
    automatic_files = sorted(automatic_src.glob("*.nii.gz"))

    if len(manual_files) != 482:
        raise RuntimeError(
            f"Expected 482 manual labels, found {len(manual_files)}"
        )

    if len(automatic_files) != 1756:
        raise RuntimeError(
            f"Expected 1756 automatic labels, found {len(automatic_files)}"
        )

    manual_dst.mkdir(parents=True, exist_ok=True)
    automatic_dst.mkdir(parents=True, exist_ok=True)

    for p in manual_files:
        shutil.copy2(p, manual_dst / p.name)

    for p in automatic_files:
        shutil.copy2(p, automatic_dst / p.name)

    print()
    print("=" * 72)
    print("PANORAMA LABEL BOOTSTRAP COMPLETE")
    print("=" * 72)
    print("Commit             :", COMMIT)
    print("Manual labels      :", len(manual_files))
    print("Automatic labels   :", len(automatic_files))
    print("Total labels       :", len(manual_files) + len(automatic_files))
    print("Destination        :", raw)
    print("STATUS             : PASS")


if __name__ == "__main__":
    main()
