#!/usr/bin/env python3
"""
BraTS 2026 Task 1 — containerized inference entrypoint.

Reads each case folder from /input (read-only), runs the ResEnc-L ensemble,
and writes ONE flat <case>.nii.gz per case to /output.

Challenge spec compliance:
  - /input is read-only; we never write there.
  - /output is FLAT: no subfolders, filename = <5-digit case>-<3-digit tp>.nii.gz
  - No network access: model weights are baked into the image (see Dockerfile).
  - Each /input/<case>/ folder contains 4 modalities:
        <case>-t1c.nii.gz  <case>-t1n.nii.gz  <case>-t2f.nii.gz  <case>-t2w.nii.gz

This script wraps nnU-Net v2 predict + ensemble. It stages each case into the
nnU-Net "imagesTs" naming convention (_0000.._0003), predicts with the selected
folds, softmax-averages, and writes the final label map flat to /output.
"""

import os
import sys
import shutil
import subprocess
import tempfile
from pathlib import Path

INPUT_DIR  = Path("/input")
OUTPUT_DIR = Path("/output")

# nnU-Net expects these; results (checkpoints) are baked into the image.
os.environ.setdefault("nnUNet_raw", "/opt/nnunet/nnUNet_raw")
os.environ.setdefault("nnUNet_preprocessed", "/opt/nnunet/nnUNet_preprocessed")
os.environ.setdefault("nnUNet_results", "/opt/nnunet/nnUNet_results")
os.environ.setdefault("nnUNet_compile", "F")   # avoid torch.compile hang

DATASET = "501"
CONFIG = "3d_fullres"
PLANS = "nnUNetResEncUNetLPlans"
TRAINER = "nnUNetTrainer_500epochs"
# RC-competent folds only (exclude the RC-dead fold). Update after retraining.
FOLDS = ["0", "1", "3"]
CHECKPOINT = "checkpoint_best.pth"

# nnU-Net channel-to-modality mapping. Confirm against your dataset.json
# "channel_names". Standard BraTS ordering used here:
MODALITY_SUFFIX = {
    "0000": "t1n",   # native T1
    "0001": "t1c",   # post-contrast T1
    "0002": "t2w",   # T2
    "0003": "t2f",   # T2-FLAIR
}


def find_case_folders(input_dir: Path):
    """Each subfolder of /input is one case."""
    return sorted([p for p in input_dir.iterdir() if p.is_dir()])


def stage_case(case_dir: Path, staging: Path):
    """
    Copy a case's 4 modalities into nnU-Net imagesTs naming:
        <case>_0000.nii.gz ... <case>_0003.nii.gz
    Returns the case identifier (folder name).
    """
    case_id = case_dir.name
    for ch, suffix in MODALITY_SUFFIX.items():
        src = case_dir / f"{case_id}-{suffix}.nii.gz"
        if not src.exists():
            raise FileNotFoundError(f"missing modality: {src}")
        dst = staging / f"{case_id}_{ch}.nii.gz"
        shutil.copy(str(src), str(dst))
    return case_id


def run(cmd):
    print("RUN:", " ".join(cmd), flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise RuntimeError(f"command failed ({r.returncode}): {' '.join(cmd)}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cases = find_case_folders(INPUT_DIR)
    print(f"found {len(cases)} case folders in {INPUT_DIR}", flush=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        staging = tmp / "imagesTs"
        staging.mkdir()

        # stage all cases into nnU-Net naming
        for case_dir in cases:
            stage_case(case_dir, staging)

        # per-fold prediction (softmax saved for ensembling)
        fold_out_dirs = []
        for f in FOLDS:
            out_f = tmp / f"fold{f}_pred"
            out_f.mkdir()
            run([
                "nnUNetv2_predict",
                "-i", str(staging),
                "-o", str(out_f),
                "-d", DATASET, "-c", CONFIG,
                "-p", PLANS, "-tr", TRAINER,
                "-f", f, "-chk", CHECKPOINT,
                "--save_probabilities",
            ])
            fold_out_dirs.append(str(out_f))

        # ensemble (or single fold)
        ens_out = tmp / "ensemble_pred"
        ens_out.mkdir()
        if len(fold_out_dirs) > 1:
            run(["nnUNetv2_ensemble", "-i", *fold_out_dirs,
                 "-o", str(ens_out), "-np", "4"])
            final_dir = ens_out
        else:
            final_dir = Path(fold_out_dirs[0])

        # write predictions FLAT to /output
        n = 0
        for nii in sorted(final_dir.glob("*.nii.gz")):
            shutil.copy(str(nii), str(OUTPUT_DIR / nii.name))
            n += 1
        print(f"wrote {n} predictions to {OUTPUT_DIR}", flush=True)

    print("done.", flush=True)


if __name__ == "__main__":
    main()
