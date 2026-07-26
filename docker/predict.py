#!/usr/bin/env python3
"""
BraTS 2026 Task 1 - containerized inference entrypoint.

Pipeline:
  1. stage each /input/<case>/ folder into nnU-Net imagesTs naming
  2. predict with the RC-oversampled fold 0 and the vanilla ResEnc-L fold 3
  3. softmax-average the two (nnUNetv2_ensemble)
  4. remove resection-cavity connected components below MIN_RC_VOLUME_MM3
  5. write ONE flat <case>.nii.gz per case to /output

Validated on the BraTS 2026 validation leaderboard:
  rc+f3            (sub 9771889): ET 0.650 / RC 0.461 / TC 0.681 / WT 0.649
  rc+f3 + CC-filter(sub 9772323): ET 0.650 / RC 0.518 / TC 0.681 / WT 0.649
  The filter cuts RC false positives 128 -> 11 at a cost of 2 true positives;
  ET/TC/WT are untouched by construction (only label 4 is modified).

Challenge spec compliance:
  - /input is read-only; we never write there.
  - /output is FLAT: no subfolders, filename = <case>.nii.gz
  - No network access: model weights baked into the image (see Dockerfile).
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

INPUT_DIR  = Path("/input")
OUTPUT_DIR = Path("/output")

os.environ.setdefault("nnUNet_raw", "/opt/nnunet/nnUNet_raw")
os.environ.setdefault("nnUNet_preprocessed", "/opt/nnunet/nnUNet_preprocessed")
os.environ.setdefault("nnUNet_results", "/opt/nnunet/nnUNet_results")
os.environ.setdefault("nnUNet_compile", "F")   # avoid torch.compile hang

DATASET = "501"
CONFIG = "3d_fullres"
PLANS = "nnUNetResEncUNetLPlans"
CHECKPOINT = "checkpoint_best.pth"

# Validated 2-model ensemble: (trainer_class_name, fold).
MODELS = [
    ("nnUNetTrainerRCOversample", "0"),   # RC-oversampled ResEnc-L, fold 0
    ("nnUNetTrainer_500epochs",   "3"),   # vanilla ResEnc-L, fold 3
]

# Resection-cavity false-positive suppression.
RC_LABEL = 4
MIN_RC_VOLUME_MM3 = 100.0

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
    """Copy a case's 4 modalities into nnU-Net imagesTs naming."""
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


def postprocess_rc(src_dir: Path, dst_dir: Path, min_volume_mm3: float = MIN_RC_VOLUME_MM3):
    """
    Remove resection-cavity (label 4) connected components smaller than
    min_volume_mm3. Only label 4 is touched; all other labels pass through
    unchanged. Components are 3D-connected with the default structuring
    element (6-connectivity).
    """
    import numpy as np
    import SimpleITK as sitk
    from scipy.ndimage import label as cc_label

    dst_dir.mkdir(parents=True, exist_ok=True)
    n_removed = 0
    n_files = 0
    for f in sorted(src_dir.glob("*.nii.gz")):
        img = sitk.ReadImage(str(f))
        arr = sitk.GetArrayFromImage(img)
        vox_mm3 = float(np.prod(img.GetSpacing()))
        rc = (arr == RC_LABEL)
        if rc.any():
            lab, n = cc_label(rc)
            for k in range(1, n + 1):
                mask = (lab == k)
                if mask.sum() * vox_mm3 < min_volume_mm3:
                    arr[mask] = 0
                    n_removed += 1
        out = sitk.GetImageFromArray(arr)
        out.CopyInformation(img)
        sitk.WriteImage(out, str(dst_dir / f.name))
        n_files += 1
    print(f"postprocessing: removed {n_removed} RC components < {min_volume_mm3:.0f} mm^3 "
          f"across {n_files} cases", flush=True)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cases = find_case_folders(INPUT_DIR)
    print(f"found {len(cases)} case folders in {INPUT_DIR}", flush=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        staging = tmp / "imagesTs"
        staging.mkdir()

        for case_dir in cases:
            stage_case(case_dir, staging)

        # per-model prediction (softmax saved for ensembling)
        pred_dirs = []
        for trainer, fold in MODELS:
            out_f = tmp / f"{trainer}_f{fold}_pred"
            out_f.mkdir()
            run([
                "nnUNetv2_predict",
                "-i", str(staging),
                "-o", str(out_f),
                "-d", DATASET, "-c", CONFIG,
                "-p", PLANS, "-tr", trainer,
                "-f", fold, "-chk", CHECKPOINT,
                "--save_probabilities",
                "-npp", "2", "-nps", "2",
            ])
            pred_dirs.append(str(out_f))

        # softmax-average the models
        ens_out = tmp / "ensemble_pred"
        ens_out.mkdir()
        if len(pred_dirs) > 1:
            run(["nnUNetv2_ensemble", "-i", *pred_dirs,
                 "-o", str(ens_out), "-np", "2"])
            merged = ens_out
        else:
            merged = Path(pred_dirs[0])

        # RC false-positive suppression
        final_dir = tmp / "final_pred"
        postprocess_rc(merged, final_dir)

        # write predictions FLAT to /output
        n = 0
        for nii in sorted(final_dir.glob("*.nii.gz")):
            shutil.copy(str(nii), str(OUTPUT_DIR / nii.name))
            n += 1
        print(f"wrote {n} predictions to {OUTPUT_DIR}", flush=True)

    print("done.", flush=True)


if __name__ == "__main__":
    main()
