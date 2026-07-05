# Containerized inference — BraTS 2026 Task 1

This container runs the ResEnc-L ensemble on the challenge testing data. It
follows the BraTS 2026 submission spec: reads case folders from a read-only
`/input`, writes flat `<case>.nii.gz` predictions to `/output`, and requires no
network access at runtime (weights are baked into the image).

## Contents

- `Dockerfile` — builds the image (installs `nnunetv2==2.7.0`, copies weights).
- `predict.py` — entrypoint: stages each case, predicts per fold, softmax
  ensembles the RC-competent folds, writes flat output.

## Before building

Place the trained nnU-Net results tree next to the `Dockerfile` so the `COPY`
step can bake it in:

```
docker/
├── Dockerfile
├── predict.py
└── nnUNet_results/
    └── Dataset501_BraTSMET2026/
        └── nnUNetTrainer_500epochs__nnUNetResEncUNetLPlans__3d_fullres/
            ├── dataset.json
            ├── plans.json
            ├── fold_0/checkpoint_best.pth
            ├── fold_1/checkpoint_best.pth
            └── fold_3/checkpoint_best.pth
```

> The `dataset.json` and `plans.json` must sit at the trainer level — nnU-Net
> reads them during inference. Copy them from your local `nnUNet_results` tree.

## Build

```bash
cd docker
docker build -t brats-uniandes-met:latest .
```

## Test locally (mimics the challenge harness)

```bash
docker run \
  --rm \
  --network none \
  --gpus=all \
  --volume $PWD/Validation/:/input:ro \
  --volume $PWD/results/:/output:rw \
  --memory=48G --shm-size=16G \
  brats-uniandes-met:latest
```

Then verify the output:

- `results/` contains one `.nii.gz` per input case, **flat** (no subfolders).
- Filenames end with the 5-digit case ID + 3-digit timepoint (e.g.
  `BraTS-MET-12345-000.nii.gz`).
- Each prediction matches the spatial characteristics (dims, spacing, origin,
  orientation) of its input.

## Notes / gotchas

- **`--network none`** replicates the no-internet runtime. If the build
  installed everything correctly, inference must not need the network.
- **Modality mapping:** `predict.py` maps nnU-Net channels `_0000.._0003` to
  `t1n/t1c/t2w/t2f`. Confirm this against your `dataset.json` `channel_names`
  before submitting — a wrong mapping silently produces bad predictions.
- **Folds:** `predict.py` uses folds `0,1,3` (RC-competent). Update the `FOLDS`
  list after retraining on the corrected labels.
- **12-hour limit:** ensemble inference over the full test set must finish
  within 12 h on an A10G. Time a local run on the validation set to estimate.
