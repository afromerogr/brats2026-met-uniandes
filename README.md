# BraTS 2026 Task 1 — Brain Metastases Segmentation (Team: BraTS Uniandes)

Residual-encoder nnU-Net (ResEnc-L) for pre- and post-treatment brain
metastases segmentation, developed for **BraTS 2026 Task 1**.

**Team:** BraTS Uniandes · **Synapse:** `afromerogr`
**Affiliation:** Center for Research and Formation in Artificial Intelligence
(CinfonIA), Universidad de los Andes, Bogotá, Colombia

---

## Overview

We segment the four-label BM scheme — non-enhancing tumor core (NETC),
surrounding FLAIR hyperintensity (SNFH), enhancing tumor (ET), and, for
post-treatment cases, the resection cavity (RC) — with an nnU-Net v2
residual-encoder configuration.

The resection cavity is the binding constraint on this task: it is present in
only 167 of the 1,296 training cases (12.9%), and folds routinely fail to learn
it at all. The method here is built around that problem, and the final
configuration is **not** a plain multi-fold ensemble.

**Final configuration (`rc + f3 + CC`):**

1. **`rc`** — a ResEnc-L variant trained on fold 0 with RC-targeted
   optimization: foreground oversampling raised from 0.33 to **0.90**, and a
   class-weighted cross-entropy with `w = [1, 1, 1, 1, 3]` over
   (background, NETC, SNFH, ET, RC).
2. **`f3`** — a vanilla ResEnc-L baseline fold, softmax-averaged **one-to-one**
   with `rc`. The pairing ratio matters: diluted 1-in-4 into a four-model
   ensemble, the entire RC gain disappears.
3. **`CC`** — a connected-component filter that removes RC (label 4) components
   below **100 mm³**. All other labels are untouched, so ET/TC/WT are
   unchanged by construction.

### Validation results (official leaderboard, N = 179)

| Configuration | ET | TC | WT | RC |
|---|---|---|---|---|
| Baseline ensemble (f0+f1+f3) | 0.655 | 0.679 | 0.647 | 0.394 |
| `rc` alone | 0.650 | 0.673 | 0.636 | 0.389 |
| `rc` + f0+f1+f3 (1:4, diluted) | 0.661 | 0.689 | 0.646 | 0.385 |
| f0 + f3 (control) | 0.660 | 0.692 | 0.660 | 0.366 |
| f1 + f3 (control) | 0.648 | 0.676 | 0.645 | 0.401 |
| `rc` + f3 (1:1) | 0.650 | 0.681 | 0.649 | 0.461 |
| **`rc` + f3 + CC (final)** | **0.650** | **0.681** | **0.649** | **0.518** |

Lesion-wise Dice. The RC arc is **0.394 → 0.461 → 0.518**, a 31% relative gain.

RC instance counts show the mechanism is false-positive suppression at constant
sensitivity — the true-positive count is unchanged across every ensemble built
on `rc`:

| Configuration | RC TP | RC FP | RC F1 | RC DSC |
|---|---|---|---|---|
| `rc` + f0+f1+f3 (1:4) | 20 | 171 | 0.202 | 0.385 |
| `rc` + f3 (1:1) | 20 | 128 | 0.217 | 0.461 |
| **`rc` + f3 + CC** | 18 | **11** | **0.516** | **0.518** |

A short paper describing the method, the three-level analysis of the resection
cavity, and two negative results (RC-stratified cross-validation; an aborted
transformer probe) accompanies this repository (BraTS 2026 proceedings,
Springer LNCS).

---

## Repository structure

```
brats2026-met-uniandes/
├── notebooks/
│   ├── BraTS2026_Training.ipynb        # fold-selectable ResEnc-L training
│   └── BraTS2026_Inference.ipynb       # per-fold prediction + softmax ensembling
├── docker/                             # containerized inference (challenge submission)
│   ├── Dockerfile
│   ├── predict.py                      # entrypoint: stage → 2 models → ensemble → CC filter
│   └── nnUNetTrainerRCOversample.py    # RC-targeted trainer (required at inference time)
├── requirements.txt
├── LICENSE                             # Apache 2.0
└── README.md
```

---

## Reproducing

### Environment

```bash
pip install -r requirements.txt
```

Core dependency: `nnunetv2==2.7.0` (PyTorch backend). Training and inference
were performed on a single NVIDIA A100 GPU.

> **Note on `nnUNet_compile`:** on some GPU/driver combinations, `torch.compile`
> can hang. Set `nnUNet_compile=F` in the environment before training or
> inference if you encounter a stall at startup.

### Data

Obtain the BraTS 2026 Task 1 dataset from the official challenge (Synapse). Set
the standard nnU-Net environment variables:

```bash
export nnUNet_raw=/path/to/nnUNet_raw
export nnUNet_preprocessed=/path/to/nnUNet_preprocessed
export nnUNet_results=/path/to/nnUNet_results
```

### Training

Baseline folds (f0–f3):

```bash
nnUNetv2_train 501 3d_fullres <FOLD> \
  -p nnUNetResEncUNetLPlans \
  -tr nnUNetTrainer_500epochs \
  --npz
```

RC-targeted variant (`rc`, fold 0). Copy
`docker/nnUNetTrainerRCOversample.py` into
`nnunetv2/training/nnUNetTrainer/variants/` inside your nnU-Net installation
first, then:

```bash
nnUNetv2_train 501 3d_fullres 0 \
  -p nnUNetResEncUNetLPlans \
  -tr nnUNetTrainerRCOversample \
  --npz
```

`--npz` saves softmax probabilities, required for ensembling.

### Inference + ensembling + postprocessing

```bash
# RC-targeted model, fold 0
nnUNetv2_predict -i <IMAGES_TS> -o rc_pred \
  -d 501 -c 3d_fullres -p nnUNetResEncUNetLPlans \
  -tr nnUNetTrainerRCOversample -f 0 -chk checkpoint_best.pth \
  --save_probabilities

# vanilla baseline, fold 3
nnUNetv2_predict -i <IMAGES_TS> -o f3_pred \
  -d 501 -c 3d_fullres -p nnUNetResEncUNetLPlans \
  -tr nnUNetTrainer_500epochs -f 3 -chk checkpoint_best.pth \
  --save_probabilities

# one-to-one softmax average
nnUNetv2_ensemble -i rc_pred f3_pred -o ensemble_pred -np 4
```

The connected-component filter is then applied to `ensemble_pred`; see
`postprocess_rc()` in `docker/predict.py` (threshold `MIN_RC_VOLUME_MM3 = 100.0`,
label 4 only, 6-connectivity, volume = voxel count × product of image spacing).

> **Note:** `nnUNetv2_predict` resolves the trainer class from the checkpoint's
> `trainer_name`, so `nnUNetTrainerRCOversample` must be importable in the
> environment at *inference* time, not only at training time. The Dockerfile
> installs it into site-packages and asserts it resolves at build time.

The `notebooks/` versions wrap these steps for a Colab/Drive workflow.

---

## Containerized submission

`docker/` contains the inference container used for the challenge testing phase.
It reads case folders from a read-only `/input` directory, runs the two-model
ensemble followed by the connected-component filter, and writes flat `.nii.gz`
predictions to `/output`, following the BraTS 2026 submission specification.

```bash
docker build -t brats-uniandes-met:rcf3 docker/

docker run --rm --network none --gpus=all \
  --volume /PATH/TO/INPUT:/input:ro \
  --volume /PATH/TO/OUTPUT:/output:rw \
  --memory=48G --shm-size=16G \
  brats-uniandes-met:rcf3
```

`--shm-size` is required: PyTorch's data-loading workers pass tensors through
`/dev/shm`, and the Docker default of 64 MB is not enough. The challenge
evaluation environment passes `--shm-size=16G`.

Runtime is roughly 26 s/case on consumer hardware, well inside the challenge's
12-hour limit.

---

## Citation

If you use this code, please cite the accompanying short paper (BraTS 2026
proceedings) and the BraTS-METS challenge manuscripts. Data used in this work
were obtained through the BraTS 2026 challenge (Synapse ID `syn74274097`).

---

## License

Released under the Apache License 2.0. See [`LICENSE`](LICENSE).
