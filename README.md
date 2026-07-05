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
residual-encoder configuration. Because the challenge aggregates ranks across
many region-level metrics, the method prioritizes **cross-region consistency**
and **small-lesion sensitivity** over peak performance on any single region.

Key design points:

- **Architecture:** nnU-Net ResEnc-L (6-stage residual encoder, patch
  160×224×192, batch 2, deep supervision).
- **Training:** 500 epochs/fold, SGD + Nesterov, Dice + cross-entropy loss,
  default nnU-Net augmentation.
- **Ensembling:** softmax-averaged multi-fold ensemble, with fold selection
  to exclude folds that fail to learn the rare RC region.

A short paper describing the method and a fold-composition analysis of the
resection cavity accompanies this repository (BraTS 2026 proceedings, Springer
LNCS).

---

## Repository structure

```
brats2026-met-uniandes/
├── notebooks/
│   ├── BraTS2026_Training.ipynb    # fold-selectable ResEnc-L training
│   └── BraTS2026_Inference.ipynb   # per-fold prediction + softmax ensembling
├── docker/                         # containerized inference (challenge submission)
│   ├── Dockerfile
│   └── predict.py
├── requirements.txt
├── LICENSE                         # Apache 2.0
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

> **Note on `nnUNet_compile`:** on some GPU/driver combinations,
> `torch.compile` can hang. Set `nnUNet_compile=F` in the environment before
> training or inference if you encounter a stall at startup.

### Data

Obtain the BraTS 2026 Task 1 dataset from the official challenge (Synapse). Set
the standard nnU-Net environment variables:

```bash
export nnUNet_raw=/path/to/nnUNet_raw
export nnUNet_preprocessed=/path/to/nnUNet_preprocessed
export nnUNet_results=/path/to/nnUNet_results
```

### Training (per fold)

```bash
nnUNetv2_train 501 3d_fullres <FOLD> \
  -p nnUNetResEncUNetLPlans \
  -tr nnUNetTrainer_500epochs \
  --npz
```

`--npz` saves softmax probabilities, required for ensembling.

### Inference + ensembling

```bash
# per-fold prediction
nnUNetv2_predict -i <IMAGES_TS> -o fold<N>_pred \
  -d 501 -c 3d_fullres -p nnUNetResEncUNetLPlans \
  -tr nnUNetTrainer_500epochs -f <N> -chk checkpoint_best.pth \
  --save_probabilities

# softmax-averaged ensemble of selected (RC-competent) folds
nnUNetv2_ensemble -i fold0_pred fold1_pred fold3_pred -o ensemble_pred -np 4
```

The `notebooks/` versions wrap these steps for a Colab/Drive workflow.

---

## Containerized submission

`docker/` contains the inference container used for the challenge testing phase.
It reads cases from a read-only `/input` directory and writes flat `.nii.gz`
predictions to `/output`, following the BraTS 2026 submission specification. See
`docker/README` for build and local-test instructions.

---

## Citation

If you use this code, please cite the accompanying short paper (BraTS 2026
proceedings) and the BraTS-METS challenge manuscripts. Data used in this work
were obtained through the BraTS 2026 challenge (Synapse ID `syn74274097`).

---

## License

Released under the Apache License 2.0. See [`LICENSE`](LICENSE).
