# BraTS 2026 Task 1 — inference container
#
# Challenge environment: NVIDIA A10G (24 GB), 16 vCPU, 48 GiB RAM,
# max CUDA 13.0, driver 580.xx, ZERO network access at runtime.
# All dependencies + model weights MUST be baked into this image.
#
# Base image: a CUDA runtime compatible with the challenge driver.
# CUDA 12.x runtime is compatible with driver 580.xx (max CUDA 13.0).
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV nnUNet_raw=/opt/nnunet/nnUNet_raw \
    nnUNet_preprocessed=/opt/nnunet/nnUNet_preprocessed \
    nnUNet_results=/opt/nnunet/nnUNet_results \
    nnUNet_compile=F

# --- system deps (installed at build time; no runtime network) ---
RUN apt-get update && apt-get install -y --no-install-recommends \
        git && \
    rm -rf /var/lib/apt/lists/*

# --- python deps ---
# Pin nnunetv2 to the training version for reproducibility.
RUN pip install --no-cache-dir nnunetv2==2.7.0

# --- bake in model weights ---
# Copy the trained nnU-Net results tree (checkpoints + plans + dataset.json).
# Only the folds used at inference need to be present.
#   nnUNet_results/Dataset501_BraTSMET2026/
#     nnUNetTrainer_500epochs__nnUNetResEncUNetLPlans__3d_fullres/
#       fold_0/checkpoint_best.pth
#       fold_1/checkpoint_best.pth
#       fold_3/checkpoint_best.pth
#       dataset.json  plans.json  (nnU-Net needs these alongside folds)
COPY nnUNet_results /opt/nnunet/nnUNet_results

# nnU-Net checks these dirs exist even for inference; create empty ones.
RUN mkdir -p /opt/nnunet/nnUNet_raw /opt/nnunet/nnUNet_preprocessed

# --- inference entrypoint ---
COPY predict.py /opt/predict.py

# /input (ro) and /output (rw) are mounted by the challenge harness.
ENTRYPOINT ["python", "/opt/predict.py"]
