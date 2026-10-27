# Pixal3D (TencentARC, MIT) on a 16 GB Blackwell GPU (RTX 5070 Ti, sm_120).
# torch 2.11 + CUDA 13.0 wheels, and the authors' prebuilt extension wheels for that pair
# (built for the RTX PRO 6000, also sm_120): from the HF demo's requirements_th211_cu130.txt.
# flash-attn is left out: run with ATTN_BACKEND=sdpa.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-venv python3-dev build-essential git ca-certificates libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

RUN pip install --upgrade pip \
 && pip install torch==2.11.0 torchvision==0.26.0 triton==3.6.0 --index-url https://download.pytorch.org/whl/cu130

RUN pip install pillow==12.0.0 imageio==2.37.2 imageio-ffmpeg==0.6.0 tqdm==4.67.1 easydict==1.13 \
      opencv-python-headless==4.12.0.88 trimesh==4.10.1 transformers==4.57.3 zstandard==0.25.0 kornia==0.8.2 \
      timm==1.0.22 diffusers==0.37.1 accelerate==1.13.0 einops==0.8.2 plyfile==1.1.3 huggingface_hub \
      "git+https://github.com/microsoft/MoGe.git@b942f00bdc2a2a23ebb474fbe034d487e6dcceec"

RUN pip install \
      "https://github.com/SHI-Labs/NATTEN/releases/download/v0.21.6/natten-0.21.6%2Btorch2110cu130-cp312-cp312-linux_x86_64.whl" \
      "https://github.com/LDYang694/Storages/releases/download/rtxpro6000/flex_gemm-1.0.0%2Btorch2.11.0.cu130-cp312-cp312-linux_x86_64.whl" \
      "https://github.com/LDYang694/Storages/releases/download/rtxpro6000/nvdiffrast-0.4.0%2Btorch2.11.0.cu130-cp312-cp312-linux_x86_64.whl" \
      "https://github.com/LDYang694/Storages/releases/download/rtxpro6000/nvdiffrec_render-0.0.0%2Btorch2.11.0.cu130-cp312-cp312-linux_x86_64.whl" \
      "https://github.com/LDYang694/Storages/releases/download/rtxpro6000/cumesh-0.0.1%2Btorch2.11.0.cu130-cp312-cp312-linux_x86_64.whl" \
      "https://github.com/LDYang694/Storages/releases/download/rtxpro6000/o_voxel-0.0.1%2Btorch2.11.0.cu130-cp312-cp312-linux_x86_64.whl" \
      "https://github.com/LDYang694/Storages/releases/download/20260430/utils3d-0.0.2-py3-none-any.whl"

ENV ATTN_BACKEND=sdpa HF_HOME=/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
WORKDIR /work
