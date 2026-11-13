# Policy runtime image for the BEHAVIOR-1K challenge submission.
#
# NOTE: reconstructed from the environment of the submitted image
# (robot-policy-runtime:20261013-r01: Python 3.11, torch 2.6.0+cu124). It has not
# been rebuilt bit-for-bit; verify it on your machine before relying on it.
#
# Build from the repository root:
#   docker build -f challenge_docker/Dockerfile -t fastwam-policy-runtime .
#
# The image contains code only. Provide the third-party base assets (Wan2.2-TI2V-5B,
# Wan2.2 VAE, DINOv3 ViT-B/16, ActionDiT init, T5 text embeddings) at
# /opt/fastwam/runtime-assets via a read-only bind mount, and one released model
# directory (checkpoint.pt, checkpoint_contract.json, norm_stats.json,
# model_config.json) at /runtime/model.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/opt/fastwam:/opt/fastwam/src \
    FASTWAM_B1K_INFERENCE_STEPS=20 \
    FASTWAM_EXECUTE_HORIZON=32 \
    FASTWAM_TEMPORAL_STRATEGY=speed_2x \
    FASTWAM_RESAMPLING=zero_order_hold \
    FASTWAM_RUNTIME_ASSETS=/opt/fastwam/runtime-assets \
    FASTWAM_B1K_MODEL_ID=/opt/fastwam/runtime-assets/models/Wan-AI/Wan2.2-TI2V-5B \
    FASTWAM_B1K_ACTION_DIT_PATH=/opt/fastwam/runtime-assets/models/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
    FASTWAM_B1K_DINO_PATH=/opt/fastwam/runtime-assets/models/dinov3/vit_base_patch16_dinov3.lvd1689m/model.safetensors \
    DIFFSYNTH_MODEL_BASE_PATH=/opt/fastwam/runtime-assets/models \
    FASTWAM_B1K_TEXT_CACHE_DIR=/opt/fastwam/runtime-assets/text_embeds \
    FASTWAM_MODEL_DIR=/runtime/model \
    FASTWAM_MODEL_CONFIG=/runtime/model/model_config.json \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    DIFFSYNTH_SKIP_DOWNLOAD=true

WORKDIR /opt/fastwam

COPY challenge_docker/requirements.runtime.txt /tmp/requirements.runtime.txt
RUN python -m pip install -r /tmp/requirements.runtime.txt

COPY pyproject.toml /opt/fastwam/
COPY src /opt/fastwam/src
COPY configs /opt/fastwam/configs
COPY experiments /opt/fastwam/experiments
COPY scripts /opt/fastwam/scripts
COPY challenge_docker /opt/fastwam/challenge_docker
RUN chmod +x /opt/fastwam/challenge_docker/entrypoint.sh

ENTRYPOINT ["/opt/fastwam/challenge_docker/entrypoint.sh"]
