# Notice

Xiaomi CYBERWAM
Copyright 2026 Xiaomi Corporation

This repository is licensed under the Apache License, Version 2.0; see `LICENSE`. Parts of it are derived from
other projects that are released under their own licenses, as listed below.

## Upstream code

This repository is derived from the open-source FastWAM codebase
(*Fast-WAM: Do World Action Models Need Test-time Future Imagination?*,
https://github.com/yuantianyuan01/FastWAM), released under the MIT License. The original MIT license text and
copyright notice (Copyright (c) 2026 The FastWAM Authors) are preserved in `third_party_licenses/MIT-FastWAM.txt`.
The files we changed carry a note in their header.

Changes in this repository relative to upstream focus on the BEHAVIOR-1K 2026
challenge: the B1K data pipeline and checkpoint contract, the Flex / Joint model
variants, the policy server (`challenge_docker/`) and the evaluation adapters
(`experiments/b1k_2026/`). The LIBERO and RoboTwin experiment code from upstream
is not included here.

## Method reference

The multi-stream (Flex) model follows the design of Flex-π
(*Flex-π: A Multi-Stream World-Action Model with Compute Flexibility*,
https://arxiv.org/abs/2608.10860; code at https://github.com/geyan21/flex-pi, MIT License).
The Flex-specific modules of this repository (`src/fastwam/models/wan22/fastwam_flex.py`,
`fastwam_multimodal_base.py`, `dino_encoder.py`, `helpers/flex_warmstart.py`) are our own
implementation on top of the FastWAM codebase and do not copy code from the Flex-π repository.

## Protocol compatibility

`challenge_docker/protocol.py` implements the websocket wire format of the official BEHAVIOR-1K evaluator
(`omnigibson.eval.utils.network_utils` in https://github.com/StanfordVL/BEHAVIOR-1K). That module is adapted
from openpi (https://github.com/Physical-Intelligence/openpi, Apache-2.0; see
`third_party_licenses/Apache-2.0.txt`), and its numpy encoding originates from msgpack-numpy
(https://github.com/lebedov/msgpack-numpy, BSD-3-Clause). The message layout (field names such as
`__ndarray__`, the metadata handshake and the `/healthz` check) is fixed by that protocol so that the
official evaluator can connect. The code in `protocol.py` is our own implementation of this format.

## Code adapted from Wan2.2 and DiffSynth-Studio

The following files in `src/fastwam/models/wan22/` are adapted from DiffSynth-Studio
(https://github.com/modelscope/DiffSynth-Studio, Apache-2.0), whose Wan model code is based on the official
Wan2.2 code (https://github.com/Wan-Video/Wan2.2, Apache-2.0; see `third_party_licenses/Apache-2.0.txt`):
`wan_video_vae.py`, `wan_video_text_encoder.py`, `wan_video_dit.py` and `helpers/gradient.py`. They came into
this repository through the FastWAM codebase; the original files carry no per-file notices. `wan_video_vae.py`
was changed by us and says so in its header.

## Third-party code included in this repository

The following files keep their original copyright and license headers:

| Files | Origin | License |
|---|---|---|
| `src/fastwam/datasets/lerobot/constants.py`, `src/fastwam/datasets/lerobot/lerobot/` (`lerobot_dataset.py`, `datasets/compute_stats.py`, `datasets/utils.py`, `datasets/video_utils.py`) | LeRobot, Copyright 2024 The HuggingFace Inc. team | Apache-2.0, `third_party_licenses/Apache-2.0.txt` |
| `src/fastwam/utils/misc.py` | Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES | Apache-2.0, `third_party_licenses/Apache-2.0.txt` |
| `src/fastwam/datasets/lerobot/transforms/rotation.py`, `src/fastwam/datasets/lerobot/utils/rotation.py` | adapted from PyTorch3D, Copyright (c) Meta Platforms, Inc. and affiliates | BSD-3-Clause, `third_party_licenses/PyTorch3D-BSD-3-Clause.txt` |

These files came into this repository through the FastWAM codebase.

## Released model weights

The released checkpoint (`model_a`) is released under the Apache License 2.0.

- Its video expert is fine-tuned from Wan2.2-TI2V-5B (Wan-AI,
  https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B), which is released under the
  Apache License 2.0. The fine-tuned weights are a derivative of that model; see
  `third_party_licenses/Apache-2.0.txt`.
- The checkpoint contains the trainable input projection and output head of the DINO stream,
  but **not** the DINOv3 encoder weights. DINOv3 is loaded separately at run time and is
  governed by the DINOv3 License
  (https://ai.meta.com/resources/models-and-libraries/dinov3-license/).
- The checkpoint was trained on the BEHAVIOR-1K 2026 challenge demonstrations; the
  demonstrations themselves are not distributed.

## Third-party assets not distributed with this repository

The following assets must be downloaded separately and are governed by their own licenses:
Wan2.2-TI2V-5B and the Wan2.2 VAE (Wan-AI, Apache-2.0), the Wan2.1 tokenizer and T5 text
encoder (Wan-AI), DINOv3 ViT-B/16 (`vit_base_patch16_dinov3.lvd1689m`, DINOv3 License),
and the BEHAVIOR-1K simulator and its datasets. Review those licenses before redistributing
them or images that contain them.
