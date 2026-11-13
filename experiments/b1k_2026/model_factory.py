# Copyright (c) 2026 Xiaomi Corporation

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "configs"
DEFAULT_CONFIG_NAME = "train"
DEFAULT_HYDRA_TASK = "b1k_2026_fastwam_flex_3cam_direct384x256_mb32_gc_on_128gpu"


def _dtype(value: str) -> torch.dtype:
    normalized = value.strip().lower()
    aliases = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if normalized not in aliases:
        raise ValueError(
            f"FASTWAM_B1K_DTYPE must be one of {sorted(aliases)}, got {value!r}."
        )
    return aliases[normalized]


def _device() -> str:
    configured = os.environ.get("FASTWAM_B1K_DEVICE")
    if configured:
        return configured
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _extra_overrides() -> list[str]:
    raw = os.environ.get("FASTWAM_B1K_CONFIG_OVERRIDES", "").strip()
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "FASTWAM_B1K_CONFIG_OVERRIDES must be a JSON list of Hydra override strings."
        ) from exc
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(
            "FASTWAM_B1K_CONFIG_OVERRIDES must be a JSON list of Hydra override strings."
        )
    return value


def compose_b1k_config() -> DictConfig:
    """Compose the training-compatible B1K model config for short smoke inference."""

    config_dir = (
        Path(os.environ.get("FASTWAM_B1K_CONFIG_DIR", str(DEFAULT_CONFIG_DIR)))
        .expanduser()
        .resolve()
    )
    config_name = os.environ.get("FASTWAM_B1K_CONFIG_NAME", DEFAULT_CONFIG_NAME)
    hydra_task = os.environ.get("FASTWAM_B1K_HYDRA_TASK", DEFAULT_HYDRA_TASK)
    if not config_dir.is_dir():
        raise FileNotFoundError(
            f"FASTWAM_B1K_CONFIG_DIR is not a directory: {config_dir}"
        )

    overrides = [f"task={hydra_task}", *_extra_overrides()]
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = compose(config_name=config_name, overrides=overrides)
    if cfg.get("model") is None:
        raise ValueError("Composed B1K config does not define cfg.model.")

    # Smoke inference uses the exact precomputed T5 cache consumed in training.
    # Loading a second text encoder here would add memory and create two text paths.
    with open_dict(cfg.model):
        cfg.model.load_text_encoder = False
        if os.environ.get("FASTWAM_B1K_MODEL_ID"):
            cfg.model.model_id = os.environ["FASTWAM_B1K_MODEL_ID"]
        if os.environ.get("FASTWAM_B1K_TOKENIZER_MODEL_ID"):
            cfg.model.tokenizer_model_id = os.environ["FASTWAM_B1K_TOKENIZER_MODEL_ID"]
        if os.environ.get("FASTWAM_B1K_DINO_PATH"):
            cfg.model.dino_pretrained_path = os.environ["FASTWAM_B1K_DINO_PATH"]
        if os.environ.get("FASTWAM_B1K_ACTION_DIT_PATH"):
            cfg.model.action_dit_pretrained_path = os.environ[
                "FASTWAM_B1K_ACTION_DIT_PATH"
            ]
    return cfg


def model_factory() -> Any:
    """Plugin entry point used by ``run_short_smoke.py --model-factory``.

    The smoke runner owns strict checkpoint loading after this function returns.
    """

    cfg = compose_b1k_config()
    model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
    if bool(model_cfg.get("load_text_encoder", True)):
        raise RuntimeError(
            "B1K short smoke must instantiate with load_text_encoder=false."
        )
    model = instantiate(
        model_cfg,
        model_dtype=_dtype(os.environ.get("FASTWAM_B1K_DTYPE", "bf16")),
        device=_device(),
    )
    model._fastwam_b1k_model_target = str(model_cfg.get("_target_", ""))
    model._fastwam_b1k_requires_depth = _model_target_requires_depth(
        model._fastwam_b1k_model_target
    )
    # Evaluation config commonly has val=null.  The training processor is the
    # authoritative temporal contract in that case (33 observations -> 32
    # action tokens, and 9 image frames for Flex).
    data_split = cfg.data.val if cfg.data.get("val") is not None else cfg.data.train
    processor = data_split.get("processor")
    if processor is not None:
        action_horizon = processor.get("action_horizon")
        if action_horizon is None:
            num_obs_steps = processor.get("num_obs_steps")
            action_horizon = (
                int(num_obs_steps) - 1 if num_obs_steps is not None else None
            )
        if action_horizon is None:
            num_frames = data_split.get("num_frames")
            action_horizon = int(num_frames) - 1 if num_frames is not None else None
        num_image_steps = processor.get("num_image_steps")
        if action_horizon is not None:
            model._fastwam_b1k_action_horizon = int(action_horizon)
        if num_image_steps is not None:
            model._fastwam_b1k_num_video_frames = int(num_image_steps)
    if (
        getattr(model, "text_encoder", None) is not None
        or getattr(model, "tokenizer", None) is not None
    ):
        raise RuntimeError(
            "B1K short-smoke model unexpectedly loaded a text encoder/tokenizer; "
            "cache-only text is required."
        )
    return model


def _model_target_requires_depth(target: str) -> bool:
    return target == "fastwam.runtime.create_fastwam_flex"


def checkpoint_probe(model: Any, *, seed: int) -> dict[str, torch.Tensor]:
    """Small deterministic probe covering proprio, video, and action weights.

    A full inference rollout belongs to the subsequent closed-loop smoke.  The
    round-trip check needs a cheap, exact serialization probe that can run while
    two 6B model instances briefly coexist on one GPU.
    """

    proprio_encoder = getattr(model, "proprio_encoder", None)
    if proprio_encoder is None:
        raise RuntimeError("B1K checkpoint probe requires the 23D proprio encoder.")
    parameter = next(proprio_encoder.parameters())
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    proprio = torch.randn((1, 23), generator=generator, dtype=torch.float32).to(
        device=parameter.device, dtype=parameter.dtype
    )
    proprio_output = proprio_encoder(proprio).detach().to(device="cpu", dtype=torch.float32)

    def parameter_sample(module: Any, label: str) -> torch.Tensor:
        named = list(module.named_parameters())
        if not named:
            raise RuntimeError(f"B1K checkpoint probe found no {label} parameters.")
        tensors = []
        for _, value in (named[0], named[len(named) // 2], named[-1]):
            flat = value.detach().reshape(-1)
            if flat.numel() == 0:
                raise RuntimeError(f"B1K checkpoint probe found an empty {label} parameter.")
            indices = torch.linspace(
                0, flat.numel() - 1, steps=min(32, flat.numel()), device=flat.device
            ).long()
            tensors.append(flat.index_select(0, indices).to(device="cpu", dtype=torch.float32))
        return torch.cat(tensors)

    result = {
        "proprio_output": proprio_output,
        "video_parameter_sample": parameter_sample(model.video_expert, "video expert"),
        "action_parameter_sample": parameter_sample(model.action_expert, "action expert"),
    }
    for name in ("dino_embedder", "dino_proj_out", "pt_patch_embedding", "pt_head"):
        module = getattr(model, name, None)
        if module is not None:
            result[f"{name}_parameter_sample"] = parameter_sample(module, name)
    return result
