# Copyright (c) 2026 Xiaomi Corporation

from __future__ import annotations

import hashlib
import inspect
import os
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F

from fastwam.datasets.b1k.text import format_b1k_prompt
from fastwam.datasets.b1k.pointmap import (
    MAX_DEPTH_M,
    MIN_DEPTH_M,
    make_b1k_pointmap_mosaic,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEXT_CACHE_DIR = PROJECT_ROOT / "artifacts" / "b1k_2026" / "text_embeds"
DEFAULT_CONTEXT_LEN = 128
DEFAULT_NUM_VIDEO_FRAMES = 9
DEFAULT_ACTION_HORIZON = 32


def load_cached_text_context(task: str) -> tuple[torch.Tensor, torch.Tensor, Path]:
    """Load the same prompt-keyed T5 payload used by the B1K training dataset."""

    prompt = format_b1k_prompt(task)
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    context_len = int(os.environ.get("FASTWAM_B1K_CONTEXT_LEN", DEFAULT_CONTEXT_LEN))
    cache_dir = (
        Path(os.environ.get("FASTWAM_B1K_TEXT_CACHE_DIR", str(DEFAULT_TEXT_CACHE_DIR)))
        .expanduser()
        .resolve()
    )
    cache_path = cache_dir / f"{digest}.t5_len{context_len}.wan22ti2v5b.pt"
    if not cache_path.is_file():
        raise FileNotFoundError(
            f"Missing B1K text cache for prompt {prompt!r}: {cache_path}. "
            "Run scripts/precompute_text_embeds.py first."
        )
    payload = torch.load(cache_path, map_location="cpu")
    if (
        not isinstance(payload, Mapping)
        or "context" not in payload
        or "mask" not in payload
    ):
        raise ValueError(f"Invalid B1K text cache payload in {cache_path}.")
    context = torch.as_tensor(payload["context"])
    context_mask = torch.as_tensor(payload["mask"], dtype=torch.bool)
    if context.ndim != 2 or context.shape[0] != context_len:
        raise ValueError(
            f"Cached context must have shape [{context_len}, D], got {tuple(context.shape)} "
            f"in {cache_path}."
        )
    if context_mask.shape != (context_len,):
        raise ValueError(
            f"Cached mask must have shape ({context_len},), got {tuple(context_mask.shape)} "
            f"in {cache_path}."
        )
    if not bool(context_mask.any().item()):
        raise ValueError(f"Cached mask has no valid tokens in {cache_path}.")
    context = context.clone()
    context[~context_mask] = 0
    return context, context_mask, cache_path


def _image_tensor(value: Any, name: str) -> torch.Tensor:
    image = torch.as_tensor(np.asarray(value))
    if image.ndim != 3:
        raise ValueError(f"{name} image must be HWC or CHW, got {tuple(image.shape)}.")
    if image.shape[-1] in (3, 4):
        image = image[..., :3].permute(2, 0, 1)
    elif image.shape[0] in (3, 4):
        image = image[:3]
    else:
        raise ValueError(
            f"{name} image must have three RGB channels, got {tuple(image.shape)}."
        )
    image = image.to(dtype=torch.float32)
    if float(image.max().item()) > 1.0 + 1e-6:
        image = image / 255.0
    return image.clamp(0.0, 1.0)


def build_b1k_mosaic(images: Mapping[str, Any]) -> torch.Tensor:
    required = ("head", "left_wrist", "right_wrist")
    missing = [name for name in required if name not in images]
    if missing:
        raise KeyError(f"B1K policy observation is missing cameras {missing}.")

    def resize(name: str, height: int, width: int) -> torch.Tensor:
        image = _image_tensor(images[name], name).unsqueeze(0)
        return F.interpolate(
            image,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )[0]

    head = resize("head", 256, 256)
    wrists = torch.cat(
        [resize("left_wrist", 128, 128), resize("right_wrist", 128, 128)], dim=-1
    )
    mosaic = torch.cat([head, wrists], dim=-2)
    if mosaic.shape != (3, 384, 256):
        raise RuntimeError(
            f"B1K mosaic must be (3,384,256), got {tuple(mosaic.shape)}."
        )
    return mosaic * 2.0 - 1.0


def build_b1k_pointmap(depths: Mapping[str, Any]) -> torch.Tensor:
    """Build the current-frame pointmap with the training projection/layout."""

    required = ("head", "left_wrist", "right_wrist")
    missing = [name for name in required if name not in depths]
    if missing:
        raise KeyError(f"B1K policy observation is missing depth cameras {missing}.")
    depth_tensors: dict[str, torch.Tensor] = {}
    for name in required:
        depth = torch.as_tensor(np.asarray(depths[name]), dtype=torch.float32)
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        if depth.ndim == 3 and depth.shape[0] == 1:
            depth = depth[0]
        if depth.ndim != 2:
            raise ValueError(
                f"{name} depth must be HxW or HxWx1, got {tuple(depth.shape)}."
            )
        # Training/inference far-field clamp parity.
        #
        # Training pointmaps were projected from the official 12-bit log
        # quantized depth videos, whose encoder clamps to
        # [MIN_DEPTH_M, MAX_DEPTH_M]. The simulator hands us raw metric depth
        # with no clamp (OmniGibson's default far clip is 1e7 m) and +inf where
        # a ray hits nothing. depth_to_pointmap treats anything outside
        # (MIN_DEPTH_M, 10.1) as invalid and zeroes it *before* the [0,3] m
        # clamp, so a genuine far surface at 15 m encoded as (0, 0, -1) at
        # inference while training saw (x, y, +1) -- opposite ends of the z
        # channel, on 7.66% of head pixels on average and up to 31% outdoors.
        #
        # Reproducing the encoder's clamp here restores parity. Relaxing the
        # validity upper bound instead would not: x = (u - cx) / f * z keeps
        # growing with z, so near-axis pixels would still disagree.
        depth = torch.nan_to_num(
            depth, nan=MAX_DEPTH_M, posinf=MAX_DEPTH_M, neginf=MIN_DEPTH_M
        ).clamp(MIN_DEPTH_M, MAX_DEPTH_M)
        depth_tensors[name] = depth.unsqueeze(0)
    pointmap = make_b1k_pointmap_mosaic(depth_tensors)
    if pointmap.shape != (3, 1, 384, 256):
        raise RuntimeError(
            "Current B1K pointmap must be (3,1,384,256), got "
            f"{tuple(pointmap.shape)}."
        )
    return pointmap[:, 0]


def policy(
    model: Any,
    *,
    observation: Mapping[str, Any],
    normalized_proprio: np.ndarray,
    step: int,
    seed: int,
) -> torch.Tensor:
    """FastWAM Flex inference plugin used by the B1K evaluation runner."""

    del step
    preprocess_start = time.perf_counter()
    if not isinstance(observation, Mapping):
        raise TypeError("B1K policy observation must be a mapping.")
    images = observation.get("images")
    if not isinstance(images, Mapping):
        raise KeyError("B1K policy observation must contain an 'images' mapping.")
    task = observation.get("task")
    if not isinstance(task, str) or not task:
        raise ValueError("B1K policy observation must contain a non-empty task string.")

    if getattr(model, "text_encoder", None) is not None:
        raise RuntimeError(
            "B1K smoke policy is cache-only and requires model.text_encoder to be None."
        )
    context, context_mask, _ = load_cached_text_context(task)

    first_parameter = next(model.parameters(), None)
    default_device = (
        torch.device("cpu") if first_parameter is None else first_parameter.device
    )
    default_dtype = torch.float32 if first_parameter is None else first_parameter.dtype
    model_device = torch.device(getattr(model, "device", default_device))
    model_dtype = getattr(model, "torch_dtype", default_dtype)
    input_image = (
        build_b1k_mosaic(images).unsqueeze(0).to(device=model_device, dtype=model_dtype)
    )
    proprio = torch.as_tensor(
        normalized_proprio, dtype=model_dtype, device=model_device
    )
    if proprio.shape != (23,):
        raise ValueError(
            f"normalized_proprio must have shape (23,), got {tuple(proprio.shape)}."
        )

    infer_action_parameters = inspect.signature(model.infer_action).parameters
    # FastWAM Flex intentionally exposes a forwarding ``(*args, **kwargs)``
    # wrapper, while its action-only implementation still requires
    # ``num_video_frames``.  Do not infer support from VAR_KEYWORD alone:
    # legacy action-only models also commonly use ``**kwargs``.
    is_flex_model = bool(
        getattr(model, "_fastwam_b1k_requires_depth", False)
        or getattr(model, "_fastwam_b1k_model_target", "")
        == "fastwam.runtime.create_fastwam_flex"
    )
    requires_pointmap = bool(
        getattr(model, "_fastwam_b1k_requires_depth", False)
        or "input_pointmap" in infer_action_parameters
    )
    action_horizon = int(
        os.environ.get(
            "FASTWAM_B1K_ACTION_HORIZON",
            getattr(model, "_fastwam_b1k_action_horizon", DEFAULT_ACTION_HORIZON),
        )
    )
    if action_horizon <= 0:
        raise ValueError(f"action horizon must be positive, got {action_horizon}")
    infer_action_kwargs = dict(
        prompt=None,
        input_image=input_image,
        action_horizon=action_horizon,
        proprio=proprio,
        context=context.to(device=model_device, dtype=model_dtype),
        context_mask=context_mask.to(device=model_device, dtype=torch.bool),
        negative_prompt=os.environ.get("FASTWAM_B1K_NEGATIVE_PROMPT", ""),
        text_cfg_scale=float(os.environ.get("FASTWAM_B1K_TEXT_CFG_SCALE", "1.0")),
        num_inference_steps=int(os.environ.get("FASTWAM_B1K_INFERENCE_STEPS", "10")),
        sigma_shift=(
            None
            if os.environ.get("FASTWAM_B1K_SIGMA_SHIFT") is None
            else float(os.environ["FASTWAM_B1K_SIGMA_SHIFT"])
        ),
        seed=seed,
        rand_device=os.environ.get("FASTWAM_B1K_RAND_DEVICE", "cpu"),
        tiled=_environment_bool("FASTWAM_B1K_TILED", False),
    )
    if requires_pointmap:
        depths = observation.get("depths")
        if not isinstance(depths, Mapping):
            raise KeyError(
                "B1K Stage-1 policy observation must contain a 'depths' mapping."
            )
        infer_action_kwargs["input_pointmap"] = (
            build_b1k_pointmap(depths)
            .unsqueeze(0)
            .to(device=model_device, dtype=model_dtype)
        )
    if "num_video_frames" in infer_action_parameters or is_flex_model:
        infer_action_kwargs["num_video_frames"] = int(
            os.environ.get(
                "FASTWAM_B1K_NUM_VIDEO_FRAMES",
                getattr(
                    model,
                    "_fastwam_b1k_num_video_frames",
                    DEFAULT_NUM_VIDEO_FRAMES,
                ),
            )
        )
    preprocess_ms = (time.perf_counter() - preprocess_start) * 1000
    model_start = time.perf_counter()
    result = model.infer_action(**infer_action_kwargs)
    action = result["action"] if isinstance(result, Mapping) else result
    action = torch.as_tensor(action, dtype=torch.float32, device="cpu")
    if action.shape != (action_horizon, 23):
        raise ValueError(
            f"FastWAM infer_action must return ({action_horizon},23), got {tuple(action.shape)}."
        )
    model._fastwam_last_timing = dict(preprocess_ms=preprocess_ms,
                                     model_infer_ms=(time.perf_counter() - model_start) * 1000)
    return action


def _environment_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value, got {value!r}.")
