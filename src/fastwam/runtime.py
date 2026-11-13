# Copyright (c) 2026 Xiaomi Corporation
# Portions derived from FastWAM (MIT License, https://github.com/yuantianyuan01/FastWAM); see NOTICE.md.

import hashlib
import logging
import os
import inspect
import re
import tempfile
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from PIL import Image
import numpy as np
from einops import repeat
from omegaconf import OmegaConf

from .utils.logging_config import get_logger, setup_logging
from .utils.pytorch_utils import (
    attach_random_module_initialization_fingerprint,
    set_global_seed,
)
from .utils.video_io import save_mp4
from .utils import misc

logger = get_logger(__name__)
Wan22Trainer = None


def _normalize_mixed_precision(mixed_precision: str) -> str:
    if not isinstance(mixed_precision, str):
        raise ValueError(f"`mixed_precision` must be str, got {type(mixed_precision)}")
    key = mixed_precision.strip().lower()
    if key not in {"no", "fp16", "bf16"}:
        raise ValueError(
            f"Unsupported mixed_precision: {mixed_precision}. "
            "Expected one of: ['no', 'fp16', 'bf16']."
        )
    return key


def _mixed_precision_to_model_dtype(mixed_precision: str) -> torch.dtype:
    precision = _normalize_mixed_precision(mixed_precision)
    if precision == "no":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16


def create_wan22_model(
    model_id: str,
    tokenizer_model_id: str,
    dit_config,
    tokenizer_max_len: int = 512,
    train_shift: float = 5.0,
    infer_shift: float = 5.0,
    num_train_timesteps: int = 1000,
    redirect_common_files: bool = True,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    from .models.wan22.wan22 import Wan22Core

    if isinstance(dit_config, DictConfig):
        dit_config = OmegaConf.to_container(dit_config, resolve=True)
    if not isinstance(dit_config, dict):
        raise ValueError(f"`dit_config` must resolve to a dict, got {type(dit_config)}")

    return Wan22Core.from_wan22_pretrained(
        device=device,
        torch_dtype=model_dtype,
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_max_len=int(tokenizer_max_len),
        redirect_common_files=bool(redirect_common_files),
        dit_config=dit_config,
        train_shift=float(train_shift),
        infer_shift=float(infer_shift),
        num_train_timesteps=int(num_train_timesteps),
    )


def create_fastwam(
    model_id: str,
    tokenizer_model_id: str,
    video_dit_config,
    tokenizer_max_len: int = 512,
    load_text_encoder: bool = True,
    proprio_dim: int | None = None,
    action_dit_config=None,
    action_dit_pretrained_path: str | None = None,
    skip_dit_load_from_pretrain: bool = False,
    video_scheduler=None,
    action_scheduler=None,
    loss=None,
    mot_checkpoint_mixed_attn: bool = True,
    redirect_common_files: bool = True,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    from .models.wan22.fastwam import FastWAM

    if isinstance(video_dit_config, DictConfig):
        video_dit_config = OmegaConf.to_container(video_dit_config, resolve=True)
    if not isinstance(video_dit_config, dict):
        raise ValueError(f"`video_dit_config` must resolve to a dict, got {type(video_dit_config)}")

    if isinstance(action_dit_config, DictConfig):
        action_dit_config = OmegaConf.to_container(action_dit_config, resolve=True)
    if action_dit_config is None:
        action_dit_config = {}
    if not isinstance(action_dit_config, dict):
        raise ValueError(f"`action_dit_config` must resolve to a dict, got {type(action_dit_config)}")

    if isinstance(video_scheduler, DictConfig):
        video_scheduler = OmegaConf.to_container(video_scheduler, resolve=True)
    if video_scheduler is None:
        video_scheduler = {}
    if not isinstance(video_scheduler, dict):
        raise ValueError(f"`video_scheduler` must be dict-like, got {type(video_scheduler)}")

    if isinstance(action_scheduler, DictConfig):
        action_scheduler = OmegaConf.to_container(action_scheduler, resolve=True)
    if action_scheduler is None:
        raise ValueError("`action_scheduler` is required for FastWAM.")
    if not isinstance(action_scheduler, dict):
        raise ValueError(f"`action_scheduler` must be dict-like, got {type(action_scheduler)}")
    required_action_scheduler_keys = {"train_shift", "infer_shift", "num_train_timesteps"}
    missing_keys = required_action_scheduler_keys - set(action_scheduler.keys())
    if missing_keys:
        raise ValueError(
            f"`action_scheduler` missing required keys: {sorted(missing_keys)}. "
            "Expected keys: train_shift, infer_shift, num_train_timesteps."
        )

    if isinstance(loss, DictConfig):
        loss = OmegaConf.to_container(loss, resolve=True)
    if loss is None:
        loss = {}
    if not isinstance(loss, dict):
        raise ValueError(f"`loss` must be dict-like, got {type(loss)}")

    return FastWAM.from_wan22_pretrained(
        device=device,
        torch_dtype=model_dtype,
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_max_len=int(tokenizer_max_len),
        load_text_encoder=bool(load_text_encoder),
        proprio_dim=(None if proprio_dim is None else int(proprio_dim)),
        redirect_common_files=bool(redirect_common_files),
        video_dit_config=video_dit_config,
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=action_dit_pretrained_path,
        skip_dit_load_from_pretrain=bool(skip_dit_load_from_pretrain),
        mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn),
        video_train_shift=float(video_scheduler.get("train_shift", 5.0)),
        video_infer_shift=float(video_scheduler.get("infer_shift", 5.0)),
        video_num_train_timesteps=int(video_scheduler.get("num_train_timesteps", 1000)),
        action_train_shift=float(action_scheduler["train_shift"]),
        action_infer_shift=float(action_scheduler["infer_shift"]),
        action_num_train_timesteps=int(action_scheduler["num_train_timesteps"]),
        loss_lambda_video=float(loss.get("lambda_video", 1.0)),
        loss_lambda_action=float(loss.get("lambda_action", 1.0)),
    )


def create_fastwam_joint(
    model_id: str,
    tokenizer_model_id: str,
    video_dit_config,
    tokenizer_max_len: int = 512,
    load_text_encoder: bool = True,
    proprio_dim: int | None = None,
    proprio_token_dropout_prob: float = 0.0,
    action_dit_config=None,
    action_dit_pretrained_path: str | None = None,
    skip_dit_load_from_pretrain: bool = False,
    video_scheduler=None,
    action_scheduler=None,
    loss=None,
    mot_checkpoint_mixed_attn: bool = True,
    redirect_common_files: bool = True,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    from .models.wan22.fastwam_joint import FastWAMJoint

    if isinstance(video_dit_config, DictConfig):
        video_dit_config = OmegaConf.to_container(video_dit_config, resolve=True)
    if not isinstance(video_dit_config, dict):
        raise ValueError(f"`video_dit_config` must resolve to a dict, got {type(video_dit_config)}")

    if isinstance(action_dit_config, DictConfig):
        action_dit_config = OmegaConf.to_container(action_dit_config, resolve=True)
    if action_dit_config is None:
        action_dit_config = {}
    if not isinstance(action_dit_config, dict):
        raise ValueError(f"`action_dit_config` must resolve to a dict, got {type(action_dit_config)}")

    if isinstance(video_scheduler, DictConfig):
        video_scheduler = OmegaConf.to_container(video_scheduler, resolve=True)
    if video_scheduler is None:
        video_scheduler = {}
    if not isinstance(video_scheduler, dict):
        raise ValueError(f"`video_scheduler` must be dict-like, got {type(video_scheduler)}")

    if isinstance(action_scheduler, DictConfig):
        action_scheduler = OmegaConf.to_container(action_scheduler, resolve=True)
    if action_scheduler is None:
        raise ValueError("`action_scheduler` is required for FastWAM.")
    if not isinstance(action_scheduler, dict):
        raise ValueError(f"`action_scheduler` must be dict-like, got {type(action_scheduler)}")
    required_action_scheduler_keys = {"train_shift", "infer_shift", "num_train_timesteps"}
    missing_keys = required_action_scheduler_keys - set(action_scheduler.keys())
    if missing_keys:
        raise ValueError(
            f"`action_scheduler` missing required keys: {sorted(missing_keys)}. "
            "Expected keys: train_shift, infer_shift, num_train_timesteps."
        )

    if isinstance(loss, DictConfig):
        loss = OmegaConf.to_container(loss, resolve=True)
    if loss is None:
        loss = {}
    if not isinstance(loss, dict):
        raise ValueError(f"`loss` must be dict-like, got {type(loss)}")

    return FastWAMJoint.from_wan22_pretrained(
        device=device,
        torch_dtype=model_dtype,
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_max_len=int(tokenizer_max_len),
        load_text_encoder=bool(load_text_encoder),
        proprio_dim=(None if proprio_dim is None else int(proprio_dim)),
        proprio_token_dropout_prob=float(proprio_token_dropout_prob),
        redirect_common_files=bool(redirect_common_files),
        video_dit_config=video_dit_config,
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=action_dit_pretrained_path,
        skip_dit_load_from_pretrain=bool(skip_dit_load_from_pretrain),
        mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn),
        video_train_shift=float(video_scheduler.get("train_shift", 5.0)),
        video_infer_shift=float(video_scheduler.get("infer_shift", 5.0)),
        video_num_train_timesteps=int(video_scheduler.get("num_train_timesteps", 1000)),
        action_train_shift=float(action_scheduler["train_shift"]),
        action_infer_shift=float(action_scheduler["infer_shift"]),
        action_num_train_timesteps=int(action_scheduler["num_train_timesteps"]),
        shared_train_timestep=bool(video_scheduler.get("shared_train_timestep", False)),
        loss_lambda_video=float(loss.get("lambda_video", 1.0)),
        loss_lambda_action=float(loss.get("lambda_action", 1.0)),
    )


def create_fastwam_flex(
    model_id: str,
    tokenizer_model_id: str,
    video_dit_config,
    tokenizer_max_len: int = 512,
    load_text_encoder: bool = True,
    proprio_dim: int | None = None,
    proprio_token_dropout_prob: float = 0.0,
    action_dit_config=None,
    action_dit_pretrained_path: str | None = None,
    skip_dit_load_from_pretrain: bool = False,
    video_scheduler=None,
    dino_scheduler=None,
    pointmap_scheduler=None,
    action_scheduler=None,
    loss=None,
    dino_model_name: str = "vit_base_patch16_dinov3.lvd1689m",
    dino_pretrained_path: str | None = None,
    dino_dim: int = 3072,
    dino_output_grid_size: int = 7,
    dino_pred_x0: bool = True,
    hbridge_enabled: bool = True,
    hbridge_bottom_ratio: float = 0.25,
    hbridge_top_ratio: float = 0.25,
    present_probabilities=None,
    output_probabilities=None,
    routing_sampling_scope: str = "fixed_all_modalities",
    cross_modal_predict=None,
    inference_present_modalities=None,
    inference_output_modalities=None,
    mot_checkpoint_mixed_attn: bool = True,
    modality_profile: bool = False,
    redirect_common_files: bool = True,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    from .models.wan22.fastwam_flex import FastWAMFlex

    def as_dict(value, name, default=None):
        if isinstance(value, DictConfig):
            value = OmegaConf.to_container(value, resolve=True)
        if value is None:
            value = {} if default is None else default
        if not isinstance(value, dict):
            raise ValueError(f"`{name}` must resolve to a dict, got {type(value)}")
        return value

    def modality_tuple(value, name, default):
        value = as_dict(value, name, default)
        required = {"rgb", "dino", "pointmap"}
        missing = required - set(value)
        if missing:
            raise ValueError(f"`{name}` missing required keys: {sorted(missing)}")
        return tuple(value[key] for key in ("rgb", "dino", "pointmap"))

    video_dit_config = as_dict(video_dit_config, "video_dit_config")
    action_dit_config = as_dict(action_dit_config, "action_dit_config")
    video_scheduler = as_dict(video_scheduler, "video_scheduler")
    dino_scheduler = as_dict(dino_scheduler, "dino_scheduler")
    pointmap_scheduler = as_dict(pointmap_scheduler, "pointmap_scheduler")
    action_scheduler = as_dict(action_scheduler, "action_scheduler")
    loss = as_dict(loss, "loss")
    required = {"train_shift", "infer_shift", "num_train_timesteps"}
    for name, scheduler in (
        ("dino_scheduler", dino_scheduler),
        ("pointmap_scheduler", pointmap_scheduler),
        ("action_scheduler", action_scheduler),
    ):
        missing = required - set(scheduler)
        if missing:
            raise ValueError(f"`{name}` missing required keys: {sorted(missing)}")

    present_probabilities = modality_tuple(
        present_probabilities,
        "present_probabilities",
        {"rgb": 1.0, "dino": 1.0, "pointmap": 1.0},
    )
    output_probabilities = modality_tuple(
        output_probabilities,
        "output_probabilities",
        {"rgb": 1.0, "dino": 1.0, "pointmap": 1.0},
    )
    cross_modal_predict = modality_tuple(
        cross_modal_predict,
        "cross_modal_predict",
        {"rgb": True, "dino": True, "pointmap": True},
    )
    inference_present_modalities = modality_tuple(
        inference_present_modalities,
        "inference_present_modalities",
        {"rgb": True, "dino": True, "pointmap": True},
    )
    inference_output_modalities = modality_tuple(
        inference_output_modalities,
        "inference_output_modalities",
        {"rgb": True, "dino": True, "pointmap": True},
    )

    return FastWAMFlex.from_wan22_pretrained(
        device=device,
        torch_dtype=model_dtype,
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_max_len=int(tokenizer_max_len),
        load_text_encoder=bool(load_text_encoder),
        proprio_dim=None if proprio_dim is None else int(proprio_dim),
        proprio_token_dropout_prob=float(proprio_token_dropout_prob),
        redirect_common_files=bool(redirect_common_files),
        video_dit_config=video_dit_config,
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=action_dit_pretrained_path,
        skip_dit_load_from_pretrain=bool(skip_dit_load_from_pretrain),
        mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn),
        video_train_shift=float(video_scheduler.get("train_shift", 5.0)),
        video_infer_shift=float(video_scheduler.get("infer_shift", 5.0)),
        video_num_train_timesteps=int(video_scheduler.get("num_train_timesteps", 1000)),
        action_train_shift=float(action_scheduler["train_shift"]),
        action_infer_shift=float(action_scheduler["infer_shift"]),
        action_num_train_timesteps=int(action_scheduler["num_train_timesteps"]),
        shared_train_timestep=bool(video_scheduler.get("shared_train_timestep", False)),
        dino_model_name=dino_model_name,
        dino_pretrained_path=dino_pretrained_path,
        dino_dim=int(dino_dim),
        dino_output_grid_size=int(dino_output_grid_size),
        dino_pred_x0=bool(dino_pred_x0),
        dino_train_shift=float(dino_scheduler["train_shift"]),
        dino_infer_shift=float(dino_scheduler["infer_shift"]),
        dino_num_train_timesteps=int(dino_scheduler["num_train_timesteps"]),
        pointmap_train_shift=float(pointmap_scheduler["train_shift"]),
        pointmap_infer_shift=float(pointmap_scheduler["infer_shift"]),
        pointmap_num_train_timesteps=int(pointmap_scheduler["num_train_timesteps"]),
        loss_lambda_video=float(loss.get("lambda_video", 1.0)),
        loss_lambda_action=float(loss.get("lambda_action", 1.0)),
        loss_lambda_dino=float(loss.get("lambda_dino", 1.0)),
        loss_lambda_pointmap=float(loss.get("lambda_pointmap", 1.0)),
        modality_profile=bool(modality_profile),
        hbridge_enabled=bool(hbridge_enabled),
        hbridge_bottom_ratio=float(hbridge_bottom_ratio),
        hbridge_top_ratio=float(hbridge_top_ratio),
        present_probabilities=tuple(float(value) for value in present_probabilities),
        output_probabilities=tuple(float(value) for value in output_probabilities),
        routing_sampling_scope=str(routing_sampling_scope),
        cross_modal_predict=tuple(bool(value) for value in cross_modal_predict),
        inference_present_modalities=tuple(
            bool(value) for value in inference_present_modalities
        ),
        inference_output_modalities=tuple(
            bool(value) for value in inference_output_modalities
        ),
    )


def create_fastwam_idm(
    model_id: str,
    tokenizer_model_id: str,
    video_dit_config,
    tokenizer_max_len: int = 512,
    load_text_encoder: bool = True,
    proprio_dim: int | None = None,
    action_dit_config=None,
    action_dit_pretrained_path: str | None = None,
    skip_dit_load_from_pretrain: bool = False,
    video_scheduler=None,
    action_scheduler=None,
    loss=None,
    mot_checkpoint_mixed_attn: bool = True,
    redirect_common_files: bool = True,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    from .models.wan22.fastwam_idm import (
        FastWAMIDM,
    )

    if isinstance(video_dit_config, DictConfig):
        video_dit_config = OmegaConf.to_container(video_dit_config, resolve=True)
    if not isinstance(video_dit_config, dict):
        raise ValueError(f"`video_dit_config` must resolve to a dict, got {type(video_dit_config)}")

    if isinstance(action_dit_config, DictConfig):
        action_dit_config = OmegaConf.to_container(action_dit_config, resolve=True)
    if action_dit_config is None:
        action_dit_config = {}
    if not isinstance(action_dit_config, dict):
        raise ValueError(f"`action_dit_config` must resolve to a dict, got {type(action_dit_config)}")

    if isinstance(video_scheduler, DictConfig):
        video_scheduler = OmegaConf.to_container(video_scheduler, resolve=True)
    if video_scheduler is None:
        video_scheduler = {}
    if not isinstance(video_scheduler, dict):
        raise ValueError(f"`video_scheduler` must be dict-like, got {type(video_scheduler)}")

    if isinstance(action_scheduler, DictConfig):
        action_scheduler = OmegaConf.to_container(action_scheduler, resolve=True)
    if action_scheduler is None:
        raise ValueError("`action_scheduler` is required for FastWAM.")
    if not isinstance(action_scheduler, dict):
        raise ValueError(f"`action_scheduler` must be dict-like, got {type(action_scheduler)}")
    required_action_scheduler_keys = {"train_shift", "infer_shift", "num_train_timesteps"}
    missing_keys = required_action_scheduler_keys - set(action_scheduler.keys())
    if missing_keys:
        raise ValueError(
            f"`action_scheduler` missing required keys: {sorted(missing_keys)}. "
            "Expected keys: train_shift, infer_shift, num_train_timesteps."
        )

    if isinstance(loss, DictConfig):
        loss = OmegaConf.to_container(loss, resolve=True)
    if loss is None:
        loss = {}
    if not isinstance(loss, dict):
        raise ValueError(f"`loss` must be dict-like, got {type(loss)}")

    return FastWAMIDM.from_wan22_pretrained(
        device=device,
        torch_dtype=model_dtype,
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_max_len=int(tokenizer_max_len),
        load_text_encoder=bool(load_text_encoder),
        proprio_dim=(None if proprio_dim is None else int(proprio_dim)),
        redirect_common_files=bool(redirect_common_files),
        video_dit_config=video_dit_config,
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=action_dit_pretrained_path,
        skip_dit_load_from_pretrain=bool(skip_dit_load_from_pretrain),
        mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn),
        video_train_shift=float(video_scheduler.get("train_shift", 5.0)),
        video_infer_shift=float(video_scheduler.get("infer_shift", 5.0)),
        video_num_train_timesteps=int(video_scheduler.get("num_train_timesteps", 1000)),
        action_train_shift=float(action_scheduler["train_shift"]),
        action_infer_shift=float(action_scheduler["infer_shift"]),
        action_num_train_timesteps=int(action_scheduler["num_train_timesteps"]),
        loss_lambda_video=float(loss.get("lambda_video", 1.0)),
        loss_lambda_action=float(loss.get("lambda_action", 1.0)),
    )


def build_datasets(data_cfg: DictConfig):
    train_ds = instantiate(data_cfg.train)
    mixture_cfg = data_cfg.get("train_mixture")
    if mixture_cfg is not None:
        from .datasets.mixture import WeightedMixtureDataset

        additional_cfgs = mixture_cfg.get("additional_datasets", [])
        if not additional_cfgs:
            raise ValueError("data.train_mixture.additional_datasets must not be empty")
        datasets = [train_ds]
        resolved_train_cfg = OmegaConf.to_container(data_cfg.train, resolve=True)
        for override in additional_cfgs:
            source_cfg = OmegaConf.merge(resolved_train_cfg, override)
            datasets.append(instantiate(source_cfg))
        train_ds = WeightedMixtureDataset(
            datasets,
            mixture_cfg.get("weights", []),
            epoch_size=mixture_cfg.get("epoch_size"),
            seed=int(mixture_cfg.get("seed", 0)),
        )
        logger.info(
            "Built weighted train mixture: weights=%s source_lengths=%s "
            "nominal_source_counts=%s epoch_size=%d",
            train_ds.weights,
            train_ds.dataset_lengths,
            train_ds.source_counts,
            len(train_ds),
        )
    if data_cfg.get("val") is None:
        val_ds = None
    else:
        train_stats_path = data_cfg.train.get("pretrained_norm_stats")
        default_stats_path = os.path.join(misc.get_work_dir(), "dataset_stats.json")
        val_stats_path = data_cfg.val.get("pretrained_norm_stats")
        pretrained_norm_stats = val_stats_path or train_stats_path or default_stats_path
        logger.info("Building val dataset with pretrained_norm_stats: %s", pretrained_norm_stats)
        val_ds = instantiate(data_cfg.val, pretrained_norm_stats=pretrained_norm_stats)
    return train_ds, val_ds


def _resolve_train_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    device_count = torch.cuda.device_count()
    if device_count <= 1:
        return "cuda:0"
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if local_rank < 0 or local_rank >= device_count:
        return "cuda:0"
    return f"cuda:{local_rank}"


def _resolve_global_rank_before_accelerator() -> int:
    """Resolve global rank without requiring Accelerator initialization."""
    if torch.distributed.is_initialized():
        return int(torch.distributed.get_rank())

    rank_text = os.environ.get("RANK")
    world_size_text = os.environ.get("WORLD_SIZE")
    try:
        world_size = 1 if world_size_text is None else int(world_size_text)
    except ValueError as error:
        raise RuntimeError(f"Invalid WORLD_SIZE={world_size_text!r}") from error
    if world_size <= 0:
        raise RuntimeError(f"WORLD_SIZE must be positive, got {world_size}")

    if rank_text is None:
        if world_size > 1:
            raise RuntimeError(
                "WORLD_SIZE indicates distributed execution but RANK is unset; "
                "refusing to let multiple processes persist the resolved config"
            )
        return 0
    try:
        rank = int(rank_text)
    except ValueError as error:
        raise RuntimeError(f"Invalid RANK={rank_text!r}") from error
    if rank < 0:
        raise RuntimeError(f"RANK must be non-negative, got {rank}")
    if world_size_text is not None and rank >= world_size:
        raise RuntimeError(f"RANK={rank} is outside WORLD_SIZE={world_size}")
    return rank


def _write_fsynced_temporary_file(destination: Path, payload: bytes) -> Path:
    fd, temporary_path = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_path)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _restore_file_after_failed_replace(destination: Path, previous: bytes | None) -> None:
    if previous is None:
        destination.unlink(missing_ok=True)
        return
    temporary = _write_fsynced_temporary_file(destination, previous)
    try:
        # Use rename for rollback so an injected os.replace failure cannot leave
        # a new config paired with an old checksum.
        os.rename(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    directory_fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _persist_resolved_config(
    cfg: DictConfig, *, global_rank: int | None = None
) -> tuple[Path, Path] | None:
    """Atomically persist a deterministic resolved config on global rank zero."""
    if global_rank is None:
        global_rank = _resolve_global_rank_before_accelerator()
    if global_rank != 0:
        return None

    output_dir = Path(str(cfg.output_dir)).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "config.yaml"
    checksum_path = output_dir / "config.yaml.sha256"
    config_bytes = OmegaConf.to_yaml(cfg, resolve=True, sort_keys=True).encode("utf-8")
    digest = hashlib.sha256(config_bytes).hexdigest()
    checksum_bytes = f"{digest}  {config_path.name}\n".encode("ascii")
    previous_config = config_path.read_bytes() if config_path.exists() else None
    previous_checksum = checksum_path.read_bytes() if checksum_path.exists() else None

    config_temporary: Path | None = None
    checksum_temporary: Path | None = None
    config_replaced = False
    checksum_replaced = False
    try:
        config_temporary = _write_fsynced_temporary_file(config_path, config_bytes)
        checksum_temporary = _write_fsynced_temporary_file(checksum_path, checksum_bytes)
        os.replace(config_temporary, config_path)
        config_temporary = None
        config_replaced = True
        os.replace(checksum_temporary, checksum_path)
        checksum_temporary = None
        checksum_replaced = True

        _fsync_directory(output_dir)
    except BaseException:
        if checksum_replaced:
            _restore_file_after_failed_replace(checksum_path, previous_checksum)
        if config_replaced:
            _restore_file_after_failed_replace(config_path, previous_config)
        if checksum_replaced or config_replaced:
            _fsync_directory(output_dir)
        raise
    finally:
        if config_temporary is not None:
            config_temporary.unlink(missing_ok=True)
        if checksum_temporary is not None:
            checksum_temporary.unlink(missing_ok=True)
    attempt_id = os.environ.get("FASTWAM_AUTO_RESUME_ATTEMPT_ID", "").strip()
    if attempt_id:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", attempt_id):
            raise ValueError(f"unsafe FASTWAM_AUTO_RESUME_ATTEMPT_ID: {attempt_id!r}")
        archive_dir = (
            output_dir / "reports" / "auto-resume" / f"attempt-{attempt_id}"
        )
        archive_dir.mkdir(parents=True, exist_ok=True)
        archive_config = archive_dir / "resolved-config.yaml"
        archive_checksum = archive_dir / "resolved-config.yaml.sha256"
        archive_checksum_bytes = (
            f"{digest}  {archive_config.name}\n".encode("ascii")
        )
        previous_archive_config = (
            archive_config.read_bytes() if archive_config.exists() else None
        )
        previous_archive_checksum = (
            archive_checksum.read_bytes() if archive_checksum.exists() else None
        )
        archive_config_tmp: Path | None = None
        archive_checksum_tmp: Path | None = None
        archive_config_replaced = False
        archive_checksum_replaced = False
        try:
            archive_config_tmp = _write_fsynced_temporary_file(
                archive_config, config_bytes
            )
            archive_checksum_tmp = _write_fsynced_temporary_file(
                archive_checksum, archive_checksum_bytes
            )
            os.replace(archive_config_tmp, archive_config)
            archive_config_tmp = None
            archive_config_replaced = True
            os.replace(archive_checksum_tmp, archive_checksum)
            archive_checksum_tmp = None
            archive_checksum_replaced = True
            _fsync_directory(archive_dir)
        except BaseException:
            if archive_checksum_replaced:
                _restore_file_after_failed_replace(
                    archive_checksum, previous_archive_checksum
                )
            if archive_config_replaced:
                _restore_file_after_failed_replace(
                    archive_config, previous_archive_config
                )
            if archive_checksum_replaced or archive_config_replaced:
                _fsync_directory(archive_dir)
            raise
        finally:
            if archive_config_tmp is not None:
                archive_config_tmp.unlink(missing_ok=True)
            if archive_checksum_tmp is not None:
                archive_checksum_tmp.unlink(missing_ok=True)
    return config_path, checksum_path


def run_training(cfg: DictConfig):
    global_rank = _resolve_global_rank_before_accelerator()
    setup_logging(
        log_level=logging.INFO,
        is_main_process=global_rank == 0,
    )
    misc.register_work_dir(cfg.output_dir)
    if global_rank == 0:
        _persist_resolved_config(cfg, global_rank=global_rank)

    set_global_seed(int(cfg.seed), rank_offset=False)
    model_device = _resolve_train_device()
    mixed_precision = _normalize_mixed_precision(cfg.mixed_precision)
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)
    model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
    initialization_fingerprint = attach_random_module_initialization_fingerprint(model)
    logger.info(
        "Random-module initialization fingerprint: sha256=%s",
        initialization_fingerprint,
    )
    train_ds, val_ds = build_datasets(cfg.data)

    from .datasets.b1k.checkpoint import build_training_checkpoint_contract

    checkpoint_contract = build_training_checkpoint_contract(cfg, train_ds)

    trainer_cls = Wan22Trainer
    if trainer_cls is None:
        from .trainer import Wan22Trainer as trainer_cls

    trainer = trainer_cls(
        cfg=cfg,
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        checkpoint_contract=checkpoint_contract,
    )
    trainer.train()

def run_inference(cfg: DictConfig):
    setup_logging(log_level=logging.INFO)
    inference_cfg = cfg.inference
    mixed_precision = _normalize_mixed_precision(cfg.mixed_precision)
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)

    model = instantiate(cfg.model, model_dtype=model_dtype, device=str(inference_cfg.device))
    checkpoint_path = inference_cfg.get("checkpoint_path")
    if checkpoint_path:
        ckpt = Path(checkpoint_path)
        if ckpt.exists():
            logger.info("Loading finetuned checkpoint: %s", checkpoint_path)
            model.load_checkpoint(checkpoint_path)
        else:
            logger.warning("Checkpoint not found, skipping load: %s", checkpoint_path)
    model.eval()
    
    def center_crop_resize(img: Image, width: int, height: int) -> Image.Image:
        src_w, src_h = img.size
        scale = max(width / src_w, height / src_h)
        resized = img.resize((round(src_w * scale), round(src_h * scale)), resample=Image.BILINEAR)
        rw, rh = resized.size
        left = max((rw - width) // 2, 0)
        top = max((rh - height) // 2, 0)
        return resized.crop((left, top, left + width, top + height))

    input_image = Image.open(str(inference_cfg.input_image_path)).convert("RGB")
    input_image = center_crop_resize(input_image, width=inference_cfg.width, height=inference_cfg.height)
    arr = np.array(input_image, dtype=np.float32)
    x = torch.from_numpy(arr)
    x = x.to(device=model.device, dtype=model.torch_dtype)
    x = x * (2.0 / 255.0) - 1.0
    x = repeat(x, "H W C -> B C H W", B=1)
    output_mp4 = str(inference_cfg.output_mp4)

    infer_kwargs = {
        "prompt": str(inference_cfg.prompt),
        "negative_prompt": str(inference_cfg.negative_prompt),
        "text_cfg_scale": float(inference_cfg.text_cfg_scale),
        "action_cfg_scale": float(inference_cfg.action_cfg_scale),
        "input_image": x,
        "num_frames": int(inference_cfg.num_frames),
        "num_inference_steps": int(inference_cfg.num_inference_steps),
        "sigma_shift": None if inference_cfg.get("sigma_shift") is None else float(inference_cfg.sigma_shift),
        "seed": int(inference_cfg.seed),
        "rand_device": str(inference_cfg.rand_device),
        "tiled": bool(inference_cfg.tiled),
    }

    infer_out = model.infer(**infer_kwargs)
    video = infer_out["video"]
    save_mp4(video, output_mp4, fps=15)
    logger.info("Saved inference video to %s", output_mp4)
    return output_mp4
