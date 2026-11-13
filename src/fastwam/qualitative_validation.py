# Copyright (c) 2026 Xiaomi Corporation

"""Deterministic distributed qualitative validation for B1K FastWAM.

The formal 128-GPU evaluator assigns one fixed validation window to every
rank.  This module deliberately stays independent of the training loop so the
same evaluator can be used by a standalone profiling job and by the trainer.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
import traceback
import uuid
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from .utils.video_io import save_mp4
from .utils.video_metrics import pil_frames_to_video_tensor, video_psnr, video_ssim


Window = tuple[int, int, int]

_METRIC_NAMES = (
    "rollout_future_psnr_vs_gt",
    "rollout_future_ssim_vs_gt",
    "rollout_future_psnr_vs_vae",
    "rollout_future_ssim_vs_vae",
    "vae_psnr_vs_gt",
    "vae_ssim_vs_gt",
    "action_l1_denormalized",
    "action_mse_denormalized",
)
_TIMING_NAMES = (
    "data_load_s",
    "inference_s",
    "vae_reconstruction_s",
    "metrics_s",
    "video_write_s",
    "barrier_s",
    "report_prepare_barrier_s",
    "total_s",
    "outer_s",
)


@dataclass(frozen=True)
class QualitativeValidationConfig:
    """Immutable execution contract for one qualitative validation pass."""

    num_windows: int = 128
    expected_tasks: int = 100
    videos_to_save: int = 8
    num_inference_steps: int = 10
    base_seed: int = 42
    fps: int = 8
    rand_device: str = "cpu"
    tiled: bool = False
    strict_one_window_per_rank: bool = True

    def __post_init__(self) -> None:
        for name in (
            "num_windows",
            "expected_tasks",
            "num_inference_steps",
            "fps",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if (
            isinstance(self.videos_to_save, bool)
            or not isinstance(self.videos_to_save, int)
            or not 0 <= self.videos_to_save <= self.num_windows
        ):
            raise ValueError(
                "videos_to_save must be an integer in [0, num_windows], "
                f"got {self.videos_to_save!r}"
            )
        if self.num_windows < self.expected_tasks:
            raise ValueError("num_windows must be at least expected_tasks to cover every task")
        if isinstance(self.base_seed, bool) or not isinstance(self.base_seed, int):
            raise TypeError("base_seed must be an integer")
        if not 0 <= self.base_seed < 2**63:
            raise ValueError("base_seed must be in [0, 2**63)")
        if not isinstance(self.rand_device, str) or not self.rand_device:
            raise ValueError("rand_device must be a non-empty string")


def select_qualitative_windows(
    fixed_windows: Sequence[Window],
    *,
    num_windows: int = 128,
    expected_tasks: int = 100,
) -> list[Window]:
    """Round-robin fixed windows so every task is represented before repeats."""
    if num_windows <= 0 or expected_tasks <= 0:
        raise ValueError("num_windows and expected_tasks must be positive")
    if num_windows < expected_tasks:
        raise ValueError("num_windows must be at least expected_tasks")

    grouped: OrderedDict[int, list[Window]] = OrderedDict()
    identities: set[Window] = set()
    global_indices: set[int] = set()
    for raw_window in fixed_windows:
        if len(raw_window) != 3:
            raise ValueError(f"window must be (window_id, episode, global_index), got {raw_window!r}")
        window = tuple(int(value) for value in raw_window)
        if window in identities or window[2] in global_indices:
            raise ValueError(f"fixed windows contain duplicate identity/global index: {window}")
        identities.add(window)
        global_indices.add(window[2])
        grouped.setdefault(window[1], []).append(window)

    if len(grouped) != expected_tasks:
        raise ValueError(
            f"expected fixed windows from {expected_tasks} tasks, got {len(grouped)}"
        )
    if len(identities) < num_windows:
        raise ValueError(f"need {num_windows} unique fixed windows, got {len(identities)}")

    selected: list[Window] = []
    depth = 0
    while len(selected) < num_windows:
        added = 0
        for task_windows in grouped.values():
            if depth < len(task_windows):
                selected.append(task_windows[depth])
                added += 1
                if len(selected) == num_windows:
                    break
        if added == 0:
            raise RuntimeError("fixed-window round-robin selection made no progress")
        depth += 1
    return selected


def qualitative_window_seed(window: Window, *, base_seed: int = 42) -> int:
    """Return a stable, domain-separated 32-bit seed for one window."""
    window_id, episode_index, global_index = (int(value) for value in window)
    payload = (
        f"b1k-qualitative-validation-v1:{int(base_seed)}:"
        f"{window_id}:{episode_index}:{global_index}"
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def qualitative_window_set_sha256(windows: Sequence[Window]) -> str:
    digest = hashlib.sha256(b"b1k-qualitative-validation-window-set-v1\0")
    for window in windows:
        payload = json.dumps([int(value) for value in window], separators=(",", ":")).encode(
            "ascii"
        )
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


def qualitative_video_window_positions(num_windows: int, videos_to_save: int) -> tuple[int, ...]:
    """Choose fixed, evenly spaced window positions including both endpoints."""
    if num_windows <= 0:
        raise ValueError("num_windows must be positive")
    if not 0 <= videos_to_save <= num_windows:
        raise ValueError("videos_to_save must be in [0, num_windows]")
    if videos_to_save == 0:
        return ()
    if videos_to_save == 1:
        return (0,)
    denominator = videos_to_save - 1
    return tuple(
        (index * (num_windows - 1) + denominator // 2) // denominator
        for index in range(videos_to_save)
    )


def qualitative_rank_windows(
    windows: Sequence[Window],
    *,
    rank: int,
    world_size: int,
    strict_one_window_per_rank: bool = True,
) -> list[Window]:
    if world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError(f"invalid rank/world_size pair: {rank}/{world_size}")
    if strict_one_window_per_rank and len(windows) != world_size:
        raise ValueError(
            "strict qualitative validation requires exactly one window per rank: "
            f"windows={len(windows)}, world_size={world_size}"
        )
    local = list(windows[rank::world_size])
    if strict_one_window_per_rank and len(local) != 1:
        raise RuntimeError(f"rank {rank} received {len(local)} windows instead of one")
    return local


def qualitative_video_filename(
    *, step: int, position: int, rank: int, window: Window, seed: int
) -> str:
    if min(step, position, rank, *window, seed) < 0:
        raise ValueError("qualitative video filename components must be non-negative")
    window_id, episode_index, global_index = window
    return (
        f"step-{step:06d}_window-{position:03d}_rank-{rank:03d}_"
        f"episode-{episode_index:06d}_index-{global_index:09d}_"
        f"seed-{seed:010d}.mp4"
    )


@torch.no_grad()
def run_qualitative_validation(
    *,
    model: Any,
    dataset: Any,
    accelerator: Any,
    fixed_windows: Sequence[Window],
    batch_sample: Callable[[dict[str, Any]], dict[str, Any]],
    action_denormalizer: Callable[[torch.Tensor], torch.Tensor],
    output_dir: str | os.PathLike[str],
    step: int,
    config: QualitativeValidationConfig = QualitativeValidationConfig(),
    initialization_fingerprint_sha256: str | None = None,
) -> dict[str, Any]:
    """Run one deterministic distributed qualitative validation pass.

    The caller owns model execution-mode restoration.  Use
    :func:`run_trainer_qualitative_validation` from a live trainer.
    """
    invocation_started = time.perf_counter()
    if step < 0:
        raise ValueError("step must be non-negative")
    selected = select_qualitative_windows(
        fixed_windows,
        num_windows=config.num_windows,
        expected_tasks=config.expected_tasks,
    )
    rank = int(accelerator.process_index)
    world_size = int(accelerator.num_processes)
    local_windows = qualitative_rank_windows(
        selected,
        rank=rank,
        world_size=world_size,
        strict_one_window_per_rank=config.strict_one_window_per_rank,
    )
    save_positions = set(
        qualitative_video_window_positions(config.num_windows, config.videos_to_save)
    )
    position_by_window = {window: position for position, window in enumerate(selected)}
    final_step_dir = Path(output_dir) / f"step-{step:06d}"
    preflight = _qualitative_preflight(
        accelerator=accelerator,
        final_step_dir=final_step_dir,
        selected=selected,
        step=step,
        world_size=world_size,
        config=config,
        initialization_fingerprint_sha256=initialization_fingerprint_sha256,
    )
    if preflight["mode"] == "reuse":
        return preflight["report"]
    attempt_dir = Path(preflight["attempt_dir"])

    total_started = time.perf_counter()
    local_error = None
    try:
        metric_sums, metric_counts, local_timing = _run_local_windows(
            model=model,
            dataset=dataset,
            accelerator=accelerator,
            local_windows=local_windows,
            position_by_window=position_by_window,
            save_positions=save_positions,
            batch_sample=batch_sample,
            action_denormalizer=action_denormalizer,
            videos_dir=attempt_dir / "videos",
            step=step,
            rank=rank,
            config=config,
        )
    except Exception as exc:  # Every rank must still enter the failure rendezvous.
        local_error = _exception_record(stage="local_evaluation", rank=rank, exc=exc)
        metric_sums = {name: 0.0 for name in _METRIC_NAMES}
        metric_counts = {name: 0 for name in _METRIC_NAMES}
        local_timing = {name: 0.0 for name in _TIMING_NAMES}

    barrier_started = time.perf_counter()
    _raise_if_any_rank_failed(accelerator, local_error, stage="local_evaluation")
    local_timing["barrier_s"] = time.perf_counter() - barrier_started

    metric_report = _reduce_metrics(metric_sums, metric_counts, accelerator)
    artifact_error = None
    windows_and_videos = None
    report_barrier_started = time.perf_counter()
    if accelerator.is_main_process:
        try:
            windows_and_videos = _collect_artifact_records(
                selected=selected,
                save_positions=save_positions,
                output_dir=attempt_dir,
                step=step,
                world_size=world_size,
                config=config,
            )
        except Exception as exc:
            artifact_error = _exception_record(stage="artifact_validation", rank=rank, exc=exc)
    _raise_if_any_rank_failed(accelerator, artifact_error, stage="artifact_validation")
    local_timing["report_prepare_barrier_s"] = time.perf_counter() - report_barrier_started
    local_timing["total_s"] = time.perf_counter() - total_started
    local_timing["outer_s"] = time.perf_counter() - invocation_started
    timing_report = _reduce_timings(local_timing, accelerator)

    publish_error = None
    report = None
    if accelerator.is_main_process:
        try:
            windows, videos = windows_and_videos
            report = _build_report(
                selected=selected,
                windows=windows,
                videos=videos,
                metric_report=metric_report,
                timing_report=timing_report,
                step=step,
                world_size=world_size,
                config=config,
                initialization_fingerprint_sha256=initialization_fingerprint_sha256,
            )
            _atomic_write_json(attempt_dir / "report.json", report)
            os.rename(attempt_dir, final_step_dir)
            _fsync_directory(final_step_dir.parent)
        except Exception as exc:
            publish_error = _exception_record(stage="report_publish", rank=rank, exc=exc)
    _raise_if_any_rank_failed(accelerator, publish_error, stage="report_publish")
    if not accelerator.is_main_process:
        report = _read_completed_report(final_step_dir / "report.json")
    return report


def _run_local_windows(
    *,
    model: Any,
    dataset: Any,
    accelerator: Any,
    local_windows: Sequence[Window],
    position_by_window: dict[Window, int],
    save_positions: set[int],
    batch_sample: Callable[[dict[str, Any]], dict[str, Any]],
    action_denormalizer: Callable[[torch.Tensor], torch.Tensor],
    videos_dir: Path,
    step: int,
    rank: int,
    config: QualitativeValidationConfig,
) -> tuple[dict[str, float], dict[str, int], dict[str, float]]:
    metric_sums = {name: 0.0 for name in _METRIC_NAMES}
    metric_counts = {name: 0 for name in _METRIC_NAMES}
    local_timing = {name: 0.0 for name in _TIMING_NAMES}
    model.eval()
    for window in local_windows:
        position = position_by_window[window]
        _, _, global_index = window
        seed = qualitative_window_seed(window, base_seed=config.base_seed)

        started = time.perf_counter()
        sample = batch_sample(dataset[global_index])
        local_timing["data_load_s"] += time.perf_counter() - started
        video_batch, action_batch, proprio_batch = _validate_eval_sample(sample)
        video_gt_normalized = video_batch[0]
        video_gt = (
            (video_gt_normalized.detach().float().cpu().clamp(-1.0, 1.0) + 1.0) * 0.5
        ).contiguous()
        input_image = video_gt_normalized[:, 0].unsqueeze(0)
        action_gt = action_batch[0].detach().to(device="cpu", dtype=torch.float32)
        proprio = proprio_batch[0, 0]

        infer_kwargs: dict[str, Any] = {
            "prompt": None,
            "input_image": input_image,
            "num_video_frames": int(video_gt.shape[1]),
            "action_horizon": int(action_gt.shape[0]),
            "action": None,
            "proprio": proprio,
            "text_cfg_scale": 1.0,
            "num_inference_steps": config.num_inference_steps,
            "seed": seed,
            "rand_device": config.rand_device,
            "tiled": config.tiled,
            "test_action_with_infer_action": False,
        }
        if sample.get("context") is not None:
            infer_kwargs["context"] = sample["context"][0]
            infer_kwargs["context_mask"] = sample["context_mask"][0]
        else:
            infer_kwargs["prompt"] = sample["prompt"][0]

        _synchronize_device(accelerator.device)
        started = time.perf_counter()
        with accelerator.autocast():
            prediction = model.infer_joint(**infer_kwargs)
        _synchronize_device(accelerator.device)
        local_timing["inference_s"] += time.perf_counter() - started
        if not isinstance(prediction, dict) or "video" not in prediction or "action" not in prediction:
            raise ValueError("infer_joint must return a mapping with video and action")
        rollout = pil_frames_to_video_tensor(prediction["video"])
        action_pred = torch.as_tensor(prediction["action"], dtype=torch.float32, device="cpu")
        if rollout.shape != video_gt.shape:
            raise ValueError(
                f"rollout/GT shape mismatch: {tuple(rollout.shape)} vs {tuple(video_gt.shape)}"
            )
        if action_pred.shape != action_gt.shape:
            raise ValueError(
                f"predicted/GT action shape mismatch: {tuple(action_pred.shape)} vs {tuple(action_gt.shape)}"
            )

        _synchronize_device(accelerator.device)
        started = time.perf_counter()
        with accelerator.autocast():
            gt_for_vae = video_gt_normalized.unsqueeze(0).to(
                device=model.device, dtype=model.torch_dtype
            )
            vae_latents = model._encode_video_latents(gt_for_vae, tiled=config.tiled)
            vae_frames = model._decode_latents(vae_latents, tiled=config.tiled)
        _synchronize_device(accelerator.device)
        vae_reconstruction = pil_frames_to_video_tensor(vae_frames)
        local_timing["vae_reconstruction_s"] += time.perf_counter() - started
        if vae_reconstruction.shape != video_gt.shape:
            raise ValueError(
                "VAE reconstruction/GT shape mismatch: "
                f"{tuple(vae_reconstruction.shape)} vs {tuple(video_gt.shape)}"
            )

        started = time.perf_counter()
        action_pred_denormalized = action_denormalizer(action_pred.clone())
        action_gt_denormalized = action_denormalizer(action_gt.clone())
        if action_pred_denormalized.shape != action_gt_denormalized.shape:
            raise ValueError("denormalized predicted/GT action shape mismatch")
        action_diff = action_pred_denormalized.float() - action_gt_denormalized.float()
        rollout_future = rollout[:, 1:]
        video_gt_future = video_gt[:, 1:]
        vae_future = vae_reconstruction[:, 1:]
        values = {
            "rollout_future_psnr_vs_gt": video_psnr(rollout_future, video_gt_future),
            "rollout_future_ssim_vs_gt": video_ssim(rollout_future, video_gt_future),
            "rollout_future_psnr_vs_vae": video_psnr(rollout_future, vae_future),
            "rollout_future_ssim_vs_vae": video_ssim(rollout_future, vae_future),
            "vae_psnr_vs_gt": video_psnr(vae_reconstruction, video_gt),
            "vae_ssim_vs_gt": video_ssim(vae_reconstruction, video_gt),
            "action_l1_denormalized": float(action_diff.abs().mean().item()),
            "action_mse_denormalized": float(action_diff.square().mean().item()),
        }
        for name, value in values.items():
            if not np.isfinite(value):
                raise RuntimeError(
                    f"non-finite qualitative metric {name} for window {window}: {value}"
                )
            metric_sums[name] += float(value)
            metric_counts[name] += 1
        local_timing["metrics_s"] += time.perf_counter() - started

        if position in save_positions:
            started = time.perf_counter()
            stitched_frames = _stitch_videos(rollout, vae_reconstruction, video_gt)
            filename = qualitative_video_filename(
                step=step, position=position, rank=rank, window=window, seed=seed
            )
            _atomic_save_mp4(stitched_frames, videos_dir / filename, fps=config.fps)
            local_timing["video_write_s"] += time.perf_counter() - started
    return metric_sums, metric_counts, local_timing


def run_trainer_qualitative_validation(
    trainer: Any,
    *,
    output_dir: str | os.PathLike[str] | None = None,
    step: int | None = None,
    config: QualitativeValidationConfig = QualitativeValidationConfig(),
    fixed_windows: Sequence[Window] | None = None,
    fixed_anchors_per_task: int = 10,
    fixed_anchor_stride: int = 6,
) -> dict[str, Any]:
    """Trainer-facing API with exact mixed train/eval state restoration."""
    if trainer.val_dataset is None:
        raise ValueError("qualitative validation requires trainer.val_dataset")
    if fixed_windows is None:
        # Delayed import keeps this standalone module free of trainer cycles.
        from .trainer import build_fixed_validation_window_indices

        fixed_windows = build_fixed_validation_window_indices(
            trainer.val_dataset,
            tasks=config.expected_tasks,
            anchors_per_task=fixed_anchors_per_task,
            stride=fixed_anchor_stride,
        )
    unwrapped_model = trainer.accelerator.unwrap_model(trainer.model)
    execution_state = trainer._capture_model_execution_state(trainer.model, unwrapped_model)
    try:
        return run_qualitative_validation(
            model=unwrapped_model,
            dataset=trainer.val_dataset,
            accelerator=trainer.accelerator,
            fixed_windows=fixed_windows,
            batch_sample=trainer._to_batched_eval_sample,
            action_denormalizer=_b1k_action_denormalizer(trainer.val_dataset),
            output_dir=(
                Path(trainer.output_dir) / "qualitative-validation"
                if output_dir is None
                else output_dir
            ),
            step=int(trainer.global_step if step is None else step),
            config=config,
            initialization_fingerprint_sha256=getattr(
                trainer, "initialization_fingerprint_sha256", None
            ),
        )
    finally:
        trainer._restore_model_execution_state(execution_state)


def _validate_eval_sample(sample: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    video = sample.get("video")
    action = sample.get("action")
    proprio = sample.get("proprio")
    if not isinstance(video, torch.Tensor) or video.ndim != 5 or video.shape[0] != 1:
        raise ValueError("qualitative video must have shape [1,3,T,H,W]")
    if video.shape[1] != 3 or video.shape[2] <= 1:
        raise ValueError("qualitative video requires three channels and at least two frames")
    if not isinstance(action, torch.Tensor) or action.ndim != 3 or action.shape[0] != 1:
        raise ValueError("qualitative action must have shape [1,T,D]")
    if not isinstance(proprio, torch.Tensor) or proprio.ndim != 3 or proprio.shape[0] != 1:
        raise ValueError("qualitative proprio must have shape [1,T,D]")
    context = sample.get("context")
    context_mask = sample.get("context_mask")
    if (context is None) != (context_mask is None):
        raise ValueError("qualitative context and context_mask must appear together")
    return video, action, proprio


def _b1k_action_denormalizer(dataset: Any) -> Callable[[torch.Tensor], torch.Tensor]:
    processor = getattr(getattr(dataset, "lerobot_dataset", dataset), "processor", None)
    if processor is None:
        raise TypeError("B1K validation dataset does not expose a processor")
    nested_normalizer = processor.normalizer
    mixed_normalizer = getattr(nested_normalizer, "normalizer", None)
    denormalize = getattr(mixed_normalizer, "denormalize_action", None)
    if not callable(denormalize):
        raise TypeError("B1K processor does not expose mixed action denormalization")
    return denormalize


def _stitch_videos(*videos: torch.Tensor) -> list[Image.Image]:
    if not videos or any(video.shape != videos[0].shape for video in videos):
        raise ValueError("stitched videos must be non-empty and have identical shapes")
    stitched = torch.cat(videos, dim=2).float().cpu().clamp(0.0, 1.0)
    frames = []
    for index in range(stitched.shape[1]):
        array = (
            stitched[:, index].permute(1, 2, 0).contiguous().numpy() * 255.0
        ).round().astype(np.uint8)
        frames.append(Image.fromarray(array, mode="RGB"))
    return frames


def _reduce_metrics(sums: dict[str, float], counts: dict[str, int], accelerator: Any) -> dict[str, Any]:
    values = [sums[name] for name in _METRIC_NAMES] + [counts[name] for name in _METRIC_NAMES]
    tensor = torch.tensor(values, device=accelerator.device, dtype=torch.float64)
    reduced = accelerator.reduce(tensor, reduction="sum").detach().cpu().tolist()
    report: dict[str, Any] = {}
    offset = len(_METRIC_NAMES)
    for index, name in enumerate(_METRIC_NAMES):
        total = float(reduced[index])
        count = int(round(reduced[offset + index]))
        if count <= 0:
            raise RuntimeError(f"qualitative metric {name} has zero global count")
        mean = total / count
        if not np.isfinite(total) or not np.isfinite(mean):
            raise RuntimeError(f"qualitative metric {name} is non-finite after reduction")
        report[name] = {"sum": total, "count": count, "mean": mean}
    return report


def _reduce_timings(local: dict[str, float], accelerator: Any) -> dict[str, Any]:
    tensor = torch.tensor(
        [local[name] for name in _TIMING_NAMES],
        device=accelerator.device,
        dtype=torch.float64,
    ).unsqueeze(0)
    # accelerate<=1.12 only implements SUM-backed reduce operations; passing
    # reduction="max" silently behaves like SUM.  Gather the tiny timing
    # vector and compute the exact rank maximum explicitly.
    gathered = accelerator.gather(tensor).detach().cpu()
    world_size = int(accelerator.num_processes)
    if gathered.numel() != world_size * len(_TIMING_NAMES):
        raise RuntimeError(
            "qualitative timing gather returned an unexpected shape: "
            f"{tuple(gathered.shape)}, world_size={world_size}"
        )
    gathered = gathered.reshape(world_size, len(_TIMING_NAMES))
    sums = gathered.sum(dim=0).tolist()
    maxima = gathered.max(dim=0).values.tolist()
    return {
        name: {
            "sum_seconds": float(sums[index]),
            "mean_seconds": float(sums[index]) / world_size,
            "max_seconds": float(maxima[index]),
            "rank_count": world_size,
        }
        for index, name in enumerate(_TIMING_NAMES)
    }


def _qualitative_preflight(
    *,
    accelerator: Any,
    final_step_dir: Path,
    selected: Sequence[Window],
    step: int,
    world_size: int,
    config: QualitativeValidationConfig,
    initialization_fingerprint_sha256: str | None,
) -> dict[str, Any]:
    decision = None
    if accelerator.is_main_process:
        try:
            final_step_dir.parent.mkdir(parents=True, exist_ok=True)
            if final_step_dir.exists():
                report = _read_completed_report(final_step_dir / "report.json")
                _validate_reusable_report(
                    report,
                    final_step_dir=final_step_dir,
                    selected=selected,
                    step=step,
                    world_size=world_size,
                    config=config,
                    initialization_fingerprint_sha256=initialization_fingerprint_sha256,
                )
                decision = {"mode": "reuse", "report": report}
            else:
                attempt_dir = final_step_dir.parent / (
                    f".{final_step_dir.name}.attempt-{uuid.uuid4().hex}"
                )
                attempt_dir.mkdir(parents=False, exist_ok=False)
                decision = {"mode": "run", "attempt_dir": str(attempt_dir)}
        except Exception as exc:
            decision = {
                "mode": "error",
                "error": _exception_record(stage="preflight", rank=0, exc=exc),
            }
    decision = _broadcast_main_object(accelerator, decision)
    if not isinstance(decision, dict) or decision.get("mode") not in {"reuse", "run", "error"}:
        raise RuntimeError(f"invalid qualitative preflight decision: {decision!r}")
    if decision["mode"] == "error":
        raise RuntimeError(_format_distributed_errors("preflight", [decision["error"]]))
    return decision


def _broadcast_main_object(accelerator: Any, value: Any) -> Any:
    if int(accelerator.num_processes) == 1:
        return value
    from accelerate.utils import broadcast_object_list

    payload = [value if accelerator.is_main_process else None]
    broadcast_object_list(payload, from_process=0)
    return payload[0]


def _raise_if_any_rank_failed(accelerator: Any, error: dict[str, Any] | None, *, stage: str) -> None:
    status = {"rank": int(accelerator.process_index), "error": error}
    if int(accelerator.num_processes) == 1:
        statuses = [status]
    else:
        from accelerate.utils import gather_object

        statuses = gather_object([status])
    failures = [item["error"] for item in statuses if item.get("error") is not None]
    if failures:
        raise RuntimeError(_format_distributed_errors(stage, failures))


def _exception_record(*, stage: str, rank: int, exc: Exception) -> dict[str, Any]:
    return {
        "stage": stage,
        "rank": int(rank),
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-8000:],
    }


def _format_distributed_errors(stage: str, failures: Sequence[dict[str, Any]]) -> str:
    details = "; ".join(
        f"rank={failure.get('rank')} {failure.get('type')}: {failure.get('message')}"
        for failure in failures
    )
    return f"qualitative validation failed coherently during {stage}: {details}"


def _collect_artifact_records(
    *,
    selected: Sequence[Window],
    save_positions: set[int],
    output_dir: Path,
    step: int,
    world_size: int,
    config: QualitativeValidationConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    windows = []
    videos = []
    for position, window in enumerate(selected):
        seed = qualitative_window_seed(window, base_seed=config.base_seed)
        window_id, episode_index, global_index = window
        record = {
            "position": position,
            "rank": position % world_size,
            "window_id": window_id,
            "episode_index": episode_index,
            "global_index": global_index,
            "seed": seed,
        }
        windows.append(record)
        if position in save_positions:
            filename = qualitative_video_filename(
                step=step,
                position=position,
                rank=position % world_size,
                window=window,
                seed=seed,
            )
            path = output_dir / "videos" / filename
            if not path.is_file():
                raise RuntimeError(f"missing completed qualitative video: {path}")
            videos.append(
                record
                | {
                    "file": f"videos/{filename}",
                    "bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                    "layout": "prediction|vae_reconstruction|ground_truth",
                }
            )
    return windows, videos


def _build_report(
    *,
    selected: Sequence[Window],
    windows: list[dict[str, Any]],
    videos: list[dict[str, Any]],
    metric_report: dict[str, Any],
    timing_report: dict[str, Any],
    step: int,
    world_size: int,
    config: QualitativeValidationConfig,
    initialization_fingerprint_sha256: str | None,
) -> dict[str, Any]:
    return {
        "schema": "b1k_qualitative_validation.v1",
        "status": "passed",
        "step": int(step),
        "world_size": int(world_size),
        "num_windows": len(selected),
        "num_tasks": len({window[1] for window in selected}),
        "base_seed": config.base_seed,
        "window_set_sha256": qualitative_window_set_sha256(selected),
        "initialization_fingerprint_sha256": initialization_fingerprint_sha256,
        "inference": {
            "num_inference_steps": config.num_inference_steps,
            "rand_device": config.rand_device,
            "tiled": config.tiled,
            "gt_action_conditioning": False,
            "duplicate_action_debug_rollout": False,
        },
        "timing_contract": {
            "clock": "wall_seconds_with_cuda_stage_boundary_synchronization",
            "inference_s": "joint_denoise_plus_rollout_vae_decode",
            "vae_reconstruction_s": "ground_truth_vae_encode_plus_decode",
            "barrier_s": "rank_local_evaluation_failure_rendezvous",
            "report_prepare_barrier_s": "main_rank_artifact_hashing_rendezvous",
            "total_s": "data_through_artifact_validation_before_atomic_report_commit",
            "outer_s": "preflight_through_artifact_validation_before_atomic_report_commit",
            "excluded_from_outer_s": "final_report_json_fsync_and_directory_rename",
        },
        "metric_contract": {
            "rollout_video_metrics": "future_frames_only_excluding_condition_frame_0",
            "vae_vs_gt_metrics": "all_frames",
            "action_metrics": "all_action_tokens_after_b1k_denormalization",
        },
        "metrics": metric_report,
        "timings": timing_report,
        "windows": windows,
        "videos": videos,
    }


def _atomic_save_mp4(frames: Sequence[Image.Image], path: Path, *, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.stem}.", suffix=".tmp.mp4"
    )
    os.close(fd)
    try:
        save_mp4(frames, temporary, fps=fps)
        with open(temporary, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _read_completed_report(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"failed to read completed qualitative report: {path}") from exc
    if not isinstance(payload, dict) or payload.get("status") != "passed":
        raise RuntimeError(f"invalid completed qualitative report: {path}")
    return payload


def _validate_reusable_report(
    report: dict[str, Any],
    *,
    final_step_dir: Path,
    selected: Sequence[Window],
    step: int,
    world_size: int,
    config: QualitativeValidationConfig,
    initialization_fingerprint_sha256: str | None,
) -> None:
    expected = {
        "schema": "b1k_qualitative_validation.v1",
        "status": "passed",
        "step": int(step),
        "world_size": int(world_size),
        "num_windows": len(selected),
        "num_tasks": len({window[1] for window in selected}),
        "base_seed": config.base_seed,
        "window_set_sha256": qualitative_window_set_sha256(selected),
        "initialization_fingerprint_sha256": initialization_fingerprint_sha256,
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise RuntimeError(
                f"existing qualitative report identity mismatch for {key}: "
                f"expected={value!r}, actual={report.get(key)!r}"
            )
    inference = report.get("inference")
    if not isinstance(inference, dict) or inference.get("num_inference_steps") != config.num_inference_steps:
        raise RuntimeError("existing qualitative report inference contract mismatch")
    videos = report.get("videos")
    if not isinstance(videos, list) or len(videos) != config.videos_to_save:
        raise RuntimeError("existing qualitative report video inventory mismatch")
    for artifact in videos:
        relative = artifact.get("file")
        if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise RuntimeError("existing qualitative report contains an unsafe video path")
        path = final_step_dir / relative
        if not path.is_file() or path.stat().st_size != artifact.get("bytes"):
            raise RuntimeError(f"existing qualitative video size mismatch: {path}")
        if _sha256_file(path) != artifact.get("sha256"):
            raise RuntimeError(f"existing qualitative video hash mismatch: {path}")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _synchronize_device(device: torch.device | str) -> None:
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
