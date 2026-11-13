# Copyright (c) 2026 Xiaomi Corporation

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from fastwam.datasets.b1k.contract import (
    ACTION_CONSTANT_DIM,
    ACTION_DIM,
    ACTION_GRIPPER_DIMS,
    postprocess_b1k_action,
)

TEMPORAL_STRATEGIES = (
    "baseline",
    "speed_2x",
    "speed_1p5_m21",
    "speed_3x_m11",
    "adaptive_upper_body_2x",
)
FIXED_ACCELERATED_TARGET_TICKS = {
    "speed_1p5_m21": 21,
    "speed_3x_m11": 11,
}
ACCELERATED_STRATEGIES = {"speed_2x", *FIXED_ACCELERATED_TARGET_TICKS}
BASE_VELOCITY_SLICE = slice(0, 3)
BASE_PHYSICAL_SCALE = np.array([0.75, 0.75, 1.0], dtype=np.float64)
ADAPTIVE_SCORE_THRESHOLDS = (0.01, 0.02, 0.03, 0.05, 0.075, 0.1)
QPOS_DIMS = tuple(
    dim
    for dim in range(3, ACTION_DIM)
    if dim not in (ACTION_CONSTANT_DIM, *ACTION_GRIPPER_DIMS)
)


@dataclass(frozen=True)
class B1KExecutionConfig:
    """Timing contract for the first B1K closed-loop execution policy."""

    policy_hz: int = 15
    native_hz: int = 30
    prediction_horizon: int = 32
    execute_horizon: int = 4
    resampling: str = "zero_order_hold"
    temporal_strategy: str = "baseline"
    base_stationary_threshold: float = 0.02
    adaptive_high_threshold: float = 0.05
    adaptive_low_threshold: float = 0.02
    adaptive_low_tokens: int = 2

    def __post_init__(self) -> None:
        for name in ("policy_hz", "native_hz", "prediction_horizon", "execute_horizon"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}.")
        if self.native_hz % self.policy_hz != 0:
            raise ValueError(
                "native_hz must be an integer multiple of policy_hz for zero-order hold, "
                f"got native_hz={self.native_hz}, policy_hz={self.policy_hz}."
            )
        if self.execute_horizon > self.prediction_horizon:
            raise ValueError(
                "execute_horizon cannot exceed prediction_horizon, "
                f"got {self.execute_horizon} > {self.prediction_horizon}."
            )
        if self.resampling not in {"zero_order_hold", "hybrid_linear"}:
            raise ValueError(
                "resampling must be 'zero_order_hold' or 'hybrid_linear', "
                f"got {self.resampling!r}."
            )
        if (
            self.resampling == "hybrid_linear"
            and self.execute_horizon >= self.prediction_horizon
        ):
            raise ValueError(
                "hybrid_linear requires one prediction token after execute_horizon"
            )
        if not 0 <= self.adaptive_low_threshold < self.adaptive_high_threshold:
            raise ValueError("adaptive thresholds must satisfy 0 <= low < high")
        if (
            isinstance(self.adaptive_low_tokens, bool)
            or not isinstance(self.adaptive_low_tokens, int)
            or self.adaptive_low_tokens < 1
        ):
            raise ValueError("adaptive_low_tokens must be a positive integer")
        if self.temporal_strategy not in TEMPORAL_STRATEGIES:
            raise ValueError(
                "temporal_strategy must be one of "
                f"{TEMPORAL_STRATEGIES}, got {self.temporal_strategy!r}."
            )
        if self.temporal_strategy != "baseline":
            if self.resampling != "zero_order_hold":
                raise ValueError(
                    f"temporal_strategy {self.temporal_strategy!r} only supports zero_order_hold"
                )
            if self.temporal_strategy != "speed_2x" and (
                self.prediction_horizon != 32 or self.execute_horizon != 16
            ):
                raise ValueError(
                    f"temporal_strategy {self.temporal_strategy!r} requires "
                    "prediction_horizon=32 and execute_horizon=16"
                )
        if (
            isinstance(self.base_stationary_threshold, bool)
            or not isinstance(self.base_stationary_threshold, (int, float))
            or not np.isfinite(self.base_stationary_threshold)
            or self.base_stationary_threshold < 0
        ):
            raise ValueError(
                "base_stationary_threshold must be a finite non-negative number, "
                f"got {self.base_stationary_threshold!r}."
            )

    @property
    def ticks_per_policy_token(self) -> int:
        return self.native_hz // self.policy_hz

    @property
    def output_ticks(self) -> int | None:
        if self.temporal_strategy == "baseline":
            return self.execute_horizon * self.ticks_per_policy_token
        if self.temporal_strategy == "speed_2x":
            return self.execute_horizon
        return FIXED_ACCELERATED_TARGET_TICKS.get(self.temporal_strategy)

    @property
    def source_native_ticks(self) -> int | None:
        if self.temporal_strategy == "speed_2x":
            return self.execute_horizon * self.ticks_per_policy_token
        if self.temporal_strategy in FIXED_ACCELERATED_TARGET_TICKS:
            return 32
        return None

    @property
    def base_velocity_scale(self) -> float:
        return (
            self.source_native_ticks / self.output_ticks
            if self.source_native_ticks is not None and self.output_ticks is not None
            else 1.0
        )

    @property
    def uses_full_horizon_speed_2x(self) -> bool:
        return (
            self.temporal_strategy == "speed_2x"
            and self.execute_horizon == self.prediction_horizon
        )

    def metadata(self) -> dict[str, Any]:
        metadata = {
            "schema": "b1k_execution_v1",
            "action_dim": ACTION_DIM,
            "input_denormalized": True,
            "policy_hz": self.policy_hz,
            "native_hz": self.native_hz,
            "prediction_horizon": self.prediction_horizon,
            "execute_horizon": self.execute_horizon,
            "ticks_per_policy_token": self.ticks_per_policy_token,
            "output_ticks": self.output_ticks,
            "resampling": self.resampling,
            "temporal_strategy": self.temporal_strategy,
            "base_velocity_scale": self.base_velocity_scale,
            "base_stationary_threshold": None,
            "adaptive_high_threshold": (
                self.adaptive_high_threshold
                if self.temporal_strategy == "adaptive_upper_body_2x"
                else None
            ),
            "adaptive_low_threshold": (
                self.adaptive_low_threshold
                if self.temporal_strategy == "adaptive_upper_body_2x"
                else None
            ),
            "adaptive_low_tokens": (
                self.adaptive_low_tokens
                if self.temporal_strategy == "adaptive_upper_body_2x"
                else None
            ),
            "constant_zero_dims": [ACTION_CONSTANT_DIM],
            "binary_gripper_dims": list(ACTION_GRIPPER_DIMS),
            "gripper_threshold": 0.0,
            "algorithm_revision": (
                "se2_trajectory_preserving_v1"
                if self.temporal_strategy in ACCELERATED_STRATEGIES
                else None
            ),
            "source_native_ticks": (
                self.source_native_ticks
            ),
            "target_native_ticks": self.output_ticks,
            "time_compression_ratio": (
                self.source_native_ticks / self.output_ticks
                if self.source_native_ticks is not None and self.output_ticks is not None
                else None
            ),
            "base_resampling_method": (
                "se2_exp_integrate_bin_log"
                if self.temporal_strategy in ACCELERATED_STRATEGIES
                else None
            ),
            "base_action_input_limits": [[-1.0, 1.0]] * 3,
            "base_physical_scales": BASE_PHYSICAL_SCALE.tolist(),
            "base_physical_limits": [
                [-float(scale), float(scale)] for scale in BASE_PHYSICAL_SCALE
            ],
            "qpos_dims": list(QPOS_DIMS),
            "qpos_resampling_method": (
                "linear_at_compressed_tick_starts_without_lookahead_at_full_prediction_horizon"
                if self.uses_full_horizon_speed_2x
                else "linear_at_compressed_tick_starts_with_one_token_lookahead"
                if self.temporal_strategy in ACCELERATED_STRATEGIES
                else None
            ),
            "gripper_resampling_method": (
                "zero_order_hold"
                if self.temporal_strategy in ACCELERATED_STRATEGIES
                else self.resampling
            ),
            "adaptive_linear_speed_reference": (
                0.35 if self.temporal_strategy == "adaptive_upper_body_2x" else None
            ),
            "adaptive_angular_speed_reference": (
                0.30 if self.temporal_strategy == "adaptive_upper_body_2x" else None
            ),
            "adaptive_lookahead_tokens": (
                1 if self.temporal_strategy == "adaptive_upper_body_2x" else None
            ),
        }
        if self.uses_full_horizon_speed_2x:
            metadata.update(
                qpos_lookahead_tokens=0,
                prediction_horizon_boundary="execute_all_tokens_without_lookahead",
            )
        return metadata


@dataclass
class B1KExecutionState:
    """State carried between chunks of one adaptive rollout."""

    adaptive_moving: bool = False
    adaptive_low_streak: int = 0


@dataclass(frozen=True)
class B1KExecutionBatch:
    """Native-rate commands and the exact contract used to construct them."""

    ticks: torch.Tensor | np.ndarray
    metadata: dict[str, Any]
    next_state: B1KExecutionState


def _se2_compose(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    c, s = np.cos(left[2]), np.sin(left[2])
    return np.array(
        [
            left[0] + c * right[0] - s * right[1],
            left[1] + s * right[0] + c * right[1],
            left[2] + right[2],
        ],
        dtype=np.float64,
    )


def _se2_exp(twist: np.ndarray, duration: float) -> np.ndarray:
    vx, vy, wz = (float(value) for value in twist)
    theta = wz * duration
    if abs(theta) < 1e-8:
        a = duration * (1.0 - theta * theta / 6.0)
        b = duration * (theta / 2.0 - theta**3 / 24.0)
    else:
        a = np.sin(theta) / wz
        b = (1.0 - np.cos(theta)) / wz
    return np.array([a * vx - b * vy, b * vx + a * vy, theta])


def _se2_log(transform: np.ndarray, duration: float) -> np.ndarray:
    x, y, theta = (float(value) for value in transform)
    if abs(theta) < 1e-8:
        a = duration * (1.0 - theta * theta / 6.0)
        b = duration * (theta / 2.0 - theta**3 / 24.0)
    else:
        wz = theta / duration
        a = np.sin(theta) / wz
        b = (1.0 - np.cos(theta)) / wz
    denominator = a * a + b * b
    return np.array(
        [
            (a * x + b * y) / denominator,
            (-b * x + a * y) / denominator,
            theta / duration,
        ]
    )


def _integrate_twists(twists: np.ndarray, duration: float = 1.0 / 30.0) -> np.ndarray:
    pose = np.zeros(3, dtype=np.float64)
    for twist in twists:
        pose = _se2_compose(pose, _se2_exp(twist, duration))
    return pose


def _resample_accelerated(
    selected: np.ndarray, following: np.ndarray | None, target_ticks: int
) -> tuple[np.ndarray, dict[str, Any]]:
    source = np.repeat(selected, 2, axis=0)
    raw_source_base = source[:, :3].astype(np.float64)
    clipped_source_base = np.clip(raw_source_base, -1.0, 1.0)
    source_twists = clipped_source_base * BASE_PHYSICAL_SCALE
    output_twists: list[np.ndarray] = []
    source_ticks = len(source_twists)
    for target_index in range(target_ticks):
        start = target_index * source_ticks / target_ticks
        end = (target_index + 1) * source_ticks / target_ticks
        relative = np.zeros(3, dtype=np.float64)
        first = int(np.floor(start))
        last = int(np.ceil(end))
        for source_index in range(first, last):
            overlap = max(0.0, min(end, source_index + 1.0) - max(start, source_index))
            if overlap:
                relative = _se2_compose(
                    relative,
                    _se2_exp(
                        source_twists[min(source_index, source_ticks - 1)],
                        overlap / 30.0,
                    ),
                )
        output_twists.append(_se2_log(relative, 1.0 / 30.0))
    output_twists_array = np.stack(output_twists)
    raw_output_base = output_twists_array / BASE_PHYSICAL_SCALE
    clipped_output_base = np.clip(raw_output_base, -1.0, 1.0)

    ticks = np.empty((target_ticks, ACTION_DIM), dtype=selected.dtype)
    source_times = (
        np.arange(target_ticks, dtype=np.float64) * source_ticks / target_ticks
    )
    token_coordinates = source_times / 2.0
    continuous_dims = [
        dim
        for dim in range(ACTION_DIM)
        if dim not in (*range(3), ACTION_CONSTANT_DIM, *ACTION_GRIPPER_DIMS)
    ]
    if following is None:
        if token_coordinates[-1] > len(selected) - 1:
            raise RuntimeError(
                "Accelerated resampling without lookahead exceeded selected tokens."
            )
        control_points = selected
    else:
        control_points = np.concatenate((selected, following[None]), axis=0)
    for dim in continuous_dims:
        ticks[:, dim] = np.interp(
            token_coordinates,
            np.arange(len(control_points), dtype=np.float64),
            control_points[:, dim],
        )
    source_token_indices = np.minimum(
        np.floor(token_coordinates).astype(np.int64), len(selected) - 1
    )
    for dim in ACTION_GRIPPER_DIMS:
        ticks[:, dim] = selected[source_token_indices, dim]
    ticks[:, ACTION_CONSTANT_DIM] = 0
    ticks[:, :3] = clipped_output_base.astype(selected.dtype)

    source_endpoint = _integrate_twists(source_twists)
    preclip_endpoint = _integrate_twists(output_twists_array)
    output_endpoint = _integrate_twists(clipped_output_base * BASE_PHYSICAL_SCALE)
    preclip_delta = preclip_endpoint - source_endpoint
    preclip_delta[2] = np.arctan2(np.sin(preclip_delta[2]), np.cos(preclip_delta[2]))
    postclip_delta = output_endpoint - source_endpoint
    postclip_delta[2] = np.arctan2(np.sin(postclip_delta[2]), np.cos(postclip_delta[2]))
    excess = np.maximum(np.abs(raw_output_base) - 1.0, 0.0)
    diagnostics = {
        "source_preclip_count": int(np.count_nonzero(np.abs(raw_source_base) > 1.0)),
        "source_max_preclip_abs": float(np.max(np.abs(raw_source_base))),
        "preclip_saturation_count": int(np.count_nonzero(excess > 0.0)),
        "preclip_saturation_token_count": int(
            np.count_nonzero(np.any(excess > 0.0, axis=1))
        ),
        "max_preclip_abs": float(np.max(np.abs(raw_output_base))),
        "max_preclip_excess": float(np.max(excess)),
        "se2_preclip_endpoint_translation_error": float(
            np.linalg.norm(preclip_delta[:2])
        ),
        "se2_preclip_endpoint_yaw_error": float(abs(preclip_delta[2])),
        "se2_postclip_endpoint_translation_error": float(
            np.linalg.norm(postclip_delta[:2])
        ),
        "se2_postclip_endpoint_yaw_error": float(abs(postclip_delta[2])),
        "qpos_output_finite": bool(np.isfinite(ticks[:, QPOS_DIMS]).all()),
        "qpos_output_min": float(np.min(ticks[:, QPOS_DIMS])),
        "qpos_output_max": float(np.max(ticks[:, QPOS_DIMS])),
    }
    return ticks, diagnostics


def _adaptive_repeats(
    selected: np.ndarray,
    lookahead: np.ndarray,
    config: B1KExecutionConfig,
    state: B1KExecutionState,
) -> tuple[np.ndarray, B1KExecutionState, dict[str, Any]]:
    base = np.concatenate((selected[:, :3], lookahead[None, :3]), axis=0)
    scores = np.maximum(
        np.linalg.norm(base[:, :2], axis=1) / 0.35, np.abs(base[:, 2]) / 0.30
    )
    moving = bool(state.adaptive_moving)
    low_streak = int(state.adaptive_low_streak)
    start_moving = moving
    repeats: list[int] = []
    transitions = 0
    for index, score in enumerate(scores[:-1]):
        previous = moving
        if (
            score > config.adaptive_high_threshold
            or scores[index + 1] > config.adaptive_high_threshold
        ):
            moving = True
            low_streak = 0
        elif moving:
            if score <= config.adaptive_low_threshold:
                low_streak += 1
                if low_streak >= config.adaptive_low_tokens:
                    moving = False
                    low_streak = 0
            else:
                low_streak = 0
        transitions += int(previous != moving)
        repeats.append(2 if moving else 1)
    repeats_array = np.asarray(repeats, dtype=np.int64)
    next_state = B1KExecutionState(moving, low_streak)
    diagnostics = {
        "start_state": {
            "moving": start_moving,
            "low_streak": int(state.adaptive_low_streak),
        },
        "end_state": {"moving": moving, "low_streak": low_streak},
        "moving_tokens": int(np.count_nonzero(repeats_array == 2)),
        "stationary_tokens": int(np.count_nonzero(repeats_array == 1)),
        "state_transitions": transitions,
        "score_threshold_counts": {
            str(threshold): int(np.count_nonzero(scores[:-1] <= threshold))
            for threshold in ADAPTIVE_SCORE_THRESHOLDS
        },
        "adaptive_score_max": float(np.max(scores[:-1])),
        "token_repeats": repeats_array.tolist(),
    }
    return repeats_array, next_state, diagnostics


def _validate_action_chunk(
    action_chunk: torch.Tensor | np.ndarray,
    config: B1KExecutionConfig,
) -> None:
    if not isinstance(action_chunk, (torch.Tensor, np.ndarray)):
        raise TypeError(
            "action_chunk must be a torch.Tensor or numpy.ndarray, "
            f"got {type(action_chunk)!r}."
        )
    if action_chunk.ndim < 2:
        raise ValueError(
            "action_chunk must have shape [..., tokens, 23], "
            f"got {tuple(action_chunk.shape)}."
        )
    if action_chunk.shape[-1] != ACTION_DIM:
        raise ValueError(
            f"action_chunk must have action dimension {ACTION_DIM}, "
            f"got shape {tuple(action_chunk.shape)}."
        )
    if action_chunk.shape[-2] != config.prediction_horizon:
        raise ValueError(
            "action_chunk token dimension must equal prediction_horizon "
            f"{config.prediction_horizon}, got shape {tuple(action_chunk.shape)}."
        )
    if isinstance(action_chunk, torch.Tensor):
        finite = bool(torch.isfinite(action_chunk).all().item())
    else:
        finite = bool(np.isfinite(action_chunk).all())
    if not finite:
        raise ValueError("action_chunk contains non-finite values.")


def prepare_b1k_execution(
    action_chunk: torch.Tensor | np.ndarray,
    config: B1KExecutionConfig | None = None,
    *,
    chunk_index: int = 0,
    state: B1KExecutionState | None = None,
) -> B1KExecutionBatch:
    """Convert a denormalized policy chunk into native-rate B1K commands.

    The first ``execute_horizon`` policy tokens are postprocessed and converted
    to native-rate ticks according to ``temporal_strategy``. Accelerated
    strategies time-resample absolute targets and the integrated SE(2) base path.
    """

    config = config or B1KExecutionConfig()
    if not isinstance(config, B1KExecutionConfig):
        raise TypeError(f"config must be B1KExecutionConfig, got {type(config)!r}.")
    if (
        isinstance(chunk_index, bool)
        or not isinstance(chunk_index, int)
        or chunk_index < 0
    ):
        raise ValueError(
            f"chunk_index must be a non-negative integer, got {chunk_index!r}."
        )
    _validate_action_chunk(action_chunk, config)
    if state is not None and not isinstance(state, B1KExecutionState):
        raise TypeError(f"state must be B1KExecutionState, got {type(state)!r}.")
    state = state or B1KExecutionState()
    if config.temporal_strategy != "baseline" and action_chunk.ndim != 2:
        raise ValueError(
            f"temporal_strategy {config.temporal_strategy!r} only supports an unbatched "
            f"[{config.prediction_horizon}, 23] action chunk"
        )

    selected = postprocess_b1k_action(
        action_chunk[..., : config.execute_horizon, :], clone=True
    )
    diagnostics: dict[str, Any] = {}
    next_state = B1KExecutionState(state.adaptive_moving, state.adaptive_low_streak)
    if config.temporal_strategy in ACCELERATED_STRATEGIES:
        selected_numpy = (
            selected.detach().cpu().numpy()
            if isinstance(selected, torch.Tensor)
            else selected
        )
        following_numpy = None
        if not config.uses_full_horizon_speed_2x:
            following = postprocess_b1k_action(
                action_chunk[config.execute_horizon], clone=True
            )
            following_numpy = (
                following.detach().cpu().numpy()
                if isinstance(following, torch.Tensor)
                else following
            )
        target_ticks = (
            config.execute_horizon
            if config.temporal_strategy == "speed_2x"
            else FIXED_ACCELERATED_TARGET_TICKS[config.temporal_strategy]
        )
        ticks_numpy, diagnostics = _resample_accelerated(
            selected_numpy, following_numpy, target_ticks
        )
        ticks = (
            torch.as_tensor(ticks_numpy, device=selected.device, dtype=selected.dtype)
            if isinstance(selected, torch.Tensor)
            else ticks_numpy
        )
    elif config.temporal_strategy == "adaptive_upper_body_2x":
        selected_numpy = (
            selected.detach().cpu().numpy()
            if isinstance(selected, torch.Tensor)
            else selected
        )
        lookahead = postprocess_b1k_action(
            action_chunk[config.execute_horizon], clone=True
        )
        lookahead_numpy = (
            lookahead.detach().cpu().numpy()
            if isinstance(lookahead, torch.Tensor)
            else lookahead
        )
        repeats, next_state, diagnostics = _adaptive_repeats(
            selected_numpy, lookahead_numpy, config, state
        )
        if isinstance(selected, torch.Tensor):
            repeat_tensor = torch.as_tensor(repeats, device=selected.device)
            ticks = selected.repeat_interleave(repeat_tensor, dim=-2)
        else:
            ticks = np.repeat(selected, repeats, axis=-2)
    elif config.resampling == "zero_order_hold":
        if isinstance(selected, torch.Tensor):
            ticks = selected.repeat_interleave(config.ticks_per_policy_token, dim=-2)
        else:
            ticks = np.repeat(selected, config.ticks_per_policy_token, axis=-2)
    else:
        following = postprocess_b1k_action(
            action_chunk[..., 1 : config.execute_horizon + 1, :], clone=True
        )
        if isinstance(selected, torch.Tensor):
            alpha = (
                torch.arange(
                    config.ticks_per_policy_token,
                    device=selected.device,
                    dtype=selected.dtype,
                )
                / config.ticks_per_policy_token
            )
            ticks = selected.unsqueeze(-2) + alpha.view(
                *((1,) * (selected.ndim - 1)), -1, 1
            ) * (following - selected).unsqueeze(-2)
            ticks[..., ACTION_GRIPPER_DIMS[0]] = selected[
                ..., ACTION_GRIPPER_DIMS[0]
            ].unsqueeze(-1)
            ticks[..., ACTION_GRIPPER_DIMS[1]] = selected[
                ..., ACTION_GRIPPER_DIMS[1]
            ].unsqueeze(-1)
            ticks = ticks.flatten(start_dim=-3, end_dim=-2)
        else:
            alpha = (
                np.arange(config.ticks_per_policy_token, dtype=selected.dtype)
                / config.ticks_per_policy_token
            )
            ticks = (
                selected[..., :, None, :]
                + alpha.reshape(*((1,) * (selected.ndim - 1)), -1, 1)
                * (following - selected)[..., :, None, :]
            )
            ticks[..., ACTION_GRIPPER_DIMS[0]] = selected[
                ..., ACTION_GRIPPER_DIMS[0], None
            ]
            ticks[..., ACTION_GRIPPER_DIMS[1]] = selected[
                ..., ACTION_GRIPPER_DIMS[1], None
            ]
            ticks = ticks.reshape(*selected.shape[:-2], config.output_ticks, ACTION_DIM)

    actual_output_ticks = int(ticks.shape[-2])
    expected_shape = (*action_chunk.shape[:-2], actual_output_ticks, ACTION_DIM)
    if tuple(ticks.shape) != expected_shape:
        raise RuntimeError(
            f"Internal B1K execution shape error: expected {expected_shape}, "
            f"got {tuple(ticks.shape)}."
        )
    metadata = config.metadata()
    metadata["chunk_index"] = chunk_index
    metadata["output_ticks"] = actual_output_ticks
    metadata["input_shape"] = list(action_chunk.shape)
    metadata["output_shape"] = list(ticks.shape)
    metadata.update(diagnostics)
    return B1KExecutionBatch(ticks=ticks, metadata=metadata, next_state=next_state)


class B1KExecutionWrapper:
    """Callable wrapper suitable for a closed-loop policy client."""

    def __init__(self, config: B1KExecutionConfig | None = None) -> None:
        self.config = config or B1KExecutionConfig()
        if not isinstance(self.config, B1KExecutionConfig):
            raise TypeError(
                f"config must be B1KExecutionConfig, got {type(self.config)!r}."
            )
        self._state = B1KExecutionState()

    @property
    def metadata(self) -> dict[str, Any]:
        return self.config.metadata()

    @property
    def state(self) -> B1KExecutionState:
        return B1KExecutionState(
            self._state.adaptive_moving, self._state.adaptive_low_streak
        )

    def reset(self) -> None:
        self._state = B1KExecutionState()

    def __call__(
        self, action_chunk: torch.Tensor | np.ndarray, *, chunk_index: int = 0
    ) -> B1KExecutionBatch:
        batch = prepare_b1k_execution(
            action_chunk, self.config, chunk_index=chunk_index, state=self._state
        )
        self._state = batch.next_state
        return batch
