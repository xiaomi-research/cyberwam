# Copyright (c) 2026 Xiaomi Corporation

from __future__ import annotations

from collections.abc import Sequence

import torch

from .contract import (
    ACTION_CONSTANT_DIM,
    ACTION_DIM,
    ACTION_GRIPPER_DIMS,
    PROPRIO_DIM,
)


DEFAULT_CLAMP_VALUE = 5.0
DEFAULT_EPS = 1e-8


class B1KMixedNormalizer:
    """Apply the B1K action/proprio normalization contract.

    Action and proprio statistics are deliberately independent. Action gripper
    commands and the constant action dimension bypass z-score normalization.
    Continuous actions are intentionally left unclamped so training and
    inference preserve the tails of the action distribution. Proprioception is
    clamped to protect low-variance state dimensions from extreme values.
    """

    def __init__(
        self,
        *,
        action_mean: torch.Tensor | Sequence[float],
        action_std: torch.Tensor | Sequence[float],
        proprio_mean: torch.Tensor | Sequence[float],
        proprio_std: torch.Tensor | Sequence[float],
        clamp_value: float = DEFAULT_CLAMP_VALUE,
        eps: float = DEFAULT_EPS,
    ) -> None:
        self.action_mean = _validate_stat(action_mean, "action_mean", ACTION_DIM)
        self.action_std = _validate_stat(action_std, "action_std", ACTION_DIM, is_std=True)
        self.proprio_mean = _validate_stat(proprio_mean, "proprio_mean", PROPRIO_DIM)
        self.proprio_std = _validate_stat(
            proprio_std, "proprio_std", PROPRIO_DIM, is_std=True
        )

        if not isinstance(clamp_value, (int, float)) or not torch.isfinite(
            torch.tensor(float(clamp_value))
        ):
            raise ValueError(f"clamp_value must be finite, got {clamp_value!r}.")
        if clamp_value <= 0:
            raise ValueError(f"clamp_value must be positive, got {clamp_value!r}.")
        if not isinstance(eps, (int, float)) or not torch.isfinite(
            torch.tensor(float(eps))
        ):
            raise ValueError(f"eps must be finite, got {eps!r}.")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps!r}.")

        self.clamp_value = float(clamp_value)
        self.eps = float(eps)

    def normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        """Z-score a raw 23D action without clipping continuous dimensions."""
        _validate_input(action, "action", ACTION_DIM)
        mean, std = self._action_stats_like(action)
        normalized = (action.to(mean.dtype) - mean) / (std + self.eps)
        normalized[..., ACTION_CONSTANT_DIM] = 0
        for dim in ACTION_GRIPPER_DIMS:
            normalized[..., dim] = _binary_threshold(action[..., dim])
        return normalized.to(action.dtype)

    def denormalize_action(self, normalized_action: torch.Tensor) -> torch.Tensor:
        """Invert action z-score without clipping continuous dimensions."""
        _validate_input(normalized_action, "normalized_action", ACTION_DIM)
        mean, std = self._action_stats_like(normalized_action)
        action = normalized_action.to(mean.dtype) * (std + self.eps) + mean
        action[..., ACTION_CONSTANT_DIM] = 0
        for dim in ACTION_GRIPPER_DIMS:
            action[..., dim] = _binary_threshold(normalized_action[..., dim])
        return action.to(normalized_action.dtype)

    def normalize_proprio(self, proprio: torch.Tensor) -> torch.Tensor:
        """Z-score and clamp every dimension of a 23D proprio tensor."""
        _validate_input(proprio, "proprio", PROPRIO_DIM)
        mean, std = self._proprio_stats_like(proprio)
        normalized = ((proprio.to(mean.dtype) - mean) / (std + self.eps)).clamp(
            -self.clamp_value, self.clamp_value
        )
        return normalized.to(proprio.dtype)

    def denormalize_proprio(self, normalized_proprio: torch.Tensor) -> torch.Tensor:
        """Invert proprio z-score values; callers control any output clipping."""
        _validate_input(normalized_proprio, "normalized_proprio", PROPRIO_DIM)
        mean, std = self._proprio_stats_like(normalized_proprio)
        proprio = normalized_proprio.to(mean.dtype) * (std + self.eps) + mean
        return proprio.to(normalized_proprio.dtype)

    def _action_stats_like(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return _stats_like(self.action_mean, self.action_std, value)

    def _proprio_stats_like(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return _stats_like(self.proprio_mean, self.proprio_std, value)


def _validate_stat(
    value: torch.Tensor | Sequence[float],
    name: str,
    expected_dim: int,
    *,
    is_std: bool = False,
) -> torch.Tensor:
    try:
        tensor = torch.as_tensor(value, dtype=torch.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a numeric tensor or sequence.") from exc
    if tensor.shape != (expected_dim,):
        raise ValueError(
            f"{name} must have shape ({expected_dim},), got {tuple(tensor.shape)}."
        )
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"{name} must contain only finite values.")
    if is_std and bool((tensor < 0).any().item()):
        raise ValueError(f"{name} must contain non-negative values.")
    return tensor.clone()


def _validate_input(value: torch.Tensor, name: str, expected_dim: int) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value)!r}.")
    if value.ndim == 0 or value.shape[-1] != expected_dim:
        raise ValueError(
            f"{name} must have last dimension {expected_dim}, got shape {tuple(value.shape)}."
        )
    if not value.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype, got {value.dtype}.")
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} must contain only finite values.")


def _stats_like(
    mean: torch.Tensor, std: torch.Tensor, value: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    dtype = torch.float32 if value.dtype in (torch.float16, torch.bfloat16) else value.dtype
    return (
        mean.to(device=value.device, dtype=dtype),
        std.to(device=value.device, dtype=dtype),
    )


def _binary_threshold(value: torch.Tensor) -> torch.Tensor:
    return torch.where(value >= 0, torch.ones_like(value), -torch.ones_like(value))
