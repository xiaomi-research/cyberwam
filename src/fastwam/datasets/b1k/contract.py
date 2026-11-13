# Copyright (c) 2026 Xiaomi Corporation

from __future__ import annotations

from typing import Any

import numpy as np
import torch


ACTION_DIM = 23
RAW_STATE_DIM = 61
R1PRO_SIMULATOR_STATE_DIM = 256
PROPRIO_DIM = 23

ACTION_CONSTANT_DIM = 6
ACTION_GRIPPER_DIMS = (14, 22)

ACTION_LABELS = (
    "base_velocity_0",
    "base_velocity_1",
    "base_velocity_2",
    "trunk_absolute_qpos_0",
    "trunk_absolute_qpos_1",
    "trunk_absolute_qpos_2",
    "trunk_absolute_qpos_3",
    *(f"left_arm_absolute_qpos_{idx}" for idx in range(7)),
    "left_gripper_command",
    *(f"right_arm_absolute_qpos_{idx}" for idx in range(7)),
    "right_gripper_command",
)

PROPRIO_LABELS = (
    "base_qvel_0",
    "base_qvel_1",
    "base_qvel_2",
    *(f"trunk_qpos_{idx}" for idx in range(4)),
    *(f"left_arm_qpos_{idx}" for idx in range(7)),
    "left_gripper_width",
    *(f"right_arm_qpos_{idx}" for idx in range(7)),
    "right_gripper_width",
)

assert len(ACTION_LABELS) == ACTION_DIM
assert len(PROPRIO_LABELS) == PROPRIO_DIM


def _shape(value: Any) -> tuple[int, ...]:
    try:
        return tuple(value.shape)
    except AttributeError as exc:
        raise TypeError(f"Expected a torch.Tensor or numpy.ndarray, got {type(value)!r}.") from exc


def _validate_last_dim(value: Any, expected: int, name: str) -> None:
    shape = _shape(value)
    if not shape or shape[-1] != expected:
        raise ValueError(f"{name} must have last dimension {expected}, got shape {shape}.")


def _is_finite(value: torch.Tensor | np.ndarray) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all().item())
    if isinstance(value, np.ndarray):
        return bool(np.isfinite(value).all())
    raise TypeError(f"Expected a torch.Tensor or numpy.ndarray, got {type(value)!r}.")


def validate_b1k_raw_state(raw_state: torch.Tensor | np.ndarray) -> None:
    _validate_last_dim(raw_state, RAW_STATE_DIM, "B1K raw state")
    if not _is_finite(raw_state):
        raise ValueError("B1K raw state contains non-finite values.")


def validate_b1k_action(
    action: torch.Tensor | np.ndarray,
    *,
    constant_atol: float = 1e-6,
    require_binary_gripper: bool = True,
) -> None:
    _validate_last_dim(action, ACTION_DIM, "B1K action")
    if not _is_finite(action):
        raise ValueError("B1K action contains non-finite values.")

    constant = action[..., ACTION_CONSTANT_DIM]
    if isinstance(action, torch.Tensor):
        constant_ok = bool((constant.abs() <= constant_atol).all().item())
    else:
        constant_ok = bool((np.abs(constant) <= constant_atol).all())
    if not constant_ok:
        raise ValueError(
            f"B1K action dimension {ACTION_CONSTANT_DIM} must be constant zero "
            f"within atol={constant_atol}."
        )

    if not require_binary_gripper:
        return
    for dim in ACTION_GRIPPER_DIMS:
        gripper = action[..., dim]
        if isinstance(action, torch.Tensor):
            valid = torch.logical_or(gripper == -1, gripper == 1)
            valid_all = bool(valid.all().item())
        else:
            valid_all = bool(np.logical_or(gripper == -1, gripper == 1).all())
        if not valid_all:
            raise ValueError(f"B1K action gripper dimension {dim} must contain only -1 or +1.")


def extract_b1k_proprio(raw_state: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    validate_b1k_raw_state(raw_state)
    parts = (
        raw_state[..., 0:3],
        raw_state[..., 53:57],
        raw_state[..., 3:10],
        raw_state[..., 24:26].sum(axis=-1, keepdims=True),
        raw_state[..., 28:35],
        raw_state[..., 49:51].sum(axis=-1, keepdims=True),
    )
    if isinstance(raw_state, torch.Tensor):
        proprio = torch.cat(parts, dim=-1)
    else:
        proprio = np.concatenate(parts, axis=-1)
    _validate_last_dim(proprio, PROPRIO_DIM, "B1K proprio")
    return proprio


def extract_r1pro_simulator_proprio(
    simulator_state: torch.Tensor | np.ndarray,
) -> torch.Tensor | np.ndarray:
    _validate_last_dim(
        simulator_state, R1PRO_SIMULATOR_STATE_DIM, "R1Pro simulator state"
    )
    if not _is_finite(simulator_state):
        raise ValueError("R1Pro simulator state contains non-finite values.")
    parts = (
        simulator_state[..., 253:256],
        simulator_state[..., 236:240],
        simulator_state[..., 158:165],
        simulator_state[..., 193:195].sum(axis=-1, keepdims=True),
        simulator_state[..., 197:204],
        simulator_state[..., 232:234].sum(axis=-1, keepdims=True),
    )
    if isinstance(simulator_state, torch.Tensor):
        proprio = torch.cat(parts, dim=-1)
    else:
        proprio = np.concatenate(parts, axis=-1)
    _validate_last_dim(proprio, PROPRIO_DIM, "B1K proprio")
    return proprio


def postprocess_b1k_action(
    action: torch.Tensor | np.ndarray,
    *,
    clone: bool = True,
) -> torch.Tensor | np.ndarray:
    _validate_last_dim(action, ACTION_DIM, "B1K predicted action")
    if not _is_finite(action):
        raise ValueError("B1K predicted action contains non-finite values.")

    if isinstance(action, torch.Tensor):
        result = action.clone() if clone else action
        result[..., ACTION_CONSTANT_DIM] = 0
        for dim in ACTION_GRIPPER_DIMS:
            result[..., dim] = torch.where(
                result[..., dim] >= 0,
                torch.ones_like(result[..., dim]),
                -torch.ones_like(result[..., dim]),
            )
        return result

    if isinstance(action, np.ndarray):
        result = action.copy() if clone else action
        result[..., ACTION_CONSTANT_DIM] = 0
        for dim in ACTION_GRIPPER_DIMS:
            result[..., dim] = np.where(result[..., dim] >= 0, 1, -1)
        return result
    raise TypeError(f"Expected a torch.Tensor or numpy.ndarray, got {type(action)!r}.")


class B1KActionStateTransform:
    """Hydra-instantiable B1K action/state contract transform."""

    def __init__(
        self,
        action_key: str = "default",
        state_key: str = "default",
        constant_atol: float = 1e-6,
    ) -> None:
        self.action_key = action_key
        self.state_key = state_key
        self.constant_atol = constant_atol

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        if "state" not in batch or self.state_key not in batch["state"]:
            raise KeyError(f"Missing B1K state key {self.state_key!r}.")
        batch["state"][self.state_key] = extract_b1k_proprio(batch["state"][self.state_key])

        if "action" in batch:
            if self.action_key not in batch["action"]:
                raise KeyError(f"Missing B1K action key {self.action_key!r}.")
            validate_b1k_action(
                batch["action"][self.action_key],
                constant_atol=self.constant_atol,
                require_binary_gripper=True,
            )
        return batch

    def backward(self, batch: dict[str, Any]) -> dict[str, Any]:
        if "action" in batch:
            if self.action_key not in batch["action"]:
                raise KeyError(f"Missing B1K action key {self.action_key!r}.")
            batch["action"][self.action_key] = postprocess_b1k_action(
                batch["action"][self.action_key], clone=False
            )
        return batch
