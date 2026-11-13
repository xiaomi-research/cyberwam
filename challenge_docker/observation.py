# Copyright (c) 2026 Xiaomi Corporation

"""Pure observation adaptation for the official BEHAVIOR websocket API.

The simulator sends a flattened dictionary.  FastWAM trains on a canonical
23-dimensional B1K state, so this module owns the only conversion between the
official R1Pro observation and that training contract.  It deliberately does
not inspect simulator state, object poses, or task metadata beyond ``task_id``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


PROPRIO_DIM = 23
ACTION_DIM = 23


def _array(value: Any, name: str, *, allow_nonfinite: bool = False) -> np.ndarray:
    result = np.asarray(value)
    if not np.issubdtype(result.dtype, np.number):
        raise TypeError(f"{name} must be numeric, got {result.dtype}.")
    result = result.astype(np.float32, copy=False)
    if not allow_nonfinite and not np.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite values.")
    return result


def collapse_proprio(value: Any) -> np.ndarray:
    """Convert the challenge robot vector (or a B1K raw state) to 23D.

    The supplied challenge robot config exposes ``base_qvel``, trunk qpos,
    both arm qpos, and two-finger gripper qpos.  That is 25 values; each pair
    of finger positions is collapsed to the width used by the B1K dataset.
    A 23D value is already canonical.  A 61D training state is accepted for
    local compatibility and uses the same fixed B1K state slices as training.
    """

    value = _array(value, "proprio").reshape(-1)
    if value.size == 256:
        from fastwam.datasets.b1k.contract import extract_r1pro_simulator_proprio

        return extract_r1pro_simulator_proprio(value)
    if value.size == PROPRIO_DIM:
        return value.copy()
    if value.size == 25:
        return np.concatenate(
            (
                value[0:3],
                value[3:7],
                value[7:14],
                value[14:16].sum(keepdims=True),
                value[16:23],
                value[23:25].sum(keepdims=True),
            )
        ).astype(np.float32, copy=False)
    if value.size == 61:
        return np.concatenate(
            (
                value[0:3],
                value[53:57],
                value[3:10],
                value[24:26].sum(keepdims=True),
                value[28:35],
                value[49:51].sum(keepdims=True),
            )
        ).astype(np.float32, copy=False)
    raise ValueError(
        "Official proprioception must contain 23 canonical, 25 challenge-robot, "
        f"or 61 raw B1K / 256 simulator values; got {value.size}."
    )


def normalize_proprio(raw: Any, mean: Any, std: Any, *, clamp: float = 5.0) -> np.ndarray:
    """Apply the B1K independent per-dimension z-score contract."""

    value = collapse_proprio(raw)
    mean = _array(mean, "proprio mean").reshape(-1)
    std = _array(std, "proprio std").reshape(-1)
    if mean.shape != (PROPRIO_DIM,) or std.shape != (PROPRIO_DIM,):
        raise ValueError("Proprio normalization statistics must each have shape (23,).")
    if (std < 0).any() or clamp <= 0:
        raise ValueError("Proprio standard deviations must be non-negative and clamp positive.")
    return np.clip((value - mean) / (std + 1e-8), -float(clamp), float(clamp)).astype(
        np.float32, copy=False
    )


def _find_modality(
    observation: Mapping[str, Any],
    role: str,
    modality: str,
) -> tuple[str, np.ndarray]:
    """Find a camera modality without depending on one sensor-name spelling."""

    exact = [
        key
        for key in observation
        if str(key).endswith(f"::{modality}")
        and (
            (role == "head" and any(token in str(key).lower() for token in ("zed", "head")))
            or (role == "left_wrist" and any(token in str(key).lower() for token in ("left_realsense", "left_wrist")))
            or (role == "right_wrist" and any(token in str(key).lower() for token in ("right_realsense", "right_wrist")))
        )
    ]
    if len(exact) != 1:
        raise KeyError(
            f"Expected exactly one {modality} observation for {role}, found {exact}."
        )
    # No-hit depth is sanitized by the original FLEX pointmap projection.
    return str(exact[0]), _array(
        observation[exact[0]], f"{role} {modality}",
        allow_nonfinite=modality in ("depth", "depth_linear"),
    )


def adapt_observation(
    observation: Mapping[str, Any],
    *,
    task_lookup: Mapping[int, str],
    proprio_mean: Any,
    proprio_std: Any,
    require_depth: bool,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, str]:
    """Adapt one official flattened observation for the FastWAM policy.

    Returns ``(policy_observation, raw_proprio, normalized_proprio, task)``.
    The policy observation contains only RGB, optional depth, task text, and
    the canonical state; no simulator-only fields are forwarded.
    """

    if not isinstance(observation, Mapping):
        raise TypeError("Websocket observation must be a mapping.")
    proprio_key = next((key for key in observation if str(key).endswith("::proprio")), None)
    if proprio_key is None:
        raise KeyError("Official observation is missing a robot ::proprio field.")
    raw_proprio = collapse_proprio(observation[proprio_key])
    normalized = normalize_proprio(raw_proprio, proprio_mean, proprio_std)

    task_id_value = observation.get("task_id")
    if task_id_value is None:
        raise KeyError("Official observation is missing task_id.")
    task_id = int(np.asarray(task_id_value).reshape(-1)[0])
    try:
        task = str(task_lookup[task_id])
    except KeyError as exc:
        raise KeyError(f"Unknown BEHAVIOR task_id {task_id}.") from exc

    images: dict[str, np.ndarray] = {}
    depths: dict[str, np.ndarray] = {}
    for role in ("head", "left_wrist", "right_wrist"):
        _, images[role] = _find_modality(observation, role, "rgb")
        if require_depth:
            try:
                _, depths[role] = _find_modality(observation, role, "depth_linear")
            except KeyError:
                _, depths[role] = _find_modality(observation, role, "depth")

    policy_observation: dict[str, Any] = {
        "task": task,
        "task_id": task_id,
        "proprio": raw_proprio,
        "images": images,
    }
    if require_depth:
        policy_observation["depths"] = depths
    return policy_observation, raw_proprio, normalized, task
