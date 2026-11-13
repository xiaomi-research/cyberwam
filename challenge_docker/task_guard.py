#!/usr/bin/env python3
# Copyright (c) 2026 Xiaomi Corporation

"""Optional task and batch guards for directory-mounted policy bundles."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np


_cached_path: Path | None | object = object()
_cached_mtime_ns: int | None = None
_cached_config: dict[str, Any] | None = None


def _load_config() -> dict[str, Any] | None:
    global _cached_path, _cached_mtime_ns, _cached_config
    raw = os.environ.get("FASTWAM_MODEL_CONFIG", "").strip()
    path = Path(raw).expanduser() if raw else None
    if path is None:
        _cached_path = None
        _cached_mtime_ns = None
        _cached_config = None
        return None
    try:
        mtime_ns = path.stat().st_mtime_ns
    except OSError as exc:
        raise ValueError(f"model config is unavailable: {path}: {exc}") from exc
    if path != _cached_path or mtime_ns != _cached_mtime_ns:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read model config {path}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"model config must be an object: {path}")
        allowed = value.get("allowed_task_ids")
        if not isinstance(allowed, list) or not allowed:
            raise ValueError(f"model config allowed_task_ids must be non-empty: {path}")
        if any(isinstance(item, bool) or not isinstance(item, int) for item in allowed):
            raise ValueError(f"model config task ids must be integers: {path}")
        batch_size = value.get("batch_size", 1)
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(f"model config batch_size must be positive: {path}")
        _cached_path = path
        _cached_mtime_ns = mtime_ns
        _cached_config = {
            "model_name": str(value.get("model_name", path.parent.name)),
            "allowed_task_ids": frozenset(int(item) for item in allowed),
            "batch_size": int(batch_size),
        }
    return _cached_config


def validate_observation_task(observation: Mapping[str, Any]) -> None:
    config = _load_config()
    if config is None:
        return
    if not isinstance(observation, Mapping):
        raise TypeError("policy observation must be a mapping")
    if "task_id" not in observation:
        raise ValueError("model task guard requires task_id in the observation")
    value = np.asarray(observation["task_id"])
    if value.size != 1 or value.dtype.kind not in "iu":
        raise ValueError("model task guard requires one integer task_id")
    task_id = int(value.reshape(-1)[0])
    allowed = config["allowed_task_ids"]
    if task_id not in allowed:
        listed = ",".join(str(item) for item in sorted(allowed))
        raise ValueError(
            f"model {config['model_name']!r} does not support task_id={task_id}; "
            f"allowed_task_ids=[{listed}]"
        )


def expected_batch_size() -> int | None:
    config = _load_config()
    return None if config is None else int(config["batch_size"])


def guard_metadata() -> dict[str, Any]:
    config = _load_config()
    if config is None:
        return {"enabled": False}
    return {
        "enabled": True,
        "model_name": config["model_name"],
        "allowed_task_ids": sorted(config["allowed_task_ids"]),
        "batch_size": config["batch_size"],
    }
