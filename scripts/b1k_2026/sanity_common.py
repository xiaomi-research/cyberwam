# Copyright (c) 2026 Xiaomi Corporation

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from fastwam.datasets.b1k.contract import ACTION_DIM, PROPRIO_DIM
from fastwam.datasets.b1k.normalizer import B1KMixedNormalizer
from fastwam.utils.checkpoint_contract import CheckpointContract, read_checkpoint_contract


def import_callable(spec: str) -> Callable[..., Any]:
    if ":" not in spec:
        raise ValueError(f"Plugin must use 'module:callable' syntax, got {spec!r}.")
    module_name, attribute = spec.rsplit(":", 1)
    value = getattr(importlib.import_module(module_name), attribute)
    if not callable(value):
        raise TypeError(f"Plugin {spec!r} is not callable.")
    return value


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    temporary.replace(output)
    return output


def load_normalizer(stats_path: str | Path) -> tuple[B1KMixedNormalizer, dict[str, Any]]:
    path = Path(stats_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    try:
        action = payload["stats"]["action"]["default"]
        proprio = payload["stats"]["state"]["default"]
        action_mean = action["global_mean"]
        action_std = action["global_std"]
        proprio_mean = proprio["global_mean"]
        proprio_std = proprio["global_std"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Invalid B1K norm stats schema in {path}.") from exc
    if len(action_mean) != ACTION_DIM or len(action_std) != ACTION_DIM:
        raise ValueError(f"B1K action stats in {path} must be {ACTION_DIM}D.")
    if len(proprio_mean) != PROPRIO_DIM or len(proprio_std) != PROPRIO_DIM:
        raise ValueError(f"B1K proprio stats in {path} must be {PROPRIO_DIM}D.")
    normalizer = B1KMixedNormalizer(
        action_mean=action_mean,
        action_std=action_std,
        proprio_mean=proprio_mean,
        proprio_std=proprio_std,
    )
    return normalizer, payload


def load_contract(path: str | Path) -> CheckpointContract:
    return read_checkpoint_contract(Path(path).expanduser().resolve())


def assert_stats_match_contract(
    stats_path: str | Path, contract: CheckpointContract
) -> str:
    path = Path(stats_path).expanduser().resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        expected = contract.compatibility["data"]["normalization"]["stats_sha256"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Checkpoint contract is missing data.normalization.stats_sha256.") from exc
    if expected != digest:
        raise ValueError(
            f"Norm stats hash does not match checkpoint contract: expected {expected}, got {digest}."
        )
    return digest


def tensor_tree_to_numpy(value: Any) -> dict[str, np.ndarray]:
    leaves: dict[str, np.ndarray] = {}

    def visit(item: Any, prefix: str) -> None:
        if isinstance(item, torch.Tensor):
            leaves[prefix or "output"] = item.detach().cpu().numpy()
        elif isinstance(item, np.ndarray):
            leaves[prefix or "output"] = item
        elif isinstance(item, dict):
            for key in sorted(item):
                visit(item[key], f"{prefix}.{key}" if prefix else str(key))
        elif isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                visit(child, f"{prefix}.{index}" if prefix else str(index))
        elif isinstance(item, (float, int, bool, np.number)):
            leaves[prefix or "output"] = np.asarray(item)
        else:
            raise TypeError(
                f"Probe output leaf {prefix or '<root>'} has unsupported type {type(item)!r}."
            )

    visit(value, "")
    if not leaves:
        raise ValueError("Probe returned no numeric outputs.")
    for name, array in leaves.items():
        if not np.issubdtype(array.dtype, np.number) and array.dtype != np.bool_:
            raise TypeError(f"Probe output {name!r} is not numeric: {array.dtype}.")
        if not np.isfinite(array).all():
            raise ValueError(f"Probe output {name!r} contains non-finite values.")
    return leaves


def compare_tensor_trees(
    expected: Any,
    actual: Any,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    left = tensor_tree_to_numpy(expected)
    right = tensor_tree_to_numpy(actual)
    if set(left) != set(right):
        raise ValueError(
            f"Probe output structure changed: expected {sorted(left)}, got {sorted(right)}."
        )
    per_leaf: dict[str, Any] = {}
    all_close = True
    for name in sorted(left):
        if left[name].shape != right[name].shape:
            raise ValueError(
                f"Probe output {name!r} shape changed: {left[name].shape} != {right[name].shape}."
            )
        delta = np.abs(left[name].astype(np.float64) - right[name].astype(np.float64))
        close = bool(np.allclose(left[name], right[name], atol=atol, rtol=rtol))
        all_close = all_close and close
        per_leaf[name] = {
            "shape": list(left[name].shape),
            "max_abs_error": float(delta.max(initial=0.0)),
            "mean_abs_error": float(delta.mean()) if delta.size else 0.0,
            "allclose": close,
        }
    return {"allclose": all_close, "atol": atol, "rtol": rtol, "leaves": per_leaf}
