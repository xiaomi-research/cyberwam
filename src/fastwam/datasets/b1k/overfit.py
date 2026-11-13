# Copyright (c) 2026 Xiaomi Corporation

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .contract import validate_b1k_action


REPORT_SCHEMA = "b1k_2026_overfit_report.v1"


class FixedWindowSubset(Dataset):
    """Repeat a stable set of dataset indices for an overfit experiment."""

    def __init__(
        self,
        dataset: Dataset,
        sample_ids: Sequence[int],
        *,
        repeat: int = 1,
    ) -> None:
        if not isinstance(dataset, Dataset):
            raise TypeError(f"dataset must be a torch Dataset, got {type(dataset)!r}.")
        if isinstance(sample_ids, (str, bytes)) or not isinstance(sample_ids, Sequence):
            raise TypeError("sample_ids must be a sequence of integer dataset indices.")
        normalized_ids = tuple(int(sample_id) for sample_id in sample_ids)
        if not normalized_ids:
            raise ValueError("sample_ids must contain at least one index.")
        if len(normalized_ids) != len(set(normalized_ids)):
            raise ValueError(f"sample_ids must be unique, got {normalized_ids}.")
        invalid = [index for index in normalized_ids if index < 0 or index >= len(dataset)]
        if invalid:
            raise IndexError(
                f"sample_ids outside dataset range [0, {len(dataset)}): {invalid}."
            )
        if isinstance(repeat, bool) or int(repeat) != repeat or repeat <= 0:
            raise ValueError(f"repeat must be a positive integer, got {repeat!r}.")

        self.dataset = dataset
        self.sample_ids = normalized_ids
        self.repeat = int(repeat)

    def __len__(self) -> int:
        return len(self.sample_ids) * self.repeat

    def __getitem__(self, index: int) -> Any:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(f"FixedWindowSubset index {index} outside [0, {len(self)}).")
        return self.dataset[self.sample_ids[index % len(self.sample_ids)]]

    @property
    def checkpoint_data_contract(self) -> Mapping[str, Any]:
        contract = getattr(self.dataset, "checkpoint_data_contract", None)
        if not isinstance(contract, Mapping):
            raise TypeError("Wrapped dataset does not expose checkpoint_data_contract.")
        return contract

    @property
    def processor(self):
        processor = getattr(self.dataset, "processor", None)
        if processor is not None:
            return processor
        legacy_dataset = getattr(self.dataset, "lerobot_dataset", None)
        processor = getattr(legacy_dataset, "processor", None)
        if processor is None:
            raise AttributeError("Wrapped dataset does not expose a processor.")
        return processor

    @property
    def episode_data_index(self) -> Mapping[str, Any]:
        episode_index = getattr(self.dataset, "episode_data_index", None)
        if episode_index is None:
            legacy_dataset = getattr(self.dataset, "lerobot_dataset", None)
            episode_index = getattr(legacy_dataset, "episode_data_index", None)
        if not isinstance(episode_index, Mapping):
            raise AttributeError("Wrapped dataset does not expose episode_data_index.")
        return episode_index

    @property
    def lerobot_dataset(self):
        """Legacy compatibility for callers not yet migrated to top-level fields."""
        legacy_dataset = getattr(self.dataset, "lerobot_dataset", None)
        if legacy_dataset is not None:
            return legacy_dataset
        return self.dataset


def parse_sample_ids(value: str | Sequence[int]) -> tuple[int, ...]:
    """Parse a comma-separated CLI value or an integer sequence."""
    if isinstance(value, str):
        fields = [field.strip() for field in value.split(",") if field.strip()]
        if not fields:
            raise ValueError("sample id string is empty.")
        try:
            return tuple(int(field) for field in fields)
        except ValueError as exc:
            raise ValueError(f"Invalid comma-separated sample ids: {value!r}.") from exc
    if isinstance(value, (bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError("sample ids must be a comma-separated string or integer sequence.")
    return tuple(int(item) for item in value)


def episode_window_ids(
    dataset: Dataset,
    *,
    episode_position: int,
    max_windows: int,
    offset: int = 0,
    stride: int = 1,
) -> tuple[int, ...]:
    """Resolve local sample ids from one selected episode.

    ``episode_position`` addresses the selected split, not the original global
    episode id. This keeps the mapping explicit when the training split is a
    sparse list of global B1K episode ids.
    """
    if episode_position < 0:
        raise ValueError("episode_position must be non-negative.")
    if max_windows <= 0:
        raise ValueError("max_windows must be positive.")
    if offset < 0:
        raise ValueError("offset must be non-negative.")
    if stride <= 0:
        raise ValueError("stride must be positive.")

    episode_index = getattr(dataset, "episode_data_index", None)
    if episode_index is None:
        legacy_dataset = getattr(dataset, "lerobot_dataset", None)
        episode_index = getattr(legacy_dataset, "episode_data_index", None)
    if not isinstance(episode_index, Mapping) or not {"from", "to"} <= set(episode_index):
        raise TypeError("Dataset does not expose top-level or legacy episode_data_index.")
    starts = torch.as_tensor(episode_index["from"]).reshape(-1)
    stops = torch.as_tensor(episode_index["to"]).reshape(-1)
    if episode_position >= starts.numel():
        raise IndexError(
            f"episode_position {episode_position} outside selected episode count {starts.numel()}."
        )
    start = int(starts[episode_position].item()) + offset
    stop = int(stops[episode_position].item())
    if start >= stop:
        raise ValueError(
            f"Episode {episode_position} has no window at offset {offset}: range=[{start}, {stop})."
        )
    return tuple(range(start, min(stop, start + max_windows * stride), stride))


def make_metric_snapshot(
    *,
    loss_video: float,
    loss_action: float,
    denormalized_action_l1: float,
) -> dict[str, float]:
    snapshot = {
        "loss_video": float(loss_video),
        "loss_action": float(loss_action),
        "denormalized_action_l1": float(denormalized_action_l1),
    }
    if not all(math.isfinite(value) for value in snapshot.values()):
        raise ValueError(f"Overfit metrics must be finite, got {snapshot}.")
    return snapshot


def build_overfit_report(
    *,
    initial: Mapping[str, float],
    final: Mapping[str, float],
    sample_ids: Sequence[int],
    max_steps: int,
    action_contract_valid: bool,
    gradient_gate: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    required = {"loss_video", "loss_action", "denormalized_action_l1"}
    if set(initial) != required or set(final) != required:
        raise ValueError(f"initial/final metrics must contain exactly {sorted(required)}.")
    initial_snapshot = make_metric_snapshot(**initial)
    final_snapshot = make_metric_snapshot(**final)
    checks = {
        "finite": True,
        "video_loss_decreased": final_snapshot["loss_video"] < initial_snapshot["loss_video"],
        "action_loss_decreased": final_snapshot["loss_action"] < initial_snapshot["loss_action"],
        "denormalized_action_l1_decreased": (
            final_snapshot["denormalized_action_l1"]
            < initial_snapshot["denormalized_action_l1"]
        ),
        "gripper_and_constant_action_contract_valid": bool(action_contract_valid),
    }
    if gradient_gate is not None:
        gradient_checks = gradient_gate.get("checks", {})
        checks["gradient_gate_passed"] = (
            gradient_gate.get("schema") == "fastwam_gradient_gate.v1"
            and gradient_gate.get("status") == "passed"
            and gradient_checks.get("loss_finite_on_all_ranks") is True
            and gradient_checks.get("all_observed_gradients_finite") is True
            and gradient_checks.get(
                "at_least_one_finite_nonzero_trainable_gradient"
            )
            is True
            and gradient_checks.get(
                "finite_nonzero_trainable_gradient_on_every_rank"
            )
            is True
        )
    report = {
        "schema": REPORT_SCHEMA,
        "status": "passed" if all(checks.values()) else "failed",
        "sample_ids": [int(sample_id) for sample_id in sample_ids],
        "max_steps": int(max_steps),
        "initial": initial_snapshot,
        "final": final_snapshot,
        "checks": checks,
        "metadata": dict(metadata or {}),
    }
    if gradient_gate is not None:
        report["gradient_gate"] = dict(gradient_gate)
    return report


def save_overfit_report(path: str | os.PathLike[str], report: Mapping[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        dir=output.parent, prefix=f".{output.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(dict(report), stream, ensure_ascii=True, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def load_overfit_report(path: str | os.PathLike[str]) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        report = json.load(stream)
    if report.get("schema") != REPORT_SCHEMA:
        raise ValueError(
            f"Unsupported overfit report schema {report.get('schema')!r}; expected {REPORT_SCHEMA!r}."
        )
    return report


def validate_action_contract(action: torch.Tensor) -> bool:
    """Return whether denormalized actions satisfy B1K execution invariants."""
    try:
        validate_b1k_action(action, require_binary_gripper=True)
    except (TypeError, ValueError):
        return False
    return True
