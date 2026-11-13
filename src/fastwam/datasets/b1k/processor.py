# Copyright (c) 2026 Xiaomi Corporation

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from ..lerobot.processors.fastwam_processor import FastWAMProcessor
from .contract import ACTION_DIM, PROPRIO_DIM
from .normalizer import B1KMixedNormalizer, DEFAULT_CLAMP_VALUE, DEFAULT_EPS


class _B1KNestedNormalizer:
    """Adapt the B1K tensor normalizer to FastWAM's nested sample contract."""

    def __init__(
        self,
        normalizer: B1KMixedNormalizer,
        *,
        action_key: str,
        state_key: str,
    ) -> None:
        self.normalizer = normalizer
        self.action_key = action_key
        self.state_key = state_key

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        if "action" in batch:
            batch["action"][self.action_key] = self.normalizer.normalize_action(
                batch["action"][self.action_key]
            )
        batch["state"][self.state_key] = self.normalizer.normalize_proprio(
            batch["state"][self.state_key]
        )
        return batch

    def backward(self, batch: dict[str, Any]) -> dict[str, Any]:
        batch["action"][self.action_key] = self.normalizer.denormalize_action(
            batch["action"][self.action_key]
        )
        batch["state"][self.state_key] = self.normalizer.denormalize_proprio(
            batch["state"][self.state_key]
        )
        return batch


class B1KFastWAMProcessor(FastWAMProcessor):
    """FastWAM processor implementing the fixed B1K 2026 z-score contract.

    The inherited preprocessing order is intentionally retained:
    action/state transform, mixed normalization, then concatenation/padding.
    Postprocessing applies the exact inverse order.

    Hydra target: ``fastwam.datasets.b1k.processor.B1KFastWAMProcessor``.
    """

    def __init__(
        self,
        *,
        num_obs_steps: int = 33,
        num_image_steps: int = 9,
        **kwargs: Any,
    ) -> None:
        if num_obs_steps <= 1 or (num_obs_steps - 1) % 4 != 0:
            raise ValueError(
                "B1K num_obs_steps-1 must be positive and divisible by "
                f"action_video_freq_ratio=4, got {num_obs_steps}."
            )
        expected_image_steps = (num_obs_steps - 1) // 4 + 1
        if num_image_steps != expected_image_steps or (num_image_steps - 1) % 4 != 0:
            raise ValueError(
                "B1K num_image_steps must match num_obs_steps at ratio 4 and yield a "
                "VAE-compatible number of video transitions; "
                f"expected {expected_image_steps}, got {num_image_steps}."
            )
        super().__init__(
            num_obs_steps=num_obs_steps,
            num_image_steps=num_image_steps,
            **kwargs,
        )
        self._action_key = _single_field_key(
            self.shape_meta, "action", expected_dim=ACTION_DIM
        )
        self._state_key = _single_field_key(
            self.shape_meta, "state", expected_dim=PROPRIO_DIM
        )

    def set_normalizer_from_stats(
        self, dataset_stats: dict[str, Any] | str | Path | None = None
    ) -> None:
        if dataset_stats is None:
            raise ValueError("B1K processor requires train-only normalization stats.")
        stats_document = _load_stats_document(dataset_stats)
        stats = stats_document.get("stats", stats_document)

        if "split" in stats_document:
            split_name = stats_document["split"].get("name")
            if split_name != "train":
                raise ValueError(
                    "B1K normalization stats must be train-only; "
                    f"got split name {split_name!r}."
                )

        action_stats = _field_stats(stats, "action", self._action_key)
        state_stats = _field_stats(stats, "state", self._state_key)
        normalization = stats_document.get("normalization", {})
        # Legacy stats call this a global clamp. Under the v2 data contract it
        # applies only to proprio; continuous action z-scores stay unclamped.
        clamp_value = _clamp_value(normalization.get("clamp", DEFAULT_CLAMP_VALUE))
        eps = normalization.get("std_epsilon", DEFAULT_EPS)

        mixed = B1KMixedNormalizer(
            action_mean=_stat_value(action_stats, "mean"),
            action_std=_stat_value(action_stats, "std"),
            proprio_mean=_stat_value(state_stats, "mean"),
            proprio_std=_stat_value(state_stats, "std"),
            clamp_value=clamp_value,
            eps=eps,
        )
        self._normalizer = _B1KNestedNormalizer(
            mixed,
            action_key=self._action_key,
            state_key=self._state_key,
        )


def _load_stats_document(
    dataset_stats: dict[str, Any] | str | Path,
) -> dict[str, Any]:
    if isinstance(dataset_stats, (str, Path)):
        path = Path(dataset_stats).expanduser()
        with path.open("r", encoding="utf-8") as stream:
            loaded = json.load(stream)
        if not isinstance(loaded, dict):
            raise ValueError(f"B1K stats JSON must contain an object, got {type(loaded)!r}.")
        return loaded
    if not isinstance(dataset_stats, dict):
        raise TypeError(
            "dataset_stats must be a mapping or JSON path, "
            f"got {type(dataset_stats)!r}."
        )
    return dataset_stats


def _single_field_key(
    shape_meta: dict[str, Any], field: str, *, expected_dim: int
) -> str:
    metadata = shape_meta.get(field)
    if (
        not isinstance(metadata, Sequence)
        or isinstance(metadata, (str, bytes))
        or len(metadata) != 1
    ):
        raise ValueError(f"B1K v1 requires exactly one {field} field.")
    item = metadata[0]
    if item.get("shape") != expected_dim:
        raise ValueError(
            f"B1K v1 {field} transformed dimension must be {expected_dim}, "
            f"got {item.get('shape')!r}."
        )
    key = item.get("key")
    if not isinstance(key, str) or not key:
        raise ValueError(f"B1K v1 {field} field requires a non-empty string key.")
    return key


def _field_stats(stats: dict[str, Any], field: str, key: str) -> dict[str, Any]:
    try:
        fields = stats[field]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"B1K stats are missing the {field!r} section.") from exc
    if not isinstance(fields, dict):
        raise ValueError(f"B1K stats {field!r} section must be a mapping.")

    if key in fields:
        value = fields[key]
    elif len(fields) == 1:
        # Allows official nested stats whose sole field key differs from the
        # configured B1K key, while still rejecting ambiguous multi-field data.
        value = next(iter(fields.values()))
    else:
        raise ValueError(
            f"B1K stats {field!r} section does not contain field {key!r}."
        )
    if not isinstance(value, dict):
        raise ValueError(f"B1K stats for {field}.{key} must be a mapping.")
    return value


def _stat_value(stats: dict[str, Any], name: str) -> torch.Tensor | list[float]:
    global_name = f"global_{name}"
    if global_name in stats:
        return stats[global_name]
    if name in stats:
        return stats[name]
    raise ValueError(
        f"B1K stats field must contain {global_name!r} or {name!r}."
    )


def _clamp_value(value: Any) -> float:
    try:
        tensor = torch.as_tensor(value, dtype=torch.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid B1K normalization clamp {value!r}.") from exc
    if tensor.ndim == 0:
        return float(tensor.item())
    if tensor.shape == (2,):
        lower, upper = (float(item) for item in tensor.tolist())
        if lower != -upper or upper <= 0:
            raise ValueError(
                "B1K normalization clamp must be symmetric around zero, "
                f"got {value!r}."
            )
        return upper
    raise ValueError(f"Invalid B1K normalization clamp {value!r}.")
