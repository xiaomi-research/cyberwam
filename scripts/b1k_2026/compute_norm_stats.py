#!/usr/bin/env python3
# Copyright (c) 2026 Xiaomi Corporation

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from fastwam.datasets.b1k.contract import (
    ACTION_CONSTANT_DIM,
    ACTION_DIM,
    ACTION_GRIPPER_DIMS,
    ACTION_LABELS,
    PROPRIO_DIM,
    PROPRIO_LABELS,
    extract_b1k_proprio,
    validate_b1k_action,
)
from fastwam.datasets.b1k.split import build_b1k_episode_split
from fastwam.datasets.b1k.normalizer import B1KMixedNormalizer


SCHEMA_VERSION = "b1k_2026_norm_stats.v1"
TRAINING_TOKEN_DIAGNOSTICS_SCHEMA_VERSION = (
    "b1k_2026_training_token_diagnostics.v1"
)
PARQUET_COLUMNS = ("action", "observation.state")
ACTION_OFFSETS_30HZ = tuple(range(0, 64, 2))
STATE_OFFSETS_30HZ = tuple(range(0, 65, 2))
DEFAULT_WINDOW_START_STRIDE = 6
DEFAULT_DATASET_ROOT = Path(
    "data/behavior-1k/2026-challenge-demos-224-rgb-jpeg"
)


@dataclass
class StreamingMoments:
    dimension: int

    def __post_init__(self) -> None:
        self.count = 0
        self.mean = np.zeros(self.dimension, dtype=np.float64)
        self.m2 = np.zeros(self.dimension, dtype=np.float64)
        self.minimum = np.full(self.dimension, np.inf, dtype=np.float64)
        self.maximum = np.full(self.dimension, -np.inf, dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.dimension:
            raise ValueError(
                f"Expected [N, {self.dimension}] values, got {values.shape}."
            )
        if values.shape[0] == 0:
            return
        if not np.isfinite(values).all():
            raise ValueError("Cannot compute B1K norm stats from non-finite values.")

        batch_count = values.shape[0]
        batch_mean = values.mean(axis=0, dtype=np.float64)
        centered = values - batch_mean
        batch_m2 = np.einsum("ij,ij->j", centered, centered, dtype=np.float64)
        if self.count == 0:
            self.mean = batch_mean
            self.m2 = batch_m2
        else:
            delta = batch_mean - self.mean
            total = self.count + batch_count
            self.mean += delta * (batch_count / total)
            self.m2 += batch_m2 + delta * delta * (self.count * batch_count / total)
        self.count += batch_count
        self.minimum = np.minimum(self.minimum, values.min(axis=0))
        self.maximum = np.maximum(self.maximum, values.max(axis=0))

    @property
    def std(self) -> np.ndarray:
        if self.count == 0:
            raise ValueError("No samples were provided for B1K norm stats.")
        return np.sqrt(np.maximum(self.m2 / self.count, 0.0))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metadata_fingerprint(dataset_root: Path) -> dict[str, Any]:
    metadata = {}
    aggregate = hashlib.sha256()
    for path in sorted((dataset_root / "meta").glob("*")):
        if not path.is_file():
            continue
        relative_path = path.relative_to(dataset_root).as_posix()
        file_hash = _sha256(path)
        metadata[relative_path] = {"sha256": file_hash, "size_bytes": path.stat().st_size}
        aggregate.update(relative_path.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(file_hash.encode("ascii"))
        aggregate.update(b"\n")
    if not metadata:
        raise FileNotFoundError(f"No metadata files found in {dataset_root / 'meta'}.")
    return {"sha256": aggregate.hexdigest(), "files": metadata}


def _load_info(dataset_root: Path) -> dict[str, Any]:
    info_path = dataset_root / "meta" / "info.json"
    with info_path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _episode_path(dataset_root: Path, info: dict[str, Any], episode_index: int) -> Path:
    chunks_size = int(info.get("chunks_size", 1000))
    pattern = info.get(
        "data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    )
    relative_path = pattern.format(
        episode_chunk=episode_index // chunks_size, episode_index=episode_index
    )
    return dataset_root / relative_path


def _fixed_size_list_to_numpy(column: Any, dimension: int, name: str) -> np.ndarray:
    if column.null_count:
        raise ValueError(f"Parquet column {name!r} contains null values.")
    array = column.combine_chunks()
    list_size = getattr(array.type, "list_size", None)
    if list_size != dimension:
        raise ValueError(
            f"Parquet column {name!r} must contain {dimension}D fixed-size lists, "
            f"got {array.type}."
        )
    values = array.values.to_numpy(zero_copy_only=False)
    return np.asarray(values, dtype=np.float64).reshape(len(array), dimension)


def _read_episode(path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("compute_norm_stats.py requires pyarrow.") from exc

    if not path.is_file():
        raise FileNotFoundError(f"Missing B1K episode parquet: {path}")
    table = pq.read_table(path, columns=list(PARQUET_COLUMNS), use_threads=True)
    action = _fixed_size_list_to_numpy(table["action"], ACTION_DIM, "action")
    raw_state = _fixed_size_list_to_numpy(
        table["observation.state"], 61, "observation.state"
    )
    if action.shape[0] != raw_state.shape[0]:
        raise ValueError(
            f"Action/state row count mismatch in {path}: {action.shape[0]} != {raw_state.shape[0]}."
        )
    validate_b1k_action(action)
    proprio = np.asarray(extract_b1k_proprio(raw_state), dtype=np.float64)
    return action, proprio


def _iter_episodes(
    dataset_root: Path, info: dict[str, Any], episode_indices: Iterable[int]
) -> Iterable[tuple[int, np.ndarray, np.ndarray]]:
    for episode_index in episode_indices:
        action, proprio = _read_episode(_episode_path(dataset_root, info, episode_index))
        yield episode_index, action, proprio


def _field_report(
    labels: tuple[str, ...],
    moments: StreamingMoments,
    greater_than_3sigma: np.ndarray,
    clamped: np.ndarray,
    *,
    low_variance_threshold: float,
) -> dict[str, Any]:
    count = moments.count
    std = moments.std
    low_variance = std <= low_variance_threshold
    return {
        "labels": list(labels),
        "count": count,
        "mean": moments.mean.tolist(),
        "std": std.tolist(),
        "min": moments.minimum.tolist(),
        "max": moments.maximum.tolist(),
        "greater_than_3sigma_rate": (greater_than_3sigma / count).tolist(),
        "clamp_rate": (clamped / count).tolist(),
        "low_variance_threshold": low_variance_threshold,
        "low_variance": low_variance.tolist(),
        "low_variance_dimensions": np.flatnonzero(low_variance).tolist(),
    }


def _gripper_report(counts: dict[int, dict[float, int]]) -> dict[str, Any]:
    result = {}
    for dimension, dimension_counts in sorted(counts.items()):
        total = sum(dimension_counts.values())
        result[str(dimension)] = {
            "label": ACTION_LABELS[dimension],
            "unique_values": [float(value) for value in sorted(dimension_counts)],
            "counts": {
                str(float(value)): count for value, count in sorted(dimension_counts.items())
            },
            "rates": {
                str(float(value)): count / total
                for value, count in sorted(dimension_counts.items())
            },
        }
    return result


def _training_stats(field: dict[str, Any]) -> dict[str, Any]:
    return {
        "global_count": field["count"],
        "global_mean": field["mean"],
        "global_std": field["std"],
        "global_min": field["min"],
        "global_max": field["max"],
    }


def _moments_report(
    labels: tuple[str, ...],
    moments: StreamingMoments,
    clamp_boundary_counts: np.ndarray | None = None,
) -> dict[str, Any]:
    report = {
        "labels": list(labels),
        "count": moments.count,
        "mean": moments.mean.tolist(),
        "std": moments.std.tolist(),
        "min": moments.minimum.tolist(),
        "max": moments.maximum.tolist(),
    }
    if clamp_boundary_counts is not None:
        report["clamp_rate"] = (clamp_boundary_counts / moments.count).tolist()
    return report


def _normalizer_from_result(result: dict[str, Any]) -> B1KMixedNormalizer:
    try:
        action_stats = result["stats"]["action"]["default"]
        proprio_stats = result["stats"]["state"]["default"]
        normalization = result["normalization"]
        clamp_min, clamp_max = normalization["clamp"]
        std_epsilon = normalization["std_epsilon"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Invalid B1K norm stats document.") from exc
    if float(clamp_min) != -float(clamp_max) or float(clamp_max) <= 0:
        raise ValueError(
            f"B1KMixedNormalizer requires a symmetric positive clamp, got "
            f"{normalization['clamp']!r}."
        )
    return B1KMixedNormalizer(
        action_mean=action_stats["global_mean"],
        action_std=action_stats["global_std"],
        proprio_mean=proprio_stats["global_mean"],
        proprio_std=proprio_stats["global_std"],
        clamp_value=float(clamp_max),
        eps=float(std_epsilon),
    )


def add_post_normalization_diagnostics(
    result: dict[str, Any], dataset_root: str | Path
) -> dict[str, Any]:
    """Scan the recorded train split once and add final mixed-normalizer moments."""
    dataset_root = Path(dataset_root).resolve()
    if result.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Expected schema_version {SCHEMA_VERSION!r}, got "
            f"{result.get('schema_version')!r}."
        )
    recorded_root = Path(result.get("dataset", {}).get("root", "")).resolve()
    if recorded_root != dataset_root:
        raise ValueError(
            f"Stats dataset root {recorded_root} does not match requested root {dataset_root}."
        )
    try:
        train_episodes = [int(index) for index in result["split"]["selected_episode_indices"]]
        expected_frame_count = int(result["split"]["frame_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Stats document does not record a valid train split.") from exc
    if not train_episodes:
        raise ValueError("Stats document records an empty train split.")

    normalizer = _normalizer_from_result(result)
    info = _load_info(dataset_root)
    action_moments = StreamingMoments(ACTION_DIM)
    proprio_moments = StreamingMoments(PROPRIO_DIM)
    action_clamp_counts = np.zeros(ACTION_DIM, dtype=np.int64)
    proprio_clamp_counts = np.zeros(PROPRIO_DIM, dtype=np.int64)
    for _, action, proprio in _iter_episodes(dataset_root, info, train_episodes):
        normalized_action = normalizer.normalize_action(torch.from_numpy(action)).numpy()
        normalized_proprio = normalizer.normalize_proprio(torch.from_numpy(proprio)).numpy()
        action_moments.update(normalized_action)
        proprio_moments.update(normalized_proprio)
        action_clamp_counts += np.count_nonzero(
            np.abs(normalized_action) >= normalizer.clamp_value, axis=0
        )
        proprio_clamp_counts += np.count_nonzero(
            np.abs(normalized_proprio) >= normalizer.clamp_value, axis=0
        )

    if action_moments.count != expected_frame_count or proprio_moments.count != expected_frame_count:
        raise ValueError(
            "Post-normalization scan frame count does not match the stats document: "
            f"expected {expected_frame_count}, got action={action_moments.count}, "
            f"proprio={proprio_moments.count}."
        )
    result["fields"]["action"]["post_normalization"] = _moments_report(
        ACTION_LABELS, action_moments, action_clamp_counts
    )
    result["fields"]["proprio"]["post_normalization"] = _moments_report(
        PROPRIO_LABELS, proprio_moments, proprio_clamp_counts
    )
    return result


def _valid_training_token_rows(
    values: np.ndarray, offsets: tuple[int, ...], window_start_stride: int
) -> tuple[np.ndarray, int, int]:
    """Expand episode-local windows and return only non-padding token rows."""
    if window_start_stride <= 0:
        raise ValueError("window_start_stride must be positive.")
    if values.ndim != 2:
        raise ValueError(f"Expected [T, D] episode values, got {values.shape}.")
    if values.shape[0] == 0:
        return values.copy(), 0, 0

    anchors = np.arange(0, values.shape[0], window_start_stride, dtype=np.int64)
    rows = anchors[:, None] + np.asarray(offsets, dtype=np.int64)[None, :]
    valid = rows < values.shape[0]
    valid_rows = values[rows[valid]]
    return valid_rows, len(anchors), int(rows.size - np.count_nonzero(valid))


def _normalized_distribution_comparison(
    labels: tuple[str, ...],
    training_moments: StreamingMoments,
    training_clamp_counts: np.ndarray,
    natural_report: dict[str, Any],
) -> dict[str, Any]:
    training = _moments_report(labels, training_moments, training_clamp_counts)
    try:
        natural_mean = np.asarray(natural_report["mean"], dtype=np.float64)
        natural_std = np.asarray(natural_report["std"], dtype=np.float64)
        natural_clamp_rate = np.asarray(
            natural_report["clamp_rate"], dtype=np.float64
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Natural-frame post-normalization diagnostics are missing or invalid."
        ) from exc
    if (
        natural_mean.shape != (len(labels),)
        or natural_std.shape != (len(labels),)
        or natural_clamp_rate.shape != (len(labels),)
    ):
        raise ValueError("Natural-frame diagnostics have an unexpected dimension.")

    training_mean = np.asarray(training["mean"], dtype=np.float64)
    training_std = np.asarray(training["std"], dtype=np.float64)
    training_clamp_rate = np.asarray(training["clamp_rate"], dtype=np.float64)
    return {
        "training_tokens": training,
        "natural_frames": {
            "count": natural_report["count"],
            "mean": natural_mean.tolist(),
            "std": natural_std.tolist(),
            "clamp_rate": natural_clamp_rate.tolist(),
        },
        "training_minus_natural": {
            "mean": (training_mean - natural_mean).tolist(),
            "std": (training_std - natural_std).tolist(),
            "clamp_rate": (training_clamp_rate - natural_clamp_rate).tolist(),
        },
    }


def compute_training_token_diagnostics(
    result: dict[str, Any],
    dataset_root: str | Path,
    *,
    window_start_stride: int = DEFAULT_WINDOW_START_STRIDE,
) -> dict[str, Any]:
    """Compare stride-sampled training tokens with natural-frame norm statistics.

    The normalizer always comes from the recorded natural-frame train-only stats.
    Episode-tail padding is deliberately excluded rather than repeated in the
    diagnostic distribution.
    """
    if window_start_stride <= 0:
        raise ValueError("window_start_stride must be positive.")
    dataset_root = Path(dataset_root).resolve()
    if result.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Expected schema_version {SCHEMA_VERSION!r}, got "
            f"{result.get('schema_version')!r}."
        )
    recorded_root = Path(result.get("dataset", {}).get("root", "")).resolve()
    if recorded_root != dataset_root:
        raise ValueError(
            f"Stats dataset root {recorded_root} does not match requested root {dataset_root}."
        )
    try:
        train_episodes = [int(index) for index in result["split"]["selected_episode_indices"]]
        action_natural = result["fields"]["action"]["post_normalization"]
        proprio_natural = result["fields"]["proprio"]["post_normalization"]
        source_fps = float(result["dataset"]["fps"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Stats document does not contain the train split and natural-frame diagnostics."
        ) from exc
    if not train_episodes:
        raise ValueError("Stats document records an empty train split.")

    normalizer = _normalizer_from_result(result)
    info = _load_info(dataset_root)
    action_moments = StreamingMoments(ACTION_DIM)
    proprio_moments = StreamingMoments(PROPRIO_DIM)
    action_clamp_counts = np.zeros(ACTION_DIM, dtype=np.int64)
    proprio_clamp_counts = np.zeros(PROPRIO_DIM, dtype=np.int64)
    anchor_count = 0
    action_padding_excluded = 0
    proprio_padding_excluded = 0

    for _, action, proprio in _iter_episodes(dataset_root, info, train_episodes):
        action_tokens, episode_anchor_count, action_padding = _valid_training_token_rows(
            action, ACTION_OFFSETS_30HZ, window_start_stride
        )
        proprio_tokens, proprio_anchor_count, proprio_padding = (
            _valid_training_token_rows(
                proprio, STATE_OFFSETS_30HZ, window_start_stride
            )
        )
        if proprio_anchor_count != episode_anchor_count:
            raise AssertionError("Action and proprio anchor counts must match.")
        anchor_count += episode_anchor_count
        action_padding_excluded += action_padding
        proprio_padding_excluded += proprio_padding

        normalized_action = normalizer.normalize_action(
            torch.from_numpy(action_tokens)
        ).numpy()
        normalized_proprio = normalizer.normalize_proprio(
            torch.from_numpy(proprio_tokens)
        ).numpy()
        action_moments.update(normalized_action)
        proprio_moments.update(normalized_proprio)
        action_clamp_counts += np.count_nonzero(
            np.abs(normalized_action) >= normalizer.clamp_value, axis=0
        )
        proprio_clamp_counts += np.count_nonzero(
            np.abs(normalized_proprio) >= normalizer.clamp_value, axis=0
        )

    return {
        "schema_version": TRAINING_TOKEN_DIAGNOSTICS_SCHEMA_VERSION,
        "normalization_source": "train_only_natural_frame_stats",
        "sampling": {
            "source_fps": source_fps,
            "window_start_stride": window_start_stride,
            "anchor_frequency_hz": source_fps / window_start_stride,
            "action_offsets_30hz": list(ACTION_OFFSETS_30HZ),
            "proprio_offsets_30hz": list(STATE_OFFSETS_30HZ),
            "padding_policy": "exclude",
        },
        "counts": {
            "episode_count": len(train_episodes),
            "anchor_count": anchor_count,
            "action_valid_token_count": action_moments.count,
            "action_padding_token_count_excluded": action_padding_excluded,
            "proprio_valid_token_count": proprio_moments.count,
            "proprio_padding_token_count_excluded": proprio_padding_excluded,
        },
        "fields": {
            "action": _normalized_distribution_comparison(
                ACTION_LABELS,
                action_moments,
                action_clamp_counts,
                action_natural,
            ),
            "proprio": _normalized_distribution_comparison(
                PROPRIO_LABELS,
                proprio_moments,
                proprio_clamp_counts,
                proprio_natural,
            ),
        },
    }


def _find_reference_stats(dataset_root: Path) -> Path | None:
    candidates = [dataset_root / "meta" / "stats.json"]
    suffix = "-224-rgb-jpeg"
    if dataset_root.name.endswith(suffix):
        candidates.append(
            dataset_root.with_name(dataset_root.name.removesuffix(suffix)) / "meta" / "stats.json"
        )
    return next((path for path in candidates if path.is_file()), None)


def _reference_comparison(dataset_root: Path, action_field: dict[str, Any]) -> dict[str, Any]:
    reference_path = _find_reference_stats(dataset_root)
    if reference_path is None:
        return {"status": "not_found"}
    with reference_path.open("r", encoding="utf-8") as stream:
        reference = json.load(stream)
    try:
        reference_action = reference["action"]
        reference_mean = np.asarray(reference_action["mean"], dtype=np.float64)
        reference_std = np.asarray(reference_action["std"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid reference action stats in {reference_path}.") from exc
    if reference_mean.shape != (ACTION_DIM,) or reference_std.shape != (ACTION_DIM,):
        raise ValueError(
            f"Reference action stats in {reference_path} must be {ACTION_DIM}D, got "
            f"mean={reference_mean.shape}, std={reference_std.shape}."
        )
    train_mean = np.asarray(action_field["mean"], dtype=np.float64)
    train_std = np.asarray(action_field["std"], dtype=np.float64)
    return {
        "status": "available",
        "path": str(reference_path),
        "sha256": _sha256(reference_path),
        "reference_scope": "original_full_dataset",
        "reference_count": reference_action.get("count"),
        "action": {
            "reference_mean": reference_mean.tolist(),
            "reference_std": reference_std.tolist(),
            "train_minus_reference_mean": (train_mean - reference_mean).tolist(),
            "train_minus_reference_std": (train_std - reference_std).tolist(),
        },
        "proprio": {
            "status": "not_directly_comparable",
            "reason": (
                "The reference contains 61D raw-state marginal stats. They cannot recover "
                "the variance of summed finger positions used by the canonical 23D proprio extractor."
            ),
        },
    }


def compute_norm_stats(
    dataset_root: str | Path,
    *,
    max_episodes: int | None = None,
    expected_tasks: int = 100,
    episodes_per_task: int = 200,
    std_epsilon: float = 1e-8,
    low_variance_threshold: float = 1e-8,
) -> dict[str, Any]:
    dataset_root = Path(dataset_root).resolve()
    if max_episodes is not None and max_episodes <= 0:
        raise ValueError("max_episodes must be positive when provided.")
    info = _load_info(dataset_root)
    train_episodes = build_b1k_episode_split(
        dataset_root,
        "train",
        expected_tasks=expected_tasks,
        episodes_per_task=episodes_per_task,
    )
    if max_episodes is not None:
        train_episodes = train_episodes[:max_episodes]
    if not train_episodes:
        raise ValueError("The selected B1K train split contains no episodes.")

    action_moments = StreamingMoments(ACTION_DIM)
    proprio_moments = StreamingMoments(PROPRIO_DIM)
    gripper_counts = {dimension: {} for dimension in ACTION_GRIPPER_DIMS}
    episode_lengths: dict[str, int] = {}
    for episode_index, action, proprio in _iter_episodes(dataset_root, info, train_episodes):
        action_moments.update(action)
        proprio_moments.update(proprio)
        episode_lengths[str(episode_index)] = action.shape[0]
        for dimension in ACTION_GRIPPER_DIMS:
            values, counts = np.unique(action[:, dimension], return_counts=True)
            for value, count in zip(values, counts, strict=True):
                key = float(value)
                gripper_counts[dimension][key] = (
                    gripper_counts[dimension].get(key, 0) + int(count)
                )

    action_outlier_counts = np.zeros(ACTION_DIM, dtype=np.int64)
    action_clamp_counts = np.zeros(ACTION_DIM, dtype=np.int64)
    proprio_outlier_counts = np.zeros(PROPRIO_DIM, dtype=np.int64)
    proprio_clamp_counts = np.zeros(PROPRIO_DIM, dtype=np.int64)
    action_denominator = action_moments.std + std_epsilon
    proprio_denominator = proprio_moments.std + std_epsilon
    normalizer = B1KMixedNormalizer(
        action_mean=action_moments.mean,
        action_std=action_moments.std,
        proprio_mean=proprio_moments.mean,
        proprio_std=proprio_moments.std,
        clamp_value=5.0,
        eps=std_epsilon,
    )
    normalized_action_moments = StreamingMoments(ACTION_DIM)
    normalized_proprio_moments = StreamingMoments(PROPRIO_DIM)
    normalized_action_clamp_counts = np.zeros(ACTION_DIM, dtype=np.int64)
    normalized_proprio_clamp_counts = np.zeros(PROPRIO_DIM, dtype=np.int64)
    for _, action, proprio in _iter_episodes(dataset_root, info, train_episodes):
        action_z = np.abs((action - action_moments.mean) / action_denominator)
        proprio_z = np.abs((proprio - proprio_moments.mean) / proprio_denominator)
        action_outlier_counts += np.count_nonzero(action_z > 3.0, axis=0)
        action_clamp_counts += np.count_nonzero(action_z > 5.0, axis=0)
        proprio_outlier_counts += np.count_nonzero(proprio_z > 3.0, axis=0)
        proprio_clamp_counts += np.count_nonzero(proprio_z > 5.0, axis=0)
        normalized_action = normalizer.normalize_action(torch.from_numpy(action)).numpy()
        normalized_proprio = normalizer.normalize_proprio(torch.from_numpy(proprio)).numpy()
        normalized_action_moments.update(normalized_action)
        normalized_proprio_moments.update(normalized_proprio)
        normalized_action_clamp_counts += np.count_nonzero(
            np.abs(normalized_action) >= normalizer.clamp_value, axis=0
        )
        normalized_proprio_clamp_counts += np.count_nonzero(
            np.abs(normalized_proprio) >= normalizer.clamp_value, axis=0
        )

    action_field = _field_report(
        ACTION_LABELS,
        action_moments,
        action_outlier_counts,
        action_clamp_counts,
        low_variance_threshold=low_variance_threshold,
    )
    action_field["gripper"] = _gripper_report(gripper_counts)
    action_field["constant_zero_dimension"] = {
        "index": ACTION_CONSTANT_DIM,
        "label": ACTION_LABELS[ACTION_CONSTANT_DIM],
        "valid": bool(
            action_moments.minimum[ACTION_CONSTANT_DIM] == 0.0
            and action_moments.maximum[ACTION_CONSTANT_DIM] == 0.0
        ),
    }
    action_field["post_normalization"] = _moments_report(
        ACTION_LABELS, normalized_action_moments, normalized_action_clamp_counts
    )
    proprio_field = _field_report(
        PROPRIO_LABELS,
        proprio_moments,
        proprio_outlier_counts,
        proprio_clamp_counts,
        low_variance_threshold=low_variance_threshold,
    )
    proprio_field["post_normalization"] = _moments_report(
        PROPRIO_LABELS, normalized_proprio_moments, normalized_proprio_clamp_counts
    )

    result = {
        "schema_version": SCHEMA_VERSION,
        "dataset": {
            "root": str(dataset_root),
            "robot_type": info.get("robot_type"),
            "fps": info.get("fps"),
            "source_metadata_fingerprint": _metadata_fingerprint(dataset_root),
        },
        "split": {
            "name": "train",
            "rule": f"per-task local episode 0..{episodes_per_task - 3}",
            "expected_tasks": expected_tasks,
            "episodes_per_task": episodes_per_task,
            "selected_episode_count": len(train_episodes),
            "selected_episode_indices": train_episodes,
            "max_episodes": max_episodes,
            "weighting": "natural_frame",
            "frame_count": action_moments.count,
            "episode_lengths": episode_lengths,
        },
        "normalization": {
            "mode": "global_per_dimension_z_score",
            "std_ddof": 0,
            "std_epsilon": std_epsilon,
            "clamp": [-5.0, 5.0],
            "stepwise": False,
            "action_and_proprio_share_stats": False,
            "action_gripper_dimensions": list(ACTION_GRIPPER_DIMS),
            "action_gripper_mode": "identity_binary_minus_one_plus_one",
            "action_constant_zero_dimension": ACTION_CONSTANT_DIM,
        },
        "stats": {
            "action": {"default": _training_stats(action_field)},
            "state": {"default": _training_stats(proprio_field)},
        },
        "fields": {"action": action_field, "proprio": proprio_field},
        "reference_comparison": _reference_comparison(dataset_root, action_field),
    }
    return result


def _diagnostic_report(
    result: dict[str, Any],
    training_token_distribution: dict[str, Any] | None = None,
) -> dict[str, Any]:
    report = {
        "schema_version": result["schema_version"],
        "dataset": result["dataset"],
        "split": result["split"],
        "normalization": result["normalization"],
        "reference_comparison": result["reference_comparison"],
        "diagnostics": {
            name: {
                key: value
                for key, value in field.items()
                if key
                in {
                    "labels",
                    "count",
                    "greater_than_3sigma_rate",
                    "clamp_rate",
                    "low_variance_threshold",
                    "low_variance",
                    "low_variance_dimensions",
                    "gripper",
                    "constant_zero_dimension",
                    "post_normalization",
                }
            }
            for name, field in result["fields"].items()
        },
    }
    if training_token_distribution is not None:
        report["training_token_distribution"] = training_token_distribution
    return report


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    temporary_path.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute train-only B1K 2026 23D action/proprio norm stats."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/b1k_2026/norm_stats.json")
    )
    parser.add_argument(
        "--diagnostics-output",
        type=Path,
        default=Path("artifacts/b1k_2026/norm_stats_diagnostics.json"),
    )
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--expected-tasks", type=int, default=100)
    parser.add_argument("--episodes-per-task", type=int, default=200)
    parser.add_argument("--std-epsilon", type=float, default=1e-8)
    parser.add_argument("--low-variance-threshold", type=float, default=1e-8)
    parser.add_argument(
        "--window-start-stride",
        type=int,
        default=DEFAULT_WINDOW_START_STRIDE,
        help=(
            "Episode-local anchor stride used only for the training-token "
            "distribution diagnostic (default: 6, or 5 Hz for 30 Hz data)."
        ),
    )
    parser.add_argument(
        "--reuse-stats",
        type=Path,
        help=(
            "Reuse an existing norm_stats.json and run only the one-pass "
            "post-normalization diagnostic scan."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.reuse_stats is not None:
        with args.reuse_stats.open("r", encoding="utf-8") as stream:
            result = add_post_normalization_diagnostics(json.load(stream), args.dataset_root)
    else:
        result = compute_norm_stats(
            args.dataset_root,
            max_episodes=args.max_episodes,
            expected_tasks=args.expected_tasks,
            episodes_per_task=args.episodes_per_task,
            std_epsilon=args.std_epsilon,
            low_variance_threshold=args.low_variance_threshold,
        )
    training_token_distribution = compute_training_token_diagnostics(
        result,
        args.dataset_root,
        window_start_stride=args.window_start_stride,
    )
    _write_json(args.output, result)
    _write_json(
        args.diagnostics_output,
        _diagnostic_report(result, training_token_distribution),
    )
    print(
        f"Saved {result['split']['frame_count']} train frames from "
        f"{result['split']['selected_episode_count']} episodes to {args.output}."
    )


if __name__ == "__main__":
    main()
