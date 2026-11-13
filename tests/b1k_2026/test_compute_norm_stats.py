# Copyright (c) 2026 Xiaomi Corporation

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from scripts.b1k_2026.compute_norm_stats import (  # noqa: E402
    SCHEMA_VERSION,
    TRAINING_TOKEN_DIAGNOSTICS_SCHEMA_VERSION,
    StreamingMoments,
    _diagnostic_report,
    _read_episode,
    _valid_training_token_rows,
    add_post_normalization_diagnostics,
    compute_norm_stats,
    compute_training_token_diagnostics,
)
from fastwam.datasets.b1k.normalizer import B1KMixedNormalizer  # noqa: E402


def _write_episode(path: Path, episode_index: int, length: int) -> tuple[np.ndarray, np.ndarray]:
    frame = np.arange(length, dtype=np.float32)
    action = np.zeros((length, 23), dtype=np.float32)
    action[:, 0] = episode_index + frame
    action[:, 1] = 2 * frame
    action[:, 3] = 1.0 + episode_index
    action[:, 14] = np.where((frame.astype(int) + episode_index) % 2, -1.0, 1.0)
    action[:, 22] = np.where(frame.astype(int) % 3, 1.0, -1.0)

    raw_state = np.zeros((length, 61), dtype=np.float32)
    raw_state[:, 0:3] = action[:, 0:3]
    raw_state[:, 53:57] = episode_index + 0.25
    raw_state[:, 3:10] = np.arange(7, dtype=np.float32) + frame[:, None]
    raw_state[:, 24] = 0.01 * (episode_index + 1)
    raw_state[:, 25] = 0.02 * (episode_index + 1)
    raw_state[:, 28:35] = np.arange(7, dtype=np.float32) - frame[:, None]
    raw_state[:, 49] = 0.03 * (episode_index + 1)
    raw_state[:, 50] = 0.04 * (episode_index + 1)

    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "action": pa.FixedSizeListArray.from_arrays(
                pa.array(action.reshape(-1)), list_size=23
            ),
            "observation.state": pa.FixedSizeListArray.from_arrays(
                pa.array(raw_state.reshape(-1)), list_size=61
            ),
        }
    )
    pq.write_table(table, path)
    return action.astype(np.float64), raw_state.astype(np.float64)


def _make_dataset(root: Path) -> tuple[np.ndarray, np.ndarray]:
    (root / "meta").mkdir(parents=True)
    info = {
        "chunks_size": 4,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "fps": 30,
        "robot_type": "R1Pro-test",
    }
    (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    all_action = []
    all_state = []
    with (root / "meta" / "episodes.jsonl").open("w", encoding="utf-8") as stream:
        for task_index in range(2):
            for local_index in range(4):
                episode_index = task_index * 4 + local_index
                length = 2 + episode_index % 3
                stream.write(
                    json.dumps(
                        {
                            "episode_index": episode_index,
                            "task_index": task_index,
                            "length": length,
                        }
                    )
                    + "\n"
                )
                action, state = _write_episode(
                    root
                    / "data"
                    / f"chunk-{episode_index // 4:03d}"
                    / f"episode_{episode_index:06d}.parquet",
                    episode_index,
                    length,
                )
                if local_index < 2:
                    all_action.append(action)
                    all_state.append(state)
    return np.concatenate(all_action), np.concatenate(all_state)


def test_streaming_moments_matches_numpy_across_batches() -> None:
    values = np.arange(33, dtype=np.float64).reshape(11, 3) - 5.0
    moments = StreamingMoments(3)
    moments.update(values[:2])
    moments.update(values[2:7])
    moments.update(values[7:])

    np.testing.assert_allclose(moments.mean, values.mean(axis=0), atol=1e-14)
    np.testing.assert_allclose(moments.std, values.std(axis=0), atol=1e-14)
    np.testing.assert_array_equal(moments.minimum, values.min(axis=0))
    np.testing.assert_array_equal(moments.maximum, values.max(axis=0))


def test_training_token_rows_expand_episode_local_stride_and_drop_tail_padding() -> None:
    values = np.arange(9, dtype=np.float64)[:, None]

    tokens, anchor_count, padding_count = _valid_training_token_rows(
        values, (0, 2, 4), window_start_stride=6
    )

    assert anchor_count == 2
    assert padding_count == 1
    np.testing.assert_array_equal(tokens[:, 0], [0.0, 2.0, 4.0, 6.0, 8.0])


def test_compute_norm_stats_uses_only_train_episodes_and_natural_frames(tmp_path: Path) -> None:
    expected_action, expected_raw_state = _make_dataset(tmp_path)
    expected_proprio = np.concatenate(
        (
            expected_raw_state[:, 0:3],
            expected_raw_state[:, 53:57],
            expected_raw_state[:, 3:10],
            expected_raw_state[:, 24:26].sum(axis=-1, keepdims=True),
            expected_raw_state[:, 28:35],
            expected_raw_state[:, 49:51].sum(axis=-1, keepdims=True),
        ),
        axis=-1,
    )

    result = compute_norm_stats(tmp_path, expected_tasks=2, episodes_per_task=4)

    assert result["schema_version"] == SCHEMA_VERSION
    assert result["split"]["selected_episode_indices"] == [0, 1, 4, 5]
    assert result["split"]["frame_count"] == len(expected_action)
    assert result["split"]["weighting"] == "natural_frame"
    np.testing.assert_allclose(
        result["fields"]["action"]["mean"], expected_action.mean(0), atol=1e-15
    )
    np.testing.assert_allclose(
        result["fields"]["action"]["std"], expected_action.std(0), atol=1e-15
    )
    np.testing.assert_allclose(
        result["fields"]["proprio"]["mean"], expected_proprio.mean(0), atol=1e-15
    )
    np.testing.assert_allclose(
        result["fields"]["proprio"]["std"], expected_proprio.std(0), atol=1e-15
    )
    assert result["fields"]["action"]["constant_zero_dimension"]["valid"] is True
    assert result["fields"]["action"]["gripper"]["14"]["unique_values"] == [-1.0, 1.0]
    assert result["stats"]["action"]["default"]["global_count"] == len(expected_action)
    assert len(result["fields"]["action"]["greater_than_3sigma_rate"]) == 23
    assert len(result["fields"]["proprio"]["clamp_rate"]) == 23
    action_post = result["fields"]["action"]["post_normalization"]
    proprio_post = result["fields"]["proprio"]["post_normalization"]
    normalizer = B1KMixedNormalizer(
        action_mean=expected_action.mean(0),
        action_std=expected_action.std(0),
        proprio_mean=expected_proprio.mean(0),
        proprio_std=expected_proprio.std(0),
    )
    expected_normalized_action = normalizer.normalize_action(
        torch.from_numpy(expected_action)
    ).numpy()
    expected_normalized_proprio = normalizer.normalize_proprio(
        torch.from_numpy(expected_proprio)
    ).numpy()
    np.testing.assert_allclose(
        action_post["mean"], expected_normalized_action.mean(0), atol=1e-15
    )
    np.testing.assert_allclose(action_post["std"], expected_normalized_action.std(0))
    np.testing.assert_allclose(
        proprio_post["mean"], expected_normalized_proprio.mean(0), atol=1e-15
    )
    np.testing.assert_allclose(proprio_post["std"], expected_normalized_proprio.std(0))
    np.testing.assert_array_equal(expected_normalized_action[:, 6], 0.0)
    np.testing.assert_array_equal(expected_normalized_action[:, 14], expected_action[:, 14])
    np.testing.assert_array_equal(expected_normalized_action[:, 22], expected_action[:, 22])
    assert action_post["clamp_rate"][6] == 0.0
    assert action_post["clamp_rate"][14] == 0.0
    assert action_post["clamp_rate"][22] == 0.0
    assert result["reference_comparison"] == {"status": "not_found"}
    assert "stats" not in _diagnostic_report(result)
    assert "post_normalization" in _diagnostic_report(result)["diagnostics"]["action"]


def test_post_normalization_diagnostics_can_reuse_existing_stats(tmp_path: Path) -> None:
    _make_dataset(tmp_path)
    result = compute_norm_stats(tmp_path, expected_tasks=2, episodes_per_task=4)
    expected = result["fields"]["action"].pop("post_normalization")
    result["fields"]["proprio"].pop("post_normalization")

    reused = add_post_normalization_diagnostics(result, tmp_path)

    assert reused["fields"]["action"]["post_normalization"] == expected


def test_training_token_diagnostics_uses_stride6_and_excludes_padding(
    tmp_path: Path,
) -> None:
    _make_dataset(tmp_path)
    result = compute_norm_stats(tmp_path, expected_tasks=2, episodes_per_task=4)
    original_result = deepcopy(result)

    diagnostic = compute_training_token_diagnostics(
        result, tmp_path, window_start_stride=6
    )

    assert result == original_result
    assert diagnostic["schema_version"] == TRAINING_TOKEN_DIAGNOSTICS_SCHEMA_VERSION
    assert diagnostic["normalization_source"] == "train_only_natural_frame_stats"
    assert diagnostic["sampling"] == {
        "source_fps": 30.0,
        "window_start_stride": 6,
        "anchor_frequency_hz": 5.0,
        "action_offsets_30hz": list(range(0, 64, 2)),
        "proprio_offsets_30hz": list(range(0, 65, 2)),
        "padding_policy": "exclude",
    }
    assert diagnostic["counts"] == {
        "episode_count": 4,
        "anchor_count": 4,
        "action_valid_token_count": 7,
        "action_padding_token_count_excluded": 121,
        "proprio_valid_token_count": 7,
        "proprio_padding_token_count_excluded": 125,
    }

    action_tokens = []
    proprio_tokens = []
    for episode_index in (0, 1, 4, 5):
        action, proprio = _read_episode(
            tmp_path
            / "data"
            / f"chunk-{episode_index // 4:03d}"
            / f"episode_{episode_index:06d}.parquet"
        )
        action_rows = np.arange(0, 64, 2)
        proprio_rows = np.arange(0, 65, 2)
        action_tokens.append(action[action_rows[action_rows < len(action)]])
        proprio_tokens.append(proprio[proprio_rows[proprio_rows < len(proprio)]])
    action_tokens = np.concatenate(action_tokens)
    proprio_tokens = np.concatenate(proprio_tokens)
    normalizer = B1KMixedNormalizer(
        action_mean=result["stats"]["action"]["default"]["global_mean"],
        action_std=result["stats"]["action"]["default"]["global_std"],
        proprio_mean=result["stats"]["state"]["default"]["global_mean"],
        proprio_std=result["stats"]["state"]["default"]["global_std"],
    )
    normalized_action = normalizer.normalize_action(torch.from_numpy(action_tokens)).numpy()
    normalized_proprio = normalizer.normalize_proprio(
        torch.from_numpy(proprio_tokens)
    ).numpy()
    action_report = diagnostic["fields"]["action"]
    proprio_report = diagnostic["fields"]["proprio"]
    np.testing.assert_allclose(
        action_report["training_tokens"]["mean"], normalized_action.mean(axis=0)
    )
    np.testing.assert_allclose(
        action_report["training_tokens"]["std"], normalized_action.std(axis=0)
    )
    np.testing.assert_allclose(
        proprio_report["training_tokens"]["mean"], normalized_proprio.mean(axis=0)
    )
    np.testing.assert_allclose(
        proprio_report["training_tokens"]["std"], normalized_proprio.std(axis=0)
    )
    natural_action_mean = np.asarray(
        action_report["natural_frames"]["mean"], dtype=np.float64
    )
    np.testing.assert_allclose(
        action_report["training_minus_natural"]["mean"],
        normalized_action.mean(axis=0) - natural_action_mean,
    )
    report = _diagnostic_report(result, diagnostic)
    assert report["training_token_distribution"] == diagnostic
    assert "training_token_distribution" not in _diagnostic_report(result)


def test_training_token_diagnostics_rejects_invalid_stride(tmp_path: Path) -> None:
    _make_dataset(tmp_path)
    result = compute_norm_stats(tmp_path, expected_tasks=2, episodes_per_task=4)

    with pytest.raises(ValueError, match="window_start_stride must be positive"):
        compute_training_token_diagnostics(
            result, tmp_path, window_start_stride=0
        )


def test_post_normalization_reuse_rejects_different_dataset_root(tmp_path: Path) -> None:
    _make_dataset(tmp_path)
    result = compute_norm_stats(tmp_path, expected_tasks=2, episodes_per_task=4)

    with pytest.raises(ValueError, match="does not match requested root"):
        add_post_normalization_diagnostics(result, tmp_path / "other")


def test_max_episodes_is_applied_after_train_split(tmp_path: Path) -> None:
    _make_dataset(tmp_path)
    result = compute_norm_stats(
        tmp_path, max_episodes=2, expected_tasks=2, episodes_per_task=4
    )
    assert result["split"]["selected_episode_indices"] == [0, 1]
    assert result["split"]["selected_episode_count"] == 2


def test_invalid_binary_gripper_is_rejected(tmp_path: Path) -> None:
    _make_dataset(tmp_path)
    path = tmp_path / "data" / "chunk-000" / "episode_000000.parquet"
    table = pq.read_table(path)
    action = np.asarray(table["action"].combine_chunks().values).reshape(-1, 23).copy()
    action[0, 14] = 0.0
    pq.write_table(
        pa.table(
            {
                "action": pa.FixedSizeListArray.from_arrays(
                    pa.array(action.reshape(-1)), list_size=23
                ),
                "observation.state": table["observation.state"],
            }
        ),
        path,
    )

    with pytest.raises(ValueError, match="gripper dimension 14"):
        compute_norm_stats(
            tmp_path, max_episodes=1, expected_tasks=2, episodes_per_task=4
        )
