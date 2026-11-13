# Copyright (c) 2026 Xiaomi Corporation

import json

import numpy as np
import pytest
import torch

from fastwam.b1k_execution import (
    B1KExecutionConfig,
    B1KExecutionState,
    B1KExecutionWrapper,
    prepare_b1k_execution,
)


def _chunk(factory, *batch_shape):
    shape = (*batch_shape, 32, 23)
    if factory is torch:
        return torch.arange(np.prod(shape), dtype=torch.float64).reshape(shape)
    return np.arange(np.prod(shape), dtype=np.float64).reshape(shape)


@pytest.mark.parametrize("factory", [torch, np])
def test_default_execution_exactly_repeats_first_four_tokens(factory):
    chunk = _chunk(factory, 2)
    chunk[..., 6] = 99
    chunk[..., 14] = (
        factory.tensor([-0.1, 0.0, 2.0, -3.0] * 16).reshape(2, 32)
        if factory is torch
        else np.array([-0.1, 0.0, 2.0, -3.0] * 16).reshape(2, 32)
    )
    chunk[..., 22] = (
        factory.tensor([0.0, -0.1, 0.1, -2.0] * 16).reshape(2, 32)
        if factory is torch
        else np.array([0.0, -0.1, 0.1, -2.0] * 16).reshape(2, 32)
    )

    original = chunk.copy() if factory is np else chunk.clone()
    result = prepare_b1k_execution(chunk)
    expected = (
        original[..., :4, :].copy() if factory is np else original[..., :4, :].clone()
    )
    expected[..., 6] = 0
    expected[..., 14] = (
        factory.where(expected[..., 14] >= 0, 1.0, -1.0)
        if factory is torch
        else np.where(expected[..., 14] >= 0, 1.0, -1.0)
    )
    expected[..., 22] = (
        factory.where(expected[..., 22] >= 0, 1.0, -1.0)
        if factory is torch
        else np.where(expected[..., 22] >= 0, 1.0, -1.0)
    )
    expected = (
        expected.repeat_interleave(2, dim=-2)
        if factory is torch
        else np.repeat(expected, 2, axis=-2)
    )

    if factory is torch:
        assert torch.equal(result.ticks, expected)
        assert result.ticks.dtype == chunk.dtype
        assert torch.equal(chunk, original)
    else:
        np.testing.assert_array_equal(result.ticks, expected)
        assert result.ticks.dtype == chunk.dtype
        np.testing.assert_array_equal(chunk, original)


def test_metadata_is_complete_and_json_serializable():
    wrapper = B1KExecutionWrapper()
    result = wrapper(torch.zeros(3, 32, 23))

    assert result.ticks.shape == (3, 8, 23)
    expected = {
        "schema": "b1k_execution_v1",
        "action_dim": 23,
        "input_denormalized": True,
        "policy_hz": 15,
        "native_hz": 30,
        "prediction_horizon": 32,
        "execute_horizon": 4,
        "ticks_per_policy_token": 2,
        "output_ticks": 8,
        "resampling": "zero_order_hold",
        "temporal_strategy": "baseline",
        "base_velocity_scale": 1.0,
        "base_stationary_threshold": None,
        "constant_zero_dims": [6],
        "binary_gripper_dims": [14, 22],
        "gripper_threshold": 0.0,
        "chunk_index": 0,
        "input_shape": [3, 32, 23],
        "output_shape": [3, 8, 23],
    }
    for key, value in expected.items():
        assert result.metadata[key] == value
    assert result.next_state == B1KExecutionState()
    json.dumps(result.metadata, allow_nan=False)
    assert "input_shape" not in wrapper.metadata


@pytest.mark.parametrize(
    ("chunk", "error", "message"),
    [
        (torch.zeros(32, 22), ValueError, "action dimension 23"),
        (torch.zeros(31, 23), ValueError, "prediction_horizon 32"),
        (torch.zeros(23), ValueError, r"shape \[\.\.\., tokens, 23\]"),
        ([[0.0] * 23] * 32, TypeError, "torch.Tensor or numpy.ndarray"),
    ],
)
def test_rejects_invalid_input_shape_and_type(chunk, error, message):
    with pytest.raises(error, match=message):
        prepare_b1k_execution(chunk)


@pytest.mark.parametrize("factory", [torch, np])
def test_rejects_nonfinite_input(factory):
    chunk = factory.zeros((32, 23))
    chunk[7, 3] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        prepare_b1k_execution(chunk)


def test_config_rejects_nonintegral_timing_and_invalid_horizons():
    with pytest.raises(ValueError, match="integer multiple"):
        B1KExecutionConfig(policy_hz=20, native_hz=30)
    with pytest.raises(ValueError, match="cannot exceed"):
        B1KExecutionConfig(prediction_horizon=4, execute_horizon=5)
    with pytest.raises(ValueError, match="positive integer"):
        B1KExecutionConfig(execute_horizon=0)
    with pytest.raises(ValueError, match="temporal_strategy must be one of"):
        B1KExecutionConfig(temporal_strategy="unknown")
    with pytest.raises(ValueError, match="only supports zero_order_hold"):
        B1KExecutionConfig(
            execute_horizon=16,
            temporal_strategy="speed_2x",
            resampling="hybrid_linear",
        )
    with pytest.raises(ValueError, match="finite non-negative"):
        B1KExecutionConfig(base_stationary_threshold=float("nan"))


@pytest.mark.parametrize(
    ("execute_horizon", "output_ticks"),
    [(8, 16), (16, 32), (32, 64), (48, 96), (64, 128)],
)
def test_longctx_zoh_horizons_expand_to_native_ticks(execute_horizon, output_ticks):
    chunk = np.zeros((64, 23), dtype=np.float32)
    chunk[:, 3] = np.arange(64)
    result = prepare_b1k_execution(
        chunk,
        B1KExecutionConfig(
            prediction_horizon=64,
            execute_horizon=execute_horizon,
        ),
    )

    assert result.ticks.shape == (output_ticks, 23)
    np.testing.assert_array_equal(
        result.ticks[:, 3], np.repeat(np.arange(execute_horizon), 2)
    )


@pytest.mark.parametrize("execute_horizon", [8, 16, 32, 48, 64])
def test_longctx_speed_2x_compresses_executed_prefix(execute_horizon):
    chunk = np.zeros((64, 23), dtype=np.float32)
    chunk[:, 3] = np.arange(64)
    result = prepare_b1k_execution(
        chunk,
        B1KExecutionConfig(
            prediction_horizon=64,
            execute_horizon=execute_horizon,
            temporal_strategy="speed_2x",
        ),
    )

    assert result.ticks.shape == (execute_horizon, 23)
    assert result.metadata["source_native_ticks"] == 2 * execute_horizon
    assert result.metadata["target_native_ticks"] == execute_horizon
    assert result.metadata["time_compression_ratio"] == 2.0
    assert result.metadata["execute_horizon"] == execute_horizon
    np.testing.assert_array_equal(result.ticks[:, 3], np.arange(execute_horizon))
    if execute_horizon < 64:
        assert result.metadata["qpos_resampling_method"] == (
            "linear_at_compressed_tick_starts_with_one_token_lookahead"
        )
        assert "qpos_lookahead_tokens" not in result.metadata
        assert "prediction_horizon_boundary" not in result.metadata
    else:
        assert result.metadata["qpos_resampling_method"] == (
            "linear_at_compressed_tick_starts_without_lookahead_at_full_prediction_horizon"
        )
        assert result.metadata["qpos_lookahead_tokens"] == 0
        assert result.metadata["prediction_horizon_boundary"] == (
            "execute_all_tokens_without_lookahead"
        )


def test_custom_valid_config_preserves_prefix_dimensions():
    config = B1KExecutionConfig(
        policy_hz=10,
        native_hz=30,
        prediction_horizon=6,
        execute_horizon=2,
    )
    chunk = np.zeros((2, 3, 6, 23), dtype=np.float32)
    result = prepare_b1k_execution(chunk, config)
    assert result.ticks.shape == (2, 3, 6, 23)
    assert result.metadata["ticks_per_policy_token"] == 3


@pytest.mark.parametrize("factory", [torch, np])
def test_hybrid_linear_interpolates_continuous_dims_and_holds_grippers(factory):
    chunk = factory.zeros((32, 23), dtype=factory.float64)
    for token in range(5):
        chunk[token] = token * 10
        chunk[token, 6] = 99
        chunk[token, 14] = -1 if token < 2 else 1
        chunk[token, 22] = 1 if token < 3 else -1
    original = chunk.copy() if factory is np else chunk.clone()

    result = prepare_b1k_execution(
        chunk, B1KExecutionConfig(resampling="hybrid_linear")
    )

    expected_continuous = (
        factory.tensor([0, 5, 10, 15, 20, 25, 30, 35], dtype=factory.float64)
        if factory is torch
        else np.array([0, 5, 10, 15, 20, 25, 30, 35], dtype=np.float64)
    )
    if factory is torch:
        assert torch.equal(result.ticks[:, 0], expected_continuous)
        assert torch.equal(result.ticks[:, 6], torch.zeros(8, dtype=torch.float64))
        assert torch.equal(
            result.ticks[:, 14],
            torch.tensor([-1, -1, -1, -1, 1, 1, 1, 1], dtype=torch.float64),
        )
        assert torch.equal(
            result.ticks[:, 22],
            torch.tensor([1, 1, 1, 1, 1, 1, -1, -1], dtype=torch.float64),
        )
        assert torch.equal(chunk, original)
    else:
        np.testing.assert_array_equal(result.ticks[:, 0], expected_continuous)
        np.testing.assert_array_equal(result.ticks[:, 6], np.zeros(8))
        np.testing.assert_array_equal(result.ticks[:, 14], [-1, -1, -1, -1, 1, 1, 1, 1])
        np.testing.assert_array_equal(result.ticks[:, 22], [1, 1, 1, 1, 1, 1, -1, -1])
        np.testing.assert_array_equal(chunk, original)
    assert result.metadata["resampling"] == "hybrid_linear"


def test_hybrid_linear_requires_following_prediction_token():
    with pytest.raises(ValueError, match="one prediction token"):
        B1KExecutionConfig(
            prediction_horizon=4,
            execute_horizon=4,
            resampling="hybrid_linear",
        )


@pytest.mark.parametrize("factory", [torch, np])
@pytest.mark.parametrize(
    ("strategy", "output_ticks"),
    [("speed_2x", 16), ("speed_1p5_m21", 21), ("speed_3x_m11", 11)],
)
def test_accelerated_resampling_preserves_type_shape_and_input(
    factory, strategy, output_ticks
):
    chunk = factory.zeros((32, 23), dtype=factory.float64)
    chunk[:, :3] = (
        factory.tensor([0.2, -0.1, 0.15]) if factory is torch else [0.2, -0.1, 0.15]
    )
    chunk[:, 3] = factory.arange(32, dtype=factory.float64)
    chunk[:, 6] = 99
    chunk[:, 14] = 1
    chunk[:, 22] = -1
    original = chunk.clone() if factory is torch else chunk.copy()

    result = prepare_b1k_execution(
        chunk,
        B1KExecutionConfig(execute_horizon=16, temporal_strategy=strategy),
        chunk_index=7,
    )
    if factory is torch:
        assert result.ticks.dtype == chunk.dtype
        assert torch.equal(chunk, original)
    else:
        assert result.ticks.dtype == chunk.dtype
        np.testing.assert_array_equal(chunk, original)
    assert result.ticks.shape == (output_ticks, 23)
    np.testing.assert_allclose(result.ticks[:, 6], 0)
    np.testing.assert_allclose(result.ticks[:, 14], 1)
    np.testing.assert_allclose(result.ticks[:, 22], -1)
    assert result.metadata["output_ticks"] == output_ticks
    assert result.metadata["chunk_index"] == 7
    assert result.metadata["algorithm_revision"] == "se2_trajectory_preserving_v1"
    assert result.metadata["source_native_ticks"] == 32
    assert result.metadata["target_native_ticks"] == output_ticks
    assert result.metadata["preclip_saturation_count"] == 0
    assert result.metadata["se2_preclip_endpoint_translation_error"] < 1e-12
    assert result.metadata["se2_preclip_endpoint_yaw_error"] < 1e-12
    assert result.metadata["se2_postclip_endpoint_translation_error"] < 1e-12
    assert result.metadata["se2_postclip_endpoint_yaw_error"] < 1e-12


@pytest.mark.parametrize(
    ("strategy", "output_ticks"),
    [("speed_1p5_m21", 21), ("speed_3x_m11", 11)],
)
def test_fractional_acceleration_linearly_interpolates_qpos_and_zoh_gripper(
    strategy, output_ticks
):
    chunk = np.zeros((32, 23), dtype=np.float64)
    chunk[:, 3] = np.arange(32)
    chunk[:, 14] = np.where(np.arange(32) < 5, -1, 1)
    chunk[:, 22] = 1
    result = prepare_b1k_execution(
        chunk, B1KExecutionConfig(execute_horizon=16, temporal_strategy=strategy)
    )
    coordinates = np.arange(output_ticks) * 16 / output_ticks
    np.testing.assert_allclose(result.ticks[:, 3], coordinates, atol=1e-12)
    source_indices = np.floor(coordinates).astype(int)
    np.testing.assert_array_equal(
        result.ticks[:, 14], np.where(source_indices < 5, -1, 1)
    )


@pytest.mark.parametrize(
    "strategy", ["speed_2x", "speed_1p5_m21", "speed_3x_m11"]
)
def test_se2_arc_endpoint_is_preserved_without_saturation(strategy):
    chunk = np.zeros((32, 23), dtype=np.float64)
    chunk[:16, :3] = [0.2, 0.08, 0.15]
    result = prepare_b1k_execution(
        chunk, B1KExecutionConfig(execute_horizon=16, temporal_strategy=strategy)
    )
    assert result.metadata["preclip_saturation_count"] == 0
    assert result.metadata["se2_preclip_endpoint_translation_error"] < 1e-12
    assert result.metadata["se2_preclip_endpoint_yaw_error"] < 1e-12
    assert result.metadata["se2_postclip_endpoint_translation_error"] < 1e-12
    assert result.metadata["se2_postclip_endpoint_yaw_error"] < 1e-12


@pytest.mark.parametrize("strategy", ["speed_1p5_m21", "speed_3x_m11"])
def test_fractional_acceleration_preserves_varying_se2_path_endpoint(strategy):
    rng = np.random.default_rng(7)
    chunk = np.zeros((32, 23), dtype=np.float64)
    chunk[:16, :3] = rng.uniform([-0.2, -0.2, -0.15], [0.2, 0.2, 0.15], (16, 3))
    result = prepare_b1k_execution(
        chunk,
        B1KExecutionConfig(execute_horizon=16, temporal_strategy=strategy),
    )
    assert result.metadata["preclip_saturation_count"] == 0
    assert result.metadata["se2_preclip_endpoint_translation_error"] < 1e-12
    assert result.metadata["se2_preclip_endpoint_yaw_error"] < 1e-12
    assert result.metadata["se2_postclip_endpoint_translation_error"] < 1e-12
    assert result.metadata["se2_postclip_endpoint_yaw_error"] < 1e-12


def test_accelerated_base_clips_and_reports_saturation():
    chunk = np.zeros((32, 23), dtype=np.float64)
    chunk[:16, 0] = 0.8
    result = prepare_b1k_execution(
        chunk, B1KExecutionConfig(execute_horizon=16, temporal_strategy="speed_2x")
    )
    assert np.max(np.abs(result.ticks[:, :3])) <= 1
    assert result.metadata["preclip_saturation_count"] == 16
    assert result.metadata["max_preclip_abs"] == pytest.approx(1.6)
    assert result.metadata["se2_preclip_endpoint_translation_error"] < 1e-12
    assert result.metadata["se2_postclip_endpoint_translation_error"] > 0


@pytest.mark.parametrize("factory", [torch, np])
def test_adaptive_hysteresis_lookahead_and_debounce(factory):
    chunk = factory.zeros((32, 23), dtype=factory.float64)
    chunk[:, 3] = factory.arange(32, dtype=factory.float64)
    chunk[:, 14] = 1
    chunk[:, 22] = 1
    chunk[1, 0] = 0.1
    original = chunk.clone() if factory is torch else chunk.copy()
    config = B1KExecutionConfig(
        execute_horizon=16, temporal_strategy="adaptive_upper_body_2x"
    )

    result = prepare_b1k_execution(chunk, config)
    expected_indices = np.array([0, 0, 1, 1, 2, 2, *range(3, 16)])
    expected = original[expected_indices]
    if factory is torch:
        assert torch.equal(result.ticks, expected)
        assert torch.equal(chunk, original)
    else:
        np.testing.assert_array_equal(result.ticks, expected)
        np.testing.assert_array_equal(chunk, original)
    assert result.metadata["moving_tokens"] == 3
    assert result.metadata["stationary_tokens"] == 13
    assert result.metadata["state_transitions"] == 2
    assert result.next_state == B1KExecutionState()


def test_adaptive_state_crosses_chunk_and_wrapper_reset():
    config = B1KExecutionConfig(
        execute_horizon=16, temporal_strategy="adaptive_upper_body_2x"
    )
    wrapper = B1KExecutionWrapper(config)
    first = np.zeros((32, 23), dtype=np.float64)
    first[15, 2] = 0.1
    result1 = wrapper(first)
    assert result1.next_state.adaptive_moving
    assert wrapper.state.adaptive_moving

    result2 = wrapper(np.zeros((32, 23), dtype=np.float64))
    assert result2.ticks.shape == (17, 23)
    assert result2.metadata["start_state"]["moving"]
    assert not result2.next_state.adaptive_moving
    wrapper.reset()
    assert wrapper.state == B1KExecutionState()


def test_adaptive_lookahead_prevents_low_streak_from_stopping_motion():
    chunk = np.zeros((32, 23), dtype=np.float64)
    chunk[:, 3] = np.arange(32)
    chunk[1, 0] = 0.1
    result = prepare_b1k_execution(
        chunk,
        B1KExecutionConfig(
            execute_horizon=16, temporal_strategy="adaptive_upper_body_2x"
        ),
        state=B1KExecutionState(adaptive_moving=True, adaptive_low_streak=1),
    )
    assert result.metadata["token_repeats"][:4] == [2, 2, 2, 1]


@pytest.mark.parametrize(("base_velocity", "output_ticks"), [(0.0, 16), (0.5, 32)])
def test_adaptive_upper_body_2x_output_tick_bounds(base_velocity, output_ticks):
    chunk = np.zeros((32, 23), dtype=np.float32)
    chunk[:16, 0] = base_velocity
    config = B1KExecutionConfig(
        execute_horizon=16, temporal_strategy="adaptive_upper_body_2x"
    )

    result = prepare_b1k_execution(chunk, config)

    assert result.ticks.shape == (output_ticks, 23)
    assert result.metadata["output_ticks"] == output_ticks


def test_accelerated_strategies_reject_batched_chunks_and_invalid_chunk_index():
    config = B1KExecutionConfig(execute_horizon=16, temporal_strategy="speed_2x")
    with pytest.raises(ValueError, match="only supports an unbatched"):
        prepare_b1k_execution(np.zeros((2, 32, 23)), config)
    with pytest.raises(ValueError, match="chunk_index must be a non-negative integer"):
        prepare_b1k_execution(np.zeros((32, 23)), config, chunk_index=-1)
