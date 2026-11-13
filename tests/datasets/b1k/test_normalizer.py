# Copyright (c) 2026 Xiaomi Corporation

import pytest
import torch

from fastwam.datasets.b1k.normalizer import B1KMixedNormalizer


def _normalizer() -> B1KMixedNormalizer:
    return B1KMixedNormalizer(
        action_mean=torch.arange(23, dtype=torch.float64),
        action_std=torch.full((23,), 2.0, dtype=torch.float64),
        proprio_mean=torch.arange(23, dtype=torch.float64) - 5,
        proprio_std=torch.arange(1, 24, dtype=torch.float64),
    )


def test_action_mixed_normalization_supports_arbitrary_leading_dims_without_clamp():
    normalizer = _normalizer()
    action = torch.arange(23, dtype=torch.float32).expand(2, 3, 4, 23).clone()
    action[..., 0] = 1_000
    action[..., 6] = 99
    action[..., 14] = -0.01
    action[..., 22] = 0

    normalized = normalizer.normalize_action(action)

    assert normalized.shape == (2, 3, 4, 23)
    assert normalized.dtype == action.dtype
    assert normalized[..., 0].eq(500).all()
    assert normalized[..., 1:6].eq(0).all()
    assert normalized[..., 6].eq(0).all()
    assert normalized[..., 14].eq(-1).all()
    assert normalized[..., 22].eq(1).all()
    assert action[..., 6].eq(99).all(), "normalization must not mutate its input"


def test_action_inverse_round_trip_including_distribution_tail():
    normalizer = _normalizer()
    action = torch.arange(23, dtype=torch.float64).repeat(5, 1)
    action[:, 0:6] += torch.linspace(-40, 40, 5).unsqueeze(-1)
    action[:, 6] = 0
    action[:, 7:14] -= 1
    action[:, 14] = torch.tensor([-1, 1, -1, 1, -1], dtype=torch.float64)
    action[:, 15:22] += 1
    action[:, 22] = torch.tensor([1, -1, 1, -1, 1], dtype=torch.float64)

    restored = normalizer.denormalize_action(normalizer.normalize_action(action))

    torch.testing.assert_close(restored, action, atol=1e-7, rtol=1e-7)


def test_action_denormalization_preserves_continuous_tails_and_enforces_special_dims():
    normalizer = _normalizer()
    prediction = torch.zeros(2, 23)
    prediction[:, 0] = torch.tensor([-10.0, 10.0])
    prediction[:, 6] = -4
    prediction[:, 14] = torch.tensor([-0.001, 0.0])
    prediction[:, 22] = torch.tensor([3.0, -3.0])

    action = normalizer.denormalize_action(prediction)

    torch.testing.assert_close(action[:, 0], torch.tensor([-20.0, 20.0]))
    assert action[:, 6].eq(0).all()
    assert torch.equal(action[:, 14], torch.tensor([-1.0, 1.0]))
    assert torch.equal(action[:, 22], torch.tensor([1.0, -1.0]))


def test_proprio_zscores_every_dimension_clamps_and_round_trips():
    normalizer = _normalizer()
    mean = torch.arange(23, dtype=torch.float64) - 5
    std = torch.arange(1, 24, dtype=torch.float64)
    normalized_source = torch.linspace(-4, 4, 23, dtype=torch.float64)
    proprio = mean + normalized_source * (std + 1e-8)

    normalized = normalizer.normalize_proprio(proprio.reshape(1, 1, 23))
    restored = normalizer.denormalize_proprio(normalized)

    torch.testing.assert_close(normalized[0, 0], normalized_source)
    torch.testing.assert_close(restored[0, 0], proprio)

    outlier = proprio.clone()
    outlier[14] = mean[14] + 100 * (std[14] + 1e-8)
    assert normalizer.normalize_proprio(outlier)[14].item() == 5


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"action_mean": torch.zeros(22)}, r"action_mean must have shape \(23,\)"),
        ({"action_std": torch.full((23,), -1.0)}, "non-negative"),
        ({"proprio_mean": torch.full((23,), torch.nan)}, "finite"),
        ({"proprio_std": torch.zeros(2, 23)}, r"proprio_std must have shape \(23,\)"),
    ],
)
def test_stats_validation_is_explicit(kwargs, message):
    valid = {
        "action_mean": torch.zeros(23),
        "action_std": torch.ones(23),
        "proprio_mean": torch.zeros(23),
        "proprio_std": torch.ones(23),
    }
    valid.update(kwargs)
    with pytest.raises(ValueError, match=message):
        B1KMixedNormalizer(**valid)


def test_input_validation_is_explicit():
    normalizer = _normalizer()
    with pytest.raises(ValueError, match="last dimension 23"):
        normalizer.normalize_action(torch.zeros(2, 22))
    with pytest.raises(TypeError, match="floating-point dtype"):
        normalizer.normalize_proprio(torch.zeros(23, dtype=torch.int64))
    with pytest.raises(ValueError, match="finite"):
        value = torch.zeros(23)
        value[0] = torch.inf
        normalizer.denormalize_action(value)


def test_zero_std_is_stable_for_dimensions_equal_to_the_mean():
    normalizer = B1KMixedNormalizer(
        action_mean=torch.zeros(23),
        action_std=torch.zeros(23),
        proprio_mean=torch.zeros(23),
        proprio_std=torch.zeros(23),
    )
    action = torch.zeros(23)
    action[14] = -1
    action[22] = 1

    normalized_action = normalizer.normalize_action(action)
    normalized_proprio = normalizer.normalize_proprio(torch.zeros(23))

    assert torch.isfinite(normalized_action).all()
    assert torch.isfinite(normalized_proprio).all()
    assert normalized_action[6].item() == 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_zero_std_uses_float32_internal_math(dtype):
    normalizer = B1KMixedNormalizer(
        action_mean=torch.zeros(23),
        action_std=torch.zeros(23),
        proprio_mean=torch.zeros(23),
        proprio_std=torch.zeros(23),
    )
    action = torch.zeros(23, dtype=dtype)
    action[14] = -1
    action[22] = 1

    normalized_action = normalizer.normalize_action(action)
    normalized_proprio = normalizer.normalize_proprio(torch.zeros(23, dtype=dtype))

    assert normalized_action.dtype == dtype
    assert normalized_proprio.dtype == dtype
    assert torch.isfinite(normalized_action).all()
    assert torch.isfinite(normalized_proprio).all()
