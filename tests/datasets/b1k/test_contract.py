# Copyright (c) 2026 Xiaomi Corporation

import numpy as np
import pytest
import torch

from fastwam.datasets.b1k.contract import (
    ACTION_LABELS,
    PROPRIO_LABELS,
    B1KActionStateTransform,
    extract_b1k_proprio,
    extract_r1pro_simulator_proprio,
    postprocess_b1k_action,
    validate_b1k_action,
)


def _valid_action(*shape: int) -> torch.Tensor:
    action = torch.zeros(*shape, 23)
    action[..., 14] = -1
    action[..., 22] = 1
    return action


def test_labels_capture_confirmed_action_and_proprio_semantics():
    assert ACTION_LABELS[7] == "left_arm_absolute_qpos_0"
    assert ACTION_LABELS[15] == "right_arm_absolute_qpos_0"
    assert PROPRIO_LABELS[7] == "left_arm_qpos_0"
    assert PROPRIO_LABELS[15] == "right_arm_qpos_0"


@pytest.mark.parametrize("factory", [torch.arange, np.arange])
def test_extract_b1k_proprio_exact_mapping(factory):
    raw = factory(2 * 61, dtype=torch.float32 if factory is torch.arange else np.float32).reshape(2, 61)
    proprio = extract_b1k_proprio(raw)
    expected = [
        *range(0, 3),
        *range(53, 57),
        *range(3, 10),
        24 + 25,
        *range(28, 35),
        49 + 50,
    ]
    np.testing.assert_allclose(np.asarray(proprio[0]), expected)
    np.testing.assert_allclose(np.asarray(proprio[1]), np.asarray(expected) + np.array([
        *([61] * 14), 122, *([61] * 7), 122
    ]))


def test_extract_rejects_bad_shape_and_nonfinite_state():
    with pytest.raises(ValueError, match="last dimension 61"):
        extract_b1k_proprio(torch.zeros(60))
    raw = torch.zeros(61)
    raw[3] = torch.nan
    with pytest.raises(ValueError, match="non-finite"):
        extract_b1k_proprio(raw)


@pytest.mark.parametrize("factory", [torch.arange, np.arange])
def test_extract_r1pro_simulator_proprio_exact_mapping(factory):
    dtype = torch.float32 if factory is torch.arange else np.float32
    raw = factory(256, dtype=dtype)

    proprio = extract_r1pro_simulator_proprio(raw)

    expected = [
        *range(253, 256),
        *range(236, 240),
        *range(158, 165),
        193 + 194,
        *range(197, 204),
        232 + 233,
    ]
    np.testing.assert_allclose(np.asarray(proprio), expected)


def test_validate_action_rejects_contract_violations():
    validate_b1k_action(_valid_action(2, 3))

    action = _valid_action(1)
    action[0, 6] = 0.1
    with pytest.raises(ValueError, match="constant zero"):
        validate_b1k_action(action)

    action = _valid_action(1)
    action[0, 14] = 0
    with pytest.raises(ValueError, match=r"only -1 or \+1"):
        validate_b1k_action(action)

    action = _valid_action(1)
    action[0, 0] = torch.inf
    with pytest.raises(ValueError, match="non-finite"):
        validate_b1k_action(action)


def test_postprocess_forces_constant_and_thresholds_grippers():
    prediction = torch.zeros(2, 23)
    prediction[:, 6] = 5
    prediction[:, 14] = torch.tensor([-0.01, 0.0])
    prediction[:, 22] = torch.tensor([0.2, -0.2])
    result = postprocess_b1k_action(prediction)
    assert torch.equal(result[:, 6], torch.zeros(2))
    assert torch.equal(result[:, 14], torch.tensor([-1.0, 1.0]))
    assert torch.equal(result[:, 22], torch.tensor([1.0, -1.0]))
    assert prediction[:, 6].eq(5).all()


def test_action_state_transform_forward_and_backward():
    transform = B1KActionStateTransform()
    batch = {
        "state": {"default": torch.zeros(4, 61)},
        "action": {"default": _valid_action(3)},
    }
    result = transform.forward(batch)
    assert result["state"]["default"].shape == (4, 23)

    result["action"]["default"][:, 6] = 9
    result["action"]["default"][:, 14] = 0
    result = transform.backward(result)
    assert result["action"]["default"][:, 6].eq(0).all()
    assert result["action"]["default"][:, 14].eq(1).all()
