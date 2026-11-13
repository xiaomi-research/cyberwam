# Copyright (c) 2026 Xiaomi Corporation

import pytest
import torch

from fastwam.utils.pytorch_utils import (
    INITIALIZATION_FINGERPRINT_ATTRIBUTE,
    attach_random_module_initialization_fingerprint,
    compute_random_module_initialization_fingerprint,
)


class _FingerprintModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.action_expert = torch.nn.Module()
        self.action_expert.action_encoder = torch.nn.Linear(3, 5)
        self.action_expert.head = torch.nn.Linear(5, 3)
        self.action_expert.pretrained_backbone = torch.nn.Linear(5, 5)
        self.proprio_encoder = torch.nn.Linear(3, 5)


def test_initialization_fingerprint_is_stable_and_attached():
    torch.manual_seed(7)
    first = _FingerprintModel()
    torch.manual_seed(7)
    second = _FingerprintModel()

    expected = compute_random_module_initialization_fingerprint(first)
    assert expected == compute_random_module_initialization_fingerprint(second)
    assert attach_random_module_initialization_fingerprint(first) == expected
    assert getattr(first, INITIALIZATION_FINGERPRINT_ATTRIBUTE) == expected


def test_initialization_fingerprint_only_tracks_random_modules():
    torch.manual_seed(11)
    model = _FingerprintModel()
    original = compute_random_module_initialization_fingerprint(model)

    with torch.no_grad():
        model.action_expert.pretrained_backbone.weight.add_(1)
    assert compute_random_module_initialization_fingerprint(model) == original

    with torch.no_grad():
        model.action_expert.action_encoder.weight[0, 0].add_(1)
    assert compute_random_module_initialization_fingerprint(model) != original


def test_initialization_fingerprint_rejects_model_without_target_parameters():
    with pytest.raises(ValueError, match="none of action_encoder"):
        compute_random_module_initialization_fingerprint(torch.nn.Linear(2, 2))
