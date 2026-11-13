# Copyright (c) 2026 Xiaomi Corporation

import pytest
import torch

from fastwam.trainer import Wan22Trainer


class _LegacyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dit = torch.nn.Linear(2, 2)
        self.proprio_encoder = torch.nn.Linear(2, 2)
        self.vae = torch.nn.Linear(2, 2)


class _MultimodalModel(torch.nn.Module):
    trainable_module_names = (
        "dit",
        "dino_adapter",
        "pointmap_head",
        "proprio_encoder",
    )

    def __init__(self):
        super().__init__()
        self.dit = torch.nn.Linear(2, 2)
        self.dino_adapter = torch.nn.Linear(2, 2)
        self.pointmap_head = torch.nn.Linear(2, 2)
        self.proprio_encoder = torch.nn.Linear(2, 2)
        self.dino_encoder = torch.nn.Linear(2, 2)
        self.vae = torch.nn.Linear(2, 2)


def test_legacy_training_boundary_remains_dit_and_proprio_only():
    model = _LegacyModel()

    Wan22Trainer._apply_dit_only_train_mode(model)

    assert model.dit.training and model.proprio_encoder.training
    assert not model.vae.training
    assert all(parameter.requires_grad for parameter in model.dit.parameters())
    assert all(parameter.requires_grad for parameter in model.proprio_encoder.parameters())
    assert all(not parameter.requires_grad for parameter in model.vae.parameters())


def test_multimodal_training_boundary_includes_adapters_and_keeps_encoders_frozen():
    model = _MultimodalModel()

    Wan22Trainer._apply_dit_only_train_mode(model)
    parameters = Wan22Trainer._trainable_parameters(model)

    expected = {
        id(parameter)
        for name in model.trainable_module_names
        for parameter in getattr(model, name).parameters()
    }
    assert {id(parameter) for parameter in parameters} == expected
    assert model.dino_adapter.training and model.pointmap_head.training
    assert not model.dino_encoder.training and not model.vae.training
    assert all(not parameter.requires_grad for parameter in model.dino_encoder.parameters())
    assert all(not parameter.requires_grad for parameter in model.vae.parameters())


def test_trainable_parameters_deduplicates_nested_or_aliased_modules():
    model = _MultimodalModel()
    model.dit = torch.nn.ModuleDict({"adapter": model.dino_adapter})

    Wan22Trainer._apply_dit_only_train_mode(model)
    parameters = Wan22Trainer._trainable_parameters(model)

    assert len(parameters) == len({id(parameter) for parameter in parameters})


def test_invalid_declared_trainable_module_fails_closed():
    model = _MultimodalModel()
    model.trainable_module_names = ("missing_adapter",)

    with pytest.raises(AttributeError, match="missing_adapter"):
        Wan22Trainer._apply_dit_only_train_mode(model)
