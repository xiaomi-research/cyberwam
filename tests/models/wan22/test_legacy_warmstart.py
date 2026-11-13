# Copyright (c) 2026 Xiaomi Corporation

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from fastwam.models.wan22.helpers.legacy_warmstart import load_legacy_warmstart


class _LegacyModel(nn.Module):
    multimodal_new_state_prefixes = (
        "dino_embedder",
        "dino_proj_out",
        "pt_patch_embedding",
        "pt_head",
    )

    def __init__(self):
        super().__init__()
        self.video_expert = nn.Module()
        self.video_expert.patch_embedding = nn.Linear(3, 4)
        self.video_expert.head = nn.Linear(4, 3)
        self.action_expert = nn.Linear(2, 2)
        self.mot = nn.ModuleDict(
            {"video": self.video_expert, "action": self.action_expert}
        )
        self.proprio_encoder = nn.Linear(23, 4)
        self.dino_embedder = nn.Linear(8, 4)
        self.dino_proj_out = nn.Linear(4, 8)
        self.pt_patch_embedding = nn.Linear(3, 4)
        self.pt_head = nn.Linear(4, 3)
        self.copy_count = 0

    def initialize_pointmap_from_rgb(self):
        self.pt_patch_embedding.load_state_dict(
            self.video_expert.patch_embedding.state_dict(), strict=True
        )
        self.pt_head.load_state_dict(self.video_expert.head.state_dict(), strict=True)
        self.copy_count += 1


class _SameArchitectureModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mot = nn.Linear(4, 4)
        self.proprio_encoder = nn.Linear(23, 4)


def _fill(module: nn.Module, value: float) -> None:
    with torch.no_grad():
        for tensor in module.parameters():
            tensor.fill_(value)


def _save_source(path: Path, *, step: int = 83800) -> _LegacyModel:
    source = _LegacyModel()
    _fill(source.mot, 2.0)
    _fill(source.proprio_encoder, 3.0)
    torch.save(
        {
            "mot": source.mot.state_dict(),
            "proprio_encoder": source.proprio_encoder.state_dict(),
            "optimizer": {"must": "not be loaded"},
            "step": step,
        },
        path,
    )
    return source


def test_legacy_warmstart_loads_legacy_weights_then_copies_rgb_to_pointmap(tmp_path):
    path = tmp_path / "step_083800.pt"
    source = _save_source(path)
    target = _LegacyModel()
    _fill(target.mot, -1.0)
    _fill(target.proprio_encoder, -2.0)
    _fill(target.pt_patch_embedding, 7.0)
    _fill(target.pt_head, 7.0)

    report = load_legacy_warmstart(target, path)

    assert report.path == path.resolve()
    assert report.source_step == 83800
    assert report.new_state_prefixes == target.multimodal_new_state_prefixes
    assert target.copy_count == 1
    for key, value in source.mot.state_dict().items():
        torch.testing.assert_close(target.mot.state_dict()[key], value)
    for key, value in source.proprio_encoder.state_dict().items():
        torch.testing.assert_close(target.proprio_encoder.state_dict()[key], value)
    for key, value in target.video_expert.patch_embedding.state_dict().items():
        torch.testing.assert_close(target.pt_patch_embedding.state_dict()[key], value)
    for key, value in target.video_expert.head.state_dict().items():
        torch.testing.assert_close(target.pt_head.state_dict()[key], value)


def test_legacy_warmstart_loads_same_architecture_without_pointmap_hook(tmp_path):
    path = tmp_path / "step_083800.pt"
    source = _SameArchitectureModel()
    _fill(source.mot, 2.0)
    _fill(source.proprio_encoder, 3.0)
    torch.save(
        {
            "mot": source.mot.state_dict(),
            "proprio_encoder": source.proprio_encoder.state_dict(),
            "step": 83800,
        },
        path,
    )
    target = _SameArchitectureModel()
    _fill(target.mot, -1.0)
    _fill(target.proprio_encoder, -2.0)

    report = load_legacy_warmstart(target, path)

    assert report.source_step == 83800
    assert report.new_state_prefixes == ()
    for key, value in source.mot.state_dict().items():
        torch.testing.assert_close(target.mot.state_dict()[key], value)
    for key, value in source.proprio_encoder.state_dict().items():
        torch.testing.assert_close(target.proprio_encoder.state_dict()[key], value)


@pytest.mark.parametrize("mutation", ["missing", "unexpected", "shape"])
def test_legacy_warmstart_rejects_inexact_mot_without_partial_load(tmp_path, mutation):
    path = tmp_path / "source.pt"
    _save_source(path)
    payload = torch.load(path, map_location="cpu")
    first_key = next(iter(payload["mot"]))
    if mutation == "missing":
        del payload["mot"][first_key]
        match = "missing keys"
    elif mutation == "unexpected":
        payload["mot"]["unexpected"] = torch.zeros(1)
        match = "unexpected keys"
    else:
        payload["mot"][first_key] = torch.zeros(1)
        match = "shape mismatch"
    torch.save(payload, path)

    target = _LegacyModel()
    before = {key: value.clone() for key, value in target.mot.state_dict().items()}
    with pytest.raises(RuntimeError, match=match):
        load_legacy_warmstart(target, path)
    assert target.copy_count == 0
    for key, value in before.items():
        torch.testing.assert_close(target.mot.state_dict()[key], value)


def test_legacy_warmstart_validates_proprio_before_loading_mot(tmp_path):
    path = tmp_path / "source.pt"
    _save_source(path)
    payload = torch.load(path, map_location="cpu")
    payload["proprio_encoder"]["weight"] = torch.zeros(1)
    torch.save(payload, path)

    target = _LegacyModel()
    before = {key: value.clone() for key, value in target.mot.state_dict().items()}
    with pytest.raises(RuntimeError, match="proprio_encoder.weight.*shape mismatch|shape mismatch.*proprio_encoder.weight"):
        load_legacy_warmstart(target, path)
    assert target.copy_count == 0
    for key, value in before.items():
        torch.testing.assert_close(target.mot.state_dict()[key], value)


@pytest.mark.parametrize("missing_payload", ["mot", "proprio_encoder"])
def test_legacy_warmstart_requires_both_legacy_payloads(tmp_path, missing_payload):
    path = tmp_path / "source.pt"
    _save_source(path)
    payload = torch.load(path, map_location="cpu")
    del payload[missing_payload]
    torch.save(payload, path)

    with pytest.raises(ValueError, match=missing_payload):
        load_legacy_warmstart(_LegacyModel(), path)


def test_legacy_warmstart_rejects_undeclared_or_nonexistent_new_prefixes(tmp_path):
    path = tmp_path / "source.pt"
    _save_source(path)
    model = _LegacyModel()
    model.multimodal_new_state_prefixes = ("does_not_exist",)

    with pytest.raises(RuntimeError, match="prefixes with no parameters"):
        load_legacy_warmstart(model, path)
