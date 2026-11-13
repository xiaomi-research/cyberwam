# Copyright (c) 2026 Xiaomi Corporation

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from fastwam.models.wan22.helpers.flex_warmstart import load_flex_warmstart


class _FlexModel(nn.Module):
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
        self.mot = nn.ModuleDict({"video": self.video_expert, "action": self.action_expert})
        self.proprio_encoder = nn.Linear(23, 4)
        self.dino_embedder = nn.Linear(8, 4)
        self.dino_proj_out = nn.Linear(4, 8)
        self.pt_patch_embedding = nn.Linear(3, 4)
        self.pt_head = nn.Linear(4, 3)


def _fill(module: nn.Module, value: float) -> None:
    with torch.no_grad():
        for tensor in module.parameters():
            tensor.fill_(value)


def _save_flex(path: Path, *, step: int = 100) -> _FlexModel:
    source = _FlexModel()
    _fill(source.mot, 2.0)
    _fill(source.proprio_encoder, 3.0)
    _fill(source.dino_embedder, 4.0)
    _fill(source.dino_proj_out, 5.0)
    _fill(source.pt_patch_embedding, 6.0)
    _fill(source.pt_head, 7.0)
    torch.save(
        {
            "mot": source.mot.state_dict(),
            "proprio_encoder": source.proprio_encoder.state_dict(),
            "multimodal": {
                name: getattr(source, name).state_dict()
                for name in source.multimodal_new_state_prefixes
            },
            "optimizer": {"must": "not be loaded"},
            "step": step,
        },
        path,
    )
    return source


def test_flex_warmstart_loads_mot_proprio_and_all_multimodal_adapters(tmp_path):
    path = tmp_path / "multimodal.pt"
    source = _save_flex(path, step=83800)
    target = _FlexModel()
    _fill(target.mot, -1.0)
    _fill(target.proprio_encoder, -2.0)
    _fill(target.dino_embedder, -3.0)

    report = load_flex_warmstart(target, path)

    assert report.path == path.resolve()
    assert report.source_step == 83800
    assert report.adapter_names == target.multimodal_new_state_prefixes
    assert report.adapter_tensor_count == sum(
        len(getattr(target, name).state_dict()) for name in target.multimodal_new_state_prefixes
    )
    for name in ("mot", "proprio_encoder", *target.multimodal_new_state_prefixes):
        source_state = getattr(source, name).state_dict() if name != "mot" else source.mot.state_dict()
        target_state = getattr(target, name).state_dict() if name != "mot" else target.mot.state_dict()
        for key, value in source_state.items():
            torch.testing.assert_close(target_state[key], value)


@pytest.mark.parametrize("missing", ["mot", "proprio_encoder", "multimodal"])
def test_flex_warmstart_requires_multimodal_payloads(tmp_path, missing):
    path = tmp_path / "multimodal.pt"
    _save_flex(path)
    payload = torch.load(path, map_location="cpu")
    del payload[missing]
    torch.save(payload, path)

    with pytest.raises(ValueError, match=missing):
        load_flex_warmstart(_FlexModel(), path)


def test_flex_warmstart_validates_everything_before_mutating(tmp_path):
    path = tmp_path / "multimodal.pt"
    _save_flex(path)
    payload = torch.load(path, map_location="cpu")
    first_key = next(iter(payload["multimodal"]["dino_embedder"]))
    payload["multimodal"]["dino_embedder"][first_key] = torch.zeros(1)
    torch.save(payload, path)

    target = _FlexModel()
    before = {key: value.clone() for key, value in target.mot.state_dict().items()}
    with pytest.raises(RuntimeError, match="shape mismatch.*multimodal.dino_embedder"):
        load_flex_warmstart(target, path)
    for key, value in before.items():
        torch.testing.assert_close(target.mot.state_dict()[key], value)


def test_flex_warmstart_rejects_adapter_module_mismatch(tmp_path):
    path = tmp_path / "multimodal.pt"
    _save_flex(path)
    payload = torch.load(path, map_location="cpu")
    del payload["multimodal"]["pt_head"]
    torch.save(payload, path)

    with pytest.raises(RuntimeError, match="module mismatch.*pt_head"):
        load_flex_warmstart(_FlexModel(), path)
