# Copyright (c) 2026 Xiaomi Corporation

from __future__ import annotations

from pathlib import Path

import torch
import pytest

pytest.importorskip("einops")
from fastwam.models.wan22.helpers import loader


def test_dit_only_loader_does_not_resolve_vae_t5_or_tokenizer(
    tmp_path: Path, monkeypatch
) -> None:
    calls: list[str] = []

    def fake_download(self) -> None:
        calls.append(str(self.origin_file_pattern))
        self.path = str(tmp_path / "dit.safetensors")

    expected_model = object()

    def fake_load(path, model_name, torch_dtype, device, model_kwargs_override=None):
        assert model_name == "wan_video_dit"
        assert model_kwargs_override == {"hidden_dim": 1}
        return expected_model

    monkeypatch.setattr(loader, "_validate_dit_config", lambda value: value)
    monkeypatch.setattr(loader.ModelConfig, "download_if_necessary", fake_download)
    monkeypatch.setattr(loader, "_load_registered_model", fake_load)

    model, path = loader.load_wan22_ti2v_5b_dit(
        device="cpu",
        torch_dtype=torch.float32,
        dit_config={"hidden_dim": 1},
    )

    assert model is expected_model
    assert path == str(tmp_path / "dit.safetensors")
    assert calls == ["diffusion_pytorch_model*.safetensors"]
