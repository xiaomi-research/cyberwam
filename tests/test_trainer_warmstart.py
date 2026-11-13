# Copyright (c) 2026 Xiaomi Corporation

from types import SimpleNamespace

import pytest

import fastwam.models.wan22.helpers.flex_warmstart as flex_warmstart_module
from fastwam.trainer import (
    Wan22Trainer,
    _resolve_initial_checkpoint_sources,
    _validate_initial_checkpoint_sources,
)


@pytest.mark.parametrize(
    "values",
    [
        {"resume": "state", "legacy_warmstart": "legacy.pt"},
        {"resume": "state", "flex_warmstart": "flex.pt"},
        {
            "resume": None,
            "legacy_warmstart": "legacy.pt",
            "flex_warmstart": "flex.pt",
        },
    ],
)
def test_initial_checkpoint_sources_are_mutually_exclusive(values):
    with pytest.raises(ValueError, match="mutually exclusive"):
        _validate_initial_checkpoint_sources(**values)


def test_initial_checkpoint_sources_use_canonical_fields():
    assert _resolve_initial_checkpoint_sources(
        resume=None,
        legacy_warmstart="legacy.pt",
        flex_warmstart=None,
    ) == ("legacy.pt", None)
    assert _resolve_initial_checkpoint_sources(
        resume=None,
        legacy_warmstart=None,
        flex_warmstart="flex.pt",
    ) == (None, "flex.pt")


def test_flex_warmstart_uses_unwrapped_model(monkeypatch):
    model = object()
    wrapped = object()
    calls = []
    expected_report = SimpleNamespace(
        path="flex.pt",
        source_step=123,
        mot_tensor_count=10,
        proprio_tensor_count=2,
        adapter_tensor_count=8,
        adapter_names=("dino_embedder", "pt_head"),
    )
    monkeypatch.setattr(
        flex_warmstart_module,
        "load_flex_warmstart",
        lambda target, path: calls.append((target, path)) or expected_report,
    )
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.model = wrapped
    trainer.accelerator = SimpleNamespace(unwrap_model=lambda value: model)
    trainer.flex_warmstart = "flex.pt"
    trainer.legacy_warmstart = None
    trainer.resume = None

    trainer._resume_or_load_checkpoint()

    assert calls == [(model, "flex.pt")]
