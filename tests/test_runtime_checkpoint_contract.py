# Copyright (c) 2026 Xiaomi Corporation

from __future__ import annotations

import torch
from omegaconf import OmegaConf

import fastwam.datasets.b1k.checkpoint as checkpoint_module
import fastwam.runtime as runtime
import fastwam.utils.pytorch_utils as pytorch_utils


def test_build_datasets_does_not_alias_train_as_validation(monkeypatch):
    train_dataset = object()
    monkeypatch.setattr(runtime, "instantiate", lambda *_args, **_kwargs: train_dataset)
    cfg = OmegaConf.create({"train": {"_target_": "example.Dataset"}, "val": None})

    actual_train, actual_val = runtime.build_datasets(cfg)

    assert actual_train is train_dataset
    assert actual_val is None


def test_build_datasets_adds_weighted_sources_without_changing_validation(monkeypatch):
    calls = []

    class FakeDataset:
        def __init__(self, root):
            self.root = root

        def __len__(self):
            return 10

    def fake_instantiate(cfg, **kwargs):
        resolved = OmegaConf.to_container(cfg, resolve=True)
        calls.append((resolved, kwargs))
        return FakeDataset(resolved["dataset_root"])

    monkeypatch.setattr(runtime, "instantiate", fake_instantiate)
    cfg = OmegaConf.create(
        {
            "train": {
                "_target_": "example.Dataset",
                "dataset_root": "/original",
                "numeric_dataset_root": "/original-numeric",
                "processor": {"action_output_dim": 23},
                "pretrained_norm_stats": "/stats.json",
            },
            "train_mixture": {
                "weights": [0.7, 0.3],
                "epoch_size": 100,
                "seed": 42,
                "additional_datasets": [
                    {
                        "dataset_root": "/rft",
                        "numeric_dataset_root": "/rft-numeric",
                    }
                ],
            },
            "val": {
                "_target_": "example.Dataset",
                "dataset_root": "/validation",
                "pretrained_norm_stats": "/stats.json",
            },
        }
    )

    train, val = runtime.build_datasets(cfg)

    assert train.weights == (0.7, 0.3)
    assert [dataset.root for dataset in train.datasets] == ["/original", "/rft"]
    assert train.epoch_size == 100
    assert val.root == "/validation"
    assert calls[1][0]["processor"] == {"action_output_dim": 23}
    assert calls[1][0]["pretrained_norm_stats"] == "/stats.json"


def test_run_training_passes_b1k_checkpoint_contract_to_trainer(tmp_path, monkeypatch):
    cfg = OmegaConf.create(
        {
            "output_dir": str(tmp_path / "run"),
            "mixed_precision": "bf16",
            "seed": 42,
            "model": {"_target_": "example.Model"},
            "data": {"train": {"_target_": checkpoint_module.B1K_DATASET_TARGET}},
        }
    )
    model = object()
    train_dataset = object()
    val_dataset = object()
    expected_contract = object()
    captured = {}

    monkeypatch.setattr(runtime, "instantiate", lambda *_args, **_kwargs: model)
    monkeypatch.setattr(runtime, "attach_random_module_initialization_fingerprint", lambda _model: "test")
    monkeypatch.setattr(runtime, "build_datasets", lambda _cfg: (train_dataset, val_dataset))
    monkeypatch.setattr(
        checkpoint_module,
        "build_training_checkpoint_contract",
        lambda actual_cfg, actual_dataset: (
            expected_contract
            if actual_cfg is cfg and actual_dataset is train_dataset
            else None
        ),
    )

    class FakeTrainer:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def train(self):
            captured["trained"] = True

    monkeypatch.setattr(runtime, "Wan22Trainer", FakeTrainer)
    runtime.run_training(cfg)

    assert captured["checkpoint_contract"] is expected_contract
    assert captured["model"] is model
    assert captured["train_dataset"] is train_dataset
    assert captured["val_dataset"] is val_dataset
    assert captured["trained"] is True


def test_runtime_defers_training_only_dependencies():
    assert runtime.Wan22Trainer is None


def test_run_training_seeds_before_model_instantiation(tmp_path, monkeypatch):
    cfg = OmegaConf.create(
        {
            "output_dir": str(tmp_path / "run"),
            "mixed_precision": "bf16",
            "seed": 123,
            "model": {"_target_": "example.Model"},
            "data": {"train": {"_target_": checkpoint_module.B1K_DATASET_TARGET}},
        }
    )
    events = []

    monkeypatch.setattr(
        runtime,
        "set_global_seed",
        lambda seed, *, rank_offset: events.append(("seed", seed, rank_offset)),
    )
    monkeypatch.setattr(
        runtime,
        "instantiate",
        lambda *_args, **_kwargs: events.append(("instantiate", None)) or object(),
    )
    monkeypatch.setattr(runtime, "attach_random_module_initialization_fingerprint", lambda _model: "test")
    monkeypatch.setattr(runtime, "build_datasets", lambda _cfg: (object(), object()))
    monkeypatch.setattr(checkpoint_module, "build_training_checkpoint_contract", lambda *_args: object())

    class FakeTrainer:
        def __init__(self, **_kwargs):
            pass

        def train(self):
            pass

    monkeypatch.setattr(runtime, "Wan22Trainer", FakeTrainer)

    runtime.run_training(cfg)

    assert events[:2] == [("seed", 123, False), ("instantiate", None)]


def test_run_training_same_seed_reproduces_random_model_initialization_across_ranks(
    tmp_path, monkeypatch
):
    initialized_weights = []
    ranks = iter((0, 17))

    def instantiate_model(*_args, **_kwargs):
        model = torch.nn.Linear(4, 3)
        initialized_weights.append(model.weight.detach().clone())
        return model

    monkeypatch.setattr(runtime, "instantiate", instantiate_model)
    monkeypatch.setattr(runtime, "attach_random_module_initialization_fingerprint", lambda _model: "test")
    monkeypatch.setattr(runtime, "build_datasets", lambda _cfg: (object(), object()))
    monkeypatch.setattr(checkpoint_module, "build_training_checkpoint_contract", lambda *_args: object())

    class FakeTrainer:
        def __init__(self, **_kwargs):
            pass

        def train(self):
            pass

    monkeypatch.setattr(runtime, "Wan22Trainer", FakeTrainer)

    for run_name in ("run-a", "run-b"):
        rank = next(ranks)
        monkeypatch.setattr(pytorch_utils, "_resolve_global_rank", lambda rank=rank: rank)
        cfg = OmegaConf.create(
            {
                "output_dir": str(tmp_path / run_name),
                "mixed_precision": "bf16",
                "seed": 987,
                "model": {"_target_": "example.Model"},
                "data": {"train": {"_target_": checkpoint_module.B1K_DATASET_TARGET}},
            }
        )
        runtime.run_training(cfg)

    assert torch.equal(initialized_weights[0], initialized_weights[1])


def test_set_global_seed_keeps_rank_offset_enabled_by_default(monkeypatch):
    samples = []
    for rank in (0, 17):
        monkeypatch.setattr(pytorch_utils, "_resolve_global_rank", lambda rank=rank: rank)
        pytorch_utils.set_global_seed(987)
        samples.append(torch.rand(4))

    assert not torch.equal(samples[0], samples[1])
