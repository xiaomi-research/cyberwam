# Copyright (c) 2026 Xiaomi Corporation

from contextlib import nullcontext

import pytest
import torch

from fastwam.trainer import (
    Wan22Trainer,
    _summarize_fixed_validation_rank_timings,
    build_fixed_validation_window_indices,
    fixed_validation_due,
    fixed_validation_seed,
    fixed_validation_window_set_sha256,
)


class _Dataset:
    selected_episodes = list(range(100))
    window_start_stride = 6
    episode_data_index = {
        "from": torch.arange(0, 1000, 10),
        "to": torch.arange(10, 1010, 10),
    }


def test_fixed_windows_are_episode_local_and_deterministic():
    windows = build_fixed_validation_window_indices(_Dataset())
    assert len(windows) == 1000
    assert windows[0] == (0, 0, 0)
    assert windows[9] == (9, 0, 9)
    assert windows[10] == (10, 1, 10)
    assert windows[-1] == (999, 99, 999)
    assert windows == build_fixed_validation_window_indices(_Dataset())



def test_fixed_window_seed_is_stable_and_identity_specific():
    a = fixed_validation_seed(1, 2, 3)
    assert a == fixed_validation_seed(1, 2, 3)
    assert a != fixed_validation_seed(1, 2, 4)
    assert 0 <= a < 2**32


def test_fixed_validation_window_hash_is_order_and_identity_sensitive():
    windows = [(0, 10, 100), (1, 11, 200)]
    fingerprint = fixed_validation_window_set_sha256(windows)
    assert fingerprint == fixed_validation_window_set_sha256(list(windows))
    assert fingerprint != fixed_validation_window_set_sha256(list(reversed(windows)))
    assert fingerprint != fixed_validation_window_set_sha256(
        [(0, 10, 100), (1, 11, 201)]
    )


def test_fixed_validation_due_supports_explicit_and_periodic_steps():
    assert fixed_validation_due(0, explicit_steps=[], every_steps=125)
    assert fixed_validation_due(125, explicit_steps=[], every_steps=125)
    assert not fixed_validation_due(124, explicit_steps=[], every_steps=125)
    assert fixed_validation_due(17, explicit_steps=[17], every_steps=125)
    assert not fixed_validation_due(0, explicit_steps=[], every_steps=0)


def test_fixed_validation_due_rejects_invalid_values():
    import pytest

    with pytest.raises(ValueError, match="step must be non-negative"):
        fixed_validation_due(-1, explicit_steps=[], every_steps=0)
    with pytest.raises(ValueError, match="every_steps must be non-negative"):
        fixed_validation_due(0, explicit_steps=[], every_steps=-1)


class _EvalDataset:
    selected_episodes = [0]
    window_start_stride = 6
    episode_data_index = {"from": torch.tensor([0]), "to": torch.tensor([1])}

    def __getitem__(self, _index):
        return {
            "video": torch.zeros(3, 2, 2, 2),
            "pointmap": torch.zeros(3, 2, 2, 2),
            "prompt": "test",
            "action": torch.zeros(1, 1),
            "image_is_pad": torch.zeros(2, dtype=torch.bool),
        }


class _MixedModeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dit = torch.nn.Linear(1, 1)
        self.proprio_encoder = torch.nn.Linear(1, 1)
        self.frozen_encoder = torch.nn.Linear(1, 1)
        self.seeds = []

    def training_loss(self, _sample, *, rng_seed=None):
        assert not self.training
        assert all(not module.training for module in self.modules())
        assert _sample["pointmap"].shape == (1, 3, 2, 2, 2)
        assert _sample["image_is_pad"].shape == (1, 2)
        self.seeds.append(rng_seed)
        loss = self.dit.weight.sum() * 0 + 1.0
        return loss, {
            "loss_action": 0.3,
            "loss_video": 0.3,
            "loss_dino": 0.2,
            "loss_pointmap": 0.2,
            "diagnostic": 99.0,
        }


class _DeepSpeedLikeEngine(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def training_loss(self, *args, **kwargs):
        return self.module.training_loss(*args, **kwargs)


class _EvalAccelerator:
    process_index = 0
    num_processes = 1
    device = torch.device("cpu")

    @staticmethod
    def unwrap_model(model):
        return getattr(model, "module", model)

    @staticmethod
    def autocast():
        return nullcontext()

    @staticmethod
    def reduce(tensor, reduction):
        assert reduction == "sum"
        return tensor

    @staticmethod
    def gather(tensor):
        return tensor


def _fixed_eval_trainer(*, training_seed, validation_seed=42):
    model = _MixedModeModel()
    engine = _DeepSpeedLikeEngine(model)
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.model = engine
    trainer.accelerator = _EvalAccelerator()
    trainer.val_dataset = _EvalDataset()
    trainer.seed = training_seed
    trainer.fixed_validation_seed = validation_seed
    trainer.initialization_fingerprint_sha256 = "a" * 64
    trainer.global_step = 0
    trainer._apply_dit_only_train_mode(model)
    engine.training = True
    return trainer, engine, model


def _execution_flags(*roots):
    modules = {}
    parameters = {}
    for root in roots:
        for module in root.modules():
            modules[id(module)] = module.training
            for parameter in module.parameters(recurse=False):
                parameters[id(parameter)] = parameter.requires_grad
    return modules, parameters


def test_step0_fixed_validation_restores_mixed_mode_and_requires_grad_through_engine():
    trainer, engine, model = _fixed_eval_trainer(training_seed=42)
    before = _execution_flags(engine, model)

    trainer.evaluate_fixed_validation_loss(tasks=1, anchors_per_task=1)

    assert _execution_flags(engine, model) == before
    assert model.dit.training
    assert model.proprio_encoder.training
    assert not model.frozen_encoder.training
    assert model.dit.weight.requires_grad
    assert model.proprio_encoder.weight.requires_grad
    assert not model.frozen_encoder.weight.requires_grad
    # This is the forward immediately following step-0 validation.
    assert model.dit(torch.ones(1, 1)).isfinite().all()
    assert not model.frozen_encoder.training


def test_fixed_validation_noise_is_independent_of_training_seed():
    first, _, first_model = _fixed_eval_trainer(training_seed=42)
    second, _, second_model = _fixed_eval_trainer(training_seed=31415)

    first.evaluate_fixed_validation_loss(tasks=1, anchors_per_task=1)
    second.evaluate_fixed_validation_loss(tasks=1, anchors_per_task=1)

    assert first_model.seeds == second_model.seeds
    assert first_model.seeds == [fixed_validation_seed(0, 0, 0, base_seed=42)]


def test_fixed_validation_report_contains_complete_rank_timing():
    trainer, _, _ = _fixed_eval_trainer(training_seed=42)
    trainer.fixed_validation_max_duration_seconds = 900

    report = trainer.evaluate_fixed_validation_loss(tasks=1, anchors_per_task=1)
    assert report["loss_dino"] == pytest.approx(0.2)
    assert report["loss_pointmap"] == pytest.approx(0.2)
    assert "diagnostic" not in report

    timings = report["timings"]
    assert timings["rank_count"] == 1
    assert timings["window_count"] == 1
    assert timings["ranks"][0]["rank"] == 0
    assert timings["ranks"][0]["windows"] == 1
    assert timings["total_s"]["max_seconds"] >= 0


def test_fixed_validation_128_rank_timing_inventory_matches_partition():
    rows = torch.zeros(128, 7, dtype=torch.float64)
    rows[:, 0] = torch.arange(128)
    rows[:, 1] = torch.tensor([8] * 104 + [7] * 24)
    rows[:, 2] = torch.arange(128, dtype=torch.float64) / 10
    rows[:, 3] = torch.arange(128, dtype=torch.float64) / 5
    rows[:, 4] = rows[:, 2] + rows[:, 3]
    rows[:, 5] = 1000.0
    rows[:, 6] = 1000.0 + rows[:, 4]

    report = _summarize_fixed_validation_rank_timings(rows)

    assert report["rank_count"] == 128
    assert report["window_count"] == 1000
    assert [record["rank"] for record in report["ranks"]] == list(range(128))
    assert [record["windows"] for record in report["ranks"]].count(8) == 104
    assert [record["windows"] for record in report["ranks"]].count(7) == 24


def test_fixed_validation_duration_gate_fails_closed():
    import pytest

    trainer, _, _ = _fixed_eval_trainer(training_seed=42)
    trainer.fixed_validation_max_duration_seconds = 0.0

    with pytest.raises(RuntimeError, match="duration gate"):
        trainer.evaluate_fixed_validation_loss(tasks=1, anchors_per_task=1)
