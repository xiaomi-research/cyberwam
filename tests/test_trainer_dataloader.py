# Copyright (c) 2026 Xiaomi Corporation

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

try:
    import accelerate  # noqa: F401
except ImportError:
    accelerate = ModuleType("accelerate")
    accelerate.Accelerator = object
    sys.modules["accelerate"] = accelerate

from fastwam.trainer import (
    Wan22Trainer,
    _resolve_dataloader_config,
    _resolve_lr_warmup_steps,
)


def test_resolve_lr_warmup_steps_records_effective_scheduler_length():
    assert _resolve_lr_warmup_steps(1000, None) == 50
    assert _resolve_lr_warmup_steps(1000, 100) == 100
    assert _resolve_lr_warmup_steps(6, 100) == 5
    assert _resolve_lr_warmup_steps(1, None) == 0


def test_resolve_lr_warmup_steps_rejects_invalid_values():
    with pytest.raises(ValueError, match="total_train_steps"):
        _resolve_lr_warmup_steps(0, None)
    with pytest.raises(ValueError, match="lr_warmup_steps"):
        _resolve_lr_warmup_steps(100, -1)


class _Dataset(torch.utils.data.Dataset):
    def __len__(self):
        return 16

    def __getitem__(self, index):
        return index


def _trainer(*, num_workers, prefetch_factor, persistent_workers):
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.seed = 42
    trainer.batch_size = 2
    trainer.num_workers = num_workers
    trainer.prefetch_factor = prefetch_factor
    trainer.persistent_workers = persistent_workers
    trainer.accelerator = SimpleNamespace(num_processes=1)
    return trainer


def test_build_loader_passes_worker_tuning_options(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    trainer = _trainer(
        num_workers=3,
        prefetch_factor=7,
        persistent_workers=True,
    )

    loader = trainer._build_loader(_Dataset())

    assert loader.num_workers == 3
    assert loader.prefetch_factor == 7
    assert loader.persistent_workers is True
    assert loader.pin_memory is True


def test_zero_workers_disables_prefetch_and_uses_main_process_loader():
    settings = _resolve_dataloader_config(
        num_workers=0,
        prefetch_factor=8,
        persistent_workers=False,
    )
    assert settings == (0, None, False)

    trainer = _trainer(
        num_workers=settings[0],
        prefetch_factor=settings[1],
        persistent_workers=settings[2],
    )
    loader = trainer._build_loader(_Dataset())
    assert loader.num_workers == 0
    assert loader.prefetch_factor is None
    assert loader.persistent_workers is False


@pytest.mark.parametrize("prefetch_factor", [0, -1, 1.5, True, "invalid"])
def test_invalid_prefetch_factor_is_rejected(prefetch_factor):
    with pytest.raises((TypeError, ValueError), match="prefetch_factor"):
        _resolve_dataloader_config(
            num_workers=2,
            prefetch_factor=prefetch_factor,
            persistent_workers=False,
        )


@pytest.mark.parametrize("num_workers", [-1, 1.5, True, "invalid"])
def test_invalid_num_workers_is_rejected(num_workers):
    with pytest.raises((TypeError, ValueError), match="num_workers"):
        _resolve_dataloader_config(
            num_workers=num_workers,
            prefetch_factor=2,
            persistent_workers=False,
        )


def test_persistent_workers_requires_worker_processes():
    with pytest.raises(ValueError, match="requires `num_workers` greater than zero"):
        _resolve_dataloader_config(
            num_workers=0,
            prefetch_factor=2,
            persistent_workers=True,
        )


def test_persistent_workers_must_be_boolean():
    with pytest.raises(TypeError, match="persistent_workers"):
        _resolve_dataloader_config(
            num_workers=2,
            prefetch_factor=2,
            persistent_workers="true",
        )


@pytest.mark.parametrize(
    ("dataset_length", "batch_size", "world_size", "gradient_accumulation", "epochs", "expected"),
    [
        (34_821_338, 32, 128, 1, 10, 85_010),
        (101, 4, 2, 3, 2, 8),
    ],
)
def test_total_train_steps_match_sampler_tail_drop(
    dataset_length,
    batch_size,
    world_size,
    gradient_accumulation,
    epochs,
    expected,
):
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.max_steps = None
    trainer.train_dataset = range(dataset_length)
    trainer.batch_size = batch_size
    trainer.gradient_accumulation_steps = gradient_accumulation
    trainer.num_epochs = epochs
    trainer.accelerator = SimpleNamespace(num_processes=world_size)

    assert trainer._estimate_total_train_steps() == expected


def test_total_train_steps_reject_dataset_smaller_than_distributed_batch():
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.max_steps = None
    trainer.train_dataset = range(7)
    trainer.batch_size = 4
    trainer.gradient_accumulation_steps = 1
    trainer.num_epochs = 1
    trainer.accelerator = SimpleNamespace(num_processes=2)

    with pytest.raises(ValueError, match="complete distributed batch"):
        trainer._estimate_total_train_steps()
