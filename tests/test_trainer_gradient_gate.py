# Copyright (c) 2026 Xiaomi Corporation

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from fastwam.trainer import (
    Wan22Trainer,
    _GradientGateBackwardObserver,
    _local_gradient_gate_stats,
)


class _Accelerator:
    is_main_process = True
    num_processes = 1

    def gather(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor


def _trainer(model: nn.Module, *, loss_finite: bool = True) -> Wan22Trainer:
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.model = model
    trainer.accelerator = _Accelerator()
    trainer.global_step = 0
    trainer._gradient_gate_report = None
    trainer._gradient_gate_loss_finite = torch.tensor(loss_finite)
    trainer._gradient_gate_loss_observations = 2
    return trainer


def test_local_gradient_gate_stats_counts_finite_nonzero_trainable_gradients() -> None:
    model = nn.Sequential(nn.Linear(3, 2), nn.Linear(2, 1))
    model[1].weight.requires_grad_(False)
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.grad = torch.zeros_like(parameter)
    model[0].weight.grad[0, 0] = 0.25

    stats = _local_gradient_gate_stats(
        model.parameters(), torch.tensor(True)
    ).squeeze(0)

    assert stats.tolist() == [1, 3, 3, 3, 1]


def test_local_gradient_gate_reads_deepspeed_zero12_fragment_without_full_gather() -> None:
    parameter = nn.Parameter(torch.ones(2))
    parameter.grad = None
    parameter._index_in_param_group = 7
    parameter._hp_mapping = SimpleNamespace(
        get_lp_grad_fragment=lambda index: torch.tensor([0.0, 0.75])
        if index == 7
        else None
    )

    stats = _local_gradient_gate_stats([parameter], torch.tensor(True)).squeeze(0)

    assert stats.tolist() == [1, 1, 1, 1, 1]


def test_local_gradient_gate_reads_deepspeed_grad_accum() -> None:
    parameter = nn.Parameter(torch.ones(2))
    parameter.grad = None
    parameter.grad_accum = torch.tensor([0.0, 0.5])

    stats = _local_gradient_gate_stats([parameter], torch.tensor(True)).squeeze(0)

    assert stats.tolist() == [1, 1, 1, 1, 1]


def test_backward_observer_retains_real_evidence_after_grad_fields_are_cleared() -> None:
    model = nn.Linear(2, 1)
    observer = _GradientGateBackwardObserver(model.parameters())

    model(torch.ones(1, 2)).sum().backward()
    for parameter in model.parameters():
        parameter.grad = None

    stats = observer.stats(torch.tensor(True)).squeeze(0)
    observer.close()

    assert stats.tolist() == [1, 2, 2, 2, 2]


def test_backward_observer_accumulates_evidence_across_micro_steps() -> None:
    parameter = nn.Parameter(torch.ones(2))
    observer = _GradientGateBackwardObserver([parameter])

    (parameter * 2).sum().backward()
    parameter.grad = None
    (parameter * 0).sum().backward()
    parameter.grad = None

    stats = observer.stats(torch.tensor(True)).squeeze(0)
    observer.close()

    assert stats.tolist() == [1, 1, 1, 1, 1]


def test_gradient_gate_passes_once_and_records_collectable_evidence() -> None:
    model = nn.Linear(2, 1)
    model.weight.grad = torch.tensor([[0.5, 0.0]])
    model.bias.grad = torch.zeros_like(model.bias)
    trainer = _trainer(model)

    trainer._validate_gradient_gate()
    first_report = trainer._gradient_gate_report
    trainer._validate_gradient_gate()

    assert trainer._gradient_gate_report is first_report
    assert first_report["status"] == "passed"
    assert first_report["gradient_accumulation_micro_steps_observed"] == 2
    assert first_report["checks"] == {
        "loss_finite_on_all_ranks": True,
        "all_observed_gradients_finite": True,
        "at_least_one_finite_nonzero_trainable_gradient": True,
        "finite_nonzero_trainable_gradient_on_every_rank": True,
    }
    assert first_report["rank_evidence"][0][
        "finite_nonzero_gradient_parameter_tensors"
    ] == 1


@pytest.mark.parametrize("failure", ["loss", "zero_gradient", "nan_gradient"])
def test_gradient_gate_fails_fast_on_invalid_real_training_signal(failure: str) -> None:
    model = nn.Linear(2, 1)
    model.weight.grad = torch.ones_like(model.weight)
    model.bias.grad = torch.zeros_like(model.bias)
    trainer = _trainer(model, loss_finite=failure != "loss")
    if failure == "zero_gradient":
        model.weight.grad.zero_()
    elif failure == "nan_gradient":
        model.weight.grad[0, 0] = float("nan")

    with pytest.raises(FloatingPointError, match="loss/gradient gate"):
        trainer._validate_gradient_gate()

    assert trainer._gradient_gate_report["status"] == "failed"


def test_gradient_gate_aggregates_rank_evidence_before_deciding() -> None:
    model = nn.Linear(1, 1)
    model.weight.grad = torch.ones_like(model.weight)
    model.bias.grad = torch.zeros_like(model.bias)
    trainer = _trainer(model)
    trainer.accelerator = SimpleNamespace(
        is_main_process=True,
        num_processes=2,
        gather=lambda local: torch.cat(
            [local, torch.tensor([[1, 2, 2, 2, 1]], dtype=local.dtype)], dim=0
        ),
    )

    trainer._validate_gradient_gate()

    assert trainer._gradient_gate_report["status"] == "passed"
    assert len(trainer._gradient_gate_report["rank_evidence"]) == 2


def test_gradient_gate_requires_nonzero_gradient_evidence_on_every_rank() -> None:
    model = nn.Linear(1, 1)
    model.weight.grad = torch.zeros_like(model.weight)
    model.bias.grad = torch.zeros_like(model.bias)
    trainer = _trainer(model)
    trainer.accelerator = SimpleNamespace(
        is_main_process=True,
        num_processes=2,
        gather=lambda local: torch.cat(
            [local, torch.tensor([[1, 2, 2, 2, 1]], dtype=local.dtype)], dim=0
        ),
    )

    with pytest.raises(FloatingPointError, match="loss/gradient gate"):
        trainer._validate_gradient_gate()

    assert trainer._gradient_gate_report["checks"][
        "finite_nonzero_trainable_gradient_on_every_rank"
    ] is False
