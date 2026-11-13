# Copyright (c) 2026 Xiaomi Corporation

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from fastwam.datasets.b1k.contract import ACTION_CONSTANT_DIM, ACTION_GRIPPER_DIMS
from fastwam.datasets.b1k.overfit import (
    FixedWindowSubset,
    build_overfit_report,
    episode_window_ids,
    load_overfit_report,
    parse_sample_ids,
    save_overfit_report,
    validate_action_contract,
)


class _IndexDataset(Dataset):
    checkpoint_data_contract = {"schema": "toy"}

    def __init__(self) -> None:
        self.processor = object()
        self.episode_data_index = {
            "from": torch.tensor([0, 4]),
            "to": torch.tensor([4, 10]),
        }
        self.lerobot_dataset = SimpleNamespace(
            processor=object(),
            episode_data_index={
                "from": torch.tensor([0]),
                "to": torch.tensor([10]),
            }
        )

    def __len__(self) -> int:
        return 10

    def __getitem__(self, index: int) -> int:
        return index


def test_fixed_window_subset_is_stable_and_delegates_contract() -> None:
    base = _IndexDataset()
    subset = FixedWindowSubset(base, [7, 2], repeat=3)

    assert len(subset) == 6
    assert [subset[index] for index in range(len(subset))] == [7, 2, 7, 2, 7, 2]
    assert subset[-1] == 2
    assert subset.checkpoint_data_contract == {"schema": "toy"}
    assert subset.processor is base.processor
    assert subset.episode_data_index is base.episode_data_index
    assert subset.lerobot_dataset is base.lerobot_dataset


def test_episode_window_ids_and_cli_parser() -> None:
    base = _IndexDataset()
    assert episode_window_ids(
        base, episode_position=1, max_windows=2, offset=1, stride=2
    ) == (5, 7)
    assert parse_sample_ids("5, 7,9") == (5, 7, 9)
    with pytest.raises(IndexError, match="episode_position"):
        episode_window_ids(base, episode_position=2, max_windows=1)


def test_legacy_lerobot_fields_remain_supported() -> None:
    legacy = _IndexDataset()
    del legacy.processor
    del legacy.episode_data_index
    legacy.lerobot_dataset.episode_data_index = {
        "from": torch.tensor([0, 5]),
        "to": torch.tensor([5, 10]),
    }
    subset = FixedWindowSubset(legacy, [0])

    assert subset.processor is legacy.lerobot_dataset.processor
    assert subset.episode_data_index is legacy.lerobot_dataset.episode_data_index
    assert episode_window_ids(
        legacy, episode_position=1, max_windows=2, offset=0, stride=2
    ) == (5, 7)


class _ToyOverfitModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.video_head = nn.Linear(3, 2)
        self.action_head = nn.Linear(3, 23)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.video_head(features), self.action_head(features)


def _metrics(
    model: _ToyOverfitModel,
    features: torch.Tensor,
    video_target: torch.Tensor,
    action_target: torch.Tensor,
) -> dict[str, float]:
    with torch.no_grad():
        video, action = model(features)
        return {
            "loss_video": torch.nn.functional.mse_loss(video, video_target).item(),
            "loss_action": torch.nn.functional.mse_loss(action, action_target).item(),
            "denormalized_action_l1": torch.nn.functional.l1_loss(
                action, action_target
            ).item(),
        }


def test_toy_fixed_window_overfit_converges_and_writes_honest_report(
    tmp_path: Path,
) -> None:
    torch.manual_seed(7)
    features = torch.tensor([[0.25, -0.5, 1.0]])
    video_target = torch.tensor([[0.6, -0.2]])
    action_target = torch.linspace(-0.8, 0.8, 23).unsqueeze(0)
    action_target[:, ACTION_CONSTANT_DIM] = 0
    for dim, value in zip(ACTION_GRIPPER_DIMS, (-1.0, 1.0), strict=True):
        action_target[:, dim] = value

    model = _ToyOverfitModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05, weight_decay=0.0)
    initial = _metrics(model, features, video_target, action_target)
    for _ in range(150):
        video, action = model(features)
        loss_video = torch.nn.functional.mse_loss(video, video_target)
        loss_action = torch.nn.functional.mse_loss(action, action_target)
        optimizer.zero_grad(set_to_none=True)
        (loss_video + loss_action).backward()
        optimizer.step()
    final = _metrics(model, features, video_target, action_target)

    executable_action = action_target.clone()
    assert validate_action_contract(executable_action)
    report = build_overfit_report(
        initial=initial,
        final=final,
        sample_ids=[0],
        max_steps=150,
        action_contract_valid=True,
        metadata={"metrics_source": "toy_unit_test"},
    )
    output = tmp_path / "overfit-report.json"
    save_overfit_report(output, report)

    loaded = load_overfit_report(output)
    assert loaded["status"] == "passed"
    assert loaded["checks"] == {
        "action_loss_decreased": True,
        "denormalized_action_l1_decreased": True,
        "finite": True,
        "gripper_and_constant_action_contract_valid": True,
        "video_loss_decreased": True,
    }
    assert loaded["metadata"]["metrics_source"] == "toy_unit_test"


def test_report_rejects_non_finite_metrics() -> None:
    finite = {
        "loss_video": 1.0,
        "loss_action": 1.0,
        "denormalized_action_l1": 1.0,
    }
    with pytest.raises(ValueError, match="finite"):
        build_overfit_report(
            initial=finite,
            final={**finite, "loss_action": float("nan")},
            sample_ids=[0],
            max_steps=1,
            action_contract_valid=True,
        )


def test_report_records_and_requires_real_gradient_gate_evidence() -> None:
    metrics = {
        "loss_video": 1.0,
        "loss_action": 1.0,
        "denormalized_action_l1": 1.0,
    }
    gradient_gate = {
        "schema": "fastwam_gradient_gate.v1",
        "status": "passed",
        "checks": {
            "loss_finite_on_all_ranks": True,
            "all_observed_gradients_finite": True,
            "at_least_one_finite_nonzero_trainable_gradient": True,
            "finite_nonzero_trainable_gradient_on_every_rank": True,
        },
        "rank_evidence": [],
    }

    report = build_overfit_report(
        initial=metrics,
        final={key: value / 2 for key, value in metrics.items()},
        sample_ids=[0],
        max_steps=1,
        action_contract_valid=True,
        gradient_gate=gradient_gate,
    )

    assert report["status"] == "passed"
    assert report["checks"]["gradient_gate_passed"] is True
    assert report["gradient_gate"] == gradient_gate

    failed_gate = {**gradient_gate, "status": "failed"}
    failed_report = build_overfit_report(
        initial=metrics,
        final={key: value / 2 for key, value in metrics.items()},
        sample_ids=[0],
        max_steps=1,
        action_contract_valid=True,
        gradient_gate=failed_gate,
    )
    assert failed_report["status"] == "failed"
    assert failed_report["checks"]["gradient_gate_passed"] is False
