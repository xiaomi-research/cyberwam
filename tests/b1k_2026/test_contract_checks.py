# Copyright (c) 2026 Xiaomi Corporation

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest

from scripts.b1k_2026.contract_checks import (
    apply_contract_scheduler_overrides,
    assert_model_target_matches_contract,
    assert_model_schedulers_match_contract,
    assert_model_temporal_contract_matches_contract,
    inference_metadata,
)


UNCOND_TARGET = "fastwam.runtime.create_fastwam"
JOINT_TARGET = "fastwam.runtime.create_fastwam_joint"


@pytest.mark.parametrize("target", [UNCOND_TARGET, JOINT_TARGET])
def test_model_target_accepts_matching_uncond_and_joint_contracts(target: str) -> None:
    model = SimpleNamespace(_fastwam_b1k_model_target=target)
    contract = SimpleNamespace(compatibility={"model": {"target": target}})

    assert_model_target_matches_contract(model, contract)


@pytest.mark.parametrize(
    ("configured", "checkpoint"),
    [(UNCOND_TARGET, JOINT_TARGET), (JOINT_TARGET, UNCOND_TARGET)],
)
def test_model_target_rejects_cross_variant_contracts(
    configured: str, checkpoint: str
) -> None:
    model = SimpleNamespace(_fastwam_b1k_model_target=configured)
    contract = SimpleNamespace(compatibility={"model": {"target": checkpoint}})

    with pytest.raises(ValueError, match="does not match checkpoint contract"):
        assert_model_target_matches_contract(model, contract)


def test_model_target_requires_factory_metadata_and_contract_target() -> None:
    with pytest.raises(ValueError, match="configured None"):
        assert_model_target_matches_contract(
            SimpleNamespace(),
            SimpleNamespace(compatibility={"model": {"target": UNCOND_TARGET}}),
        )
    with pytest.raises(ValueError, match="missing compatibility.model.target"):
        assert_model_target_matches_contract(
            SimpleNamespace(_fastwam_b1k_model_target=UNCOND_TARGET),
            SimpleNamespace(compatibility={"model": {}}),
        )


def _scheduler(shift: float, steps: int = 1000) -> SimpleNamespace:
    return SimpleNamespace(shift=shift, num_train_timesteps=steps)


def _scheduler_model(
    *, action_shift: float = 5.0, shared_train_timestep: bool = True
) -> SimpleNamespace:
    return SimpleNamespace(
        train_video_scheduler=_scheduler(5.0),
        infer_video_scheduler=_scheduler(5.0),
        train_action_scheduler=_scheduler(action_shift),
        infer_action_scheduler=_scheduler(action_shift),
        train_dino_scheduler=_scheduler(5.0),
        infer_dino_scheduler=_scheduler(5.0),
        train_pointmap_scheduler=_scheduler(5.0),
        infer_pointmap_scheduler=_scheduler(5.0),
        shared_train_timestep=shared_train_timestep,
    )


def _scheduler_contract(
    *, action_shift: float = 5.0, shared_train_timestep: bool = True
) -> SimpleNamespace:
    stream = lambda shift: {
        "train_shift": shift,
        "infer_shift": shift,
        "num_train_timesteps": 1000,
    }
    return SimpleNamespace(
        compatibility={
            "scheduler": {
                "video": {**stream(5.0), "shared_train_timestep": shared_train_timestep},
                "action": stream(action_shift),
                "dino": stream(5.0),
                "pointmap": stream(5.0),
            }
        }
    )


def test_model_schedulers_accept_matching_checkpoint_contract() -> None:
    assert_model_schedulers_match_contract(_scheduler_model(), _scheduler_contract())


def test_model_schedulers_reject_action_shift_mismatch() -> None:
    with pytest.raises(ValueError, match=r"action\.infer_shift.*expected 5\.0, got 1\.0"):
        assert_model_schedulers_match_contract(
            _scheduler_model(action_shift=1.0), _scheduler_contract()
        )


def test_model_schedulers_reject_shared_timestep_mismatch() -> None:
    with pytest.raises(ValueError, match="shared_train_timestep"):
        assert_model_schedulers_match_contract(
            _scheduler_model(shared_train_timestep=False), _scheduler_contract()
        )


def test_contract_scheduler_overrides_add_or_replace_hydra_fields(monkeypatch) -> None:
    monkeypatch.setenv(
        "FASTWAM_B1K_CONFIG_OVERRIDES",
        json.dumps(["model.proprio_token_dropout_prob=0.3"]),
    )

    apply_contract_scheduler_overrides(_scheduler_contract())

    overrides = json.loads(os.environ["FASTWAM_B1K_CONFIG_OVERRIDES"])
    assert overrides[0] == "model.proprio_token_dropout_prob=0.3"
    assert "++model.video_scheduler.shared_train_timestep=true" in overrides
    assert "++model.video_scheduler.train_shift=5.0" in overrides
    assert "++model.action_scheduler.infer_shift=5.0" in overrides
    assert all(
        not item.startswith("+model.")
        for item in overrides
    )


def _temporal_contract(*, action_horizon: int = 64, video_frames: int = 17) -> SimpleNamespace:
    return SimpleNamespace(
        compatibility={
            "data": {
                "action": {"horizon": action_horizon},
                "video": {
                    "dataset_image_indices": list(range(video_frames)),
                    "raw_frame_indices": list(range(video_frames)),
                },
            }
        }
    )


def test_model_temporal_contract_accepts_matching_runtime() -> None:
    model = SimpleNamespace(
        _fastwam_b1k_action_horizon=64,
        _fastwam_b1k_num_video_frames=17,
    )
    assert_model_temporal_contract_matches_contract(model, _temporal_contract())


def test_model_temporal_contract_rejects_runtime_override(monkeypatch) -> None:
    monkeypatch.setenv("FASTWAM_B1K_ACTION_HORIZON", "32")
    model = SimpleNamespace(
        _fastwam_b1k_action_horizon=64,
        _fastwam_b1k_num_video_frames=17,
    )
    with pytest.raises(ValueError, match="action_horizon.*expected 64, got 32"):
        assert_model_temporal_contract_matches_contract(model, _temporal_contract())


def test_inference_metadata_exposes_model_contract_and_fixed_step_default(monkeypatch):
    monkeypatch.delenv("FASTWAM_B1K_ACTION_HORIZON", raising=False)
    monkeypatch.delenv("FASTWAM_B1K_NUM_VIDEO_FRAMES", raising=False)
    monkeypatch.delenv("FASTWAM_B1K_INFERENCE_STEPS", raising=False)

    assert inference_metadata(
        SimpleNamespace(
            _fastwam_b1k_action_horizon=32,
            _fastwam_b1k_num_video_frames=9,
        )
    ) == {
        "action_horizon": 32,
        "num_video_frames": 9,
        "inference_steps": 10,
    }


def test_inference_metadata_honors_explicit_runtime_overrides(monkeypatch):
    monkeypatch.setenv("FASTWAM_B1K_ACTION_HORIZON", "16")
    monkeypatch.setenv("FASTWAM_B1K_NUM_VIDEO_FRAMES", "5")
    monkeypatch.setenv("FASTWAM_B1K_INFERENCE_STEPS", "7")

    assert inference_metadata(SimpleNamespace()) == {
        "action_horizon": 16,
        "num_video_frames": 5,
        "inference_steps": 7,
    }
