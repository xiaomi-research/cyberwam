# Copyright (c) 2026 Xiaomi Corporation

from types import SimpleNamespace

import pytest

import fastwam.qualitative_validation as qualitative
from fastwam.trainer import Wan22Trainer, qualitative_validation_due


METRIC_NAMES = (
    "rollout_future_psnr_vs_gt",
    "rollout_future_ssim_vs_gt",
    "rollout_future_psnr_vs_vae",
    "rollout_future_ssim_vs_vae",
    "vae_psnr_vs_gt",
    "vae_ssim_vs_gt",
    "action_l1_denormalized",
    "action_mse_denormalized",
)


def _report(*, step=8501, world_size=128, windows=128, tasks=100, duration=100.0):
    return {
        "schema": "b1k_qualitative_validation.v1",
        "status": "passed",
        "step": step,
        "world_size": world_size,
        "num_windows": windows,
        "num_tasks": tasks,
        "metrics": {
            name: {"sum": float(windows), "count": windows, "mean": 1.0}
            for name in METRIC_NAMES
        },
        "timings": {
            "total_s": {"max_seconds": duration},
            "outer_s": {"max_seconds": duration},
        },
        "windows": [{} for _ in range(windows)],
        "videos": [{} for _ in range(8)],
    }


def _trainer(*, step=8501):
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.global_step = step
    trainer.accelerator = SimpleNamespace(
        num_processes=128, is_main_process=True
    )
    trainer.qualitative_validation_enabled = True
    trainer.qualitative_validation_steps = ()
    trainer.qualitative_validation_every_steps = 8501
    trainer.qualitative_validation_num_windows = 128
    trainer.qualitative_validation_expected_tasks = 100
    trainer.qualitative_validation_videos = 8
    trainer.qualitative_validation_inference_steps = 10
    trainer.qualitative_validation_seed = 42
    trainer.qualitative_validation_fps = 8
    trainer.qualitative_validation_max_duration_seconds = 900.0
    trainer.qualitative_validation_report_dir = "/tmp/qualitative-validation"
    trainer.fixed_validation_anchors_per_task = 10
    trainer.fixed_validation_stride = 6
    trainer._log_qualitative_validation_metrics = lambda report: None
    return trainer


def test_qualitative_schedule_excludes_step_zero_and_runs_periodically():
    assert not qualitative_validation_due(0, explicit_steps=[0], every_steps=8501)
    assert qualitative_validation_due(8501, explicit_steps=[], every_steps=8501)
    assert qualitative_validation_due(17, explicit_steps=[17], every_steps=0)
    assert not qualitative_validation_due(8500, explicit_steps=[], every_steps=8501)
    with pytest.raises(ValueError, match="non-negative"):
        qualitative_validation_due(-1, explicit_steps=[], every_steps=1)
    with pytest.raises(ValueError, match="non-negative"):
        qualitative_validation_due(1, explicit_steps=[], every_steps=-1)


def test_trainer_runs_all_rank_qualitative_validation_with_locked_config(monkeypatch):
    trainer = _trainer()
    calls = []

    def run(received_trainer, **kwargs):
        calls.append((received_trainer, kwargs))
        return _report()

    monkeypatch.setattr(qualitative, "run_trainer_qualitative_validation", run)

    assert trainer._run_qualitative_validation_if_due() == _report()
    assert calls[0][0] is trainer
    config = calls[0][1]["config"]
    assert config.num_windows == 128
    assert config.expected_tasks == 100
    assert config.videos_to_save == 8
    assert config.num_inference_steps == 10
    assert config.base_seed == 42
    assert config.fps == 8
    assert calls[0][1]["fixed_anchors_per_task"] == 10
    assert calls[0][1]["fixed_anchor_stride"] == 6


@pytest.mark.parametrize(
    ("report", "message"),
    [
        (_report(world_size=64), "world_size"),
        (_report(windows=127), "num_windows"),
        (_report() | {"videos": []}, "video inventory"),
        (_report(duration=900.01), "duration gate"),
        (
            _report()
            | {
                "metrics": _report()["metrics"]
                | {
                    "action_l1_denormalized": {
                        "sum": float("nan"),
                        "count": 128,
                        "mean": float("nan"),
                    }
                }
            },
            "metric contract",
        ),
    ],
)
def test_trainer_fails_closed_on_invalid_qualitative_report(
    monkeypatch, report, message
):
    trainer = _trainer()
    monkeypatch.setattr(
        qualitative, "run_trainer_qualitative_validation", lambda *args, **kwargs: report
    )

    with pytest.raises(RuntimeError, match=message):
        trainer._run_qualitative_validation_if_due()


def test_trainer_does_not_run_qualitative_validation_at_step_zero(monkeypatch):
    trainer = _trainer(step=0)
    monkeypatch.setattr(
        qualitative,
        "run_trainer_qualitative_validation",
        lambda *args, **kwargs: pytest.fail("step-zero inference ran"),
    )

    assert trainer._run_qualitative_validation_if_due() is None
