# Copyright (c) 2026 Xiaomi Corporation

from types import SimpleNamespace

import pytest

from fastwam.trainer import Wan22Trainer


class FakeTensorBoardLogger:
    def __init__(self):
        self.records = []
        self.flush_count = 0
        self.close_count = 0

    def log_scalars(self, metrics, *, step):
        self.records.append((dict(metrics), step))

    def flush(self):
        self.flush_count += 1

    def close(self):
        self.close_count += 1


def _trainer_with_fake_logger(*, is_main_process=True):
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.global_step = 17
    trainer.accelerator = SimpleNamespace(is_main_process=is_main_process)
    trainer.tensorboard_logger = FakeTensorBoardLogger()
    trainer.wandb_run = None
    return trainer


def test_tensorboard_scalars_use_explicit_or_current_global_step():
    trainer = _trainer_with_fake_logger()

    trainer._log_tensorboard_scalars({"train/loss_total": 1.25})
    trainer._log_tensorboard_scalars(
        {"validation/loss_total": 0.5}, step=0, flush=True
    )

    assert trainer.tensorboard_logger.records == [
        ({"train/loss_total": 1.25}, 17),
        ({"validation/loss_total": 0.5}, 0),
    ]
    assert trainer.tensorboard_logger.flush_count == 1


def test_fixed_validation_logs_canonical_scalars_and_flushes():
    trainer = _trainer_with_fake_logger()
    wandb_payloads = []
    trainer._wandb_log = wandb_payloads.append

    trainer._log_fixed_validation_metrics(
        {
            "step": 0,
            "loss_total": 3.0,
            "loss_action": 1.0,
            "loss_video": 2.0,
            "loss_dino": 0.25,
            "loss_pointmap": 0.5,
        },
        duration_seconds=12.5,
    )

    expected = {
        "validation/loss_total": 3.0,
        "validation/loss_action": 1.0,
        "validation/loss_video": 2.0,
        "validation/loss_dino": 0.25,
        "validation/loss_pointmap": 0.5,
        "timing/fixed_validation_s": 12.5,
    }
    assert trainer.tensorboard_logger.records == [(expected, 0)]
    assert trainer.tensorboard_logger.flush_count == 1
    assert wandb_payloads == [expected]


def _qualitative_report(*, step=8501, duration=123.0):
    names = {
        "rollout_future_psnr_vs_gt": 20.0,
        "rollout_future_ssim_vs_gt": 0.7,
        "rollout_future_psnr_vs_vae": 21.0,
        "rollout_future_ssim_vs_vae": 0.8,
        "vae_psnr_vs_gt": 30.0,
        "vae_ssim_vs_gt": 0.9,
        "action_l1_denormalized": 0.1,
        "action_mse_denormalized": 0.2,
    }
    return {
        "step": step,
        "metrics": {
            name: {"mean": value, "sum": value * 128, "count": 128}
            for name, value in names.items()
        },
        "timings": {
            "total_s": {"max_seconds": duration},
            "outer_s": {"max_seconds": duration},
        },
    }


def test_qualitative_validation_logs_metrics_and_flushes():
    trainer = _trainer_with_fake_logger()
    wandb_payloads = []
    trainer._wandb_log = wandb_payloads.append

    trainer._log_qualitative_validation_metrics(_qualitative_report())

    expected = {
        "qualitative/future_psnr_vs_gt": 20.0,
        "qualitative/future_ssim_vs_gt": 0.7,
        "qualitative/future_psnr_vs_vae": 21.0,
        "qualitative/future_ssim_vs_vae": 0.8,
        "qualitative/vae_psnr_vs_gt": 30.0,
        "qualitative/vae_ssim_vs_gt": 0.9,
        "qualitative/action_l1_denormalized": 0.1,
        "qualitative/action_mse_denormalized": 0.2,
        "timing/qualitative_validation_s": 123.0,
    }
    assert trainer.tensorboard_logger.records == [(expected, 8501)]
    assert trainer.tensorboard_logger.flush_count == 1
    assert wandb_payloads == [expected]


def test_training_log_contains_complete_canonical_metric_set():
    trainer = _trainer_with_fake_logger()
    wandb_payloads = []
    trainer._wandb_log = wandb_payloads.append

    trainer._log_training_metrics(
        loss_total=3.0,
        loss_metrics={"loss_action": 1.0, "loss_video": 2.0},
        learning_rate=2e-4,
        grad_norm=1.5,
        clip_event_rate=0.25,
        steps_per_second=0.1,
        samples_per_second=409.6,
        optimizer_skipped=2,
    )

    expected = {
        "train/loss_total": 3.0,
        "train/loss_action": 1.0,
        "train/loss_video": 2.0,
        "train/lr": 2e-4,
        "train/grad_norm": 1.5,
        "train/clip_event_rate": 0.25,
        "train/steps_per_sec": 0.1,
        "train/samples_per_sec": 409.6,
        "train/optimizer_skipped": 2,
    }
    assert trainer.tensorboard_logger.records == [(expected, 17)]
    assert wandb_payloads == [expected]


def test_non_main_rank_does_not_log_fixed_validation():
    trainer = _trainer_with_fake_logger(is_main_process=False)
    trainer._wandb_log = lambda payload: pytest.fail("non-main rank logged to W&B")

    trainer._log_fixed_validation_metrics(
        {
            "step": 1,
            "loss_total": 3.0,
            "loss_action": 1.0,
            "loss_video": 2.0,
        },
        duration_seconds=1.0,
    )

    assert trainer.tensorboard_logger.records == []


def test_non_main_rank_does_not_log_generic_scalars():
    trainer = _trainer_with_fake_logger(is_main_process=False)

    trainer._log_tensorboard_scalars({"train/loss_total": 1.0}, step=3)

    assert trainer.tensorboard_logger.records == []


def test_train_closes_tensorboard_and_wandb_when_training_raises():
    trainer = _trainer_with_fake_logger()
    finished = []
    trainer._finish_wandb = lambda: finished.append(True)

    def fail():
        raise RuntimeError("training failed")

    trainer._train_impl = fail

    with pytest.raises(RuntimeError, match="training failed"):
        trainer.train()

    assert trainer.tensorboard_logger.flush_count == 1
    assert trainer.tensorboard_logger.close_count == 1
    assert finished == [True]


def test_train_closes_loggers_after_early_successful_return():
    trainer = _trainer_with_fake_logger()
    finished = []
    trainer._finish_wandb = lambda: finished.append(True)
    trainer._train_impl = lambda: "profile complete"

    assert trainer.train() == "profile complete"
    assert trainer.tensorboard_logger.flush_count == 1
    assert trainer.tensorboard_logger.close_count == 1
    assert finished == [True]
