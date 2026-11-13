# Copyright (c) 2026 Xiaomi Corporation

from __future__ import annotations

from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from fastwam.utils.tensorboard import create_tensorboard_logger


def _scalars(log_dir: Path, tag: str) -> list[tuple[int, float]]:
    accumulator = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    return [(event.step, event.value) for event in accumulator.Scalars(tag)]


def test_real_tensorboard_writer_archives_and_resumes(tmp_path: Path) -> None:
    output_dir = tmp_path / "formal-run"
    platform_dir = tmp_path / "scheduler-tensorboard"
    config = {
        "enabled": True,
        "archive": True,
        "max_queue": 1,
        "flush_secs": 1,
    }

    logger = create_tensorboard_logger(
        output_dir=output_dir,
        config=config,
        is_main_process=True,
        environ={"TENSORBOARD_LOG_PATH": str(platform_dir)},
    )
    logger.log_scalars(
        {"train/loss_total": 1.25, "validation/loss_action": 0.75},
        step=1,
    )
    logger.flush()
    logger.close()

    archive_dir = output_dir / "tensorboard"
    assert _scalars(platform_dir, "train/loss_total") == [(1, 1.25)]
    assert _scalars(archive_dir, "validation/loss_action") == [(1, 0.75)]

    resumed = create_tensorboard_logger(
        output_dir=output_dir,
        config=config,
        is_main_process=True,
        resumed_completed_step=1,
        environ={"TENSORBOARD_LOG_PATH": str(platform_dir)},
    )
    resumed.log_scalars(
        {"train/loss_total": 1.0, "validation/loss_action": 0.5},
        step=2,
    )
    resumed.flush()
    resumed.close()

    expected_loss = [(1, 1.25), (2, 1.0)]
    expected_validation = [(1, 0.75), (2, 0.5)]
    assert _scalars(platform_dir, "train/loss_total") == expected_loss
    assert _scalars(archive_dir, "train/loss_total") == expected_loss
    assert _scalars(platform_dir, "validation/loss_action") == expected_validation
    assert _scalars(archive_dir, "validation/loss_action") == expected_validation
