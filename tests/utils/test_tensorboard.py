# Copyright (c) 2026 Xiaomi Corporation

from pathlib import Path

import pytest

from fastwam.utils.tensorboard import (
    TENSORBOARD_LOG_PATH_ENV,
    TensorBoardSettings,
    create_tensorboard_logger,
    resolve_tensorboard_paths,
    resume_purge_step,
)


class FakeWriter:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.scalars = []
        self.flush_count = 0
        self.close_count = 0

    def add_scalar(self, tag, scalar_value, global_step):
        self.scalars.append((tag, scalar_value, global_step))

    def flush(self):
        self.flush_count += 1

    def close(self):
        self.close_count += 1


def test_settings_defaults_and_validation():
    assert TensorBoardSettings.from_config(None) == TensorBoardSettings()
    assert TensorBoardSettings.from_config({"enabled": True}).enabled is True
    with pytest.raises(ValueError, match="max_queue"):
        TensorBoardSettings.from_config({"max_queue": 0})
    with pytest.raises(ValueError, match="flush_secs"):
        TensorBoardSettings.from_config({"flush_secs": -1})


def test_paths_fall_back_to_output_directory(tmp_path: Path):
    paths = resolve_tensorboard_paths(tmp_path / "run", environ={})

    assert paths.log_dir == (tmp_path / "run" / "tensorboard").resolve()
    assert paths.archive_dir is None
    assert paths.uses_platform_path is False


def test_platform_path_is_primary_and_output_is_archive(tmp_path: Path):
    cloud = tmp_path / "cloud-events"
    paths = resolve_tensorboard_paths(
        tmp_path / "run",
        environ={TENSORBOARD_LOG_PATH_ENV: str(cloud)},
    )

    assert paths.log_dir == cloud.resolve()
    assert paths.archive_dir == (tmp_path / "run" / "tensorboard").resolve()
    assert paths.all_dirs == (paths.log_dir, paths.archive_dir)
    assert paths.uses_platform_path is True


def test_archive_can_be_disabled_for_platform_path(tmp_path: Path):
    paths = resolve_tensorboard_paths(
        tmp_path / "run",
        environ={TENSORBOARD_LOG_PATH_ENV: str(tmp_path / "cloud-events")},
        archive=False,
    )

    assert paths.archive_dir is None
    assert len(paths.all_dirs) == 1


def test_same_platform_and_archive_path_creates_only_one_writer(tmp_path: Path):
    fallback = tmp_path / "run" / "tensorboard"
    created = []

    logger = create_tensorboard_logger(
        output_dir=tmp_path / "run",
        config={"enabled": True},
        is_main_process=True,
        environ={TENSORBOARD_LOG_PATH_ENV: str(fallback)},
        writer_factory=lambda **kwargs: created.append(FakeWriter(**kwargs)) or created[-1],
    )

    assert len(created) == 1
    assert logger.paths.archive_dir is None


def test_non_main_rank_does_not_create_writer_or_directory(tmp_path: Path):
    calls = []
    cloud = tmp_path / "cloud-events"
    logger = create_tensorboard_logger(
        output_dir=tmp_path / "run",
        config={"enabled": True},
        is_main_process=False,
        environ={TENSORBOARD_LOG_PATH_ENV: str(cloud)},
        writer_factory=lambda **kwargs: calls.append(kwargs),
    )

    assert logger.enabled is False
    assert calls == []
    assert not cloud.exists()
    assert not (tmp_path / "run" / "tensorboard").exists()


def test_disabled_logger_does_not_require_tensorboard_dependency(tmp_path: Path):
    logger = create_tensorboard_logger(
        output_dir=tmp_path,
        config={"enabled": False},
        is_main_process=True,
        environ={},
    )

    logger.log_scalars({"train/loss": 1.0}, step=1)
    logger.flush()
    logger.close()
    assert logger.enabled is False


def test_resume_uses_next_step_as_purge_boundary_and_dual_writes(tmp_path: Path):
    created = []
    logger = create_tensorboard_logger(
        output_dir=tmp_path / "run",
        config={"enabled": True, "archive": True, "max_queue": 7, "flush_secs": 11},
        is_main_process=True,
        resumed_completed_step=1000,
        environ={TENSORBOARD_LOG_PATH_ENV: str(tmp_path / "cloud")},
        writer_factory=lambda **kwargs: created.append(FakeWriter(**kwargs)) or created[-1],
    )

    assert logger.purge_step == 1001
    assert len(created) == 2
    assert all(writer.kwargs["purge_step"] == 1001 for writer in created)
    assert all(writer.kwargs["max_queue"] == 7 for writer in created)
    assert all(writer.kwargs["flush_secs"] == 11 for writer in created)

    logger.log_scalars({"train/loss_video": 2, "train/loss_action": 1.5}, step=1001)
    expected = [
        ("train/loss_action", 1.5, 1001),
        ("train/loss_video", 2.0, 1001),
    ]
    assert all(writer.scalars == expected for writer in created)

    logger.flush()
    logger.close()
    logger.close()
    assert all(writer.flush_count == 1 for writer in created)
    assert all(writer.close_count == 1 for writer in created)
    assert logger.enabled is False


def test_resume_purge_step_validation():
    assert resume_purge_step(None) is None
    assert resume_purge_step(0) == 1
    assert resume_purge_step(42) == 43
    with pytest.raises(ValueError, match="non-negative"):
        resume_purge_step(-1)


def test_invalid_scalar_fails_before_any_writer_is_modified(tmp_path: Path):
    created = []
    logger = create_tensorboard_logger(
        output_dir=tmp_path,
        config={"enabled": True},
        is_main_process=True,
        environ={},
        writer_factory=lambda **kwargs: created.append(FakeWriter(**kwargs)) or created[-1],
    )

    with pytest.raises(TypeError, match="not numeric"):
        logger.log_scalars({"train/loss": object()}, step=1)
    assert created[0].scalars == []
    with pytest.raises(ValueError, match="non-empty strings"):
        logger.log_scalars({"train/loss": 1.0, 2: 3.0}, step=1)
    assert created[0].scalars == []
    with pytest.raises(ValueError, match="non-negative"):
        logger.log_scalars({"train/loss": 1.0}, step=-1)
