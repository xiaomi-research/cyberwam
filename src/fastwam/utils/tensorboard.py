# Copyright (c) 2026 Xiaomi Corporation

"""Rank-safe TensorBoard logging with scheduler path integration."""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol


logger = logging.getLogger(__name__)

TENSORBOARD_LOG_PATH_ENV = "TENSORBOARD_LOG_PATH"


class _SummaryWriter(Protocol):
    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> Any: ...

    def flush(self) -> Any: ...

    def close(self) -> Any: ...


WriterFactory = Callable[..., _SummaryWriter]


@dataclass(frozen=True)
class TensorBoardSettings:
    """TensorBoard settings parsed from the optional training config section."""

    enabled: bool = False
    archive: bool = True
    max_queue: int = 100
    flush_secs: int = 30

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None) -> "TensorBoardSettings":
        if config is None:
            return cls()
        get = getattr(config, "get", None)
        if get is None:
            raise TypeError("tensorboard config must provide a mapping-like get() method")

        settings = cls(
            enabled=bool(get("enabled", False)),
            archive=bool(get("archive", True)),
            max_queue=int(get("max_queue", 100)),
            flush_secs=int(get("flush_secs", 30)),
        )
        if settings.max_queue <= 0:
            raise ValueError("tensorboard.max_queue must be positive")
        if settings.flush_secs <= 0:
            raise ValueError("tensorboard.flush_secs must be positive")
        return settings


@dataclass(frozen=True)
class TensorBoardPaths:
    """Primary scheduler/fallback log directory and optional local archive."""

    log_dir: Path
    archive_dir: Path | None
    uses_platform_path: bool

    @property
    def all_dirs(self) -> tuple[Path, ...]:
        if self.archive_dir is None:
            return (self.log_dir,)
        return (self.log_dir, self.archive_dir)


def _normalized_path(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def resolve_tensorboard_paths(
    output_dir: str | os.PathLike[str],
    *,
    environ: Mapping[str, str] | None = None,
    archive: bool = True,
) -> TensorBoardPaths:
    """Resolve the scheduler-provided path first, with ``output/tensorboard`` as fallback.

    When the scheduler path points elsewhere, a second writer archives identical events in
    the persistent training output. Empty environment values are treated as
    absent.
    """

    env = os.environ if environ is None else environ
    fallback = _normalized_path(Path(output_dir) / "tensorboard")
    platform_value = str(env.get(TENSORBOARD_LOG_PATH_ENV, "")).strip()
    if not platform_value:
        return TensorBoardPaths(
            log_dir=fallback,
            archive_dir=None,
            uses_platform_path=False,
        )

    platform_dir = _normalized_path(Path(platform_value))
    archive_dir = fallback if archive and platform_dir != fallback else None
    return TensorBoardPaths(
        log_dir=platform_dir,
        archive_dir=archive_dir,
        uses_platform_path=True,
    )


def resume_purge_step(completed_step: int | None) -> int | None:
    """Map a restored completed step to TensorBoard's first stale step.

    A checkpoint at step N has valid events through N. TensorBoard must purge
    stale events beginning at N+1 before the resumed process writes N+1 again.
    """

    if completed_step is None:
        return None
    completed_step = int(completed_step)
    if completed_step < 0:
        raise ValueError("completed_step must be non-negative")
    return completed_step + 1


def _torch_writer_factory(**kwargs: Any) -> _SummaryWriter:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise RuntimeError(
            "TensorBoard logging is enabled but the tensorboard dependency is not "
            "installed. Install the fastwam project dependencies."
        ) from exc
    return SummaryWriter(**kwargs)


class TensorBoardLogger:
    """Small scalar-only facade over one or two TensorBoard writers."""

    def __init__(
        self,
        *,
        paths: TensorBoardPaths,
        writers: tuple[_SummaryWriter, ...] = (),
        purge_step: int | None = None,
    ) -> None:
        self.paths = paths
        self._writers = writers
        self.purge_step = purge_step
        self._closed = False

    @property
    def enabled(self) -> bool:
        return bool(self._writers) and not self._closed

    def log_scalars(self, metrics: Mapping[str, Any], *, step: int) -> None:
        """Write the same scalar mapping to every configured event directory."""

        if not self.enabled:
            return
        step = int(step)
        if step < 0:
            raise ValueError("TensorBoard global step must be non-negative")
        normalized: list[tuple[str, float]] = []
        for tag, value in metrics.items():
            if not isinstance(tag, str) or not tag.strip():
                raise ValueError("TensorBoard scalar tags must be non-empty strings")
            try:
                scalar = float(value)
            except (TypeError, ValueError) as exc:
                raise TypeError(f"TensorBoard scalar {tag!r} is not numeric: {value!r}") from exc
            normalized.append((tag, scalar))
        normalized.sort(key=lambda item: item[0])

        for writer in self._writers:
            for tag, value in normalized:
                writer.add_scalar(tag, value, global_step=step)

    def flush(self) -> None:
        if self._closed:
            return
        for writer in self._writers:
            writer.flush()

    def close(self) -> None:
        if self._closed:
            return
        errors: list[BaseException] = []
        for writer in self._writers:
            try:
                writer.close()
            except BaseException as exc:  # close every archive before propagating
                errors.append(exc)
        self._closed = True
        if errors:
            raise errors[0]

    def __enter__(self) -> "TensorBoardLogger":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


def create_tensorboard_logger(
    *,
    output_dir: str | os.PathLike[str],
    config: Mapping[str, Any] | None,
    is_main_process: bool,
    resumed_completed_step: int | None = None,
    environ: Mapping[str, str] | None = None,
    writer_factory: WriterFactory | None = None,
) -> TensorBoardLogger:
    """Create a rank-0 TensorBoard logger suitable for trainer integration.

    Non-primary ranks and disabled configurations never import TensorBoard,
    create directories, or instantiate writers. Pass the restored trainer's
    completed global step to ``resumed_completed_step`` after loading state.
    """

    settings = TensorBoardSettings.from_config(config)
    paths = resolve_tensorboard_paths(
        output_dir,
        environ=environ,
        archive=settings.archive,
    )
    purge_step = resume_purge_step(resumed_completed_step)
    if not settings.enabled or not is_main_process:
        return TensorBoardLogger(paths=paths, purge_step=purge_step)

    factory = _torch_writer_factory if writer_factory is None else writer_factory
    writers: list[_SummaryWriter] = []
    try:
        for path in paths.all_dirs:
            path.mkdir(parents=True, exist_ok=True)
            writers.append(
                factory(
                    log_dir=str(path),
                    purge_step=purge_step,
                    max_queue=settings.max_queue,
                    flush_secs=settings.flush_secs,
                )
            )
    except BaseException:
        for writer in writers:
            writer.close()
        raise

    logger.info(
        "Initialized TensorBoard: log_dir=%s archive_dir=%s purge_step=%s",
        paths.log_dir,
        paths.archive_dir,
        purge_step,
    )
    return TensorBoardLogger(
        paths=paths,
        writers=tuple(writers),
        purge_step=purge_step,
    )
