# Copyright (c) 2026 Xiaomi Corporation

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest
from omegaconf import OmegaConf

import fastwam.runtime as runtime


def _cfg(output_dir: Path, *, value: int = 3):
    return OmegaConf.create(
        {
            "z_value": "${base}",
            "output_dir": str(output_dir),
            "base": value,
        }
    )


def _force_environment_rank(monkeypatch: pytest.MonkeyPatch, rank: int, world_size: int) -> None:
    monkeypatch.setattr(runtime.torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setenv("RANK", str(rank))
    monkeypatch.setenv("WORLD_SIZE", str(world_size))


def test_rank_zero_persists_deterministic_resolved_config_and_checksum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _force_environment_rank(monkeypatch, rank=0, world_size=8)
    output_dir = tmp_path / "run"

    paths = runtime._persist_resolved_config(_cfg(output_dir))

    assert paths == (output_dir / "config.yaml", output_dir / "config.yaml.sha256")
    payload = paths[0].read_bytes()
    assert payload == OmegaConf.to_yaml(
        _cfg(output_dir), resolve=True, sort_keys=True
    ).encode("utf-8")
    assert b"${base}" not in payload
    assert paths[1].read_text(encoding="ascii") == (
        f"{hashlib.sha256(payload).hexdigest()}  config.yaml\n"
    )
    assert not list(output_dir.glob(".*.tmp"))


def test_rank_zero_archives_resolved_config_under_resume_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _force_environment_rank(monkeypatch, rank=0, world_size=128)
    monkeypatch.setenv("FASTWAM_AUTO_RESUME_ATTEMPT_ID", "123")
    output_dir = tmp_path / "run"

    runtime._persist_resolved_config(_cfg(output_dir))

    archive = output_dir / "reports/auto-resume/attempt-123"
    payload = (output_dir / "config.yaml").read_bytes()
    assert (archive / "resolved-config.yaml").read_bytes() == payload
    assert (archive / "resolved-config.yaml.sha256").read_text(encoding="ascii") == (
        f"{hashlib.sha256(payload).hexdigest()}  resolved-config.yaml\n"
    )


def test_resume_attempt_id_rejects_path_traversal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _force_environment_rank(monkeypatch, rank=0, world_size=128)
    monkeypatch.setenv("FASTWAM_AUTO_RESUME_ATTEMPT_ID", "../escape")

    with pytest.raises(ValueError, match="unsafe FASTWAM_AUTO_RESUME_ATTEMPT_ID"):
        runtime._persist_resolved_config(_cfg(tmp_path / "run"))


def test_resume_attempt_archive_replace_failure_does_not_leave_partial_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _force_environment_rank(monkeypatch, rank=0, world_size=128)
    monkeypatch.setenv("FASTWAM_AUTO_RESUME_ATTEMPT_ID", "replace-failure")
    output_dir = tmp_path / "run"
    real_replace = os.replace
    calls = 0

    def fail_archive_checksum(source, destination):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise OSError("injected archive checksum replace failure")
        return real_replace(source, destination)

    monkeypatch.setattr(runtime.os, "replace", fail_archive_checksum)

    with pytest.raises(OSError, match="injected archive checksum replace failure"):
        runtime._persist_resolved_config(_cfg(output_dir))

    archive = output_dir / "reports/auto-resume/attempt-replace-failure"
    assert not (archive / "resolved-config.yaml").exists()
    assert not (archive / "resolved-config.yaml.sha256").exists()
    assert not list(archive.glob(".*.tmp"))


def test_nonzero_rank_does_not_create_or_write_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _force_environment_rank(monkeypatch, rank=7, world_size=8)
    output_dir = tmp_path / "run"

    assert runtime._persist_resolved_config(_cfg(output_dir)) is None

    assert not output_dir.exists()


def test_initialized_torch_distributed_rank_takes_precedence_over_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runtime.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(runtime.torch.distributed, "get_rank", lambda: 5)
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")

    output_dir = tmp_path / "run"
    assert runtime._persist_resolved_config(_cfg(output_dir)) is None
    assert not output_dir.exists()


def test_overwrite_uses_atomic_replacement_and_fsyncs_files_and_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _force_environment_rank(monkeypatch, rank=0, world_size=1)
    output_dir = tmp_path / "run"
    runtime._persist_resolved_config(_cfg(output_dir, value=1))
    config_path = output_dir / "config.yaml"
    old_payload = config_path.read_bytes()
    real_replace = os.replace
    replace_destinations = []
    fsync_kinds = []

    def observed_replace(source, destination):
        destination = Path(destination)
        replace_destinations.append(destination.name)
        if destination == config_path:
            assert destination.read_bytes() == old_payload
        return real_replace(source, destination)

    real_fsync = os.fsync

    def observed_fsync(fd):
        fsync_kinds.append("directory" if os.path.isdir(f"/proc/self/fd/{fd}") else "file")
        return real_fsync(fd)

    monkeypatch.setattr(runtime.os, "replace", observed_replace)
    monkeypatch.setattr(runtime.os, "fsync", observed_fsync)

    runtime._persist_resolved_config(_cfg(output_dir, value=9))

    assert replace_destinations == ["config.yaml", "config.yaml.sha256"]
    assert fsync_kinds.count("file") == 2
    assert fsync_kinds.count("directory") == 1
    assert config_path.read_bytes() != old_payload
    assert not list(output_dir.glob(".*.tmp"))


def test_replace_failure_preserves_existing_files_and_cleans_temporary_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _force_environment_rank(monkeypatch, rank=0, world_size=1)
    output_dir = tmp_path / "run"
    runtime._persist_resolved_config(_cfg(output_dir, value=1))
    config_path = output_dir / "config.yaml"
    checksum_path = output_dir / "config.yaml.sha256"
    old_config = config_path.read_bytes()
    old_checksum = checksum_path.read_bytes()

    def fail_replace(_source, _destination):
        raise OSError("injected replace failure")

    monkeypatch.setattr(runtime.os, "replace", fail_replace)

    with pytest.raises(OSError, match="injected replace failure"):
        runtime._persist_resolved_config(_cfg(output_dir, value=9))

    assert config_path.read_bytes() == old_config
    assert checksum_path.read_bytes() == old_checksum
    assert not list(output_dir.glob(".*.tmp"))


def test_checksum_replace_failure_rolls_back_config_and_checksum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _force_environment_rank(monkeypatch, rank=0, world_size=1)
    output_dir = tmp_path / "run"
    runtime._persist_resolved_config(_cfg(output_dir, value=1))
    config_path = output_dir / "config.yaml"
    checksum_path = output_dir / "config.yaml.sha256"
    old_config = config_path.read_bytes()
    old_checksum = checksum_path.read_bytes()
    real_replace = os.replace
    calls = 0

    def fail_second_replace(source, destination):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected checksum replace failure")
        return real_replace(source, destination)

    monkeypatch.setattr(runtime.os, "replace", fail_second_replace)

    with pytest.raises(OSError, match="injected checksum replace failure"):
        runtime._persist_resolved_config(_cfg(output_dir, value=9))

    assert config_path.read_bytes() == old_config
    assert checksum_path.read_bytes() == old_checksum
    assert not list(output_dir.glob(".*.tmp"))


@pytest.mark.parametrize(
    ("rank", "world_size", "message"),
    [
        (None, "8", "RANK is unset"),
        ("8", "8", "outside WORLD_SIZE"),
        ("0", "0", "must be positive"),
    ],
)
def test_invalid_distributed_environment_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    rank: str | None,
    world_size: str,
    message: str,
) -> None:
    monkeypatch.setattr(runtime.torch.distributed, "is_initialized", lambda: False)
    if rank is None:
        monkeypatch.delenv("RANK", raising=False)
    else:
        monkeypatch.setenv("RANK", rank)
    monkeypatch.setenv("WORLD_SIZE", world_size)

    with pytest.raises(RuntimeError, match=message):
        runtime._resolve_global_rank_before_accelerator()
