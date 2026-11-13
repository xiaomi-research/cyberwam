# Copyright (c) 2026 Xiaomi Corporation
# Portions derived from FastWAM (MIT License, https://github.com/yuantianyuan01/FastWAM); see NOTICE.md.

import logging
import json
import hashlib
import inspect
import os
import re
import shutil
import tempfile
from math import ceil
from pathlib import Path
import time

import numpy as np
import torch
from accelerate import Accelerator
from omegaconf import DictConfig
from PIL import Image
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from .utils.fs import ensure_dir
from .utils.checkpoint_contract import (
    CheckpointContract,
    CheckpointContractMismatchError,
    CheckpointContractIntegrityError,
    assert_checkpoint_compatible,
    read_checkpoint_contract,
    write_checkpoint_contract,
)
from .utils.logging_config import get_logger, setup_logging
from .utils.pytorch_utils import (
    INITIALIZATION_FINGERPRINT_ATTRIBUTE,
    set_global_seed,
)
from .utils.samplers import ResumableEpochSampler
from .utils.tensorboard import create_tensorboard_logger
from .utils.video_io import save_mp4
from .utils.video_metrics import pil_frames_to_video_tensor, video_psnr, video_ssim

logger = get_logger(__name__)

_CHECKPOINT_COMPLETE_MARKER = "checkpoint_complete.json"
_CHECKPOINT_COMPLETE_SCHEMA_VERSION = 1
_CHECKPOINT_INVENTORY_SCHEMA_VERSION = 1
_CHECKPOINT_INVENTORY_SHA256_MAX_BYTES = 1024 * 1024
_TRAINING_COMPLETE_MARKER = "training_complete.json"
_TRAINING_COMPLETE_SCHEMA_VERSION = 1


def _validate_initial_checkpoint_sources(
    *,
    resume,
    legacy_warmstart=None,
    flex_warmstart=None,
) -> None:
    """Ensure a run selects at most one initial model/checkpoint source."""
    _resolve_initial_checkpoint_sources(
        resume=resume,
        legacy_warmstart=legacy_warmstart,
        flex_warmstart=flex_warmstart,
    )


def _resolve_initial_checkpoint_sources(
    *,
    resume,
    legacy_warmstart=None,
    flex_warmstart=None,
) -> tuple[object, object]:
    """Resolve the mutually exclusive canonical checkpoint sources."""
    sources = {
        "resume": resume,
        "legacy_warmstart": legacy_warmstart,
        "flex_warmstart": flex_warmstart,
    }
    selected = [name for name, value in sources.items() if value]
    if len(selected) > 1:
        raise ValueError(
            "`resume`, `legacy_warmstart`, and `flex_warmstart` are mutually "
            f"exclusive; selected={selected}."
        )
    return legacy_warmstart, flex_warmstart


def _checkpoint_inventory_is_excluded(relative_path: Path) -> bool:
    """Exclude the commit marker and atomic-write scratch files from inventory."""
    if relative_path == Path(_CHECKPOINT_COMPLETE_MARKER):
        return True
    return relative_path.name.endswith(".tmp")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_inventory_entry(path: Path, *, inventory_path: str) -> dict:
    """Describe one checkpoint file without hashing multi-GiB model shards."""
    initial = path.stat()
    if not path.is_file():
        raise RuntimeError(f"Checkpoint inventory path is not a regular file: {path}")
    entry = {
        "path": inventory_path,
        "size_bytes": int(initial.st_size),
    }
    if initial.st_size <= _CHECKPOINT_INVENTORY_SHA256_MAX_BYTES:
        entry["sha256"] = _sha256_file(path)
        final = path.stat()
        if (final.st_size, final.st_mtime_ns) != (
            initial.st_size,
            initial.st_mtime_ns,
        ):
            raise RuntimeError(
                f"Checkpoint file changed while its inventory was built: {path}"
            )
    return entry


def _build_checkpoint_inventory(state_path: Path, weights_path: Path) -> dict:
    """Build the exact file inventory committed by a checkpoint marker.

    Small files are content-bound. Large optimizer/model shards deliberately
    use size-only validation so publishing the marker does not reread roughly
    91 GiB from shared storage after every save.
    """
    if not state_path.is_dir():
        raise RuntimeError(f"Checkpoint state directory is missing: {state_path}")
    if not weights_path.is_file():
        raise RuntimeError(f"Checkpoint weights file is missing: {weights_path}")

    entries = []
    for path in sorted(state_path.rglob("*")):
        if not path.is_file():
            continue
        relative_path = path.relative_to(state_path)
        if _checkpoint_inventory_is_excluded(relative_path):
            continue
        entries.append(
            _checkpoint_inventory_entry(
                path,
                inventory_path=f"state/{relative_path.as_posix()}",
            )
        )
    entries.append(
        _checkpoint_inventory_entry(
            weights_path,
            inventory_path=f"weights/{weights_path.name}",
        )
    )
    entries.sort(key=lambda item: item["path"])
    return {
        "schema_version": _CHECKPOINT_INVENTORY_SCHEMA_VERSION,
        "sha256_algorithm": "sha256",
        "sha256_max_bytes": _CHECKPOINT_INVENTORY_SHA256_MAX_BYTES,
        "large_file_validation": "path_and_size_only",
        "files": entries,
    }


def _checkpoint_inventory_is_valid(
    payload: object,
    *,
    state_path: Path,
    weights_path: Path,
) -> bool:
    """Validate inventory structure and compare it with the exact live file set."""
    if not isinstance(payload, dict):
        return False
    required_header = {
        "schema_version": _CHECKPOINT_INVENTORY_SCHEMA_VERSION,
        "sha256_algorithm": "sha256",
        "sha256_max_bytes": _CHECKPOINT_INVENTORY_SHA256_MAX_BYTES,
        "large_file_validation": "path_and_size_only",
    }
    if set(payload) != {*required_header, "files"}:
        return False
    if any(payload.get(key) != value for key, value in required_header.items()):
        return False
    files = payload.get("files")
    if not isinstance(files, list) or not files:
        return False
    paths = []
    for entry in files:
        if not isinstance(entry, dict) or set(entry) not in (
            {"path", "size_bytes"},
            {"path", "size_bytes", "sha256"},
        ):
            return False
        inventory_path = entry.get("path")
        size_bytes = entry.get("size_bytes")
        if (
            not isinstance(inventory_path, str)
            or not inventory_path
            or not isinstance(size_bytes, int)
            or isinstance(size_bytes, bool)
            or size_bytes < 0
        ):
            return False
        paths.append(inventory_path)
        sha256 = entry.get("sha256")
        if size_bytes <= _CHECKPOINT_INVENTORY_SHA256_MAX_BYTES:
            if (
                not isinstance(sha256, str)
                or re.fullmatch(r"[0-9a-f]{64}", sha256) is None
            ):
                return False
        elif "sha256" in entry:
            return False
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        return False
    try:
        actual = _build_checkpoint_inventory(state_path, weights_path)
    except (OSError, RuntimeError):
        return False
    return payload == actual


def build_fixed_validation_window_indices(
    dataset,
    *,
    tasks: int = 100,
    anchors_per_task: int = 10,
    stride: int = 6,
) -> list[tuple[int, int, int]]:
    """Return deterministic ``(window_id, episode, global_index)`` tuples.

    B1K validation contains one held-out episode per task.  Anchors are
    selected episode-locally at ``0, stride, ...``; this avoids accidentally
    crossing an episode boundary when the dataset is flattened.
    """
    if tasks <= 0 or anchors_per_task <= 0 or stride <= 0:
        raise ValueError("tasks, anchors_per_task and stride must be positive")
    episodes = getattr(dataset, "selected_episodes", None)
    ranges = getattr(dataset, "episode_data_index", None)
    if episodes is None or ranges is None:
        raise TypeError("fixed B1K validation requires selected_episodes and episode_data_index")
    dataset_stride = getattr(dataset, "window_start_stride", None)
    if dataset_stride is not None and int(dataset_stride) != int(stride):
        raise ValueError(
            f"validation dataset window_start_stride={dataset_stride} does not match anchor_stride={stride}"
        )
    if len(episodes) != tasks:
        raise ValueError(f"expected {tasks} validation episodes (one per task), got {len(episodes)}")
    starts = ranges["from"].tolist()
    ends = ranges["to"].tolist()
    if len(starts) != len(episodes) or len(ends) != len(episodes):
        raise ValueError("episode_data_index does not match selected_episodes")
    output: list[tuple[int, int, int]] = []
    for episode_pos, episode in enumerate(episodes):
        length = int(ends[episode_pos] - starts[episode_pos])
        if length < anchors_per_task:
            raise ValueError(
                f"validation episode {episode} has only {length} anchors, need {anchors_per_task}"
            )
        # Dataset ranges are already in anchor-index space when stride is
        # enabled. Select evenly over the complete held-out episode; do not
        # apply stride a second time.
        anchor_positions = torch.linspace(0, length - 1, anchors_per_task).round().to(torch.long).tolist()
        anchors = [int(value) for value in anchor_positions]
        for anchor in anchors:
            output.append((len(output), int(episode), int(starts[episode_pos] + anchor)))
    return output


def fixed_validation_seed(
    window_id: int, episode_index: int, frame_index: int, *, base_seed: int = 42
) -> int:
    """Stable 32-bit diffusion RNG seed derived from the window identity."""
    payload = f"b1k-fixed-validation-v1:{base_seed}:{window_id}:{episode_index}:{frame_index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def fixed_validation_window_set_sha256(windows: list[tuple[int, int, int]]) -> str:
    """Hash the ordered fixed-validation window identities."""
    digest = hashlib.sha256(b"b1k-fixed-validation-window-set-v1\0")
    for window_id, episode_index, global_index in windows:
        payload = json.dumps(
            [int(window_id), int(episode_index), int(global_index)],
            separators=(",", ":"),
        ).encode("ascii")
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


def _summarize_fixed_validation_rank_timings(rows: torch.Tensor) -> dict:
    """Build a JSON-safe timing report from one row per distributed rank."""
    if rows.ndim != 2 or rows.shape[1] != 7:
        raise ValueError(
            "fixed-validation timing rows must have shape [world_size, 7], "
            f"got {tuple(rows.shape)}"
        )
    records = []
    for row in rows.detach().cpu().double().tolist():
        records.append(
            {
                "rank": int(row[0]),
                "windows": int(row[1]),
                "data_load_s": float(row[2]),
                "forward_s": float(row[3]),
                "total_s": float(row[4]),
                "started_unix_s": float(row[5]),
                "finished_unix_s": float(row[6]),
            }
        )
    records.sort(key=lambda record: record["rank"])
    ranks = [record["rank"] for record in records]
    if ranks != list(range(len(records))):
        raise RuntimeError(
            "fixed-validation timing rank inventory is incomplete: "
            f"expected={list(range(len(records)))}, got={ranks}"
        )

    def summary(key: str) -> dict[str, float]:
        values = np.asarray([record[key] for record in records], dtype=np.float64)
        return {
            "min_seconds": float(values.min()),
            "mean_seconds": float(values.mean()),
            "p50_seconds": float(np.percentile(values, 50)),
            "p95_seconds": float(np.percentile(values, 95)),
            "max_seconds": float(values.max()),
        }

    return {
        "rank_count": len(records),
        "window_count": int(sum(record["windows"] for record in records)),
        "data_load_s": summary("data_load_s"),
        "forward_s": summary("forward_s"),
        "total_s": summary("total_s"),
        "wall_span_s": float(
            max(record["finished_unix_s"] for record in records)
            - min(record["started_unix_s"] for record in records)
        ),
        "ranks": records,
    }


def fixed_validation_due(
    step: int,
    *,
    explicit_steps: tuple[int, ...] | list[int],
    every_steps: int = 0,
) -> bool:
    """Return whether deterministic loss validation is due at ``step``.

    Explicit steps are useful for short experiments such as LR sweeps, while
    ``every_steps`` supports long training without materializing a large step
    list. A positive interval includes step zero so a run records its baseline.
    """
    step = int(step)
    every_steps = int(every_steps)
    if step < 0:
        raise ValueError("fixed-validation step must be non-negative")
    if every_steps < 0:
        raise ValueError("fixed_validation.every_steps must be non-negative")
    return step in explicit_steps or (every_steps > 0 and step % every_steps == 0)


def qualitative_validation_due(
    step: int,
    *,
    explicit_steps: tuple[int, ...] | list[int],
    every_steps: int = 0,
) -> bool:
    """Return whether qualitative inference is due, always excluding step zero."""
    step = int(step)
    every_steps = int(every_steps)
    if step < 0:
        raise ValueError("qualitative-validation step must be non-negative")
    if every_steps < 0:
        raise ValueError("qualitative_validation.every_steps must be non-negative")
    if step == 0:
        return False
    return step in explicit_steps or (every_steps > 0 and step % every_steps == 0)


def _atomic_write_json(path: str | os.PathLike[str], payload: dict) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=True, indent=2, sort_keys=True)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, output_path)
        # Persist the rename itself.  Without syncing the containing directory,
        # a power loss can lose an otherwise fsynced completion marker.
        directory_fd = os.open(output_path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
        raise


def _read_checkpoint_complete_marker(
    state_path: Path,
    weights_path: Path,
) -> dict | None:
    """Return a validated completion marker, or ``None`` for incomplete state.

    The marker is the commit record for the two-directory checkpoint.  Merely
    observing a state directory and a weights file is not sufficient because
    either may be visible while another rank is still writing its shard/probe.
    """
    marker_path = state_path / _CHECKPOINT_COMPLETE_MARKER
    if not marker_path.is_file():
        return None
    try:
        payload = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None

    match = re.fullmatch(r"step_(\d+)", state_path.name)
    if match is None:
        return None
    step = int(match.group(1))
    required = {
        "schema_version": _CHECKPOINT_COMPLETE_SCHEMA_VERSION,
        "global_step": step,
        "state_directory": state_path.name,
        "weights_file": weights_path.name,
    }
    if any(payload.get(key) != value for key, value in required.items()):
        return None
    retention_class = payload.get("retention_class")
    if retention_class is not None and retention_class not in {
        "rolling",
        "milestone",
        "final",
    }:
        return None
    trainer_state_path = state_path / "trainer_state.json"
    if not weights_path.is_file() or not trainer_state_path.is_file():
        return None
    try:
        trainer_state = json.loads(trainer_state_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if trainer_state.get("global_step") != step:
        return None
    contract_path = state_path / "checkpoint_contract.json"
    if payload.get("checkpoint_contract_required") and not contract_path.is_file():
        return None
    if payload.get("checkpoint_contract_required"):
        try:
            actual_contract = read_checkpoint_contract(contract_path)
        except (OSError, ValueError, TypeError, CheckpointContractIntegrityError):
            return None
        expected_hash = payload.get("checkpoint_contract_sha256")
        if (
            expected_hash != actual_contract.contract_sha256
            or trainer_state.get("checkpoint_contract_sha256") != expected_hash
        ):
            return None
    elif payload.get("checkpoint_contract_sha256") is not None:
        return None

    try:
        world_size = int(payload["world_size"])
        completed_at_ns = int(payload["completed_at_unix_ns"])
        resume_probe_count = int(payload["resume_probe_count"])
    except (KeyError, TypeError, ValueError):
        return None
    if world_size <= 0 or completed_at_ns <= 0 or resume_probe_count < 0:
        return None
    actual_probe_count = sum(
        1 for _ in state_path.glob("resume_probe_rank_*.json")
    )
    if actual_probe_count != resume_probe_count:
        return None
    if payload.get("resume_probes_required") and resume_probe_count != world_size:
        return None
    # Inventory was added without bumping the completion-marker schema so
    # existing checkpoints remain resumable. New markers always contain it.
    if "inventory" in payload and not _checkpoint_inventory_is_valid(
        payload["inventory"],
        state_path=state_path,
        weights_path=weights_path,
    ):
        return None
    return payload


def read_valid_training_completion(
    output_dir: str | os.PathLike[str],
    *,
    expected_max_steps: int,
    expected_world_size: int,
    expected_run_manifest_sha256: str,
) -> dict | None:
    """Return a validated natural-completion record, never infer it from step."""
    if (
        isinstance(expected_max_steps, bool)
        or not isinstance(expected_max_steps, int)
        or expected_max_steps <= 0
    ):
        raise ValueError("expected_max_steps must be a positive integer")
    if (
        isinstance(expected_world_size, bool)
        or not isinstance(expected_world_size, int)
        or expected_world_size <= 0
    ):
        raise ValueError("expected_world_size must be a positive integer")
    if re.fullmatch(r"[0-9a-f]{64}", expected_run_manifest_sha256) is None:
        raise ValueError("expected_run_manifest_sha256 must be a lowercase SHA256")

    root = Path(output_dir).expanduser().resolve(strict=False)
    marker_path = root / _TRAINING_COMPLETE_MARKER
    try:
        payload = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None

    required_keys = {
        "schema_version",
        "status",
        "global_step",
        "max_steps",
        "world_size",
        "state_directory",
        "weights_file",
        "checkpoint_complete_sha256",
        "checkpoint_contract_sha256",
        "run_manifest_sha256",
        "run_id",
        "training_summary_sha256",
        "completed_at_unix_ns",
    }
    if set(payload) != required_keys:
        return None
    required_values = {
        "schema_version": _TRAINING_COMPLETE_SCHEMA_VERSION,
        "status": "completed",
        "global_step": expected_max_steps,
        "max_steps": expected_max_steps,
        "world_size": expected_world_size,
        "run_manifest_sha256": expected_run_manifest_sha256,
    }
    if any(payload.get(key) != value for key, value in required_values.items()):
        return None
    if (
        not isinstance(payload.get("completed_at_unix_ns"), int)
        or isinstance(payload.get("completed_at_unix_ns"), bool)
        or payload["completed_at_unix_ns"] <= 0
        or not isinstance(payload.get("run_id"), str)
        or not payload["run_id"]
    ):
        return None
    hash_keys = (
        "checkpoint_complete_sha256",
        "checkpoint_contract_sha256",
        "training_summary_sha256",
    )
    if any(
        re.fullmatch(r"[0-9a-f]{64}", payload.get(key, "")) is None
        for key in hash_keys
    ):
        return None

    step_tag = f"step_{expected_max_steps:06d}"
    weights_name = f"{step_tag}.pt"
    if (
        payload["state_directory"] != step_tag
        or payload["weights_file"] != weights_name
    ):
        return None
    state_path = root / "checkpoints" / "state" / step_tag
    weights_path = root / "checkpoints" / "weights" / weights_name
    checkpoint_marker_path = state_path / _CHECKPOINT_COMPLETE_MARKER
    summary_path = root / "training-summary.json"
    if not checkpoint_marker_path.is_file() or not summary_path.is_file():
        return None
    try:
        if (
            _sha256_file(checkpoint_marker_path)
            != payload["checkpoint_complete_sha256"]
        ):
            return None
        if _sha256_file(summary_path) != payload["training_summary_sha256"]:
            return None
    except OSError:
        return None

    checkpoint = _read_checkpoint_complete_marker(state_path, weights_path)
    if checkpoint is None:
        return None
    if (
        checkpoint.get("global_step") != expected_max_steps
        or checkpoint.get("world_size") != expected_world_size
        or checkpoint.get("checkpoint_contract_required") is not True
        or checkpoint.get("checkpoint_contract_sha256")
        != payload["checkpoint_contract_sha256"]
        or checkpoint.get("resume_probes_required") is not True
        or checkpoint.get("resume_probe_count") != expected_world_size
        or "inventory" not in checkpoint
    ):
        return None
    try:
        contract = read_checkpoint_contract(state_path / "checkpoint_contract.json")
    except (OSError, ValueError, TypeError, CheckpointContractIntegrityError):
        return None
    contract_manifest = contract.provenance.get("run_manifest")
    if (
        not isinstance(contract_manifest, dict)
        or contract_manifest.get("sha256") != expected_run_manifest_sha256
        or contract_manifest.get("run_id") != payload["run_id"]
    ):
        return None
    return payload


def discover_latest_complete_checkpoint(
    output_dir: str | os.PathLike[str],
    *,
    expected_world_size: int | None = None,
    require_strict_metadata: bool = False,
    maximum_step_exclusive: int | None = None,
) -> dict:
    """Discover the highest-step resumable full-state checkpoint.

    A standalone weights file is intentionally never eligible: resuming formal
    training must restore optimizer, scheduler, sampler and trainer progress as
    one committed state. Invalid or interrupted candidates are reported rather
    than selected so callers can make the automatic decision observable.
    Strict mode additionally requires the current inventory, checkpoint
    contract and one resume probe per process for the expected world size.
    """
    if expected_world_size is not None:
        if (
            isinstance(expected_world_size, bool)
            or not isinstance(expected_world_size, int)
            or expected_world_size <= 0
        ):
            raise ValueError("expected_world_size must be a positive integer or None")
    if require_strict_metadata and expected_world_size is None:
        raise ValueError(
            "require_strict_metadata=True requires expected_world_size"
        )
    if maximum_step_exclusive is not None:
        if (
            isinstance(maximum_step_exclusive, bool)
            or not isinstance(maximum_step_exclusive, int)
            or maximum_step_exclusive <= 0
        ):
            raise ValueError("maximum_step_exclusive must be a positive integer or None")

    output_root = Path(output_dir).expanduser().resolve(strict=False)
    state_root = output_root / "checkpoints" / "state"
    weights_root = output_root / "checkpoints" / "weights"
    rejected: list[str] = []
    complete: list[tuple[int, Path]] = []
    state_tags: set[str] = set()

    if state_root.is_dir():
        for state_path in sorted(
            state_root.iterdir(), key=lambda path: path.name
        ):
            match = re.fullmatch(r"step_(\d+)", state_path.name)
            if match is None or not state_path.is_dir():
                continue
            state_tags.add(state_path.name)
            step = int(match.group(1))
            if (
                maximum_step_exclusive is not None
                and step >= maximum_step_exclusive
            ):
                rejected.append(
                    f"{state_path}: step is not below resume ceiling "
                    f"{maximum_step_exclusive}"
                )
                continue
            weights_path = weights_root / f"{state_path.name}.pt"
            marker_path = state_path / _CHECKPOINT_COMPLETE_MARKER
            marker = _read_checkpoint_complete_marker(state_path, weights_path)
            if marker is None:
                reason = (
                    "invalid completion marker or committed inventory"
                    if marker_path.exists()
                    else "missing completion marker"
                )
                rejected.append(f"{state_path}: {reason}")
                continue
            metadata_rejections = []
            if expected_world_size is not None and marker.get("world_size") != (
                expected_world_size
            ):
                metadata_rejections.append(
                    "world size mismatch "
                    f"(checkpoint={marker.get('world_size')}, "
                    f"expected={expected_world_size})"
                )
            if require_strict_metadata:
                if "inventory" not in marker:
                    metadata_rejections.append("missing committed inventory")
                if marker.get("checkpoint_contract_required") is not True:
                    metadata_rejections.append("strict checkpoint contract not required")
                if marker.get("resume_probes_required") is not True:
                    metadata_rejections.append("rank resume probes not required")
                if marker.get("resume_probe_count") != expected_world_size:
                    metadata_rejections.append(
                        "resume probe count mismatch "
                        f"(checkpoint={marker.get('resume_probe_count')}, "
                        f"expected={expected_world_size})"
                    )
            if metadata_rejections:
                rejected.append(
                    f"{state_path}: " + "; ".join(metadata_rejections)
                )
                continue
            complete.append((step, state_path))

    # A natural-end weights-only checkpoint can have the greatest step. It is
    # useful for inference but must never shadow the latest full training state.
    if weights_root.is_dir():
        for weights_path in sorted(
            weights_root.iterdir(), key=lambda path: path.name
        ):
            match = re.fullmatch(r"step_(\d+)\.pt", weights_path.name)
            if match is None or not weights_path.is_file():
                continue
            state_tag = f"step_{match.group(1)}"
            if state_tag not in state_tags:
                rejected.append(
                    f"{weights_path}: standalone weights have no full training state"
                )

    # Step is the training timeline authority. Completion timestamps are only
    # retention metadata and may be equal or externally restored from backup.
    selected = max(complete, key=lambda item: item[0]) if complete else None
    return {
        "mode": "latest_complete",
        "resume_path": str(selected[1]) if selected is not None else None,
        "global_step": selected[0] if selected is not None else None,
        "rejected_candidates": rejected,
    }


def _path_size_bytes(path: str | os.PathLike[str] | None) -> int:
    if path is None:
        return 0
    input_path = Path(path)
    if not input_path.exists():
        return 0
    if input_path.is_file():
        return input_path.stat().st_size
    return sum(
        child.stat().st_size
        for child in input_path.rglob("*")
        if child.is_file()
    )


def _distribution_summary(values) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"median": 0.0, "p90": 0.0, "mean": 0.0, "max": 0.0}
    return {
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "mean": float(np.mean(array)),
        "max": float(np.max(array)),
    }


def _resolve_dataloader_config(
    *,
    num_workers,
    prefetch_factor=2,
    persistent_workers=False,
) -> tuple[int, int | None, bool]:
    """Validate loader settings and return the values that DataLoader receives."""
    if isinstance(num_workers, bool):
        raise TypeError("`num_workers` must be a non-negative integer, not bool.")
    try:
        resolved_num_workers = int(num_workers)
    except (TypeError, ValueError) as error:
        raise TypeError("`num_workers` must be a non-negative integer.") from error
    if resolved_num_workers < 0:
        raise ValueError("`num_workers` must be non-negative.")
    if isinstance(num_workers, float) and not num_workers.is_integer():
        raise TypeError("`num_workers` must be a non-negative integer.")

    if prefetch_factor is None:
        resolved_prefetch_factor = None
    else:
        if isinstance(prefetch_factor, bool):
            raise TypeError("`prefetch_factor` must be a positive integer or null.")
        try:
            resolved_prefetch_factor = int(prefetch_factor)
        except (TypeError, ValueError) as error:
            raise TypeError(
                "`prefetch_factor` must be a positive integer or null."
            ) from error
        if resolved_prefetch_factor <= 0:
            raise ValueError("`prefetch_factor` must be positive or null.")
        if isinstance(prefetch_factor, float) and not prefetch_factor.is_integer():
            raise TypeError("`prefetch_factor` must be a positive integer or null.")

    if not isinstance(persistent_workers, bool):
        raise TypeError("`persistent_workers` must be a boolean.")
    if resolved_num_workers == 0:
        if persistent_workers:
            raise ValueError(
                "`persistent_workers=true` requires `num_workers` greater than zero."
            )
        # PyTorch requires prefetch_factor=None when loading in the main process.
        resolved_prefetch_factor = None

    return resolved_num_workers, resolved_prefetch_factor, persistent_workers


def _resolve_checkpoint_retention_max_to_keep(value) -> int | None:
    """Validate the optional number of complete checkpoint pairs to retain."""
    if value in (None, "", "null"):
        return None
    if isinstance(value, bool):
        raise TypeError("`checkpoint_retention.max_to_keep` must be a positive integer or null.")
    try:
        resolved = int(value)
    except (TypeError, ValueError) as error:
        raise TypeError(
            "`checkpoint_retention.max_to_keep` must be a positive integer or null."
        ) from error
    if resolved <= 0 or (isinstance(value, float) and not value.is_integer()):
        raise ValueError(
            "`checkpoint_retention.max_to_keep` must be a positive integer or null."
        )
    return resolved


def _resolve_checkpoint_retention_keep_every_n_steps(value) -> int | None:
    """Validate the optional checkpoint milestone interval."""
    if isinstance(value, bool):
        raise TypeError(
            "`checkpoint_retention.keep_every_n_steps` must be a positive integer or null."
        )
    if value in (None, "", "null", 0, "0"):
        return None
    try:
        resolved = int(value)
    except (TypeError, ValueError) as error:
        raise TypeError(
            "`checkpoint_retention.keep_every_n_steps` must be a positive integer or null."
        ) from error
    if resolved <= 0 or (isinstance(value, float) and not value.is_integer()):
        raise ValueError(
            "`checkpoint_retention.keep_every_n_steps` must be a positive integer or null."
        )
    return resolved


def _resolve_permanent_weights_every_n_steps(value) -> int | None:
    if isinstance(value, bool):
        raise TypeError(
            "`checkpoint_retention.permanent_weights_every_n_steps` must be "
            "a positive integer or null."
        )
    if value in (None, "", "null", 0, "0"):
        return None
    if not isinstance(value, (int, float, str)):
        raise TypeError(
            "`checkpoint_retention.permanent_weights_every_n_steps` must be "
            "a positive integer or null."
        )
    try:
        resolved = int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "`checkpoint_retention.permanent_weights_every_n_steps` must be "
            "a positive integer or null."
        ) from exc
    if resolved <= 0 or str(value).strip() not in {str(resolved), f"{resolved}.0"}:
        raise ValueError(
            "`checkpoint_retention.permanent_weights_every_n_steps` must be "
            "a positive integer or null."
        )
    return resolved


def _resolve_checkpoint_retention_keep_final(value) -> bool:
    """Validate whether the natural-end checkpoint is permanent."""
    if not isinstance(value, bool):
        raise TypeError("`checkpoint_retention.keep_final` must be a boolean.")
    return value


def _resolve_lr_warmup_steps(total_train_steps: int, configured_warmup) -> int:
    """Resolve the exact warmup length used by the scheduler."""
    total_train_steps = int(total_train_steps)
    if total_train_steps <= 0:
        raise ValueError("total_train_steps must be positive")
    warmup_steps = (
        int(total_train_steps * 0.05)
        if configured_warmup is None
        else int(configured_warmup)
    )
    if warmup_steps < 0:
        raise ValueError("lr_warmup_steps must be non-negative")
    return min(warmup_steps, total_train_steps - 1)


def _accepts_keyword(callable_obj, keyword: str) -> bool:
    signature = inspect.signature(callable_obj)
    return keyword in signature.parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


def _get_local_parameter_gradient(parameter: torch.nn.Parameter):
    """Return a gradient tensor without materializing a full ZeRO gradient."""
    if parameter.grad is not None:
        return parameter.grad

    # DeepSpeed ZeRO-1/2 can move the low-precision gradient here during its
    # backward epilogue, before user code regains control.
    grad_accum = getattr(parameter, "grad_accum", None)
    if grad_accum is not None:
        return grad_accum

    # ZeRO-3 keeps only a local gradient partition on each rank.
    if hasattr(parameter, "ds_id"):
        from deepspeed.utils import safe_get_local_grad

        return safe_get_local_grad(parameter)

    # ZeRO-1/2 may clear ``param.grad`` after reduction while retaining the
    # local low-precision fragment until ``engine.step()``.
    mapping = getattr(parameter, "_hp_mapping", None)
    parameter_index = getattr(parameter, "_index_in_param_group", None)
    if mapping is not None and parameter_index is not None:
        try:
            return mapping.get_lp_grad_fragment(parameter_index)
        except ValueError:
            return None
    return None


def _local_gradient_gate_stats(parameters, loss_finite: torch.Tensor) -> torch.Tensor:
    """Build fixed-size local statistics for the first real backward pass."""
    trainable_count = 0
    gradient_count = 0
    finite_flags = []
    nonzero_finite_flags = []
    for parameter in parameters:
        if not parameter.requires_grad:
            continue
        trainable_count += 1
        gradient = _get_local_parameter_gradient(parameter)
        if gradient is None:
            continue
        gradient_count += 1
        if gradient.is_sparse:
            gradient = gradient.coalesce().values()
        finite = torch.isfinite(gradient).all()
        finite_flags.append(finite)
        nonzero_finite_flags.append(finite & torch.any(gradient != 0))

    device = loss_finite.device
    if finite_flags:
        finite_count = torch.stack(finite_flags).sum(dtype=torch.int64)
        nonzero_finite_count = torch.stack(nonzero_finite_flags).sum(dtype=torch.int64)
    else:
        finite_count = torch.zeros((), device=device, dtype=torch.int64)
        nonzero_finite_count = torch.zeros((), device=device, dtype=torch.int64)
    return torch.stack(
        [
            loss_finite.to(device=device, dtype=torch.int64),
            torch.tensor(trainable_count, device=device, dtype=torch.int64),
            torch.tensor(gradient_count, device=device, dtype=torch.int64),
            finite_count.to(device=device),
            nonzero_finite_count.to(device=device),
        ]
    ).unsqueeze(0)


class _GradientGateBackwardObserver:
    """Retain scalar evidence from gradients before ZeRO clears ``param.grad``."""

    def __init__(self, parameters) -> None:
        self._parameters = [parameter for parameter in parameters if parameter.requires_grad]
        self._observations = {}
        self._handles = [
            parameter.register_hook(self._make_hook(index))
            for index, parameter in enumerate(self._parameters)
        ]

    def _make_hook(self, index: int):
        def observe(gradient: torch.Tensor) -> torch.Tensor:
            values = gradient.coalesce().values() if gradient.is_sparse else gradient
            finite = torch.isfinite(values).all().detach()
            finite_nonzero = (finite & torch.any(values != 0)).detach()
            previous = self._observations.get(index)
            if previous is None:
                self._observations[index] = (finite, finite_nonzero)
            else:
                self._observations[index] = (
                    previous[0] & finite,
                    previous[1] | finite_nonzero,
                )
            return gradient

        return observe

    def stats(self, loss_finite: torch.Tensor) -> torch.Tensor:
        device = loss_finite.device
        observations = list(self._observations.values())
        if observations:
            finite_count = torch.stack(
                [finite.to(device=device) for finite, _ in observations]
            ).sum(dtype=torch.int64)
            finite_nonzero_count = torch.stack(
                [finite_nonzero.to(device=device) for _, finite_nonzero in observations]
            ).sum(dtype=torch.int64)
        else:
            finite_count = torch.zeros((), device=device, dtype=torch.int64)
            finite_nonzero_count = torch.zeros((), device=device, dtype=torch.int64)
        return torch.stack(
            [
                loss_finite.to(device=device, dtype=torch.int64),
                torch.tensor(len(self._parameters), device=device, dtype=torch.int64),
                torch.tensor(len(observations), device=device, dtype=torch.int64),
                finite_count,
                finite_nonzero_count,
            ]
        ).unsqueeze(0)

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._observations.clear()


def _json_digest(payload) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _jsonable_state(value):
    if isinstance(value, torch.Tensor):
        tensor = value.detach()
        flat = tensor.reshape(-1)
        samples = []
        if flat.numel():
            for index in sorted({0, flat.numel() // 2, flat.numel() - 1}):
                samples.append(float(flat[index].float().cpu().item()).hex())
        total = float(tensor.sum(dtype=torch.float64).cpu().item()).hex() if tensor.numel() else 0.0.hex()
        return {
            "tensor": True,
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "sum_float64_hex": total,
            "samples_float32_hex": samples,
        }
    if isinstance(value, dict):
        return {
            str(key): _jsonable_state(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable_state(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


class Wan22Trainer:
    def __init__(
        self,
        model,
        train_dataset,
        val_dataset=None,
        *,
        cfg: DictConfig,
        checkpoint_contract: CheckpointContract | None = None,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.cfg = cfg
        if checkpoint_contract is not None and not isinstance(
            checkpoint_contract, CheckpointContract
        ):
            raise TypeError(
                "`checkpoint_contract` must be a CheckpointContract or None, "
                f"got {type(checkpoint_contract).__name__}"
            )
        self.checkpoint_contract = checkpoint_contract
        self.output_dir = str(cfg.output_dir)
        self.learning_rate = float(cfg.learning_rate)
        configured_min_learning_rate = cfg.get("min_learning_rate", None)
        self.min_learning_rate = (
            self.learning_rate * 0.01
            if configured_min_learning_rate is None
            else float(configured_min_learning_rate)
        )
        if not 0.0 <= self.min_learning_rate <= self.learning_rate:
            raise ValueError(
                "min_learning_rate must be between zero and learning_rate, "
                f"got {self.min_learning_rate} for learning_rate={self.learning_rate}"
            )
        self.weight_decay = float(cfg.weight_decay)
        self.adam_betas = (0.9, 0.95)
        self.batch_size = int(cfg.batch_size)
        self.num_workers, self.prefetch_factor, self.persistent_workers = (
            _resolve_dataloader_config(
                num_workers=cfg.num_workers,
                prefetch_factor=cfg.get("prefetch_factor", 2),
                persistent_workers=cfg.get("persistent_workers", False),
            )
        )
        self.num_epochs = int(cfg.num_epochs)
        max_steps = cfg.max_steps
        self.max_steps = int(max_steps) if max_steps is not None else None
        # An extension is a full-state continuation whose only contract drift
        # is the requested terminal step. All model/data/optimizer semantics
        # remain strict, so this cannot silently become a warm-start run.
        self.resume_extension = bool(cfg.get("resume_extension", False))
        self.resume_lr_override = bool(cfg.get("resume_lr_override", False))
        self.resume_lr_override_value = float(cfg.get("resume_lr_override_value", 1e-5))
        self.log_every = int(cfg.log_every)
        self.save_every = int(cfg.save_every)
        self.final_checkpoint_mode = str(cfg.get("final_checkpoint_mode", "full")).strip().lower()
        if self.final_checkpoint_mode not in {"full", "weights_only"}:
            raise ValueError("final_checkpoint_mode must be 'full' or 'weights_only'")
        retention_cfg = cfg.get("checkpoint_retention", {})
        self.checkpoint_retention_max_to_keep = (
            _resolve_checkpoint_retention_max_to_keep(
                retention_cfg.get("max_to_keep", None)
            )
        )
        self.checkpoint_retention_keep_every_n_steps = (
            _resolve_checkpoint_retention_keep_every_n_steps(
                retention_cfg.get("keep_every_n_steps", None)
            )
        )
        self.checkpoint_retention_keep_final = (
            _resolve_checkpoint_retention_keep_final(
                retention_cfg.get("keep_final", False)
            )
        )
        self.permanent_weights_every_n_steps = (
            _resolve_permanent_weights_every_n_steps(
                retention_cfg.get("permanent_weights_every_n_steps", None)
            )
        )
        self.eval_every = int(cfg.eval_every)
        self.eval_num_inference_steps = int(cfg.eval_num_inference_steps)
        self.gradient_accumulation_steps = int(cfg.gradient_accumulation_steps)
        self.max_grad_norm = float(cfg.max_grad_norm)
        self.seed = int(cfg.seed)
        self.clip_events = 0
        self.optimizer_steps_observed = 0
        self.optimizer_skipped = 0
        self._summary_losses: list[dict[str, float]] = []
        self._summary_grad_norms: list[float] = []
        self._summary_samples = 0
        
        self.resume = cfg.resume
        configured_legacy_warmstart = cfg.get("legacy_warmstart", None)
        configured_flex_warmstart = cfg.get("flex_warmstart", None)
        self.legacy_warmstart, self.flex_warmstart = (
            _resolve_initial_checkpoint_sources(
                resume=self.resume,
                legacy_warmstart=configured_legacy_warmstart,
                flex_warmstart=configured_flex_warmstart,
            )
        )
        self.mixed_precision = str(cfg.mixed_precision).strip().lower()
        if self.mixed_precision not in {"no", "fp16", "bf16"}:
            raise ValueError(
                f"Unsupported mixed_precision: {cfg.mixed_precision}. "
                "Expected one of: ['no', 'fp16', 'bf16']."
            )
        self.wandb_enabled = bool(cfg.wandb.enabled)

        profiling_cfg = cfg.get("profiling", {})
        self.profiling_enabled = bool(profiling_cfg.get("enabled", False))
        self.profiling_warmup_steps = int(profiling_cfg.get("warmup_steps", 50))
        self.profiling_measure_steps = int(profiling_cfg.get("measure_steps", 100))
        self.profiling_stop_after_profile = bool(
            profiling_cfg.get("stop_after_profile", True)
        )
        report_path = profiling_cfg.get("report_path", None)
        if report_path in (None, "", "null"):
            report_path = os.path.join(self.output_dir, "profile-report.json")
        elif not os.path.isabs(str(report_path)):
            report_path = os.path.join(self.output_dir, str(report_path))
        self.profiling_report_path = str(report_path)
        if self.profiling_warmup_steps < 0:
            raise ValueError("`profiling.warmup_steps` must be non-negative.")
        if self.profiling_measure_steps <= 0:
            raise ValueError("`profiling.measure_steps` must be positive.")

        resume_validation_cfg = cfg.get("resume_validation", {})
        self.resume_validation_enabled = bool(
            resume_validation_cfg.get("enabled", False)
        )
        self.resume_validation_continuation_steps = int(
            resume_validation_cfg.get("continuation_steps", 5)
        )
        resume_report_path = resume_validation_cfg.get("report_path", None)
        if resume_report_path in (None, "", "null"):
            resume_report_path = os.path.join(
                self.output_dir, "resume-validation-report.json"
            )
        elif not os.path.isabs(str(resume_report_path)):
            resume_report_path = os.path.join(
                self.output_dir, str(resume_report_path)
            )
        self.resume_validation_report_path = str(resume_report_path)
        if self.resume_validation_continuation_steps <= 0:
            raise ValueError("`resume_validation.continuation_steps` must be positive.")
        if self.resume_validation_enabled and not self.resume:
            raise ValueError("Resume validation requires a full-state `resume` directory.")
        self._resume_validation_start_probe = None
        self._resume_validation_target_step = None

        fixed_cfg = cfg.get("fixed_validation", {})
        self.fixed_validation_enabled = bool(fixed_cfg.get("enabled", False))
        self.fixed_validation_steps = tuple(int(x) for x in fixed_cfg.get("steps", [0, 250, 500, 750, 1000]))
        self.fixed_validation_every_steps = int(fixed_cfg.get("every_steps", 0))
        self.fixed_validation_tasks = int(fixed_cfg.get("tasks", 100))
        self.fixed_validation_anchors_per_task = int(fixed_cfg.get("anchors_per_task", 10))
        self.fixed_validation_stride = int(fixed_cfg.get("anchor_stride", 6))
        self.fixed_validation_seed = int(fixed_cfg.get("seed", 42))
        self.fixed_validation_max_duration_seconds = float(
            fixed_cfg.get("max_duration_seconds", 900)
        )
        fixed_report_dir = fixed_cfg.get("report_dir", None)
        if fixed_report_dir in (None, "", "null"):
            fixed_report_dir = os.path.join(self.output_dir, "fixed-validation")
        elif not os.path.isabs(str(fixed_report_dir)):
            fixed_report_dir = os.path.join(self.output_dir, str(fixed_report_dir))
        self.fixed_validation_report_dir = str(fixed_report_dir)
        if self.fixed_validation_enabled and any(step < 0 for step in self.fixed_validation_steps):
            raise ValueError("fixed_validation.steps must be non-negative")
        if self.fixed_validation_every_steps < 0:
            raise ValueError("fixed_validation.every_steps must be non-negative")
        if self.fixed_validation_max_duration_seconds <= 0:
            raise ValueError("fixed_validation.max_duration_seconds must be positive")

        qualitative_cfg = cfg.get("qualitative_validation", {})
        self.qualitative_validation_enabled = bool(
            qualitative_cfg.get("enabled", False)
        )
        self.qualitative_validation_steps = tuple(
            int(x) for x in qualitative_cfg.get("steps", [])
        )
        self.qualitative_validation_every_steps = int(
            qualitative_cfg.get("every_steps", 0)
        )
        self.qualitative_validation_num_windows = int(
            qualitative_cfg.get("num_windows", 128)
        )
        self.qualitative_validation_expected_tasks = int(
            qualitative_cfg.get("expected_tasks", 100)
        )
        self.qualitative_validation_videos = int(qualitative_cfg.get("videos", 8))
        self.qualitative_validation_inference_steps = int(
            qualitative_cfg.get("inference_steps", 10)
        )
        self.qualitative_validation_seed = int(qualitative_cfg.get("seed", 42))
        self.qualitative_validation_fps = int(qualitative_cfg.get("fps", 8))
        self.qualitative_validation_max_duration_seconds = float(
            qualitative_cfg.get("max_duration_seconds", 900)
        )
        qualitative_report_dir = qualitative_cfg.get("report_dir", None)
        if qualitative_report_dir in (None, "", "null"):
            qualitative_report_dir = os.path.join(
                self.output_dir, "qualitative-validation"
            )
        elif not os.path.isabs(str(qualitative_report_dir)):
            qualitative_report_dir = os.path.join(
                self.output_dir, str(qualitative_report_dir)
            )
        self.qualitative_validation_report_dir = str(qualitative_report_dir)
        if any(step < 0 for step in self.qualitative_validation_steps):
            raise ValueError("qualitative_validation.steps must be non-negative")
        if self.qualitative_validation_every_steps < 0:
            raise ValueError(
                "qualitative_validation.every_steps must be non-negative"
            )
        if self.qualitative_validation_max_duration_seconds <= 0:
            raise ValueError(
                "qualitative_validation.max_duration_seconds must be positive"
            )

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision=self.mixed_precision,
            step_scheduler_with_optimizer=False,
        )
        self.initialization_fingerprint_sha256 = getattr(
            self.model, INITIALIZATION_FINGERPRINT_ATTRIBUTE, None
        )
        self._assert_initialization_fingerprint_consistent()
        self._fixed_validation_window_set_sha256 = None
        
        logger.info(
            "Accelerate training: distributed_type=%s zero_stage=%s world_size=%d process_index=%d cfg_mixed_precision=%s accelerator_mixed_precision=%s grad_accum=%d grad_clip=%.4f",
            self.accelerator.distributed_type,
            self.accelerator.state.deepspeed_plugin.deepspeed_config.get("zero_optimization", {}).get("stage", "unknown"),
            self.accelerator.num_processes,
            self.accelerator.process_index,
            self.mixed_precision,
            self.accelerator.mixed_precision,
            self.gradient_accumulation_steps,
            self.max_grad_norm,
        )
        logger.info("using accelerator.device=%s", self.accelerator.device)
        logger.info(
            "DataLoader: batch_size=%d num_workers=%d prefetch_factor=%s "
            "persistent_workers=%s pin_memory=%s",
            self.batch_size,
            self.num_workers,
            self.prefetch_factor,
            self.persistent_workers,
            torch.cuda.is_available(),
        )
        worker_init_fn = set_global_seed(self.seed, get_worker_init_fn=True)
        self._assert_dataset_length_consistent(self.train_dataset, "train_dataset")
        if self.val_dataset is not None:
            self._assert_dataset_length_consistent(self.val_dataset, "val_dataset")

        # Freeze non-trainable modules before optimizer/deepspeed initialization.
        # This keeps DiT (+ optional proprio encoder) as trainable when ZeRO builds optimizer state.
        self._apply_dit_only_train_mode(self.model)
        trainable_params = self._trainable_parameters(self.model)
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=self.adam_betas,
        )
        
        self.train_loader = self._build_loader(self.train_dataset, worker_init_fn=worker_init_fn)
        total_train_steps = self._estimate_total_train_steps()
        self.max_steps = total_train_steps
        warmup_steps = _resolve_lr_warmup_steps(
            total_train_steps, cfg.get("lr_warmup_steps", None)
        )
        self.lr_scheduler_type = str(cfg.lr_scheduler_type).strip().lower()
        self.lr_warmup_steps = warmup_steps
        self.scheduler = self._build_scheduler(
            scheduler_type=self.lr_scheduler_type,
            total_train_steps=total_train_steps,
            warmup_steps=warmup_steps,
        )
        from .datasets.b1k.checkpoint import bind_optimizer_scheduler_contract

        self.checkpoint_contract = bind_optimizer_scheduler_contract(
            self.checkpoint_contract,
            learning_rate=self.optimizer.param_groups[0]["lr"],
            min_learning_rate=self.min_learning_rate,
            lr_scheduler_type=self.lr_scheduler_type,
            lr_warmup_steps=self.lr_warmup_steps,
            max_steps=self.max_steps,
            weight_decay=self.optimizer.param_groups[0]["weight_decay"],
            betas=self.optimizer.param_groups[0]["betas"],
            epsilon=self.optimizer.param_groups[0]["eps"],
            amsgrad=self.optimizer.param_groups[0]["amsgrad"],
            max_grad_norm=self.max_grad_norm,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
        )
        self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0

        self.checkpoint_root = os.path.join(self.output_dir, "checkpoints")
        self.weights_dir = os.path.join(self.checkpoint_root, "weights")
        self.permanent_weights_dir = os.path.join(
            self.checkpoint_root, "permanent_weights"
        )
        self.state_dir = os.path.join(self.checkpoint_root, "state")
        self.eval_dir = os.path.join(self.output_dir, "eval")

        ensure_dir(self.output_dir)
        ensure_dir(self.checkpoint_root)
        ensure_dir(self.weights_dir)
        ensure_dir(self.permanent_weights_dir)
        ensure_dir(self.state_dir)
        ensure_dir(self.eval_dir)

        self.model, self.optimizer, self.train_loader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.scheduler
        )
        self.optimizer.zero_grad(set_to_none=True)
        self.wandb_run = None
        self._init_wandb()
        self._resume_or_load_checkpoint()
        self.tensorboard_logger = create_tensorboard_logger(
            output_dir=self.output_dir,
            config=cfg.get("tensorboard", None),
            is_main_process=self.accelerator.is_main_process,
            # A weights-only load starts a new optimizer timeline at step 0;
            # only a full-state directory has completed events to purge.
            resumed_completed_step=(
                self.global_step
                if self.resume and Path(str(self.resume)).is_dir()
                else None
            ),
        )
        self._init_profiling_state()
        self._gradient_gate_report = None
        self._gradient_gate_loss_finite = torch.ones(
            (), device=self.accelerator.device, dtype=torch.bool
        )
        self._gradient_gate_loss_observations = 0
        self._gradient_gate_backward_observer = _GradientGateBackwardObserver(
            self.model.parameters()
        )

        val_size = len(self.val_dataset) if self.val_dataset is not None else len(self.train_dataset)
        logger.info("Train/val dataset size: %d/%d", len(self.train_dataset), val_size)

    def _assert_initialization_fingerprint_consistent(self) -> None:
        fingerprint = self.initialization_fingerprint_sha256
        if not isinstance(fingerprint, str) or not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
            raise RuntimeError(
                "Model is missing a valid random-module initialization fingerprint. "
                "Instantiate it through fastwam.runtime.run_training()."
            )
        local = torch.tensor(
            list(bytes.fromhex(fingerprint)),
            dtype=torch.uint8,
            device=self.accelerator.device,
        )
        gathered = self.accelerator.gather(local).detach().cpu().reshape(-1, 32)
        fingerprints = [bytes(row.tolist()).hex() for row in gathered]
        unique = sorted(set(fingerprints))
        if len(unique) != 1:
            raise RuntimeError(
                "Random-module initialization differs across distributed ranks: "
                f"fingerprints={unique}"
            )
        logger.info(
            "Random-module initialization is identical across %d ranks: sha256=%s",
            len(fingerprints),
            fingerprint,
        )

    def _init_profiling_state(self):
        self._profiling_start_global_step = int(self.global_step)
        self._profiling_completed_steps = 0
        self._profiling_step_data_seconds = 0.0
        self._profiling_step_forward_seconds = 0.0
        self._profiling_step_backward_seconds = 0.0
        self._profiling_step_optimizer_seconds = 0.0
        self._profiling_step_wall_started_at = None
        self._profiling_records = []
        self._profiling_optimizer_skipped_steps = 0
        self._profiling_memory_reset = False
        self._profiling_report = None
        self._profiling_checkpoint = None
        if not self.profiling_enabled:
            return
        required_steps = self.profiling_warmup_steps + self.profiling_measure_steps
        remaining_steps = int(self.max_steps) - int(self.global_step)
        if remaining_steps < required_steps:
            raise ValueError(
                "Profiling does not fit before max_steps: "
                f"remaining_steps={remaining_steps}, required_steps={required_steps}."
            )
        logger.info(
            "Profiling enabled: warmup_steps=%d measure_steps=%d stop_after_profile=%s report=%s",
            self.profiling_warmup_steps,
            self.profiling_measure_steps,
            self.profiling_stop_after_profile,
            self.profiling_report_path,
        )

    def _cuda_synchronize(self):
        if self.profiling_enabled and torch.cuda.is_available():
            torch.cuda.synchronize(self.accelerator.device)

    def _profiling_prepare_micro_step(self):
        if not self.profiling_enabled:
            return
        if (
            self._profiling_completed_steps >= self.profiling_warmup_steps
            and not self._profiling_memory_reset
        ):
            self._cuda_synchronize()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(self.accelerator.device)
            self._profiling_memory_reset = True
        self._cuda_synchronize()
        if (
            self._profiling_measurement_active()
            and self._profiling_step_wall_started_at is None
        ):
            self._profiling_step_wall_started_at = time.perf_counter()

    def _gradient_gate_observe_loss(self, loss: torch.Tensor) -> None:
        if self._gradient_gate_report is not None:
            return
        self._gradient_gate_loss_finite &= torch.isfinite(loss.detach()).all()
        self._gradient_gate_loss_observations += 1

    def _validate_gradient_gate(self) -> None:
        """Fail before the first optimizer step if real loss/gradients are invalid."""
        if self._gradient_gate_report is not None:
            return
        observer = getattr(self, "_gradient_gate_backward_observer", None)
        if observer is None:
            local = _local_gradient_gate_stats(
                self.model.parameters(), self._gradient_gate_loss_finite
            )
            evidence_source = "post_backward_parameter_storage"
        else:
            local = observer.stats(self._gradient_gate_loss_finite)
            observer.close()
            self._gradient_gate_backward_observer = None
            evidence_source = "autograd_parameter_hooks_after_backward"
        gathered = self.accelerator.gather(local).detach().cpu().reshape(-1, 5)
        world_size = int(self.accelerator.num_processes)
        if gathered.shape[0] != world_size:
            raise RuntimeError(
                "Gradient gate gathered an unexpected number of rank records: "
                f"expected={world_size}, actual={gathered.shape[0]}."
            )

        rank_records = []
        for rank, row in enumerate(gathered.tolist()):
            rank_records.append(
                {
                    "rank": rank,
                    "loss_finite": bool(row[0]),
                    "trainable_parameter_tensors": int(row[1]),
                    "gradient_parameter_tensors": int(row[2]),
                    "finite_gradient_parameter_tensors": int(row[3]),
                    "finite_nonzero_gradient_parameter_tensors": int(row[4]),
                }
            )
        loss_finite = all(record["loss_finite"] for record in rank_records)
        nonfinite_gradient_count = sum(
            record["gradient_parameter_tensors"]
            - record["finite_gradient_parameter_tensors"]
            for record in rank_records
        )
        finite_nonzero_gradient_count = sum(
            record["finite_nonzero_gradient_parameter_tensors"]
            for record in rank_records
        )
        finite_nonzero_gradient_on_every_rank = all(
            record["finite_nonzero_gradient_parameter_tensors"] > 0
            for record in rank_records
        )
        passed = (
            loss_finite
            and nonfinite_gradient_count == 0
            and finite_nonzero_gradient_count > 0
            and finite_nonzero_gradient_on_every_rank
        )
        self._gradient_gate_report = {
            "schema": "fastwam_gradient_gate.v1",
            "status": "passed" if passed else "failed",
            "checked_before_global_step": int(self.global_step + 1),
            "world_size": world_size,
            "gradient_evidence_source": evidence_source,
            "gradient_accumulation_micro_steps_observed": int(
                self._gradient_gate_loss_observations
            ),
            "checks": {
                "loss_finite_on_all_ranks": loss_finite,
                "all_observed_gradients_finite": nonfinite_gradient_count == 0,
                "at_least_one_finite_nonzero_trainable_gradient": (
                    finite_nonzero_gradient_count > 0
                ),
                "finite_nonzero_trainable_gradient_on_every_rank": (
                    finite_nonzero_gradient_on_every_rank
                ),
            },
            "rank_evidence": rank_records,
        }
        if not passed:
            raise FloatingPointError(
                "First real training batch failed the loss/gradient gate: "
                f"{json.dumps(self._gradient_gate_report, sort_keys=True)}"
            )
        if self.accelerator.is_main_process:
            logger.info(
                "First-batch gradient gate passed before step=%d: finite_nonzero_gradient_tensors=%d",
                self.global_step + 1,
                finite_nonzero_gradient_count,
            )

    def _profiling_measurement_active(self) -> bool:
        return (
            self.profiling_enabled
            and self.profiling_warmup_steps
            <= self._profiling_completed_steps
            < self.profiling_warmup_steps + self.profiling_measure_steps
        )

    def _profiling_record_data_wait(self, started_at: float):
        if not self.profiling_enabled:
            return
        self._cuda_synchronize()
        if self._profiling_measurement_active():
            self._profiling_step_data_seconds += time.perf_counter() - started_at

    def _profiling_record_forward(self, started_at: float):
        if not self.profiling_enabled:
            return
        self._cuda_synchronize()
        if self._profiling_measurement_active():
            self._profiling_step_forward_seconds += time.perf_counter() - started_at

    def _profiling_record_backward(self, started_at: float):
        if not self.profiling_enabled:
            return
        self._cuda_synchronize()
        if self._profiling_measurement_active():
            self._profiling_step_backward_seconds += time.perf_counter() - started_at

    def _profiling_record_optimizer(self, started_at: float):
        if not self.profiling_enabled:
            return None
        self._cuda_synchronize()
        ended_at = time.perf_counter()
        if self._profiling_measurement_active():
            self._profiling_step_optimizer_seconds += ended_at - started_at
        return ended_at

    def _profiling_finish_optimizer_step(
        self,
        post_optimizer_started_at: float | None,
        *,
        optimizer_skipped: bool = False,
    ) -> bool:
        if not self.profiling_enabled:
            return False
        self._cuda_synchronize()
        ended_at = time.perf_counter()
        if self._profiling_measurement_active():
            if self._profiling_step_wall_started_at is None:
                raise RuntimeError(
                    "Profiling step wall timer was not started before optimizer completion."
                )
            metrics_logging_seconds = (
                ended_at - post_optimizer_started_at
                if post_optimizer_started_at is not None
                else 0.0
            )
            phase_sum_seconds = (
                self._profiling_step_data_seconds
                + self._profiling_step_forward_seconds
                + self._profiling_step_backward_seconds
                + self._profiling_step_optimizer_seconds
                + metrics_logging_seconds
            )
            step_wall_seconds = ended_at - self._profiling_step_wall_started_at
            self._profiling_records.append(
                [
                    self._profiling_step_data_seconds,
                    self._profiling_step_forward_seconds,
                    self._profiling_step_backward_seconds,
                    self._profiling_step_optimizer_seconds,
                    metrics_logging_seconds,
                    phase_sum_seconds,
                    step_wall_seconds,
                ]
            )
        self._profiling_step_data_seconds = 0.0
        self._profiling_step_forward_seconds = 0.0
        self._profiling_step_backward_seconds = 0.0
        self._profiling_step_optimizer_seconds = 0.0
        self._profiling_step_wall_started_at = None
        if optimizer_skipped:
            self._profiling_optimizer_skipped_steps += 1
        self._profiling_completed_steps += 1
        if len(self._profiling_records) < self.profiling_measure_steps:
            return False
        self._finalize_profiling_report()
        return self.profiling_stop_after_profile

    def _finalize_profiling_report(self):
        if self._profiling_report is not None:
            return
        local_timings = torch.tensor(
            self._profiling_records,
            device=self.accelerator.device,
            dtype=torch.float64,
        )
        gathered_timings = self.accelerator.gather(local_timings).detach().cpu().numpy()
        world_size = int(self.accelerator.num_processes)
        gathered_timings = gathered_timings.reshape(
            world_size, self.profiling_measure_steps, 7
        )

        if torch.cuda.is_available():
            local_memory = torch.tensor(
                [
                    torch.cuda.max_memory_allocated(self.accelerator.device),
                    torch.cuda.max_memory_reserved(self.accelerator.device),
                ],
                device=self.accelerator.device,
                dtype=torch.int64,
            )
        else:
            local_memory = torch.zeros(2, device=self.accelerator.device, dtype=torch.int64)
        gathered_memory = (
            self.accelerator.gather(local_memory).detach().cpu().reshape(world_size, 2)
        )

        phases = {
            "data_wait": _distribution_summary(gathered_timings[:, :, 0].reshape(-1)),
            "forward": _distribution_summary(gathered_timings[:, :, 1].reshape(-1)),
            "backward": _distribution_summary(gathered_timings[:, :, 2].reshape(-1)),
            "forward_backward": _distribution_summary(
                (gathered_timings[:, :, 1] + gathered_timings[:, :, 2]).reshape(-1)
            ),
            "optimizer": _distribution_summary(gathered_timings[:, :, 3].reshape(-1)),
            "metrics_logging": _distribution_summary(
                gathered_timings[:, :, 4].reshape(-1)
            ),
            # `total_step` keeps the v1 meaning: sum of instrumented phases.
            "total_step": _distribution_summary(gathered_timings[:, :, 5].reshape(-1)),
            "step_wall": _distribution_summary(gathered_timings[:, :, 6].reshape(-1)),
        }
        # Throughput must use end-to-end wall time, not the optimistic sum of
        # instrumented phases. This includes metric collectives, logging, Python
        # overhead, and inter-rank stalls across every accumulation micro-step.
        distributed_step_seconds = gathered_timings[:, :, 5].max(axis=0)
        distributed_step_wall_seconds = gathered_timings[:, :, 6].max(axis=0)
        effective_batch = (
            self.batch_size * world_size * self.gradient_accumulation_steps
        )
        elapsed = float(distributed_step_wall_seconds.sum())
        throughput = (
            effective_batch * self.profiling_measure_steps / elapsed
            if elapsed > 0
            else 0.0
        )
        mean_distributed_step_seconds = float(distributed_step_wall_seconds.mean())
        estimated_configured_run_wall_seconds = mean_distributed_step_seconds * int(
            self.max_steps
        )
        estimated_configured_remaining_wall_seconds = mean_distributed_step_seconds * max(
            0, int(self.max_steps) - int(self.global_step)
        )
        estimated_wall_seconds_per_million_samples = (
            1_000_000.0 / throughput if throughput > 0 else 0.0
        )
        # Persist the resolved runtime contract alongside timings.  Profile
        # launchers must validate these values from the report rather than
        # trusting Hydra overrides or inherited launcher environment.
        accelerator_state = getattr(self.accelerator, "state", None)
        deepspeed_plugin = getattr(accelerator_state, "deepspeed_plugin", None)
        deepspeed_config = getattr(deepspeed_plugin, "deepspeed_config", {}) or {}
        zero_optimization = deepspeed_config.get("zero_optimization", {})
        zero_stage = zero_optimization.get("stage")
        config = getattr(self, "cfg", {}) or {}
        model_cfg = config.get("model", {}) if hasattr(config, "get") else {}
        gradient_checkpointing = bool(
            model_cfg.get("mot_checkpoint_mixed_attn", False)
        )
        train_stride = getattr(
            getattr(self, "train_dataset", None), "window_start_stride", None
        )
        configured_lr = getattr(self, "learning_rate", None)
        checkpoint_contract = getattr(self, "checkpoint_contract", None)
        distributed_type = getattr(
            self.accelerator, "distributed_type", "UNKNOWN"
        )
        runtime_contract = {
            "learning_rate": float(configured_lr) if configured_lr is not None else None,
            "gradient_checkpointing": gradient_checkpointing,
            "window_start_stride": (
                int(train_stride) if train_stride is not None else None
            ),
            "zero_stage": int(zero_stage) if zero_stage is not None else None,
            # Accelerate exposes DistributedType as an enum. Persist its stable
            # value instead of the implementation-specific enum repr.
            "distributed_type": str(
                getattr(distributed_type, "value", distributed_type)
            ),
            "checkpoint_contract_sha256": (
                checkpoint_contract.contract_sha256
                if checkpoint_contract is not None
                else None
            ),
        }
        report = {
            "schema_version": 2,
            "world_size": world_size,
            "per_gpu_micro_batch": int(self.batch_size),
            "dataloader": {
                "num_workers_per_rank": int(getattr(self, "num_workers", 0)),
                "prefetch_factor": getattr(self, "prefetch_factor", None),
                "persistent_workers": bool(
                    getattr(self, "persistent_workers", False)
                ),
                "pin_memory": bool(torch.cuda.is_available()),
            },
            "gradient_accumulation_steps": int(self.gradient_accumulation_steps),
            "effective_global_batch": int(effective_batch),
            "start_global_step": int(self._profiling_start_global_step),
            "warmup_steps": int(self.profiling_warmup_steps),
            "measure_steps": int(self.profiling_measure_steps),
            "measured_global_step_start": int(
                self._profiling_start_global_step + self.profiling_warmup_steps + 1
            ),
            "measured_global_step_end": int(self.global_step),
            "timing_seconds": phases,
            "distributed_step_seconds": _distribution_summary(distributed_step_seconds),
            "distributed_step_wall_seconds": _distribution_summary(
                distributed_step_wall_seconds
            ),
            "effective_samples_per_second": float(throughput),
            "throughput_basis": "distributed_end_to_end_step_wall",
            "configured_max_steps": int(self.max_steps),
            "estimated_configured_run_wall_seconds": estimated_configured_run_wall_seconds,
            "estimated_configured_run_wall_hours": estimated_configured_run_wall_seconds
            / 3600.0,
            "estimated_configured_remaining_wall_seconds": estimated_configured_remaining_wall_seconds,
            "estimated_configured_remaining_wall_hours": estimated_configured_remaining_wall_seconds
            / 3600.0,
            "estimated_wall_seconds_per_million_samples": estimated_wall_seconds_per_million_samples,
            # This is deliberately a train-only wall-clock estimate.  It does
            # not include validation/inference, checkpoint I/O, preemption,
            # queueing, or restart overhead.
            "estimated_train_only_run_wall_seconds": estimated_configured_run_wall_seconds,
            "estimated_train_only_run_wall_hours": estimated_configured_run_wall_seconds
            / 3600.0,
            # Keep the historical field stable for existing report readers;
            # the explicit train-only list is used by new profile summaries.
            "wall_time_estimate_excludes": ["evaluation", "checkpoint_save"],
            "train_only_estimate_excludes": [
                "evaluation",
                "checkpoint_save",
                "queueing",
                "preemption_restart",
            ],
            "resolved_training_contract": runtime_contract,
            "optimizer_skipped_steps": int(self._profiling_optimizer_skipped_steps),
            "gradient_gate": getattr(self, "_gradient_gate_report", None),
            "rank_peak_memory": [
                {
                    "rank": rank,
                    "max_allocated_bytes": int(gathered_memory[rank, 0]),
                    "max_reserved_bytes": int(gathered_memory[rank, 1]),
                }
                for rank in range(world_size)
            ],
            "checkpoint": self._profiling_checkpoint
            or {
                "measured": False,
                "reason": "no checkpoint completed during the profiling window",
            },
        }
        self._profiling_report = report
        if self.accelerator.is_main_process:
            _atomic_write_json(self.profiling_report_path, report)
            logger.info("Wrote profiling report to %s", self.profiling_report_path)
        self.accelerator.wait_for_everyone()

    def _record_checkpoint_profile(self, elapsed_seconds: float, ckpt_info: dict):
        if not self.profiling_enabled:
            return
        local_elapsed = torch.tensor(
            [elapsed_seconds], device=self.accelerator.device, dtype=torch.float64
        )
        gathered_elapsed = self.accelerator.gather(local_elapsed).detach().cpu().numpy()
        checkpoint_profile = {
            "measured": True,
            "elapsed_seconds": _distribution_summary(gathered_elapsed),
            "weights_size_bytes": _path_size_bytes(ckpt_info.get("weights_path"))
            if self.accelerator.is_main_process
            else 0,
            "state_size_bytes": _path_size_bytes(ckpt_info.get("state_path"))
            if self.accelerator.is_main_process
            else 0,
            "global_step": int(self.global_step),
        }
        self._profiling_checkpoint = checkpoint_profile
        if self._profiling_report is not None:
            self._profiling_report["checkpoint"] = checkpoint_profile
            if self.accelerator.is_main_process:
                _atomic_write_json(self.profiling_report_path, self._profiling_report)
        self.accelerator.wait_for_everyone()

    def _init_wandb(self):
        if not self.wandb_enabled or not self.accelerator.is_main_process:
            return
        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                "wandb logging is enabled in config (`wandb.enabled=true`) but wandb is not installed."
            ) from e

        self.wandb_run = wandb.init(
            entity=self.cfg.wandb.workspace,
            project=self.cfg.wandb.project,
            name=self.cfg.wandb.name,
            group=None if self.cfg.wandb.group in (None, "null", "") else str(self.cfg.wandb.group),
            mode=self.cfg.wandb.mode,
            dir=self.output_dir,
        )
        logger.info(
            "Initialized wandb run: workspace=%s project=%s name=%s",
            self.cfg.wandb.workspace,
            self.cfg.wandb.project,
            self.cfg.wandb.name,
        )

    def _wandb_log(self, payload: dict):
        if getattr(self, "wandb_run", None) is None:
            return
        self.wandb_run.log(payload, step=self.global_step)

    def _finish_wandb(self):
        if getattr(self, "wandb_run", None) is None:
            return
        self.wandb_run.finish()
        self.wandb_run = None

    def _log_tensorboard_scalars(
        self,
        payload: dict,
        *,
        step: int | None = None,
        flush: bool = False,
    ) -> None:
        """Write rank-0 scalar metrics without making W&B a dependency."""
        if not self.accelerator.is_main_process:
            return
        tensorboard_logger = getattr(self, "tensorboard_logger", None)
        if tensorboard_logger is None:
            return
        tensorboard_logger.log_scalars(
            payload,
            step=self.global_step if step is None else int(step),
        )
        if flush:
            tensorboard_logger.flush()

    def _log_fixed_validation_metrics(
        self,
        metrics: dict[str, float | int | str],
        *,
        duration_seconds: float,
    ) -> None:
        if not self.accelerator.is_main_process:
            return
        payload = {
            "validation/loss_total": float(metrics["loss_total"]),
            "timing/fixed_validation_s": float(duration_seconds),
        }
        for key, value in metrics.items():
            if key.startswith("loss_") and key != "loss_total":
                payload[f"validation/{key}"] = float(value)
        timings = metrics.get("timings")
        if isinstance(timings, dict):
            payload.update(
                {
                    "timing/fixed_validation_data_load_max_s": float(
                        timings["data_load_s"]["max_seconds"]
                    ),
                    "timing/fixed_validation_forward_max_s": float(
                        timings["forward_s"]["max_seconds"]
                    ),
                    "timing/fixed_validation_rank_total_max_s": float(
                        timings["total_s"]["max_seconds"]
                    ),
                }
            )
        self._log_tensorboard_scalars(
            payload,
            step=int(metrics["step"]),
            flush=True,
        )
        self._wandb_log(payload)

    def _log_qualitative_validation_metrics(self, report: dict) -> None:
        metrics = report["metrics"]
        timing = report["timings"]["outer_s"]
        payload = {
            "qualitative/future_psnr_vs_gt": float(
                metrics["rollout_future_psnr_vs_gt"]["mean"]
            ),
            "qualitative/future_ssim_vs_gt": float(
                metrics["rollout_future_ssim_vs_gt"]["mean"]
            ),
            "qualitative/future_psnr_vs_vae": float(
                metrics["rollout_future_psnr_vs_vae"]["mean"]
            ),
            "qualitative/future_ssim_vs_vae": float(
                metrics["rollout_future_ssim_vs_vae"]["mean"]
            ),
            "qualitative/vae_psnr_vs_gt": float(
                metrics["vae_psnr_vs_gt"]["mean"]
            ),
            "qualitative/vae_ssim_vs_gt": float(
                metrics["vae_ssim_vs_gt"]["mean"]
            ),
            "qualitative/action_l1_denormalized": float(
                metrics["action_l1_denormalized"]["mean"]
            ),
            "qualitative/action_mse_denormalized": float(
                metrics["action_mse_denormalized"]["mean"]
            ),
            "timing/qualitative_validation_s": float(timing["max_seconds"]),
        }
        self._log_tensorboard_scalars(
            payload, step=int(report["step"]), flush=True
        )
        if self.accelerator.is_main_process:
            self._wandb_log(payload)

    def _run_qualitative_validation_if_due(self) -> dict | None:
        if not self.qualitative_validation_enabled or not qualitative_validation_due(
            self.global_step,
            explicit_steps=self.qualitative_validation_steps,
            every_steps=self.qualitative_validation_every_steps,
        ):
            return None

        from .qualitative_validation import (
            QualitativeValidationConfig,
            run_trainer_qualitative_validation,
        )

        config = QualitativeValidationConfig(
            num_windows=self.qualitative_validation_num_windows,
            expected_tasks=self.qualitative_validation_expected_tasks,
            videos_to_save=self.qualitative_validation_videos,
            num_inference_steps=self.qualitative_validation_inference_steps,
            base_seed=self.qualitative_validation_seed,
            fps=self.qualitative_validation_fps,
        )
        report = run_trainer_qualitative_validation(
            self,
            output_dir=self.qualitative_validation_report_dir,
            step=self.global_step,
            config=config,
            fixed_anchors_per_task=self.fixed_validation_anchors_per_task,
            fixed_anchor_stride=self.fixed_validation_stride,
        )
        if not isinstance(report, dict):
            raise RuntimeError(
                "qualitative validation returned a non-object report: "
                f"{type(report).__name__}"
            )
        expected = {
            "schema": "b1k_qualitative_validation.v1",
            "status": "passed",
            "step": int(self.global_step),
            "world_size": int(self.accelerator.num_processes),
            "num_windows": self.qualitative_validation_num_windows,
            "num_tasks": self.qualitative_validation_expected_tasks,
        }
        for key, value in expected.items():
            if report.get(key) != value:
                raise RuntimeError(
                    "qualitative validation report contract failed for "
                    f"{key}: expected={value!r}, actual={report.get(key)!r}"
                )
        windows = report.get("windows")
        videos = report.get("videos")
        if not isinstance(windows, list) or len(windows) != self.qualitative_validation_num_windows:
            raise RuntimeError(
                "qualitative validation window inventory is incomplete: "
                f"expected={self.qualitative_validation_num_windows}, "
                f"actual={len(windows) if isinstance(windows, list) else None}"
            )
        if not isinstance(videos, list) or len(videos) != self.qualitative_validation_videos:
            raise RuntimeError(
                "qualitative validation video inventory is incomplete: "
                f"expected={self.qualitative_validation_videos}, "
                f"actual={len(videos) if isinstance(videos, list) else None}"
            )
        metrics = report.get("metrics")
        required_metrics = {
            "rollout_future_psnr_vs_gt",
            "rollout_future_ssim_vs_gt",
            "rollout_future_psnr_vs_vae",
            "rollout_future_ssim_vs_vae",
            "vae_psnr_vs_gt",
            "vae_ssim_vs_gt",
            "action_l1_denormalized",
            "action_mse_denormalized",
        }
        if not isinstance(metrics, dict) or set(metrics) != required_metrics:
            raise RuntimeError(
                "qualitative validation metric inventory mismatch: "
                f"expected={sorted(required_metrics)}, "
                f"actual={sorted(metrics) if isinstance(metrics, dict) else None}"
            )
        for name in sorted(required_metrics):
            metric = metrics[name]
            if (
                not isinstance(metric, dict)
                or int(metric.get("count", -1)) != self.qualitative_validation_num_windows
                or not np.isfinite(float(metric.get("mean", float("nan"))))
                or not np.isfinite(float(metric.get("sum", float("nan"))))
            ):
                raise RuntimeError(
                    f"qualitative validation metric contract failed: {name}={metric!r}"
                )
        total_timing = report.get("timings", {}).get(
            "outer_s", report.get("timings", {}).get("total_s", {})
        )
        maximum = float(total_timing.get("max_seconds", float("nan")))
        if not np.isfinite(maximum) or maximum > self.qualitative_validation_max_duration_seconds:
            raise RuntimeError(
                "qualitative validation exceeded its duration gate: "
                f"max_seconds={maximum}, limit={self.qualitative_validation_max_duration_seconds}"
            )
        self._log_qualitative_validation_metrics(report)
        if self.accelerator.is_main_process:
            logger.info(
                "[qualitative-eval] step=%d windows=%d tasks=%d max_seconds=%.3f report=%s",
                self.global_step,
                report["num_windows"],
                report["num_tasks"],
                maximum,
                Path(self.qualitative_validation_report_dir)
                / f"step-{self.global_step:06d}"
                / "report.json",
            )
        return report

    def _log_training_metrics(
        self,
        *,
        loss_total: float,
        loss_metrics: dict[str, float],
        learning_rate: float,
        grad_norm: float,
        clip_event_rate: float,
        steps_per_second: float,
        samples_per_second: float,
        optimizer_skipped: int,
    ) -> None:
        payload = {
            "train/loss_total": float(loss_total),
            "train/lr": float(learning_rate),
            "train/grad_norm": float(grad_norm),
            "train/clip_event_rate": float(clip_event_rate),
            "train/steps_per_sec": float(steps_per_second),
            "train/samples_per_sec": float(samples_per_second),
            "train/optimizer_skipped": int(optimizer_skipped),
        }
        for key, value in loss_metrics.items():
            payload[f"train/{key}"] = float(value)
        self._log_tensorboard_scalars(payload)
        self._wandb_log(payload)

    def _close_loggers(self) -> None:
        """Flush durable local events before closing optional remote logging."""
        tensorboard_logger = getattr(self, "tensorboard_logger", None)
        try:
            if tensorboard_logger is not None:
                try:
                    tensorboard_logger.flush()
                finally:
                    tensorboard_logger.close()
        finally:
            self._finish_wandb()

    def _build_loader(self, dataset, worker_init_fn=None):
        self.train_sampler = ResumableEpochSampler(
            dataset=dataset,
            seed=self.seed,
            batch_size=self.batch_size,
            num_processes=self.accelerator.num_processes,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=self.train_sampler,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=worker_init_fn,
        )

    def _assert_dataset_length_consistent(self, dataset, dataset_name: str):
        if not hasattr(dataset, "__len__"):
            raise TypeError(f"`{dataset_name}` must implement __len__ for rank consistency checks.")

        local_length = len(dataset)
        gathered_lengths = self.accelerator.gather(
            torch.tensor([local_length], device=self.accelerator.device, dtype=torch.int64)
        ).reshape(-1)
        if torch.all(gathered_lengths == gathered_lengths[0]):
            return

        if self.accelerator.is_main_process:
            print(f"[dataset-check] {dataset_name} length mismatch across ranks after initialization:")
            for rank, rank_length in enumerate(gathered_lengths.cpu().tolist()):
                print(f"rank {rank}: {rank_length}")
        self.accelerator.wait_for_everyone()
        raise RuntimeError(
            f"{dataset_name} length mismatch across ranks: {gathered_lengths.cpu().tolist()}"
        )

    def _estimate_total_train_steps(self) -> int:
        if self.max_steps is not None:
            return max(int(self.max_steps), 1)

        if not hasattr(self.train_dataset, "__len__"):
            raise TypeError("`train_dataset` must implement __len__ when `max_steps` is None.")

        num_processes = max(int(self.accelerator.num_processes), 1)
        global_batch_size = max(self.batch_size * num_processes, 1)
        # ResumableEpochSampler drops a partial distributed batch rather than
        # allowing Accelerate to pad it with duplicate samples.
        retained_samples = len(self.train_dataset) - len(self.train_dataset) % global_batch_size
        if retained_samples == 0:
            raise ValueError(
                "train dataset must contain at least one complete distributed batch: "
                f"dataset_length={len(self.train_dataset)} global_batch_size={global_batch_size}"
            )
        micro_steps_per_epoch = retained_samples // global_batch_size
        opt_steps_per_epoch = max(
            ceil(micro_steps_per_epoch / self.gradient_accumulation_steps),
            1,
        )
        return max(opt_steps_per_epoch * self.num_epochs, 1)

    def _build_scheduler(self, scheduler_type, total_train_steps: int, warmup_steps: int = 0):
        scheduler_type = str(scheduler_type).strip().lower()
        total_train_steps = max(int(total_train_steps), 1)
        warmup_steps = min(max(int(warmup_steps), 0), total_train_steps - 1)

        remaining_steps = max(total_train_steps - warmup_steps, 1)
        if scheduler_type == "cosine":
            main_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=remaining_steps,
                eta_min=self.min_learning_rate,
            )
        elif scheduler_type == "constant":
            main_scheduler = ConstantLR(self.optimizer, factor=1.0, total_iters=remaining_steps)
        else:
            raise ValueError(
                f"Unsupported lr_scheduler_type: {scheduler_type}. "
                "Expected one of: ['cosine', 'constant']."
            )

        if warmup_steps <= 0:
            return main_scheduler

        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=1.0 / warmup_steps,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps],
        )
    
    def _estimate_eta(self):
        elapsed = max(time.perf_counter() - self.run_start_time, 1e-6)
        done_steps = max(self.global_step - self.run_start_step, 1)
        steps_per_sec = done_steps / elapsed
        remaining_steps = max(self.max_steps - self.global_step, 0)
        eta_seconds = int(remaining_steps / max(steps_per_sec, 1e-9))
        eta_h, eta_rem = divmod(eta_seconds, 3600)
        eta_m, eta_s = divmod(eta_rem, 60)
        return f"{eta_h:02d}:{eta_m:02d}:{eta_s:02d}", steps_per_sec

    def _resume_or_load_checkpoint(self):
        flex_warmstart = getattr(self, "flex_warmstart", None)
        if flex_warmstart:
            from .models.wan22.helpers.flex_warmstart import load_flex_warmstart

            model = self.accelerator.unwrap_model(self.model)
            report = load_flex_warmstart(model, flex_warmstart)
            logger.info(
                "Loaded FastWAM Flex weights-only warm-start from %s "
                "(source_step=%s, mot_tensors=%d, proprio_tensors=%d, "
                "adapter_tensors=%d, adapters=%s).",
                report.path,
                report.source_step,
                report.mot_tensor_count,
                report.proprio_tensor_count,
                report.adapter_tensor_count,
                report.adapter_names,
            )
            logger.warning(
                "FastWAM Flex warm-start did not restore optimizer/scheduler/step; "
                "training starts a new timeline at step 0."
            )
            return
        legacy_warmstart = getattr(self, "legacy_warmstart", None)
        if legacy_warmstart:
            from .models.wan22.helpers.legacy_warmstart import load_legacy_warmstart

            model = self.accelerator.unwrap_model(self.model)
            report = load_legacy_warmstart(model, legacy_warmstart)
            logger.info(
                "Loaded legacy FastWAM weights-only warm-start from %s "
                "(source_step=%s, mot_tensors=%d, proprio_tensors=%d, new_prefixes=%s).",
                report.path,
                report.source_step,
                report.mot_tensor_count,
                report.proprio_tensor_count,
                report.new_state_prefixes,
            )
            logger.warning(
                "Legacy FastWAM warm-start did not restore optimizer/scheduler/step; "
                "training starts a new timeline at step 0."
            )
            return
        resume = self.resume
        if not resume:
            return
        resume_path = Path(str(resume))
        if resume_path.is_dir():
            logger.info("Resuming full training state from directory: %s", resume)
            self.load_training_state(str(resume_path))
            return
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume}")
        logger.info("Loading weight checkpoint only: %s", resume)
        model = self.accelerator.unwrap_model(self.model)
        load_kwargs = {"optimizer": None}
        if self.checkpoint_contract is not None:
            if not (
                _accepts_keyword(model.load_checkpoint, "expected_contract")
                and _accepts_keyword(model.load_checkpoint, "contract_mode")
            ):
                raise TypeError(
                    f"{type(model).__name__}.load_checkpoint does not support strict "
                    "checkpoint contract validation."
                )
            load_kwargs.update(
                expected_contract=self.checkpoint_contract,
                contract_mode="strict",
            )
        model.load_checkpoint(str(resume_path), **load_kwargs)
        logger.warning("Loaded .pt weights only; optimizer/scheduler/step were not restored under ZeRO2.")

    def _set_dit_only_train_mode(self):
        # Match DiffSynth's freeze_except("dit"): only DiT stays trainable/in-train-mode.
        logger.info("Setting DiT to train mode and freezing other model components.")
        model = self.accelerator.unwrap_model(self.model)
        self._apply_dit_only_train_mode(model)
        self._assert_configured_mechanisms_plumbed(model)

    @staticmethod
    def _apply_dit_only_train_mode(model):
        model.eval()
        model.requires_grad_(False)
        for module in Wan22Trainer._trainable_modules(model):
            module.train()
            module.requires_grad_(True)
        Wan22Trainer._assert_training_mechanisms_effective(model)

    @staticmethod
    def _assert_training_mechanisms_effective(model) -> None:
        """Fail if a configured training mechanism cannot actually take effect.

        State-token dropout is gated on ``proprio_encoder.training``. Since train
        mode is established by putting the whole model in eval mode and then
        re-enabling only ``trainable_module_names``, removing ``proprio_encoder``
        from that whitelist would disable the dropout silently: the config would
        still say it is on and the loss curve would look unremarkable.
        """
        probability = float(getattr(model, "proprio_token_dropout_prob", 0.0) or 0.0)
        shared_timestep = bool(getattr(model, "shared_train_timestep", False))

        if probability > 0.0:
            encoder = getattr(model, "proprio_encoder", None)
            if encoder is None:
                raise RuntimeError(
                    "proprio_token_dropout_prob="
                    f"{probability} but the model exposes no `proprio_encoder`, so the "
                    "state-token dropout could never apply."
                )
            if not encoder.training:
                raise RuntimeError(
                    "proprio_token_dropout_prob="
                    f"{probability} but `proprio_encoder` is in eval mode after entering "
                    "train mode, so the state-token dropout would never fire. Add "
                    "`proprio_encoder` to `trainable_module_names`."
                )

        if shared_timestep:
            # Sharing the RGB timestep value is only valid when every stream
            # interprets that value on the same schedule. Otherwise equal `t`
            # means unequal sigma (different step counts), and the receiving
            # stream's loss weights were normalized for a shift distribution
            # that was never sampled. Flex inference likewise advances all
            # streams on one step index, so its stream-local schedules must be
            # mutually aligned too. Train and inference shifts may differ from
            # each other intentionally; alignment is required within each phase.
            for phase in ("train", "infer"):
                schedulers = []
                for stream in ("video", "dino", "pointmap", "action"):
                    name = f"{phase}_{stream}_scheduler"
                    scheduler = getattr(model, name, None)
                    if scheduler is not None:
                        schedulers.append((name, scheduler))
                if len(schedulers) < 2:
                    raise RuntimeError(
                        "shared_train_timestep=True requires at least two participating "
                        f"{phase} schedulers, found {[name for name, _ in schedulers]}."
                    )
                signatures = {
                    name: (
                        int(getattr(scheduler, "num_train_timesteps")),
                        float(getattr(scheduler, "shift")),
                    )
                    for name, scheduler in schedulers
                }
                if len(set(signatures.values())) != 1:
                    raise RuntimeError(
                        "shared_train_timestep=True requires mutually aligned "
                        f"{phase} schedulers (num_train_timesteps, shift), got "
                        f"{signatures}. Keep shift overrides run-specific and align "
                        "every participating stream in that run's task config."
                    )

        logger.info(
            "Training mechanisms: state_token_dropout=%.3f (%s) shared_train_timestep=%s",
            probability,
            "active" if probability > 0.0 else "disabled",
            shared_timestep,
        )

    def _assert_configured_mechanisms_plumbed(self, model) -> None:
        """Fail if the requested mechanism config never reached the model.

        Catches builders that accept a switch but forget to forward it: the
        persisted config would advertise the mechanism while training ignores it.
        """

        def read(container, key):
            if container is None:
                return None
            try:
                return container.get(key, None)
            except Exception:  # noqa: BLE001 - plain objects and DictConfig both allowed
                return getattr(container, key, None)

        model_cfg = read(self.cfg, "model")
        if model_cfg is None:
            return

        requested_probability = read(model_cfg, "proprio_token_dropout_prob")
        if requested_probability is not None:
            actual = float(getattr(model, "proprio_token_dropout_prob", 0.0) or 0.0)
            if abs(float(requested_probability) - actual) > 1e-9:
                raise RuntimeError(
                    "config requested proprio_token_dropout_prob="
                    f"{float(requested_probability)} but the model was built with {actual}; "
                    "the model builder is not forwarding this setting."
                )

        requested_shared = read(read(model_cfg, "video_scheduler"), "shared_train_timestep")
        if requested_shared is not None:
            actual_shared = bool(getattr(model, "shared_train_timestep", False))
            if bool(requested_shared) != actual_shared:
                raise RuntimeError(
                    "config requested shared_train_timestep="
                    f"{bool(requested_shared)} but the model was built with {actual_shared}; "
                    "the model builder is not forwarding this setting."
                )

    @staticmethod
    def _trainable_modules(model) -> tuple[torch.nn.Module, ...]:
        """Resolve the model-owned training boundary without duplicating parameters."""
        names = getattr(model, "trainable_module_names", None)
        if names is None:
            names = ("dit", "proprio_encoder")
        if not isinstance(names, (tuple, list)) or not names:
            raise TypeError("`trainable_module_names` must be a non-empty tuple/list.")

        modules = []
        seen = set()
        for name in names:
            if not isinstance(name, str) or not name:
                raise TypeError("Every `trainable_module_names` entry must be a non-empty string.")
            module = getattr(model, name, None)
            if module is None:
                if name == "proprio_encoder":
                    continue
                raise AttributeError(f"Trainable module `{name}` does not exist on {type(model).__name__}.")
            if not isinstance(module, torch.nn.Module):
                raise TypeError(f"Trainable attribute `{name}` must be a torch.nn.Module.")
            if id(module) not in seen:
                seen.add(id(module))
                modules.append(module)
        return tuple(modules)

    @staticmethod
    def _trainable_parameters(model) -> list[torch.nn.Parameter]:
        parameters = []
        seen = set()
        for module in Wan22Trainer._trainable_modules(model):
            for parameter in module.parameters():
                if parameter.requires_grad and id(parameter) not in seen:
                    seen.add(id(parameter))
                    parameters.append(parameter)
        if not parameters:
            raise RuntimeError("Model exposes no trainable parameters.")
        return parameters

    @staticmethod
    def _capture_model_execution_state(*roots):
        """Capture per-object mode and gradient flags without flattening mixed modes."""
        modules = []
        parameters = []
        seen_modules = set()
        seen_parameters = set()
        for root in roots:
            if root is None:
                continue
            root_modules = root.modules() if isinstance(root, torch.nn.Module) else (root,)
            for module in root_modules:
                module_id = id(module)
                if module_id not in seen_modules and hasattr(module, "training"):
                    seen_modules.add(module_id)
                    modules.append((module, bool(module.training)))
                if not isinstance(module, torch.nn.Module):
                    continue
                for parameter in module.parameters(recurse=False):
                    parameter_id = id(parameter)
                    if parameter_id not in seen_parameters:
                        seen_parameters.add(parameter_id)
                        parameters.append((parameter, bool(parameter.requires_grad)))
        return modules, parameters

    @staticmethod
    def _restore_model_execution_state(state):
        modules, parameters = state
        # Assign each flag directly: Module.train() recurses and destroys mixed
        # DiT-train / frozen-component-eval state.
        for module, training in modules:
            module.training = training
        for parameter, requires_grad in parameters:
            parameter.requires_grad_(requires_grad)

    @staticmethod
    def _to_batched_eval_sample(sample):
        video = sample["video"]
        prompt = sample["prompt"]
        action = sample.get("action", None)
        proprio = sample.get("proprio", None)
        context = sample.get("context", None)
        context_mask = sample.get("context_mask", None)

        if not isinstance(video, torch.Tensor):
            raise TypeError(
                f"Expected tensor video for evaluation, got {type(video)}. "
                "Evaluation now expects `video` with shape [3,T,H,W] or [B,3,T,H,W]."
            )
        if video.ndim == 4:
            video = video.unsqueeze(0)
        if video.ndim != 5:
            raise ValueError(f"Expected video shape [3,T,H,W] or [B,3,T,H,W], got {tuple(video.shape)}")
        num_video_frames = video.shape[2]
        if num_video_frames <= 1:
            raise ValueError(f"`sample['video']` must have at least 2 frames for action evaluation, got {num_video_frames}")

        if isinstance(prompt, str):
            prompt = [prompt]
        elif isinstance(prompt, tuple):
            prompt = list(prompt)
        elif not isinstance(prompt, list):
            raise TypeError(f"Expected prompt type str/list[str], got {type(prompt)}")
        if len(prompt) != video.shape[0]:
            raise ValueError(f"Prompt batch mismatch: len(prompt)={len(prompt)} vs video batch={video.shape[0]}")
        
        action_horizon = None
        action = None
        if "action" in sample:
            action = sample["action"]
            if not isinstance(action, torch.Tensor):
                raise TypeError(
                    f"`sample['action']` must be a torch.Tensor, got {type(action)}"
                )
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3:
                raise ValueError(f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}")
            if action.shape[1] % (num_video_frames - 1) != 0:
                raise ValueError(f"`sample['action']` temporal dimension must be divisible by video frames-1={num_video_frames - 1}, got {action.shape[1]}")
            action_horizon = int(action.shape[1])

        proprio = None
        if "proprio" in sample:
            proprio = sample["proprio"]
            if not isinstance(proprio, torch.Tensor):
                raise TypeError(f"`sample['proprio']` must be a torch.Tensor, got {type(proprio)}")
            if proprio.ndim == 2:
                proprio = proprio.unsqueeze(0)
            if proprio.ndim != 3:
                raise ValueError(f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}")

        if context is not None or context_mask is not None:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must both exist in eval sample.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )

        result = dict(sample)
        result.update({
            "video": video,
            "prompt": prompt,
            "action": action,
            "proprio": proprio,
            "context": context,
            "context_mask": context_mask,
            "action_horizon": action_horizon,
        })
        pointmap = sample.get("pointmap")
        if pointmap is not None:
            if not isinstance(pointmap, torch.Tensor):
                raise TypeError(
                    f"`sample['pointmap']` must be a torch.Tensor, got {type(pointmap)}"
                )
            if pointmap.ndim == 4:
                pointmap = pointmap.unsqueeze(0)
            if pointmap.ndim != 5 or pointmap.shape[:3] != video.shape[:3]:
                raise ValueError(
                    "`sample['pointmap']` must match video [B,3,T,...], got "
                    f"{tuple(pointmap.shape)} vs video {tuple(video.shape)}"
                )
            result["pointmap"] = pointmap
        for key in ("action_is_pad", "image_is_pad"):
            value = sample.get(key)
            if isinstance(value, torch.Tensor) and value.ndim == 1:
                result[key] = value.unsqueeze(0)
        return result

    @torch.no_grad()
    def evaluate_fixed_validation_loss(
        self,
        *,
        step: int | None = None,
        tasks: int = 100,
        anchors_per_task: int = 10,
        stride: int = 6,
    ) -> dict[str, float | int | str]:
        """Evaluate fixed B1K windows with reproducible diffusion randomness.

        The same 1000 windows and per-window seed are used at every requested
        checkpoint step, so LR sweeps can compare action/video loss directly.
        This evaluator intentionally reports the model's weighted losses, as
        returned by ``training_loss``.
        """
        if self.val_dataset is None:
            raise ValueError("fixed validation requires `val_dataset`")
        windows = build_fixed_validation_window_indices(
            self.val_dataset, tasks=tasks, anchors_per_task=anchors_per_task, stride=stride
        )
        window_set_sha256 = fixed_validation_window_set_sha256(windows)
        self._fixed_validation_window_set_sha256 = window_set_sha256
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        model = self.model if hasattr(self.model, "training_loss") else unwrapped_model
        execution_state = self._capture_model_execution_state(model, unwrapped_model)
        metric_totals: dict[str, float] = {"loss_total": 0.0}
        local_data_load_s = 0.0
        local_forward_s = 0.0
        local_started_unix_s = time.time()
        local_started = time.perf_counter()
        try:
            model.eval()
            local_windows = windows[self.accelerator.process_index :: self.accelerator.num_processes]
            for window_id, episode_index, global_index in local_windows:
                data_started = time.perf_counter()
                sample = self._to_batched_eval_sample(self.val_dataset[global_index])
                local_data_load_s += time.perf_counter() - data_started
                seed = fixed_validation_seed(
                    window_id, episode_index, global_index, base_seed=self.fixed_validation_seed
                )
                if torch.cuda.is_available() and torch.device(self.accelerator.device).type == "cuda":
                    torch.cuda.synchronize(self.accelerator.device)
                forward_started = time.perf_counter()
                with self.accelerator.autocast():
                    loss, loss_dict = model.training_loss(sample, rng_seed=seed)
                if torch.cuda.is_available() and torch.device(self.accelerator.device).type == "cuda":
                    torch.cuda.synchronize(self.accelerator.device)
                local_forward_s += time.perf_counter() - forward_started
                metric_totals["loss_total"] += float(loss.detach().float().item())
                for name, value in loss_dict.items():
                    if not str(name).startswith("loss_"):
                        continue
                    metric_totals[name] = metric_totals.get(name, 0.0) + float(value)
        finally:
            self._restore_model_execution_state(execution_state)
        local_total_s = time.perf_counter() - local_started
        local_finished_unix_s = time.time()
        metric_names = sorted(metric_totals)
        totals = torch.tensor(
            [metric_totals[name] for name in metric_names] + [float(len(local_windows))],
            device=self.accelerator.device,
            dtype=torch.float64,
        )
        totals = self.accelerator.reduce(totals, reduction="sum")
        reduced = [float(x) for x in totals.detach().cpu()]
        count_value = reduced[-1]
        count = int(count_value)
        if count <= 0:
            raise RuntimeError("fixed validation produced no windows")
        timing_row = torch.tensor(
            [
                float(self.accelerator.process_index),
                float(len(local_windows)),
                local_data_load_s,
                local_forward_s,
                local_total_s,
                local_started_unix_s,
                local_finished_unix_s,
            ],
            dtype=torch.float64,
            device=self.accelerator.device,
        )
        timing_rows = self.accelerator.gather(timing_row).reshape(-1, 7)
        timings = _summarize_fixed_validation_rank_timings(timing_rows)
        max_duration_seconds = float(
            getattr(self, "fixed_validation_max_duration_seconds", float("inf"))
        )
        if timings["total_s"]["max_seconds"] > max_duration_seconds:
            raise RuntimeError(
                "fixed validation exceeded its duration gate: "
                f"max_seconds={timings['total_s']['max_seconds']}, "
                f"limit={max_duration_seconds}"
            )
        return {
            "step": int(self.global_step if step is None else step),
            "num_windows": count,
            **{name: value / count for name, value in zip(metric_names, reduced[:-1])},
            "stride": int(stride),
            "fixed_validation_seed": int(self.fixed_validation_seed),
            "initialization_fingerprint_sha256": self.initialization_fingerprint_sha256,
            "window_set_sha256": window_set_sha256,
            "timings": timings,
        }

    def _persist_fixed_validation_metrics(self, metrics: dict[str, float | int | str]) -> None:
        """Write per-step JSON and an append-only summary on the main rank."""
        if not self.accelerator.is_main_process:
            return
        metrics = dict(metrics)
        metrics.setdefault(
            "initialization_fingerprint_sha256",
            getattr(self, "initialization_fingerprint_sha256", None),
        )
        metrics.setdefault(
            "window_set_sha256",
            getattr(self, "_fixed_validation_window_set_sha256", None),
        )
        report_dir = Path(self.fixed_validation_report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        summary_path = report_dir / "summary.json"
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            payload = None

        if payload is None:
            records = []
        else:
            if not isinstance(payload, dict):
                raise ValueError(
                    f"Fixed-validation summary must be a JSON object: {summary_path}"
                )
            records = payload.get("records")
            if not isinstance(records, list):
                raise ValueError(
                    f"Fixed-validation summary records must be a JSON list: {summary_path}"
                )
            for key in (
                "fixed_validation_seed",
                "initialization_fingerprint_sha256",
                "window_set_sha256",
            ):
                previous = payload.get(key)
                current = metrics.get(key)
                if previous is not None and current is not None and previous != current:
                    raise RuntimeError(
                        f"Fixed-validation {key} changed within one run: "
                        f"previous={previous}, current={current}"
                    )
        _atomic_write_json(
            str(report_dir / f"step-{int(metrics['step']):06d}.json"), metrics
        )
        records = [record for record in records if int(record.get("step", -1)) != int(metrics["step"])]
        records.append(metrics)
        records.sort(key=lambda record: int(record["step"]))
        _atomic_write_json(
            str(summary_path),
            {
                "schema": "b1k_fixed_validation.v1",
                "status": "passed",
                "steps": [int(record["step"]) for record in records],
                "num_windows": int(metrics["num_windows"]),
                "fixed_validation_seed": metrics.get("fixed_validation_seed"),
                "initialization_fingerprint_sha256": metrics.get(
                    "initialization_fingerprint_sha256"
                ),
                "window_set_sha256": metrics.get("window_set_sha256"),
                "records": records,
            },
        )

    @torch.no_grad()
    def evaluate(self):
        if self.val_dataset is None:
            return None

        model = self.accelerator.unwrap_model(self.model)
        was_dit_training = model.dit.training
        model.eval()

        # eval_index = (self.global_step + self.accelerator.process_index) % len(self.val_dataset)
        rng = torch.Generator(device="cpu").manual_seed(self.global_step + self.accelerator.process_index)
        eval_index = torch.randint(0, len(self.val_dataset), (1,), generator=rng).item()
        sample = self._to_batched_eval_sample(self.val_dataset[eval_index])

        # 1. training loss
        with self.accelerator.autocast():
            val_loss, _ = model.training_loss(sample)
            val_loss = val_loss.float().item()
        
        prompt = sample["prompt"][0]
        video0 = sample["video"][0] # Tensor [3, T, H, W] in (-1, 1)
        action = sample["action"][0] if "action" in sample and sample["action"] is not None else None
        proprio = sample["proprio"][0, 0] if "proprio" in sample and sample["proprio"] is not None else None # from [1, T, d] to [d]
        input_image = video0[:, 0].unsqueeze(0)
        _, num_frames, _, _ = video0.shape

        # 2. inference and video saving
        infer_kwargs = {
            "input_image": input_image,
            "num_frames": num_frames,
            "action": action,
            "action_horizon": sample['action_horizon'],
            "proprio": proprio,
            "text_cfg_scale": 1.0,
            "action_cfg_scale": 1.0,
            "num_inference_steps": self.eval_num_inference_steps,
            "seed": 42,
            "tiled": False,
        }
        if sample["context"] is not None:
            infer_kwargs["prompt"] = None
            infer_kwargs["context"] = sample["context"][0]
            infer_kwargs["context_mask"] = sample["context_mask"][0]
        else:
            infer_kwargs["prompt"] = prompt

        pred = model.infer(
            **infer_kwargs,
        )
        
        pred_video = pred["video"]
        pred_action = pred.get("action", None)

        # 3. inference metrics against GT video
        pred_video_tensor = pil_frames_to_video_tensor(pred_video)
        gt_video_tensor = ((video0.detach().float().cpu().clamp(-1.0, 1.0) + 1.0) * 0.5).contiguous()

        assert pred_video_tensor.shape == gt_video_tensor.shape, (
            "Eval infer prediction/GT shape mismatch: "
            f"pred={tuple(pred_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
        )

        psnr_rollout_vs_gt = video_psnr(pred=pred_video_tensor, target=gt_video_tensor)
        ssim_rollout_vs_gt = video_ssim(pred=pred_video_tensor, target=gt_video_tensor)

        action_l1 = None
        action_l2 = None
        if action is not None and pred_action is not None:
            if sample["proprio"] is None:
                raise ValueError("Eval sample must contain `proprio` for action denormalization.")
            proprio = sample["proprio"].detach().to(device="cpu", dtype=torch.float32)
            
            processor = self.val_dataset.lerobot_dataset.processor

            denorm_actions = {}
            action_meta = processor.shape_meta["action"]
            state_meta = processor.shape_meta["state"]
            for action_name, raw_action in (("pred", pred_action), ("gt", action)):
                if not isinstance(raw_action, torch.Tensor):
                    raise TypeError(f"{action_name} action must be a torch.Tensor, got {type(raw_action)}")
                if raw_action.ndim == 2:
                    action_btd = raw_action.unsqueeze(0)
                elif raw_action.ndim == 3 and raw_action.shape[0] == 1:
                    action_btd = raw_action
                else:
                    raise ValueError(
                        f"{action_name} action must have shape [T, D] or [1, T, D], got {tuple(raw_action.shape)}"
                    )
                action_btd = action_btd.detach().to(device="cpu", dtype=torch.float32)

                batch = {
                    "action": action_btd,
                    "state": proprio,
                }
                batch = processor.action_state_merger.backward(batch)
                batch = processor.normalizer.backward(batch)
                merged_batch = {
                    "action": {meta["key"]: batch["action"][meta["key"]].squeeze(0) for meta in action_meta},
                    "state": {meta["key"]: batch["state"][meta["key"]].squeeze(0) for meta in state_meta},
                }
                merged_batch = processor.action_state_merger.forward(merged_batch)
                denorm_action = merged_batch["action"].unsqueeze(0)
                if denorm_action.ndim != 3 or denorm_action.shape[0] != 1:
                    raise ValueError(
                        f"Denormalized {action_name} action must have shape [1, T, D], got {tuple(denorm_action.shape)}"
                    )
                denorm_actions[action_name] = denorm_action

            pred_action_denorm = denorm_actions["pred"]
            gt_action_denorm = denorm_actions["gt"]

            if pred_action_denorm.shape != gt_action_denorm.shape:
                raise ValueError(
                    "Predicted action/GT action shape mismatch after denormalization: "
                    f"pred={tuple(pred_action_denorm.shape)} vs gt={tuple(gt_action_denorm.shape)}"
                )
            action_diff = pred_action_denorm - gt_action_denorm
            action_l1 = action_diff.abs().mean().item()
            action_l2 = action_diff.pow(2).mean().item()

        # 4. VAE reconstruction metrics against GT video
        gt_video_batch = video0.unsqueeze(0).to(device=model.device, dtype=model.torch_dtype)
        vae_latents = model._encode_video_latents(gt_video_batch, tiled=False)
        vae_recon_video = model._decode_latents(vae_latents, tiled=False)
        vae_video_tensor = pil_frames_to_video_tensor(vae_recon_video)

        assert vae_video_tensor.shape == gt_video_tensor.shape, (
            "Eval VAE reconstruction/GT shape mismatch: "
            f"vae={tuple(vae_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
        )

        psnr_decode_vs_gt = video_psnr(pred=vae_video_tensor, target=gt_video_tensor)
        ssim_decode_vs_gt = video_ssim(pred=vae_video_tensor, target=gt_video_tensor)

        psnr_rollout_vs_decode = video_psnr(pred=pred_video_tensor, target=vae_video_tensor)
        ssim_rollout_vs_decode = video_ssim(pred=pred_video_tensor, target=vae_video_tensor)

        stitched_video_tensor = torch.cat(
            [pred_video_tensor, vae_video_tensor, gt_video_tensor],
            dim=2,
        ).contiguous()
        stitched_frames = []
        for t in range(stitched_video_tensor.shape[1]):
            frame = (stitched_video_tensor[:, t].permute(1, 2, 0).clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
            stitched_frames.append(Image.fromarray(frame))

        video_path = os.path.join(
            self.eval_dir,
            f"step_{self.global_step:06d}_rank_{self.accelerator.process_index:03d}.mp4",
        )
        save_mp4(stitched_frames, video_path, fps=8)

        local_metrics = torch.tensor(
            [
                float(val_loss),
                float(psnr_rollout_vs_gt),
                float(ssim_rollout_vs_gt),
                float(psnr_rollout_vs_decode),
                float(ssim_rollout_vs_decode),
                float(psnr_decode_vs_gt),
                float(ssim_decode_vs_gt),
                float(action_l2) if action_l2 is not None else -1.0,
                float(action_l1) if action_l1 is not None else -1.0,
            ],
            device=self.accelerator.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        gathered_metrics = self.accelerator.gather_for_metrics(local_metrics)
        mean_metrics = gathered_metrics[:, :7].mean(dim=0)
        action_l2_mean = gathered_metrics[:, 7].mean().item() if action_l2 is not None else None
        action_l1_mean = gathered_metrics[:, 8].mean().item() if action_l1 is not None else None

        if was_dit_training:
            self._set_dit_only_train_mode()

        result = {
            "val_loss": float(mean_metrics[0].item()),
            "psnr_rg": float(mean_metrics[1].item()),
            "ssim_rg": float(mean_metrics[2].item()),
            "psnr_rd": float(mean_metrics[3].item()),
            "ssim_rd": float(mean_metrics[4].item()),
            "psnr_dg": float(mean_metrics[5].item()),
            "ssim_dg": float(mean_metrics[6].item()),
            "video_path": video_path,
        }
        if action_l2_mean is not None:
            result["action_l2"] = float(action_l2_mean)
        if action_l1_mean is not None:
            result["action_l1"] = float(action_l1_mean)
        return result

    def _save_weights_checkpoint(self, step_tag: str):
        model = self.accelerator.unwrap_model(self.model)
        ckpt_path = os.path.join(self.weights_dir, f"{step_tag}.pt")
        save_kwargs = {"optimizer": None, "step": self.global_step}
        if self.checkpoint_contract is not None:
            if not _accepts_keyword(model.save_checkpoint, "checkpoint_contract"):
                raise TypeError(
                    f"{type(model).__name__}.save_checkpoint does not accept "
                    "`checkpoint_contract`; refusing to write an unverifiable checkpoint."
                )
            save_kwargs["checkpoint_contract"] = self.checkpoint_contract
        model.save_checkpoint(ckpt_path, **save_kwargs)
        return ckpt_path

    def _save_permanent_weights_checkpoint(self, step_tag: str):
        """Save a standalone model snapshot that is independent of resume-state pruning."""
        model = self.accelerator.unwrap_model(self.model)
        ckpt_path = os.path.join(self.permanent_weights_dir, f"{step_tag}.pt")
        save_kwargs = {"optimizer": None, "step": self.global_step}
        if self.checkpoint_contract is not None:
            if not _accepts_keyword(model.save_checkpoint, "checkpoint_contract"):
                raise TypeError(
                    f"{type(model).__name__}.save_checkpoint does not accept "
                    "`checkpoint_contract`; refusing to write an unverifiable checkpoint."
                )
            save_kwargs["checkpoint_contract"] = self.checkpoint_contract
        descriptor, temporary_path = tempfile.mkstemp(
            dir=self.permanent_weights_dir,
            prefix=f".{step_tag}.",
            suffix=".tmp",
        )
        os.close(descriptor)
        try:
            model.save_checkpoint(temporary_path, **save_kwargs)
            with open(temporary_path, "rb") as stream:
                os.fsync(stream.fileno())
            os.replace(temporary_path, ckpt_path)
            directory_fd = os.open(
                self.permanent_weights_dir, os.O_RDONLY | os.O_DIRECTORY
            )
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            Path(temporary_path).unlink(missing_ok=True)
        return ckpt_path

    def _publish_permanent_weights_link(self, weights_path: str, step_tag: str) -> str:
        """Expose final weights in the permanent directory without duplicating storage."""
        destination = Path(self.permanent_weights_dir) / f"{step_tag}.pt"
        temporary = Path(self.permanent_weights_dir) / f".{step_tag}.{os.getpid()}.tmp"
        try:
            temporary.unlink(missing_ok=True)
            os.link(weights_path, temporary)
            os.replace(temporary, destination)
            directory_fd = os.open(
                self.permanent_weights_dir, os.O_RDONLY | os.O_DIRECTORY
            )
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            temporary.unlink(missing_ok=True)
        return str(destination)

    def _save_trainer_state(self, state_path: str):
        state_file = os.path.join(state_path, "trainer_state.json")
        payload = {
            "global_step": int(self.global_step),
            "epoch": int(self.epoch),
            "batch_in_epoch": int(self.batch_in_epoch),
        }
        if self.checkpoint_contract is not None:
            payload["checkpoint_contract_sha256"] = self.checkpoint_contract.contract_sha256
        _atomic_write_json(state_file, payload)

    def _optimizer_resume_signature(self) -> dict:
        optimizer = self.optimizer
        visited = set()
        while id(optimizer) not in visited:
            visited.add(id(optimizer))
            state = getattr(optimizer, "state", None)
            if state:
                break
            nested = getattr(optimizer, "optimizer", None)
            if nested is None:
                break
            optimizer = nested
        state = getattr(optimizer, "state", None)
        param_groups = getattr(optimizer, "param_groups", None)
        if not state or not param_groups:
            raise RuntimeError(
                "Optimizer state is empty; full-state resume cannot be validated. "
                "The checkpoint probe must be created after at least one optimizer step."
            )

        parameter_positions = {}
        group_payload = []
        for group_index, group in enumerate(param_groups):
            group_parameters = list(group.get("params", []))
            for parameter_index, parameter in enumerate(group_parameters):
                parameter_positions[id(parameter)] = (group_index, parameter_index)
            group_payload.append(
                {
                    str(key): _jsonable_state(value)
                    for key, value in sorted(group.items())
                    if key != "params"
                }
            )

        state_payload = []
        for parameter, parameter_state in state.items():
            position = parameter_positions.get(id(parameter))
            if position is None:
                raise RuntimeError("Optimizer state contains an untracked parameter.")
            state_payload.append(
                {
                    "group": position[0],
                    "parameter": position[1],
                    "state": _jsonable_state(parameter_state),
                }
            )
        state_payload.sort(key=lambda item: (item["group"], item["parameter"]))
        payload = {"param_groups": group_payload, "parameter_states": state_payload}
        return {
            "sha256": _json_digest(payload),
            "parameter_state_count": len(state_payload),
            "param_group_count": len(group_payload),
        }

    def _scheduler_resume_signature(self) -> dict:
        raw_state = self.scheduler.state_dict()
        state = _jsonable_state(raw_state)
        return {
            "sha256": _json_digest(state),
            "last_epoch": int(
                raw_state.get("last_epoch", getattr(self.scheduler, "last_epoch", -1))
            ),
        }

    def _resume_probe_payload(self) -> dict:
        sampler_preview = self.train_sampler.preview_indices(
            epoch=self.epoch,
            batch_in_epoch=self.batch_in_epoch,
            count=32,
        )
        return {
            "schema_version": 1,
            "rank": int(self.accelerator.process_index),
            "world_size": int(self.accelerator.num_processes),
            "global_step": int(self.global_step),
            "epoch": int(self.epoch),
            "batch_in_epoch": int(self.batch_in_epoch),
            "dataloader": {
                "dataset_length": int(len(self.train_dataset)),
                "per_gpu_batch": int(self.batch_size),
                "gradient_accumulation_steps": int(
                    self.gradient_accumulation_steps
                ),
                "num_workers": int(self.num_workers),
                "prefetch_factor": getattr(
                    self,
                    "prefetch_factor",
                    2 if int(self.num_workers) > 0 else None,
                ),
                "persistent_workers": bool(
                    getattr(self, "persistent_workers", False)
                ),
                "sampler_type": type(self.train_sampler).__name__,
                "sampler_seed": int(self.train_sampler.seed),
                "next_global_indices": sampler_preview,
            },
            "optimizer": self._optimizer_resume_signature(),
            "scheduler": self._scheduler_resume_signature(),
            "checkpoint_contract_sha256": (
                self.checkpoint_contract.contract_sha256
                if self.checkpoint_contract is not None
                else None
            ),
        }

    def _write_resume_probe(self, state_path: str) -> None:
        if not all(
            hasattr(self, name)
            for name in ("optimizer", "scheduler", "train_sampler", "train_dataset")
        ):
            return
        probe_path = Path(state_path) / (
            f"resume_probe_rank_{self.accelerator.process_index:03d}.json"
        )
        _atomic_write_json(probe_path, self._resume_probe_payload())

    def _write_checkpoint_complete_marker(
        self,
        *,
        state_path: str,
        weights_path: str,
        retention_class: str = "rolling",
    ) -> Path:
        state = Path(state_path)
        weights = Path(weights_path)
        probe_count = sum(1 for _ in state.glob("resume_probe_rank_*.json"))
        probes_required = all(
            hasattr(self, name)
            for name in ("optimizer", "scheduler", "train_sampler", "train_dataset")
        )
        if not weights.is_file():
            raise RuntimeError(
                f"Cannot commit checkpoint because weights are missing: {weights}"
            )
        if not (state / "trainer_state.json").is_file():
            raise RuntimeError(
                "Cannot commit checkpoint because trainer_state.json is missing: "
                f"{state}"
            )
        if self.checkpoint_contract is not None and not (
            state / "checkpoint_contract.json"
        ).is_file():
            raise RuntimeError(
                "Cannot commit checkpoint because checkpoint_contract.json is missing: "
                f"{state}"
            )
        if probes_required and probe_count != int(self.accelerator.num_processes):
            raise RuntimeError(
                "Cannot commit checkpoint before every rank resume probe is visible: "
                f"expected={self.accelerator.num_processes}, actual={probe_count}, "
                f"state={state}"
            )
        if retention_class not in {"rolling", "milestone", "final"}:
            raise ValueError(
                "checkpoint retention_class must be rolling, milestone, or final"
            )
        marker_path = state / _CHECKPOINT_COMPLETE_MARKER
        inventory = _build_checkpoint_inventory(state, weights)
        _atomic_write_json(
            marker_path,
            {
                "schema_version": _CHECKPOINT_COMPLETE_SCHEMA_VERSION,
                "global_step": int(self.global_step),
                "state_directory": state.name,
                "weights_file": weights.name,
                "world_size": int(self.accelerator.num_processes),
                "checkpoint_contract_required": self.checkpoint_contract is not None,
                "checkpoint_contract_sha256": (
                    self.checkpoint_contract.contract_sha256
                    if self.checkpoint_contract is not None
                    else None
                ),
                "resume_probes_required": probes_required,
                "resume_probe_count": probe_count,
                "completed_at_unix_ns": time.time_ns(),
                # This decision is committed with the checkpoint so a future
                # resume cannot accidentally prune it after config changes.
                "retention_class": retention_class,
                "inventory": inventory,
            },
        )
        return marker_path

    def _checkpoint_retention_class(self, *, is_final: bool = False) -> str:
        if is_final and getattr(self, "checkpoint_retention_keep_final", False):
            return "final"
        interval = getattr(
            self, "checkpoint_retention_keep_every_n_steps", None
        )
        if interval is not None and self.global_step % interval == 0:
            return "milestone"
        return "rolling"

    def save_checkpoint(self, *, is_final: bool = False):
        step_tag = f"step_{self.global_step:06d}"
        state_path = os.path.join(self.state_dir, step_tag)
        weights_path = Path(self.weights_dir) / f"{step_tag}.pt"
        retention_class = self._checkpoint_retention_class(is_final=is_final)
        checkpoint_started_at = time.perf_counter()
        if getattr(self, "profiling_enabled", False):
            self._cuda_synchronize()
            checkpoint_started_at = time.perf_counter()

        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            # Re-saving the same tag must first revoke any old commit record.
            # Otherwise a preemption during the overwrite can leave an old
            # marker falsely blessing a mixture of old and new files.
            old_marker = Path(state_path) / _CHECKPOINT_COMPLETE_MARKER
            if old_marker.exists():
                old_completion = _read_checkpoint_complete_marker(
                    Path(state_path), weights_path
                )
                if old_completion is not None and old_completion.get(
                    "retention_class"
                ) in {"milestone", "final"}:
                    retention_class = old_completion["retention_class"]
                old_marker.unlink()
                directory_fd = os.open(old_marker.parent, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        self.accelerator.wait_for_everyone()

        ckpt_path = None
        if self.accelerator.is_main_process:
            ckpt_path = self._save_weights_checkpoint(step_tag=step_tag)
        self.accelerator.wait_for_everyone()

        ensure_dir(state_path)
        if self.accelerator.is_main_process and self.checkpoint_contract is not None:
            write_checkpoint_contract(
                Path(state_path) / "checkpoint_contract.json",
                self.checkpoint_contract,
            )
        self.accelerator.wait_for_everyone()
        self.accelerator.save_state(output_dir=state_path)
        self._write_resume_probe(state_path)
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            self._save_trainer_state(state_path)
        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            # This is the commit point.  Every full-state shard, rank probe,
            # contract and trainer metadata file is globally visible before the
            # main rank publishes the atomic marker.
            self._write_checkpoint_complete_marker(
                state_path=state_path,
                weights_path=ckpt_path,
                retention_class=retention_class,
            )
        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            self._prune_complete_checkpoint_pairs(protect_step_tag=step_tag)
        self.accelerator.wait_for_everyone()

        ckpt_info = {"weights_path": ckpt_path, "state_path": state_path}
        if getattr(self, "profiling_enabled", False):
            self._cuda_synchronize()
        checkpoint_seconds = time.perf_counter() - checkpoint_started_at
        if getattr(self, "profiling_enabled", False):
            self._record_checkpoint_profile(checkpoint_seconds, ckpt_info)
        if self.accelerator.is_main_process:
            checkpoint_payload = {
                "timing/checkpoint_save_s": checkpoint_seconds
            }
            self._log_tensorboard_scalars(
                checkpoint_payload,
                flush=True,
            )
            self._wandb_log(checkpoint_payload)
        return ckpt_info

    def _prune_complete_checkpoint_pairs(
        self,
        *,
        protect_step_tag: str | None = None,
    ) -> list[int]:
        """Remove oldest complete state/weights pairs after a successful save.

        Only standard, paired ``step_N`` artifacts are eligible.  Unpaired or
        non-standard paths are retained so an interrupted save or an operator
        artifact is never silently discarded.
        """
        max_to_keep = getattr(self, "checkpoint_retention_max_to_keep", None)
        if max_to_keep is None:
            return []

        state_root = Path(self.state_dir)
        weights_root = Path(self.weights_dir)
        complete: list[tuple[int, int, Path, Path, bool]] = []
        for state_path in state_root.iterdir():
            match = re.fullmatch(r"step_(\d+)", state_path.name)
            if match is None or not state_path.is_dir():
                continue
            weights_path = weights_root / f"{state_path.name}.pt"
            marker = _read_checkpoint_complete_marker(state_path, weights_path)
            if marker is not None:
                retention_class = marker.get("retention_class")
                if retention_class is None:
                    # Schema-v1 markers written before milestone retention did
                    # not record a class. Preserve an old interval checkpoint
                    # when the current policy can identify it.
                    interval = getattr(
                        self, "checkpoint_retention_keep_every_n_steps", None
                    )
                    permanent = (
                        interval is not None
                        and int(match.group(1)) % interval == 0
                    )
                else:
                    permanent = retention_class in {"milestone", "final"}
                complete.append(
                    (
                        int(marker["completed_at_unix_ns"]),
                        int(match.group(1)),
                        state_path,
                        weights_path,
                        permanent,
                    )
                )

        # Completion order, rather than global step, remains correct after a
        # resume forks from an older checkpoint.  Explicitly reserve one slot
        # for the checkpoint just committed even if marker timestamps are
        # externally modified or collide.
        complete.sort(key=lambda item: (item[0], item[2].name))
        rolling = [item for item in complete if not item[4]]
        protected = [item for item in rolling if item[2].name == protect_step_tag]
        candidates = [item for item in rolling if item[2].name != protect_step_tag]
        keep_candidate_count = max_to_keep - min(1, len(protected))
        stale = candidates[:-keep_candidate_count] if keep_candidate_count else candidates
        if not protected:
            stale = rolling[:-max_to_keep]
        for _, step, state_path, weights_path, _ in stale:
            # Remove the standalone copy first.  If deleting the full state
            # fails, the still-resumable state remains and training fails loud.
            weights_path.unlink()
            shutil.rmtree(state_path)
            logger.info(
                "[ckpt] retention pruned complete checkpoint step=%d state=%s weights=%s",
                step,
                state_path,
                weights_path,
            )
        return [step for _, step, _, _, _ in stale]

    def save_final_checkpoint(self):
        """Save the configured natural-end checkpoint without changing periodic saves."""
        if self.final_checkpoint_mode == "full":
            result = self.save_checkpoint(is_final=True)
        else:
            checkpoint_started_at = time.perf_counter()
            step_tag = f"step_{self.global_step:06d}"
            self.accelerator.wait_for_everyone()
            weights_path = None
            if self.accelerator.is_main_process:
                weights_path = self._save_weights_checkpoint(step_tag=step_tag)
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                checkpoint_payload = {
                    "timing/checkpoint_save_s": (
                        time.perf_counter() - checkpoint_started_at
                    )
                }
                self._log_tensorboard_scalars(
                    checkpoint_payload,
                    flush=True,
                )
                self._wandb_log(checkpoint_payload)
            result = {"weights_path": weights_path, "state_path": None}

        permanent_interval = getattr(
            self, "permanent_weights_every_n_steps", None
        )
        if permanent_interval is not None and self.global_step % permanent_interval == 0:
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                self._publish_permanent_weights_link(
                    result["weights_path"], f"step_{self.global_step:06d}"
                )
            self.accelerator.wait_for_everyone()
        return result

    def load_training_state(self, state_dir: str):
        state_path = Path(state_dir)
        standard_tag = re.fullmatch(r"step_(\d+)", state_path.name)
        if standard_tag is not None:
            # Pair against the checkpoint root being resumed, not this run's
            # output directory.  A resumed run may intentionally write future
            # checkpoints under a different output_dir.
            weights_path = state_path.parent.parent / "weights" / f"{state_path.name}.pt"
            completion = _read_checkpoint_complete_marker(state_path, weights_path)
            if completion is None:
                raise CheckpointContractIntegrityError(
                    "Refusing to resume an incomplete checkpoint without a valid "
                    f"{_CHECKPOINT_COMPLETE_MARKER}: state={state_path}, "
                    f"weights={weights_path}"
                )
            if int(completion["world_size"]) != int(self.accelerator.num_processes):
                raise CheckpointContractIntegrityError(
                    "Full-state checkpoint world size does not match runtime: "
                    f"checkpoint={completion['world_size']} runtime="
                    f"{self.accelerator.num_processes}"
                )
        state_file = Path(state_dir) / "trainer_state.json"
        contract_file = Path(state_dir) / "checkpoint_contract.json"
        payload = None
        if state_file.exists():
            with open(state_file, "r", encoding="utf-8") as f:
                payload = json.load(f)

        actual_contract = None
        if contract_file.exists():
            actual_contract = read_checkpoint_contract(contract_file)
        if self.checkpoint_contract is not None:
            if actual_contract is None:
                raise CheckpointContractIntegrityError(
                    f"Training state is missing required checkpoint contract: {contract_file}"
                )
            try:
                assert_checkpoint_compatible(self.checkpoint_contract, actual_contract)
            except CheckpointContractMismatchError as error:
                allowed = (
                    (getattr(self, "resume_extension", False) or getattr(self, "resume_lr_override", False))
                    and error.differences
                    and all(
                        difference.startswith("$.compatibility.scheduler.optimizer_lr.max_steps:")
                        or (
                            getattr(self, "resume_lr_override", False)
                            and difference.startswith(
                                (
                                    "$.compatibility.scheduler.optimizer_lr.learning_rate:",
                                    "$.compatibility.scheduler.optimizer_lr.min_learning_rate:",
                                    "$.compatibility.scheduler.optimizer_lr.lr_scheduler_type:",
                                    "$.compatibility.scheduler.optimizer_lr.lr_warmup_steps:",
                                )
                            )
                        )
                        for difference in error.differences
                    )
                )
                if not allowed:
                    raise
                logger.warning(
                    "Allowing restricted full-state continuation contract drift "
                    "for terminal step and/or explicitly requested LR override."
                )
            if payload is None:
                raise CheckpointContractIntegrityError(
                    f"Training state is missing required trainer metadata: {state_file}"
                )
            recorded_hash = payload.get("checkpoint_contract_sha256")
            if recorded_hash != actual_contract.contract_sha256:
                raise CheckpointContractIntegrityError(
                    "trainer_state.json checkpoint_contract_sha256 does not match "
                    f"checkpoint_contract.json: expected {actual_contract.contract_sha256}, "
                    f"got {recorded_hash!r}"
                )
        elif payload is not None and "checkpoint_contract_sha256" in payload:
            if actual_contract is None:
                raise CheckpointContractIntegrityError(
                    "trainer_state.json references a checkpoint contract, but "
                    f"{contract_file} is missing"
                )
            if payload["checkpoint_contract_sha256"] != actual_contract.contract_sha256:
                raise CheckpointContractIntegrityError(
                    "trainer_state.json checkpoint_contract_sha256 does not match "
                    "checkpoint_contract.json"
                )

        probe_path = Path(state_dir) / (
            f"resume_probe_rank_{self.accelerator.process_index:03d}.json"
        )
        expected_probe = None
        if getattr(self, "resume_validation_enabled", False):
            for rank in range(self.accelerator.num_processes):
                rank_probe_path = Path(state_dir) / f"resume_probe_rank_{rank:03d}.json"
                if not rank_probe_path.is_file():
                    raise CheckpointContractIntegrityError(
                        f"Training state is missing rank resume probe: {rank_probe_path}"
                    )
                rank_probe = json.loads(rank_probe_path.read_text(encoding="utf-8"))
                if rank_probe.get("rank") != rank:
                    raise CheckpointContractIntegrityError(
                        f"Resume probe rank mismatch in {rank_probe_path}"
                    )
                if rank_probe.get("world_size") != int(
                    self.accelerator.num_processes
                ):
                    raise CheckpointContractIntegrityError(
                        "Resume validation requires the same distributed world size: "
                        f"checkpoint={rank_probe.get('world_size')} runtime="
                        f"{self.accelerator.num_processes}"
                    )
                if (
                    actual_contract is not None
                    and rank_probe.get("checkpoint_contract_sha256")
                    != actual_contract.contract_sha256
                ):
                    raise CheckpointContractIntegrityError(
                        "Resume probe checkpoint_contract_sha256 does not match "
                        f"checkpoint_contract.json in {rank_probe_path}: expected "
                        f"{actual_contract.contract_sha256}, got "
                        f"{rank_probe.get('checkpoint_contract_sha256')!r}"
                    )
            expected_probe = json.loads(probe_path.read_text(encoding="utf-8"))

        # Never mutate model/optimizer/scheduler state until all checkpoint metadata validates.
        self.accelerator.load_state(input_dir=state_dir)
        if payload is not None:
            self.global_step = int(payload["global_step"])

            if getattr(self, "resume_lr_override", False):
                override_value = float(getattr(self, "resume_lr_override_value", 1e-5))
                if override_value <= 0:
                    raise ValueError("resume_lr_override_value must be positive")
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = override_value
                    param_group["initial_lr"] = override_value
                replacement = self._build_scheduler(
                    scheduler_type="constant",
                    total_train_steps=self.max_steps,
                    warmup_steps=0,
                )
                if hasattr(self.scheduler, "scheduler"):
                    self.scheduler.scheduler = replacement
                else:
                    self.scheduler = replacement
                logger.warning(
                    "Overriding resumed learning rate after full-state restore: "
                    "scheduler=constant lr=%g at step=%d",
                    override_value,
                    self.global_step,
                )

            if "epoch" in payload and "batch_in_epoch" in payload:
                self.epoch = int(payload["epoch"])
                self.batch_in_epoch = int(payload["batch_in_epoch"])
                self.train_sampler.set_epoch_offset(self.epoch)
                self.train_sampler.set_resume_batch_offset(self.batch_in_epoch)
                logger.info(
                    "Restored dataloader progress: epoch=%d batch_in_epoch=%d sample_offset=%d",
                    self.epoch,
                    self.batch_in_epoch,
                    self.batch_in_epoch * self.batch_size * self.accelerator.num_processes,
                )
            else:
                self.epoch = 0
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                logger.warning(
                    "State file does not contain `epoch`/`batch_in_epoch`; "
                    "optimizer/scheduler were restored, but dataloader progress resume is skipped."
                )
            if expected_probe is not None:
                actual_probe = self._resume_probe_payload()
                mismatches = [
                    key
                    for key in expected_probe
                    if key != "checkpoint_contract_sha256"
                    and actual_probe.get(key) != expected_probe.get(key)
                ]
                mismatch_flag = torch.tensor(
                    [bool(mismatches)],
                    device=self.accelerator.device,
                    dtype=torch.int64,
                )
                gathered_mismatches = self.accelerator.gather(mismatch_flag)
                if bool(gathered_mismatches.any().item()):
                    raise CheckpointContractIntegrityError(
                        "Full-state resume probe mismatch after Accelerate restore; "
                        f"local mismatches on rank {self.accelerator.process_index}: "
                        + (", ".join(mismatches) if mismatches else "none")
                    )
                self._resume_validation_start_probe = actual_probe
                self._resume_validation_target_step = (
                    self.global_step + self.resume_validation_continuation_steps
                )
                logger.info(
                    "Validated optimizer/scheduler/global-step/dataloader/sampler "
                    "resume probe at step=%d on rank=%d",
                    self.global_step,
                    self.accelerator.process_index,
                )
            self.accelerator.wait_for_everyone()
            return

        match = re.search(r"step[_-](\d+)$", str(state_dir).rstrip("/"))
        if match:
            self.global_step = int(match.group(1))
        else:
            self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0
        self.train_sampler.clear_resume_batch_offset()
        self.accelerator.wait_for_everyone()
        logger.info("Loaded accelerate training state from %s at step=%d", state_dir, self.global_step)
        logger.warning(
            "State file `%s` is missing; dataloader progress resume is skipped.",
            state_file,
        )

    def _save_checkpoint_after_step(self) -> bool:
        """Save at most once for this optimizer step and report whether training is done."""
        terminal = self.global_step >= self.max_steps
        # Terminal checkpoints are handled once by the training epilogue so
        # final_checkpoint_mode and training-summary are always honored.
        periodic = (
            not terminal
            and self.save_every > 0
            and self.global_step % self.save_every == 0
        )
        if periodic:
            ckpt_info = self.save_checkpoint()
            if self.accelerator.is_main_process:
                logger.info(
                    "[ckpt] checkpoint saved step=%d weights=%s state=%s",
                    self.global_step,
                    ckpt_info["weights_path"],
                    ckpt_info["state_path"],
                )

        permanent_interval = getattr(
            self, "permanent_weights_every_n_steps", None
        )
        permanent_due = (
            not terminal
            and permanent_interval is not None
            and self.global_step % permanent_interval == 0
        )
        if permanent_due:
            step_tag = f"step_{self.global_step:06d}"
            self.accelerator.wait_for_everyone()
            permanent_path = None
            if self.accelerator.is_main_process:
                permanent_path = self._save_permanent_weights_checkpoint(step_tag)
                logger.info(
                    "[ckpt] permanent weights-only checkpoint saved step=%d weights=%s",
                    self.global_step,
                    permanent_path,
                )
            self.accelerator.wait_for_everyone()
        return False

    def _write_training_summary(self) -> None:
        if not self.accelerator.is_main_process:
            return
        _atomic_write_json(
            os.path.join(self.output_dir, "training-summary.json"),
            {
                "initialization_fingerprint_sha256": self.initialization_fingerprint_sha256,
                "fixed_validation_window_set_sha256": self._fixed_validation_window_set_sha256,
                "fixed_validation_seed": int(self.fixed_validation_seed),
                "optimizer_steps": int(self.optimizer_steps_observed),
                "clip_events": int(self.clip_events),
                "clip_event_rate": self.clip_events
                / max(1, self.optimizer_steps_observed),
                "optimizer_skipped": int(self.optimizer_skipped),
                "train_loss_last": self._summary_losses[-1]
                if self._summary_losses
                else {},
                "train_loss_mean": {
                    key: float(np.mean([row[key] for row in self._summary_losses]))
                    for key in (
                        self._summary_losses[0] if self._summary_losses else {}
                    )
                },
                "preclip_grad_norm_mean": float(np.mean(self._summary_grad_norms))
                if self._summary_grad_norms
                else 0.0,
                "preclip_grad_norm_max": float(np.max(self._summary_grad_norms))
                if self._summary_grad_norms
                else 0.0,
                "samples_per_sec": self._summary_samples
                / max(time.perf_counter() - self.run_start_time, 1e-6),
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated())
                if torch.cuda.is_available()
                else 0,
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved())
                if torch.cuda.is_available()
                else 0,
            },
        )

    def _write_training_complete_marker(self, ckpt_info: dict) -> Path | None:
        """Publish natural completion only after the final checkpoint and summary."""
        if not self.accelerator.is_main_process:
            return None
        if getattr(self, "checkpoint_contract", None) is None:
            return None
        run_manifest = self.checkpoint_contract.provenance.get("run_manifest")
        if not isinstance(run_manifest, dict):
            logger.warning(
                "Skipping training completion marker because the checkpoint contract "
                "is not bound to an immutable run manifest."
            )
            return None
        run_manifest_sha256 = run_manifest.get("sha256")
        run_id = run_manifest.get("run_id")
        if (
            re.fullmatch(r"[0-9a-f]{64}", str(run_manifest_sha256)) is None
            or not isinstance(run_id, str)
            or not run_id
        ):
            raise RuntimeError("Checkpoint contract has invalid run-manifest provenance.")
        if self.final_checkpoint_mode != "full":
            return None
        if int(self.global_step) != int(self.max_steps):
            raise RuntimeError(
                "Refusing to mark incomplete training as complete: "
                f"global_step={self.global_step}, max_steps={self.max_steps}."
            )

        state_path = Path(ckpt_info.get("state_path") or "")
        weights_path = Path(ckpt_info.get("weights_path") or "")
        expected_tag = f"step_{self.global_step:06d}"
        if state_path.name != expected_tag or weights_path.name != f"{expected_tag}.pt":
            raise RuntimeError(
                "Final checkpoint paths do not match the terminal global step: "
                f"state={state_path}, weights={weights_path}."
            )
        checkpoint = _read_checkpoint_complete_marker(state_path, weights_path)
        if checkpoint is None:
            raise RuntimeError(
                f"Final checkpoint is not fully committed: {state_path}"
            )
        contract_sha256 = self.checkpoint_contract.contract_sha256
        if checkpoint.get("checkpoint_contract_sha256") != contract_sha256:
            raise RuntimeError(
                "Final checkpoint contract does not match the active training contract."
            )
        checkpoint_marker_path = state_path / _CHECKPOINT_COMPLETE_MARKER
        summary_path = Path(self.output_dir) / "training-summary.json"
        if not summary_path.is_file():
            raise RuntimeError(
                f"Cannot mark training complete before the summary exists: {summary_path}"
            )

        marker_path = Path(self.output_dir) / _TRAINING_COMPLETE_MARKER
        _atomic_write_json(
            marker_path,
            {
                "schema_version": _TRAINING_COMPLETE_SCHEMA_VERSION,
                "status": "completed",
                "global_step": int(self.global_step),
                "max_steps": int(self.max_steps),
                "world_size": int(self.accelerator.num_processes),
                "state_directory": state_path.name,
                "weights_file": weights_path.name,
                "checkpoint_complete_sha256": _sha256_file(
                    checkpoint_marker_path
                ),
                "checkpoint_contract_sha256": contract_sha256,
                "run_manifest_sha256": run_manifest_sha256,
                "run_id": run_id,
                "training_summary_sha256": _sha256_file(summary_path),
                "completed_at_unix_ns": time.time_ns(),
            },
        )
        return marker_path

    def _revoke_training_complete_marker(self) -> None:
        """Invalidate a previous success record before a real Trainer attempt."""
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            marker_path = Path(self.output_dir) / _TRAINING_COMPLETE_MARKER
            try:
                marker_path.unlink()
            except FileNotFoundError:
                pass
            else:
                directory_fd = os.open(
                    marker_path.parent, os.O_RDONLY | os.O_DIRECTORY
                )
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        self.accelerator.wait_for_everyone()

    def _finish_resume_validation_if_ready(self) -> bool:
        if not self.resume_validation_enabled:
            return False
        if self._resume_validation_start_probe is None:
            raise RuntimeError("Resume validation was enabled but no start probe was loaded.")
        if self.global_step < self._resume_validation_target_step:
            return False

        start = self._resume_validation_start_probe
        final = self._resume_probe_payload()
        expected_micro_batches = (
            self.resume_validation_continuation_steps
            * self.gradient_accumulation_steps
        )
        checks = {
            "global_step_continuity": self.global_step
            == self._resume_validation_target_step,
            "epoch_unchanged": self.epoch == start["epoch"],
            "dataloader_batch_continuity": self.batch_in_epoch
            == start["batch_in_epoch"] + expected_micro_batches,
            "sampler_stream_advanced": final["dataloader"]["next_global_indices"]
            != start["dataloader"]["next_global_indices"],
            "optimizer_state_advanced": final["optimizer"]["sha256"]
            != start["optimizer"]["sha256"],
            "scheduler_state_advanced": final["scheduler"]["last_epoch"]
            == start["scheduler"]["last_epoch"]
            + self.resume_validation_continuation_steps,
            "optimizer_structure_stable": (
                final["optimizer"]["parameter_state_count"]
                == start["optimizer"]["parameter_state_count"]
                and final["optimizer"]["param_group_count"]
                == start["optimizer"]["param_group_count"]
            ),
            "checkpoint_contract_stable": final["checkpoint_contract_sha256"]
            == start["checkpoint_contract_sha256"],
        }
        local_report = {
            "rank": int(self.accelerator.process_index),
            "status": "passed" if all(checks.values()) else "failed",
            "checks": checks,
            "start": start,
            "final": final,
            "continuation_steps": self.resume_validation_continuation_steps,
            "expected_consumed_micro_batches": expected_micro_batches,
        }
        report_path = Path(self.resume_validation_report_path)
        rank_report_path = report_path.with_name(
            f"{report_path.stem}.rank-{self.accelerator.process_index:03d}"
            f"{report_path.suffix or '.json'}"
        )
        _atomic_write_json(rank_report_path, local_report)
        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            rank_reports = []
            for rank in range(self.accelerator.num_processes):
                path = report_path.with_name(
                    f"{report_path.stem}.rank-{rank:03d}{report_path.suffix or '.json'}"
                )
                if not path.is_file():
                    raise RuntimeError(f"Missing resume validation rank report: {path}")
                rank_reports.append(json.loads(path.read_text(encoding="utf-8")))
            report = {
                "schema_version": 1,
                "status": (
                    "passed"
                    if all(item["status"] == "passed" for item in rank_reports)
                    else "failed"
                ),
                "world_size": int(self.accelerator.num_processes),
                "checkpoint_global_step": int(start["global_step"]),
                "final_global_step": int(self.global_step),
                "continuation_steps": self.resume_validation_continuation_steps,
                "ranks": rank_reports,
            }
            _atomic_write_json(report_path, report)
        self.accelerator.wait_for_everyone()
        if not all(checks.values()):
            failed = [name for name, passed in checks.items() if not passed]
            raise RuntimeError(
                "Resume continuation validation failed on rank "
                f"{self.accelerator.process_index}: {', '.join(failed)}"
            )
        logger.info(
            "Full-state resume validation passed through step=%d on rank=%d",
            self.global_step,
            self.accelerator.process_index,
        )
        return True

    def train(self):
        try:
            ckpt_info = self._train_impl()
        finally:
            self._close_loggers()
        is_full_checkpoint = (
            isinstance(ckpt_info, dict)
            and ckpt_info.get("state_path") is not None
            and ckpt_info.get("weights_path") is not None
        )
        if is_full_checkpoint:
            if self.accelerator.is_main_process:
                self._write_training_complete_marker(ckpt_info)
            self.accelerator.wait_for_everyone()
        return ckpt_info

    def _train_impl(self):
        self._set_dit_only_train_mode()

        unwrapped_model = self.accelerator.unwrap_model(self.model)

        if self.max_steps is None:
            raise ValueError("`max_steps` must be set before entering the while-step training loop.")

        self._revoke_training_complete_marker()

        logger.info("Starting training with max_steps=%d.", self.max_steps)
        data_iter = iter(self.train_loader)
        self.run_start_step = self.global_step
        self.run_start_time = time.perf_counter()

        if self.fixed_validation_enabled and fixed_validation_due(
            self.global_step,
            explicit_steps=self.fixed_validation_steps,
            every_steps=self.fixed_validation_every_steps,
        ):
            fixed_validation_started_at = time.perf_counter()
            fixed_metrics = self.evaluate_fixed_validation_loss(
                step=self.global_step,
                tasks=self.fixed_validation_tasks,
                anchors_per_task=self.fixed_validation_anchors_per_task,
                stride=self.fixed_validation_stride,
            )
            fixed_validation_seconds = (
                time.perf_counter() - fixed_validation_started_at
            )
            self._persist_fixed_validation_metrics(fixed_metrics)
            self._log_fixed_validation_metrics(
                fixed_metrics,
                duration_seconds=fixed_validation_seconds,
            )

        while self.global_step < self.max_steps:
            self._profiling_prepare_micro_step()
            data_wait_started_at = time.perf_counter()
            try:
                sample = next(data_iter)
                self.batch_in_epoch += 1
            except StopIteration:
                self._profiling_record_data_wait(data_wait_started_at)
                self.epoch += 1
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                data_iter = iter(self.train_loader)
                continue
            self._profiling_record_data_wait(data_wait_started_at)

            with self.accelerator.accumulate(self.model):
                train_model = self.model if hasattr(self.model, "training_loss") else self.accelerator.unwrap_model(self.model)

                self._cuda_synchronize()
                forward_started_at = time.perf_counter()
                with self.accelerator.autocast():
                    loss, loss_dict = train_model.training_loss(sample)
                self._gradient_gate_observe_loss(loss)
                self._profiling_record_forward(forward_started_at)

                self._cuda_synchronize()
                backward_started_at = time.perf_counter()
                self.accelerator.backward(loss)
                self._profiling_record_backward(backward_started_at)

                if self.accelerator.sync_gradients:
                    self._validate_gradient_gate()
                    self._cuda_synchronize()
                    optimizer_started_at = time.perf_counter()
                    grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer_steps_observed += 1
                    if self.max_grad_norm > 0 and float(grad_norm) > self.max_grad_norm:
                        self.clip_events += 1
                    self.optimizer.step()
                    if not self.accelerator.optimizer_step_was_skipped:
                        self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1
                    post_optimizer_started_at = self._profiling_record_optimizer(
                        optimizer_started_at
                    )
                    global_loss = float(
                        self.accelerator.gather(loss.detach().float().reshape(1)).mean().item()
                    )
                    global_loss_metrics = {}
                    for key, value in loss_dict.items():
                        metric_tensor = torch.tensor(float(value), device=loss.device, dtype=torch.float32).reshape(1)
                        global_loss_metrics[key] = float(
                            self.accelerator.gather(metric_tensor).mean().item()
                        )
                    grad_norm_tensor = torch.tensor(grad_norm, device=loss.device, dtype=torch.float32)
                    global_grad_norm = float(self.accelerator.gather(grad_norm_tensor).mean().item())
                    self._summary_grad_norms.append(global_grad_norm)
                    self._summary_losses.append({key: float(value) for key, value in global_loss_metrics.items()} | {"loss_total": global_loss})
                    if self.accelerator.optimizer_step_was_skipped:
                        self.optimizer_skipped += 1
                    self._summary_samples += self.batch_size * self.accelerator.num_processes * self.gradient_accumulation_steps

                    current_lr = float(self.optimizer.param_groups[0]["lr"])

                    if self.log_every > 0 and self.global_step % self.log_every == 0 and self.accelerator.is_main_process:
                        eta_str, steps_per_sec = self._estimate_eta()
                        description = "[train] epoch=%d step=%d/%d loss=%.4f " % (
                            self.epoch,
                            self.global_step,
                            self.max_steps,
                            global_loss,
                        )
                        if global_loss_metrics:
                            detail_str = " ".join([f"{k}={v:.4f}" for k, v in sorted(global_loss_metrics.items())])
                            description += detail_str + " "
                        description += "lr=%.2e speed=%.2f step/s, %.2f samples/s eta=%s" % (
                            current_lr,
                            steps_per_sec,
                            steps_per_sec
                            * self.batch_size
                            * self.accelerator.num_processes
                            * self.gradient_accumulation_steps,
                            eta_str,
                        )
                        logger.info(description)

                        self._log_training_metrics(
                            loss_total=global_loss,
                            loss_metrics=global_loss_metrics,
                            learning_rate=current_lr,
                            grad_norm=global_grad_norm,
                            clip_event_rate=(
                                self.clip_events
                                / max(1, self.optimizer_steps_observed)
                            ),
                            steps_per_second=steps_per_sec,
                            samples_per_second=(
                                steps_per_sec
                                * self.batch_size
                                * self.accelerator.num_processes
                                * self.gradient_accumulation_steps
                            ),
                            optimizer_skipped=self.optimizer_skipped,
                        )

                    stop_after_profile = self._profiling_finish_optimizer_step(
                        post_optimizer_started_at,
                        optimizer_skipped=self.accelerator.optimizer_step_was_skipped,
                    )
                    if stop_after_profile:
                        if self.accelerator.is_main_process:
                            logger.info(
                                "Profiling complete at step=%d; stopping without forcing a checkpoint.",
                                self.global_step,
                            )
                        return

                    if (
                        self.eval_every > 0
                        and self.val_dataset is not None
                        and self.global_step % self.eval_every == 0
                    ):
                        metrics = self.evaluate()
                        self.accelerator.wait_for_everyone()
                        if metrics is not None and self.accelerator.is_main_process:
                            description = "[eval] step=%d val_loss=%.4f infer_psnr=%.4f infer_ssim=%.4f" % (
                                self.global_step,
                                metrics["val_loss"],
                                metrics["psnr_rd"],
                                metrics["ssim_rd"],
                            )
                            if "action_l2" in metrics:
                                description += " action_l2=%.4f" % metrics["action_l2"]
                            if "action_l1" in metrics:
                                description += " action_l1=%.4f" % metrics["action_l1"]
                            logger.info(description)
                            eval_payload = {
                                "eval/val_loss": float(metrics["val_loss"]),
                                "eval/psnr_rg": float(metrics["psnr_rg"]),
                                "eval/ssim_rg": float(metrics["ssim_rg"]),
                                "eval/psnr_rd": float(metrics["psnr_rd"]),
                                "eval/ssim_rd": float(metrics["ssim_rd"]),
                                "eval/psnr_dg": float(metrics["psnr_dg"]),
                                "eval/ssim_dg": float(metrics["ssim_dg"]),
                            }
                            if "action_l2" in metrics:
                                eval_payload["eval/action_l2"] = float(metrics["action_l2"])
                            if "action_l1" in metrics:
                                eval_payload["eval/action_l1"] = float(metrics["action_l1"])
                            self._wandb_log(eval_payload)

                    if self.fixed_validation_enabled and fixed_validation_due(
                        self.global_step,
                        explicit_steps=self.fixed_validation_steps,
                        every_steps=self.fixed_validation_every_steps,
                    ):
                        fixed_validation_started_at = time.perf_counter()
                        fixed_metrics = self.evaluate_fixed_validation_loss(
                            step=self.global_step,
                            tasks=self.fixed_validation_tasks,
                            anchors_per_task=self.fixed_validation_anchors_per_task,
                            stride=self.fixed_validation_stride,
                        )
                        fixed_validation_seconds = (
                            time.perf_counter() - fixed_validation_started_at
                        )
                        if self.accelerator.is_main_process:
                            self._persist_fixed_validation_metrics(fixed_metrics)
                            report_path = Path(self.fixed_validation_report_dir) / f"step-{self.global_step:06d}.json"
                            logger.info(
                                "[fixed-eval] step=%d total=%.6f action=%.6f video=%.6f windows=%d report=%s",
                                self.global_step,
                                fixed_metrics["loss_total"],
                                fixed_metrics["loss_action"],
                                fixed_metrics["loss_video"],
                                fixed_metrics["num_windows"],
                                report_path,
                            )
                            self._log_fixed_validation_metrics(
                                fixed_metrics,
                                duration_seconds=fixed_validation_seconds,
                            )

                    # Keep expensive joint inference after the deterministic
                    # loss pass at each epoch boundary. All ranks participate.
                    self._run_qualitative_validation_if_due()

                    if self._finish_resume_validation_if_ready():
                        return

                    if self._save_checkpoint_after_step():
                        self._write_training_summary()
                        return

        ckpt_info = self.save_final_checkpoint()
        if self.accelerator.is_main_process:
            self._write_training_summary()
            logger.info(
                "[done] training finished step=%d weights=%s state=%s",
                self.global_step,
                ckpt_info["weights_path"],
                ckpt_info["state_path"],
            )
        self.accelerator.wait_for_everyone()
        return ckpt_info
