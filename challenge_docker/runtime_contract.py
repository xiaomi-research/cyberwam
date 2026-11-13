#!/usr/bin/env python3
# Copyright (c) 2026 Xiaomi Corporation

"""Small, dependency-free helpers for mounted FastWAM model bundles."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


def canonical_json(value: Any) -> str:
    """Match fastwam.utils.checkpoint_contract canonical JSON serialization."""
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_model_config(root: Path) -> tuple[Path | None, dict[str, Any] | None]:
    """Load the optional task and batch guard configuration."""
    path = root / "model_config.json"
    if not path.exists():
        return None, None
    if not path.is_file():
        raise ValueError(f"model config path is not a file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read model config {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"model config must be an object: {path}")
    if value.get("schema") != "robot_policy_task_guard.v1":
        raise ValueError(f"unsupported model config schema in {path}")
    allowed = value.get("allowed_task_ids")
    if not isinstance(allowed, list) or not allowed:
        raise ValueError(f"model config allowed_task_ids must be non-empty: {path}")
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in allowed):
        raise ValueError(f"model config task ids must be non-negative integers: {path}")
    if len(set(allowed)) != len(allowed):
        raise ValueError(f"model config task ids must be unique: {path}")
    batch_size = value.get("batch_size", 1)
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError(f"model config batch_size must be positive: {path}")
    model_name = value.get("model_name", root.name)
    if not isinstance(model_name, str) or not model_name:
        raise ValueError(f"model config model_name must be non-empty: {path}")
    normalized = dict(value)
    normalized.update(
        model_name=model_name,
        allowed_task_ids=[int(item) for item in allowed],
        batch_size=int(batch_size),
    )
    return path, normalized


def _contract_payload(contract: dict[str, Any]) -> dict[str, Any]:
    required = {
        "kind",
        "schema_version",
        "compatibility",
        "provenance",
        "compatibility_sha256",
        "contract_sha256",
    }
    missing = sorted(required - set(contract))
    unexpected = sorted(set(contract) - required)
    if missing or unexpected:
        raise ValueError(
            "Malformed checkpoint contract keys: "
            f"missing={missing}, unexpected={unexpected}"
        )
    if contract["kind"] != "fastwam_checkpoint_contract":
        raise ValueError(f"Unsupported checkpoint contract kind: {contract['kind']!r}")
    if not isinstance(contract["compatibility"], dict):
        raise ValueError("checkpoint contract compatibility must be an object")
    if not isinstance(contract["provenance"], dict):
        raise ValueError("checkpoint contract provenance must be an object")
    return contract


def load_contract(path: Path) -> dict[str, Any]:
    try:
        contract = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read checkpoint contract {path}: {exc}") from exc
    if not isinstance(contract, dict):
        raise ValueError(f"checkpoint contract must be an object: {path}")
    _contract_payload(contract)
    compatibility_sha256 = sha256_bytes(
        canonical_json(contract["compatibility"]).encode("utf-8")
    )
    unsigned = {
        "kind": contract["kind"],
        "schema_version": contract["schema_version"],
        "compatibility": contract["compatibility"],
        "provenance": contract["provenance"],
    }
    contract_sha256 = sha256_bytes(canonical_json(unsigned).encode("utf-8"))
    if contract["compatibility_sha256"] != compatibility_sha256:
        raise ValueError(
            "checkpoint compatibility_sha256 does not match its compatibility payload: "
            f"expected {compatibility_sha256}, got {contract['compatibility_sha256']}"
        )
    if contract["contract_sha256"] != contract_sha256:
        raise ValueError(
            "checkpoint contract_sha256 does not match its contract payload: "
            f"expected {contract_sha256}, got {contract['contract_sha256']}"
        )
    return contract


def derive_hydra_task(contract: dict[str, Any]) -> str:
    """Derive the supported evaluation config from compatibility metadata."""
    compatibility = contract["compatibility"]
    model = compatibility.get("model")
    if not isinstance(model, dict):
        raise ValueError("checkpoint contract is missing compatibility.model")
    target = model.get("target")
    if target == "fastwam.runtime.create_fastwam_joint":
        return "b1k_2026_joint_sharedt_task_finetune_32gpu"
    if target != "fastwam.runtime.create_fastwam_flex":
        raise ValueError(f"unsupported checkpoint model target: {target!r}")

    schedulers = compatibility.get("scheduler")
    if not isinstance(schedulers, dict):
        raise ValueError("checkpoint contract is missing compatibility.scheduler")
    action = schedulers.get("action")
    video = schedulers.get("video")
    if not isinstance(action, dict) or not isinstance(video, dict):
        raise ValueError("checkpoint contract has incomplete action/video schedulers")
    try:
        action_shift = float(action["infer_shift"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("checkpoint contract is missing action.infer_shift") from exc
    shared = video.get("shared_train_timestep")
    if math.isclose(action_shift, 1.0, rel_tol=0.0, abs_tol=1e-9) and shared in (None, False):
        return "b1k_2026_fastwam_flex_3cam_direct384x256_mb32_gc_on_128gpu"
    if math.isclose(action_shift, 5.0, rel_tol=0.0, abs_tol=1e-9) and shared is True:
        return "b1k_2026_fastwam_flex_statedrop10_sharedt_ashift5_256gpu"
    raise ValueError(
        "unsupported Flex scheduler combination: "
        f"action.infer_shift={action_shift!r}, video.shared_train_timestep={shared!r}"
    )


def _pick_file(root: Path, exact_name: str, *, suffix: str | None = None) -> Path:
    exact = root / exact_name
    if exact.is_file():
        return exact
    if suffix is None:
        raise ValueError(f"missing required file: {exact}")
    candidates = sorted(
        path for path in root.iterdir() if path.is_file() and path.name.endswith(suffix)
    )
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        names = ", ".join(path.name for path in candidates)
        raise ValueError(
            f"missing {exact_name}; found multiple {suffix} files in {root}: {names}"
        )
    raise ValueError(f"missing required file: {exact}")


def resolve_bundle_files(root: Path) -> tuple[Path, Path, Path]:
    root = root.expanduser()
    if not root.is_dir():
        raise ValueError(f"model bundle directory does not exist: {root}")
    checkpoint = _pick_file(root, "checkpoint.pt", suffix=".pt")
    contract = _pick_file(root, "checkpoint_contract.json")
    stats = _pick_file(root, "norm_stats.json")
    return checkpoint, contract, stats


def validate_bundle(root: Path) -> dict[str, Any]:
    checkpoint, contract_path, stats_path = resolve_bundle_files(root)
    if checkpoint.stat().st_size <= 0:
        raise ValueError(f"checkpoint is empty: {checkpoint}")
    contract = load_contract(contract_path)
    compatibility = contract["compatibility"]
    try:
        expected_stats_sha256 = compatibility["data"]["normalization"]["stats_sha256"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "checkpoint contract is missing compatibility.data.normalization.stats_sha256"
        ) from exc
    actual_stats_sha256 = sha256_file(stats_path)
    if expected_stats_sha256 != actual_stats_sha256:
        raise ValueError(
            "norm_stats.json hash does not match checkpoint contract: "
            f"expected {expected_stats_sha256}, got {actual_stats_sha256}"
        )
    model = compatibility.get("model")
    if not isinstance(model, dict) or not isinstance(model.get("target"), str):
        raise ValueError("checkpoint contract is missing compatibility.model.target")
    model_config_path, model_config = load_model_config(root)
    return {
        "root": root,
        "checkpoint": checkpoint,
        "contract": contract_path,
        "stats": stats_path,
        "contract_sha256": contract["contract_sha256"],
        "stats_sha256": actual_stats_sha256,
        "hydra_task": derive_hydra_task(contract),
        "model_target": model["target"],
        "model_config": model_config,
        "model_config_path": model_config_path,
    }
