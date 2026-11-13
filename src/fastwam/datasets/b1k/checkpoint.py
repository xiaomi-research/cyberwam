# Copyright (c) 2026 Xiaomi Corporation

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from fastwam.utils.checkpoint_contract import CheckpointContract, canonical_json_sha256

from .contract import (
    ACTION_CONSTANT_DIM,
    ACTION_DIM,
    ACTION_GRIPPER_DIMS,
    ACTION_LABELS,
    PROPRIO_DIM,
    PROPRIO_LABELS,
)
from .text import B1K_PROMPT_TEMPLATE


B1K_DATASET_TARGET = "fastwam.datasets.b1k.dataset.B1KRobotVideoDataset"
B1K_DIRECT_VIDEO_DATASET_TARGET = (
    "fastwam.datasets.b1k.video_dataset.B1KDirectVideoDataset"
)
B1K_DATASET_TARGETS = {B1K_DATASET_TARGET, B1K_DIRECT_VIDEO_DATASET_TARGET}
_MODEL_METADATA_SUFFIXES = (".json", ".yaml", ".yml", ".txt", ".md", ".index.json")
_MODEL_WEIGHT_SUFFIXES = (".pt", ".pth", ".bin", ".safetensors")


class B1KCheckpointContractError(ValueError):
    """Raised when a strict B1K checkpoint contract cannot be constructed."""


def bind_optimizer_scheduler_contract(
    contract: CheckpointContract | None,
    *,
    learning_rate: float,
    min_learning_rate: float,
    lr_scheduler_type: str,
    lr_warmup_steps: int,
    max_steps: int,
    weight_decay: float,
    betas: tuple[float, float],
    epsilon: float,
    amsgrad: bool,
    max_grad_norm: float,
    gradient_accumulation_steps: int,
) -> CheckpointContract | None:
    """Bind resolved Trainer optimizer/scheduler semantics to a B1K contract.

    The dataset-level contract is built before Accelerate knows the effective
    world size.  Consequently, total optimizer steps and the resolved warmup
    cannot be made authoritative until the Trainer has built its loader.  This
    function returns a new immutable contract instead of mutating the base
    contract in place.
    """
    if contract is None:
        return None
    if not isinstance(contract, CheckpointContract):
        raise TypeError(
            "contract must be a CheckpointContract or None, "
            f"got {type(contract).__name__}"
        )
    normalized_scheduler_type = str(lr_scheduler_type).strip().lower()
    if normalized_scheduler_type not in {"constant", "cosine"}:
        raise B1KCheckpointContractError(
            f"Unsupported lr_scheduler_type: {lr_scheduler_type!r}."
        )
    if len(betas) != 2:
        raise B1KCheckpointContractError("AdamW betas must contain exactly two values.")
    if int(lr_warmup_steps) < 0 or int(max_steps) <= 0:
        raise B1KCheckpointContractError(
            "Resolved lr_warmup_steps must be non-negative and max_steps must be positive."
        )

    compatibility = _json_copy(contract.compatibility)
    scheduler = _required_mapping(compatibility, "scheduler")
    scheduler["optimizer_lr"] = {
        "optimizer_type": "AdamW",
        "learning_rate": float(learning_rate),
        "min_learning_rate": float(min_learning_rate),
        "lr_scheduler_type": normalized_scheduler_type,
        "lr_warmup_steps": int(lr_warmup_steps),
        "max_steps": int(max_steps),
        "weight_decay": float(weight_decay),
        "betas": [float(betas[0]), float(betas[1])],
        "epsilon": float(epsilon),
        "amsgrad": bool(amsgrad),
        "max_grad_norm": float(max_grad_norm),
        "gradient_accumulation_steps": int(gradient_accumulation_steps),
    }
    return CheckpointContract(
        compatibility=compatibility,
        provenance=contract.provenance,
        schema_version=contract.schema_version,
    )


def build_b1k_checkpoint_contract(
    cfg: DictConfig | Mapping[str, Any],
    dataset: Any,
    *,
    run_manifest_path: str | os.PathLike[str] | None = None,
) -> CheckpointContract:
    """Build the strict B1K checkpoint contract from resolved runtime inputs."""
    resolved = _resolved_config(cfg)
    train_cfg = _required_mapping(_required_mapping(resolved, "data"), "train")
    if train_cfg.get("_target_") not in B1K_DATASET_TARGETS:
        raise B1KCheckpointContractError(
            f"Expected a B1K train target in {sorted(B1K_DATASET_TARGETS)!r}, "
            f"got {train_cfg.get('_target_')!r}."
        )

    public_contract = getattr(dataset, "checkpoint_data_contract", None)
    if callable(public_contract):
        public_contract = public_contract()
    if not isinstance(public_contract, Mapping):
        raise B1KCheckpointContractError(
            "B1K dataset must expose a mapping-valued `checkpoint_data_contract`."
        )
    public_contract = _json_copy(public_contract)
    _validate_public_data_contract(public_contract)
    _validate_config_against_data_contract(resolved, public_contract)

    norm_path = _required_file(train_cfg.get("pretrained_norm_stats"), "normalization stats")
    model_cfg = _required_mapping(resolved, "model")
    action_dit_path = _required_file(
        model_cfg.get("action_dit_pretrained_path"), "ActionDiT initialization checkpoint"
    )
    manifest_path = _find_run_manifest(resolved, explicit=run_manifest_path)
    manifest = _load_run_manifest(manifest_path) if manifest_path is not None else None

    norm_sha256 = _sha256_file(norm_path)
    if manifest is not None:
        _validate_manifest_artifact(
            manifest,
            key="norm_stats",
            configured_path=norm_path,
            full_file_hash=True,
            allow_configured_descendant=False,
        )
    wan_fingerprint = _artifact_fingerprint(
        manifest,
        key="wan_model",
        fallback_path=_resolve_wan_model_path(str(model_cfg.get("model_id", ""))),
        full_file_hash=False,
    )
    action_dit_fingerprint = _artifact_fingerprint(
        manifest,
        key="action_dit",
        fallback_path=action_dit_path,
        full_file_hash=True,
    )

    data_contract = _json_copy(public_contract)
    data_contract["normalization"]["stats_sha256"] = norm_sha256
    mixture_contract = getattr(dataset, "mixture_contract", None)
    if mixture_contract is not None:
        data_contract["mixture"] = _json_copy(mixture_contract)
    compatibility = {
        "model": {
            "variant": _model_variant(str(model_cfg.get("_target_", ""))),
            "target": model_cfg.get("_target_"),
            "wan": {
                "id": model_cfg.get("model_id"),
                "fingerprint": wan_fingerprint,
            },
            "action_dit_initialization": {
                "id": action_dit_path.name,
                "fingerprint": action_dit_fingerprint,
            },
        },
        "data": data_contract,
        "scheduler": {
            "video": _json_copy(_required_mapping(model_cfg, "video_scheduler")),
            "action": _json_copy(_required_mapping(model_cfg, "action_scheduler")),
        },
        "loss": _json_copy(_required_mapping(model_cfg, "loss")),
    }
    if compatibility["model"]["variant"] == "fastwam_flex":
        dino_contract = {
            "model_name": model_cfg.get("dino_model_name"),
            "feature_dim": int(model_cfg.get("dino_dim", 0)),
            "prediction": "x0" if bool(model_cfg.get("dino_pred_x0", False)) else "velocity",
        }
        # Keep legacy 7x7 checkpoint contracts byte-for-byte compatible. Flex
        # records this only when it explicitly changes the DINO token geometry.
        if "dino_output_grid_size" in model_cfg:
            dino_contract["spatial_grid_per_camera"] = int(
                model_cfg["dino_output_grid_size"]
            )
            dino_contract["spatial_resample"] = "bilinear"
        if model_cfg.get("dino_pretrained_path") is not None:
            dino_contract["pretrained_path"] = model_cfg.get("dino_pretrained_path")
        compatibility["model"]["dino"] = dino_contract
    if compatibility["model"]["variant"] == "fastwam_flex":
        compatibility["model"]["routing"] = {
            "hbridge": {
                "enabled": bool(model_cfg.get("hbridge_enabled", True)),
                "bottom_ratio": float(model_cfg.get("hbridge_bottom_ratio", 0.0)),
                "top_ratio": float(model_cfg.get("hbridge_top_ratio", 0.0)),
            },
            "present_probabilities": _json_copy(
                _required_mapping(model_cfg, "present_probabilities")
            ),
            "output_probabilities": _json_copy(
                _required_mapping(model_cfg, "output_probabilities")
            ),
            "sampling_scope": str(
                model_cfg.get("routing_sampling_scope", "per_sample")
            ),
            "cross_modal_predict": _json_copy(
                _required_mapping(model_cfg, "cross_modal_predict")
            ),
            "inference": {
                "present_modalities": _json_copy(
                    _required_mapping(model_cfg, "inference_present_modalities")
                ),
                "output_modalities": _json_copy(
                    _required_mapping(model_cfg, "inference_output_modalities")
                ),
            },
        }
    for stream in ("dino", "pointmap"):
        scheduler_key = f"{stream}_scheduler"
        if scheduler_key in model_cfg:
            compatibility["scheduler"][stream] = _json_copy(
                _required_mapping(model_cfg, scheduler_key)
            )

    provenance: dict[str, Any] = {
        "resolved_hydra_config": resolved,
        "artifacts": {
            "norm_stats": {"path": str(norm_path), "sha256": norm_sha256},
            "wan_model": {"id": model_cfg.get("model_id"), "fingerprint": wan_fingerprint},
            "action_dit": {
                "path": str(action_dit_path),
                "fingerprint": action_dit_fingerprint,
            },
        },
        "git": _git_provenance(),
    }
    if manifest is not None:
        provenance["run_manifest"] = {
            "path": str(manifest_path),
            "sha256": _sha256_file(manifest_path),
            "run_id": manifest.get("run_id"),
            "code_snapshot": manifest.get("code_snapshot"),
            "git": manifest.get("git"),
        }
    return CheckpointContract(compatibility=compatibility, provenance=provenance)


def build_training_checkpoint_contract(
    cfg: DictConfig | Mapping[str, Any], dataset: Any
) -> CheckpointContract | None:
    """Return the strict B1K contract, while preserving every non-B1K path."""
    resolved = _resolved_config(cfg)
    data = resolved.get("data")
    train = data.get("train") if isinstance(data, Mapping) else None
    if not isinstance(train, Mapping) or train.get("_target_") not in B1K_DATASET_TARGETS:
        return None
    return build_b1k_checkpoint_contract(resolved, dataset)


def _resolved_config(cfg: DictConfig | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(cfg, DictConfig):
        value = OmegaConf.to_container(cfg, resolve=True)
    elif isinstance(cfg, Mapping):
        value = _json_copy(cfg)
    else:
        raise TypeError(f"cfg must be a DictConfig or mapping, got {type(cfg)!r}.")
    if not isinstance(value, dict):
        raise B1KCheckpointContractError("Resolved Hydra config must be a mapping.")
    return value


def _validate_config_against_data_contract(
    resolved: Mapping[str, Any], data_contract: Mapping[str, Any]
) -> None:
    train = _required_mapping(_required_mapping(resolved, "data"), "train")
    model = _required_mapping(resolved, "model")
    processor = _required_mapping(train, "processor")
    action = _required_mapping(data_contract, "action")
    sampling = _required_mapping(data_contract, "sampling")
    proprio = _required_mapping(data_contract, "proprio")
    video = _required_mapping(data_contract, "video")
    text = _required_mapping(data_contract, "text")
    checks = {
        "action.dim": (processor.get("action_output_dim"), action.get("dim")),
        "proprio.dim": (processor.get("proprio_output_dim"), proprio.get("dim")),
        "action.horizon": (int(train.get("num_frames", 0)) - 1, action.get("horizon")),
        "action.sample_stride": (train.get("global_sample_stride"), action.get("sample_stride")),
        "sampling.window_start_stride": (
            train.get("window_start_stride"),
            sampling.get("window_start_stride"),
        ),
        "video.canvas": (
            {
                "height": train.get("video_size", [None, None])[0],
                "width": train.get("video_size", [None, None])[1],
            },
            video.get("canvas"),
        ),
        "text.context_length": (train.get("context_len"), text.get("context_length")),
        "model.action_dim": (
            _required_mapping(model, "action_dit_config").get("action_dim"),
            action.get("dim"),
        ),
        "model.proprio_dim": (model.get("proprio_dim"), proprio.get("dim")),
    }
    mismatches = [
        f"{name}: config={configured!r}, dataset={published!r}"
        for name, (configured, published) in checks.items()
        if configured != published
    ]
    dataset_image_indices = list(
        range(
            0,
            int(action["horizon"]) + 1,
            int(train["action_video_freq_ratio"]),
        )
    )
    if video.get("dataset_image_indices") != dataset_image_indices:
        mismatches.append(
            "video.dataset_image_indices: "
            f"config-derived={dataset_image_indices!r}, "
            f"dataset={video.get('dataset_image_indices')!r}"
        )
    raw_frame_indices = [
        index * int(action["sample_stride"]) for index in dataset_image_indices
    ]
    if video.get("raw_frame_indices") != raw_frame_indices:
        mismatches.append(
            "video.raw_frame_indices: "
            f"config-derived={raw_frame_indices!r}, "
            f"dataset={video.get('raw_frame_indices')!r}"
        )
    if mismatches:
        raise B1KCheckpointContractError(
            "Resolved config disagrees with the B1K dataset public contract:\n  - "
            + "\n  - ".join(mismatches)
        )


def _validate_public_data_contract(data_contract: Mapping[str, Any]) -> None:
    action = _required_mapping(data_contract, "action")
    sampling = _required_mapping(data_contract, "sampling")
    proprio = _required_mapping(data_contract, "proprio")
    video = _required_mapping(data_contract, "video")
    text = _required_mapping(data_contract, "text")
    normalization = _required_mapping(data_contract, "normalization")
    expected = {
        "schema": "b1k_2026_data_contract.v2",
        "sampling.source_fps": 30,
        "sampling.tail_policy": "replicate_last_frame_with_padding_mask",
        "action.dim": ACTION_DIM,
        "action.labels": list(ACTION_LABELS),
        "action.gripper_dims": list(ACTION_GRIPPER_DIMS),
        "action.constant_dims": [ACTION_CONSTANT_DIM],
        "action.gripper_mask": [
            index in ACTION_GRIPPER_DIMS for index in range(ACTION_DIM)
        ],
        "action.constant_dim_mask": [
            index == ACTION_CONSTANT_DIM for index in range(ACTION_DIM)
        ],
        "proprio.dim": PROPRIO_DIM,
        "proprio.labels": list(PROPRIO_LABELS),
        "video.camera_order": ["head", "left_wrist", "right_wrist"],
        "text.prompt_template": B1K_PROMPT_TEMPLATE,
        "text.attention_mask": "real_token_mask",
        "normalization.mode": "mixed_z_score",
        "normalization.action_continuous_mode": (
            "global_per_dimension_z_score_unclamped"
        ),
        "normalization.proprio_continuous_mode": (
            "global_per_dimension_z_score_clamped"
        ),
        "normalization.proprio_clamp": [-5.0, 5.0],
        "normalization.stepwise": False,
        "normalization.gripper_action": "fixed_binary_minus_one_plus_one",
        "normalization.constant_action": "fixed_zero",
    }
    actual = {
        "schema": data_contract.get("schema"),
        "sampling.source_fps": sampling.get("source_fps"),
        "sampling.tail_policy": sampling.get("tail_policy"),
        "action.dim": action.get("dim"),
        "action.labels": action.get("labels"),
        "action.gripper_dims": action.get("gripper_dims"),
        "action.constant_dims": action.get("constant_dims"),
        "action.gripper_mask": action.get("gripper_mask"),
        "action.constant_dim_mask": action.get("constant_dim_mask"),
        "proprio.dim": proprio.get("dim"),
        "proprio.labels": proprio.get("labels"),
        "video.camera_order": video.get("camera_order"),
        "text.prompt_template": text.get("prompt_template"),
        "text.attention_mask": text.get("attention_mask"),
        "normalization.mode": normalization.get("mode"),
        "normalization.action_continuous_mode": normalization.get(
            "action_continuous_mode"
        ),
        "normalization.proprio_continuous_mode": normalization.get(
            "proprio_continuous_mode"
        ),
        "normalization.proprio_clamp": normalization.get("proprio_clamp"),
        "normalization.stepwise": normalization.get("stepwise"),
        "normalization.gripper_action": normalization.get("gripper_action"),
        "normalization.constant_action": normalization.get("constant_action"),
    }
    mismatches = [
        f"{key}: expected={value!r}, dataset={actual[key]!r}"
        for key, value in expected.items()
        if actual[key] != value
    ]
    stride = sampling.get("window_start_stride")
    if isinstance(stride, bool) or not isinstance(stride, int) or stride <= 0:
        mismatches.append(
            f"sampling.window_start_stride: expected=positive integer, dataset={stride!r}"
        )
    valid_stride = isinstance(stride, int) and not isinstance(stride, bool) and stride > 0
    expected_frequency = 30.0 / stride if valid_stride else None
    if sampling.get("anchor_frequency_hz") != expected_frequency:
        mismatches.append(
            "sampling.anchor_frequency_hz: "
            f"expected={expected_frequency!r}, dataset={sampling.get('anchor_frequency_hz')!r}"
        )
    sample_count = sampling.get("sample_count")
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count <= 0:
        mismatches.append(
            f"sampling.sample_count: expected=positive integer, dataset={sample_count!r}"
        )
    if mismatches:
        raise B1KCheckpointContractError(
            "Invalid B1K dataset public contract:\n  - " + "\n  - ".join(mismatches)
        )


def _model_variant(target: str) -> str:
    variants = {
        "fastwam.runtime.create_fastwam": "fastwam_uncond",
        "fastwam.runtime.create_fastwam_joint": "fastwam_joint",
        "fastwam.runtime.create_fastwam_flex": "fastwam_flex",
    }
    if target not in variants:
        raise B1KCheckpointContractError(
            "B1K 2026 supports the FastWAM uncond, joint, and flex targets, "
            f"got {target!r}."
        )
    return variants[target]


def _find_run_manifest(
    resolved: Mapping[str, Any], *, explicit: str | os.PathLike[str] | None
) -> Path | None:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit))
    if os.environ.get("FASTWAM_RUN_MANIFEST"):
        candidates.append(Path(os.environ["FASTWAM_RUN_MANIFEST"]))
    output_dir = resolved.get("output_dir")
    if isinstance(output_dir, str):
        output = Path(output_dir).expanduser()
        candidates.extend((output / "run.json", output.parent / "run.json"))
    candidates.append(Path.cwd() / "run.json")
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    if explicit is not None:
        raise B1KCheckpointContractError(f"Run manifest does not exist: {Path(explicit)}")
    return None


def _load_run_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise B1KCheckpointContractError(f"Cannot read run manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise B1KCheckpointContractError(f"Run manifest {path} must contain a JSON object.")
    return value


def _artifact_fingerprint(
    manifest: Mapping[str, Any] | None,
    *,
    key: str,
    fallback_path: Path,
    full_file_hash: bool,
) -> dict[str, Any]:
    if manifest is not None:
        return _validate_manifest_artifact(
            manifest,
            key=key,
            configured_path=fallback_path,
            full_file_hash=full_file_hash,
            allow_configured_descendant=key == "wan_model",
        )
    return _path_fingerprint(fallback_path, full_file_hash=full_file_hash)


def _validate_manifest_artifact(
    manifest: Mapping[str, Any],
    *,
    key: str,
    configured_path: Path,
    full_file_hash: bool,
    allow_configured_descendant: bool,
) -> dict[str, Any]:
    inputs = manifest.get("inputs")
    entry = inputs.get(key) if isinstance(inputs, Mapping) else None
    fingerprint = entry.get("fingerprint") if isinstance(entry, Mapping) else None
    artifact_path_value = entry.get("path") if isinstance(entry, Mapping) else None
    if not isinstance(fingerprint, Mapping) or not fingerprint.get("sha256"):
        raise B1KCheckpointContractError(
            f"Run manifest is missing inputs.{key}.fingerprint.sha256."
        )
    if not isinstance(artifact_path_value, str):
        raise B1KCheckpointContractError(f"Run manifest is missing inputs.{key}.path.")
    artifact_path = Path(artifact_path_value).expanduser().resolve(strict=True)
    configured_path = configured_path.resolve(strict=True)
    same_artifact = configured_path == artifact_path
    if allow_configured_descendant:
        same_artifact = same_artifact or artifact_path in configured_path.parents
    if not same_artifact:
        raise B1KCheckpointContractError(
            f"Configured {key} path {configured_path} disagrees with run manifest path "
            f"{artifact_path}."
        )
    actual = _path_fingerprint(artifact_path, full_file_hash=full_file_hash)
    if _json_copy(fingerprint) != actual:
        raise B1KCheckpointContractError(
            f"Run manifest fingerprint for {key} does not match artifact {artifact_path}."
        )
    return actual


def _resolve_wan_model_path(model_id: str) -> Path:
    if not model_id:
        raise B1KCheckpointContractError("model.model_id must be a non-empty string.")
    direct = Path(model_id).expanduser()
    if direct.exists():
        return direct.resolve()
    base = Path(os.environ.get("DIFFSYNTH_MODEL_BASE_PATH", "./checkpoints")).expanduser()
    candidate = base / model_id
    if not candidate.exists():
        raise B1KCheckpointContractError(
            f"Cannot fingerprint Wan model {model_id!r}; expected local path {candidate}."
        )
    return candidate.resolve()


def _path_fingerprint(path: Path, *, full_file_hash: bool) -> dict[str, Any]:
    path = Path(path).expanduser().resolve(strict=True)
    if path.is_file():
        entry: dict[str, Any] = {"path": path.name, "size_bytes": path.stat().st_size}
        if full_file_hash:
            entry["sha256"] = _sha256_file(path)
        return {
            "strategy": "full_file_content_v1" if full_file_hash else "file_inventory_v1",
            "sha256": canonical_json_sha256(entry),
            "files": [entry],
        }
    entries: list[dict[str, Any]] = []
    for candidate in sorted(item for item in path.rglob("*") if item.is_file()):
        if candidate.name.endswith(_MODEL_METADATA_SUFFIXES):
            entry = {
                "path": candidate.relative_to(path).as_posix(),
                "size_bytes": candidate.stat().st_size,
                "sha256": _sha256_file(candidate),
            }
        elif candidate.suffix.lower() in _MODEL_WEIGHT_SUFFIXES:
            entry = {
                "path": candidate.relative_to(path).as_posix(),
                "size_bytes": candidate.stat().st_size,
            }
        else:
            continue
        entries.append(entry)
    if not entries:
        raise B1KCheckpointContractError(
            f"Model path contains no recognized metadata or weight files: {path}"
        )
    return {
        "strategy": "metadata_content_and_weight_inventory_v1",
        "sha256": canonical_json_sha256(entries),
        "files": entries,
    }


def _required_file(value: Any, description: str) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise B1KCheckpointContractError(f"{description} path must be configured.")
    path = Path(value).expanduser()
    if not path.is_file():
        raise B1KCheckpointContractError(f"{description} does not exist: {path}")
    return path.resolve()


def _required_mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    child = value.get(key)
    if not isinstance(child, Mapping):
        raise B1KCheckpointContractError(f"Required config/contract mapping {key!r} is missing.")
    return child


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=True, allow_nan=False))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _git_provenance() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[4]
    values: dict[str, Any] = {}
    for key, args in {
        "commit": ("rev-parse", "HEAD"),
        "branch": ("symbolic-ref", "--short", "HEAD"),
    }.items():
        result = subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True, check=False
        )
        values[key] = result.stdout.strip() or None
    return values
