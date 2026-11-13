# Copyright (c) 2026 Xiaomi Corporation

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch
import torch.nn as nn


@dataclass(frozen=True)
class LegacyWarmstartReport:
    path: Path
    source_step: int | None
    mot_tensor_count: int
    proprio_tensor_count: int
    new_state_prefixes: tuple[str, ...]


def _validate_exact_state_dict(
    *,
    module_name: str,
    expected: Mapping[str, torch.Tensor],
    checkpoint: object,
) -> None:
    if not isinstance(checkpoint, Mapping):
        raise TypeError(
            f"legacy FastWAM warm-start `{module_name}` must be a mapping, "
            f"got {type(checkpoint).__name__}."
        )

    expected_keys = set(expected)
    checkpoint_keys = set(checkpoint)
    missing = sorted(expected_keys - checkpoint_keys)
    unexpected = sorted(checkpoint_keys - expected_keys)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing keys: {missing}")
        if unexpected:
            details.append(f"unexpected keys: {unexpected}")
        raise RuntimeError(
            f"legacy FastWAM warm-start `{module_name}` state is not exact; " + "; ".join(details)
        )

    for key, expected_tensor in expected.items():
        checkpoint_tensor = checkpoint[key]
        if not isinstance(checkpoint_tensor, torch.Tensor):
            raise TypeError(
                f"legacy FastWAM warm-start `{module_name}.{key}` must be a tensor, "
                f"got {type(checkpoint_tensor).__name__}."
            )
        if checkpoint_tensor.shape != expected_tensor.shape:
            raise RuntimeError(
                f"legacy FastWAM warm-start shape mismatch for `{module_name}.{key}`: "
                f"checkpoint={tuple(checkpoint_tensor.shape)}, "
                f"model={tuple(expected_tensor.shape)}."
            )


def _legacy_new_state_prefixes(model: nn.Module) -> tuple[str, ...]:
    raw_prefixes = getattr(model, "multimodal_new_state_prefixes", None)
    if raw_prefixes is None:
        return ()
    if not isinstance(raw_prefixes, (tuple, list)) or not raw_prefixes:
        raise TypeError(
            "`multimodal_new_state_prefixes` must be a non-empty tuple/list when declared."
        )
    prefixes = tuple(raw_prefixes)
    if any(not isinstance(prefix, str) or not prefix for prefix in prefixes):
        raise TypeError("Every `multimodal_new_state_prefixes` entry must be a non-empty string.")

    model_keys = tuple(model.state_dict())
    absent = [
        prefix
        for prefix in prefixes
        if not any(key == prefix or key.startswith(f"{prefix}.") for key in model_keys)
    ]
    if absent:
        raise RuntimeError(
            "legacy FastWAM model declarations do not match its state_dict; prefixes with no "
            f"parameters or buffers: {absent}"
        )
    return prefixes


def load_legacy_warmstart(model: nn.Module, path: str | Path) -> LegacyWarmstartReport:
    """Strictly warm-start a legacy FastWAM model from a legacy FastWAM weights file.

    This intentionally loads only the legacy ``mot`` and ``proprio_encoder``
    payloads. Optimizer, scheduler, and training progress are never restored.
    For a same-architecture model, all model state is restored exactly. If the target
    declares ``multimodal_new_state_prefixes``, those new modules remain at their
    initialization except that pointmap input/output modules are refreshed from the
    loaded RGB modules through ``initialize_pointmap_from_rgb``.
    """
    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"legacy FastWAM warm-start checkpoint not found: {checkpoint_path}")

    mot = getattr(model, "mot", None)
    if not isinstance(mot, nn.Module):
        raise TypeError("legacy FastWAM warm-start model must expose `mot` as torch.nn.Module.")
    proprio_encoder = getattr(model, "proprio_encoder", None)
    if not isinstance(proprio_encoder, nn.Module):
        raise TypeError(
            "legacy FastWAM warm-start requires `proprio_encoder`; the source B1K checkpoint "
            "was trained with proprioception."
        )
    new_state_prefixes = _legacy_new_state_prefixes(model)
    copy_pointmap = getattr(model, "initialize_pointmap_from_rgb", None)
    if new_state_prefixes and not callable(copy_pointmap):
        raise TypeError(
            "legacy FastWAM warm-start model with new multimodal state must implement "
            "`initialize_pointmap_from_rgb()`."
        )

    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise TypeError(
            "legacy FastWAM warm-start checkpoint payload must be a mapping, "
            f"got {type(payload).__name__}: {checkpoint_path}"
        )
    if "mot" not in payload:
        raise ValueError(f"legacy FastWAM warm-start checkpoint is missing `mot`: {checkpoint_path}")
    if "proprio_encoder" not in payload:
        raise ValueError(
            f"legacy FastWAM warm-start checkpoint is missing `proprio_encoder`: {checkpoint_path}"
        )

    mot_state = mot.state_dict()
    proprio_state = proprio_encoder.state_dict()
    _validate_exact_state_dict(
        module_name="mot",
        expected=mot_state,
        checkpoint=payload["mot"],
    )
    _validate_exact_state_dict(
        module_name="proprio_encoder",
        expected=proprio_state,
        checkpoint=payload["proprio_encoder"],
    )

    # Validation above is intentionally complete before either module mutates.
    mot.load_state_dict(payload["mot"], strict=True)
    proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
    if new_state_prefixes:
        copy_pointmap()

    source_step = payload.get("step")
    if source_step is not None:
        source_step = int(source_step)
    return LegacyWarmstartReport(
        path=checkpoint_path,
        source_step=source_step,
        mot_tensor_count=len(mot_state),
        proprio_tensor_count=len(proprio_state),
        new_state_prefixes=new_state_prefixes,
    )
