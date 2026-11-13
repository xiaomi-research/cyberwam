# Copyright (c) 2026 Xiaomi Corporation

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch
import torch.nn as nn


@dataclass(frozen=True)
class FlexWarmstartReport:
    path: Path
    source_step: int | None
    mot_tensor_count: int
    proprio_tensor_count: int
    adapter_tensor_count: int
    adapter_names: tuple[str, ...]


def _validate_exact_state_dict(
    *,
    module_name: str,
    expected: Mapping[str, torch.Tensor],
    checkpoint: object,
) -> None:
    if not isinstance(checkpoint, Mapping):
        raise TypeError(
            f"FastWAM Flex warm-start `{module_name}` must be a mapping, "
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
            f"FastWAM Flex warm-start `{module_name}` state is not exact; "
            + "; ".join(details)
        )

    for key, expected_tensor in expected.items():
        checkpoint_tensor = checkpoint[key]
        if not isinstance(checkpoint_tensor, torch.Tensor):
            raise TypeError(
                f"FastWAM Flex warm-start `{module_name}.{key}` must be a tensor, "
                f"got {type(checkpoint_tensor).__name__}."
            )
        if checkpoint_tensor.shape != expected_tensor.shape:
            raise RuntimeError(
                f"FastWAM Flex warm-start shape mismatch for `{module_name}.{key}`: "
                f"checkpoint={tuple(checkpoint_tensor.shape)}, "
                f"model={tuple(expected_tensor.shape)}."
            )


def _multimodal_adapter_modules(model: nn.Module) -> dict[str, nn.Module]:
    raw_names = getattr(model, "multimodal_new_state_prefixes", None)
    if not isinstance(raw_names, (tuple, list)) or not raw_names:
        raise TypeError(
            "FastWAM Flex warm-start requires the model to declare a non-empty "
            "`multimodal_new_state_prefixes` tuple/list."
        )
    names = tuple(raw_names)
    if any(not isinstance(name, str) or not name for name in names):
        raise TypeError("Every `multimodal_new_state_prefixes` entry must be a non-empty string.")
    if len(set(names)) != len(names):
        raise ValueError("`multimodal_new_state_prefixes` must not contain duplicate entries.")

    modules: dict[str, nn.Module] = {}
    for name in names:
        module = getattr(model, name, None)
        if not isinstance(module, nn.Module):
            raise TypeError(
                f"FastWAM Flex warm-start adapter `{name}` must be a torch.nn.Module."
            )
        modules[name] = module
    return modules


def load_flex_warmstart(model: nn.Module, path: str | Path) -> FlexWarmstartReport:
    """Strictly initialize FastWAM Flex from a Flex weights checkpoint.

    Only the multimodal trainable payload is restored: ``mot``,
    ``proprio_encoder``, and every module in the checkpoint's ``multimodal``
    adapter mapping. Optimizer, scheduler, and training progress are ignored.
    All payloads are validated before any model tensor is mutated.
    """
    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"FastWAM Flex warm-start checkpoint not found: {checkpoint_path}")

    mot = getattr(model, "mot", None)
    if not isinstance(mot, nn.Module):
        raise TypeError("FastWAM Flex warm-start model must expose `mot` as torch.nn.Module.")
    proprio_encoder = getattr(model, "proprio_encoder", None)
    if not isinstance(proprio_encoder, nn.Module):
        raise TypeError(
            "FastWAM Flex warm-start requires `proprio_encoder`; its source "
            "checkpoint must contain the same proprioception adapter."
        )
    adapters = _multimodal_adapter_modules(model)

    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise TypeError(
            "FastWAM Flex warm-start checkpoint payload must be a mapping, "
            f"got {type(payload).__name__}: {checkpoint_path}"
        )
    for required in ("mot", "proprio_encoder", "multimodal"):
        if required not in payload:
            raise ValueError(
                f"FastWAM Flex warm-start checkpoint is missing `{required}`: {checkpoint_path}"
            )

    multimodal = payload["multimodal"]
    if not isinstance(multimodal, Mapping):
        raise TypeError(
            "FastWAM Flex warm-start `multimodal` must be a mapping, "
            f"got {type(multimodal).__name__}."
        )
    expected_adapter_names = set(adapters)
    checkpoint_adapter_names = set(multimodal)
    missing = sorted(expected_adapter_names - checkpoint_adapter_names)
    unexpected = sorted(checkpoint_adapter_names - expected_adapter_names)
    if missing or unexpected:
        raise RuntimeError(
            "FastWAM Flex warm-start `multimodal` module mismatch: "
            f"missing={missing}, unexpected={unexpected}."
        )

    mot_state = mot.state_dict()
    proprio_state = proprio_encoder.state_dict()
    adapter_states = {name: module.state_dict() for name, module in adapters.items()}
    _validate_exact_state_dict(
        module_name="mot", expected=mot_state, checkpoint=payload["mot"]
    )
    _validate_exact_state_dict(
        module_name="proprio_encoder",
        expected=proprio_state,
        checkpoint=payload["proprio_encoder"],
    )
    for name, state in adapter_states.items():
        _validate_exact_state_dict(
            module_name=f"multimodal.{name}", expected=state, checkpoint=multimodal[name]
        )

    # The validation pass above is intentionally complete before any mutation.
    mot.load_state_dict(payload["mot"], strict=True)
    proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
    for name, module in adapters.items():
        module.load_state_dict(multimodal[name], strict=True)

    source_step = payload.get("step")
    if source_step is not None:
        source_step = int(source_step)
    return FlexWarmstartReport(
        path=checkpoint_path,
        source_step=source_step,
        mot_tensor_count=len(mot_state),
        proprio_tensor_count=len(proprio_state),
        adapter_tensor_count=sum(len(state) for state in adapter_states.values()),
        adapter_names=tuple(adapters),
    )
