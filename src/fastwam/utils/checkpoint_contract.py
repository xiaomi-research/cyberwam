# Copyright (c) 2026 Xiaomi Corporation

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


CHECKPOINT_CONTRACT_KIND = "fastwam_checkpoint_contract"
CHECKPOINT_CONTRACT_SCHEMA_VERSION = 1

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_MISSING = object()


class CheckpointContractError(ValueError):
    """Base error for invalid or incompatible checkpoint contracts."""


class CheckpointContractIntegrityError(CheckpointContractError):
    """Raised when a serialized contract is malformed or its hash is invalid."""


class CheckpointContractMismatchError(CheckpointContractError):
    """Raised when a checkpoint is incompatible with the current runtime."""

    def __init__(self, differences: Sequence[str], *, section: str = "compatibility"):
        self.differences = tuple(differences)
        self.section = section
        detail = "\n".join(f"  - {difference}" for difference in self.differences)
        super().__init__(f"Checkpoint {section} contract mismatch:\n{detail}")


def _normalize_json(value: Any, *, path: str = "$") -> Any:
    """Convert a value to a deterministic JSON-compatible representation."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        value = dataclasses.asdict(value)

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError(f"Non-finite float is not allowed at {path}: {value!r}")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        normalized = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"Checkpoint contract mapping keys must be strings at {path}, "
                    f"got {type(key).__name__}"
                )
            normalized[key] = _normalize_json(item, path=_child_path(path, key))
        return normalized
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _normalize_json(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(
        f"Unsupported checkpoint contract value at {path}: {type(value).__name__}"
    )


def canonical_json_dumps(value: Any) -> str:
    """Serialize JSON data canonically for hashing and reproducibility."""
    normalized = _normalize_json(value)
    return json.dumps(
        normalized,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_dumps(value).encode("utf-8")).hexdigest()


def _child_path(parent: str, key: str) -> str:
    if _IDENTIFIER_RE.fullmatch(key):
        return f"{parent}.{key}"
    return f"{parent}[{json.dumps(key, ensure_ascii=True)}]"


def _display_value(value: Any) -> str:
    if value is _MISSING:
        return "<missing>"
    return canonical_json_dumps(value)


def json_path_differences(
    expected: Any,
    actual: Any,
    *,
    root: str = "$",
    max_differences: int = 100,
) -> list[str]:
    """Return deterministic, human-readable differences with JSON paths."""
    if max_differences <= 0:
        raise ValueError("max_differences must be positive")

    expected = _normalize_json(expected, path=root)
    actual = _normalize_json(actual, path=root)
    differences: list[str] = []

    def add(path: str, expected_value: Any, actual_value: Any) -> None:
        if len(differences) >= max_differences:
            return
        differences.append(
            f"{path}: expected {_display_value(expected_value)}, "
            f"got {_display_value(actual_value)}"
        )

    def visit(expected_value: Any, actual_value: Any, path: str) -> None:
        if len(differences) >= max_differences:
            return
        if isinstance(expected_value, dict) and isinstance(actual_value, dict):
            for key in sorted(set(expected_value) | set(actual_value)):
                visit(
                    expected_value.get(key, _MISSING),
                    actual_value.get(key, _MISSING),
                    _child_path(path, key),
                )
            return
        if isinstance(expected_value, list) and isinstance(actual_value, list):
            for index in range(max(len(expected_value), len(actual_value))):
                visit(
                    expected_value[index] if index < len(expected_value) else _MISSING,
                    actual_value[index] if index < len(actual_value) else _MISSING,
                    f"{path}[{index}]",
                )
            return
        if expected_value is _MISSING or actual_value is _MISSING:
            add(path, expected_value, actual_value)
            return
        if type(expected_value) is not type(actual_value) or expected_value != actual_value:
            add(path, expected_value, actual_value)

    visit(expected, actual, root)
    if len(differences) == max_differences:
        differences.append(f"{root}: further differences omitted")
    return differences


@dataclasses.dataclass(frozen=True)
class CheckpointContract:
    compatibility: dict[str, Any]
    provenance: dict[str, Any] = dataclasses.field(default_factory=dict)
    schema_version: int = CHECKPOINT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.schema_version, int) or self.schema_version <= 0:
            raise TypeError("schema_version must be a positive integer")
        compatibility = _normalize_json(self.compatibility, path="$.compatibility")
        provenance = _normalize_json(self.provenance, path="$.provenance")
        if not isinstance(compatibility, dict):
            raise TypeError("compatibility must be a mapping")
        if not isinstance(provenance, dict):
            raise TypeError("provenance must be a mapping")
        object.__setattr__(self, "compatibility", compatibility)
        object.__setattr__(self, "provenance", provenance)

    @property
    def compatibility_sha256(self) -> str:
        return canonical_json_sha256(self.compatibility)

    @property
    def contract_sha256(self) -> str:
        return canonical_json_sha256(self._unsigned_payload())

    def _unsigned_payload(self) -> dict[str, Any]:
        return {
            "kind": CHECKPOINT_CONTRACT_KIND,
            "schema_version": self.schema_version,
            "compatibility": self.compatibility,
            "provenance": self.provenance,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._unsigned_payload(),
            "compatibility_sha256": self.compatibility_sha256,
            "contract_sha256": self.contract_sha256,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CheckpointContract":
        normalized = _normalize_json(payload)
        if not isinstance(normalized, dict):
            raise CheckpointContractIntegrityError("Checkpoint contract payload must be a mapping")

        required = {
            "kind",
            "schema_version",
            "compatibility",
            "provenance",
            "compatibility_sha256",
            "contract_sha256",
        }
        missing = sorted(required - set(normalized))
        unexpected = sorted(set(normalized) - required)
        if missing or unexpected:
            raise CheckpointContractIntegrityError(
                f"Malformed checkpoint contract keys: missing={missing}, unexpected={unexpected}"
            )
        if normalized["kind"] != CHECKPOINT_CONTRACT_KIND:
            raise CheckpointContractIntegrityError(
                f"Unsupported checkpoint contract kind: {normalized['kind']!r}"
            )
        contract = cls(
            schema_version=normalized["schema_version"],
            compatibility=normalized["compatibility"],
            provenance=normalized["provenance"],
        )
        if normalized["compatibility_sha256"] != contract.compatibility_sha256:
            raise CheckpointContractIntegrityError(
                "Checkpoint compatibility_sha256 does not match its compatibility payload"
            )
        if normalized["contract_sha256"] != contract.contract_sha256:
            raise CheckpointContractIntegrityError(
                "Checkpoint contract_sha256 does not match its contract payload"
            )
        return contract


def write_checkpoint_contract(path: str | os.PathLike[str], contract: CheckpointContract) -> Path:
    """Atomically write a checkpoint contract and return its resolved path."""
    if not isinstance(contract, CheckpointContract):
        raise TypeError(f"contract must be CheckpointContract, got {type(contract).__name__}")
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        contract.to_dict(),
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        indent=2,
    )
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_file.write(serialized)
            temporary_file.write("\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
            temporary_path = temporary_file.name
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None and os.path.exists(temporary_path):
            os.unlink(temporary_path)
    return output_path


def read_checkpoint_contract(path: str | os.PathLike[str]) -> CheckpointContract:
    input_path = Path(path).expanduser().resolve()
    try:
        with input_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        raise CheckpointContractIntegrityError(
            f"Failed to read checkpoint contract {input_path}: {error}"
        ) from error
    return CheckpointContract.from_dict(payload)


def assert_checkpoint_compatible(
    expected: CheckpointContract,
    actual: CheckpointContract,
    *,
    check_provenance: bool = False,
    max_differences: int = 100,
) -> None:
    """Raise a path-rich error unless two contracts are load-compatible."""
    if not isinstance(expected, CheckpointContract) or not isinstance(actual, CheckpointContract):
        raise TypeError("expected and actual must both be CheckpointContract instances")

    schema_differences = json_path_differences(
        expected.schema_version,
        actual.schema_version,
        root="$.schema_version",
        max_differences=max_differences,
    )
    if schema_differences:
        raise CheckpointContractMismatchError(schema_differences, section="schema")

    compatibility_differences = json_path_differences(
        expected.compatibility,
        actual.compatibility,
        root="$.compatibility",
        max_differences=max_differences,
    )
    if compatibility_differences:
        raise CheckpointContractMismatchError(
            compatibility_differences,
            section="compatibility",
        )

    if check_provenance:
        provenance_differences = json_path_differences(
            expected.provenance,
            actual.provenance,
            root="$.provenance",
            max_differences=max_differences,
        )
        if provenance_differences:
            raise CheckpointContractMismatchError(
                provenance_differences,
                section="provenance",
            )
