# Copyright (c) 2026 Xiaomi Corporation

import json
from pathlib import Path

import pytest

from fastwam.utils.checkpoint_contract import (
    CheckpointContract,
    CheckpointContractIntegrityError,
    CheckpointContractMismatchError,
    assert_checkpoint_compatible,
    canonical_json_dumps,
    canonical_json_sha256,
    json_path_differences,
    read_checkpoint_contract,
    write_checkpoint_contract,
)


def _contract(*, action_dim: int = 23, commit: str = "abc") -> CheckpointContract:
    return CheckpointContract(
        compatibility={
            "model": {"variant": "fastwam_uncond", "action_dim": action_dim},
            "data": {
                "action_labels": ["base_x", "base_y", "base_yaw"],
                "norm": {"mode": "mixed-z-score", "stats_sha256": "123"},
            },
        },
        provenance={"git": {"commit": commit}, "output_dir": Path("runs/example")},
    )


def test_canonical_json_and_hash_ignore_mapping_order():
    left = {"b": (2, 3), "a": {"value": 1}}
    right = {"a": {"value": 1}, "b": [2, 3]}

    assert canonical_json_dumps(left) == '{"a":{"value":1},"b":[2,3]}'
    assert canonical_json_sha256(left) == canonical_json_sha256(right)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_canonical_json_rejects_non_finite_floats(value):
    with pytest.raises(TypeError, match=r"Non-finite float.*\$\.value"):
        canonical_json_dumps({"value": value})


def test_contract_file_round_trip_and_atomic_write(tmp_path):
    contract = _contract()
    path = write_checkpoint_contract(tmp_path / "state" / "checkpoint_contract.json", contract)

    assert path == (tmp_path / "state" / "checkpoint_contract.json").resolve()
    assert read_checkpoint_contract(path) == contract
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["compatibility_sha256"] == contract.compatibility_sha256
    assert payload["contract_sha256"] == contract.contract_sha256
    assert not list(path.parent.glob(f".{path.name}.*.tmp"))


def test_read_rejects_tampered_payload(tmp_path):
    contract = _contract()
    path = write_checkpoint_contract(tmp_path / "checkpoint_contract.json", contract)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["compatibility"]["model"]["action_dim"] = 14
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CheckpointContractIntegrityError, match="compatibility_sha256"):
        read_checkpoint_contract(path)


def test_compatibility_ignores_provenance_by_default():
    assert_checkpoint_compatible(_contract(commit="old"), _contract(commit="new"))

    with pytest.raises(CheckpointContractMismatchError) as error:
        assert_checkpoint_compatible(
            _contract(commit="old"),
            _contract(commit="new"),
            check_provenance=True,
        )
    assert '$.provenance.git.commit: expected "old", got "new"' in str(error.value)


def test_compatibility_mismatch_reports_precise_json_path():
    with pytest.raises(CheckpointContractMismatchError) as error:
        assert_checkpoint_compatible(_contract(action_dim=23), _contract(action_dim=14))

    assert error.value.section == "compatibility"
    assert error.value.differences == (
        "$.compatibility.model.action_dim: expected 23, got 14",
    )


def test_json_path_differences_report_missing_extra_and_escaped_keys():
    differences = json_path_differences(
        {"items": [1, 2], "odd.key": {"expected": True}},
        {"items": [1, 3, 4], "odd.key": {"actual": True}},
    )

    assert differences == [
        "$.items[1]: expected 2, got 3",
        "$.items[2]: expected <missing>, got 4",
        '$["odd.key"].actual: expected <missing>, got true',
        '$["odd.key"].expected: expected true, got <missing>',
    ]


def test_schema_version_is_part_of_compatibility():
    expected = _contract()
    actual = CheckpointContract(
        compatibility=expected.compatibility,
        provenance=expected.provenance,
        schema_version=2,
    )

    with pytest.raises(CheckpointContractMismatchError) as error:
        assert_checkpoint_compatible(expected, actual)
    assert error.value.section == "schema"
    assert "$.schema_version: expected 1, got 2" in str(error.value)
