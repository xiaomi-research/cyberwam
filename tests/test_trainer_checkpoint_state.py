# Copyright (c) 2026 Xiaomi Corporation

import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from types import ModuleType

import pytest
import torch

try:
    import accelerate  # noqa: F401
except ImportError:
    accelerate = ModuleType("accelerate")
    accelerate.Accelerator = object
    sys.modules["accelerate"] = accelerate

from fastwam.trainer import (
    Wan22Trainer,
    _atomic_write_json,
    _build_checkpoint_inventory,
    _read_checkpoint_complete_marker,
    _sha256_file,
    _resolve_checkpoint_retention_keep_every_n_steps,
    _resolve_checkpoint_retention_keep_final,
    _resolve_checkpoint_retention_max_to_keep,
    _resolve_permanent_weights_every_n_steps,
    discover_latest_complete_checkpoint,
    read_valid_training_completion,
)
from fastwam.utils.checkpoint_contract import (
    CheckpointContract,
    CheckpointContractIntegrityError,
    CheckpointContractMismatchError,
    read_checkpoint_contract,
    write_checkpoint_contract,
)


def _contract(*, action_dim: int = 23) -> CheckpointContract:
    return CheckpointContract(
        compatibility={
            "model": "fastwam-uncond",
            "action_dim": action_dim,
            "proprio_dim": 23,
        },
        provenance={"test": True},
    )


class _FakeAccelerator:
    is_main_process = True
    num_processes = 8
    process_index = 0
    device = "cpu"

    def __init__(self, model=None):
        self.model = model
        self.load_calls = []
        self.save_calls = []

    def unwrap_model(self, model):
        return self.model or model

    def wait_for_everyone(self):
        return None

    def save_state(self, *, output_dir):
        self.save_calls.append(output_dir)

    def load_state(self, *, input_dir):
        self.load_calls.append(input_dir)

    def gather(self, tensor):
        return tensor.repeat(self.num_processes)


def test_initialization_fingerprint_rank_check_accepts_identical_hashes():
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.accelerator = _FakeAccelerator()
    trainer.initialization_fingerprint_sha256 = "ab" * 32
    trainer._assert_initialization_fingerprint_consistent()


def test_initialization_fingerprint_rank_check_rejects_mismatch():
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.initialization_fingerprint_sha256 = "ab" * 32

    class _MismatchAccelerator(_FakeAccelerator):
        num_processes = 2

        def gather(self, tensor):
            different = torch.tensor(
                list(bytes.fromhex("cd" * 32)), dtype=torch.uint8
            )
            return torch.cat([tensor, different])

    trainer.accelerator = _MismatchAccelerator()
    with pytest.raises(RuntimeError, match="differs across distributed ranks"):
        trainer._assert_initialization_fingerprint_consistent()


class _ContractAwareModel:
    def __init__(self):
        self.saved = None

    def save_checkpoint(
        self, path, optimizer=None, step=None, checkpoint_contract=None
    ):
        self.saved = {
            "path": path,
            "optimizer": optimizer,
            "step": step,
            "checkpoint_contract": checkpoint_contract,
        }


class _WritingCheckpointModel(_ContractAwareModel):
    def save_checkpoint(
        self, path, optimizer=None, step=None, checkpoint_contract=None
    ):
        super().save_checkpoint(
            path,
            optimizer=optimizer,
            step=step,
            checkpoint_contract=checkpoint_contract,
        )
        Path(path).write_bytes(b"weights")


class _InventoryAccelerator(_FakeAccelerator):
    def save_state(self, *, output_dir):
        super().save_state(output_dir=output_dir)
        output = Path(output_dir)
        (output / "optimizer").mkdir(parents=True, exist_ok=True)
        (output / "optimizer" / "rank_000.bin").write_bytes(
            b"x" * (1024 * 1024 + 1)
        )
        (output / "random_states_0.pkl").write_bytes(b"small-metadata")


def _save_inventory_checkpoint(tmp_path: Path) -> tuple[Wan22Trainer, Path, Path]:
    model = _WritingCheckpointModel()
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.model = model
    trainer.accelerator = _InventoryAccelerator(model)
    trainer.checkpoint_contract = None
    trainer.global_step = 20
    trainer.epoch = 0
    trainer.batch_in_epoch = 0
    trainer.weights_dir = str(tmp_path / "checkpoints" / "weights")
    trainer.state_dir = str(tmp_path / "checkpoints" / "state")
    Path(trainer.weights_dir).mkdir(parents=True)
    Path(trainer.state_dir).mkdir(parents=True)
    result = trainer.save_checkpoint()
    return trainer, Path(result["state_path"]), Path(result["weights_path"])


def test_discover_latest_complete_checkpoint_returns_empty_report(tmp_path):
    output = tmp_path / "run"

    report = discover_latest_complete_checkpoint(output)

    assert report == {
        "mode": "latest_complete",
        "resume_path": None,
        "global_step": None,
        "rejected_candidates": [],
    }


def test_discover_latest_complete_checkpoint_ignores_weights_only(tmp_path):
    output = tmp_path / "run"
    weights = output / "checkpoints" / "weights"
    weights.mkdir(parents=True)
    weights_path = weights / "step_000030.pt"
    weights_path.write_bytes(b"inference-only")

    report = discover_latest_complete_checkpoint(output)

    assert report["resume_path"] is None
    assert report["global_step"] is None
    assert report["rejected_candidates"] == [
        f"{weights_path}: standalone weights have no full training state"
    ]


def test_discover_latest_complete_checkpoint_falls_back_from_invalid_latest(
    tmp_path,
):
    output = tmp_path / "run"
    state_root = output / "checkpoints" / "state"
    weights_root = output / "checkpoints" / "weights"
    state_root.mkdir(parents=True)
    weights_root.mkdir(parents=True)
    valid_state, _ = _write_complete_pair(
        state_root,
        weights_root,
        step=20,
        completed_at=200,
        world_size=8,
    )
    invalid_state, _ = _write_complete_pair(
        state_root,
        weights_root,
        step=30,
        completed_at=300,
        world_size=8,
    )
    invalid_marker = json.loads(
        (invalid_state / "checkpoint_complete.json").read_text(encoding="utf-8")
    )
    invalid_marker["global_step"] = 31
    (invalid_state / "checkpoint_complete.json").write_text(
        json.dumps(invalid_marker), encoding="utf-8"
    )

    report = discover_latest_complete_checkpoint(output)

    assert report["resume_path"] == str(valid_state.resolve())
    assert report["global_step"] == 20
    assert report["rejected_candidates"] == [
        f"{invalid_state.resolve()}: invalid completion marker or committed inventory"
    ]


def test_discover_latest_complete_checkpoint_sorts_by_step_when_times_equal(
    tmp_path,
):
    output = tmp_path / "run"
    state_root = output / "checkpoints" / "state"
    weights_root = output / "checkpoints" / "weights"
    state_root.mkdir(parents=True)
    weights_root.mkdir(parents=True)
    _write_complete_pair(
        state_root,
        weights_root,
        step=20,
        completed_at=123,
        world_size=8,
    )
    latest_state, _ = _write_complete_pair(
        state_root,
        weights_root,
        step=40,
        completed_at=123,
        world_size=8,
    )

    report = discover_latest_complete_checkpoint(output)

    assert report["resume_path"] == str(latest_state.resolve())
    assert report["global_step"] == 40
    assert report["rejected_candidates"] == []


def test_discover_latest_complete_checkpoint_honors_resume_ceiling(tmp_path):
    output = tmp_path / "run"
    state_root = output / "checkpoints" / "state"
    weights_root = output / "checkpoints" / "weights"
    state_root.mkdir(parents=True)
    weights_root.mkdir(parents=True)
    earlier_state, _ = _write_complete_pair(
        state_root, weights_root, step=10, completed_at=10
    )
    terminal_state, _ = _write_complete_pair(
        state_root, weights_root, step=20, completed_at=20
    )

    report = discover_latest_complete_checkpoint(
        output, maximum_step_exclusive=20
    )

    assert report["global_step"] == 10
    assert report["resume_path"] == str(earlier_state.resolve())
    assert any(
        str(terminal_state.resolve()) in reason and "resume ceiling 20" in reason
        for reason in report["rejected_candidates"]
    )


def test_discover_latest_complete_checkpoint_strict_rejects_legacy_marker(
    tmp_path,
):
    output = tmp_path / "run"
    state_root = output / "checkpoints" / "state"
    weights_root = output / "checkpoints" / "weights"
    state_root.mkdir(parents=True)
    weights_root.mkdir(parents=True)
    state, _ = _write_complete_pair(
        state_root,
        weights_root,
        step=20,
        completed_at=123,
        world_size=8,
    )

    report = discover_latest_complete_checkpoint(
        output,
        expected_world_size=8,
        require_strict_metadata=True,
    )

    assert report["resume_path"] is None
    assert report["global_step"] is None
    assert report["rejected_candidates"] == [
        f"{state.resolve()}: missing committed inventory; "
        "strict checkpoint contract not required; rank resume probes not required; "
        "resume probe count mismatch (checkpoint=0, expected=8)"
    ]


def test_discover_latest_complete_checkpoint_rejects_wrong_world_size(tmp_path):
    output = tmp_path / "run"
    state_root = output / "checkpoints" / "state"
    weights_root = output / "checkpoints" / "weights"
    state_root.mkdir(parents=True)
    weights_root.mkdir(parents=True)
    state, _ = _write_complete_pair(
        state_root,
        weights_root,
        step=20,
        completed_at=123,
        world_size=4,
    )

    report = discover_latest_complete_checkpoint(output, expected_world_size=8)

    assert report["resume_path"] is None
    assert report["rejected_candidates"] == [
        f"{state.resolve()}: world size mismatch (checkpoint=4, expected=8)"
    ]


def test_discover_latest_complete_checkpoint_selects_valid_strict_state(
    tmp_path,
):
    output = tmp_path / "run"
    state_root = output / "checkpoints" / "state"
    weights_root = output / "checkpoints" / "weights"
    state_root.mkdir(parents=True)
    weights_root.mkdir(parents=True)
    state, weights = _write_complete_pair(
        state_root,
        weights_root,
        step=20,
        completed_at=123,
        world_size=8,
    )
    contract = _contract()
    write_checkpoint_contract(state / "checkpoint_contract.json", contract)
    (state / "trainer_state.json").write_text(
        json.dumps(
            {
                "global_step": 20,
                "checkpoint_contract_sha256": contract.contract_sha256,
            }
        ),
        encoding="utf-8",
    )
    for rank in range(8):
        (state / f"resume_probe_rank_{rank:03d}.json").write_text(
            json.dumps({"rank": rank}), encoding="utf-8"
        )
    marker_path = state / "checkpoint_complete.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker.update(
        {
            "checkpoint_contract_required": True,
            "checkpoint_contract_sha256": contract.contract_sha256,
            "resume_probes_required": True,
            "resume_probe_count": 8,
            "inventory": _build_checkpoint_inventory(state, weights),
        }
    )
    marker_path.write_text(json.dumps(marker), encoding="utf-8")

    report = discover_latest_complete_checkpoint(
        output,
        expected_world_size=8,
        require_strict_metadata=True,
    )

    assert report["resume_path"] == str(state.resolve())
    assert report["global_step"] == 20
    assert report["rejected_candidates"] == []


def test_save_checkpoint_writes_contract_and_hash_bound_trainer_state(tmp_path):
    contract = _contract()
    model = _WritingCheckpointModel()
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.model = model
    trainer.accelerator = _FakeAccelerator(model)
    trainer.checkpoint_contract = contract
    trainer.global_step = 17
    trainer.epoch = 3
    trainer.batch_in_epoch = 11
    trainer.weights_dir = str(tmp_path / "weights")
    trainer.state_dir = str(tmp_path / "state")
    Path(trainer.weights_dir).mkdir()
    Path(trainer.state_dir).mkdir()

    result = trainer.save_checkpoint()

    state_path = Path(result["state_path"])
    contract_payload = json.loads(
        (state_path / "checkpoint_contract.json").read_text(encoding="utf-8")
    )
    trainer_payload = json.loads(
        (state_path / "trainer_state.json").read_text(encoding="utf-8")
    )
    assert contract_payload == contract.to_dict()
    assert trainer_payload == {
        "global_step": 17,
        "epoch": 3,
        "batch_in_epoch": 11,
        "checkpoint_contract_sha256": contract.contract_sha256,
    }
    assert model.saved["checkpoint_contract"] is contract
    assert model.saved["step"] == 17
    completion = json.loads(
        (state_path / "checkpoint_complete.json").read_text(encoding="utf-8")
    )
    assert completion["global_step"] == 17
    assert completion["weights_file"] == "step_000017.pt"
    assert completion["checkpoint_contract_required"] is True
    assert completion["checkpoint_contract_sha256"] == contract.contract_sha256
    assert completion["resume_probe_count"] == 0
    assert completion["inventory"]["sha256_max_bytes"] == 1024 * 1024
    assert completion["inventory"]["large_file_validation"] == "path_and_size_only"
    assert [entry["path"] for entry in completion["inventory"]["files"]] == [
        "state/checkpoint_contract.json",
        "state/trainer_state.json",
        "weights/step_000017.pt",
    ]
    assert all("sha256" in entry for entry in completion["inventory"]["files"])


def test_checkpoint_inventory_accepts_unchanged_checkpoint_and_ignores_atomic_temp(
    tmp_path,
):
    _, state, weights = _save_inventory_checkpoint(tmp_path)

    marker = _read_checkpoint_complete_marker(state, weights)
    assert marker is not None
    entries = {entry["path"]: entry for entry in marker["inventory"]["files"]}
    assert "sha256" not in entries["state/optimizer/rank_000.bin"]
    assert "sha256" in entries["state/random_states_0.pkl"]
    (state / ".trainer_state.json.abcd.tmp").write_bytes(b"uncommitted")
    (state / "accelerate-save.tmp").write_bytes(b"uncommitted")
    assert _read_checkpoint_complete_marker(state, weights) is not None


@pytest.mark.parametrize(
    "mutation",
    ["delete", "truncate", "extra", "metadata_bitflip"],
)
def test_checkpoint_inventory_rejects_file_set_size_and_small_hash_corruption(
    tmp_path, mutation
):
    trainer, state, weights = _save_inventory_checkpoint(tmp_path)
    shard = state / "optimizer" / "rank_000.bin"
    metadata = state / "random_states_0.pkl"
    if mutation == "delete":
        shard.unlink()
    elif mutation == "truncate":
        with shard.open("r+b") as file:
            file.truncate(shard.stat().st_size - 1)
    elif mutation == "extra":
        (state / "unexpected.bin").write_bytes(b"extra")
    else:
        original = metadata.read_bytes()
        metadata.write_bytes(b"X" + original[1:])

    assert _read_checkpoint_complete_marker(state, weights) is None
    trainer.accelerator.load_calls.clear()
    with pytest.raises(CheckpointContractIntegrityError, match="incomplete checkpoint"):
        trainer.load_training_state(str(state))
    assert trainer.accelerator.load_calls == []


def test_checkpoint_inventory_rejects_standalone_weights_truncation(tmp_path):
    trainer, state, weights = _save_inventory_checkpoint(tmp_path)
    weights.write_bytes(b"weight")

    assert _read_checkpoint_complete_marker(state, weights) is None
    with pytest.raises(CheckpointContractIntegrityError, match="incomplete checkpoint"):
        trainer.load_training_state(str(state))


def test_checkpoint_inventory_corruption_is_not_eligible_for_retention(tmp_path):
    trainer, state, weights = _save_inventory_checkpoint(tmp_path)
    trainer.checkpoint_retention_max_to_keep = 1
    (state / "unexpected.bin").write_bytes(b"extra")
    state_root = Path(trainer.state_dir)
    weights_root = Path(trainer.weights_dir)
    old_valid_state, old_valid_weights = _write_complete_pair(
        state_root, weights_root, step=30, completed_at=30
    )
    new_valid_state, new_valid_weights = _write_complete_pair(
        state_root, weights_root, step=40, completed_at=40
    )

    assert trainer._prune_complete_checkpoint_pairs() == [30]
    assert state.exists()
    assert weights.exists()
    assert not old_valid_state.exists()
    assert not old_valid_weights.exists()
    assert new_valid_state.exists()
    assert new_valid_weights.exists()


def _write_complete_pair(
    state_root: Path,
    weights_root: Path,
    *,
    step: int,
    completed_at: int,
    world_size: int = 1,
    retention_class: str | None = None,
) -> tuple[Path, Path]:
    tag = f"step_{step:06d}"
    state = state_root / tag
    state.mkdir()
    (state / "trainer_state.json").write_text(
        json.dumps({"global_step": step}), encoding="utf-8"
    )
    weights = weights_root / f"{tag}.pt"
    weights.write_bytes(b"weights")
    marker = {
        "schema_version": 1,
        "global_step": step,
        "state_directory": tag,
        "weights_file": weights.name,
        "world_size": world_size,
        "checkpoint_contract_required": False,
        "checkpoint_contract_sha256": None,
        "resume_probes_required": False,
        "resume_probe_count": 0,
        "completed_at_unix_ns": completed_at,
    }
    if retention_class is not None:
        marker["retention_class"] = retention_class
    (state / "checkpoint_complete.json").write_text(
        json.dumps(marker),
        encoding="utf-8",
    )
    return state, weights


def _write_strict_training_completion(
    output: Path,
    *,
    step: int = 20,
    world_size: int = 2,
    manifest_sha256: str = "ab" * 32,
) -> dict:
    state_root = output / "checkpoints" / "state"
    weights_root = output / "checkpoints" / "weights"
    state_root.mkdir(parents=True)
    weights_root.mkdir(parents=True)
    tag = f"step_{step:06d}"
    state = state_root / tag
    state.mkdir()
    weights = weights_root / f"{tag}.pt"
    weights.write_bytes(b"weights")
    contract = CheckpointContract(
        compatibility=_contract().compatibility,
        provenance={
            "run_manifest": {
                "sha256": manifest_sha256,
                "run_id": "test-run",
            }
        },
    )
    write_checkpoint_contract(state / "checkpoint_contract.json", contract)
    _atomic_write_json(
        state / "trainer_state.json",
        {
            "global_step": step,
            "checkpoint_contract_sha256": contract.contract_sha256,
        },
    )
    for rank in range(world_size):
        _atomic_write_json(state / f"resume_probe_rank_{rank:03d}.json", {"rank": rank})
    checkpoint_marker = {
        "schema_version": 1,
        "global_step": step,
        "state_directory": tag,
        "weights_file": weights.name,
        "world_size": world_size,
        "checkpoint_contract_required": True,
        "checkpoint_contract_sha256": contract.contract_sha256,
        "resume_probes_required": True,
        "resume_probe_count": world_size,
        "completed_at_unix_ns": 1,
        "retention_class": "final",
        "inventory": _build_checkpoint_inventory(state, weights),
    }
    _atomic_write_json(state / "checkpoint_complete.json", checkpoint_marker)
    _atomic_write_json(output / "training-summary.json", {"optimizer_steps": step})
    completion = {
        "schema_version": 1,
        "status": "completed",
        "global_step": step,
        "max_steps": step,
        "world_size": world_size,
        "state_directory": tag,
        "weights_file": weights.name,
        "checkpoint_complete_sha256": _sha256_file(
            state / "checkpoint_complete.json"
        ),
        "checkpoint_contract_sha256": contract.contract_sha256,
        "run_manifest_sha256": manifest_sha256,
        "run_id": "test-run",
        "training_summary_sha256": _sha256_file(output / "training-summary.json"),
        "completed_at_unix_ns": 2,
    }
    _atomic_write_json(output / "training_complete.json", completion)
    return completion


def test_training_completion_requires_bound_valid_artifacts(tmp_path):
    completion = _write_strict_training_completion(tmp_path)

    assert read_valid_training_completion(
        tmp_path,
        expected_max_steps=20,
        expected_world_size=2,
        expected_run_manifest_sha256="ab" * 32,
    ) == completion


def test_trainer_publishes_completion_after_final_checkpoint_and_summary(tmp_path):
    _write_strict_training_completion(tmp_path)
    (tmp_path / "training_complete.json").unlink()
    state = tmp_path / "checkpoints/state/step_000020"
    weights = tmp_path / "checkpoints/weights/step_000020.pt"
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.accelerator = _FakeAccelerator()
    trainer.accelerator.num_processes = 2
    trainer.output_dir = str(tmp_path)
    trainer.global_step = 20
    trainer.max_steps = 20
    trainer.final_checkpoint_mode = "full"
    trainer.checkpoint_contract = read_checkpoint_contract(
        state / "checkpoint_contract.json"
    )

    marker = trainer._write_training_complete_marker(
        {"state_path": str(state), "weights_path": str(weights)}
    )

    assert marker == tmp_path / "training_complete.json"
    assert read_valid_training_completion(
        tmp_path,
        expected_max_steps=20,
        expected_world_size=2,
        expected_run_manifest_sha256="ab" * 32,
    ) is not None


def test_trainer_revokes_stale_completion_before_running(tmp_path):
    marker = tmp_path / "training_complete.json"
    _atomic_write_json(marker, {"status": "completed"})
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.accelerator = _FakeAccelerator()
    trainer.output_dir = str(tmp_path)

    trainer._revoke_training_complete_marker()

    assert not marker.exists()


def test_train_publishes_completion_only_after_loggers_close():
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.accelerator = _FakeAccelerator()
    calls = []
    checkpoint = {"state_path": "state", "weights_path": "weights"}
    trainer._train_impl = lambda: checkpoint
    trainer._close_loggers = lambda: calls.append("close")
    trainer._write_training_complete_marker = lambda value: calls.append(
        ("complete", value)
    )

    assert trainer.train() == checkpoint
    assert calls == ["close", ("complete", checkpoint)]


def test_train_does_not_publish_completion_when_logger_close_fails():
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.accelerator = _FakeAccelerator()
    trainer._train_impl = lambda: {"state_path": "state", "weights_path": "weights"}
    trainer._write_training_complete_marker = lambda value: pytest.fail(
        "completion marker must not be published"
    )

    def fail_close():
        raise RuntimeError("flush failed")

    trainer._close_loggers = fail_close
    with pytest.raises(RuntimeError, match="flush failed"):
        trainer.train()


def test_training_completion_rejects_tampered_summary(tmp_path):
    _write_strict_training_completion(tmp_path)
    (tmp_path / "training-summary.json").write_text("{}\n", encoding="utf-8")

    assert read_valid_training_completion(
        tmp_path,
        expected_max_steps=20,
        expected_world_size=2,
        expected_run_manifest_sha256="ab" * 32,
    ) is None


def test_terminal_checkpoint_without_training_completion_is_not_complete(tmp_path):
    _write_strict_training_completion(tmp_path)
    (tmp_path / "training_complete.json").unlink()

    assert read_valid_training_completion(
        tmp_path,
        expected_max_steps=20,
        expected_world_size=2,
        expected_run_manifest_sha256="ab" * 32,
    ) is None
    discovery = discover_latest_complete_checkpoint(
        tmp_path, expected_world_size=2, require_strict_metadata=True
    )
    assert discovery["global_step"] == 20
    assert discovery["resume_path"].endswith("step_000020")


def test_legacy_completion_marker_without_inventory_remains_valid(tmp_path):
    state_root = tmp_path / "state"
    weights_root = tmp_path / "weights"
    state_root.mkdir()
    weights_root.mkdir()
    state, weights = _write_complete_pair(
        state_root,
        weights_root,
        step=5,
        completed_at=5,
    )

    marker = _read_checkpoint_complete_marker(state, weights)

    assert marker is not None
    assert "inventory" not in marker


@pytest.mark.parametrize("value", [None, "", "null"])
def test_checkpoint_retention_can_be_disabled(value):
    assert _resolve_checkpoint_retention_max_to_keep(value) is None


@pytest.mark.parametrize("value, expected", [(1, 1), ("3", 3), (4.0, 4)])
def test_checkpoint_retention_accepts_positive_integers(value, expected):
    assert _resolve_checkpoint_retention_max_to_keep(value) == expected


@pytest.mark.parametrize("value", [True, False, "many"])
def test_checkpoint_retention_rejects_non_integers(value):
    with pytest.raises(TypeError, match="positive integer or null"):
        _resolve_checkpoint_retention_max_to_keep(value)


@pytest.mark.parametrize("value", [0, -1, 1.5])
def test_checkpoint_retention_rejects_non_positive_or_fractional_values(value):
    with pytest.raises(ValueError, match="positive integer or null"):
        _resolve_checkpoint_retention_max_to_keep(value)


@pytest.mark.parametrize("value", [None, "", "null", 0, "0"])
def test_checkpoint_milestone_interval_can_be_disabled(value):
    assert _resolve_checkpoint_retention_keep_every_n_steps(value) is None


@pytest.mark.parametrize("value, expected", [(1, 1), ("10000", 10000), (4.0, 4)])
def test_checkpoint_milestone_interval_accepts_positive_integers(value, expected):
    assert _resolve_checkpoint_retention_keep_every_n_steps(value) == expected


@pytest.mark.parametrize("value", [True, False, "many"])
def test_checkpoint_milestone_interval_rejects_non_integers(value):
    with pytest.raises(TypeError, match="positive integer or null"):
        _resolve_checkpoint_retention_keep_every_n_steps(value)


@pytest.mark.parametrize("value", [-1, 1.5])
def test_checkpoint_milestone_interval_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="positive integer or null"):
        _resolve_checkpoint_retention_keep_every_n_steps(value)


def test_checkpoint_keep_final_requires_boolean():
    assert _resolve_checkpoint_retention_keep_final(True) is True
    assert _resolve_checkpoint_retention_keep_final(False) is False
    with pytest.raises(TypeError, match="must be a boolean"):
        _resolve_checkpoint_retention_keep_final("true")


@pytest.mark.parametrize("value", [None, "", "null", 0, "0"])
def test_permanent_weights_interval_can_be_disabled(value):
    assert _resolve_permanent_weights_every_n_steps(value) is None


@pytest.mark.parametrize("value, expected", [(1, 1), ("17002", 17002), (4.0, 4)])
def test_permanent_weights_interval_accepts_positive_integers(value, expected):
    assert _resolve_permanent_weights_every_n_steps(value) == expected


@pytest.mark.parametrize("value", [True, False, "many", -1, 1.5])
def test_permanent_weights_interval_rejects_invalid_values(value):
    with pytest.raises((TypeError, ValueError), match="positive integer or null"):
        _resolve_permanent_weights_every_n_steps(value)


def test_checkpoint_retention_prunes_only_old_complete_pairs(tmp_path):
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.state_dir = str(tmp_path / "state")
    trainer.weights_dir = str(tmp_path / "weights")
    trainer.checkpoint_retention_max_to_keep = 2
    state_root = Path(trainer.state_dir)
    weights_root = Path(trainer.weights_dir)
    state_root.mkdir()
    weights_root.mkdir()

    for completed_at, step in enumerate((10, 20, 30, 40), start=1):
        _write_complete_pair(
            state_root,
            weights_root,
            step=step,
            completed_at=completed_at,
        )

    # Interrupted/non-standard artifacts must remain available for diagnosis.
    (state_root / "step_000005").mkdir()
    (weights_root / "step_000007.pt").write_bytes(b"orphan")
    (state_root / "notes").mkdir()

    assert trainer._prune_complete_checkpoint_pairs() == [10, 20]
    assert sorted(path.name for path in state_root.iterdir()) == [
        "notes",
        "step_000005",
        "step_000030",
        "step_000040",
    ]
    assert sorted(path.name for path in weights_root.iterdir()) == [
        "step_000007.pt",
        "step_000030.pt",
        "step_000040.pt",
    ]


def test_checkpoint_retention_keeps_milestones_outside_rolling_budget(tmp_path):
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.state_dir = str(tmp_path / "state")
    trainer.weights_dir = str(tmp_path / "weights")
    trainer.checkpoint_retention_max_to_keep = 2
    trainer.checkpoint_retention_keep_every_n_steps = 10000
    state_root = Path(trainer.state_dir)
    weights_root = Path(trainer.weights_dir)
    state_root.mkdir()
    weights_root.mkdir()

    for completed_at, step in enumerate((9000, 10000, 11000, 12000, 20000, 21000)):
        _write_complete_pair(
            state_root,
            weights_root,
            step=step,
            completed_at=completed_at + 1,
            retention_class=(
                "milestone" if step % 10000 == 0 else "rolling"
            ),
        )

    assert trainer._prune_complete_checkpoint_pairs() == [9000, 11000]
    assert sorted(path.name for path in state_root.iterdir()) == [
        "step_010000",
        "step_012000",
        "step_020000",
        "step_021000",
    ]


def test_checkpoint_retention_honors_permanent_marker_after_policy_changes(tmp_path):
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.state_dir = str(tmp_path / "state")
    trainer.weights_dir = str(tmp_path / "weights")
    trainer.checkpoint_retention_max_to_keep = 1
    trainer.checkpoint_retention_keep_every_n_steps = None
    state_root = Path(trainer.state_dir)
    weights_root = Path(trainer.weights_dir)
    state_root.mkdir()
    weights_root.mkdir()
    permanent_state, permanent_weights = _write_complete_pair(
        state_root,
        weights_root,
        step=10000,
        completed_at=1,
        retention_class="milestone",
    )
    _write_complete_pair(
        state_root, weights_root, step=11000, completed_at=2, retention_class="rolling"
    )
    _write_complete_pair(
        state_root, weights_root, step=12000, completed_at=3, retention_class="rolling"
    )

    assert trainer._prune_complete_checkpoint_pairs() == [11000]
    assert permanent_state.exists() and permanent_weights.exists()


def test_checkpoint_retention_infers_old_marker_milestone_for_compatibility(tmp_path):
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.state_dir = str(tmp_path / "state")
    trainer.weights_dir = str(tmp_path / "weights")
    trainer.checkpoint_retention_max_to_keep = 1
    trainer.checkpoint_retention_keep_every_n_steps = 10000
    state_root = Path(trainer.state_dir)
    weights_root = Path(trainer.weights_dir)
    state_root.mkdir()
    weights_root.mkdir()
    old_state, old_weights = _write_complete_pair(
        state_root, weights_root, step=10000, completed_at=1
    )
    _write_complete_pair(state_root, weights_root, step=11000, completed_at=2)
    _write_complete_pair(state_root, weights_root, step=12000, completed_at=3)

    assert trainer._prune_complete_checkpoint_pairs() == [11000]
    assert old_state.exists() and old_weights.exists()


def test_checkpoint_retention_never_prunes_unknown_retention_class(tmp_path):
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.state_dir = str(tmp_path / "state")
    trainer.weights_dir = str(tmp_path / "weights")
    trainer.checkpoint_retention_max_to_keep = 1
    state_root = Path(trainer.state_dir)
    weights_root = Path(trainer.weights_dir)
    state_root.mkdir()
    weights_root.mkdir()
    state, weights = _write_complete_pair(
        state_root,
        weights_root,
        step=10000,
        completed_at=1,
        retention_class="corrupt",
    )
    _write_complete_pair(
        state_root, weights_root, step=11000, completed_at=2, retention_class="rolling"
    )
    _write_complete_pair(
        state_root, weights_root, step=12000, completed_at=3, retention_class="rolling"
    )

    assert trainer._prune_complete_checkpoint_pairs() == [11000]
    assert state.exists() and weights.exists()


def test_checkpoint_retention_disabled_preserves_all_pairs(tmp_path):
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.state_dir = str(tmp_path / "state")
    trainer.weights_dir = str(tmp_path / "weights")
    trainer.checkpoint_retention_max_to_keep = None
    Path(trainer.state_dir).mkdir()
    Path(trainer.weights_dir).mkdir()
    state = Path(trainer.state_dir) / "step_000001"
    state.mkdir()
    weights = Path(trainer.weights_dir) / "step_000001.pt"
    weights.write_bytes(b"weights")

    assert trainer._prune_complete_checkpoint_pairs() == []
    assert state.exists()
    assert weights.exists()


def test_checkpoint_retention_ignores_preempted_half_written_pair(tmp_path):
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.state_dir = str(tmp_path / "state")
    trainer.weights_dir = str(tmp_path / "weights")
    trainer.checkpoint_retention_max_to_keep = 1
    state_root = Path(trainer.state_dir)
    weights_root = Path(trainer.weights_dir)
    state_root.mkdir()
    weights_root.mkdir()

    old_state, old_weights = _write_complete_pair(
        state_root, weights_root, step=10, completed_at=10
    )
    new_state, new_weights = _write_complete_pair(
        state_root, weights_root, step=20, completed_at=20
    )
    partial_state = state_root / "step_000030"
    partial_state.mkdir()
    (partial_state / "trainer_state.json").write_text("{}", encoding="utf-8")
    partial_weights = weights_root / "step_000030.pt"
    partial_weights.write_bytes(b"partial")

    assert trainer._prune_complete_checkpoint_pairs() == [10]
    assert not old_state.exists()
    assert not old_weights.exists()
    assert new_state.exists() and new_weights.exists()
    assert partial_state.exists() and partial_weights.exists()


def test_checkpoint_retention_uses_completion_order_and_protects_new_branch(tmp_path):
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.state_dir = str(tmp_path / "state")
    trainer.weights_dir = str(tmp_path / "weights")
    trainer.checkpoint_retention_max_to_keep = 1
    state_root = Path(trainer.state_dir)
    weights_root = Path(trainer.weights_dir)
    state_root.mkdir()
    weights_root.mkdir()

    high_state, high_weights = _write_complete_pair(
        state_root, weights_root, step=900, completed_at=100
    )
    branch_state, branch_weights = _write_complete_pair(
        state_root, weights_root, step=120, completed_at=200
    )

    assert trainer._prune_complete_checkpoint_pairs(
        protect_step_tag="step_000120"
    ) == [900]
    assert not high_state.exists() and not high_weights.exists()
    assert branch_state.exists() and branch_weights.exists()


def test_checkpoint_retention_waits_for_completion_marker_before_cleanup(tmp_path):
    model = _WritingCheckpointModel()
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.model = model
    trainer.accelerator = _FakeAccelerator(model)
    trainer.checkpoint_contract = None
    trainer.epoch = 0
    trainer.batch_in_epoch = 0
    trainer.weights_dir = str(tmp_path / "weights")
    trainer.state_dir = str(tmp_path / "state")
    trainer.checkpoint_retention_max_to_keep = 1
    state_root = Path(trainer.state_dir)
    weights_root = Path(trainer.weights_dir)
    state_root.mkdir()
    weights_root.mkdir()
    old_state, old_weights = _write_complete_pair(
        state_root, weights_root, step=10, completed_at=10
    )

    original_write_marker = trainer._write_checkpoint_complete_marker

    def verify_then_commit(*, state_path, weights_path, retention_class="rolling"):
        # Retention must not run while the new pair is still uncommitted.
        assert trainer._prune_complete_checkpoint_pairs() == []
        assert old_state.exists() and old_weights.exists()
        return original_write_marker(
            state_path=state_path,
            weights_path=weights_path,
            retention_class=retention_class,
        )

    trainer._write_checkpoint_complete_marker = verify_then_commit
    trainer.global_step = 20
    trainer.save_checkpoint()

    assert not old_state.exists() and not old_weights.exists()
    assert (state_root / "step_000020" / "checkpoint_complete.json").is_file()
    assert (weights_root / "step_000020.pt").is_file()


def test_interrupted_same_step_overwrite_revokes_old_completion_marker(tmp_path):
    class _InterruptedModel(_ContractAwareModel):
        def save_checkpoint(self, path, **kwargs):
            Path(path).write_bytes(b"partial-new-weights")
            raise RuntimeError("preempted while writing weights")

    model = _InterruptedModel()
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.model = model
    trainer.accelerator = _FakeAccelerator(model)
    trainer.checkpoint_contract = None
    trainer.global_step = 20
    trainer.epoch = 0
    trainer.batch_in_epoch = 0
    trainer.weights_dir = str(tmp_path / "weights")
    trainer.state_dir = str(tmp_path / "state")
    trainer.checkpoint_retention_max_to_keep = 1
    state_root = Path(trainer.state_dir)
    weights_root = Path(trainer.weights_dir)
    state_root.mkdir()
    weights_root.mkdir()
    state, weights = _write_complete_pair(
        state_root, weights_root, step=20, completed_at=20
    )

    with pytest.raises(RuntimeError, match="preempted"):
        trainer.save_checkpoint()

    assert state.exists() and weights.exists()
    assert not (state / "checkpoint_complete.json").exists()
    assert trainer._prune_complete_checkpoint_pairs() == []


def test_save_checkpoint_applies_retention_after_new_pair_is_complete(tmp_path):
    model = _WritingCheckpointModel()
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.model = model
    trainer.accelerator = _FakeAccelerator(model)
    trainer.checkpoint_contract = None
    trainer.epoch = 0
    trainer.batch_in_epoch = 0
    trainer.weights_dir = str(tmp_path / "weights")
    trainer.state_dir = str(tmp_path / "state")
    trainer.checkpoint_retention_max_to_keep = 1
    Path(trainer.weights_dir).mkdir()
    Path(trainer.state_dir).mkdir()

    trainer.global_step = 10
    trainer.save_checkpoint()
    trainer.global_step = 20
    trainer.save_checkpoint()

    assert not (Path(trainer.state_dir) / "step_000010").exists()
    assert not (Path(trainer.weights_dir) / "step_000010.pt").exists()
    assert (Path(trainer.state_dir) / "step_000020" / "trainer_state.json").is_file()
    assert (Path(trainer.state_dir) / "step_000020" / "checkpoint_complete.json").is_file()
    assert (Path(trainer.weights_dir) / "step_000020.pt").is_file()


def test_save_checkpoint_commits_milestone_and_final_as_permanent(tmp_path):
    model = _WritingCheckpointModel()
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.model = model
    trainer.accelerator = _FakeAccelerator(model)
    trainer.checkpoint_contract = None
    trainer.epoch = 0
    trainer.batch_in_epoch = 0
    trainer.weights_dir = str(tmp_path / "weights")
    trainer.state_dir = str(tmp_path / "state")
    trainer.checkpoint_retention_max_to_keep = 2
    trainer.checkpoint_retention_keep_every_n_steps = 10000
    trainer.checkpoint_retention_keep_final = True
    trainer.final_checkpoint_mode = "full"
    Path(trainer.weights_dir).mkdir()
    Path(trainer.state_dir).mkdir()

    for step in (10000, 11000, 12000):
        trainer.global_step = step
        trainer.save_checkpoint()
    trainer.global_step = 12500
    trainer.save_final_checkpoint()

    milestone = json.loads(
        (Path(trainer.state_dir) / "step_010000" / "checkpoint_complete.json").read_text()
    )
    final = json.loads(
        (Path(trainer.state_dir) / "step_012500" / "checkpoint_complete.json").read_text()
    )
    assert milestone["retention_class"] == "milestone"
    assert final["retention_class"] == "final"
    assert sorted(path.name for path in Path(trainer.state_dir).iterdir()) == [
        "step_010000",
        "step_011000",
        "step_012000",
        "step_012500",
    ]


def test_persist_fixed_validation_summary_replaces_same_step(tmp_path):
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.accelerator = SimpleNamespace(is_main_process=True)
    trainer.fixed_validation_report_dir = str(tmp_path)
    trainer.initialization_fingerprint_sha256 = "ab" * 32
    trainer._fixed_validation_window_set_sha256 = "cd" * 32
    trainer.fixed_validation_seed = 42
    metrics = {"num_windows": 1000, "fixed_validation_seed": 42}
    trainer._persist_fixed_validation_metrics(metrics | {"step": 5, "loss_total": 2.0})
    trainer._persist_fixed_validation_metrics(metrics | {"step": 5, "loss_total": 1.0})
    trainer._persist_fixed_validation_metrics(metrics | {"step": 10, "loss_total": 0.5})
    payload = json.loads((tmp_path / "summary.json").read_text())
    assert payload["status"] == "passed"
    assert payload["steps"] == [5, 10]
    assert payload["num_windows"] == 1000
    assert payload["fixed_validation_seed"] == 42
    assert payload["initialization_fingerprint_sha256"] == "ab" * 32
    assert payload["window_set_sha256"] == "cd" * 32
    assert [row["loss_total"] for row in payload["records"]] == [1.0, 0.5]
    assert json.loads((tmp_path / "step-000005.json").read_text())["loss_total"] == 1.0
    assert len(list(tmp_path.glob("step-000005.json"))) == 1


def test_persist_fixed_validation_missing_summary_initializes_history(tmp_path):
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.accelerator = SimpleNamespace(is_main_process=True)
    trainer.fixed_validation_report_dir = str(tmp_path)
    trainer.initialization_fingerprint_sha256 = "ab" * 32
    trainer._fixed_validation_window_set_sha256 = "cd" * 32

    trainer._persist_fixed_validation_metrics(
        {"step": 0, "num_windows": 1000, "loss_total": 1.0, "fixed_validation_seed": 42}
    )

    payload = json.loads((tmp_path / "summary.json").read_text())
    assert payload["steps"] == [0]


def test_persist_fixed_validation_corrupt_summary_fails_without_overwrite(tmp_path):
    summary_path = tmp_path / "summary.json"
    summary_path.write_text('{"records": [', encoding="utf-8")
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.accelerator = SimpleNamespace(is_main_process=True)
    trainer.fixed_validation_report_dir = str(tmp_path)
    trainer.initialization_fingerprint_sha256 = "ab" * 32
    trainer._fixed_validation_window_set_sha256 = "cd" * 32

    with pytest.raises(json.JSONDecodeError):
        trainer._persist_fixed_validation_metrics(
            {"step": 1, "num_windows": 1000, "loss_total": 0.5, "fixed_validation_seed": 42}
        )

    assert summary_path.read_text(encoding="utf-8") == '{"records": ['
    assert not (tmp_path / "step-000001.json").exists()


def test_persist_fixed_validation_summary_read_oserror_fails_loud(tmp_path, monkeypatch):
    summary_path = tmp_path / "summary.json"
    summary_path.write_text('{"records": []}', encoding="utf-8")
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.accelerator = SimpleNamespace(is_main_process=True)
    trainer.fixed_validation_report_dir = str(tmp_path)
    trainer.initialization_fingerprint_sha256 = "ab" * 32
    trainer._fixed_validation_window_set_sha256 = "cd" * 32
    original_read_text = Path.read_text

    def fail_summary_read(path, *args, **kwargs):
        if path == summary_path:
            raise OSError("transient shared-filesystem read failure")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fail_summary_read)
    with pytest.raises(OSError, match="transient shared-filesystem read failure"):
        trainer._persist_fixed_validation_metrics(
            {"step": 1, "num_windows": 1000, "loss_total": 0.5, "fixed_validation_seed": 42}
        )

    assert not (tmp_path / "step-000001.json").exists()


def test_atomic_write_json_replace_failure_preserves_destination(tmp_path, monkeypatch):
    output = tmp_path / "summary.json"
    output.write_text('{"old": true}\n', encoding="utf-8")

    def fail_replace(source, destination):
        raise OSError("replace failed")

    monkeypatch.setattr("fastwam.trainer.os.replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        _atomic_write_json(output, {"old": False})

    assert output.read_text(encoding="utf-8") == '{"old": true}\n'
    assert list(tmp_path.glob(".summary.json.*.tmp")) == []


def test_persist_fixed_validation_rejects_identity_change(tmp_path):
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.accelerator = SimpleNamespace(is_main_process=True)
    trainer.fixed_validation_report_dir = str(tmp_path)
    trainer.initialization_fingerprint_sha256 = "ab" * 32
    trainer._fixed_validation_window_set_sha256 = "cd" * 32
    trainer._persist_fixed_validation_metrics(
        {"step": 0, "num_windows": 1000, "loss_total": 1.0, "fixed_validation_seed": 42}
    )

    trainer.initialization_fingerprint_sha256 = "ef" * 32
    with pytest.raises(RuntimeError, match="initialization_fingerprint_sha256 changed"):
        trainer._persist_fixed_validation_metrics(
            {"step": 1, "num_windows": 1000, "loss_total": 0.5, "fixed_validation_seed": 42}
        )


def test_persist_fixed_validation_rejects_seed_change(tmp_path):
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.accelerator = SimpleNamespace(is_main_process=True)
    trainer.fixed_validation_report_dir = str(tmp_path)
    trainer.initialization_fingerprint_sha256 = "ab" * 32
    trainer._fixed_validation_window_set_sha256 = "cd" * 32
    trainer._persist_fixed_validation_metrics(
        {"step": 0, "num_windows": 1000, "loss_total": 1.0, "fixed_validation_seed": 42}
    )
    with pytest.raises(RuntimeError, match="fixed_validation_seed changed"):
        trainer._persist_fixed_validation_metrics(
            {"step": 1, "num_windows": 1000, "loss_total": 0.5, "fixed_validation_seed": 43}
        )


def test_save_final_checkpoint_weights_only_skips_full_state(tmp_path):
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.final_checkpoint_mode = "weights_only"
    trainer.accelerator = _FakeAccelerator()
    trainer.global_step = 3
    trainer.weights_dir = str(tmp_path / "weights")
    trainer.checkpoint_contract = None
    called = {"weights": 0, "full": 0}
    trainer._save_weights_checkpoint = lambda step_tag: called.__setitem__("weights", called["weights"] + 1) or "weights"
    trainer.save_checkpoint = lambda: called.__setitem__("full", called["full"] + 1)
    result = trainer.save_final_checkpoint()
    assert result == {"weights_path": "weights", "state_path": None}
    assert called == {"weights": 1, "full": 0}


def test_resume_rejects_contract_mismatch_before_accelerator_load(tmp_path):
    actual = _contract(action_dim=14)
    expected = _contract(action_dim=23)
    write_checkpoint_contract(tmp_path / "checkpoint_contract.json", actual)
    (tmp_path / "trainer_state.json").write_text(
        json.dumps(
            {
                "global_step": 5,
                "epoch": 0,
                "batch_in_epoch": 5,
                "checkpoint_contract_sha256": actual.contract_sha256,
            }
        ),
        encoding="utf-8",
    )
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.checkpoint_contract = expected
    trainer.accelerator = _FakeAccelerator()

    with pytest.raises(CheckpointContractMismatchError, match="action_dim"):
        trainer.load_training_state(str(tmp_path))

    assert trainer.accelerator.load_calls == []


def test_resume_rejects_standard_step_without_completion_marker(tmp_path):
    state_root = tmp_path / "state"
    weights_root = tmp_path / "weights"
    state = state_root / "step_000005"
    state.mkdir(parents=True)
    weights_root.mkdir()
    (state / "trainer_state.json").write_text(
        json.dumps({"global_step": 5, "epoch": 0, "batch_in_epoch": 5}),
        encoding="utf-8",
    )
    (weights_root / "step_000005.pt").write_bytes(b"partial")
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.weights_dir = str(weights_root)
    trainer.checkpoint_contract = None
    trainer.accelerator = _FakeAccelerator()

    with pytest.raises(CheckpointContractIntegrityError, match="incomplete checkpoint"):
        trainer.load_training_state(str(state))

    assert trainer.accelerator.load_calls == []


def test_resume_standard_step_pairs_weights_from_source_checkpoint_root(tmp_path):
    source_root = tmp_path / "source-run" / "checkpoints"
    state_root = source_root / "state"
    weights_root = source_root / "weights"
    state_root.mkdir(parents=True)
    weights_root.mkdir()
    state, _ = _write_complete_pair(
        state_root,
        weights_root,
        step=5,
        completed_at=5,
        world_size=8,
    )
    (state / "trainer_state.json").write_text(
        json.dumps({"global_step": 5, "epoch": 1, "batch_in_epoch": 2}),
        encoding="utf-8",
    )
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    # Deliberately point the new run elsewhere.  Resume validation must bind
    # the state to the source run's sibling weights directory.
    trainer.weights_dir = str(tmp_path / "new-run" / "checkpoints" / "weights")
    trainer.checkpoint_contract = None
    trainer.accelerator = _FakeAccelerator()
    trainer.batch_size = 2
    trainer.train_sampler = SimpleNamespace(
        set_epoch_offset=lambda epoch: None,
        set_resume_batch_offset=lambda offset: None,
    )

    trainer.load_training_state(str(state))

    assert trainer.accelerator.load_calls == [str(state)]
    assert (trainer.global_step, trainer.epoch, trainer.batch_in_epoch) == (5, 1, 2)


def test_resume_loads_state_only_after_contract_and_hash_validate(tmp_path):
    contract = _contract()
    write_checkpoint_contract(tmp_path / "checkpoint_contract.json", contract)
    (tmp_path / "trainer_state.json").write_text(
        json.dumps(
            {
                "global_step": 41,
                "epoch": 2,
                "batch_in_epoch": 7,
                "checkpoint_contract_sha256": contract.contract_sha256,
            }
        ),
        encoding="utf-8",
    )
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.checkpoint_contract = contract
    trainer.accelerator = _FakeAccelerator()
    trainer.batch_size = 2
    trainer.train_sampler = SimpleNamespace(
        set_epoch_offset=lambda epoch: setattr(trainer, "sampler_epoch", epoch),
        set_resume_batch_offset=lambda offset: setattr(trainer, "sampler_offset", offset),
    )

    trainer.load_training_state(str(tmp_path))

    assert trainer.accelerator.load_calls == [str(tmp_path)]
    assert (trainer.global_step, trainer.epoch, trainer.batch_in_epoch) == (41, 2, 7)
    assert (trainer.sampler_epoch, trainer.sampler_offset) == (2, 7)


def test_terminal_step_is_deferred_to_final_checkpoint_epilogue():
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.global_step = 100
    trainer.max_steps = 100
    trainer.save_every = 10
    trainer.accelerator = SimpleNamespace(is_main_process=False)
    calls = []

    def save_checkpoint():
        calls.append(trainer.global_step)
        return {"weights_path": "weights.pt", "state_path": "state"}

    trainer.save_checkpoint = save_checkpoint

    assert trainer._save_checkpoint_after_step() is False
    assert calls == []


def test_permanent_weights_checkpoint_is_saved_without_train_state():
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.global_step = 17002
    trainer.max_steps = 85010
    trainer.save_every = 1000
    trainer.permanent_weights_every_n_steps = 17002
    trainer.accelerator = _FakeAccelerator()
    calls = []
    trainer._save_permanent_weights_checkpoint = lambda tag: calls.append(tag) or f"{tag}.pt"
    trainer.save_checkpoint = lambda: pytest.fail("full checkpoint must not be saved")

    assert trainer._save_checkpoint_after_step() is False
    assert calls == ["step_017002"]


def test_permanent_weights_checkpoint_failure_preserves_previous_file(tmp_path):
    class _FailingCheckpointModel(_ContractAwareModel):
        def save_checkpoint(self, path, **kwargs):
            Path(path).write_bytes(b"partial")
            raise RuntimeError("interrupted save")

    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.model = _FailingCheckpointModel()
    trainer.accelerator = _FakeAccelerator(trainer.model)
    trainer.checkpoint_contract = None
    trainer.global_step = 17002
    trainer.permanent_weights_dir = str(tmp_path)
    destination = tmp_path / "step_017002.pt"
    destination.write_bytes(b"previous-complete")

    with pytest.raises(RuntimeError, match="interrupted save"):
        trainer._save_permanent_weights_checkpoint("step_017002")

    assert destination.read_bytes() == b"previous-complete"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["step_017002.pt"]


def test_final_two_epoch_weight_is_hardlinked_into_permanent_directory(tmp_path):
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.final_checkpoint_mode = "full"
    trainer.global_step = 85010
    trainer.permanent_weights_every_n_steps = 17002
    trainer.permanent_weights_dir = str(tmp_path / "permanent_weights")
    Path(trainer.permanent_weights_dir).mkdir()
    source = tmp_path / "weights" / "step_085010.pt"
    source.parent.mkdir()
    source.write_bytes(b"final-weights")
    trainer.accelerator = _FakeAccelerator()
    trainer.save_checkpoint = lambda **kwargs: {
        "weights_path": str(source),
        "state_path": "state/step_085010",
    }

    result = trainer.save_final_checkpoint()
    published = Path(trainer.permanent_weights_dir) / "step_085010.pt"

    assert result["weights_path"] == str(source)
    assert published.samefile(source)


def test_training_summary_contains_comparison_identities(tmp_path):
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.accelerator = SimpleNamespace(is_main_process=True)
    trainer.output_dir = str(tmp_path)
    trainer.initialization_fingerprint_sha256 = "ab" * 32
    trainer._fixed_validation_window_set_sha256 = "cd" * 32
    trainer.fixed_validation_seed = 42
    trainer.optimizer_steps_observed = 2
    trainer.clip_events = 1
    trainer.optimizer_skipped = 0
    trainer._summary_losses = [
        {"loss_total": 2.0, "loss_action": 1.5},
        {"loss_total": 1.0, "loss_action": 0.5},
    ]
    trainer._summary_grad_norms = [0.5, 1.5]
    trainer._summary_samples = 128
    trainer.run_start_time = time.perf_counter() - 1.0

    trainer._write_training_summary()

    payload = json.loads((tmp_path / "training-summary.json").read_text())
    assert payload["initialization_fingerprint_sha256"] == "ab" * 32
    assert payload["fixed_validation_window_set_sha256"] == "cd" * 32
    assert payload["fixed_validation_seed"] == 42
    assert payload["optimizer_steps"] == 2
    assert payload["clip_event_rate"] == 0.5
    assert payload["train_loss_mean"] == {"loss_action": 1.0, "loss_total": 1.5}


class _SizedDataset:
    def __len__(self):
        return 128


def _resume_probe_trainer(tmp_path):
    parameter = __import__("torch").nn.Parameter(__import__("torch").tensor([1.0, 2.0]))
    optimizer = __import__("torch").optim.AdamW([parameter], lr=1e-3)
    optimizer.zero_grad()
    parameter.grad = __import__("torch").ones_like(parameter)
    optimizer.step()
    scheduler = __import__("torch").optim.lr_scheduler.ConstantLR(
        optimizer, factor=1.0, total_iters=100
    )
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.accelerator = _FakeAccelerator()
    trainer.accelerator.num_processes = 1
    trainer.optimizer = optimizer
    trainer.scheduler = scheduler
    trainer.train_dataset = _SizedDataset()
    trainer.train_sampler = __import__(
        "fastwam.utils.samplers", fromlist=["ResumableEpochSampler"]
    ).ResumableEpochSampler(
        trainer.train_dataset, seed=42, batch_size=2, num_processes=1
    )
    trainer.batch_size = 2
    trainer.gradient_accumulation_steps = 1
    trainer.num_workers = 0
    trainer.global_step = 1
    trainer.epoch = 0
    trainer.batch_in_epoch = 1
    trainer.checkpoint_contract = _contract()
    trainer.resume_validation_enabled = True
    trainer.resume_validation_continuation_steps = 2
    trainer.resume_validation_report_path = str(tmp_path / "resume-report.json")
    trainer._resume_validation_start_probe = None
    trainer._resume_validation_target_step = None
    return trainer


def test_resume_probe_validates_full_state_and_sampler_continuity(tmp_path):
    trainer = _resume_probe_trainer(tmp_path)
    write_checkpoint_contract(tmp_path / "checkpoint_contract.json", trainer.checkpoint_contract)
    payload = {
        "global_step": 1,
        "epoch": 0,
        "batch_in_epoch": 1,
        "checkpoint_contract_sha256": trainer.checkpoint_contract.contract_sha256,
    }
    (tmp_path / "trainer_state.json").write_text(json.dumps(payload), encoding="utf-8")
    probe = trainer._resume_probe_payload()
    (tmp_path / "resume_probe_rank_000.json").write_text(
        json.dumps(probe), encoding="utf-8"
    )

    trainer.load_training_state(str(tmp_path))

    assert trainer._resume_validation_start_probe == probe
    assert trainer._resume_validation_target_step == 3


def test_resume_probe_allows_compatible_contract_with_new_runtime_provenance(tmp_path):
    trainer = _resume_probe_trainer(tmp_path)
    saved_contract = _contract()
    trainer.checkpoint_contract = CheckpointContract(
        compatibility=saved_contract.compatibility,
        provenance={"test": True, "runtime": "resumed"},
    )
    write_checkpoint_contract(tmp_path / "checkpoint_contract.json", saved_contract)
    payload = {
        "global_step": 1,
        "epoch": 0,
        "batch_in_epoch": 1,
        "checkpoint_contract_sha256": saved_contract.contract_sha256,
    }
    (tmp_path / "trainer_state.json").write_text(json.dumps(payload), encoding="utf-8")
    probe = trainer._resume_probe_payload()
    probe["checkpoint_contract_sha256"] = saved_contract.contract_sha256
    (tmp_path / "resume_probe_rank_000.json").write_text(
        json.dumps(probe), encoding="utf-8"
    )

    trainer.load_training_state(str(tmp_path))

    assert trainer.accelerator.load_calls == [str(tmp_path)]
    assert (
        trainer._resume_validation_start_probe["checkpoint_contract_sha256"]
        == trainer.checkpoint_contract.contract_sha256
    )
    assert trainer._resume_validation_start_probe != probe


def test_resume_probe_rejects_tampered_saved_contract_hash_before_load(tmp_path):
    trainer = _resume_probe_trainer(tmp_path)
    trainer.accelerator.num_processes = 2
    write_checkpoint_contract(
        tmp_path / "checkpoint_contract.json", trainer.checkpoint_contract
    )
    payload = {
        "global_step": 1,
        "epoch": 0,
        "batch_in_epoch": 1,
        "checkpoint_contract_sha256": trainer.checkpoint_contract.contract_sha256,
    }
    (tmp_path / "trainer_state.json").write_text(json.dumps(payload), encoding="utf-8")
    local_probe = trainer._resume_probe_payload()
    (tmp_path / "resume_probe_rank_000.json").write_text(
        json.dumps(local_probe), encoding="utf-8"
    )
    tampered_probe = {
        **local_probe,
        "rank": 1,
        "checkpoint_contract_sha256": "0" * 64,
    }
    (tmp_path / "resume_probe_rank_001.json").write_text(
        json.dumps(tampered_probe), encoding="utf-8"
    )

    with pytest.raises(
        CheckpointContractIntegrityError,
        match="resume_probe_rank_001.json",
    ):
        trainer.load_training_state(str(tmp_path))

    assert trainer.accelerator.load_calls == []


def test_resume_probe_rejects_optimizer_state_mismatch(tmp_path):
    trainer = _resume_probe_trainer(tmp_path)
    write_checkpoint_contract(tmp_path / "checkpoint_contract.json", trainer.checkpoint_contract)
    payload = {
        "global_step": 1,
        "epoch": 0,
        "batch_in_epoch": 1,
        "checkpoint_contract_sha256": trainer.checkpoint_contract.contract_sha256,
    }
    (tmp_path / "trainer_state.json").write_text(json.dumps(payload), encoding="utf-8")
    probe = trainer._resume_probe_payload()
    probe["optimizer"]["sha256"] = "0" * 64
    (tmp_path / "resume_probe_rank_000.json").write_text(
        json.dumps(probe), encoding="utf-8"
    )

    with pytest.raises(CheckpointContractIntegrityError, match="optimizer"):
        trainer.load_training_state(str(tmp_path))


def test_resume_continuation_writes_passed_report(tmp_path):
    trainer = _resume_probe_trainer(tmp_path)
    trainer.accelerator.num_processes = 1
    trainer.train_sampler = __import__(
        "fastwam.utils.samplers", fromlist=["ResumableEpochSampler"]
    ).ResumableEpochSampler(
        trainer.train_dataset, seed=42, batch_size=2, num_processes=1
    )
    trainer._resume_validation_start_probe = trainer._resume_probe_payload()
    trainer._resume_validation_target_step = 3

    parameter = trainer.optimizer.param_groups[0]["params"][0]
    for _ in range(2):
        trainer.optimizer.zero_grad()
        parameter.grad = __import__("torch").ones_like(parameter)
        trainer.optimizer.step()
        trainer.scheduler.step()
    trainer.global_step = 3
    trainer.batch_in_epoch = 3

    assert trainer._finish_resume_validation_if_ready() is True
    report = json.loads(
        Path(trainer.resume_validation_report_path).read_text(encoding="utf-8")
    )
    assert report["status"] == "passed"
    assert report["checkpoint_global_step"] == 1
    assert report["final_global_step"] == 3
    assert all(report["ranks"][0]["checks"].values())
