# Copyright (c) 2026 Xiaomi Corporation

import json
from enum import Enum
from pathlib import Path
import sys
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
    _distribution_summary,
    _path_size_bytes,
)


class _SingleRankAccelerator:
    is_main_process = True
    num_processes = 1
    process_index = 0
    device = torch.device("cpu")

    def gather(self, tensor):
        return tensor

    def wait_for_everyone(self):
        return None


class _TwoRankAccelerator(_SingleRankAccelerator):
    num_processes = 2

    def __init__(self, timings, memory):
        self.timings = torch.tensor(timings, dtype=torch.float64)
        self.memory = torch.tensor(memory, dtype=torch.int64).reshape(-1)

    def gather(self, tensor):
        if tensor.ndim == 2:
            return self.timings
        return self.memory


def test_atomic_json_and_path_size(tmp_path: Path):
    output = tmp_path / "reports" / "profile.json"
    _atomic_write_json(output, {"value": 3})
    assert json.loads(output.read_text(encoding="utf-8")) == {"value": 3}
    assert not list(output.parent.glob(".profile.json.*.tmp"))
    assert _path_size_bytes(output.parent) == output.stat().st_size


def test_distribution_summary_reports_median_p90_and_mean():
    summary = _distribution_summary([1.0, 2.0, 3.0, 4.0])
    assert summary["median"] == pytest.approx(2.5)
    assert summary["p90"] == pytest.approx(3.7)
    assert summary["mean"] == pytest.approx(2.5)
    assert summary["max"] == pytest.approx(4.0)


def test_finalize_profile_report_computes_effective_throughput(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.accelerator = _SingleRankAccelerator()
    class _DistributedType(Enum):
        DEEPSPEED = "DEEPSPEED"

    trainer.accelerator.distributed_type = _DistributedType.DEEPSPEED
    trainer.profiling_enabled = True
    trainer.profiling_report_path = str(tmp_path / "profile.json")
    trainer.profiling_measure_steps = 2
    trainer.profiling_warmup_steps = 1
    trainer.profiling_stop_after_profile = True
    trainer._profiling_start_global_step = 10
    trainer._profiling_records = [
        [0.1, 0.2, 0.4, 0.1, 0.1, 0.9, 1.0],
        [0.2, 0.4, 0.8, 0.2, 0.2, 1.8, 2.0],
    ]
    trainer._profiling_report = None
    trainer._profiling_checkpoint = None
    trainer._profiling_optimizer_skipped_steps = 0
    trainer._gradient_gate_report = {
        "schema": "fastwam_gradient_gate.v1",
        "status": "passed",
    }
    trainer.batch_size = 2
    trainer.gradient_accumulation_steps = 4
    trainer.global_step = 13
    trainer.max_steps = 100

    trainer._finalize_profiling_report()

    report = json.loads(Path(trainer.profiling_report_path).read_text(encoding="utf-8"))
    assert "resolved_training_contract" in report
    assert "runtime_contract" not in report
    assert report["resolved_training_contract"]["distributed_type"] == "DEEPSPEED"
    assert report["effective_global_batch"] == 8
    assert report["dataloader"] == {
        "num_workers_per_rank": 0,
        "prefetch_factor": None,
        "persistent_workers": False,
        "pin_memory": False,
    }
    assert report["effective_samples_per_second"] == pytest.approx(16 / 3)
    assert report["throughput_basis"] == "distributed_end_to_end_step_wall"
    assert report["timing_seconds"]["forward"]["median"] == pytest.approx(0.3)
    assert report["timing_seconds"]["backward"]["median"] == pytest.approx(0.6)
    assert report["timing_seconds"]["forward_backward"]["median"] == pytest.approx(0.9)
    assert report["timing_seconds"]["optimizer"]["median"] == pytest.approx(0.15)
    assert report["timing_seconds"]["metrics_logging"]["median"] == pytest.approx(0.15)
    assert report["timing_seconds"]["total_step"]["median"] == pytest.approx(1.35)
    assert report["timing_seconds"]["step_wall"]["median"] == pytest.approx(1.5)
    assert report["distributed_step_seconds"]["p90"] == pytest.approx(1.71)
    assert report["distributed_step_wall_seconds"]["p90"] == pytest.approx(1.9)
    assert report["measured_global_step_start"] == 12
    assert report["measured_global_step_end"] == 13
    assert report["configured_max_steps"] == 100
    assert report["estimated_configured_run_wall_seconds"] == pytest.approx(150.0)
    assert report["estimated_configured_remaining_wall_seconds"] == pytest.approx(130.5)
    assert report["estimated_wall_seconds_per_million_samples"] == pytest.approx(
        187_500.0
    )
    assert report["wall_time_estimate_excludes"] == ["evaluation", "checkpoint_save"]
    assert report["rank_peak_memory"] == [
        {"rank": 0, "max_allocated_bytes": 0, "max_reserved_bytes": 0}
    ]
    assert report["checkpoint"]["measured"] is False
    assert report["gradient_gate"] == trainer._gradient_gate_report


def test_profile_completion_can_stop_without_checkpoint(monkeypatch):
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.profiling_enabled = True
    trainer.profiling_warmup_steps = 1
    trainer.profiling_measure_steps = 2
    trainer.profiling_stop_after_profile = True
    trainer._profiling_completed_steps = 1
    trainer._profiling_step_data_seconds = 0.1
    trainer._profiling_step_forward_seconds = 0.1
    trainer._profiling_step_backward_seconds = 0.1
    trainer._profiling_step_optimizer_seconds = 0.1
    trainer._profiling_step_wall_started_at = 9.0
    trainer._profiling_records = []
    trainer._profiling_optimizer_skipped_steps = 0
    trainer._cuda_synchronize = lambda: None
    finalized = []
    trainer._finalize_profiling_report = lambda: finalized.append(True)

    times = iter([10.4, 20.6])
    monkeypatch.setattr("fastwam.trainer.time.perf_counter", lambda: next(times))

    assert trainer._profiling_finish_optimizer_step(10.0) is False
    trainer._profiling_step_data_seconds = 0.2
    trainer._profiling_step_forward_seconds = 0.2
    trainer._profiling_step_backward_seconds = 0.2
    trainer._profiling_step_optimizer_seconds = 0.2
    trainer._profiling_step_wall_started_at = 19.0
    assert trainer._profiling_finish_optimizer_step(20.0) is True
    assert finalized == [True]
    assert len(trainer._profiling_records) == 2


def test_grad_accumulation_sums_micro_step_phase_times(monkeypatch):
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.profiling_enabled = True
    trainer.profiling_warmup_steps = 0
    trainer.profiling_measure_steps = 1
    trainer.profiling_stop_after_profile = False
    trainer._profiling_completed_steps = 0
    trainer._profiling_step_data_seconds = 0.0
    trainer._profiling_step_forward_seconds = 0.0
    trainer._profiling_step_backward_seconds = 0.0
    trainer._profiling_step_optimizer_seconds = 0.0
    trainer._profiling_step_wall_started_at = 0.0
    trainer._profiling_records = []
    trainer._profiling_optimizer_skipped_steps = 0
    trainer._cuda_synchronize = lambda: None
    trainer._finalize_profiling_report = lambda: None

    times = iter([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    monkeypatch.setattr("fastwam.trainer.time.perf_counter", lambda: next(times))
    trainer._profiling_record_data_wait(0.0)
    trainer._profiling_record_forward(0.1)
    trainer._profiling_record_backward(0.2)
    trainer._profiling_record_data_wait(0.3)
    trainer._profiling_record_forward(0.4)
    trainer._profiling_record_backward(0.5)
    post_optimizer_started_at = trainer._profiling_record_optimizer(0.6)

    assert trainer._profiling_finish_optimizer_step(post_optimizer_started_at) is False
    assert trainer._profiling_records[0] == pytest.approx(
        [0.2, 0.2, 0.2, 0.1, 0.1, 0.8, 0.8]
    )


def test_multi_rank_report_uses_slowest_rank_for_throughput(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    timings = [
        [0.1, 0.2, 0.4, 0.1, 0.1, 0.9, 1.0],
        [0.2, 0.4, 0.8, 0.2, 0.2, 1.8, 2.0],
        [0.3, 0.6, 1.2, 0.3, 0.3, 2.7, 3.0],
        [0.4, 0.8, 1.6, 0.4, 0.4, 3.6, 4.0],
    ]
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.accelerator = _TwoRankAccelerator(
        timings,
        memory=[[100, 200], [300, 400]],
    )
    trainer.profiling_enabled = True
    trainer.profiling_report_path = str(tmp_path / "profile.json")
    trainer.profiling_measure_steps = 2
    trainer.profiling_warmup_steps = 0
    trainer._profiling_start_global_step = 0
    trainer._profiling_records = timings[:2]
    trainer._profiling_report = None
    trainer._profiling_checkpoint = None
    trainer._profiling_optimizer_skipped_steps = 0
    trainer.batch_size = 2
    trainer.gradient_accumulation_steps = 2
    trainer.global_step = 2
    trainer.max_steps = 10

    trainer._finalize_profiling_report()

    report = json.loads(Path(trainer.profiling_report_path).read_text(encoding="utf-8"))
    assert report["effective_global_batch"] == 8
    assert report["distributed_step_seconds"]["median"] == pytest.approx(3.15)
    assert report["distributed_step_wall_seconds"]["median"] == pytest.approx(3.5)
    assert report["effective_samples_per_second"] == pytest.approx(16 / 7)
    assert report["rank_peak_memory"] == [
        {"rank": 0, "max_allocated_bytes": 100, "max_reserved_bytes": 200},
        {"rank": 1, "max_allocated_bytes": 300, "max_reserved_bytes": 400},
    ]


def test_checkpoint_measurement_updates_completed_report(tmp_path: Path):
    weights = tmp_path / "weights.pt"
    state = tmp_path / "state"
    weights.write_bytes(b"weights")
    state.mkdir()
    (state / "rank0.bin").write_bytes(b"state-data")

    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.accelerator = _SingleRankAccelerator()
    trainer.profiling_enabled = True
    trainer.profiling_report_path = str(tmp_path / "profile.json")
    trainer._profiling_report = {"checkpoint": {"measured": False}}
    trainer._profiling_checkpoint = None
    trainer.global_step = 160

    trainer._record_checkpoint_profile(
        2.5,
        {"weights_path": str(weights), "state_path": str(state)},
    )

    report = json.loads(Path(trainer.profiling_report_path).read_text(encoding="utf-8"))
    assert report["checkpoint"]["measured"] is True
    assert report["checkpoint"]["elapsed_seconds"]["max"] == pytest.approx(2.5)
    assert report["checkpoint"]["weights_size_bytes"] == len(b"weights")
    assert report["checkpoint"]["state_size_bytes"] == len(b"state-data")
    assert report["checkpoint"]["global_step"] == 160
