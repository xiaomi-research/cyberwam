# Copyright (c) 2026 Xiaomi Corporation

from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

import fastwam.qualitative_validation as qualitative
from fastwam.qualitative_validation import (
    QualitativeValidationConfig,
    qualitative_rank_windows,
    qualitative_video_filename,
    qualitative_video_window_positions,
    qualitative_window_seed,
    run_qualitative_validation,
    run_trainer_qualitative_validation,
    select_qualitative_windows,
)


def _fixed_windows(tasks=100, anchors=10):
    return [
        (task * anchors + anchor, 1000 + task, task * anchors + anchor)
        for task in range(tasks)
        for anchor in range(anchors)
    ]


def test_selects_128_unique_windows_and_covers_all_100_tasks_first():
    selected = select_qualitative_windows(_fixed_windows())

    assert len(selected) == len(set(selected)) == 128
    assert len({window[1] for window in selected}) == 100
    assert len({window[1] for window in selected[:100]}) == 100
    assert selected[:100] == [windows[0] for windows in (
        _fixed_windows()[task * 10 : task * 10 + 10] for task in range(100)
    )]
    assert selected[100:] == [
        _fixed_windows()[task * 10 + 1] for task in range(28)
    ]


def test_window_selection_rejects_duplicate_global_index_and_missing_task():
    windows = _fixed_windows(tasks=2, anchors=2)
    windows[-1] = (99, windows[-1][1], windows[0][2])
    with pytest.raises(ValueError, match="duplicate"):
        select_qualitative_windows(windows, num_windows=3, expected_tasks=2)
    with pytest.raises(ValueError, match="expected fixed windows from 3 tasks"):
        select_qualitative_windows(_fixed_windows(tasks=2), num_windows=3, expected_tasks=3)


def test_128_rank_partition_is_exactly_one_window_per_rank():
    selected = select_qualitative_windows(_fixed_windows())
    assigned = [
        qualitative_rank_windows(selected, rank=rank, world_size=128)
        for rank in range(128)
    ]
    assert assigned == [[window] for window in selected]
    with pytest.raises(ValueError, match="exactly one window per rank"):
        qualitative_rank_windows(selected, rank=0, world_size=64)


def test_seed_video_selection_and_filename_are_stable():
    window = (7, 1012, 3456)
    assert qualitative_window_seed(window) == qualitative_window_seed(window)
    assert qualitative_window_seed(window) != qualitative_window_seed((7, 1012, 3457))
    assert qualitative_video_window_positions(128, 8) == (0, 18, 36, 54, 73, 91, 109, 127)
    assert qualitative_video_filename(
        step=8501, position=18, rank=18, window=window, seed=123
    ) == "step-008501_window-018_rank-018_episode-001012_index-000003456_seed-0000000123.mp4"


class _Dataset:
    def __getitem__(self, _index):
        video = torch.zeros(3, 5, 12, 16)
        video[0] = 0.2
        return {
            "video": video,
            "prompt": "turning on radio",
            "action": torch.ones(4, 23) * 0.25,
            "proprio": torch.zeros(4, 23),
            "context": torch.zeros(3, 4),
            "context_mask": torch.ones(3, dtype=torch.bool),
        }


def _batch_sample(sample):
    return {
        "video": sample["video"].unsqueeze(0),
        "prompt": [sample["prompt"]],
        "action": sample["action"].unsqueeze(0),
        "proprio": sample["proprio"].unsqueeze(0),
        "context": sample["context"].unsqueeze(0),
        "context_mask": sample["context_mask"].unsqueeze(0),
        "action_horizon": sample["action"].shape[0],
    }


def _tensor_to_frames(video):
    video = ((video.float().cpu().clamp(-1, 1) + 1) * 0.5)[0]
    return [
        Image.fromarray(
            (video[:, index].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        )
        for index in range(video.shape[1])
    ]


class _Model(torch.nn.Module):
    device = torch.device("cpu")
    torch_dtype = torch.float32

    def __init__(self):
        super().__init__()
        self.dit = torch.nn.Linear(1, 1)
        self.frozen = torch.nn.Linear(1, 1)
        self.calls = []

    def infer_joint(self, **kwargs):
        self.calls.append(kwargs)
        image = kwargs["input_image"]
        video = image.unsqueeze(2).repeat(1, 1, kwargs["num_video_frames"], 1, 1)
        return {
            "video": _tensor_to_frames(video),
            "action": torch.ones(kwargs["action_horizon"], 23) * 0.25,
        }

    def _encode_video_latents(self, video, tiled=False):
        assert tiled is False
        return video

    def _decode_latents(self, video, tiled=False):
        assert tiled is False
        return _tensor_to_frames(video)


class _Accelerator:
    process_index = 0
    num_processes = 1
    device = torch.device("cpu")
    is_main_process = True

    def __init__(self):
        self.barriers = 0

    @staticmethod
    def autocast():
        return nullcontext()

    @staticmethod
    def reduce(tensor, reduction):
        assert reduction == "sum"
        return tensor

    @staticmethod
    def gather(tensor):
        return tensor

    def wait_for_everyone(self):
        self.barriers += 1

    @staticmethod
    def unwrap_model(model):
        return model


def test_end_to_end_runner_writes_atomic_report_and_disables_debug_rollout(tmp_path, monkeypatch):
    def fake_atomic_video(_frames, path, *, fps):
        assert fps == 8
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"completed-mp4")

    monkeypatch.setattr(qualitative, "_atomic_save_mp4", fake_atomic_video)
    model = _Model()
    accelerator = _Accelerator()
    config = QualitativeValidationConfig(
        num_windows=1, expected_tasks=1, videos_to_save=1, num_inference_steps=10
    )

    report = run_qualitative_validation(
        model=model,
        dataset=_Dataset(),
        accelerator=accelerator,
        fixed_windows=[(0, 1000, 0)],
        batch_sample=_batch_sample,
        action_denormalizer=lambda action: action,
        output_dir=tmp_path,
        step=8501,
        config=config,
        initialization_fingerprint_sha256="a" * 64,
    )

    persisted = json.loads((tmp_path / "step-008501" / "report.json").read_text())
    assert persisted == report
    assert report["schema"] == "b1k_qualitative_validation.v1"
    assert report["status"] == "passed"
    assert report["num_windows"] == report["num_tasks"] == report["world_size"] == 1
    assert len(report["videos"]) == 1
    assert report["videos"][0]["bytes"] == len(b"completed-mp4")
    assert report["metrics"]["action_l1_denormalized"] == {
        "sum": 0.0,
        "count": 1,
        "mean": 0.0,
    }
    assert all(value["count"] == 1 for value in report["metrics"].values())
    assert all(value["rank_count"] == 1 for value in report["timings"].values())
    assert accelerator.barriers == 0
    assert len(model.calls) == 1
    call = model.calls[0]
    assert call["seed"] == qualitative_window_seed((0, 1000, 0))
    assert call["action"] is None
    assert call["test_action_with_infer_action"] is False
    assert call["num_inference_steps"] == 10

    reused = run_qualitative_validation(
        model=model,
        dataset=_Dataset(),
        accelerator=accelerator,
        fixed_windows=[(0, 1000, 0)],
        batch_sample=_batch_sample,
        action_denormalizer=lambda action: action,
        output_dir=tmp_path,
        step=8501,
        config=config,
        initialization_fingerprint_sha256="a" * 64,
    )
    assert reused == report
    assert len(model.calls) == 1


def test_local_failure_is_reported_coherently_without_publishing_step(tmp_path):
    class BrokenModel(_Model):
        def infer_joint(self, **kwargs):
            raise RuntimeError("deliberate infer failure")

    with pytest.raises(RuntimeError, match="failed coherently.*deliberate infer failure"):
        run_qualitative_validation(
            model=BrokenModel(),
            dataset=_Dataset(),
            accelerator=_Accelerator(),
            fixed_windows=[(0, 1000, 0)],
            batch_sample=_batch_sample,
            action_denormalizer=lambda action: action,
            output_dir=tmp_path,
            step=1,
            config=QualitativeValidationConfig(
                num_windows=1, expected_tasks=1, videos_to_save=0
            ),
        )

    assert not (tmp_path / "step-000001").exists()
    assert len(list(tmp_path.glob(".step-000001.attempt-*"))) == 1


def test_report_publish_failure_leaves_only_hidden_attempt(tmp_path, monkeypatch):
    monkeypatch.setattr(
        qualitative,
        "_atomic_save_mp4",
        lambda _frames, path, fps: (
            Path(path).parent.mkdir(parents=True, exist_ok=True),
            Path(path).write_bytes(f"fps={fps}".encode()),
        ),
    )
    monkeypatch.setattr(
        qualitative,
        "_atomic_write_json",
        lambda _path, _payload: (_ for _ in ()).throw(OSError("disk unavailable")),
    )
    with pytest.raises(RuntimeError, match="failed coherently.*disk unavailable"):
        run_qualitative_validation(
            model=_Model(),
            dataset=_Dataset(),
            accelerator=_Accelerator(),
            fixed_windows=[(0, 1000, 0)],
            batch_sample=_batch_sample,
            action_denormalizer=lambda action: action,
            output_dir=tmp_path,
            step=2,
            config=QualitativeValidationConfig(
                num_windows=1, expected_tasks=1, videos_to_save=1
            ),
        )

    assert not (tmp_path / "step-000002").exists()
    attempts = list(tmp_path.glob(".step-000002.attempt-*"))
    assert len(attempts) == 1
    assert not (attempts[0] / "report.json").exists()


def test_atomic_mp4_replaces_temporary_file(tmp_path, monkeypatch):
    observed = {}

    def fake_save(_frames, path, fps):
        observed["path"] = Path(path)
        observed["fps"] = fps
        Path(path).write_bytes(b"valid-video")

    monkeypatch.setattr(qualitative, "save_mp4", fake_save)
    destination = tmp_path / "videos" / "result.mp4"
    qualitative._atomic_save_mp4([], destination, fps=7)

    assert destination.read_bytes() == b"valid-video"
    assert observed["fps"] == 7
    assert observed["path"] != destination
    assert observed["path"].name.endswith(".tmp.mp4")
    assert not observed["path"].exists()


class _MixedTrainer:
    def __init__(self, tmp_path):
        self.model = _Model()
        self.model.dit.train()
        self.model.dit.requires_grad_(True)
        self.model.frozen.eval()
        self.model.frozen.requires_grad_(False)
        self.accelerator = _Accelerator()
        self.val_dataset = _Dataset()
        self.output_dir = str(tmp_path)
        self.global_step = 3
        self.initialization_fingerprint_sha256 = "b" * 64

    @staticmethod
    def _to_batched_eval_sample(sample):
        return _batch_sample(sample)

    @staticmethod
    def _capture_model_execution_state(*roots):
        modules = []
        parameters = []
        seen_modules = set()
        seen_parameters = set()
        for root in roots:
            for module in root.modules():
                if id(module) not in seen_modules:
                    seen_modules.add(id(module))
                    modules.append((module, module.training))
                for parameter in module.parameters(recurse=False):
                    if id(parameter) not in seen_parameters:
                        seen_parameters.add(id(parameter))
                        parameters.append((parameter, parameter.requires_grad))
        return modules, parameters

    @staticmethod
    def _restore_model_execution_state(state):
        modules, parameters = state
        for module, training in modules:
            module.training = training
        for parameter, requires_grad in parameters:
            parameter.requires_grad_(requires_grad)


def test_trainer_api_restores_mixed_execution_state(tmp_path, monkeypatch):
    monkeypatch.setattr(
        qualitative,
        "_b1k_action_denormalizer",
        lambda _dataset: lambda action: action,
    )
    monkeypatch.setattr(
        qualitative,
        "_atomic_save_mp4",
        lambda _frames, path, fps: (
            Path(path).parent.mkdir(parents=True, exist_ok=True),
            Path(path).write_bytes(f"fps={fps}".encode()),
        ),
    )
    trainer = _MixedTrainer(tmp_path)
    before = (
        trainer.model.training,
        trainer.model.dit.training,
        trainer.model.frozen.training,
        trainer.model.dit.weight.requires_grad,
        trainer.model.frozen.weight.requires_grad,
    )

    report = run_trainer_qualitative_validation(
        trainer,
        fixed_windows=[(0, 1000, 0)],
        config=QualitativeValidationConfig(
            num_windows=1, expected_tasks=1, videos_to_save=1
        ),
    )

    after = (
        trainer.model.training,
        trainer.model.dit.training,
        trainer.model.frozen.training,
        trainer.model.dit.weight.requires_grad,
        trainer.model.frozen.weight.requires_grad,
    )
    assert after == before
    assert report["step"] == 3
    assert report["initialization_fingerprint_sha256"] == "b" * 64


def test_config_rejects_impossible_task_coverage():
    with pytest.raises(ValueError, match="at least expected_tasks"):
        QualitativeValidationConfig(num_windows=99, expected_tasks=100)


def test_timing_aggregation_computes_true_max_across_ranks():
    class TwoRankAccelerator:
        device = torch.device("cpu")
        num_processes = 2

        @staticmethod
        def gather(_tensor):
            count = len(qualitative._TIMING_NAMES)
            return torch.tensor(
                [
                    list(range(1, count + 1)),
                    [10 * value for value in range(1, count + 1)],
                ],
                dtype=torch.float64,
            )

    local = {name: float(index) for index, name in enumerate(qualitative._TIMING_NAMES)}
    report = qualitative._reduce_timings(local, TwoRankAccelerator())

    assert report["data_load_s"] == {
        "sum_seconds": 11.0,
        "mean_seconds": 5.5,
        "max_seconds": 10.0,
        "rank_count": 2,
    }
    assert report["total_s"]["sum_seconds"] == 88.0
    assert report["total_s"]["max_seconds"] == 80.0
    assert report["outer_s"]["sum_seconds"] == 99.0
    assert report["outer_s"]["max_seconds"] == 90.0


def test_remote_rank_failure_is_raised_on_every_rank(monkeypatch):
    class Accelerator:
        process_index = 0
        num_processes = 2

    remote_error = {
        "stage": "local_evaluation",
        "rank": 1,
        "type": "RuntimeError",
        "message": "remote ffmpeg failure",
        "traceback": "trace",
    }
    monkeypatch.setattr(
        "accelerate.utils.gather_object",
        lambda local: [local[0], {"rank": 1, "error": remote_error}],
    )

    with pytest.raises(RuntimeError, match="rank=1 RuntimeError: remote ffmpeg failure"):
        qualitative._raise_if_any_rank_failed(
            Accelerator(), None, stage="local_evaluation"
        )
