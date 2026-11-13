# Copyright (c) 2026 Xiaomi Corporation

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("datasets")
pytest.importorskip("accelerate")

from fastwam.datasets.lerobot import base_lerobot_dataset as base_module
from fastwam.datasets.b1k.dataset import B1KRobotVideoDataset
from fastwam.datasets.lerobot.base_lerobot_dataset import BaseLerobotDataset
from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from fastwam.datasets.lerobot.robot_video_dataset import RobotVideoDataset


SHAPE_META = {
    "images": [{"key": "head", "raw_shape": [3, 224, 224], "shape": [3, 224, 224]}],
    "state": [{"key": "default", "raw_shape": 23, "shape": 23}],
    "action": [{"key": "default", "raw_shape": 23, "shape": 23}],
}


class _FakeMetadata:
    def __init__(self, repo_id, root):
        self.repo_id = repo_id
        self.fps = 30
        self.total_episodes = 200


class _FakeMultiDataset:
    captured_episodes = None
    captured_delta_timestamps = None

    def __init__(self, dataset_dirs, episodes, delta_timestamps):
        type(self).captured_episodes = episodes
        type(self).captured_delta_timestamps = delta_timestamps
        selected_count = len(episodes[dataset_dirs[0]])
        self._datasets = [
            SimpleNamespace(
                episode_data_index={
                    "from": torch.arange(selected_count),
                    "to": torch.arange(1, selected_count + 1),
                }
            )
        ]
        self.num_frames = selected_count


def test_base_dataset_uses_explicit_episodes_and_sparse_image_timestamps(monkeypatch):
    monkeypatch.setattr(base_module, "LeRobotDatasetMetadata", _FakeMetadata)
    monkeypatch.setattr(base_module, "MultiLeRobotDataset", _FakeMultiDataset)

    dataset = BaseLerobotDataset(
        dataset_dirs=["/dataset"],
        shape_meta=SHAPE_META,
        obs_size=33,
        action_size=32,
        global_sample_stride=2,
        selected_episodes=[0, 7, 198],
        image_sample_indices=[0, 4, 8, 12, 16, 20, 24, 28, 32],
    )

    assert dataset.image_sample_indices == [0, 4, 8, 12, 16, 20, 24, 28, 32]
    assert _FakeMultiDataset.captured_episodes == {"/dataset": [0, 7, 198]}
    assert _FakeMultiDataset.captured_delta_timestamps["observation.images.head"] == pytest.approx(
        [index * 2 / 30 for index in range(0, 33, 4)]
    )
    assert len(_FakeMultiDataset.captured_delta_timestamps["observation.state"]) == 33
    assert len(_FakeMultiDataset.captured_delta_timestamps["action"]) == 32


def test_base_dataset_rejects_ambiguous_episode_list_for_multiple_datasets(monkeypatch):
    monkeypatch.setattr(base_module, "LeRobotDatasetMetadata", _FakeMetadata)

    with pytest.raises(ValueError, match="single dataset"):
        BaseLerobotDataset(
            dataset_dirs=["/a", "/b"],
            shape_meta=SHAPE_META,
            obs_size=33,
            action_size=32,
            selected_episodes=[0],
        )


def test_base_dataset_fail_fast_preserves_original_error():
    dataset = BaseLerobotDataset.__new__(BaseLerobotDataset)
    dataset.fail_fast = True
    dataset.multi_dataset = SimpleNamespace(
        num_frames=1,
        __getitem__=lambda _idx: None,
    )

    class _BrokenDataset:
        num_frames = 1

        def __getitem__(self, _idx):
            raise OSError("broken parquet")

    dataset.multi_dataset = _BrokenDataset()
    with pytest.raises(OSError, match="broken parquet"):
        dataset[0]


class _Identity:
    def __call__(self, value):
        return value


def test_b1k_mosaic_and_cached_t5_mask_are_preserved():
    frames = torch.zeros(3, 9, 3, 16, 16)
    frames[0] = 0.25
    frames[1] = 0.50
    frames[2] = 0.75
    sample = {
        "pixel_values": frames,
        "image_is_pad": torch.zeros(9, dtype=torch.bool),
        "action": torch.zeros(32, 23),
        "proprio": torch.zeros(33, 23),
        "instruction": "turning on radio",
        "action_is_pad": torch.zeros(32, dtype=torch.bool),
        "proprio_is_pad": torch.zeros(33, dtype=torch.bool),
    }

    dataset = RobotVideoDataset.__new__(RobotVideoDataset)
    dataset.lerobot_dataset = [sample]
    dataset.max_padding_retry = 0
    dataset.skip_padding_as_possible = False
    dataset.video_sample_indices = None
    dataset.num_image_steps = 9
    dataset.concat_multi_camera = "b1k"
    dataset.resize_transform = _Identity()
    dataset.crop_transform = _Identity()
    dataset.normalize_transform = _Identity()
    dataset.override_instruction = None
    dataset.context_len = 4
    context = torch.arange(12, dtype=torch.float32).view(4, 3)
    mask = torch.tensor([True, True, False, False])
    dataset._get_cached_text_context = lambda _prompt: (context, mask)

    result = dataset._get(0)

    assert result["video"].shape == (3, 9, 352, 224)
    assert torch.allclose(result["video"][:, :, :224], torch.full((3, 9, 224, 224), 0.25))
    assert torch.allclose(result["video"][:, :, 224:336, :112], torch.full((3, 9, 112, 112), 0.50))
    assert torch.allclose(result["video"][:, :, 224:336, 112:], torch.full((3, 9, 112, 112), 0.75))
    assert torch.allclose(result["video"][:, :, 336:], torch.full((3, 9, 16, 224), 0.5))
    assert torch.equal(result["context_mask"], mask)
    assert torch.count_nonzero(result["context"][~mask]) == 0
    assert torch.count_nonzero(context[~mask]) > 0, "The cached tensor must not be mutated in-place."


class _Merger:
    def set_shape_meta(self, shape_meta):
        self.shape_meta = shape_meta


def test_processor_supports_image_horizon_independent_of_state_horizon():
    processor = FastWAMProcessor(
        shape_meta=SHAPE_META,
        num_obs_steps=33,
        num_image_steps=9,
        num_output_cameras=1,
        action_output_dim=23,
        proprio_output_dim=23,
        action_state_transforms=None,
        use_stepwise_action_norm=False,
        norm_default_mode="z-score",
        norm_exception_mode=None,
        action_state_merger=_Merger(),
        train_transforms=[],
        val_transforms=[],
    )

    assert processor.num_obs_steps == 33
    assert processor.num_image_steps == 9


def test_b1k_dataset_uses_b1k_prompt_formatter():
    dataset = B1KRobotVideoDataset.__new__(B1KRobotVideoDataset)
    assert dataset._format_prompt("turning_on_radio").endswith("turning on radio")
