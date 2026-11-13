# Copyright (c) 2026 Xiaomi Corporation

from pathlib import Path
import json

import pytest
import torch

from fastwam.datasets.b1k import video_dataset
from fastwam.datasets.b1k.dataset import _TextContextLRU
from fastwam.datasets.b1k.video_dataset import (
    B1KDirectVideoDataset,
    _CameraLocation,
    _VideoEpisode,
    _select_direct_episodes,
)


class _NumericReader:
    def __init__(self) -> None:
        self.calls = []

    def read_rows(self, path, columns, rows, *, cache_numeric):
        self.calls.append((path, columns, list(rows), cache_numeric))
        return {
            "action": [[float(row)] * 23 for row in rows],
            "observation.state": [[float(row)] * 61 for row in rows],
        }

    def num_rows(self, path):
        return 80


class _VideoReader:
    def __init__(self) -> None:
        self.calls = []

    def read_frames(self, requests):
        self.calls.append(requests)
        return {
            path: {
                position: torch.full((3, 8, 8), position % 255, dtype=torch.uint8)
                for position in positions
            }
            for path, positions in requests.items()
        }


class _DepthVideoReader:
    def __init__(self) -> None:
        self.calls = []

    def read_frames(self, requests):
        self.calls.append(requests)
        return {
            path: {
                position: torch.full((1, 8, 8), float(position))
                for position in positions
            }
            for path, positions in requests.items()
        }


class _EquivalentNumericReader(_NumericReader):
    def read_rows(self, path, columns, rows, *, cache_numeric):
        self.calls.append((path, columns, list(rows), cache_numeric))
        logical_rows = [
            row - 100 if path == Path("/data/shared.parquet") else row for row in rows
        ]
        return {
            "action": [[float(row)] * 23 for row in logical_rows],
            "observation.state": [[float(row)] * 61 for row in logical_rows],
        }


def _bare_dataset() -> B1KDirectVideoDataset:
    dataset = B1KDirectVideoDataset.__new__(B1KDirectVideoDataset)
    data_path = Path("/data/shared.parquet")
    cameras = tuple(
        _CameraLocation(Path(f"/videos/camera-{index}.mp4"), 1000 * index)
        for index in range(3)
    )
    dataset._episodes = [
        _VideoEpisode(0, 80, "turning_on_radio", 100, data_path, 0, cameras)
    ]
    dataset._frame_prefix = [0, 80]
    dataset.window_start_stride = 1
    dataset._reader = _NumericReader()
    dataset._video_reader = _VideoReader()
    dataset.num_image_steps = 9
    dataset.normalize_transform = lambda value: (value - 0.5) / 0.5
    dataset._text_context_cache = _TextContextLRU(2)
    return dataset


def test_direct_video_full_training_split_selects_every_episode_per_task():
    cameras = (_CameraLocation(Path("/video.mp4"), 0),) * 3
    episodes = [
        _VideoEpisode(index, 10, f"task-{index // 4}", 0, Path("/data.parquet"), 0, cameras)
        for index in range(8)
    ]

    default_train = _select_direct_episodes(
        episodes, split="train", train_all_episodes=False,
        expected_tasks=2, task_indices=None,
    )
    full_train = _select_direct_episodes(
        episodes, split="train", train_all_episodes=True,
        expected_tasks=2, task_indices=None,
    )

    assert [episode.index for episode in default_train] == [0, 1, 4, 5]
    assert [episode.index for episode in full_train] == list(range(8))


def test_direct_video_full_training_split_rejects_non_train_split():
    with pytest.raises(ValueError, match="only valid for the train split"):
        _select_direct_episodes(
            [], split="validation", train_all_episodes=True,
            expected_tasks=1, task_indices=None,
        )


def test_direct_video_selects_canonical_task_indices_before_split():
    cameras = (_CameraLocation(Path("/video.mp4"), 0),) * 3
    episodes = [
        _VideoEpisode(index, 10, f"task-{index // 4}", 0, Path("/data.parquet"), 0, cameras)
        for index in range(12)
    ]

    selected = _select_direct_episodes(
        episodes,
        split="train",
        train_all_episodes=False,
        expected_tasks=3,
        task_indices=[0, 2],
    )
    full = _select_direct_episodes(
        episodes,
        split="train",
        train_all_episodes=True,
        expected_tasks=3,
        task_indices=[0, 2],
    )

    assert [episode.index for episode in selected] == [0, 1, 8, 9]
    assert [episode.index for episode in full] == [0, 1, 2, 3, 8, 9, 10, 11]


@pytest.mark.parametrize("task_indices", [[], [3], [True], ["0"], "0"])
def test_direct_video_rejects_invalid_task_indices(task_indices):
    with pytest.raises((TypeError, ValueError)):
        _select_direct_episodes(
            [], split="train", train_all_episodes=False,
            expected_tasks=3, task_indices=task_indices,
        )


def test_direct_video_batch_groups_numeric_and_video_requests():
    dataset = _bare_dataset()
    first, second = dataset._raw_samples([0, 4])

    assert len(dataset._reader.calls) == 1
    assert dataset._reader.calls[0][0] == Path("/data/shared.parquet")
    assert dataset._reader.calls[0][3] is False
    assert len(dataset._video_reader.calls) == 1
    requests = dataset._video_reader.calls[0]
    assert set(requests) == {camera.path for camera in dataset._episodes[0].cameras}
    assert requests[Path("/videos/camera-0.mp4")] == list(range(0, 69, 4))
    assert first["action"]["default"].shape == (32, 23)
    assert first["state"]["default"].shape == (33, 61)
    assert first["images"]["rgb.head"].shape == (9, 3, 8, 8)
    assert second["frame_index"] == 4


def test_direct_video_tail_padding_and_offsets_are_episode_local():
    dataset = _bare_dataset()
    sample = dataset._raw_sample(79)

    assert sample["action_is_pad"].tolist() == [False] + [True] * 31
    assert sample["state_is_pad"].tolist() == [False] + [True] * 32
    assert sample["image_is_pad"].tolist() == [False] + [True] * 8
    assert sample["action"]["default"][:, 0].eq(179).all()
    assert sample["images"]["rgb.left_wrist"][:, 0, 0, 0].eq((1079 % 255)).all()


def test_direct_video_window_start_stride_is_episode_local():
    dataset = _bare_dataset()
    second = dataset._episodes[0]
    dataset._episodes = [
        _VideoEpisode(
            second.index,
            8,
            second.task,
            second.dataset_from_index,
            second.data_path,
            second.data_file_from_index,
            second.cameras,
        ),
        _VideoEpisode(
            1,
            5,
            second.task,
            second.dataset_from_index,
            second.data_path,
            second.data_file_from_index,
            second.cameras,
        ),
    ]
    dataset.window_start_stride = 6
    dataset._frame_prefix = [0, 2, 3]

    assert len(dataset) == 3
    assert dataset._locate(0)[1] == 0
    assert dataset._locate(1)[1] == 6
    episode, local_frame = dataset._locate(2)
    assert episode.index == 1
    assert local_frame == 0
    assert dataset._locate(-1) == (episode, 0)
    with pytest.raises(IndexError):
        dataset._locate(3)


def test_direct_video_numeric_sidecar_changes_only_numeric_path():
    dataset = _bare_dataset()
    original = dataset._episodes[0]
    sidecar_path = Path("/sidecar/data/chunk-000/episode_000000.parquet")
    dataset._episodes = [
        _VideoEpisode(
            original.index,
            original.length,
            original.task,
            original.dataset_from_index,
            original.data_path,
            original.data_file_from_index,
            original.cameras,
            sidecar_path,
            original.dataset_from_index,
        )
    ]

    sample = dataset._raw_sample(4)

    assert dataset._reader.calls[0][0] == sidecar_path
    assert dataset._reader.calls[0][2][0] == 4
    assert sample["action"]["default"][0, 0].item() == 4
    requests = dataset._video_reader.calls[0]
    assert set(requests) == {camera.path for camera in original.cameras}


def test_direct_video_raw_and_sidecar_numeric_values_are_equivalent():
    raw_dataset = _bare_dataset()
    sidecar_dataset = _bare_dataset()
    raw_dataset._reader = _EquivalentNumericReader()
    sidecar_dataset._reader = _EquivalentNumericReader()
    original = sidecar_dataset._episodes[0]
    sidecar_dataset._episodes = [
        _VideoEpisode(
            original.index,
            original.length,
            original.task,
            original.dataset_from_index,
            original.data_path,
            original.data_file_from_index,
            original.cameras,
            Path("/sidecar/episode_000000.parquet"),
            original.dataset_from_index,
        )
    ]

    raw = raw_dataset._raw_sample(4)
    sidecar = sidecar_dataset._raw_sample(4)

    assert torch.equal(raw["action"]["default"], sidecar["action"]["default"])
    assert torch.equal(raw["state"]["default"], sidecar["state"]["default"])
    assert raw["episode_index"] == sidecar["episode_index"]
    assert raw["frame_index"] == sidecar["frame_index"]


def _write_sidecar_metadata(root: Path, *, fps: int = 30, action_dim: int = 23) -> None:
    meta = root / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(
        json.dumps(
            {
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                "chunks_size": 200,
                "fps": fps,
                "features": {
                    "action": {"shape": [action_dim]},
                    "observation.state": {"shape": [61]},
                },
            }
        ),
        encoding="utf-8",
    )
    (meta / "episodes.jsonl").write_text(
        json.dumps(
            {
                "episode_index": 0,
                "length": 80,
                "tasks": ["turning_on_radio"],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_direct_video_attaches_numeric_sidecar_without_changing_video_identity(tmp_path):
    dataset = _bare_dataset()
    dataset.numeric_dataset_root = tmp_path
    dataset._info = {"fps": 30}
    _write_sidecar_metadata(tmp_path)
    original = dataset._episodes[0]

    attached = dataset._attach_numeric_sidecar(dataset._episodes)

    assert [episode.index for episode in attached] == [
        episode.index for episode in dataset._episodes
    ]
    assert attached[0].length == original.length
    assert attached[0].task == original.task
    assert attached[0].data_path == original.data_path
    assert attached[0].cameras == original.cameras
    assert attached[0].numeric_path == (
        tmp_path / "data/chunk-000/episode_000000.parquet"
    )
    assert attached[0].numeric_file_from_index == original.dataset_from_index


def test_direct_video_rejects_incompatible_numeric_sidecar(tmp_path):
    dataset = _bare_dataset()
    dataset.numeric_dataset_root = tmp_path
    dataset._info = {"fps": 30}
    _write_sidecar_metadata(tmp_path, action_dim=22)

    with pytest.raises(ValueError, match="action.*shape"):
        dataset._attach_numeric_sidecar(dataset._episodes)


def test_direct_video_long_context_offsets_produce_64_actions_and_17_images():
    dataset = _bare_dataset()
    dataset._action_offsets = tuple(range(0, 128, 2))
    dataset._state_offsets = tuple(range(0, 130, 2))
    dataset._image_offsets = tuple(range(0, 130, 8))
    dataset.num_image_steps = 17

    raw = dataset._raw_sample(0)

    assert raw["action"]["default"].shape == (64, 23)
    assert raw["state"]["default"].shape == (65, 61)
    assert raw["images"]["rgb.head"].shape == (17, 3, 8, 8)


def test_numeric_sidecar_preserves_depth_video_identity(tmp_path):
    dataset = _bare_dataset()
    depth_cameras = tuple(
        _CameraLocation(Path(f"/depth/camera-{index}.mp4"), 2000 * index)
        for index in range(3)
    )
    original = dataset._episodes[0]
    dataset._episodes = [
        _VideoEpisode(
            original.index, original.length, original.task,
            original.dataset_from_index, original.data_path,
            original.data_file_from_index, original.cameras,
            depth_cameras=depth_cameras,
        )
    ]
    dataset.numeric_dataset_root = tmp_path
    dataset._info = {"fps": 30}
    _write_sidecar_metadata(tmp_path)

    attached = dataset._attach_numeric_sidecar(dataset._episodes)

    assert attached[0].depth_cameras == depth_cameras


@pytest.mark.parametrize("include_depth", [False, True])
def test_pointmap_loads_depth_internally_without_forcing_depth_into_sample(
    monkeypatch, include_depth
):
    dataset = _bare_dataset()
    original = dataset._episodes[0]
    depth_cameras = tuple(
        _CameraLocation(Path(f"/depth/camera-{index}.mp4"), 2000 * index)
        for index in range(3)
    )
    dataset._episodes = [
        _VideoEpisode(
            original.index,
            original.length,
            original.task,
            original.dataset_from_index,
            original.data_path,
            original.data_file_from_index,
            original.cameras,
            depth_cameras=depth_cameras,
        )
    ]
    dataset.include_depth = include_depth
    dataset.return_pointmap = True
    dataset._depth_video_reader = _DepthVideoReader()
    dataset.num_frames = 33
    dataset.override_instruction = None
    dataset.processor = type(
        "Processor",
        (),
        {
            "preprocess_action_state": staticmethod(
                lambda raw: {
                    "action": torch.zeros(32, 23),
                    "proprio": torch.zeros(33, 23),
                    "action_is_pad": raw["action_is_pad"],
                    "proprio_is_pad": raw["state_is_pad"],
                }
            )
        },
    )()
    dataset._make_mosaic = lambda images: torch.zeros(3, 9, 384, 256)
    dataset._get_cached_text_context = lambda prompt: (
        torch.zeros(1, 4),
        torch.ones(1, dtype=torch.bool),
    )
    pointmap_inputs = []

    def fake_pointmap(depths):
        pointmap_inputs.append(depths)
        return torch.zeros(3, 9, 384, 256)

    monkeypatch.setattr(video_dataset, "make_b1k_pointmap_mosaic", fake_pointmap)

    sample = dataset[0]

    assert len(dataset._depth_video_reader.calls) == 1
    requests = dataset._depth_video_reader.calls[0]
    assert set(requests) == {camera.path for camera in depth_cameras}
    assert len(pointmap_inputs) == 1
    assert set(pointmap_inputs[0]) == {"head", "left_wrist", "right_wrist"}
    assert sample["pointmap"].shape == (3, 9, 384, 256)
    assert ("depths" in sample) is include_depth


def test_direct_video_mosaic_uses_256_head_and_128_wrists_without_padding(monkeypatch):
    dataset = _bare_dataset()
    video = {
        "rgb.head": torch.full((9, 3, 720, 720), 64, dtype=torch.uint8),
        "rgb.left_wrist": torch.full((9, 3, 480, 480), 128, dtype=torch.uint8),
        "rgb.right_wrist": torch.full((9, 3, 480, 480), 191, dtype=torch.uint8),
    }

    resize_calls = []
    original_resize = video_dataset.transforms_F.resize

    def counted_resize(value, size, **kwargs):
        resize_calls.append((tuple(value.shape[-2:]), tuple(size)))
        return original_resize(value, size, **kwargs)

    monkeypatch.setattr(
        "fastwam.datasets.b1k.video_dataset.transforms_F.resize", counted_resize
    )
    mosaic = dataset._make_mosaic(video)

    assert mosaic.shape == (3, 9, 384, 256)
    assert resize_calls == [
        ((720, 720), (256, 256)),
        ((480, 480), (128, 128)),
        ((480, 480), (128, 128)),
    ]
    assert torch.allclose(
        mosaic[:, :, :256], torch.full((3, 9, 256, 256), 2 * 64 / 255 - 1),
        rtol=0, atol=1e-6,
    )
    assert torch.allclose(
        mosaic[:, :, 256:, :128], torch.full((3, 9, 128, 128), 2 * 128 / 255 - 1),
        rtol=0, atol=1e-6,
    )
    assert torch.allclose(
        mosaic[:, :, 256:, 128:], torch.full((3, 9, 128, 128), 2 * 191 / 255 - 1),
        rtol=0, atol=1e-6,
    )


def test_direct_video_checkpoint_contract_has_no_padding():
    dataset = _bare_dataset()
    dataset.global_sample_stride = 2
    dataset.window_start_stride = 6
    dataset._frame_prefix = [0, 14]
    dataset.image_sample_indices = list(range(0, 33, 4))
    dataset.context_len = 128

    video = dataset.checkpoint_data_contract["video"]
    sampling = dataset.checkpoint_data_contract["sampling"]

    assert video["canvas"] == {"height": 384, "width": 256}
    assert "padding" not in video
    assert video["source"] == "lerobot_v3_hevc"
    assert sampling == {
        "source_fps": 30,
        "window_start_stride": 6,
        "anchor_frequency_hz": 5.0,
        "sample_count": 14,
        "tail_policy": "replicate_last_frame_with_padding_mask",
    }


def test_long_context_depth_checkpoint_contract_is_complete():
    dataset = _bare_dataset()
    dataset.global_sample_stride = 2
    dataset.window_start_stride = 6
    dataset._frame_prefix = [0, 14]
    dataset.num_frames = 65
    dataset.image_sample_indices = list(range(0, 65, 4))
    dataset._image_offsets = tuple(range(0, 130, 8))
    dataset.context_len = 128
    dataset.include_depth = True
    dataset.return_pointmap = True

    contract = dataset.checkpoint_data_contract

    assert contract["action"]["horizon"] == 64
    assert contract["video"]["dataset_image_indices"] == list(range(0, 65, 4))
    assert contract["video"]["raw_frame_indices"] == list(range(0, 130, 8))
    assert contract["depth"]["quantization"]["bits"] == 12
    assert contract["pointmap"]["frame"] == "per_camera"
    assert contract["pointmap"]["valid_depth_m"] == [0.01, 10.1]
    assert contract["pointmap"]["normalization_bounds_m"] == {
        "x": [-1.5, 1.5],
        "y": [-1.5, 1.5],
        "z": [0.0, 3.0],
    }


def test_depth_sources_match_official_b1k_feature_names():
    from fastwam.datasets.b1k.video_dataset import _DEPTH_SOURCES

    assert [key for _, key in _DEPTH_SOURCES] == [
        "observation.depth_linear.zed_link_camera_0",
        "observation.depth_linear.left_realsense_link_camera_0",
        "observation.depth_linear.right_realsense_link_camera_0",
    ]
