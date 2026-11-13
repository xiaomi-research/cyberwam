# Copyright (c) 2026 Xiaomi Corporation

from __future__ import annotations

import hashlib
import io
import json
import pickle
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image


pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from fastwam.datasets.b1k.dataset import B1KRobotVideoDataset
from fastwam.datasets.b1k.text import format_b1k_prompt


CAMERA_COLUMNS = (
    "observation.images.rgb.head",
    "observation.images.rgb.left_wrist",
    "observation.images.rgb.right_wrist",
)
CAMERA_KEYS = ("rgb.head", "rgb.left_wrist", "rgb.right_wrist")


class _IdentityProcessor:
    def __init__(self) -> None:
        self.stats = None
        self.training = None

    def set_normalizer_from_stats(self, stats) -> None:
        self.stats = stats

    def train(self):
        self.training = True
        return self

    def eval(self):
        self.training = False
        return self

    def preprocess(self, sample):
        return {
            "pixel_values": torch.stack(
                [sample["images"][key] for key in CAMERA_KEYS], dim=0
            ),
            "action": sample["action"]["default"],
            "proprio": sample["state"]["default"][..., :23],
            "action_is_pad": sample["action_is_pad"],
            "proprio_is_pad": sample["state_is_pad"],
        }


def _jpeg_bytes(value: int) -> bytes:
    image = Image.fromarray(
        np.full((224, 224, 3), value % 256, dtype=np.uint8), mode="RGB"
    )
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


def _write_episode(path: Path, episode_index: int, length: int) -> None:
    actions = []
    states = []
    for frame in range(length):
        action = np.full(23, frame, dtype=np.float32)
        action[6] = 0
        action[14] = -1 if frame % 2 else 1
        action[22] = 1 if frame % 2 else -1
        actions.append(action.tolist())

        state = np.full(61, frame, dtype=np.float32)
        states.append(state.tolist())

    columns = {
        "action": pa.array(actions, type=pa.list_(pa.float32(), 23)),
        "observation.state": pa.array(states, type=pa.list_(pa.float32(), 61)),
        "episode_index": pa.array([episode_index] * length, type=pa.int64()),
        "frame_index": pa.array(range(length), type=pa.int64()),
    }
    image_type = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    for camera_index, column in enumerate(CAMERA_COLUMNS):
        columns[column] = pa.array(
            [
                {"bytes": _jpeg_bytes(frame + camera_index * 80), "path": None}
                for frame in range(length)
            ],
            type=image_type,
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(columns), path, row_group_size=7)


def _write_text_cache(cache_dir: Path, task: str) -> None:
    prompt = format_b1k_prompt(task)
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    cache_dir.mkdir(parents=True, exist_ok=True)
    context = torch.arange(128 * 4, dtype=torch.float32).reshape(128, 4)
    mask = torch.zeros(128, dtype=torch.bool)
    mask[:3] = True
    torch.save(
        {"context": context, "mask": mask},
        cache_dir / f"{digest}.t5_len128.wan22ti2v5b.pt",
    )


@pytest.fixture()
def synthetic_lerobot_v21(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "b1k"
    meta = root / "meta"
    meta.mkdir(parents=True)
    info = {
        "codebase_version": "v2.1",
        "fps": 30,
        "chunks_size": 3,
        "total_episodes": 3,
        "total_tasks": 1,
        "total_frames": 76,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "features": {
            "action": {"dtype": "float32", "shape": [23]},
            "observation.state": {"dtype": "float32", "shape": [61]},
            **{
                column: {"dtype": "image", "shape": [8, 8, 3]}
                for column in CAMERA_COLUMNS
            },
        },
    }
    (meta / "info.json").write_text(json.dumps(info), encoding="utf-8")
    episodes = [
        {"episode_index": 0, "length": 70, "task_index": 0, "tasks": ["test_task"]},
        {"episode_index": 1, "length": 3, "task_index": 0, "tasks": ["test_task"]},
        {"episode_index": 2, "length": 3, "task_index": 0, "tasks": ["test_task"]},
    ]
    (meta / "episodes.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in episodes), encoding="utf-8"
    )
    (meta / "tasks.jsonl").write_text(
        json.dumps({"task": "test_task", "task_index": 0}) + "\n",
        encoding="utf-8",
    )
    for episode in episodes:
        path = root / info["data_path"].format(
            episode_chunk=0, episode_index=episode["episode_index"]
        )
        _write_episode(path, episode["episode_index"], episode["length"])

    cache_dir = tmp_path / "text_cache"
    _write_text_cache(cache_dir, "test_task")
    return root, cache_dir


def _shape_meta():
    return {
        "images": [
            {"key": key, "raw_shape": [3, 8, 8], "shape": [3, 8, 8]}
            for key in CAMERA_KEYS
        ],
        "action": [{"key": "default", "raw_shape": 23, "shape": 23}],
        "state": [{"key": "default", "raw_shape": 61, "shape": 23}],
    }


def _dataset(root: Path, cache_dir: Path) -> B1KRobotVideoDataset:
    return B1KRobotVideoDataset(
        dataset_root=root,
        split="train",
        shape_meta=_shape_meta(),
        processor=_IdentityProcessor(),
        pretrained_norm_stats={"train_only": True},
        text_embedding_cache_dir=cache_dir,
        expected_tasks=1,
        episodes_per_task=3,
    )


def test_window_start_stride_changes_only_episode_local_anchors(synthetic_lerobot_v21):
    root, cache_dir = synthetic_lerobot_v21
    dataset = B1KRobotVideoDataset(
        dataset_root=root,
        split="train",
        shape_meta=_shape_meta(),
        processor=_IdentityProcessor(),
        pretrained_norm_stats={"train_only": True},
        text_embedding_cache_dir=cache_dir,
        expected_tasks=1,
        episodes_per_task=3,
        window_start_stride=6,
    )

    assert len(dataset) == 12
    assert dataset.episode_data_index["from"].tolist() == [0]
    assert dataset.episode_data_index["to"].tolist() == [12]
    assert dataset._locate(0)[1] == 0
    assert dataset._locate(1)[1] == 6
    assert dataset._locate(-1)[1] == 66
    sample = dataset[-1]
    assert sample["frame_index"] == 66
    assert sample["action"][:, 0].tolist()[:2] == [66.0, 68.0]
    assert sample["action"][:, 0].tolist()[2:] == [69.0] * 30


@pytest.mark.parametrize("stride", [0, -1])
def test_window_start_stride_must_be_positive(synthetic_lerobot_v21, stride):
    root, cache_dir = synthetic_lerobot_v21
    with pytest.raises(ValueError, match="window_start_stride must be positive"):
        B1KRobotVideoDataset(
            dataset_root=root,
            split="train",
            shape_meta=_shape_meta(),
            processor=_IdentityProcessor(),
            pretrained_norm_stats={"train_only": True},
            text_embedding_cache_dir=cache_dir,
            expected_tasks=1,
            episodes_per_task=3,
            window_start_stride=stride,
        )


def test_direct_dataset_does_not_use_robot_video_or_hf_datasets(
    synthetic_lerobot_v21, monkeypatch
):
    def fail_load_dataset(*args, **kwargs):
        raise AssertionError("B1K direct parquet loading must not call datasets.load_dataset")

    monkeypatch.setitem(
        sys.modules,
        "datasets",
        types.SimpleNamespace(load_dataset=fail_load_dataset),
    )
    root, cache_dir = synthetic_lerobot_v21
    dataset = _dataset(root, cache_dir)

    assert "RobotVideoDataset" not in {
        base.__name__ for base in B1KRobotVideoDataset.__mro__
    }
    assert len(dataset) == 70
    assert dataset.selected_episodes == [0]


def test_getitem_decodes_exactly_nine_frames_from_each_camera(
    synthetic_lerobot_v21, monkeypatch
):
    root, cache_dir = synthetic_lerobot_v21
    dataset = _dataset(root, cache_dir)
    original_decode = dataset._decode_jpeg
    decoded = []

    def counted_decode(payload):
        decoded.append(payload)
        return original_decode(payload)

    monkeypatch.setattr(dataset, "_decode_jpeg", counted_decode)
    sample = dataset[0]

    assert len(decoded) == 27
    assert sample["video"].shape == (3, 9, 352, 224)
    assert torch.allclose(sample["video"][:, :, 336:], torch.zeros_like(sample["video"][:, :, 336:]))
    assert sample["action"].shape == (32, 23)
    assert sample["proprio"].shape == (32, 23)
    assert sample["context"].shape == (128, 4)
    assert sample["context_mask"].shape == (128,)
    assert sample["image_is_pad"].shape == (9,)
    assert sample["action_is_pad"].shape == (32,)
    assert sample["proprio_is_pad"].shape == (32,)
    assert sample["context_mask"].sum().item() == 3
    assert sample["context"][3:].eq(0).all()


def test_window_crosses_parquet_row_groups_and_preserves_temporal_offsets(
    synthetic_lerobot_v21,
):
    root, cache_dir = synthetic_lerobot_v21
    parquet_path = root / "data/chunk-000/episode_000000.parquet"
    assert pq.ParquetFile(parquet_path).num_row_groups == 10

    sample = _dataset(root, cache_dir)[6]

    expected = torch.arange(6, 70, 2, dtype=torch.float32)
    torch.testing.assert_close(sample["action"][:, 0], expected)
    torch.testing.assert_close(sample["proprio"][:, 0], expected)
    assert not sample["action_is_pad"].any()
    assert sample["proprio_is_pad"].tolist() == [False] * 32
    assert sample["image_is_pad"].tolist() == [False] * 8 + [True]


def test_episode_tail_replicates_last_row_and_marks_each_padding_cadence(
    synthetic_lerobot_v21,
):
    root, cache_dir = synthetic_lerobot_v21
    sample = _dataset(root, cache_dir)[-1]

    assert sample["action"][:, 0].eq(69).all()
    assert sample["proprio"][:, 0].eq(69).all()
    assert sample["action_is_pad"].tolist() == [False] + [True] * 31
    assert sample["proprio_is_pad"].tolist() == [False] + [True] * 31
    assert sample["image_is_pad"].tolist() == [False] + [True] * 8


def test_text_context_lru_loads_once_and_does_not_cross_process_boundaries(
    synthetic_lerobot_v21, monkeypatch
):
    root, cache_dir = synthetic_lerobot_v21
    dataset = _dataset(root, cache_dir)
    prompt = format_b1k_prompt("test_task")
    original_load = torch.load
    load_calls = []

    def counted_load(*args, **kwargs):
        load_calls.append(args[0])
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", counted_load)
    first = dataset._get_cached_text_context(prompt)
    second = dataset._get_cached_text_context(prompt)
    assert len(load_calls) == 1
    assert first is second

    sample = dataset[0]
    assert sample["context"][3:].eq(0).all()
    assert not first[0][3:].eq(0).all(), "sample masking must not mutate the cache"

    restored = pickle.loads(pickle.dumps(dataset._text_context_cache))
    assert not restored._cache

    # A forked worker sees a different PID and must start with an empty cache.
    dataset._text_context_cache._pid = -1
    dataset._get_cached_text_context(prompt)
    assert len(load_calls) == 2
