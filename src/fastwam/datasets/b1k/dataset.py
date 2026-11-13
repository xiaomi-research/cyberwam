# Copyright (c) 2026 Xiaomi Corporation

from __future__ import annotations

import hashlib
import io
import json
import os
from collections import OrderedDict
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torchvision.transforms.functional as transforms_F
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from ..dataset_utils import Normalize
from .contract import (
    ACTION_CONSTANT_DIM,
    ACTION_DIM,
    ACTION_GRIPPER_DIMS,
    ACTION_LABELS,
    PROPRIO_DIM,
    PROPRIO_LABELS,
)
from .parquet_reader import B1KParquetReader
from .split import B1KSplit, build_b1k_episode_split
from .text import B1K_PROMPT_TEMPLATE, format_b1k_prompt


B1K_ACTION_OFFSETS = tuple(range(0, 64, 2))
B1K_STATE_OFFSETS = tuple(range(0, 65, 2))
B1K_IMAGE_OFFSETS = tuple(range(0, 65, 8))
B1K_IMAGE_SAMPLE_INDICES = list(range(0, 33, 4))

_ACTION_COLUMN = "action"
_STATE_COLUMN = "observation.state"
_CAMERA_COLUMNS = (
    "observation.images.rgb.head",
    "observation.images.rgb.left_wrist",
    "observation.images.rgb.right_wrist",
)
_CAMERA_KEYS = ("rgb.head", "rgb.left_wrist", "rgb.right_wrist")


@dataclass(frozen=True)
class _Episode:
    index: int
    length: int
    task: str
    parquet_path: Path


class _TextContextLRU:
    """Worker-local cache for immutable T5 context payloads."""

    def __init__(self, max_size: int) -> None:
        if max_size <= 0:
            raise ValueError(f"text_cache_size must be positive, got {max_size}.")
        self.max_size = max_size
        self._pid = os.getpid()
        self._cache: OrderedDict[str, tuple[torch.Tensor, torch.Tensor]] = OrderedDict()

    def _ensure_worker_local(self) -> None:
        pid = os.getpid()
        if pid != self._pid:
            self._pid = pid
            self._cache.clear()

    def get(self, key: str) -> tuple[torch.Tensor, torch.Tensor] | None:
        self._ensure_worker_local()
        value = self._cache.pop(key, None)
        if value is not None:
            self._cache[key] = value
        return value

    def put(self, key: str, value: tuple[torch.Tensor, torch.Tensor]) -> None:
        self._ensure_worker_local()
        self._cache.pop(key, None)
        self._cache[key] = value
        while len(self._cache) > self.max_size:
            self._cache.popitem(last=False)

    def __getstate__(self) -> dict[str, Any]:
        return {
            "max_size": self.max_size,
            "_pid": os.getpid(),
            "_cache": OrderedDict(),
        }


class B1KRobotVideoDataset(torch.utils.data.Dataset):
    """Direct per-episode B1K parquet dataset with the fixed 2026 contract."""

    def __init__(
        self,
        dataset_root: str | Path,
        split: B1KSplit,
        shape_meta: Mapping[str, Any],
        num_frames: int = 33,
        video_size: list[int] = [352, 224],
        camera_key: str | None = None,
        processor: Any | None = None,
        text_embedding_cache_dir: str | Path | None = None,
        context_len: int = 128,
        pretrained_norm_stats: str | Path | Mapping[str, Any] | None = None,
        is_training_set: bool | None = None,
        global_sample_stride: int = 2,
        window_start_stride: int = 1,
        action_video_freq_ratio: int = 4,
        skip_padding_as_possible: bool = False,
        max_padding_retry: int = 3,
        override_instruction: str | None = None,
        expected_tasks: int = 100,
        episodes_per_task: int = 200,
        parquet_file_cache_size: int = 8,
        parquet_row_group_cache_size: int = 32,
        text_cache_size: int = 128,
        **kwargs: Any,
    ) -> None:
        del camera_key, max_padding_retry
        if kwargs:
            raise TypeError(f"Unsupported B1K dataset arguments: {sorted(kwargs)}")
        if skip_padding_as_possible:
            raise ValueError("B1K v1 uses natural-frame sampling and does not skip padded tails.")
        if num_frames != 33 or global_sample_stride != 2 or action_video_freq_ratio != 4:
            raise ValueError(
                "B1K v1 requires num_frames=33, global_sample_stride=2, and "
                "action_video_freq_ratio=4."
            )
        if isinstance(window_start_stride, bool) or not isinstance(window_start_stride, int):
            raise TypeError("window_start_stride must be an integer.")
        if window_start_stride <= 0:
            raise ValueError("window_start_stride must be positive.")
        if list(video_size) != [352, 224]:
            raise ValueError(f"B1K v1 requires video_size=[352, 224], got {video_size}.")
        if context_len != 128:
            raise ValueError(f"B1K v1 requires context_len=128, got {context_len}.")
        if processor is None:
            raise ValueError("B1K dataset requires B1KFastWAMProcessor.")
        if pretrained_norm_stats is None:
            raise ValueError("B1K dataset requires precomputed train-only norm stats.")

        self.dataset_root = Path(dataset_root).expanduser()
        self.split = "validation" if split == "val" else split
        expected_training = self.split == "train"
        if is_training_set is not None and bool(is_training_set) != expected_training:
            raise ValueError("is_training_set conflicts with the requested B1K split.")
        self.is_training_set = expected_training
        self.shape_meta = (
            OmegaConf.to_container(shape_meta, resolve=True)
            if OmegaConf.is_config(shape_meta)
            else dict(shape_meta)
        )
        self.num_frames = num_frames
        self.num_image_steps = len(B1K_IMAGE_OFFSETS)
        self.context_len = context_len
        self.text_embedding_cache_dir = (
            None if text_embedding_cache_dir is None else Path(text_embedding_cache_dir)
        )
        self.override_instruction = override_instruction
        self.global_sample_stride = global_sample_stride
        self.window_start_stride = window_start_stride
        self.image_sample_indices = list(B1K_IMAGE_SAMPLE_INDICES)

        self._info = self._load_info()
        selected = build_b1k_episode_split(
            self.dataset_root,
            self.split,
            expected_tasks=expected_tasks,
            episodes_per_task=episodes_per_task,
        )
        self._episodes = self._load_episodes(selected)
        self.selected_episodes = [episode.index for episode in self._episodes]
        self._frame_prefix = [0]
        for episode in self._episodes:
            num_anchors = (
                episode.length + self.window_start_stride - 1
            ) // self.window_start_stride
            self._frame_prefix.append(self._frame_prefix[-1] + num_anchors)
        if self._frame_prefix[-1] <= 0:
            raise ValueError(f"B1K split {self.split!r} contains no frames.")
        self.episode_data_index = {
            "from": torch.tensor(self._frame_prefix[:-1], dtype=torch.long),
            "to": torch.tensor(self._frame_prefix[1:], dtype=torch.long),
        }
        # Transitional compatibility for existing trainer/overfit consumers.
        # New B1K code should access processor and episode_data_index directly.
        self.lerobot_dataset = self

        if isinstance(processor, DictConfig):
            processor = instantiate(processor)
        processor.set_normalizer_from_stats(pretrained_norm_stats)
        self.processor = processor.train() if self.is_training_set else processor.eval()
        self._reader = B1KParquetReader(
            max_open_files=parquet_file_cache_size,
            max_numeric_row_groups=parquet_row_group_cache_size,
        )
        self._text_context_cache = _TextContextLRU(text_cache_size)
        self.normalize_transform = Normalize(args={"mean": 0.5, "std": 0.5})

    def _load_info(self) -> dict[str, Any]:
        path = self.dataset_root / "meta" / "info.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing B1K dataset metadata: {path}")
        with path.open("r", encoding="utf-8") as stream:
            info = json.load(stream)
        required = {"data_path", "chunks_size", "fps", "features"}
        missing = required - set(info)
        if missing:
            raise ValueError(f"B1K info.json is missing fields: {sorted(missing)}")
        if int(info["fps"]) != 30:
            raise ValueError(f"B1K v1 requires source fps=30, got {info['fps']}.")
        required_columns = {_ACTION_COLUMN, _STATE_COLUMN, *_CAMERA_COLUMNS}
        missing_columns = required_columns - set(info["features"])
        if missing_columns:
            raise ValueError(f"B1K data is missing columns: {sorted(missing_columns)}")
        return info

    def _episode_path(self, episode_index: int) -> Path:
        episode_chunk = episode_index // int(self._info["chunks_size"])
        relative = self._info["data_path"].format(
            episode_chunk=episode_chunk, episode_index=episode_index
        )
        return self.dataset_root / relative

    def _load_episodes(self, selected: list[int]) -> list[_Episode]:
        selected_set = set(selected)
        records: dict[int, _Episode] = {}
        path = self.dataset_root / "meta" / "episodes.jsonl"
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    index = int(record["episode_index"])
                    if index not in selected_set:
                        continue
                    length = int(record["length"])
                    tasks = record["tasks"]
                    task = str(tasks[0])
                except (json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid B1K episode record at {path}:{line_number}.") from exc
                if length <= 0 or len(tasks) != 1:
                    raise ValueError(
                        f"B1K episode {index} must have positive length and exactly one task."
                    )
                records[index] = _Episode(index, length, task, self._episode_path(index))
        missing = selected_set - set(records)
        if missing:
            raise ValueError(f"Missing selected B1K episode metadata: {sorted(missing)[:8]}")
        return [records[index] for index in selected]

    def __len__(self) -> int:
        return self._frame_prefix[-1]

    def _locate(self, idx: int) -> tuple[_Episode, int]:
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(f"B1K sample index {idx} out of range for length {len(self)}.")
        episode_position = bisect_right(self._frame_prefix, idx) - 1
        local_anchor = idx - self._frame_prefix[episode_position]
        return self._episodes[episode_position], local_anchor * self.window_start_stride

    @staticmethod
    def _window(local_frame: int, offsets: tuple[int, ...], length: int) -> tuple[list[int], torch.Tensor]:
        requested = [local_frame + offset for offset in offsets]
        is_pad = torch.tensor([index >= length for index in requested], dtype=torch.bool)
        return [min(index, length - 1) for index in requested], is_pad

    @staticmethod
    def _decode_jpeg(payload: Any) -> torch.Tensor:
        if isinstance(payload, Mapping):
            payload = payload.get("bytes")
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise TypeError(f"Expected embedded JPEG bytes, got {type(payload)!r}.")
        with Image.open(io.BytesIO(payload)) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        return torch.from_numpy(rgb).permute(2, 0, 1).contiguous()

    def _raw_sample(self, idx: int) -> dict[str, Any]:
        episode, local_frame = self._locate(idx)
        actual_rows = self._reader.num_rows(episode.parquet_path)
        if actual_rows != episode.length:
            raise ValueError(
                f"B1K episode {episode.index} metadata length {episode.length} does not "
                f"match parquet rows {actual_rows}."
            )
        action_rows, action_is_pad = self._window(local_frame, B1K_ACTION_OFFSETS, episode.length)
        state_rows, state_is_pad = self._window(local_frame, B1K_STATE_OFFSETS, episode.length)
        image_rows, image_is_pad = self._window(local_frame, B1K_IMAGE_OFFSETS, episode.length)

        numeric_rows = sorted(set(action_rows + state_rows))
        numeric = self._reader.read_rows(
            episode.parquet_path,
            [_ACTION_COLUMN, _STATE_COLUMN],
            numeric_rows,
            cache_numeric=True,
        )
        numeric_lookup = {
            row: (numeric[_ACTION_COLUMN][pos], numeric[_STATE_COLUMN][pos])
            for pos, row in enumerate(numeric_rows)
        }
        action = torch.tensor(
            [numeric_lookup[row][0] for row in action_rows], dtype=torch.float32
        )
        state = torch.tensor(
            [numeric_lookup[row][1] for row in state_rows], dtype=torch.float32
        )

        encoded_images = self._reader.read_rows(
            episode.parquet_path,
            _CAMERA_COLUMNS,
            image_rows,
            cache_numeric=False,
        )
        images = {
            key: torch.stack([self._decode_jpeg(payload) for payload in encoded_images[column]])
            for key, column in zip(_CAMERA_KEYS, _CAMERA_COLUMNS, strict=True)
        }
        return {
            "idx": idx,
            "task": episode.task,
            "action": {"default": action},
            "state": {"default": state},
            "images": images,
            "action_is_pad": action_is_pad,
            "state_is_pad": state_is_pad,
            "image_is_pad": image_is_pad,
            "episode_index": episode.index,
            "frame_index": local_frame,
        }

    def _make_mosaic(self, video: torch.Tensor) -> torch.Tensor:
        if video.shape[:2] != (3, self.num_image_steps):
            raise ValueError(f"Expected B1K camera tensor [3,9,C,H,W], got {tuple(video.shape)}")
        if video.shape[-2:] != (224, 224):
            raise ValueError(
                f"B1K v1 requires decoded camera frames [224,224], got {tuple(video.shape[-2:])}."
            )
        head = video[0]
        left = transforms_F.resize(video[1], [112, 112], antialias=True)
        right = transforms_F.resize(video[2], [112, 112], antialias=True)
        mosaic = torch.cat([head, torch.cat([left, right], dim=-1)], dim=-2)
        if mosaic.dtype == torch.uint8:
            mosaic = mosaic.to(dtype=torch.float32) / 255.0
        mosaic = transforms_F.pad(mosaic, [0, 0, 0, 16], fill=0.5)
        if mosaic.shape[-2:] != (352, 224):
            raise RuntimeError(f"Unexpected B1K mosaic shape {tuple(mosaic.shape)}.")
        mosaic = self.normalize_transform(mosaic)
        return mosaic.permute(1, 0, 2, 3)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        raw = self._raw_sample(idx)
        sample = self.processor.preprocess(raw)
        video = self._make_mosaic(sample["pixel_values"])
        action = sample["action"]
        proprio = sample["proprio"][:-1]
        if action.shape != (32, ACTION_DIM) or proprio.shape != (32, PROPRIO_DIM):
            raise ValueError(
                f"Unexpected B1K processed action/proprio shapes: {action.shape}, {proprio.shape}."
            )
        task = self.override_instruction or raw["task"]
        prompt = self._format_prompt(task)
        context, context_mask = self._get_cached_text_context(prompt)
        context = context.clone()
        context[~context_mask] = 0
        return {
            "video": video,
            "action": action,
            "proprio": proprio,
            "prompt": prompt,
            "context": context,
            "context_mask": context_mask,
            "image_is_pad": raw["image_is_pad"],
            "action_is_pad": sample["action_is_pad"],
            "proprio_is_pad": sample["proprio_is_pad"][:-1],
            "idx": raw["idx"],
            "episode_index": raw["episode_index"],
            "frame_index": raw["frame_index"],
        }

    def _get_cached_text_context(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        if self.text_embedding_cache_dir is None:
            raise ValueError("text_embedding_cache_dir is not set.")
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cached = self._text_context_cache.get(digest)
        if cached is not None:
            return cached
        path = self.text_embedding_cache_dir / f"{digest}.t5_len{self.context_len}.wan22ti2v5b.pt"
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing text embedding cache: {path}. Run scripts/precompute_text_embeds.py first."
            )
        payload = torch.load(path, map_location="cpu")
        context = payload["context"]
        mask = payload["mask"].bool()
        if context.ndim != 2 or context.shape[0] != self.context_len:
            raise ValueError(f"Invalid cached B1K context shape {tuple(context.shape)} in {path}.")
        if mask.shape != (self.context_len,):
            raise ValueError(f"Invalid cached B1K mask shape {tuple(mask.shape)} in {path}.")
        cached = (context, mask)
        self._text_context_cache.put(digest, cached)
        return cached

    def _format_prompt(self, task: str) -> str:
        return format_b1k_prompt(task)

    @property
    def checkpoint_data_contract(self) -> dict[str, Any]:
        return {
            "schema": "b1k_2026_data_contract.v2",
            "sampling": {
                "source_fps": 30,
                "window_start_stride": self.window_start_stride,
                "anchor_frequency_hz": 30.0 / self.window_start_stride,
                "sample_count": len(self),
                "tail_policy": "replicate_last_frame_with_padding_mask",
            },
            "action": {
                "dim": ACTION_DIM,
                "labels": list(ACTION_LABELS),
                "horizon": 32,
                "sample_stride": self.global_sample_stride,
                "gripper_dims": list(ACTION_GRIPPER_DIMS),
                "constant_dims": [ACTION_CONSTANT_DIM],
                "gripper_mask": [i in ACTION_GRIPPER_DIMS for i in range(ACTION_DIM)],
                "constant_dim_mask": [i == ACTION_CONSTANT_DIM for i in range(ACTION_DIM)],
            },
            "proprio": {"dim": PROPRIO_DIM, "labels": list(PROPRIO_LABELS)},
            "video": {
                "dataset_image_indices": list(self.image_sample_indices),
                "raw_frame_indices": list(B1K_IMAGE_OFFSETS),
                "canvas": {"height": 352, "width": 224},
                "camera_order": ["head", "left_wrist", "right_wrist"],
                "layout": {
                    "head": {"x": 0, "y": 0, "width": 224, "height": 224},
                    "left_wrist": {"x": 0, "y": 224, "width": 112, "height": 112},
                    "right_wrist": {"x": 112, "y": 224, "width": 112, "height": 112},
                },
                "padding": {
                    "x": 0,
                    "y": 336,
                    "width": 224,
                    "height": 16,
                    "value_before_normalization": 0.5,
                    "value_after_normalization": 0.0,
                },
            },
            "text": {
                "context_length": self.context_len,
                "prompt_template": B1K_PROMPT_TEMPLATE,
                "attention_mask": "real_token_mask",
            },
            "normalization": {
                "mode": "mixed_z_score",
                "action_continuous_mode": "global_per_dimension_z_score_unclamped",
                "proprio_continuous_mode": "global_per_dimension_z_score_clamped",
                "stepwise": False,
                "proprio_clamp": [-5.0, 5.0],
                "action_and_proprio_stats": "independent",
                "gripper_action": "fixed_binary_minus_one_plus_one",
                "constant_action": "fixed_zero",
            },
        }
