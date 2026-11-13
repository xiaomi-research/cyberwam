# Copyright (c) 2026 Xiaomi Corporation

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torchvision.transforms.functional as transforms_F
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from ..dataset_utils import Normalize
from .contract import ACTION_DIM, PROPRIO_DIM
from .dataset import (
    B1K_ACTION_OFFSETS,
    B1K_IMAGE_OFFSETS,
    B1K_STATE_OFFSETS,
    B1KRobotVideoDataset,
    _TextContextLRU,
)
from .parquet_reader import B1KParquetReader
from .pointmap import make_b1k_pointmap_mosaic
from .video_reader import B1KVideoReader


_VIDEO_SOURCES = (
    ("rgb.head", "observation.rgb.zed_link_camera_0"),
    ("rgb.left_wrist", "observation.rgb.left_realsense_link_camera_0"),
    ("rgb.right_wrist", "observation.rgb.right_realsense_link_camera_0"),
)
_DEPTH_SOURCES = (
    ("head", "observation.depth_linear.zed_link_camera_0"),
    ("left_wrist", "observation.depth_linear.left_realsense_link_camera_0"),
    ("right_wrist", "observation.depth_linear.right_realsense_link_camera_0"),
)
_EPISODE_COLUMNS = {
    "episode_index",
    "tasks",
    "length",
    "data/chunk_index",
    "data/file_index",
    "dataset_from_index",
    "dataset_to_index",
}


@dataclass(frozen=True)
class _CameraLocation:
    path: Path
    from_frame: int


@dataclass(frozen=True)
class _VideoEpisode:
    index: int
    length: int
    task: str
    dataset_from_index: int
    data_path: Path
    data_file_from_index: int
    cameras: tuple[_CameraLocation, ...]
    numeric_path: Path | None = None
    numeric_file_from_index: int | None = None
    depth_cameras: tuple[_CameraLocation, ...] = ()


def _select_direct_episodes(
    episodes: Sequence[_VideoEpisode],
    *,
    split: str,
    train_all_episodes: bool,
    expected_tasks: int,
    task_indices: Sequence[int] | None,
) -> list[_VideoEpisode]:
    if train_all_episodes and split != "train":
        raise ValueError("train_all_episodes is only valid for the train split.")

    grouped: dict[str, list[_VideoEpisode]] = defaultdict(list)
    for episode in episodes:
        grouped[episode.task].append(episode)

    selected_task_indices: set[int] | None = None
    if task_indices is not None:
        if isinstance(task_indices, (str, bytes)):
            raise TypeError("task_indices must be a sequence of integer task indices.")
        selected_task_indices = set()
        for task_index in task_indices:
            if isinstance(task_index, bool) or not isinstance(task_index, int):
                raise TypeError("task_indices must contain only integers.")
            if task_index < 0 or task_index >= expected_tasks:
                raise ValueError(
                    f"task index {task_index} is outside 0..{expected_tasks - 1}."
                )
            selected_task_indices.add(task_index)
        if not selected_task_indices:
            raise ValueError("task_indices must not be empty when provided.")

    selected: list[_VideoEpisode] = []
    ordered_tasks = sorted(grouped, key=lambda value: grouped[value][0].index)
    for task_index, task in enumerate(ordered_tasks):
        if selected_task_indices is not None and task_index not in selected_task_indices:
            continue
        task_episodes = sorted(grouped[task], key=lambda episode: episode.index)
        if split == "train":
            selected.extend(task_episodes if train_all_episodes else task_episodes[:-2])
        elif split == "validation":
            selected.append(task_episodes[-2])
        else:
            selected.append(task_episodes[-1])
    return selected


class B1KDirectVideoDataset(B1KRobotVideoDataset):
    """B1K LeRobot v3 backend reading shared parquet shards and source HEVC videos."""

    def __init__(
        self,
        dataset_root: str | Path,
        split: str,
        shape_meta: Mapping[str, Any],
        numeric_dataset_root: str | Path | None = None,
        num_frames: int = 33,
        video_size: list[int] = [384, 256],
        processor: Any | None = None,
        text_embedding_cache_dir: str | Path | None = None,
        context_len: int = 128,
        pretrained_norm_stats: str | Path | Mapping[str, Any] | None = None,
        is_training_set: bool | None = None,
        global_sample_stride: int = 2,
        window_start_stride: int = 1,
        action_video_freq_ratio: int = 4,
        skip_padding_as_possible: bool = False,
        override_instruction: str | None = None,
        expected_tasks: int = 100,
        episodes_per_task: int | None = 200,
        task_indices: Sequence[int] | None = None,
        parquet_file_cache_size: int = 8,
        parquet_row_group_cache_size: int = 0,
        text_cache_size: int = 128,
        max_open_videos: int = 16,
        max_decode_gap: int = 128,
        include_depth: bool = False,
        return_pointmap: bool = False,
        train_all_episodes: bool = False,
        **kwargs: Any,
    ) -> None:
        kwargs.pop("camera_key", None)
        kwargs.pop("max_padding_retry", None)
        if kwargs:
            raise TypeError(f"Unsupported B1K direct-video arguments: {sorted(kwargs)}")
        if skip_padding_as_possible:
            raise ValueError("B1K v1 uses natural-frame sampling and does not skip padded tails.")
        if num_frames not in {33, 65} or global_sample_stride != 2 or action_video_freq_ratio != 4:
            raise ValueError(
                "B1K direct video requires num_frames in {33, 65}, global_sample_stride=2, "
                "and action_video_freq_ratio=4."
            )
        if isinstance(window_start_stride, bool) or not isinstance(window_start_stride, int):
            raise TypeError("window_start_stride must be an integer.")
        if window_start_stride <= 0:
            raise ValueError("window_start_stride must be positive.")
        if list(video_size) != [384, 256]:
            raise ValueError(f"B1K direct video requires video_size=[384, 256], got {video_size}.")
        if context_len != 128:
            raise ValueError(f"B1K v1 requires context_len=128, got {context_len}.")
        if processor is None or pretrained_norm_stats is None:
            raise ValueError("B1K direct video requires a processor and train-only norm stats.")

        self.dataset_root = Path(dataset_root).expanduser()
        self.numeric_dataset_root = (
            None if numeric_dataset_root is None else Path(numeric_dataset_root).expanduser()
        )
        self.split = "validation" if split == "val" else split
        if self.split not in {"train", "validation", "test"}:
            raise ValueError(f"Unsupported B1K split {split!r}.")
        if not isinstance(train_all_episodes, bool):
            raise TypeError("train_all_episodes must be a boolean.")
        self.train_all_episodes = train_all_episodes
        self.is_training_set = self.split == "train"
        if is_training_set is not None and bool(is_training_set) != self.is_training_set:
            raise ValueError("is_training_set conflicts with the requested B1K split.")
        self.shape_meta = (
            OmegaConf.to_container(shape_meta, resolve=True)
            if OmegaConf.is_config(shape_meta)
            else dict(shape_meta)
        )
        self.num_frames = num_frames
        self._action_offsets = tuple(range(0, 2 * (num_frames - 1), 2))
        self._state_offsets = tuple(range(0, 2 * num_frames, 2))
        self._image_offsets = tuple(range(0, 2 * num_frames, 8))
        self.num_image_steps = len(self._image_offsets)
        self.context_len = context_len
        self.text_embedding_cache_dir = (
            None if text_embedding_cache_dir is None else Path(text_embedding_cache_dir)
        )
        self.override_instruction = override_instruction
        self.global_sample_stride = global_sample_stride
        self.window_start_stride = window_start_stride
        self.image_sample_indices = list(range(0, num_frames, action_video_freq_ratio))
        self.include_depth = bool(include_depth)
        self.return_pointmap = bool(return_pointmap)
        self._info = self._load_direct_info()
        all_episodes = self._load_direct_episodes(expected_tasks, episodes_per_task)
        if self.numeric_dataset_root is not None:
            all_episodes = self._attach_numeric_sidecar(all_episodes)
        selected = _select_direct_episodes(
            all_episodes,
            split=self.split,
            train_all_episodes=self.train_all_episodes,
            expected_tasks=expected_tasks,
            task_indices=task_indices,
        )
        self._episodes = selected
        self.selected_episodes = [episode.index for episode in selected]
        self._frame_prefix = [0]
        for episode in selected:
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
        self.lerobot_dataset = self

        if isinstance(processor, DictConfig):
            processor = instantiate(processor)
        processor.set_normalizer_from_stats(pretrained_norm_stats)
        self.processor = processor.train() if self.is_training_set else processor.eval()
        self._reader = B1KParquetReader(
            max_open_files=parquet_file_cache_size,
            max_numeric_row_groups=parquet_row_group_cache_size,
        )
        self._video_reader = B1KVideoReader(
            fps=float(self._info["fps"]),
            max_open_videos=max_open_videos,
            max_decode_gap=max_decode_gap,
        )
        self._depth_video_reader = (
            B1KVideoReader(
                fps=float(self._info["fps"]),
                max_open_videos=max_open_videos,
                max_decode_gap=max_decode_gap,
                is_depth=True,
            )
            if self._needs_depth
            else None
        )
        self._text_context_cache = _TextContextLRU(text_cache_size)
        self._validated_numeric_paths: set[Path] = set()
        self.normalize_transform = Normalize(args={"mean": 0.5, "std": 0.5})

    def _attach_numeric_sidecar(
        self, episodes: list[_VideoEpisode]
    ) -> list[_VideoEpisode]:
        """Bind v2 per-episode numeric parquet files to v3 video metadata."""
        root = self.numeric_dataset_root
        assert root is not None
        info_path = root / "meta" / "info.json"
        with info_path.open("r", encoding="utf-8") as stream:
            info = json.load(stream)
        required = {"data_path", "chunks_size", "fps", "features"}
        missing = required - set(info)
        if missing:
            raise ValueError(
                f"B1K numeric sidecar info.json is missing fields: {sorted(missing)}"
            )
        if int(info["fps"]) != int(self._info["fps"]):
            raise ValueError(
                f"B1K numeric sidecar fps {info['fps']} does not match video fps "
                f"{self._info['fps']}."
            )
        expected_shapes = {"action": [ACTION_DIM], "observation.state": [61]}
        for key, expected_shape in expected_shapes.items():
            feature = info["features"].get(key)
            if feature is None or list(feature.get("shape", [])) != expected_shape:
                actual = None if feature is None else feature.get("shape")
                raise ValueError(
                    f"B1K numeric sidecar feature {key!r} must have shape "
                    f"{expected_shape}, got {actual}."
                )

        metadata_path = root / "meta" / "episodes.jsonl"
        sidecar_records: dict[int, tuple[int, str]] = {}
        with metadata_path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    index = int(record["episode_index"])
                    length = int(record["length"])
                    tasks = record["tasks"]
                    if not isinstance(tasks, list) or len(tasks) != 1:
                        raise ValueError("expected exactly one task")
                    task = str(tasks[0])
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid B1K numeric sidecar episode at "
                        f"{metadata_path}:{line_number}."
                    ) from exc
                if index in sidecar_records:
                    raise ValueError(f"Duplicate B1K numeric sidecar episode {index}.")
                sidecar_records[index] = (length, task)

        expected_indices = {episode.index for episode in episodes}
        if set(sidecar_records) != expected_indices:
            missing_indices = sorted(expected_indices - set(sidecar_records))[:8]
            extra_indices = sorted(set(sidecar_records) - expected_indices)[:8]
            raise ValueError(
                "B1K numeric sidecar episode identities do not match the video dataset: "
                f"missing={missing_indices}, extra={extra_indices}."
            )

        attached: list[_VideoEpisode] = []
        chunks_size = int(info["chunks_size"])
        for episode in episodes:
            length, task = sidecar_records[episode.index]
            if length != episode.length or task != episode.task:
                raise ValueError(
                    f"B1K numeric sidecar episode {episode.index} mismatch: "
                    f"video=(length={episode.length}, task={episode.task!r}), "
                    f"sidecar=(length={length}, task={task!r})."
                )
            relative = info["data_path"].format(
                episode_chunk=episode.index // chunks_size,
                episode_index=episode.index,
            )
            attached.append(
                _VideoEpisode(
                    episode.index,
                    episode.length,
                    episode.task,
                    episode.dataset_from_index,
                    episode.data_path,
                    episode.data_file_from_index,
                    episode.cameras,
                    root / relative,
                    episode.dataset_from_index,
                    episode.depth_cameras,
                )
            )
        return attached

    def _load_direct_info(self) -> dict[str, Any]:
        path = self.dataset_root / "meta" / "info.json"
        with path.open("r", encoding="utf-8") as stream:
            info = json.load(stream)
        required = {"data_path", "video_path", "fps", "features"}
        missing = required - set(info)
        if missing:
            raise ValueError(f"B1K info.json is missing fields: {sorted(missing)}")
        if not str(info.get("codebase_version", "")).startswith("v3.") or int(info["fps"]) != 30:
            raise ValueError("B1K direct-video backend requires LeRobot v3 at 30 FPS.")
        required_features = {"action", "observation.state", *(key for _, key in _VIDEO_SOURCES)}
        if self._needs_depth:
            required_features.update(key for _, key in _DEPTH_SOURCES)
        missing_features = required_features - set(info["features"])
        if missing_features:
            raise ValueError(f"B1K source dataset is missing features: {sorted(missing_features)}")
        return info

    def _render_path(self, template: str, **values: Any) -> Path:
        return self.dataset_root / template.format(**values)

    def _load_direct_episodes(
        self, expected_tasks: int, episodes_per_task: int | None
    ) -> list[_VideoEpisode]:
        paths = sorted(self.dataset_root.glob("meta/episodes/chunk-*/file-*.parquet"))
        if not paths:
            raise FileNotFoundError(
                f"No LeRobot v3 episode metadata found under {self.dataset_root / 'meta/episodes'}."
            )
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError("B1K direct-video loading requires pyarrow.") from exc
        sources = _VIDEO_SOURCES + (_DEPTH_SOURCES if self._needs_depth else ())
        camera_columns = {
            f"videos/{key}/{suffix}"
            for _, key in sources
            for suffix in ("chunk_index", "file_index", "from_timestamp", "to_timestamp")
        }
        columns = sorted(_EPISODE_COLUMNS | camera_columns)
        rows: list[dict[str, Any]] = []
        for path in paths:
            table = pq.read_table(path, columns=columns)
            rows.extend(table.to_pylist())
        rows.sort(key=lambda row: int(row["episode_index"]))
        if episodes_per_task is not None and len(rows) != expected_tasks * episodes_per_task:
            raise ValueError(
                f"Expected {expected_tasks * episodes_per_task} B1K episodes, found {len(rows)}."
            )

        data_file_starts: dict[Path, int] = {}
        for row in rows:
            path = self._render_path(
                self._info["data_path"],
                chunk_index=int(row["data/chunk_index"]),
                file_index=int(row["data/file_index"]),
            )
            start = int(row["dataset_from_index"])
            data_file_starts[path] = min(start, data_file_starts.get(path, start))

        episodes: list[_VideoEpisode] = []
        task_counts: dict[str, int] = defaultdict(int)
        for row in rows:
            length = int(row["length"])
            start = int(row["dataset_from_index"])
            if int(row["dataset_to_index"]) - start != length:
                raise ValueError(f"Invalid dataset range for B1K episode {row['episode_index']}.")
            tasks = row["tasks"]
            if not isinstance(tasks, list) or len(tasks) != 1:
                raise ValueError(f"B1K episode {row['episode_index']} must have exactly one task.")
            task = str(tasks[0])
            task_counts[task] += 1
            data_path = self._render_path(
                self._info["data_path"],
                chunk_index=int(row["data/chunk_index"]),
                file_index=int(row["data/file_index"]),
            )
            camera_groups: list[tuple[_CameraLocation, ...]] = []
            for source_group in (_VIDEO_SOURCES, _DEPTH_SOURCES if self._needs_depth else ()):
                cameras = []
                for _, key in source_group:
                    prefix = f"videos/{key}/"
                    from_timestamp = float(row[f"{prefix}from_timestamp"])
                    to_timestamp = float(row[f"{prefix}to_timestamp"])
                    from_frame = round(from_timestamp * float(self._info["fps"]))
                    to_frame = round(to_timestamp * float(self._info["fps"]))
                    if to_frame - from_frame != length:
                        raise ValueError(
                            f"Video range for episode {row['episode_index']} camera {key} has "
                            f"{to_frame - from_frame} frames, expected {length}."
                        )
                    cameras.append(
                        _CameraLocation(
                            self._render_path(
                                self._info["video_path"],
                                video_key=key,
                                chunk_index=int(row[f"{prefix}chunk_index"]),
                                file_index=int(row[f"{prefix}file_index"]),
                            ),
                            from_frame,
                        )
                    )
                camera_groups.append(tuple(cameras))
            episodes.append(
                _VideoEpisode(
                    int(row["episode_index"]), length, task, start, data_path,
                    data_file_starts[data_path], camera_groups[0],
                    depth_cameras=camera_groups[1] if self._needs_depth else (),
                )
            )
        invalid_counts = len(task_counts) != expected_tasks
        if episodes_per_task is not None:
            invalid_counts = invalid_counts or set(task_counts.values()) != {episodes_per_task}
        if invalid_counts:
            raise ValueError(f"Unexpected B1K task/episode counts: {dict(task_counts)}")
        return episodes

    def _raw_samples(self, indices: Sequence[int]) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        numeric_requests: dict[Path, set[int]] = defaultdict(set)
        video_requests: dict[Path, set[int]] = defaultdict(set)
        depth_requests: dict[Path, set[int]] = defaultdict(set)
        locations = []
        for idx in indices:
            episode, local_frame = self._locate(int(idx))
            action_offsets = getattr(self, "_action_offsets", B1K_ACTION_OFFSETS)
            state_offsets = getattr(self, "_state_offsets", B1K_STATE_OFFSETS)
            image_offsets = getattr(self, "_image_offsets", B1K_IMAGE_OFFSETS)
            action_rows, action_is_pad = self._window(local_frame, action_offsets, episode.length)
            state_rows, state_is_pad = self._window(local_frame, state_offsets, episode.length)
            image_rows, image_is_pad = self._window(local_frame, image_offsets, episode.length)
            numeric_path = episode.numeric_path or episode.data_path
            numeric_file_from_index = (
                episode.data_file_from_index
                if episode.numeric_file_from_index is None
                else episode.numeric_file_from_index
            )
            validated_numeric_paths = getattr(self, "_validated_numeric_paths", set())
            if (
                episode.numeric_path is not None
                and numeric_path not in validated_numeric_paths
            ):
                actual_rows = self._reader.num_rows(numeric_path)
                if actual_rows != episode.length:
                    raise ValueError(
                        f"B1K numeric sidecar episode {episode.index} metadata length "
                        f"{episode.length} does not match parquet rows {actual_rows}."
                    )
                validated_numeric_paths.add(numeric_path)
                self._validated_numeric_paths = validated_numeric_paths
            numeric_local = [
                episode.dataset_from_index + row - numeric_file_from_index
                for row in sorted(set(action_rows + state_rows))
            ]
            numeric_requests[numeric_path].update(numeric_local)
            for camera in episode.cameras:
                video_requests[camera.path].update(camera.from_frame + row for row in image_rows)
            for camera in episode.depth_cameras:
                depth_requests[camera.path].update(camera.from_frame + row for row in image_rows)
            locations.append(
                (episode, local_frame, action_rows, state_rows, image_rows,
                 action_is_pad, state_is_pad, image_is_pad)
            )

        numeric_data: dict[Path, dict[int, tuple[Any, Any]]] = {}
        for path, rows in numeric_requests.items():
            ordered = sorted(rows)
            values = self._reader.read_rows(
                path, ["action", "observation.state"], ordered, cache_numeric=False
            )
            numeric_data[path] = {
                row: (values["action"][position], values["observation.state"][position])
                for position, row in enumerate(ordered)
            }
        decoded = self._video_reader.read_frames(
            {path: sorted(positions) for path, positions in video_requests.items()}
        )
        decoded_depth = (
            self._depth_video_reader.read_frames(
                {path: sorted(positions) for path, positions in depth_requests.items()}
            )
            if self._needs_depth
            and getattr(self, "_depth_video_reader", None) is not None
            else {}
        )

        for idx, located in zip(indices, locations, strict=True):
            (episode, local_frame, action_rows, state_rows, image_rows,
             action_is_pad, state_is_pad, image_is_pad) = located
            numeric_path = episode.numeric_path or episode.data_path
            numeric_file_from_index = (
                episode.data_file_from_index
                if episode.numeric_file_from_index is None
                else episode.numeric_file_from_index
            )
            lookup = numeric_data[numeric_path]
            action_keys = [
                episode.dataset_from_index + row - numeric_file_from_index
                for row in action_rows
            ]
            state_keys = [
                episode.dataset_from_index + row - numeric_file_from_index
                for row in state_rows
            ]
            images = {
                alias: torch.stack(
                    [decoded[camera.path][camera.from_frame + row] for row in image_rows]
                )
                for (alias, _), camera in zip(_VIDEO_SOURCES, episode.cameras, strict=True)
            }
            result = {
                "idx": int(idx),
                "task": episode.task,
                "action": {"default": torch.tensor([lookup[row][0] for row in action_keys], dtype=torch.float32)},
                "state": {"default": torch.tensor([lookup[row][1] for row in state_keys], dtype=torch.float32)},
                "images": images,
                "action_is_pad": action_is_pad,
                "state_is_pad": state_is_pad,
                "image_is_pad": image_is_pad,
                "episode_index": episode.index,
                "frame_index": local_frame,
            }
            if self._needs_depth:
                result["depths"] = {
                    name: torch.stack(
                        [decoded_depth[camera.path][camera.from_frame + row] for row in image_rows]
                    )
                    for (name, _), camera in zip(
                        _DEPTH_SOURCES, episode.depth_cameras, strict=True
                    )
                }
            samples.append(result)
        return samples

    def _raw_sample(self, idx: int) -> dict[str, Any]:
        return self._raw_samples([idx])[0]

    def _make_mosaic(self, images: Mapping[str, torch.Tensor]) -> torch.Tensor:
        expected_shapes = {
            "rgb.head": (self.num_image_steps, 3, 720, 720),
            "rgb.left_wrist": (self.num_image_steps, 3, 480, 480),
            "rgb.right_wrist": (self.num_image_steps, 3, 480, 480),
        }
        for key, expected in expected_shapes.items():
            if tuple(images[key].shape) != expected:
                raise ValueError(
                    f"Expected native B1K {key} frames {expected}, got {tuple(images[key].shape)}."
                )
        head = transforms_F.resize(images["rgb.head"], [256, 256], antialias=True)
        left = transforms_F.resize(images["rgb.left_wrist"], [128, 128], antialias=True)
        right = transforms_F.resize(images["rgb.right_wrist"], [128, 128], antialias=True)
        mosaic = torch.cat([head, torch.cat([left, right], dim=-1)], dim=-2)
        if mosaic.dtype == torch.uint8:
            mosaic = mosaic.to(dtype=torch.float32) / 255.0
        if mosaic.shape[-2:] != (384, 256):
            raise RuntimeError(f"Unexpected B1K direct-video mosaic shape {tuple(mosaic.shape)}.")
        return self.normalize_transform(mosaic).permute(1, 0, 2, 3)

    def _finish_raw(self, raw: dict[str, Any]) -> dict[str, Any]:
        sample = self.processor.preprocess_action_state(raw)
        video = self._make_mosaic(raw["images"])
        action = sample["action"]
        proprio = sample["proprio"][:-1]
        horizon = self.num_frames - 1
        if action.shape != (horizon, ACTION_DIM) or proprio.shape != (horizon, PROPRIO_DIM):
            raise ValueError(
                f"Unexpected B1K processed action/proprio shapes: {action.shape}, {proprio.shape}."
            )
        prompt = self._format_prompt(self.override_instruction or raw["task"])
        context, context_mask = self._get_cached_text_context(prompt)
        context = context.clone()
        context[~context_mask] = 0
        result = {
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
        if getattr(self, "include_depth", False):
            result["depths"] = raw["depths"]
        if getattr(self, "return_pointmap", False):
            result["pointmap"] = make_b1k_pointmap_mosaic(raw["depths"])
        return result

    @property
    def _needs_depth(self) -> bool:
        return bool(
            getattr(self, "include_depth", False)
            or getattr(self, "return_pointmap", False)
        )

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self._finish_raw(self._raw_sample(idx))

    def __getitems__(self, indices: Sequence[int]) -> list[dict[str, Any]]:
        return [self._finish_raw(raw) for raw in self._raw_samples(indices)]

    @property
    def checkpoint_data_contract(self) -> dict[str, Any]:
        contract = super().checkpoint_data_contract
        contract["action"]["horizon"] = int(getattr(self, "num_frames", 33)) - 1
        contract["video"]["dataset_image_indices"] = list(self.image_sample_indices)
        contract["video"]["raw_frame_indices"] = list(
            getattr(self, "_image_offsets", B1K_IMAGE_OFFSETS)
        )
        contract["video"]["canvas"] = {"height": 384, "width": 256}
        contract["video"]["layout"] = {
            "head": {"x": 0, "y": 0, "width": 256, "height": 256},
            "left_wrist": {"x": 0, "y": 256, "width": 128, "height": 128},
            "right_wrist": {"x": 128, "y": 256, "width": 128, "height": 128},
        }
        contract["video"].pop("padding", None)
        contract["video"]["source"] = "lerobot_v3_hevc"
        if self._needs_depth:
            contract["depth"] = {
                "source": "lerobot_v3_hevc_gray12le",
                "unit": "meter",
                "quantization": {
                    "bits": 12,
                    "mode": "log_shifted",
                    "min_m": 0.01,
                    "max_m": 10.0,
                    "shift": 3.5,
                },
            }
        if getattr(self, "return_pointmap", False):
            contract["pointmap"] = {
                "frame": "per_camera",
                "valid_depth_m": [0.01, 10.1],
                "normalization_bounds_m": {
                    "x": [-1.5, 1.5],
                    "y": [-1.5, 1.5],
                    "z": [0.0, 3.0],
                },
                "resize": "nearest_after_native_unprojection",
                "canvas": {"height": 384, "width": 256},
            }
        return contract
