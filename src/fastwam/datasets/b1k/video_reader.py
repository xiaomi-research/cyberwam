# Copyright (c) 2026 Xiaomi Corporation

from __future__ import annotations

import os
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from .pointmap import dequantize_b1k_depth


class _RandomAccessVideo:
    """Worker-local persistent PyAV decoder with exact frame-number access."""

    def __init__(self, path: Path, fps: float, *, max_decode_gap: int, is_depth: bool) -> None:
        try:
            import av
        except ImportError as exc:
            raise ImportError(
                "B1K direct-video loading requires PyAV. Install the project training environment."
            ) from exc
        self.path = path
        self.fps = fps
        self.max_decode_gap = max_decode_gap
        self.is_depth = is_depth
        self.container = av.open(str(path), mode="r")
        streams = self.container.streams.video
        if len(streams) != 1:
            raise ValueError(f"Expected one video stream in {path}, found {len(streams)}.")
        self.stream = streams[0]

    def close(self) -> None:
        self.container.close()

    def read_positions(self, positions: Iterable[int]) -> dict[int, torch.Tensor]:
        ordered = sorted(set(int(position) for position in positions))
        if not ordered:
            return {}
        if ordered[0] < 0:
            raise IndexError(f"Negative video frame requested from {self.path}: {ordered[0]}.")
        result: dict[int, torch.Tensor] = {}
        start = 0
        while start < len(ordered):
            end = start + 1
            while end < len(ordered) and ordered[end] - ordered[start] <= self.max_decode_gap:
                end += 1
            result.update(self._read_span(ordered[start:end]))
            start = end
        return result

    def _frame_number(self, frame: Any) -> int:
        if frame.pts is None:
            raise ValueError(f"Decoded frame without PTS in {self.path}.")
        start_time = self.stream.start_time or 0
        seconds = float((frame.pts - start_time) * self.stream.time_base)
        return round(seconds * self.fps)

    def _read_span(self, requested: Sequence[int]) -> dict[int, torch.Tensor]:
        first, last = requested[0], requested[-1]
        start_time = self.stream.start_time or 0
        seek_pts = start_time + round((first / self.fps) / float(self.stream.time_base))
        self.container.seek(max(0, seek_pts), stream=self.stream, backward=True, any_frame=False)
        wanted = set(requested)
        result: dict[int, torch.Tensor] = {}
        previous: int | None = None
        for frame in self.container.decode(self.stream):
            position = self._frame_number(frame)
            if previous is not None and position <= previous:
                raise ValueError(
                    f"Non-increasing decoded PTS in {self.path}: {previous} -> {position}."
                )
            previous = position
            if position in wanted:
                if self.is_depth:
                    quantized = np.asarray(
                        frame.reformat(format="gray12le").to_ndarray(), dtype=np.uint16
                    ).copy()
                    depth = dequantize_b1k_depth(torch.from_numpy(quantized))
                    result[position] = depth.unsqueeze(0)
                else:
                    rgb = np.asarray(frame.to_ndarray(format="rgb24"), dtype=np.uint8).copy()
                    result[position] = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
            if position >= last:
                break
        missing = sorted(wanted - set(result))
        if missing:
            raise ValueError(f"Missing source frames in {self.path}: {missing[:8]}.")
        return result


class B1KVideoReader:
    """Bounded worker-local LRU of persistent random-access video decoders."""

    def __init__(
        self,
        *,
        fps: float = 30.0,
        max_open_videos: int = 16,
        max_decode_gap: int = 128,
        is_depth: bool = False,
    ) -> None:
        if fps <= 0 or max_open_videos <= 0 or max_decode_gap <= 0:
            raise ValueError("Invalid B1K video reader configuration.")
        self.fps = float(fps)
        self.max_open_videos = max_open_videos
        self.max_decode_gap = max_decode_gap
        self.is_depth = bool(is_depth)
        self._pid = os.getpid()
        self._videos: OrderedDict[str, _RandomAccessVideo] = OrderedDict()

    def _ensure_worker_local(self) -> None:
        pid = os.getpid()
        if pid != self._pid:
            self._pid = pid
            self._videos = OrderedDict()

    def __getstate__(self) -> dict[str, Any]:
        return {
            "fps": self.fps,
            "max_open_videos": self.max_open_videos,
            "max_decode_gap": self.max_decode_gap,
            "is_depth": self.is_depth,
            "_pid": os.getpid(),
            "_videos": OrderedDict(),
        }

    def close(self) -> None:
        for video in self._videos.values():
            video.close()
        self._videos.clear()

    def _video(self, path: str | Path) -> _RandomAccessVideo:
        self._ensure_worker_local()
        key = str(path)
        video = self._videos.pop(key, None)
        if video is None:
            if not Path(key).is_file():
                raise FileNotFoundError(f"Missing B1K source video: {key}")
            video = _RandomAccessVideo(
                Path(key), self.fps, max_decode_gap=self.max_decode_gap, is_depth=self.is_depth
            )
        self._videos[key] = video
        while len(self._videos) > self.max_open_videos:
            _, evicted = self._videos.popitem(last=False)
            evicted.close()
        return video

    def read_frames(self, requests: dict[Path, Sequence[int]]) -> dict[Path, dict[int, torch.Tensor]]:
        return {path: self._video(path).read_positions(positions) for path, positions in requests.items()}
