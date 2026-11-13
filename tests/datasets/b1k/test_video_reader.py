# Copyright (c) 2026 Xiaomi Corporation

from pathlib import Path

import torch

from fastwam.datasets.b1k.video_reader import _RandomAccessVideo


def test_random_access_reader_splits_requests_by_maximum_decode_span():
    video = _RandomAccessVideo.__new__(_RandomAccessVideo)
    video.path = Path("/fake/depth.mp4")
    video.max_decode_gap = 64
    spans = []

    def fake_read_span(requested):
        spans.append(list(requested))
        return {position: torch.tensor(position) for position in requested}

    video._read_span = fake_read_span

    decoded = video.read_positions([96, 0, 64, 32, 64])

    assert spans == [[0, 32, 64], [96]]
    assert all(span[-1] - span[0] <= video.max_decode_gap for span in spans)
    assert list(decoded) == [0, 32, 64, 96]
    assert [decoded[position].item() for position in decoded] == [0, 32, 64, 96]
