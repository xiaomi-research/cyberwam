# Copyright (c) 2026 Xiaomi Corporation

import math

import torch

from fastwam.datasets.b1k.pointmap import (
    B1K_CAMERA_INTRINSICS,
    DEPTH_QMAX,
    MAX_DEPTH_M,
    MIN_DEPTH_M,
    dequantize_b1k_depth,
    depth_to_pointmap,
    make_b1k_pointmap_mosaic,
)


def test_official_log_depth_dequantization_endpoints():
    decoded = dequantize_b1k_depth(torch.tensor([0, DEPTH_QMAX]))
    assert torch.allclose(
        decoded, torch.tensor([MIN_DEPTH_M, MAX_DEPTH_M]), rtol=0, atol=2e-6
    )


def test_pointmap_unprojects_camera_center_and_zeros_invalid_before_normalizing():
    intrinsic = torch.tensor([[2.0, 0.0, 1.0], [0.0, 2.0, 1.0], [0.0, 0.0, 1.0]])
    depth = torch.ones(3, 3)
    depth[0, 0] = 10.1
    depth[0, 2] = 10.0
    pointmap = depth_to_pointmap(depth, intrinsic)

    assert torch.allclose(pointmap[:, 1, 1], torch.tensor([0.0, 0.0, -1 / 3]))
    # Invalid XYZ is zero before Flex-pi's fixed normalization.
    assert torch.allclose(pointmap[:, 0, 0], torch.tensor([0.0, 0.0, -1.0]))
    # The official quantization endpoint remains a valid, saturated far point.
    assert torch.allclose(pointmap[:, 0, 2], torch.tensor([1.0, -1.0, 1.0]))


def test_pointmap_mosaic_shape_layout_and_normalization():
    depths = {
        "head": torch.ones(2, 1, 720, 720),
        "left_wrist": torch.ones(2, 1, 480, 480),
        "right_wrist": torch.ones(2, 1, 480, 480),
    }
    mosaic = make_b1k_pointmap_mosaic(depths)

    assert mosaic.shape == (3, 2, 384, 256)
    assert torch.isfinite(mosaic).all()
    assert mosaic.min() >= -1.0 and mosaic.max() <= 1.0
    assert math.isclose(float(mosaic[2, 0, 360, 64]), -1 / 3, abs_tol=1e-6)
    assert B1K_CAMERA_INTRINSICS["head"][0, 2] == 360
