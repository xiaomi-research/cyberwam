# Copyright (c) 2026 Xiaomi Corporation

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn.functional as F


MIN_DEPTH_M = 0.01
MAX_DEPTH_M = 10.0
DEPTH_SHIFT = 3.5
DEPTH_QMAX = (1 << 12) - 1
# Keep the official 10 m endpoint valid despite the strict upper-bound check.
POINTMAP_MAX_DEPTH_M = 10.1
POINTMAP_MIN = torch.tensor([-1.5, -1.5, 0.0], dtype=torch.float32)
POINTMAP_MAX = torch.tensor([1.5, 1.5, 3.0], dtype=torch.float32)
B1K_CAMERA_INTRINSICS = {
    "head": torch.tensor([[306.0, 0.0, 360.0], [0.0, 306.0, 360.0], [0.0, 0.0, 1.0]]),
    "left_wrist": torch.tensor([[388.6639, 0.0, 240.0], [0.0, 388.6639, 240.0], [0.0, 0.0, 1.0]]),
    "right_wrist": torch.tensor([[388.6639, 0.0, 240.0], [0.0, 388.6639, 240.0], [0.0, 0.0, 1.0]]),
}


def dequantize_b1k_depth(quantized: torch.Tensor) -> torch.Tensor:
    """Decode the official B1K 12-bit log quantization into metric depth."""
    quantized = quantized.to(dtype=torch.float32)
    log_min = torch.log(quantized.new_tensor(MIN_DEPTH_M + DEPTH_SHIFT))
    log_max = torch.log(quantized.new_tensor(MAX_DEPTH_M + DEPTH_SHIFT))
    depth = torch.exp((quantized / DEPTH_QMAX) * (log_max - log_min) + log_min) - DEPTH_SHIFT
    return depth.clamp(MIN_DEPTH_M, MAX_DEPTH_M)


def depth_to_pointmap(depth: torch.Tensor, intrinsic: torch.Tensor) -> torch.Tensor:
    """Unproject ``[..., H, W]`` metric depth to normalized camera-frame XYZ."""
    depth = depth.float()
    intrinsic = intrinsic.to(device=depth.device, dtype=depth.dtype)
    height, width = depth.shape[-2:]
    v, u = torch.meshgrid(
        torch.arange(height, device=depth.device, dtype=depth.dtype),
        torch.arange(width, device=depth.device, dtype=depth.dtype), indexing="ij"
    )
    z = depth
    x = (u - intrinsic[0, 2]) / intrinsic[0, 0] * z
    y = (v - intrinsic[1, 2]) / intrinsic[1, 1] * z
    valid = torch.isfinite(z) & (z > MIN_DEPTH_M) & (z < POINTMAP_MAX_DEPTH_M)
    xyz = torch.stack([x, y, z], dim=-3)
    xyz = torch.where(valid.unsqueeze(-3), xyz, torch.zeros_like(xyz))
    lower = POINTMAP_MIN.to(xyz).view(3, 1, 1)
    upper = POINTMAP_MAX.to(xyz).view(3, 1, 1)
    return 2.0 * (xyz.clamp(lower, upper) - lower) / (upper - lower) - 1.0


def make_b1k_pointmap_mosaic(depths: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Project native cameras, nearest-resize, and make ``[3,T,384,256]``."""
    specs = {"head": ((720, 720), (256, 256)), "left_wrist": ((480, 480), (128, 128)),
             "right_wrist": ((480, 480), (128, 128))}
    pointmaps = {}
    for name, (native, target) in specs.items():
        depth = depths[name]
        if depth.ndim == 4 and depth.shape[1] == 1:
            depth = depth[:, 0]
        if tuple(depth.shape[-2:]) != native:
            raise ValueError(f"Expected {name} depth resolution {native}, got {tuple(depth.shape)}.")
        xyz = depth_to_pointmap(depth, B1K_CAMERA_INTRINSICS[name])
        pointmaps[name] = F.interpolate(xyz, size=target, mode="nearest")
    mosaic = torch.cat(
        [pointmaps["head"], torch.cat([pointmaps["left_wrist"], pointmaps["right_wrist"]], -1)], -2
    )
    if mosaic.shape[-3:] != (3, 384, 256):
        raise RuntimeError(f"Unexpected B1K pointmap mosaic shape {tuple(mosaic.shape)}.")
    return mosaic.permute(1, 0, 2, 3).contiguous()
