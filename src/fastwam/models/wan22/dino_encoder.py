# Copyright (c) 2026 Xiaomi Corporation

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


class B1KDinoEncoder(nn.Module):
    """Frozen DINOv3 encoder for the B1K 384x256 three-camera mosaic."""

    def __init__(
        self,
        model_name: str = "vit_base_patch16_dinov3.lvd1689m",
        model: nn.Module | None = None,
        pretrained_path: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
        image_size: int = 224,
        pixel_unshuffle: int = 2,
        output_grid_size: int | None = None,
    ) -> None:
        super().__init__()
        if pretrained_path is not None and checkpoint_path is not None:
            raise ValueError("Specify only one of `pretrained_path` and `checkpoint_path`.")
        if checkpoint_path is not None:
            pretrained_path = checkpoint_path
        if model is None:
            try:
                import timm
            except ImportError as exc:
                raise ImportError(
                    "B1KDinoEncoder requires timm. Install timm or inject `model=` in tests."
                ) from exc
            model = timm.create_model(
                model_name, pretrained=pretrained_path is None, num_classes=0
            )
            if pretrained_path is not None:
                self._load_local_checkpoint(model, pretrained_path)
        self.model = model
        self.model_name = str(model_name)
        self.image_size = int(image_size)
        self.pixel_unshuffle = int(pixel_unshuffle)
        if self.image_size <= 0 or self.pixel_unshuffle <= 0:
            raise ValueError("`image_size` and `pixel_unshuffle` must be positive.")

        patch_size: Any = getattr(getattr(model, "patch_embed", None), "patch_size", 16)
        if isinstance(patch_size, (tuple, list)):
            patch_size = patch_size[0]
        self.patch_size = int(patch_size)
        self.embed_dim = int(getattr(model, "embed_dim", 768))
        self.native_grid = self.image_size // self.patch_size
        if self.native_grid % self.pixel_unshuffle:
            raise ValueError(
                f"DINO grid {self.native_grid} must be divisible by fold {self.pixel_unshuffle}."
            )
        self.folded_grid = self.native_grid // self.pixel_unshuffle
        self.output_grid_size = (
            self.folded_grid if output_grid_size is None else int(output_grid_size)
        )
        if self.output_grid_size <= 0:
            raise ValueError("`output_grid_size` must be positive.")
        self.tokens_per_camera = self.output_grid_size**2
        self.tokens_per_frame = 3 * self.tokens_per_camera
        self.output_dim = self.embed_dim * self.pixel_unshuffle**2
        self.register_buffer(
            "_mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "_std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1), persistent=False
        )
        self.eval()
        self.requires_grad_(False)

    @staticmethod
    def _resolve_checkpoint_files(path: str | Path) -> list[Path]:
        """Resolve a timm/Hugging Face checkpoint file or directory."""
        path = Path(path)
        if path.is_file():
            return [path]
        if not path.is_dir():
            raise FileNotFoundError(f"DINO checkpoint does not exist: {path}")
        preferred = (
            "model.safetensors", "pytorch_model.bin", "model.bin", "pytorch_model.pt",
        )
        for name in preferred:
            candidate = path / name
            if candidate.is_file():
                return [candidate]
        files = sorted(path.glob("*.safetensors"))
        if not files:
            files = sorted(path.glob("*.bin")) + sorted(path.glob("*.pt"))
        if not files:
            raise FileNotFoundError(f"No DINO checkpoint weights found in {path}")
        return files

    @classmethod
    def _read_local_state_dict(cls, path: str | Path) -> dict[str, torch.Tensor]:
        state: dict[str, Any] = {}
        for file_path in cls._resolve_checkpoint_files(path):
            if file_path.suffix == ".safetensors":
                try:
                    from safetensors.torch import load_file
                except ImportError as exc:
                    raise ImportError("Loading .safetensors DINO weights requires safetensors.") from exc
                state.update(load_file(str(file_path), device="cpu"))
            else:
                payload = torch.load(file_path, map_location="cpu", weights_only=True)
                if isinstance(payload, Mapping):
                    state.update(payload)
                else:
                    raise ValueError(f"DINO checkpoint must contain a state dict: {file_path}")
        # Common wrappers emitted by timm/trainers.
        for key in ("state_dict", "model", "module", "model_state"):
            nested = state.get(key)
            if isinstance(nested, Mapping):
                state = dict(nested)
                break
        # Remove a uniform wrapper prefix without touching legitimate parameter names.
        for prefix in ("module.", "model."):
            keys = list(state)
            if keys and all(key.startswith(prefix) for key in keys):
                state = {key[len(prefix):]: value for key, value in state.items()}
        return {key: value for key, value in state.items() if isinstance(value, torch.Tensor)}

    @classmethod
    def _load_local_checkpoint(cls, model: nn.Module, path: str | Path) -> None:
        state = cls._read_local_state_dict(path)
        if not state:
            raise ValueError(f"DINO checkpoint contains no tensor parameters: {path}")
        incompatible = model.load_state_dict(state, strict=False)
        # A checkpoint with no matching parameters almost certainly indicates a wrong model.
        if len(incompatible.missing_keys) == len(model.state_dict()):
            raise ValueError(f"DINO checkpoint has no parameters compatible with model: {path}")

    def train(self, mode: bool = True):
        # This encoder is a fixed target generator, including when its parent trains.
        super().train(False)
        self.model.eval()
        return self

    @staticmethod
    def _camera_crops(frames: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if frames.shape[-2:] != (384, 256):
            raise ValueError(
                "B1K DINO expects the canonical 384x256 mosaic, got "
                f"{tuple(frames.shape[-2:])}."
            )
        return (
            frames[..., :256, :256],
            frames[..., 256:384, :128],
            frames[..., 256:384, 128:256],
        )

    def _patch_tokens(self, images: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            images.float(), size=(self.image_size, self.image_size), mode="bilinear",
            align_corners=False, antialias=True,
        )
        x = ((x + 1.0) * 0.5 - self._mean.float()) / self._std.float()
        output = self.model.forward_features(x.to(dtype=next(self.model.parameters()).dtype))
        if isinstance(output, dict):
            for key in ("x_norm_patchtokens", "x_prenorm", "x"):
                if key in output:
                    output = output[key]
                    break
        if not isinstance(output, torch.Tensor) or output.ndim != 3:
            raise ValueError("DINO `forward_features` must return patch tokens [N,P,D].")
        expected = self.native_grid**2
        if output.shape[1] != expected:
            num_prefix = int(getattr(self.model, "num_prefix_tokens", output.shape[1] - expected))
            output = output[:, num_prefix:]
        if output.shape[1] != expected or output.shape[2] != self.embed_dim:
            raise ValueError(
                f"Unexpected DINO patch shape {tuple(output.shape)}; expected [N,{expected},{self.embed_dim}]."
            )
        grid = output.reshape(output.shape[0], self.native_grid, self.native_grid, self.embed_dim)
        grid = grid.permute(0, 3, 1, 2)
        grid = F.pixel_unshuffle(grid, self.pixel_unshuffle)
        if self.output_grid_size != self.folded_grid:
            grid = F.interpolate(
                grid,
                size=(self.output_grid_size, self.output_grid_size),
                mode="bilinear",
                align_corners=False,
            )
        return grid.flatten(2).transpose(1, 2).contiguous()

    @torch.no_grad()
    def encode_video(self, video: torch.Tensor, temporal_downsample_factor: int = 4) -> torch.Tensor:
        if video.ndim != 5 or video.shape[1] != 3:
            raise ValueError(f"`video` must be [B,3,T,H,W], got {tuple(video.shape)}.")
        factor = int(temporal_downsample_factor)
        if factor <= 0 or (video.shape[2] - 1) % factor:
            raise ValueError(
                f"Video T={video.shape[2]} must satisfy (T-1) % factor == 0; factor={factor}."
            )
        slots = torch.arange(0, video.shape[2], factor, device=video.device)
        sampled = video.index_select(2, slots).permute(0, 2, 1, 3, 4)
        batch, frames = sampled.shape[:2]
        flat = sampled.reshape(batch * frames, 3, *sampled.shape[-2:])
        features = [self._patch_tokens(crop) for crop in self._camera_crops(flat)]
        features = torch.cat(features, dim=1).reshape(
            batch, frames, self.tokens_per_frame, self.output_dim
        )
        return features.permute(0, 3, 1, 2).unsqueeze(-1).contiguous()
