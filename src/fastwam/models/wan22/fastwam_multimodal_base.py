# Copyright (c) 2026 Xiaomi Corporation

from __future__ import annotations

import copy
import logging
import time
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .dino_encoder import B1KDinoEncoder
from .fastwam_joint import FastWAMJoint
from .schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler
from .wan_video_dit import sinusoidal_embedding_1d


class FastWAMMultimodalBase(FastWAMJoint):
    """Fixed four-stream FastWAM: RGB, DINO, pointmap, and action.

    The three visual streams share the existing video expert blocks.  Only
    their input/output adapters differ.  This intentionally excludes Flex-pi
    stream dropout, cross-modal forcing, HBridge, and dynamic output masks.
    """

    multimodal_new_state_prefixes = (
        "dino_feature_norm",
        "dino_embedder",
        "dino_proj_out",
        "pt_patch_embedding",
        "pt_head",
    )
    trainable_module_names = (
        "mot", "dino_feature_norm", "dino_embedder", "dino_proj_out",
        "pt_patch_embedding", "pt_head", "proprio_encoder",
    )

    def __init__(
        self,
        *args,
        dino_encoder: Optional[nn.Module] = None,
        dino_pretrained_path: Optional[str] = None,
        dino_dim: int = 3072,
        dino_pred_x0: bool = True,
        dino_train_shift: float = 6.0,
        dino_infer_shift: float = 6.0,
        dino_num_train_timesteps: int = 1000,
        pointmap_train_shift: float = 6.0,
        pointmap_infer_shift: float = 6.0,
        pointmap_num_train_timesteps: int = 1000,
        loss_lambda_dino: float = 1.0,
        loss_lambda_pointmap: float = 1.0,
        modality_profile: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.dino_encoder = dino_encoder
        self.dino_dim = int(dino_dim)
        self.dino_pred_x0 = bool(dino_pred_x0)
        self.loss_lambda_dino = float(loss_lambda_dino)
        self.loss_lambda_pointmap = float(loss_lambda_pointmap)
        self.modality_profile = bool(modality_profile)

        hidden_dim = int(self.video_expert.hidden_dim)
        self.dino_feature_norm = nn.LayerNorm(self.dino_dim)
        self.dino_embedder = nn.Linear(self.dino_dim, hidden_dim)
        self.dino_proj_out = nn.Linear(hidden_dim, self.dino_dim)
        self._init_dino_layers()
        self.pt_patch_embedding = copy.deepcopy(self.video_expert.patch_embedding)
        self.pt_head = copy.deepcopy(self.video_expert.head)

        self.train_dino_scheduler = WanContinuousFlowMatchScheduler(
            dino_num_train_timesteps, dino_train_shift
        )
        self.infer_dino_scheduler = WanContinuousFlowMatchScheduler(
            dino_num_train_timesteps, dino_infer_shift
        )
        self.train_pointmap_scheduler = WanContinuousFlowMatchScheduler(
            pointmap_num_train_timesteps, pointmap_train_shift
        )
        self.infer_pointmap_scheduler = WanContinuousFlowMatchScheduler(
            pointmap_num_train_timesteps, pointmap_infer_shift
        )

        if self.dino_encoder is not None:
            self.dino_encoder.eval().requires_grad_(False)
        self.to(device=self.device, dtype=self.torch_dtype)

    @classmethod
    def from_wan22_pretrained(cls, **kwargs):
        model_init_kwargs = dict(kwargs.pop("model_init_kwargs", {}))
        multimodal_keys = {
            "dino_encoder", "dino_pretrained_path", "dino_dim", "dino_pred_x0", "dino_train_shift",
            "dino_infer_shift", "dino_num_train_timesteps", "pointmap_train_shift",
            "pointmap_infer_shift", "pointmap_num_train_timesteps", "loss_lambda_dino",
            "loss_lambda_pointmap", "modality_profile",
        }
        multimodal = {key: kwargs.pop(key) for key in list(kwargs) if key in multimodal_keys}
        dino_model_name = kwargs.pop("dino_model_name", None)
        dino_output_grid_size = kwargs.pop("dino_output_grid_size", None)
        if "dino_encoder" not in multimodal and dino_model_name is not None:
            multimodal["dino_encoder"] = B1KDinoEncoder(
                model_name=dino_model_name,
                pretrained_path=multimodal.get("dino_pretrained_path"),
                output_grid_size=dino_output_grid_size,
            )
        model_init_kwargs.update(multimodal)
        kwargs["model_init_kwargs"] = model_init_kwargs
        return super().from_wan22_pretrained(**kwargs)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.dino_encoder is not None:
            self.dino_encoder.eval()
        self.vae.eval()
        return self

    def _multimodal_state_dict(self) -> dict[str, dict[str, torch.Tensor]]:
        return {
            name: getattr(self, name).state_dict()
            for name in self.multimodal_new_state_prefixes
        }

    def _checkpoint_extra_payload(self) -> dict[str, Any]:
        return {"multimodal": self._multimodal_state_dict()}

    def load_checkpoint(self, path, optimizer=None, **kwargs):
        payload = super().load_checkpoint(path, optimizer=optimizer, **kwargs)
        multimodal = payload.get("multimodal")
        if multimodal is None:
            raise ValueError(
                "FastWAMMultimodalBase checkpoint is missing `multimodal` adapter/head weights. "
                "Use the dedicated legacy warm-start loader for a legacy FastWAM checkpoint."
            )
        expected = set(self.multimodal_new_state_prefixes)
        if set(multimodal) != expected:
            raise RuntimeError(
                "FastWAMMultimodalBase checkpoint module mismatch: "
                f"missing={sorted(expected - set(multimodal))}, "
                f"unexpected={sorted(set(multimodal) - expected)}."
            )
        for name in self.multimodal_new_state_prefixes:
            getattr(self, name).load_state_dict(multimodal[name], strict=True)
        return payload

    def _init_dino_layers(self) -> None:
        nn.init.ones_(self.dino_feature_norm.weight)
        nn.init.zeros_(self.dino_feature_norm.bias)
        for module in (self.dino_embedder, self.dino_proj_out):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
        if self.dino_pred_x0:
            nn.init.zeros_(self.dino_proj_out.weight)
            nn.init.zeros_(self.dino_proj_out.bias)

    @torch.no_grad()
    def initialize_pointmap_from_rgb(self) -> None:
        """Copy the currently loaded RGB adapters after FastWAM warm-start."""
        self.pt_patch_embedding.load_state_dict(self.video_expert.patch_embedding.state_dict())
        self.pt_head.load_state_dict(self.video_expert.head.state_dict())

    def build_inputs(self, sample, tiled: bool = False):
        profile = self.modality_profile and torch.cuda.is_available()
        if profile:
            torch.cuda.synchronize()
            rgb_started = time.perf_counter()
        inputs = super().build_inputs(sample, tiled=tiled)
        if profile:
            torch.cuda.synchronize()
            self._log_modality_profile("rgb_vae", time.perf_counter() - rgb_started)
        if self.dino_encoder is None:
            raise ValueError("FastWAMMultimodalBase requires a frozen `dino_encoder`.")
        if "pointmap" not in sample:
            raise ValueError("FastWAMMultimodalBase requires `sample['pointmap']` [B,3,T,H,W].")
        video = sample["video"].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        pointmap = sample["pointmap"]
        if pointmap.shape != video.shape:
            raise ValueError(
                f"Pointmap shape must match RGB video, got {tuple(pointmap.shape)} vs {tuple(video.shape)}."
            )
        with torch.no_grad():
            if profile:
                torch.cuda.synchronize()
                dino_started = time.perf_counter()
            dino = self.dino_encoder.encode_video(
                video, temporal_downsample_factor=int(self.vae.temporal_downsample_factor)
            )
            if profile:
                torch.cuda.synchronize()
                self._log_modality_profile("dino", time.perf_counter() - dino_started)
                torch.cuda.synchronize()
                pointmap_started = time.perf_counter()
            pointmap_latents = self._encode_video_latents(
                pointmap.to(device=self.device, dtype=self.torch_dtype, non_blocking=True), tiled=tiled
            )
            if profile:
                torch.cuda.synchronize()
                self._log_modality_profile("pointmap_vae", time.perf_counter() - pointmap_started)
        if dino.shape[1] != self.dino_dim:
            raise ValueError(f"DINO feature dim must be {self.dino_dim}, got {dino.shape[1]}.")
        if dino.shape[2] != inputs["input_latents"].shape[2]:
            raise ValueError("DINO and RGB latent frame counts must match.")
        if pointmap_latents.shape != inputs["input_latents"].shape:
            raise ValueError(
                "Pointmap and RGB VAE latent shapes must match, got "
                f"{tuple(pointmap_latents.shape)} vs {tuple(inputs['input_latents'].shape)}."
            )
        inputs["dino_features"] = dino.to(device=self.device, dtype=self.torch_dtype)
        inputs["pointmap_latents"] = pointmap_latents
        return inputs

    def _log_modality_profile(self, name: str, elapsed: float) -> None:
        if not self.modality_profile:
            return
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            if torch.distributed.get_rank() != 0:
                return
        logging.getLogger(__name__).info("modality_profile %s=%.4fs", name, elapsed)

    @staticmethod
    def build_multimodal_attention_mask(
        stream_lengths: tuple[int, int, int],
        anchor_lengths: tuple[int, int, int],
        action_seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        if any(length <= 0 for length in stream_lengths + anchor_lengths) or action_seq_len <= 0:
            raise ValueError("All stream, anchor, and action lengths must be positive.")
        if any(anchor > length for anchor, length in zip(anchor_lengths, stream_lengths)):
            raise ValueError("An anchor length cannot exceed its stream length.")
        visual_len = sum(stream_lengths)
        total = visual_len + action_seq_len
        mask = torch.zeros((total, total), dtype=torch.bool, device=device)
        offsets = []
        start = 0
        for length in stream_lengths:
            offsets.append((start, start + length))
            start += length

        # Every future visual query can read every visual token. Anchor queries
        # can only read all three clean anchors, never any future or action.
        for (begin, end), anchor in zip(offsets, anchor_lengths):
            mask[begin + anchor:end, :visual_len] = True
        for (q_begin, _), q_anchor in zip(offsets, anchor_lengths):
            for (k_begin, _), k_anchor in zip(offsets, anchor_lengths):
                mask[q_begin:q_begin + q_anchor, k_begin:k_begin + k_anchor] = True

        mask[visual_len:, :visual_len] = True
        mask[visual_len:, visual_len:] = True
        return mask

    def _prepare_attention_routing(
        self, batch_size: int, device: torch.device, *, training: bool
    ) -> Any:
        del batch_size, device, training
        return None

    def _apply_visual_input_routing(
        self,
        noisy_streams: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        routing: Any,
    ) -> None:
        del noisy_streams, routing

    def _build_routed_attention_masks(
        self,
        stream_lengths: tuple[int, int, int],
        anchor_lengths: tuple[int, int, int],
        action_seq_len: int,
        device: torch.device,
        routing: Any,
    ) -> tuple[torch.Tensor, Optional[list[torch.Tensor]]]:
        del routing
        return (
            self.build_multimodal_attention_mask(
                stream_lengths, anchor_lengths, action_seq_len, device
            ),
            None,
        )

    def _reduce_routed_loss(
        self, loss: torch.Tensor, routing: Any, modality_index: int
    ) -> torch.Tensor:
        del routing, modality_index
        return loss.mean()

    def _build_stream_tensors(
        self, timestep: torch.Tensor, frames: int, tokens_per_frame: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = timestep.shape[0]
        seq_len = frames * tokens_per_frame
        token_t = timestep.view(batch, 1).expand(batch, seq_len).clone()
        token_t[:, :tokens_per_frame] = 0
        t_emb = sinusoidal_embedding_1d(self.video_expert.freq_dim, token_t.reshape(-1))
        t = self.video_expert.time_embedding(t_emb).reshape(batch, seq_len, -1)
        t_mod = self.video_expert.time_projection(t).unflatten(
            2, (6, self.video_expert.hidden_dim)
        )
        return t, t_mod

    def _grid_freqs(self, frames: int, height: int, width: int, device: torch.device):
        ft, fh, fw = self.video_expert.freqs
        if frames > ft.shape[0] or height > fh.shape[0] or width > fw.shape[0]:
            raise ValueError("Auxiliary stream exceeds the Video DiT RoPE cache.")
        return torch.cat(
            [
                ft[:frames].view(frames, 1, 1, -1).expand(frames, height, width, -1),
                fh[:height].view(1, height, 1, -1).expand(frames, height, width, -1),
                fw[:width].view(1, 1, width, -1).expand(frames, height, width, -1),
            ],
            dim=-1,
        ).reshape(frames * height * width, 1, -1).to(device)

    def _dino_freqs(self, frames: int, tokens_per_frame: int, device: torch.device):
        if tokens_per_frame % 3:
            raise ValueError(f"B1K DINO tokens/frame must be divisible by 3, got {tokens_per_frame}.")
        per_camera = tokens_per_frame // 3
        side = int(per_camera**0.5)
        if side * side != per_camera:
            raise ValueError(f"Each B1K DINO camera token grid must be square, got {per_camera}.")
        ft, fh, fw = self.video_expert.freqs
        if frames > ft.shape[0] or 2 * side > fh.shape[0] or 2 * side > fw.shape[0]:
            raise ValueError("DINO virtual camera grid exceeds the Video DiT RoPE cache.")

        # Use disjoint regions in a virtual 2*side square. Token order remains
        # frame-major, then head/left/right camera-major, matching encode_video.
        origins = ((0, 0), (side, 0), (side, side))
        per_camera_freqs = []
        for h0, w0 in origins:
            h_idx = torch.arange(h0, h0 + side)
            w_idx = torch.arange(w0, w0 + side)
            hh, ww = torch.meshgrid(h_idx, w_idx, indexing="ij")
            spatial = torch.cat([fh[hh.reshape(-1)], fw[ww.reshape(-1)]], dim=-1)
            per_frame = torch.cat(
                [
                    ft[:frames, None, :].expand(-1, per_camera, -1),
                    spatial[None, :, :].expand(frames, -1, -1),
                ],
                dim=-1,
            )
            per_camera_freqs.append(per_frame)
        return torch.stack(per_camera_freqs, dim=1).reshape(
            frames * tokens_per_frame, 1, -1
        ).to(device)

    def _embed_dino(self, features: torch.Tensor) -> torch.Tensor:
        flat = features.squeeze(-1).permute(0, 2, 3, 1).reshape(features.shape[0], -1, self.dino_dim)
        return self.dino_embedder(self.dino_feature_norm(flat))

    def _embed_pointmap(self, latents: torch.Tensor):
        x = self.pt_patch_embedding(latents)
        grid = tuple(int(v) for v in x.shape[2:])
        return rearrange(x, "b c f h w -> b (f h w) c").contiguous(), grid

    def _project_pointmap(self, tokens: torch.Tensor, time: torch.Tensor, grid):
        f, h, w = grid
        x = self.pt_head(tokens, time)
        pt, ph, pw = self.video_expert.patch_size
        return rearrange(
            x, "b (f h w) (pt ph pw c) -> b c (f pt) (h ph) (w pw)",
            f=f, h=h, w=w, pt=pt, ph=ph, pw=pw,
        )

    @staticmethod
    def _x0_to_velocity(x_t, x0_pred, timestep, num_train_timesteps: int):
        sigma = (timestep.float() / float(num_train_timesteps)).clamp(min=0.05)
        return (x_t.float() - x0_pred.float()) / sigma

    @staticmethod
    def _dino_loss_per_sample(
        prediction: torch.Tensor,
        target: torch.Tensor,
        *,
        frames: int,
        tokens_per_frame: int,
        image_is_pad: Optional[torch.Tensor],
        temporal_factor: int,
    ) -> torch.Tensor:
        batch, sequence, channels = prediction.shape
        if target.shape != prediction.shape or sequence != frames * tokens_per_frame:
            raise ValueError("DINO prediction/target shape does not match its frame layout.")
        token_loss = F.mse_loss(
            prediction.float(), target.float(), reduction="none"
        ).mean(dim=2).reshape(batch, frames, tokens_per_frame).mean(dim=2)
        future_loss = token_loss[:, 1:]
        if image_is_pad is None:
            return future_loss.mean(dim=1)
        if (image_is_pad.shape[1] - 1) % temporal_factor:
            raise ValueError("DINO padding mask cannot be aligned to latent frames.")
        future_is_pad = image_is_pad[:, 1:].reshape(
            batch, frames - 1, temporal_factor
        ).all(dim=2)
        valid = (~future_is_pad).to(future_loss)
        return (future_loss * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)

    def _training_loss_impl(self, sample, tiled: bool = False):
        inputs = self.build_inputs(sample, tiled=tiled)
        rgb = inputs["input_latents"]
        dino = inputs["dino_features"]
        pointmap = inputs["pointmap_latents"]
        action = inputs["action"]
        context, context_mask = inputs["context"], inputs["context_mask"]
        batch = rgb.shape[0]

        def noisy_target(clean, scheduler, timestep=None):
            noise = torch.randn_like(clean)
            if timestep is None:
                timestep = scheduler.sample_training_t(batch, self.device, clean.dtype)
            else:
                timestep = timestep.to(device=self.device, dtype=clean.dtype)
            return (
                scheduler.add_noise(clean, noise, timestep),
                scheduler.training_target(clean, noise, timestep),
                timestep,
            )

        noisy_rgb, target_rgb, t_rgb = noisy_target(rgb, self.train_video_scheduler)
        # Flex inference advances all four streams on a single step index, so
        # sharing the RGB timestep here removes the train/inference mismatch
        # once every stream uses the same schedule shift.
        shared_t = t_rgb if self.shared_train_timestep else None
        noisy_dino, target_dino, t_dino = noisy_target(
            dino, self.train_dino_scheduler, shared_t
        )
        noisy_pt, target_pt, t_pt = noisy_target(
            pointmap, self.train_pointmap_scheduler, shared_t
        )
        noisy_action, target_action, t_action = noisy_target(
            action, self.train_action_scheduler, shared_t
        )
        noisy_rgb[:, :, :1] = rgb[:, :, :1]
        noisy_dino[:, :, :1] = dino[:, :, :1]
        noisy_pt[:, :, :1] = pointmap[:, :, :1]
        routing = self._prepare_attention_routing(batch, self.device, training=True)
        self._apply_visual_input_routing((noisy_rgb, noisy_dino, noisy_pt), routing)

        video_pre = self.video_expert.pre_dit(
            x=noisy_rgb, timestep=t_rgb, context=context, context_mask=context_mask,
            action=action, fuse_vae_embedding_in_latents=True,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action, timestep=t_action, context=context, context_mask=context_mask
        )
        dino_tokens = self._embed_dino(noisy_dino)
        dino_frames, dino_tpf = int(dino.shape[2]), int(dino.shape[3])
        _, dino_t_mod = self._build_stream_tensors(t_dino, dino_frames, dino_tpf)
        dino_freqs = self._dino_freqs(dino_frames, dino_tpf, dino_tokens.device)

        pt_tokens, pt_grid = self._embed_pointmap(noisy_pt)
        pt_frames, pt_h, pt_w = pt_grid
        pt_tpf = pt_h * pt_w
        pt_time, pt_t_mod = self._build_stream_tensors(t_pt, pt_frames, pt_tpf)
        pt_freqs = self._grid_freqs(pt_frames, pt_h, pt_w, pt_tokens.device)

        sv = video_pre["tokens"].shape[1]
        sd, sp, sa = dino_tokens.shape[1], pt_tokens.shape[1], action_pre["tokens"].shape[1]
        rgb_tpf = int(video_pre["meta"]["tokens_per_frame"])
        visual_tokens = torch.cat([video_pre["tokens"], dino_tokens, pt_tokens], dim=1)
        visual_freqs = torch.cat([video_pre["freqs"], dino_freqs, pt_freqs], dim=0)
        visual_t_mod = torch.cat([video_pre["t_mod"], dino_t_mod, pt_t_mod], dim=1)
        visual_context_mask = video_pre["context_mask"]
        if visual_context_mask.ndim == 3:
            aux_mask = visual_context_mask[:, :1].expand(-1, sd + sp, -1)
            visual_context_mask = torch.cat([visual_context_mask, aux_mask], dim=1)
        attention_mask, layer_attention_masks = self._build_routed_attention_masks(
            (sv, sd, sp),
            (rgb_tpf, dino_tpf, pt_tpf),
            sa,
            visual_tokens.device,
            routing,
        )
        if self.modality_profile and torch.cuda.is_available():
            torch.cuda.synchronize()
            mot_started = time.perf_counter()
        output = self.mot(
            embeds_all={"video": visual_tokens, "action": action_pre["tokens"]},
            attention_mask=attention_mask,
            freqs_all={"video": visual_freqs, "action": action_pre["freqs"]},
            context_all={
                "video": {"context": video_pre["context"], "mask": visual_context_mask},
                "action": {"context": action_pre["context"], "mask": action_pre["context_mask"]},
            },
            t_mod_all={"video": visual_t_mod, "action": action_pre["t_mod"]},
            layer_attention_masks=layer_attention_masks,
        )
        if self.modality_profile and torch.cuda.is_available():
            torch.cuda.synchronize()
            self._log_modality_profile("mot_forward", time.perf_counter() - mot_started)
        pred_rgb = self.video_expert.post_dit(output["video"][:, :sv], video_pre)[:, :, 1:]
        pred_dino = self.dino_proj_out(output["video"][:, sv:sv + sd])
        pred_pt = self._project_pointmap(output["video"][:, sv + sd:], pt_time, pt_grid)[:, :, 1:]
        pred_action = self.action_expert.post_dit(output["action"], action_pre)

        lv = self._compute_video_loss_per_sample(
            pred_rgb, target_rgb[:, :, 1:], inputs["image_is_pad"], False
        )
        loss_video = self._reduce_routed_loss(
            lv * self.train_video_scheduler.training_weight(t_rgb).to(lv)
            , routing, 0
        )

        target_dino_flat = target_dino.squeeze(-1).permute(0, 2, 3, 1).reshape(batch, sd, self.dino_dim)
        if self.dino_pred_x0:
            noisy_dino_flat = noisy_dino.squeeze(-1).permute(0, 2, 3, 1).reshape(batch, sd, self.dino_dim)
            pred_dino = self._x0_to_velocity(
                noisy_dino_flat, pred_dino, t_dino.view(batch, 1, 1),
                self.train_dino_scheduler.num_train_timesteps,
            )
        ld = self._dino_loss_per_sample(
            pred_dino,
            target_dino_flat,
            frames=dino_frames,
            tokens_per_frame=dino_tpf,
            image_is_pad=inputs["image_is_pad"],
            temporal_factor=int(self.vae.temporal_downsample_factor),
        )
        loss_dino = self._reduce_routed_loss(
            ld * self.train_dino_scheduler.training_weight(t_dino).to(ld)
            , routing, 1
        )

        lp = self._compute_video_loss_per_sample(
            pred_pt, target_pt[:, :, 1:], inputs["image_is_pad"], False
        )
        loss_pointmap = self._reduce_routed_loss(
            lp * self.train_pointmap_scheduler.training_weight(t_pt).to(lp)
            , routing, 2
        )

        la = F.mse_loss(pred_action.float(), target_action.float(), reduction="none").mean(2)
        if inputs["action_is_pad"] is not None:
            valid = (~inputs["action_is_pad"]).to(la)
            la = (la * valid).sum(1) / valid.sum(1).clamp(min=1)
        else:
            la = la.mean(1)
        loss_action = (la * self.train_action_scheduler.training_weight(t_action).to(la)).mean()

        total = (
            self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
            + self.loss_lambda_dino * loss_dino + self.loss_lambda_pointmap * loss_pointmap
        )
        return total, {
            "loss_video": self.loss_lambda_video * float(loss_video.detach()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach()),
            "loss_dino": self.loss_lambda_dino * float(loss_dino.detach()),
            "loss_pointmap": self.loss_lambda_pointmap * float(loss_pointmap.detach()),
        }

    @torch.no_grad()
    def _predict_multimodal_noise(
        self,
        *,
        latents_video: torch.Tensor,
        latents_dino: torch.Tensor,
        latents_pointmap: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_video: torch.Tensor,
        timestep_dino: torch.Tensor,
        timestep_pointmap: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one fixed-topology denoising step for all Stage-1 streams."""
        routing = self._prepare_attention_routing(
            int(latents_video.shape[0]), latents_video.device, training=False
        )
        self._apply_visual_input_routing(
            (latents_video, latents_dino, latents_pointmap), routing
        )
        video_pre = self.video_expert.pre_dit(
            x=latents_video,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        dino_tokens = self._embed_dino(latents_dino)
        dino_frames, dino_tpf = int(latents_dino.shape[2]), int(latents_dino.shape[3])
        _, dino_t_mod = self._build_stream_tensors(
            timestep_dino, dino_frames, dino_tpf
        )
        dino_freqs = self._dino_freqs(
            dino_frames, dino_tpf, dino_tokens.device
        )

        pointmap_tokens, pointmap_grid = self._embed_pointmap(latents_pointmap)
        pointmap_frames, pointmap_h, pointmap_w = pointmap_grid
        pointmap_tpf = pointmap_h * pointmap_w
        pointmap_time, pointmap_t_mod = self._build_stream_tensors(
            timestep_pointmap, pointmap_frames, pointmap_tpf
        )
        pointmap_freqs = self._grid_freqs(
            pointmap_frames,
            pointmap_h,
            pointmap_w,
            pointmap_tokens.device,
        )

        rgb_len = int(video_pre["tokens"].shape[1])
        dino_len = int(dino_tokens.shape[1])
        pointmap_len = int(pointmap_tokens.shape[1])
        action_len = int(action_pre["tokens"].shape[1])
        rgb_tpf = int(video_pre["meta"]["tokens_per_frame"])
        visual_tokens = torch.cat(
            [video_pre["tokens"], dino_tokens, pointmap_tokens], dim=1
        )
        visual_freqs = torch.cat(
            [video_pre["freqs"], dino_freqs, pointmap_freqs], dim=0
        )
        visual_t_mod = torch.cat(
            [video_pre["t_mod"], dino_t_mod, pointmap_t_mod], dim=1
        )
        visual_context_mask = video_pre["context_mask"]
        if visual_context_mask.ndim == 3:
            auxiliary_mask = visual_context_mask[:, :1].expand(
                -1, dino_len + pointmap_len, -1
            )
            visual_context_mask = torch.cat(
                [visual_context_mask, auxiliary_mask], dim=1
            )
        attention_mask, layer_attention_masks = self._build_routed_attention_masks(
            (rgb_len, dino_len, pointmap_len),
            (rgb_tpf, dino_tpf, pointmap_tpf),
            action_len,
            visual_tokens.device,
            routing,
        )
        output = self.mot(
            embeds_all={"video": visual_tokens, "action": action_pre["tokens"]},
            attention_mask=attention_mask,
            freqs_all={"video": visual_freqs, "action": action_pre["freqs"]},
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": visual_context_mask,
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={"video": visual_t_mod, "action": action_pre["t_mod"]},
            layer_attention_masks=layer_attention_masks,
        )

        pred_video = self.video_expert.post_dit(
            output["video"][:, :rgb_len], video_pre
        )
        pred_dino = self.dino_proj_out(
            output["video"][:, rgb_len:rgb_len + dino_len]
        )
        pred_dino = pred_dino.reshape(
            latents_dino.shape[0], dino_frames, dino_tpf, self.dino_dim
        ).permute(0, 3, 1, 2).unsqueeze(-1).contiguous()
        if self.dino_pred_x0:
            pred_dino = self._x0_to_velocity(
                latents_dino,
                pred_dino,
                timestep_dino.view(-1, 1, 1, 1, 1),
                self.infer_dino_scheduler.num_train_timesteps,
            ).to(dtype=latents_dino.dtype)
        pred_pointmap = self._project_pointmap(
            output["video"][:, rgb_len + dino_len:],
            pointmap_time,
            pointmap_grid,
        )
        pred_action = self.action_expert.post_dit(output["action"], action_pre)
        return pred_video, pred_dino, pred_pointmap, pred_action

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        input_pointmap: torch.Tensor,
        action_horizon: int,
        num_video_frames: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        return_joint_outputs: bool = False,
    ) -> dict[str, Any]:
        """Jointly sample RGB, DINO, pointmap futures and an action chunk."""
        del negative_prompt, text_cfg_scale
        self.eval()
        if self.dino_encoder is None:
            raise ValueError("FastWAMMultimodalBase inference requires a frozen `dino_encoder`.")

        def validate_image(value: torch.Tensor, name: str) -> torch.Tensor:
            if value.ndim == 3:
                value = value.unsqueeze(0)
            if value.ndim != 4 or value.shape[0] != 1 or value.shape[1] != 3:
                raise ValueError(
                    f"`{name}` must have shape [1,3,H,W] or [3,H,W], "
                    f"got {tuple(value.shape)}"
                )
            return value

        input_image = validate_image(input_image, "input_image")
        input_pointmap = validate_image(input_pointmap, "input_pointmap")
        if input_pointmap.shape != input_image.shape:
            raise ValueError(
                "`input_pointmap` must match `input_image`, got "
                f"{tuple(input_pointmap.shape)} vs {tuple(input_image.shape)}."
            )
        _, _, height, width = input_image.shape
        checked_h, checked_w, checked_t = self._check_resize_height_width(
            height, width, num_video_frames
        )
        if (checked_h, checked_w) != (height, width):
            raise ValueError(
                "Stage-1 inference inputs must be resized to multiples of 32; "
                f"got HxW=({height},{width})."
            )
        if checked_t != num_video_frames:
            raise ValueError(
                f"`num_video_frames` must satisfy T % 4 == 1, got {num_video_frames}."
            )
        if num_video_frames <= 1:
            raise ValueError("`num_video_frames` must include at least one future frame.")
        if action_horizon <= 0:
            raise ValueError("`action_horizon` must be positive.")

        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError(
                    "`proprio` was provided but `proprio_dim=None` disables its encoder."
                )
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            if proprio.ndim != 2 or proprio.shape != (1, self.proprio_dim):
                raise ValueError(
                    f"`proprio` must be [D] or [1,D] with D={self.proprio_dim}, "
                    f"got {tuple(proprio.shape)}."
                )
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")
        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    "`context/context_mask` must be [B,L,D]/[B,L], got "
                    f"{tuple(context.shape)} and {tuple(context_mask.shape)}."
                )
            if context.shape[0] != 1 or context_mask.shape[0] != 1:
                raise ValueError("Stage-1 inference currently supports batch size 1.")
            context = context.to(
                device=self.device, dtype=self.torch_dtype, non_blocking=True
            )
            context_mask = context_mask.to(
                device=self.device, dtype=torch.bool, non_blocking=True
            )
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context, context_mask=context_mask, proprio=proprio
            )

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        input_pointmap = input_pointmap.to(
            device=self.device, dtype=self.torch_dtype
        )
        first_video = self._encode_input_image_latents_tensor(
            input_image=input_image, tiled=tiled
        )
        first_pointmap = self._encode_input_image_latents_tensor(
            input_image=input_pointmap, tiled=tiled
        )
        first_dino = self.dino_encoder.encode_video(
            input_image.unsqueeze(2),
            temporal_downsample_factor=int(self.vae.temporal_downsample_factor),
        ).to(device=self.device, dtype=self.torch_dtype)
        if first_dino.shape[1] != self.dino_dim or first_dino.shape[2] != 1:
            raise ValueError(
                "Unexpected current DINO feature shape "
                f"{tuple(first_dino.shape)}; expected [1,{self.dino_dim},1,P,1]."
            )

        latent_frames = (
            (num_video_frames - 1) // int(self.vae.temporal_downsample_factor) + 1
        )
        latent_h, latent_w = int(first_video.shape[-2]), int(first_video.shape[-1])

        generator = (
            None
            if seed is None
            else torch.Generator(device=rand_device).manual_seed(seed)
        )

        def random_latents(shape: tuple[int, ...]) -> torch.Tensor:
            return torch.randn(
                shape,
                generator=generator,
                device=rand_device,
                dtype=torch.float32,
            ).to(device=self.device, dtype=self.torch_dtype)

        latents_video = random_latents(
            (1, int(self.vae.model.z_dim), latent_frames, latent_h, latent_w)
        )
        latents_dino = random_latents(
            (1, self.dino_dim, latent_frames, int(first_dino.shape[3]), 1)
        )
        latents_pointmap = random_latents(tuple(latents_video.shape))
        latents_action = random_latents(
            (1, action_horizon, int(self.action_expert.action_dim))
        )
        latents_video[:, :, :1] = first_video
        latents_dino[:, :, :1] = first_dino
        latents_pointmap[:, :, :1] = first_pointmap

        schedulers = (
            self.infer_video_scheduler,
            self.infer_dino_scheduler,
            self.infer_pointmap_scheduler,
            self.infer_action_scheduler,
        )
        samples = (latents_video, latents_dino, latents_pointmap, latents_action)
        schedules = [
            scheduler.build_inference_schedule(
                num_inference_steps=num_inference_steps,
                device=self.device,
                dtype=sample.dtype,
                shift_override=sigma_shift,
            )
            for scheduler, sample in zip(schedulers, samples)
        ]
        fuse_flag = bool(
            getattr(self.video_expert, "fuse_vae_embedding_in_latents", False)
        )
        for step in range(num_inference_steps):
            timesteps = [
                schedule[0][step].unsqueeze(0).to(
                    device=self.device, dtype=sample.dtype
                )
                for schedule, sample in zip(schedules, samples)
            ]
            predictions = self._predict_multimodal_noise(
                latents_video=latents_video,
                latents_dino=latents_dino,
                latents_pointmap=latents_pointmap,
                latents_action=latents_action,
                timestep_video=timesteps[0],
                timestep_dino=timesteps[1],
                timestep_pointmap=timesteps[2],
                timestep_action=timesteps[3],
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=fuse_flag,
            )
            latents_video, latents_dino, latents_pointmap, latents_action = (
                scheduler.step(prediction, schedule[1][step], sample)
                for scheduler, prediction, schedule, sample in zip(
                    schedulers, predictions, schedules, samples
                )
            )
            samples = (
                latents_video, latents_dino, latents_pointmap, latents_action
            )
            latents_video[:, :, :1] = first_video
            latents_dino[:, :, :1] = first_dino
            latents_pointmap[:, :, :1] = first_pointmap

        result: dict[str, Any] = {
            "action": latents_action[0].detach().to(
                device="cpu", dtype=torch.float32
            )
        }
        if return_joint_outputs:
            result.update(
                {
                    "video": self._decode_latents(latents_video, tiled=tiled),
                    "dino": latents_dino[0].detach().to(
                        device="cpu", dtype=torch.float32
                    ),
                    "pointmap": self._decode_latents(
                        latents_pointmap, tiled=tiled
                    ),
                }
            )
        return result
