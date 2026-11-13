# Copyright (c) 2026 Xiaomi Corporation

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import torch

from .fastwam_multimodal_base import FastWAMMultimodalBase


class FastWAMFlex(FastWAMMultimodalBase):
    """Flex-pi style routing on top of the fixed four-stream FastWAM model.

    This variant adds configurable routing to the shared multimodal backbone.
    """

    def __init__(
        self,
        *args,
        hbridge_enabled: bool = True,
        hbridge_bottom_ratio: float = 0.25,
        hbridge_top_ratio: float = 0.25,
        present_probabilities: Sequence[float] = (1.0, 1.0, 1.0),
        output_probabilities: Sequence[float] = (1.0, 1.0, 1.0),
        routing_sampling_scope: str = "fixed_all_modalities",
        cross_modal_predict: Sequence[bool] = (True, True, True),
        inference_present_modalities: Sequence[bool] = (True, True, True),
        inference_output_modalities: Sequence[bool] = (True, True, True),
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.hbridge_enabled = bool(hbridge_enabled)
        self.hbridge_bottom_ratio = float(hbridge_bottom_ratio)
        self.hbridge_top_ratio = float(hbridge_top_ratio)
        self.present_probabilities = self._validate_probabilities(present_probabilities)
        self.output_probabilities = self._validate_probabilities(output_probabilities)
        self.routing_sampling_scope = self._validate_routing_sampling_scope(
            routing_sampling_scope
        )
        self.cross_modal_predict = tuple(bool(x) for x in cross_modal_predict)
        self.inference_present_modalities = tuple(bool(x) for x in inference_present_modalities)
        self.inference_output_modalities = tuple(bool(x) for x in inference_output_modalities)
        if len(self.cross_modal_predict) != 3 or len(self.inference_present_modalities) != 3 or len(self.inference_output_modalities) != 3:
            raise ValueError("FastWAM Flex modality settings must contain three RGB/DINO/pointmap values.")

    @classmethod
    def from_wan22_pretrained(cls, **kwargs):
        names = {
            "hbridge_enabled", "hbridge_bottom_ratio", "hbridge_top_ratio",
            "present_probabilities", "output_probabilities", "cross_modal_predict",
            "routing_sampling_scope",
            "inference_present_modalities", "inference_output_modalities",
        }
        flex = {key: kwargs.pop(key) for key in list(kwargs) if key in names}
        model_init_kwargs = dict(kwargs.pop("model_init_kwargs", {}))
        model_init_kwargs.update(flex)
        kwargs["model_init_kwargs"] = model_init_kwargs
        return super().from_wan22_pretrained(**kwargs)

    @staticmethod
    def _validate_probabilities(values: Sequence[float]) -> tuple[float, float, float]:
        result = tuple(float(x) for x in values)
        if len(result) != 3 or any(x < 0 or x > 1 for x in result):
            raise ValueError("FastWAM Flex modality probabilities must contain three values in [0, 1].")
        return result  # type: ignore[return-value]

    @staticmethod
    def _validate_routing_sampling_scope(value: str) -> str:
        value = str(value)
        allowed = {"per_sample", "per_batch", "fixed_all_modalities"}
        if value not in allowed:
            raise ValueError(
                "routing_sampling_scope must be one of "
                f"{sorted(allowed)}, got {value!r}."
            )
        return value

    def _flex_metadata(self) -> dict[str, Any]:
        return {
            "hbridge_enabled": self.hbridge_enabled,
            "hbridge_bottom_ratio": self.hbridge_bottom_ratio,
            "hbridge_top_ratio": self.hbridge_top_ratio,
            "present_probabilities": self.present_probabilities,
            "output_probabilities": self.output_probabilities,
            "routing_sampling_scope": self.routing_sampling_scope,
            "cross_modal_predict": self.cross_modal_predict,
            "inference_present_modalities": self.inference_present_modalities,
            "inference_output_modalities": self.inference_output_modalities,
        }

    def _checkpoint_extra_payload(self) -> dict[str, Any]:
        payload = super()._checkpoint_extra_payload()
        payload["fastwam_flex"] = self._flex_metadata()
        return payload

    def load_checkpoint(self, path, optimizer=None, **kwargs):
        payload = super().load_checkpoint(path, optimizer=optimizer, **kwargs)
        saved = payload.get("fastwam_flex")
        expected = self._flex_metadata()
        if not isinstance(saved, Mapping):
            raise ValueError(
                "FastWAM Flex checkpoint is missing `fastwam_flex` routing metadata."
            )
        saved = dict(saved)
        # Flex checkpoints written before sampling scopes existed used the
        # current per-sample behavior implicitly.
        saved.setdefault("routing_sampling_scope", "per_sample")
        if saved != expected:
            raise RuntimeError(
                "FastWAM Flex checkpoint routing metadata does not match the model config: "
                f"checkpoint={saved}, model={expected}."
            )
        return payload

    @staticmethod
    def sample_modality_masks(
        batch_size: int,
        present_probability: Sequence[float],
        output_probability: Sequence[float],
        device: torch.device,
        generator: Optional[torch.Generator] = None,
        sampling_scope: str = "per_sample",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        sampling_scope = FastWAMFlex._validate_routing_sampling_scope(sampling_scope)
        if sampling_scope == "fixed_all_modalities":
            fixed = torch.ones((batch_size, 3), dtype=torch.bool, device=device)
            return fixed, fixed.clone()
        sample_count = 1 if sampling_scope == "per_batch" else batch_size
        probabilities = torch.tensor(tuple(present_probability), device=device)
        if probabilities.shape != (3,) or not bool((probabilities > 0).any()):
            raise ValueError("At least one present_probability must be greater than zero.")
        present = torch.rand((sample_count, 3), device=device, generator=generator) < torch.tensor(
            tuple(present_probability), device=device
        )
        # The all-missing input is rejected by the Flex-pi sampler.
        missing = ~present.any(dim=1)
        if missing.any():
            candidates = (probabilities > 0).nonzero().flatten()
            choice = torch.randint(
                0, candidates.numel(), (int(missing.sum()),), device=device, generator=generator
            )
            replacement = candidates[choice]
            present[missing] = False
            present[missing, replacement] = True
        output = torch.rand((sample_count, 3), device=device, generator=generator) < torch.tensor(
            tuple(output_probability), device=device
        )
        if sampling_scope == "per_batch":
            present = present.expand(batch_size, -1).clone()
            output = output.expand(batch_size, -1).clone()
        return present, output

    @staticmethod
    def build_flex_attention_masks(
        stream_lengths: tuple[int, int, int],
        anchor_lengths: tuple[int, int, int],
        action_seq_len: int,
        num_layers: int,
        bottom_local_layers: int,
        top_local_layers: int,
        present_modalities: torch.Tensor,
        output_modalities: torch.Tensor,
        cross_modal_predict: Sequence[bool],
        device: torch.device,
    ) -> list[torch.Tensor]:
        if num_layers <= 0 or bottom_local_layers < 0 or top_local_layers < 0 or bottom_local_layers + top_local_layers > num_layers:
            raise ValueError("Invalid HBridge layer partition.")
        if len(stream_lengths) != 3 or len(anchor_lengths) != 3 or any(a <= 0 or a > s for a, s in zip(anchor_lengths, stream_lengths)):
            raise ValueError("Invalid FastWAM Flex stream or anchor lengths.")
        if present_modalities.ndim == 1:
            present_modalities = present_modalities.unsqueeze(0)
        if output_modalities.ndim == 1:
            output_modalities = output_modalities.unsqueeze(0)
        if present_modalities.shape != output_modalities.shape or present_modalities.shape[1] != 3:
            raise ValueError("present_modalities and output_modalities must be [B,3].")
        bsz = present_modalities.shape[0]
        present_modalities = present_modalities.to(device=device, dtype=torch.bool)
        output_modalities = output_modalities.to(device=device, dtype=torch.bool)
        cross = tuple(bool(x) for x in cross_modal_predict)
        if len(cross) != 3:
            raise ValueError("cross_modal_predict must contain three values.")
        offsets = []
        start = 0
        for length in stream_lengths:
            offsets.append((start, start + length))
            start += length
        action_begin = start
        total = start + action_seq_len
        masks: list[torch.Tensor] = []

        def local_mask() -> torch.Tensor:
            mask = torch.zeros((bsz, total, total), dtype=torch.bool, device=device)
            for sample in range(bsz):
                for i, ((begin, end), anchor) in enumerate(zip(offsets, anchor_lengths)):
                    if bool(present_modalities[sample, i]) or cross[i]:
                        future_begin = begin + anchor
                        mask[sample, future_begin:end, future_begin:end] = True
                        if bool(present_modalities[sample, i]):
                            mask[sample, begin:begin + anchor, begin:begin + anchor] = True
                            mask[sample, future_begin:end, begin:begin + anchor] = True
            mask[:, action_begin:, action_begin:] = True
            return mask

        middle = torch.zeros((bsz, total, total), dtype=torch.bool, device=device)
        for sample in range(bsz):
            active = present_modalities[sample].clone()
            for i, (begin, end) in enumerate(offsets):
                if not bool(active[i]) and not cross[i]:
                    middle[sample, begin:end, :] = False
                    middle[sample, :, begin:end] = False
                elif not bool(active[i]):
                    middle[sample, begin:begin + anchor_lengths[i], :] = False
                    middle[sample, :, begin:begin + anchor_lengths[i]] = False
            # Anchors: present anchors can be read by all active future/action queries.
            for i, (begin, end) in enumerate(offsets):
                if bool(active[i]):
                    middle[sample, begin:begin + anchor_lengths[i], :].zero_()
                    for j, (kb, _) in enumerate(offsets):
                        if bool(active[j]):
                            middle[sample, begin:begin + anchor_lengths[i], kb:kb + anchor_lengths[j]] = True
            for i, (begin, end) in enumerate(offsets):
                if not bool(active[i]) and not cross[i]:
                    continue
                future = slice(begin + anchor_lengths[i], end)
                # Every future can read present anchors.
                for j, (kb, _) in enumerate(offsets):
                    if bool(active[j]):
                        middle[sample, future, kb:kb + anchor_lengths[j]] = True
                # Futures with equal m_out values are mutually visible.
                for j, (kb, ke) in enumerate(offsets):
                    if not bool(active[j]) and not cross[j]:
                        continue
                    if bool(output_modalities[sample, i] == output_modalities[sample, j]):
                        middle[sample, future, kb + anchor_lengths[j]:ke] = True
                middle[sample, future, future] = True
                if bool(output_modalities[sample, i]):
                    middle[sample, future, action_begin:] = False
            # Action sees active anchors and selected futures, plus itself.
            middle[sample, action_begin:, action_begin:] = True
            for j, (kb, ke) in enumerate(offsets):
                if bool(active[j]):
                    middle[sample, action_begin:, kb:kb + anchor_lengths[j]] = True
                if bool(output_modalities[sample, j]) and (bool(active[j]) or cross[j]):
                    middle[sample, action_begin:, kb + anchor_lengths[j]:ke] = True
        local = local_mask()
        for layer_idx in range(num_layers):
            masks.append(local if layer_idx < bottom_local_layers or layer_idx >= num_layers - top_local_layers else middle)
        return masks

    # The old name remains available through the compatibility module, while
    # new callers can use the canonical FastWAM Flex terminology.
    # Checkpoint-era compatibility for callers of the former product name.

    def _prepare_attention_routing(self, batch_size: int, device: torch.device, *, training: bool) -> Any:
        if training:
            present, output = self.sample_modality_masks(
                batch_size,
                self.present_probabilities,
                self.output_probabilities,
                device,
                sampling_scope=self.routing_sampling_scope,
            )
            shared = self.routing_sampling_scope != "per_sample"
        else:
            present = torch.tensor(self.inference_present_modalities, device=device).expand(batch_size, -1)
            output = torch.tensor(self.inference_output_modalities, device=device).expand(batch_size, -1)
            shared = True
        return {"present": present, "output": output, "shared": shared}

    def _apply_visual_input_routing(self, noisy_streams, routing) -> None:
        for idx, stream in enumerate(noisy_streams):
            absent = ~routing["present"][:, idx]
            if not absent.any():
                continue
            if self.cross_modal_predict[idx]:
                stream[absent, :, :1] = 0
            else:
                stream[absent] = 0

    def _build_routed_attention_masks(self, stream_lengths, anchor_lengths, action_seq_len, device, routing):
        layers = int(self.mot.num_layers)
        bottom = int(layers * self.hbridge_bottom_ratio) if self.hbridge_enabled else 0
        top = int(layers * self.hbridge_top_ratio) if self.hbridge_enabled else 0
        present = routing["present"][:1] if routing.get("shared", False) else routing["present"]
        output = routing["output"][:1] if routing.get("shared", False) else routing["output"]
        masks = self.build_flex_attention_masks(
            stream_lengths, anchor_lengths, action_seq_len, layers, bottom, top,
            present, output, self.cross_modal_predict, device,
        )
        if routing.get("shared", False):
            masks = [mask.squeeze(0) for mask in masks]
        return masks[0], masks

    def _reduce_routed_loss(self, loss: torch.Tensor, routing: Any, modality_index: int) -> torch.Tensor:
        # Cross-modal forcing keeps future losses active; non-forced absent streams
        # are disabled and excluded from the batch loss.
        if self.cross_modal_predict[modality_index]:
            return loss.mean()
        weight = routing["present"][:, modality_index].to(loss)
        return (loss * weight).sum() / weight.sum().clamp(min=1)

    @torch.no_grad()
    def infer_action(self, *args, **kwargs) -> dict[str, Any]:
        if any(self.inference_output_modalities):
            return super().infer_action(*args, **kwargs)
        if bool(kwargs.get("return_joint_outputs", False)):
            raise ValueError(
                "Action-only FastWAM Flex inference cannot return visual joint outputs."
            )
        return self._infer_action_only(*args, **kwargs)

    @torch.no_grad()
    def _infer_action_only(
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
        del negative_prompt, text_cfg_scale, return_joint_outputs
        self.eval()
        if self.dino_encoder is None:
            raise ValueError("FastWAM Flex inference requires a frozen `dino_encoder`.")

        def validate_image(value: torch.Tensor, name: str) -> torch.Tensor:
            if value.ndim == 3:
                value = value.unsqueeze(0)
            if value.ndim != 4 or value.shape[0] != 1 or value.shape[1] != 3:
                raise ValueError(f"`{name}` must be [1,3,H,W] or [3,H,W].")
            return value

        input_image = validate_image(input_image, "input_image")
        input_pointmap = validate_image(input_pointmap, "input_pointmap")
        if input_image.shape != input_pointmap.shape:
            raise ValueError("`input_image` and `input_pointmap` must have equal shapes.")
        _, _, height, width = input_image.shape
        checked_h, checked_w, checked_t = self._check_resize_height_width(
            height, width, num_video_frames
        )
        if (checked_h, checked_w, checked_t) != (height, width, num_video_frames):
            raise ValueError("FastWAM Flex inference inputs must satisfy the Wan spatial/temporal grid.")
        if action_horizon <= 0 or num_inference_steps <= 0:
            raise ValueError("action_horizon and num_inference_steps must be positive.")

        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but proprio encoding is disabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            if proprio.shape != (1, self.proprio_dim):
                raise ValueError(f"`proprio` must have shape [1,{self.proprio_dim}].")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        if prompt is not None and (context is not None or context_mask is not None):
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if prompt is not None:
            context, context_mask = self.encode_prompt(prompt)
        elif context is None or context_mask is None:
            raise ValueError("Either `prompt` or both `context/context_mask` are required.")
        else:
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2 or context.shape[0] != 1:
                raise ValueError("FastWAM Flex context must be [1,L,D] with mask [1,L].")
            context = context.to(device=self.device, dtype=self.torch_dtype)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context, context_mask=context_mask, proprio=proprio
            )

        image = input_image.to(device=self.device, dtype=self.torch_dtype)
        pointmap = input_pointmap.to(device=self.device, dtype=self.torch_dtype)
        first_video = self._encode_input_image_latents_tensor(image, tiled=tiled)
        first_pointmap = self._encode_input_image_latents_tensor(pointmap, tiled=tiled)
        first_dino = self.dino_encoder.encode_video(
            image.unsqueeze(2),
            temporal_downsample_factor=int(self.vae.temporal_downsample_factor),
        ).to(device=self.device, dtype=self.torch_dtype)
        zero_t = torch.zeros((1,), device=self.device, dtype=self.torch_dtype)
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))
        video_pre = self.video_expert.pre_dit(
            x=first_video, timestep=zero_t, context=context, context_mask=context_mask,
            action=None, fuse_vae_embedding_in_latents=fuse_flag,
        )
        dino_tokens = self._embed_dino(first_dino)
        dino_frames, dino_tpf = int(first_dino.shape[2]), int(first_dino.shape[3])
        _, dino_t_mod = self._build_stream_tensors(zero_t, dino_frames, dino_tpf)
        dino_freqs = self._dino_freqs(dino_frames, dino_tpf, dino_tokens.device)
        point_tokens, point_grid = self._embed_pointmap(first_pointmap)
        point_frames, point_h, point_w = point_grid
        point_tpf = point_h * point_w
        _, point_t_mod = self._build_stream_tensors(zero_t, point_frames, point_tpf)
        point_freqs = self._grid_freqs(point_frames, point_h, point_w, point_tokens.device)
        visual_tokens = torch.cat([video_pre["tokens"], dino_tokens, point_tokens], dim=1)
        visual_freqs = torch.cat([video_pre["freqs"], dino_freqs, point_freqs], dim=0)
        visual_t_mod = torch.cat([video_pre["t_mod"], dino_t_mod, point_t_mod], dim=1)
        visual_context_mask = video_pre["context_mask"]
        if visual_context_mask.ndim == 3:
            auxiliary = visual_context_mask[:, :1].expand(
                -1, dino_tokens.shape[1] + point_tokens.shape[1], -1
            )
            visual_context_mask = torch.cat([visual_context_mask, auxiliary], dim=1)

        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_action = torch.randn(
            (1, action_horizon, int(self.action_expert.action_dim)),
            generator=generator, device=rand_device, dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        routing = {
            "present": torch.tensor(self.inference_present_modalities, device=self.device).expand(1, -1),
            "output": torch.zeros((1, 3), device=self.device, dtype=torch.bool),
        }
        lengths = (
            int(video_pre["tokens"].shape[1]),
            int(dino_tokens.shape[1]),
            int(point_tokens.shape[1]),
        )
        masks = self._build_routed_attention_masks(
            lengths, lengths, action_horizon, visual_tokens.device, routing
        )[1]
        assert masks is not None
        visual_len = sum(lengths)
        cache = self.mot.prefill_video_cache(
            video_tokens=visual_tokens,
            video_freqs=visual_freqs,
            video_t_mod=visual_t_mod,
            video_context_payload={"context": video_pre["context"], "mask": visual_context_mask},
            video_attention_mask=masks[0][..., :visual_len, :visual_len],
            layer_video_attention_masks=[mask[..., :visual_len, :visual_len] for mask in masks],
        )
        timesteps, deltas = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps, device=self.device,
            dtype=latents_action.dtype, shift_override=sigma_shift,
        )
        for timestep, delta in zip(timesteps, deltas):
            action_pre = self.action_expert.pre_dit(
                action_tokens=latents_action,
                timestep=timestep.unsqueeze(0).to(latents_action),
                context=context,
                context_mask=context_mask,
            )
            action_tokens = self.mot.forward_action_with_video_cache(
                action_tokens=action_pre["tokens"],
                action_freqs=action_pre["freqs"],
                action_t_mod=action_pre["t_mod"],
                action_context_payload={"context": action_pre["context"], "mask": action_pre["context_mask"]},
                video_kv_cache=cache,
                attention_mask=masks[0],
                video_seq_len=visual_len,
                layer_attention_masks=masks,
            )
            prediction = self.action_expert.post_dit(action_tokens, action_pre)
            latents_action = self.infer_action_scheduler.step(prediction, delta, latents_action)
        return {"action": latents_action[0].detach().to(device="cpu", dtype=torch.float32)}
