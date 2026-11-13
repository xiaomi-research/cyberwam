# Copyright (c) 2026 Xiaomi Corporation

import types
import sys

import torch
import torch.nn as nn

from fastwam.models.wan22.dino_encoder import B1KDinoEncoder
from fastwam.models.wan22.fastwam_multimodal_base import FastWAMMultimodalBase


class _PatchEmbed:
    patch_size = (16, 16)


class _FakeDino(nn.Module):
    embed_dim = 8
    num_prefix_tokens = 1
    patch_embed = _PatchEmbed()

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward_features(self, x):
        batch = x.shape[0]
        values = torch.arange(197 * 8, device=x.device, dtype=x.dtype)
        return values.reshape(1, 197, 8).expand(batch, -1, -1)


def test_b1k_dino_encoder_outputs_three_lossless_folded_camera_grids():
    encoder = B1KDinoEncoder(model=_FakeDino(), image_size=224, pixel_unshuffle=2)
    video = torch.zeros(2, 3, 9, 384, 256)
    output = encoder.encode_video(video, temporal_downsample_factor=4)

    assert output.shape == (2, 32, 3, 147, 1)
    assert encoder.tokens_per_camera == 49
    assert encoder.tokens_per_frame == 147
    assert not any(parameter.requires_grad for parameter in encoder.parameters())


def test_b1k_dino_encoder_can_pool_each_camera_grid_to_25_tokens():
    encoder = B1KDinoEncoder(
        model=_FakeDino(), image_size=224, pixel_unshuffle=2, output_grid_size=5
    )
    video = torch.zeros(2, 3, 9, 384, 256)
    output = encoder.encode_video(video, temporal_downsample_factor=4)

    assert output.shape == (2, 32, 3, 75, 1)
    assert encoder.output_dim == 32
    assert encoder.tokens_per_camera == 25
    assert encoder.tokens_per_frame == 75


def test_dino_encoder_loads_wrapped_local_torch_checkpoint(tmp_path):
    checkpoint = tmp_path / "dino.pt"
    torch.save({"state_dict": {"module.anchor": torch.tensor(3.5)}}, checkpoint)

    model = _FakeDino()
    B1KDinoEncoder._load_local_checkpoint(model, checkpoint)
    assert model.anchor.item() == 3.5


def test_dino_encoder_loads_safetensors_directory(tmp_path):
    from safetensors.torch import save_file

    save_file({"anchor": torch.tensor(4.5)}, str(tmp_path / "model.safetensors"))
    model = _FakeDino()
    B1KDinoEncoder._load_local_checkpoint(model, tmp_path)
    assert model.anchor.item() == 4.5


def test_dino_encoder_local_path_disables_remote_pretrained(monkeypatch, tmp_path):
    calls = []

    def create_model(name, pretrained, num_classes):
        calls.append((name, pretrained, num_classes))
        return _FakeDino()

    monkeypatch.setitem(sys.modules, "timm", types.SimpleNamespace(create_model=create_model))
    torch.save({"anchor": torch.tensor(2.0)}, tmp_path / "weights.pt")
    B1KDinoEncoder(pretrained_path=tmp_path / "weights.pt")
    assert calls == [("vit_base_patch16_dinov3.lvd1689m", False, 0)]


def test_multimodal_attention_directions_are_fixed_full_joint():
    # Layout: RGB(2 anchor + 2 future), DINO(1+2), pointmap(2+1), action(2).
    lengths = (4, 3, 3)
    anchors = (2, 1, 2)
    mask = FastWAMMultimodalBase.build_multimodal_attention_mask(lengths, anchors, 2, torch.device("cpu"))
    visual = sum(lengths)

    assert mask.shape == (12, 12)
    # Every visual anchor sees every visual anchor, but no future or action.
    anchor_rows = (0, 1, 4, 7, 8)
    anchor_cols = set(anchor_rows)
    for row in anchor_rows:
        allowed = set(mask[row].nonzero().flatten().tolist())
        assert allowed == anchor_cols
    # Every future visual token sees all visual streams and never action.
    for row in (2, 3, 5, 6, 9):
        assert mask[row, :visual].all()
        assert not mask[row, visual:].any()
    # Action sees all visual and action tokens.
    assert mask[visual:].all()


def test_multimodal_mask_is_temporal_length_agnostic():
    for visual_frames in (3, 5):
        rgb_tpf, dino_tpf, pointmap_tpf = 96, 147, 96
        lengths = tuple(visual_frames * n for n in (rgb_tpf, dino_tpf, pointmap_tpf))
        mask = FastWAMMultimodalBase.build_multimodal_attention_mask(
            lengths, (rgb_tpf, dino_tpf, pointmap_tpf), 32, torch.device("cpu")
        )
        assert mask.shape == (sum(lengths) + 32,) * 2
        assert mask[-1].all()


def test_x0_velocity_conversion_matches_flow_matching_identity():
    clean = torch.tensor([[[2.0]]])
    noise = torch.tensor([[[6.0]]])
    sigma = 0.25
    noisy = (1 - sigma) * clean + sigma * noise
    velocity = FastWAMMultimodalBase._x0_to_velocity(noisy, clean, torch.tensor([250.0]), 1000)
    torch.testing.assert_close(velocity, noise - clean)


def test_dino_rope_assigns_disjoint_camera_regions():
    holder = object.__new__(FastWAMMultimodalBase)
    nn.Module.__init__(holder)
    holder.video_expert = type("Expert", (), {})()
    holder.video_expert.freqs = (
        torch.arange(8).view(8, 1).to(torch.complex64),
        torch.arange(32).view(32, 1).to(torch.complex64),
        (100 + torch.arange(32)).view(32, 1).to(torch.complex64),
    )

    freqs = holder._dino_freqs(frames=1, tokens_per_frame=147, device=torch.device("cpu"))
    cameras = freqs.squeeze(1).reshape(3, 49, -1)
    assert not torch.equal(cameras[0], cameras[1])
    assert not torch.equal(cameras[0], cameras[2])
    assert not torch.equal(cameras[1], cameras[2])


def test_dino_rope_supports_pooled_five_by_five_camera_grids():
    holder = object.__new__(FastWAMMultimodalBase)
    nn.Module.__init__(holder)
    holder.video_expert = type("Expert", (), {})()
    holder.video_expert.freqs = (
        torch.arange(8).view(8, 1).to(torch.complex64),
        torch.arange(32).view(32, 1).to(torch.complex64),
        (100 + torch.arange(32)).view(32, 1).to(torch.complex64),
    )

    freqs = holder._dino_freqs(frames=3, tokens_per_frame=75, device=torch.device("cpu"))
    assert freqs.shape[0] == 225


def test_dino_loss_excludes_anchor_and_padded_future_frames():
    prediction = torch.tensor([[[99.0], [1.0], [3.0]]])
    target = torch.zeros_like(prediction)
    # Raw frames: anchor, one real latent group, one fully padded latent group.
    image_is_pad = torch.tensor([[False, False, False, False, False, True, True, True, True]])

    loss = FastWAMMultimodalBase._dino_loss_per_sample(
        prediction,
        target,
        frames=3,
        tokens_per_frame=1,
        image_is_pad=image_is_pad,
        temporal_factor=4,
    )

    torch.testing.assert_close(loss, torch.tensor([1.0]))


class _InferenceDino(nn.Module):
    def encode_video(self, video, temporal_downsample_factor):
        assert video.shape == (1, 3, 1, 32, 32)
        assert temporal_downsample_factor == 4
        return torch.full((1, 8, 1, 3, 1), 0.25, device=video.device)


class _InferenceVae:
    temporal_downsample_factor = 4
    model = types.SimpleNamespace(z_dim=2)

    def eval(self):
        return self


class _InferenceScheduler:
    def __init__(self, marker):
        self.marker = marker
        self.calls = []

    def build_inference_schedule(
        self, num_inference_steps, device, dtype, shift_override
    ):
        self.calls.append((num_inference_steps, shift_override))
        timesteps = torch.full((num_inference_steps,), self.marker, device=device, dtype=dtype)
        deltas = torch.full((num_inference_steps,), -0.1, device=device, dtype=dtype)
        return timesteps, deltas

    @staticmethod
    def step(prediction, delta, sample):
        return sample + prediction * delta


def _make_mock_inference_model():
    model = object.__new__(FastWAMMultimodalBase)
    nn.Module.__init__(model)
    model.device = torch.device("cpu")
    model.torch_dtype = torch.float32
    model.dino_dim = 8
    model.dino_encoder = _InferenceDino()
    model.proprio_dim = None
    model.vae = _InferenceVae()
    model.video_expert = types.SimpleNamespace(fuse_vae_embedding_in_latents=False)
    model.action_expert = types.SimpleNamespace(action_dim=23)
    schedulers = [_InferenceScheduler(index) for index in range(1, 5)]
    (
        model.infer_video_scheduler,
        model.infer_dino_scheduler,
        model.infer_pointmap_scheduler,
        model.infer_action_scheduler,
    ) = schedulers

    def encode_current(self, input_image, tiled):
        del tiled
        return input_image.new_full((1, 2, 1, 2, 2), float(input_image.mean()))

    calls = []

    def predict(self, **kwargs):
        calls.append(
            {
                "shapes": tuple(
                    kwargs[name].shape
                    for name in (
                        "latents_video", "latents_dino",
                        "latents_pointmap", "latents_action",
                    )
                ),
                "timesteps": tuple(
                    float(kwargs[name].item())
                    for name in (
                        "timestep_video", "timestep_dino",
                        "timestep_pointmap", "timestep_action",
                    )
                ),
                "future_video": kwargs["latents_video"][:, :, 1:].clone(),
                "future_pointmap": kwargs["latents_pointmap"][:, :, 1:].clone(),
            }
        )
        return tuple(
            torch.ones_like(kwargs[name])
            for name in (
                "latents_video", "latents_dino",
                "latents_pointmap", "latents_action",
            )
        )

    def decode(self, latents, tiled):
        del tiled
        return latents.detach().clone()

    model._encode_input_image_latents_tensor = types.MethodType(encode_current, model)
    model._predict_multimodal_noise = types.MethodType(predict, model)
    model._decode_latents = types.MethodType(decode, model)
    return model, schedulers, calls


def test_multimodal_infer_action_jointly_samples_three_or_five_latent_frames():
    for num_video_frames, latent_frames in ((9, 3), (17, 5)):
        model, schedulers, calls = _make_mock_inference_model()
        result = model.infer_action(
            prompt=None,
            input_image=torch.full((3, 32, 32), 0.5),
            input_pointmap=torch.full((3, 32, 32), -0.5),
            action_horizon=32,
            num_video_frames=num_video_frames,
            context=torch.zeros(2, 4),
            context_mask=torch.ones(2, dtype=torch.bool),
            num_inference_steps=2,
            sigma_shift=7.0,
            seed=3,
            return_joint_outputs=True,
        )

        assert result["action"].shape == (32, 23)
        assert result["video"].shape == (1, 2, latent_frames, 2, 2)
        assert result["dino"].shape == (8, latent_frames, 3, 1)
        assert result["pointmap"].shape == (1, 2, latent_frames, 2, 2)
        assert len(calls) == 2
        assert calls[0]["shapes"] == (
            (1, 2, latent_frames, 2, 2),
            (1, 8, latent_frames, 3, 1),
            (1, 2, latent_frames, 2, 2),
            (1, 32, 23),
        )
        assert calls[0]["timesteps"] == (1.0, 2.0, 3.0, 4.0)
        assert not torch.equal(
            calls[0]["future_video"], calls[0]["future_pointmap"]
        )
        assert all(scheduler.calls == [(2, 7.0)] for scheduler in schedulers)
        torch.testing.assert_close(result["video"][:, :, 0], torch.full((1, 2, 2, 2), 0.5))
        torch.testing.assert_close(result["dino"][:, 0], torch.full((8, 3, 1), 0.25))
        torch.testing.assert_close(
            result["pointmap"][:, :, 0], torch.full((1, 2, 2, 2), -0.5)
        )
