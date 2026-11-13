# Copyright (c) 2026 Xiaomi Corporation

import json

import pytest
import torch
from omegaconf import OmegaConf

from fastwam.datasets.b1k.processor import B1KFastWAMProcessor
from fastwam.datasets.lerobot.transforms.action_state_merger import ConcatLeftAlign


class _ExtractStateAndMarkAction:
    """Minimal transform that makes ordering observable in processor tests."""

    def forward(self, batch):
        batch["state"]["default"] = batch["state"]["default"][..., :23]
        batch["action"]["default"] = batch["action"]["default"] + 2
        batch["action"]["default"][..., 6] = 0
        batch["action"]["default"][..., 14] = -1
        batch["action"]["default"][..., 22] = 1
        return batch

    def backward(self, batch):
        left_gripper = batch["action"]["default"][..., 14].clone()
        right_gripper = batch["action"]["default"][..., 22].clone()
        batch["action"]["default"] = batch["action"]["default"] - 2
        batch["action"]["default"][..., 6] = 0
        batch["action"]["default"][..., 14] = torch.where(
            left_gripper >= 0,
            1.0,
            -1.0,
        )
        batch["action"]["default"][..., 22] = torch.where(
            right_gripper >= 0,
            1.0,
            -1.0,
        )
        return batch


def _shape_meta():
    # Hydra supplies ListConfig values; this also exercises the inherited
    # processor's shape comparison under its real instantiation contract.
    return OmegaConf.create({
        "images": [{"key": "head", "raw_shape": [3, 2, 2], "shape": [3, 2, 2]}],
        "action": [{"key": "default", "raw_shape": 23, "shape": 23}],
        "state": [{"key": "default", "raw_shape": 61, "shape": 23}],
    })


def _stats(*, action_mean=2.0, state_mean=10.0):
    return {
        "schema_version": "b1k_2026_norm_stats.v1",
        "split": {"name": "train"},
        "normalization": {"std_epsilon": 1e-8, "clamp": [-5.0, 5.0]},
        "stats": {
            "action": {
                "default": {
                    "global_mean": [action_mean] * 23,
                    "global_std": [2.0] * 23,
                }
            },
            "state": {
                "default": {
                    "global_mean": [state_mean] * 23,
                    "global_std": [4.0] * 23,
                }
            },
        },
    }


def _processor():
    return B1KFastWAMProcessor(
        shape_meta=_shape_meta(),
        num_obs_steps=33,
        num_image_steps=9,
        num_output_cameras=1,
        action_output_dim=23,
        proprio_output_dim=23,
        action_state_transforms=[_ExtractStateAndMarkAction()],
        use_stepwise_action_norm=False,
        norm_default_mode="z-score",
        norm_exception_mode=None,
        action_state_merger=ConcatLeftAlign(),
        train_transforms=[],
        val_transforms=[],
    )


def test_preprocess_applies_transform_then_mixed_norm_then_merger():
    processor = _processor().train()
    processor.set_normalizer_from_stats(_stats())
    action = torch.zeros(32, 23)
    action[:, 6] = 99
    action[:, 14] = 1
    action[:, 22] = -1
    state = torch.full((33, 61), 14.0)
    data = {
        "task": "turning on radio",
        "images": {"head": torch.zeros(9, 3, 2, 2)},
        "image_is_pad": torch.zeros(9, dtype=torch.bool),
        "action": {"default": action},
        "state": {"default": state},
        "action_is_pad": torch.zeros(32, dtype=torch.bool),
        "state_is_pad": torch.zeros(33, dtype=torch.bool),
        "idx": 7,
    }

    sample = processor.preprocess(data)

    # Transform adds two before z-score: (2 - mean=2) / std=2 == 0.
    assert sample["action"][:, :6].eq(0).all()
    assert sample["action"][:, 7:14].eq(0).all()
    assert sample["action"][:, 15:22].eq(0).all()
    assert sample["action"][:, 6].eq(0).all()
    assert sample["action"][:, 14].eq(-1).all()
    assert sample["action"][:, 22].eq(1).all()
    assert sample["proprio"].eq(1).all()
    assert sample["action"].shape == (32, 23)
    assert sample["proprio"].shape == (33, 23)
    assert sample["pixel_values"].shape == (1, 9, 3, 2, 2)


def test_preprocess_preserves_action_zscore_tail_but_clamps_state():
    processor = _processor().train()
    processor.set_normalizer_from_stats(_stats())
    action = torch.zeros(32, 23)
    action[:, 0] = 20
    action[:, 6] = 0
    action[:, 14] = -1
    action[:, 22] = 1
    state = torch.full((33, 61), 100.0)
    data = {
        "task": "turning on radio",
        "images": {"head": torch.zeros(9, 3, 2, 2)},
        "image_is_pad": torch.zeros(9, dtype=torch.bool),
        "action": {"default": action},
        "state": {"default": state},
        "action_is_pad": torch.zeros(32, dtype=torch.bool),
        "state_is_pad": torch.zeros(33, dtype=torch.bool),
        "idx": 7,
    }

    sample = processor.preprocess(data)

    # The transform adds 2, so action z-score is (22 - 2) / 2 = 10.
    assert sample["action"][:, 0].eq(10).all()
    assert sample["proprio"].eq(5).all()


def test_postprocess_reverses_merger_norm_and_transform_and_enforces_semantics():
    processor = _processor().eval()
    processor.set_normalizer_from_stats(_stats())
    normalized_action = torch.zeros(1, 64, 23)
    normalized_action[..., 6] = 4
    normalized_action[..., 14] = -0.01
    normalized_action[..., 22] = 0
    normalized_state = torch.ones(1, 33, 23)

    result = processor.postprocess(
        {"action": normalized_action, "proprio": normalized_state}
    )

    action = result["action"]["default"]
    assert action.shape == (1, 32, 23)
    assert action[..., :6].eq(0).all()
    assert action[..., 6].eq(0).all()
    assert action[..., 14].eq(-1).all()
    assert action[..., 22].eq(1).all()
    torch.testing.assert_close(result["state"]["default"], torch.full((1, 33, 23), 14.0))


def test_postprocess_preserves_unclamped_action_zscore_tail():
    processor = _processor().eval()
    processor.set_normalizer_from_stats(_stats())
    normalized_action = torch.zeros(1, 64, 23)
    normalized_action[..., 0] = 10
    normalized_action[..., 14] = -1
    normalized_action[..., 22] = 1
    normalized_state = torch.zeros(1, 33, 23)

    result = processor.postprocess(
        {"action": normalized_action, "proprio": normalized_state}
    )

    # Denormalization gives 22, then the inverse transform subtracts 2.
    assert result["action"]["default"][..., 0].eq(20).all()


def test_set_normalizer_loads_train_only_json_and_plain_official_subtree(tmp_path):
    stats_path = tmp_path / "norm_stats.json"
    stats_path.write_text(json.dumps(_stats()), encoding="utf-8")
    processor = _processor()
    processor.set_normalizer_from_stats(stats_path)
    assert processor.normalizer.normalizer.action_mean.shape == (23,)

    processor.set_normalizer_from_stats(_stats()["stats"])
    assert processor.normalizer.normalizer.proprio_std.eq(4).all()

    tensor_loaded = _stats()
    tensor_loaded["normalization"]["clamp"] = torch.tensor([-5.0, 5.0])
    processor.set_normalizer_from_stats(tensor_loaded)
    assert processor.normalizer.normalizer.clamp_value == 5.0


def test_processor_rejects_non_train_stats_and_wrong_temporal_contract():
    processor = _processor()
    validation_stats = _stats()
    validation_stats["split"]["name"] = "val"
    with pytest.raises(ValueError, match="train-only"):
        processor.set_normalizer_from_stats(validation_stats)

    with pytest.raises(ValueError, match="num_obs_steps-1"):
        B1KFastWAMProcessor(
            shape_meta=_shape_meta(),
            num_obs_steps=32,
            num_output_cameras=1,
            action_output_dim=23,
            proprio_output_dim=23,
            action_state_transforms=None,
            use_stepwise_action_norm=False,
            norm_default_mode="z-score",
            norm_exception_mode=None,
            action_state_merger=ConcatLeftAlign(),
            train_transforms=[],
            val_transforms=[],
        )


def test_processor_accepts_long_context_temporal_contract():
    processor = B1KFastWAMProcessor(
        shape_meta=_shape_meta(),
        num_obs_steps=65,
        num_image_steps=17,
        num_output_cameras=1,
        action_output_dim=23,
        proprio_output_dim=23,
        action_state_transforms=None,
        use_stepwise_action_norm=False,
        norm_default_mode="z-score",
        norm_exception_mode=None,
        action_state_merger=ConcatLeftAlign(),
        train_transforms=[],
        val_transforms=[],
    )

    assert processor.num_obs_steps == 65
    assert processor.num_image_steps == 17


def test_processor_rejects_mismatched_or_vae_incompatible_image_steps():
    common = dict(
        shape_meta=_shape_meta(),
        num_output_cameras=1,
        action_output_dim=23,
        proprio_output_dim=23,
        action_state_transforms=None,
        use_stepwise_action_norm=False,
        norm_default_mode="z-score",
        norm_exception_mode=None,
        action_state_merger=ConcatLeftAlign(),
        train_transforms=[],
        val_transforms=[],
    )
    with pytest.raises(ValueError, match="expected 17, got 9"):
        B1KFastWAMProcessor(
            num_obs_steps=65,
            num_image_steps=9,
            **common,
        )
    with pytest.raises(ValueError, match="VAE-compatible"):
        B1KFastWAMProcessor(
            num_obs_steps=9,
            num_image_steps=3,
            **common,
        )
