# Copyright (c) 2026 Xiaomi Corporation

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


def _compose_b1k_config():
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
        return compose(
            config_name="train",
            overrides=["task=b1k_legacy_uncond_3cam_352x224_1e-4"],
        )


def _compose_b1k_direct_video_config():
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
        return compose(
            config_name="train",
            overrides=["task=b1k_legacy_uncond_3cam_direct_video_384x256_1e-4"],
        )


def _compose_b1k_128gpu_candidate_config(*overrides: str):
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
        return compose(
            config_name="train",
            overrides=[
                "task=b1k_2026_uncond_3cam_direct384x256_mb32_gc_off_128gpu",
                *overrides,
            ],
        )


def _compose_b1k_128gpu_joint_config(*overrides: str):
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
        return compose(
            config_name="train",
            overrides=[
                "task=b1k_2026_joint_3cam_direct384x256_mb32_gc_off_128gpu",
                *overrides,
            ],
        )


def _compose_b1k_fastwam_flex_config(*overrides: str):
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
        return compose(
            config_name="train",
            overrides=[
                "task=b1k_2026_fastwam_flex_3cam_direct384x256_mb32_gc_on_128gpu",
                *overrides,
            ],
        )


def test_b1k_data_contract_resolves() -> None:
    cfg = _compose_b1k_config()
    data = OmegaConf.to_container(cfg.data, resolve=True)

    assert data["train"]["_target_"] == "fastwam.datasets.b1k.dataset.B1KRobotVideoDataset"
    assert data["train"]["split"] == "train"
    assert data["val"]["split"] == "validation"
    assert data["train"]["dataset_root"].endswith("2026-challenge-demos-224-rgb-jpeg")

    train = data["train"]
    assert train["num_frames"] == 33
    assert train["global_sample_stride"] == 2
    assert train["window_start_stride"] == 6
    assert train["action_video_freq_ratio"] == 4
    assert train["video_size"] == [352, 224]
    assert [image["key"] for image in train["shape_meta"]["images"]] == [
        "rgb.head",
        "rgb.left_wrist",
        "rgb.right_wrist",
    ]
    assert train["shape_meta"]["action"] == [
        {"key": "default", "raw_shape": 23, "shape": 23}
    ]
    assert train["shape_meta"]["state"] == [
        {"key": "default", "raw_shape": 61, "shape": 23}
    ]

    processor = train["processor"]
    assert processor["_target_"] == "fastwam.datasets.b1k.processor.B1KFastWAMProcessor"
    assert processor["num_obs_steps"] == 33
    assert processor["num_image_steps"] == 9
    assert processor["num_output_cameras"] == 3
    assert processor["action_output_dim"] == 23
    assert processor["proprio_output_dim"] == 23
    assert processor["use_stepwise_action_norm"] is False
    assert processor["norm_default_mode"] == "z-score"
    assert train["context_len"] == 128


def test_b1k_uncond_model_and_optimizer_contract_resolves() -> None:
    cfg = _compose_b1k_config()

    assert cfg.model._target_ == "fastwam.runtime.create_fastwam"
    assert cfg.model.model_id == "Wan-AI/Wan2.2-TI2V-5B"
    assert cfg.model.tokenizer_max_len == 128
    assert cfg.model.proprio_dim == 23
    assert cfg.model.action_dit_pretrained_path.endswith(
        "ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt"
    )
    assert cfg.model.skip_dit_load_from_pretrain is False
    assert cfg.model.video_dit_config.action_conditioned is False
    assert cfg.model.video_dit_config.action_dim == 23
    assert cfg.model.action_dit_config.action_dim == 23
    assert cfg.model.loss.lambda_video == 1.0
    assert cfg.model.loss.lambda_action == 1.0

    assert cfg.learning_rate == 1.0e-4
    assert cfg.lr_scheduler_type == "cosine"
    assert cfg.weight_decay == 1.0e-2
    assert cfg.mixed_precision == "bf16"
    assert cfg.max_grad_norm == 1.0
    assert cfg.batch_size == 1
    assert cfg.num_workers == 8
    assert cfg.prefetch_factor == 2
    assert cfg.persistent_workers is False
    assert cfg.max_steps is None
    assert cfg.resume is None


def test_b1k_direct_video_config_preserves_native_inputs_and_384x256_canvas() -> None:
    cfg = _compose_b1k_direct_video_config()
    train = OmegaConf.to_container(cfg.data.train, resolve=True)

    assert train["_target_"] == "fastwam.datasets.b1k.video_dataset.B1KDirectVideoDataset"
    assert train["dataset_root"].endswith("b1k_2026/v3_source")
    assert train["numeric_dataset_root"] is None
    assert cfg.data.val.numeric_dataset_root is None
    assert train["video_size"] == [384, 256]
    assert train["window_start_stride"] == 6
    assert [item["raw_shape"] for item in train["shape_meta"]["images"]] == [
        [3, 720, 720],
        [3, 480, 480],
        [3, 480, 480],
    ]
    assert [item["shape"] for item in train["shape_meta"]["images"]] == [
        [3, 720, 720],
        [3, 480, 480],
        [3, 480, 480],
    ]
    assert train["processor"]["train_transforms"] == []
    assert train["parquet_row_group_cache_size"] == 0


def test_b1k_128gpu_candidate_config_resolves_locked_training_contract() -> None:
    cfg = _compose_b1k_128gpu_candidate_config()

    assert cfg.data.train._target_ == "fastwam.datasets.b1k.video_dataset.B1KDirectVideoDataset"
    assert cfg.data.train.video_size == [384, 256]
    assert cfg.data.train.window_start_stride == 6
    assert cfg.data.train.numeric_dataset_root.endswith("b1k_2026/numeric_v3")
    assert cfg.data.val.numeric_dataset_root == cfg.data.train.numeric_dataset_root
    assert cfg.batch_size == 32
    assert cfg.gradient_accumulation_steps == 1
    assert cfg.model.mot_checkpoint_mixed_attn is False
    assert cfg.learning_rate == 2.0e-4
    assert cfg.min_learning_rate == 1.0e-6
    assert cfg.num_epochs == 10
    assert cfg.save_every == 1000
    assert cfg.checkpoint_retention.max_to_keep == 2
    assert cfg.checkpoint_retention.keep_every_n_steps == 10000
    assert cfg.checkpoint_retention.keep_final is True
    assert cfg.checkpoint_retention.permanent_weights_every_n_steps is None
    assert cfg.max_steps is None
    assert cfg.num_workers == 8
    assert cfg.prefetch_factor == 1
    assert cfg.persistent_workers is True
    assert cfg.mixed_precision == "bf16"
    assert cfg.max_grad_norm == 1.0
    assert cfg.weight_decay == 1.0e-2
    assert cfg.eval_every == 0
    assert cfg.fixed_validation.enabled is True
    assert list(cfg.fixed_validation.steps) == []
    assert cfg.fixed_validation.every_steps == 8501
    assert cfg.fixed_validation.tasks == 100
    assert cfg.fixed_validation.anchors_per_task == 10
    assert cfg.fixed_validation.max_duration_seconds == 900
    assert cfg.qualitative_validation.enabled is True
    assert list(cfg.qualitative_validation.steps) == []
    assert cfg.qualitative_validation.every_steps == 8501
    assert cfg.qualitative_validation.num_windows == 128
    assert cfg.qualitative_validation.expected_tasks == 100
    assert cfg.qualitative_validation.videos == 8
    assert cfg.qualitative_validation.inference_steps == 10
    assert cfg.qualitative_validation.seed == 42
    assert cfg.qualitative_validation.fps == 8
    assert cfg.qualitative_validation.max_duration_seconds == 900
    assert cfg.model.loss.lambda_video == 1.0
    assert cfg.model.loss.lambda_action == 1.0


def test_b1k_128gpu_joint_changes_only_model_target() -> None:
    uncond = OmegaConf.to_container(
        _compose_b1k_128gpu_candidate_config(), resolve=True
    )
    joint = OmegaConf.to_container(_compose_b1k_128gpu_joint_config(), resolve=True)

    assert uncond["model"].pop("_target_") == "fastwam.runtime.create_fastwam"
    assert joint["model"].pop("_target_") == "fastwam.runtime.create_fastwam_joint"
    uncond.pop("output_dir")
    joint.pop("output_dir")
    assert joint == uncond


def test_b1k_128gpu_candidate_learning_rate_can_be_overridden() -> None:
    cfg = _compose_b1k_128gpu_candidate_config("learning_rate=5e-5")

    assert cfg.learning_rate == 5.0e-5


def test_b1k_fastwam_flex_resolves_canonical_multimodal_contract() -> None:
    cfg = _compose_b1k_fastwam_flex_config()

    assert cfg.model._target_ == "fastwam.runtime.create_fastwam_flex"
    assert cfg.model.dino_model_name == "vit_base_patch16_dinov3.lvd1689m"
    assert cfg.model.dino_dim == 3072
    assert cfg.model.dino_output_grid_size == 6
    assert cfg.model.dino_pretrained_path.endswith(
        "models/dinov3/vit_base_patch16_dinov3.lvd1689m/model.safetensors"
    )
    assert cfg.model.dino_pred_x0 is True
    assert cfg.model.hbridge_enabled is True
    assert cfg.model.hbridge_bottom_ratio == 0.25
    assert cfg.model.hbridge_top_ratio == 0.25
    assert dict(cfg.model.present_probabilities) == {
        "rgb": 1.0,
        "dino": 1.0,
        "pointmap": 1.0,
    }
    assert dict(cfg.model.output_probabilities) == {
        "rgb": 1.0,
        "dino": 1.0,
        "pointmap": 1.0,
    }
    assert cfg.model.routing_sampling_scope == "fixed_all_modalities"
    assert all(cfg.model.cross_modal_predict.values())

    split = cfg.data.train
    assert split.dataset_root.endswith(
        "b1k_2026/v3_source"
    )
    assert split.numeric_dataset_root.endswith(
        "b1k_2026/numeric_v3"
    )
    assert split.pretrained_norm_stats.endswith(
        "b1k_2026/norm_stats.json"
    )
    assert split.text_embedding_cache_dir.endswith(
        "b1k_2026/text_cache"
    )
    assert split.train_all_episodes is True
    assert cfg.data.val is None
    assert split.num_frames == 33
    assert split.processor.num_obs_steps == 33
    assert split.processor.num_image_steps == 9
    assert split.processor.num_obs_steps - 1 == 32
    assert (split.processor.num_image_steps - 1) // 4 + 1 == 3
    assert split.include_depth is False
    assert split.return_pointmap is True

    assert cfg.batch_size == 32
    assert cfg.num_epochs == 30
    assert cfg.max_steps == 316950
    assert cfg.lr_warmup_steps == 3169
    assert cfg.model.mot_checkpoint_mixed_attn is True
    assert cfg.model.model_id == "Wan-AI/Wan2.2-TI2V-5B"
    assert cfg.model.skip_dit_load_from_pretrain is False
    assert cfg.model.action_dit_pretrained_path.endswith(
        "ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt"
    )
    for scheduler_name in ("video_scheduler", "dino_scheduler", "pointmap_scheduler"):
        scheduler = cfg.model[scheduler_name]
        assert scheduler.train_shift == 5.0
        assert scheduler.infer_shift == 5.0
    assert cfg.model.action_scheduler.train_shift == 1.0
    assert cfg.model.action_scheduler.infer_shift == 1.0
    assert cfg.resume is None
    assert cfg.legacy_warmstart is None
    assert cfg.flex_warmstart is None
    assert cfg.checkpoint_retention.max_to_keep == 2
    assert cfg.checkpoint_retention.keep_every_n_steps is None
    assert cfg.checkpoint_retention.keep_final is True
    assert cfg.save_every == 250
    assert cfg.checkpoint_retention.permanent_weights_every_n_steps == 10565
    assert cfg.eval_every == 0
    assert cfg.fixed_validation.enabled is False
    assert list(cfg.fixed_validation.steps) == []
    assert cfg.fixed_validation.every_steps == 0
    assert cfg.qualitative_validation.enabled is False
    assert list(cfg.qualitative_validation.steps) == []
    assert cfg.qualitative_validation.every_steps == 0
    assert cfg.wandb.name.startswith("b1k_2026_fastwam_flex_")
