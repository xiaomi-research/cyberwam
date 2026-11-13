# Copyright (c) 2026 Xiaomi Corporation

from __future__ import annotations

import hashlib

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import fastwam.datasets.b1k.checkpoint as checkpoint_module
from fastwam.datasets.b1k.contract import (
    ACTION_CONSTANT_DIM,
    ACTION_DIM,
    ACTION_GRIPPER_DIMS,
    ACTION_LABELS,
    PROPRIO_DIM,
    PROPRIO_LABELS,
)
from fastwam.datasets.b1k.checkpoint import (
    B1KCheckpointContractError,
    bind_optimizer_scheduler_contract,
    build_b1k_checkpoint_contract,
    build_training_checkpoint_contract,
)
from fastwam.utils.checkpoint_contract import (
    CheckpointContract,
    CheckpointContractMismatchError,
    assert_checkpoint_compatible,
)
from fastwam.datasets.b1k.text import B1K_PROMPT_TEMPLATE


def _config(repo_root, tmp_path):
    with initialize_config_dir(config_dir=str(repo_root / "configs"), version_base="1.3"):
        cfg = compose(
            config_name="train",
            overrides=["task=b1k_legacy_uncond_3cam_352x224_1e-4"],
        )
    stats = tmp_path / "norm_stats.json"
    stats.write_text('{"stats":"train-only"}\n', encoding="utf-8")
    action_dit = tmp_path / "action_dit.pt"
    action_dit.write_bytes(b"official interpolation checkpoint")
    cfg.data.train.pretrained_norm_stats = str(stats)
    cfg.data.val.pretrained_norm_stats = str(stats)
    cfg.model.action_dit_pretrained_path = str(action_dit)
    cfg.output_dir = str(tmp_path / "output")
    return cfg, stats, action_dit


def _dataset_contract():
    class Dataset:
        checkpoint_data_contract = {
            "schema": "b1k_2026_data_contract.v2",
            "sampling": {
                "source_fps": 30,
                "window_start_stride": 6,
                "anchor_frequency_hz": 5.0,
                "sample_count": 12,
                "tail_policy": "replicate_last_frame_with_padding_mask",
            },
            "action": {
                "dim": ACTION_DIM,
                "labels": list(ACTION_LABELS),
                "horizon": 32,
                "sample_stride": 2,
                "gripper_dims": list(ACTION_GRIPPER_DIMS),
                "constant_dims": [ACTION_CONSTANT_DIM],
                "gripper_mask": [index in ACTION_GRIPPER_DIMS for index in range(ACTION_DIM)],
                "constant_dim_mask": [
                    index == ACTION_CONSTANT_DIM for index in range(ACTION_DIM)
                ],
            },
            "proprio": {"dim": PROPRIO_DIM, "labels": list(PROPRIO_LABELS)},
            "video": {
                "dataset_image_indices": list(range(0, 33, 4)),
                "raw_frame_indices": list(range(0, 65, 8)),
                "canvas": {"height": 352, "width": 224},
                "camera_order": ["head", "left_wrist", "right_wrist"],
                "layout": {
                    "head": {"x": 0, "y": 0, "width": 224, "height": 224},
                    "left_wrist": {"x": 0, "y": 224, "width": 112, "height": 112},
                    "right_wrist": {"x": 112, "y": 224, "width": 112, "height": 112},
                },
                "padding": {
                    "x": 0,
                    "y": 336,
                    "width": 224,
                    "height": 16,
                    "value_before_normalization": 0.5,
                    "value_after_normalization": 0.0,
                },
            },
            "text": {
                "context_length": 128,
                "prompt_template": B1K_PROMPT_TEMPLATE,
                "attention_mask": "real_token_mask",
            },
            "normalization": {
                "mode": "mixed_z_score",
                "action_continuous_mode": "global_per_dimension_z_score_unclamped",
                "proprio_continuous_mode": "global_per_dimension_z_score_clamped",
                "stepwise": False,
                "proprio_clamp": [-5.0, 5.0],
                "action_and_proprio_stats": "independent",
                "gripper_action": "fixed_binary_minus_one_plus_one",
                "constant_action": "fixed_zero",
            },
        }

    return Dataset()


def test_builder_covers_strict_b1k_compatibility_contract(tmp_path, monkeypatch):
    repo_root = __import__("pathlib").Path(__file__).resolve().parents[3]
    cfg, stats, action_dit = _config(repo_root, tmp_path)
    model_root = tmp_path / "models" / "Wan-AI" / "Wan2.2-TI2V-5B"
    model_root.mkdir(parents=True)
    (model_root / "config.json").write_text('{"model":"wan2.2-5b"}', encoding="utf-8")
    (model_root / "diffusion_pytorch_model-00001-of-00001.safetensors").write_bytes(b"weights")
    monkeypatch.setenv("DIFFSYNTH_MODEL_BASE_PATH", str(tmp_path / "models"))

    contract = build_b1k_checkpoint_contract(cfg, _dataset_contract())
    compatibility = contract.compatibility

    assert compatibility["model"]["variant"] == "fastwam_uncond"
    assert compatibility["model"]["target"] == "fastwam.runtime.create_fastwam"
    assert compatibility["model"]["wan"]["id"] == "Wan-AI/Wan2.2-TI2V-5B"
    assert len(compatibility["model"]["wan"]["fingerprint"]["sha256"]) == 64
    assert compatibility["model"]["action_dit_initialization"]["id"] == action_dit.name
    assert len(compatibility["model"]["action_dit_initialization"]["fingerprint"]["sha256"]) == 64

    data = compatibility["data"]
    assert data["action"]["dim"] == 23
    assert len(data["action"]["labels"]) == 23
    assert data["action"]["horizon"] == 32
    assert data["action"]["sample_stride"] == 2
    assert data["sampling"] == {
        "source_fps": 30,
        "window_start_stride": 6,
        "anchor_frequency_hz": 5.0,
        "sample_count": 12,
        "tail_policy": "replicate_last_frame_with_padding_mask",
    }
    assert data["action"]["gripper_dims"] == [14, 22]
    assert data["action"]["constant_dims"] == [6]
    assert [index for index, value in enumerate(data["action"]["gripper_mask"]) if value] == [14, 22]
    assert [index for index, value in enumerate(data["action"]["constant_dim_mask"]) if value] == [6]
    assert data["proprio"]["dim"] == 23
    assert len(data["proprio"]["labels"]) == 23
    assert data["video"]["raw_frame_indices"] == list(range(0, 65, 8))
    assert data["video"]["canvas"] == {"height": 352, "width": 224}
    assert data["video"]["padding"] == {
        "x": 0,
        "y": 336,
        "width": 224,
        "height": 16,
        "value_before_normalization": 0.5,
        "value_after_normalization": 0.0,
    }
    assert data["video"]["camera_order"] == ["head", "left_wrist", "right_wrist"]
    assert data["text"]["context_length"] == 128
    assert data["text"]["prompt_template"].endswith("{task}")
    assert data["normalization"]["mode"] == "mixed_z_score"
    assert data["normalization"]["action_continuous_mode"] == (
        "global_per_dimension_z_score_unclamped"
    )
    assert data["normalization"]["proprio_continuous_mode"] == (
        "global_per_dimension_z_score_clamped"
    )
    assert data["normalization"]["stepwise"] is False
    assert data["normalization"]["stats_sha256"] == hashlib.sha256(stats.read_bytes()).hexdigest()
    assert compatibility["scheduler"]["action"]["num_train_timesteps"] == 1000
    assert compatibility["scheduler"]["video"]["train_shift"] == 5.0
    # Effective optimizer steps and warmup depend on the distributed loader and
    # are bound by Wan22Trainer before any checkpoint is loaded or saved.
    assert "optimizer_lr" not in compatibility["scheduler"]
    assert compatibility["loss"] == {"lambda_video": 1.0, "lambda_action": 1.0}
    assert contract.provenance["resolved_hydra_config"]["data"]["train"]["context_len"] == 128


def test_builder_records_mixture_sampling_contract(tmp_path, monkeypatch):
    repo_root = __import__("pathlib").Path(__file__).resolve().parents[3]
    cfg, _, _ = _config(repo_root, tmp_path)
    model_root = tmp_path / "models" / "Wan-AI" / "Wan2.2-TI2V-5B"
    model_root.mkdir(parents=True)
    (model_root / "config.json").write_text('{"model":"wan2.2-5b"}', encoding="utf-8")
    monkeypatch.setenv("DIFFSYNTH_MODEL_BASE_PATH", str(tmp_path / "models"))

    dataset = _dataset_contract()
    dataset.mixture_contract = {
        "weights": [0.7, 0.3],
        "epoch_size": 83968,
        "nominal_source_counts": [58778, 25190],
        "source_lengths": [83968, 12000],
        "seed": 42,
    }

    contract = build_b1k_checkpoint_contract(cfg, dataset)

    assert contract.compatibility["data"]["mixture"] == dataset.mixture_contract


def test_model_variant_supports_canonical_targets() -> None:
    assert checkpoint_module._model_variant("fastwam.runtime.create_fastwam") == "fastwam_uncond"
    assert (
        checkpoint_module._model_variant("fastwam.runtime.create_fastwam_joint")
        == "fastwam_joint"
    )
    assert (
        checkpoint_module._model_variant("fastwam.runtime.create_fastwam_flex")
        == "fastwam_flex"
    )
    with pytest.raises(B1KCheckpointContractError, match="uncond, joint, and flex"):
        checkpoint_module._model_variant("fastwam.runtime.create_fastwam_idm")


def test_flex_contract_uses_canonical_variant_and_routing_semantics(
    tmp_path, monkeypatch
) -> None:
    repo_root = __import__("pathlib").Path(__file__).resolve().parents[3]
    cfg, _, _ = _config(repo_root, tmp_path)
    OmegaConf.set_struct(cfg.model, False)
    cfg.model._target_ = "fastwam.runtime.create_fastwam_flex"
    cfg.model.dino_model_name = "vit_base_patch16_dinov3.lvd1689m"
    cfg.model.dino_dim = 3072
    cfg.model.dino_output_grid_size = 5
    cfg.model.dino_pred_x0 = False
    cfg.model.dino_scheduler = OmegaConf.create({
        "train_shift": 5.0,
        "infer_shift": 5.0,
        "num_train_timesteps": 1000,
    })
    cfg.model.pointmap_scheduler = cfg.model.dino_scheduler
    cfg.model.hbridge_enabled = True
    cfg.model.hbridge_bottom_ratio = 0.25
    cfg.model.hbridge_top_ratio = 0.25
    cfg.model.present_probabilities = OmegaConf.create(
        {"rgb": 0.5, "dino": 0.25, "pointmap": 1.0}
    )
    cfg.model.output_probabilities = OmegaConf.create(
        {"rgb": 1.0, "dino": 0.5, "pointmap": 0.0}
    )
    cfg.model.cross_modal_predict = OmegaConf.create(
        {"rgb": True, "dino": False, "pointmap": True}
    )
    cfg.model.inference_present_modalities = OmegaConf.create(
        {"rgb": True, "dino": False, "pointmap": True}
    )
    cfg.model.inference_output_modalities = OmegaConf.create(
        {"rgb": False, "dino": True, "pointmap": True}
    )
    model_root = tmp_path / "models" / "Wan-AI" / "Wan2.2-TI2V-5B"
    model_root.mkdir(parents=True)
    (model_root / "config.json").write_text('{"model":"wan2.2-5b"}', encoding="utf-8")
    (model_root / "weights.safetensors").write_bytes(b"weights")
    monkeypatch.setenv("DIFFSYNTH_MODEL_BASE_PATH", str(tmp_path / "models"))

    compatibility = build_b1k_checkpoint_contract(
        cfg, _dataset_contract()
    ).compatibility

    assert compatibility["model"]["variant"] == "fastwam_flex"
    assert compatibility["model"]["dino"]["prediction"] == "velocity"
    assert compatibility["model"]["dino"]["spatial_grid_per_camera"] == 5
    assert compatibility["model"]["dino"]["spatial_resample"] == "bilinear"
    assert compatibility["model"]["routing"] == {
        "hbridge": {"enabled": True, "bottom_ratio": 0.25, "top_ratio": 0.25},
        "present_probabilities": {"rgb": 0.5, "dino": 0.25, "pointmap": 1.0},
        "output_probabilities": {"rgb": 1.0, "dino": 0.5, "pointmap": 0.0},
        "sampling_scope": "per_sample",
        "cross_modal_predict": {"rgb": True, "dino": False, "pointmap": True},
        "inference": {
            "present_modalities": {"rgb": True, "dino": False, "pointmap": True},
            "output_modalities": {"rgb": False, "dino": True, "pointmap": True},
        },
    }


def test_builder_uses_snapshot_artifact_fingerprints(tmp_path, monkeypatch):
    repo_root = __import__("pathlib").Path(__file__).resolve().parents[3]
    cfg, stats, action_dit = _config(repo_root, tmp_path)
    model_root = tmp_path / "models" / "Wan-AI" / "Wan2.2-TI2V-5B"
    model_root.mkdir(parents=True)
    (model_root / "config.json").write_text('{"model":"wan2.2-5b"}', encoding="utf-8")
    (model_root / "diffusion.safetensors").write_bytes(b"weights")
    monkeypatch.setenv("DIFFSYNTH_MODEL_BASE_PATH", str(tmp_path / "models"))
    manifest = tmp_path / "run.json"
    manifest.write_text(
        __import__("json").dumps(
            {
                "run_id": "b1k-test",
                "git": {"commit": "snapshot-commit"},
                "code_snapshot": {"tree_sha256": "tree-hash"},
                "inputs": {
                    "norm_stats": {
                        "path": str(stats),
                        "fingerprint": checkpoint_module._path_fingerprint(
                            stats, full_file_hash=True
                        ),
                    },
                    "wan_model": {
                        "path": str(model_root),
                        "fingerprint": checkpoint_module._path_fingerprint(
                            model_root, full_file_hash=False
                        ),
                    },
                    "action_dit": {
                        "path": str(action_dit),
                        "fingerprint": checkpoint_module._path_fingerprint(
                            action_dit, full_file_hash=True
                        ),
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    contract = build_b1k_checkpoint_contract(
        cfg, _dataset_contract(), run_manifest_path=manifest
    )

    assert contract.compatibility["model"]["wan"]["fingerprint"] == checkpoint_module._path_fingerprint(
        model_root, full_file_hash=False
    )
    assert contract.compatibility["model"]["action_dit_initialization"]["fingerprint"] == checkpoint_module._path_fingerprint(
        action_dit, full_file_hash=True
    )
    assert contract.provenance["run_manifest"]["run_id"] == "b1k-test"
    assert contract.provenance["run_manifest"]["code_snapshot"]["tree_sha256"] == "tree-hash"


def test_builder_rejects_config_dataset_contract_drift(tmp_path, monkeypatch):
    repo_root = __import__("pathlib").Path(__file__).resolve().parents[3]
    cfg, _, _ = _config(repo_root, tmp_path)
    cfg.data.train.context_len = 64
    contract = _dataset_contract()

    with pytest.raises(B1KCheckpointContractError, match="text.context_length"):
        build_b1k_checkpoint_contract(cfg, contract)


def test_builder_rejects_window_start_stride_drift(tmp_path):
    repo_root = __import__("pathlib").Path(__file__).resolve().parents[3]
    cfg, _, _ = _config(repo_root, tmp_path)
    contract = _dataset_contract()
    contract.checkpoint_data_contract["sampling"]["window_start_stride"] = 3
    contract.checkpoint_data_contract["sampling"]["anchor_frequency_hz"] = 10.0

    with pytest.raises(B1KCheckpointContractError, match="window_start_stride"):
        build_b1k_checkpoint_contract(cfg, contract)


def test_builder_rejects_legacy_clamped_action_contract(tmp_path):
    repo_root = __import__("pathlib").Path(__file__).resolve().parents[3]
    cfg, _, _ = _config(repo_root, tmp_path)
    contract = _dataset_contract()
    normalization = contract.checkpoint_data_contract["normalization"]
    contract.checkpoint_data_contract["schema"] = "b1k_2026_data_contract.v1"
    normalization.pop("action_continuous_mode")
    normalization.pop("proprio_continuous_mode")
    normalization.pop("proprio_clamp")
    normalization["continuous_mode"] = "global_per_dimension_z_score"
    normalization["clamp"] = [-5.0, 5.0]

    with pytest.raises(B1KCheckpointContractError, match="data_contract.v2"):
        build_b1k_checkpoint_contract(cfg, contract)


def test_non_b1k_training_keeps_checkpoint_contract_disabled():
    cfg = OmegaConf.create({"data": {"train": {"_target_": "example.OtherDataset"}}})
    assert build_training_checkpoint_contract(cfg, object()) is None


def _bind_optimizer(contract, **overrides):
    values = {
        "learning_rate": 1.0e-4,
        "min_learning_rate": 1.0e-6,
        "lr_scheduler_type": "cosine",
        "lr_warmup_steps": 4250,
        "max_steps": 85010,
        "weight_decay": 1.0e-2,
        "betas": (0.9, 0.95),
        "epsilon": 1.0e-8,
        "amsgrad": False,
        "max_grad_norm": 1.0,
        "gradient_accumulation_steps": 1,
    }
    values.update(overrides)
    return bind_optimizer_scheduler_contract(contract, **values)


def test_optimizer_scheduler_contract_records_resolved_runtime_values(tmp_path, monkeypatch):
    repo_root = __import__("pathlib").Path(__file__).resolve().parents[3]
    cfg, _, _ = _config(repo_root, tmp_path)
    monkeypatch.setenv("DIFFSYNTH_MODEL_BASE_PATH", str(tmp_path / "models"))
    model_root = tmp_path / "models" / "Wan-AI" / "Wan2.2-TI2V-5B"
    model_root.mkdir(parents=True)
    (model_root / "config.json").write_text("{}", encoding="utf-8")

    bound = _bind_optimizer(build_b1k_checkpoint_contract(cfg, _dataset_contract()))

    assert bound.compatibility["scheduler"]["optimizer_lr"] == {
        "optimizer_type": "AdamW",
        "learning_rate": 1.0e-4,
        "min_learning_rate": 1.0e-6,
        "lr_scheduler_type": "cosine",
        "lr_warmup_steps": 4250,
        "max_steps": 85010,
        "weight_decay": 1.0e-2,
        "betas": [0.9, 0.95],
        "epsilon": 1.0e-8,
        "amsgrad": False,
        "max_grad_norm": 1.0,
        "gradient_accumulation_steps": 1,
    }
    assert "warmup_ratio" not in bound.compatibility["scheduler"]["optimizer_lr"]


@pytest.mark.parametrize(
    "override",
    [
        {"learning_rate": 2.0e-4},
        {"min_learning_rate": 0.0},
        {"lr_scheduler_type": "constant"},
        {"lr_warmup_steps": 100},
        {"max_steps": 1000},
        {"weight_decay": 0.0},
        {"betas": (0.9, 0.999)},
        {"epsilon": 1.0e-6},
        {"amsgrad": True},
        {"max_grad_norm": 0.5},
        {"gradient_accumulation_steps": 4},
    ],
)
def test_resume_rejects_optimizer_scheduler_configuration_drift(override):
    base = CheckpointContract(
        compatibility={"scheduler": {"video": {}, "action": {}}},
        provenance={},
    )
    saved = _bind_optimizer(base)
    resumed = _bind_optimizer(base, **override)

    with pytest.raises(CheckpointContractMismatchError, match="optimizer_lr"):
        assert_checkpoint_compatible(resumed, saved)
