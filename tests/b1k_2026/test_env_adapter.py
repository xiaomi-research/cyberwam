# Copyright (c) 2026 Xiaomi Corporation

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

import experiments.b1k_2026.model_factory as model_factory_module
from experiments.b1k_2026.env_adapter import (
    B1KSmokeEnvConfig,
    B1KSmokeEnvironment,
    _build_available_tasks_from_2025_templates,
    _require_r1pro_identity,
    _select_robot_action_space,
    _select_robot_pose,
    _stabilize_task_entities,
    _task_instance_path,
    replay_gt_commands,
)
from experiments.b1k_2026.model_factory import compose_b1k_config
from experiments.b1k_2026.policy_adapter import (
    build_b1k_mosaic,
    build_b1k_pointmap,
    policy,
)
from fastwam.datasets.b1k.text import format_b1k_prompt

CAMERA_SUFFIXES = {
    "head": "head::rgb",
    "left_wrist": "left::rgb",
    "right_wrist": "right::rgb",
}


def test_generic_robot_wrapper_is_validated_by_model_and_name():
    robot = type("Robot", (), {"model": "r1pro", "name": "robot_r1"})()
    assert _require_r1pro_identity(robot) == {
        "class": "Robot",
        "model": "r1pro",
        "name": "robot_r1",
    }

    with pytest.raises(RuntimeError, match="R1Pro model"):
        _require_r1pro_identity(
            type("Robot", (), {"model": "fetch", "name": "robot_r1"})()
        )


def test_robot_pose_selection_matches_model_case_insensitively():
    pose = {"position": [1, 2, 3], "orientation": [0, 0, 0, 1]}
    assert _select_robot_pose({"R1Pro": [pose]}, "r1pro") is pose


def test_task_instance_path_uses_current_omnigibson_api(tmp_path):
    captured = []

    def get_path(scene_name, instance_name):
        captured.append((scene_name, instance_name))
        return tmp_path / f"{instance_name}.json"

    path = _task_instance_path(
        get_path,
        "Benevolence_1_int",
        "turning_on_radio",
        "instance_000",
    )

    assert captured == [
        (
            "Benevolence_1_int",
            "Benevolence_1_int_task_turning_on_radio_instances/instance_000-tro_state",
        )
    ]
    assert path == (
        tmp_path
        / "Benevolence_1_int_task_turning_on_radio_instances/instance_000-tro_state.json"
    )


def test_task_entity_stabilization_matches_current_omnigibson_types():
    calls = []

    class _BaseSystem:
        pass

    class _Entity:
        def __init__(self, name):
            self.name = name

        def keep_still(self):
            calls.append(self.name)

    robot = _Entity("robot")
    entity = _Entity("object")

    _stabilize_task_entities(
        robot,
        [entity, _BaseSystem(), None],
        _BaseSystem,
        lambda: calls.append("physics"),
        steps=2,
    )

    assert calls == [
        "physics",
        "robot",
        "object",
        "physics",
        "robot",
        "object",
        "robot",
    ]


def test_build_available_tasks_from_2025_templates(tmp_path):
    root = tmp_path / "2025-challenge-task-instances"
    metadata = root / "metadata"
    scene_json = root / "scenes/Benevolence_1_int/json"
    metadata.mkdir(parents=True)
    scene_json.mkdir(parents=True)
    (metadata / "B50_task_misc.csv").write_text(
        "Task ID,Task Name,Room Instances\n0,turning_on_radio,kitchen_0\n",
        encoding="utf-8",
    )
    (scene_json / "Benevolence_1_int_task_turning_on_radio_0_0_template.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "task": {
                        "robot_poses": {
                            "R1Pro": [
                                {"position": [1, 2, 3], "orientation": [0, 0, 0, 1]}
                            ]
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    tasks = _build_available_tasks_from_2025_templates(tmp_path)

    assert tasks["turning_on_radio"][0] == {
        "scene_model": "Benevolence_1_int",
        "robot_start_position": [1, 2, 3],
        "robot_start_orientation": [0, 0, 0, 1],
        "load_room_instances": ["kitchen_0"],
    }


def _raw_observation() -> dict:
    state = np.zeros(61, dtype=np.float32)
    state[24:26] = 0.02
    state[49:51] = 0.03
    return {
        "robot_r1::proprio": state,
        "robot_r1::head::rgb": np.zeros((720, 720, 4), dtype=np.float32),
        "robot_r1::left::rgb": np.zeros((480, 480, 3), dtype=np.uint8),
        "robot_r1::right::rgb": np.zeros((3, 480, 480), dtype=np.uint8),
    }


def _full_simulator_observation() -> dict:
    observation = _raw_observation()
    state = np.zeros(256, dtype=np.float32)
    state[253:256] = [1, 2, 3]
    state[236:240] = [4, 5, 6, 7]
    state[158:165] = np.arange(8, 15)
    state[193:195] = [0.02, 0.03]
    state[197:204] = np.arange(15, 22)
    state[232:234] = [0.04, 0.05]
    observation["robot_r1::proprio"] = state
    return observation


class _Robot:
    action_dim = 23
    model = "r1pro"
    name = "robot_r1"
    controllers = {"base": object(), "trunk": object(), "arm_left": object()}


class _Backend:
    action_space = type(
        "DictSpace",
        (),
        {
            "spaces": {
                "robot_r1": type("BoxSpace", (), {"shape": (23,)})(),
            }
        },
    )()

    def __init__(self):
        self.commands = []
        self.closed = False

    def reset(self):
        return _raw_observation(), {}

    def step(self, action, n_render_iterations):
        assert n_render_iterations == 1
        self.commands.append(action)
        return _raw_observation(), 0.25, False, False, {}

    def close(self):
        self.closed = True


def _environment() -> tuple[B1KSmokeEnvironment, _Backend]:
    backend = _Backend()
    environment = B1KSmokeEnvironment(
        B1KSmokeEnvConfig(Path("/tmp"), task="turning_on_radio", instance_id=198),
        backend=backend,
        robot=_Robot(),
        camera_suffixes=CAMERA_SUFFIXES,
        task_loader=lambda: None,
    )
    return environment, backend


def test_smoke_environment_selects_instance_and_sends_exact_23d_command():
    environment, backend = _environment()
    observation, info = environment.reset()
    assert observation["proprio"].shape == (23,)
    assert observation["images"]["head"].shape == (720, 720, 3)
    assert observation["instance_id"] == 198
    assert (
        info["environment_validation"]["controller_config_modified_by_adapter"] is False
    )
    assert info["environment_validation"]["environment_action_space_shape"] == [23]

    command = np.zeros(23, dtype=np.float32)
    command[0:3] = [0.1, -0.2, 0.3]
    command[3:7] = [0.4, 0.5, 0.6, 0.0]
    command[7:14] = np.arange(7) / 10
    command[14] = -1
    command[15:22] = -np.arange(7) / 10
    command[22] = 1
    _, reward, terminated, truncated, step_info = environment.step(command)

    assert reward == 0.25
    assert not terminated and not truncated
    assert len(backend.commands) == 1
    assert isinstance(backend.commands[0], torch.Tensor)
    np.testing.assert_array_equal(backend.commands[0].numpy(), command)
    assert step_info["environment_validation"]["command_count"] == 1
    environment.close()
    assert backend.closed


def test_robot_action_space_selects_named_subspace():
    outer = _Backend.action_space
    assert _select_robot_action_space(outer, "robot_r1") is outer.spaces["robot_r1"]
    assert _select_robot_action_space(outer, "missing") is outer


def test_smoke_environment_accepts_full_omnigibson_proprio():
    environment, _ = _environment()

    observation = environment._observation(_full_simulator_observation())

    np.testing.assert_allclose(
        observation["proprio"],
        [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            *range(8, 15),
            0.05,
            *range(15, 22),
            0.09,
        ],
    )
    assert observation["state"].shape == (256,)


def test_smoke_environment_rejects_nonbinary_gripper_before_backend_step():
    environment, backend = _environment()
    environment.reset()
    command = np.zeros(23, dtype=np.float32)
    command[14] = 0
    command[22] = 1
    with pytest.raises(ValueError, match="gripper dimension 14"):
        environment.step(command)
    assert backend.commands == []


def test_gt_replay_loads_task_and_instance_from_episode_metadata(monkeypatch, tmp_path):
    captured = {}

    class _ReplayEnvironment:
        def __init__(self, config):
            captured["config"] = config

        def reset(self):
            return {}, {}

        def step(self, command):
            return {}, 0.5, False, False, {}

        def validation_report(self):
            return {"command_count": 2}

        def close(self):
            captured["closed"] = True

    monkeypatch.setenv("FASTWAM_B1K_BEHAVIOR_ROOT", str(tmp_path))
    monkeypatch.setattr(
        "experiments.b1k_2026.env_adapter.B1KSmokeEnvironment", _ReplayEnvironment
    )
    commands = np.zeros((2, 23), dtype=np.float32)
    commands[:, 14] = -1
    commands[:, 22] = 1

    finalized = []
    report = replay_gt_commands(
        commands=commands,
        episode_index=201,
        dataset_root=tmp_path,
        metadata={
            "task": "picking_up_trash",
            "instance_id": 1,
            "_finalize_environment_report": finalized.append,
        },
    )

    assert captured["config"].task == "picking_up_trash"
    assert captured["config"].instance_id == 1
    assert captured["closed"] is True
    assert report["executed_ticks"] == 2
    assert finalized == [report]
    assert "_finalize_environment_report" not in report["source"]


def test_gt_replay_preserves_original_error_before_shutdown(monkeypatch, tmp_path):
    captured = {"closed": False}

    class _FailingEnvironment:
        def __init__(self, config):
            del config

        def reset(self):
            raise RuntimeError("reset failed")

        def close(self):
            captured["closed"] = True

    monkeypatch.setenv("FASTWAM_B1K_BEHAVIOR_ROOT", str(tmp_path))
    monkeypatch.setattr(
        "experiments.b1k_2026.env_adapter.B1KSmokeEnvironment", _FailingEnvironment
    )
    commands = np.zeros((2, 23), dtype=np.float32)
    commands[:, 14] = -1
    commands[:, 22] = 1

    with pytest.raises(RuntimeError, match="reset failed"):
        replay_gt_commands(
            commands=commands,
            episode_index=201,
            dataset_root=tmp_path,
            metadata={"task": "picking_up_trash", "instance_id": 1},
        )

    assert captured["closed"] is False


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.torch_dtype = torch.float32
        self.text_encoder = None
        self.call = None

    @property
    def device(self):
        return self.anchor.device

    def infer_action(self, **kwargs):
        self.call = kwargs
        return {"action": torch.zeros(32, 23)}


def test_policy_adapter_builds_training_layout_and_calls_flex_inference(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("FASTWAM_B1K_INFERENCE_STEPS", "3")
    monkeypatch.setenv("FASTWAM_B1K_TEXT_CACHE_DIR", str(tmp_path))
    prompt = format_b1k_prompt("turning_on_radio")
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    mask = torch.zeros(128, dtype=torch.bool)
    mask[:7] = True
    context = torch.randn(128, 16)
    torch.save(
        {"context": context, "mask": mask},
        tmp_path / f"{digest}.t5_len128.wan22ti2v5b.pt",
    )
    images = {
        "head": np.full((224, 224, 3), 255, dtype=np.uint8),
        "left_wrist": np.zeros((224, 224, 3), dtype=np.uint8),
        "right_wrist": np.full((224, 224, 3), 127, dtype=np.uint8),
    }
    mosaic = build_b1k_mosaic(images)
    assert mosaic.shape == (3, 384, 256)
    assert torch.allclose(mosaic[:, :256], torch.ones_like(mosaic[:, :256]))
    assert torch.allclose(mosaic[:, 256:, :128], -torch.ones_like(mosaic[:, 256:, :128]))

    model = _Model()
    action = policy(
        model,
        observation={"images": images, "task": "turning_on_radio"},
        normalized_proprio=np.zeros(23, dtype=np.float32),
        step=0,
        seed=7,
    )
    assert action.shape == (32, 23)
    assert model.call["input_image"].shape == (1, 3, 384, 256)
    assert model.text_encoder is None
    assert model.call["prompt"] is None
    assert torch.equal(model.call["context_mask"].cpu(), mask)
    assert not bool(model.call["context_mask"][7:].any().item())
    assert torch.count_nonzero(model.call["context"][7:]).item() == 0
    assert model.call["num_inference_steps"] == 3
    assert model.call["seed"] == 7
    assert "num_video_frames" not in model.call


def test_policy_adapter_passes_training_video_length_to_joint_model(monkeypatch):
    class _JointModel(_Model):
        def infer_action(self, *, num_video_frames, **kwargs):
            self.call = {"num_video_frames": num_video_frames, **kwargs}
            return {"action": torch.zeros(32, 23)}

    monkeypatch.setattr(
        "experiments.b1k_2026.policy_adapter.load_cached_text_context",
        lambda task: (
            torch.zeros(128, 16),
            torch.ones(128, dtype=torch.bool),
            Path("unused"),
        ),
    )
    images = {
        "head": np.zeros((224, 224, 3), dtype=np.uint8),
        "left_wrist": np.zeros((224, 224, 3), dtype=np.uint8),
        "right_wrist": np.zeros((224, 224, 3), dtype=np.uint8),
    }
    model = _JointModel()

    action = policy(
        model,
        observation={"images": images, "task": "turning_on_radio"},
        normalized_proprio=np.zeros(23, dtype=np.float32),
        step=0,
        seed=7,
    )

    assert action.shape == (32, 23)
    assert model.call["num_video_frames"] == 9


def test_policy_adapter_passes_video_length_to_flex_forwarding_wrapper(monkeypatch):
    class _FlexForwardingModel(_Model):
        _fastwam_b1k_model_target = "fastwam.runtime.create_fastwam_flex"

        def infer_action(self, *args, **kwargs):
            self.call = kwargs
            return {"action": torch.zeros(32, 23)}

    monkeypatch.setattr(
        "experiments.b1k_2026.policy_adapter.load_cached_text_context",
        lambda task: (
            torch.zeros(128, 16),
            torch.ones(128, dtype=torch.bool),
            Path("unused"),
        ),
    )
    images = {
        name: np.zeros((32, 32, 3), dtype=np.uint8)
        for name in ("head", "left_wrist", "right_wrist")
    }
    model = _FlexForwardingModel()

    action = policy(
        model,
        observation={"images": images, "task": "turning_on_radio", "depths": {}},
        normalized_proprio=np.zeros(23, dtype=np.float32),
        step=0,
        seed=7,
    )

    assert action.shape == (32, 23)
    assert model.call["num_video_frames"] == 9


def test_policy_adapter_does_not_pass_video_length_to_legacy_kwargs_model(monkeypatch):
    class _LegacyModel(_Model):
        # The data processor may expose this metadata even when the legacy
        # action-only API does not consume video length.
        _fastwam_b1k_num_video_frames = 9

    monkeypatch.setattr(
        "experiments.b1k_2026.policy_adapter.load_cached_text_context",
        lambda task: (
            torch.zeros(128, 16),
            torch.ones(128, dtype=torch.bool),
            Path("unused"),
        ),
    )
    images = {
        name: np.zeros((32, 32, 3), dtype=np.uint8)
        for name in ("head", "left_wrist", "right_wrist")
    }
    model = _LegacyModel()

    policy(
        model,
        observation={"images": images, "task": "turning_on_radio"},
        normalized_proprio=np.zeros(23, dtype=np.float32),
        step=0,
        seed=7,
    )

    assert "num_video_frames" not in model.call


def test_policy_adapter_uses_model_action_horizon_and_validates_dynamic_shape(
    monkeypatch,
):
    class _DynamicModel(_Model):
        _fastwam_b1k_action_horizon = 12

        def infer_action(self, **kwargs):
            self.call = kwargs
            return {"action": torch.zeros(12, 23)}

    monkeypatch.delenv("FASTWAM_B1K_ACTION_HORIZON", raising=False)
    monkeypatch.setattr(
        "experiments.b1k_2026.policy_adapter.load_cached_text_context",
        lambda task: (
            torch.zeros(128, 16),
            torch.ones(128, dtype=torch.bool),
            Path("unused"),
        ),
    )
    images = {
        name: np.zeros((32, 32, 3), dtype=np.uint8)
        for name in ("head", "left_wrist", "right_wrist")
    }
    model = _DynamicModel()

    action = policy(
        model,
        observation={"images": images, "task": "turning_on_radio"},
        normalized_proprio=np.zeros(23, dtype=np.float32),
        step=0,
        seed=7,
    )

    assert action.shape == (12, 23)
    assert model.call["action_horizon"] == 12
    assert model.call["num_inference_steps"] == 10


def test_policy_adapter_builds_flex_pointmap_from_metric_depth(monkeypatch):
    class _FlexModel(_Model):
        _fastwam_b1k_requires_depth = True
        _fastwam_b1k_num_video_frames = 17

        def infer_action(self, *, input_pointmap, num_video_frames, **kwargs):
            self.call = {
                "input_pointmap": input_pointmap,
                "num_video_frames": num_video_frames,
                **kwargs,
            }
            return {"action": torch.zeros(32, 23)}

    monkeypatch.setattr(
        "experiments.b1k_2026.policy_adapter.load_cached_text_context",
        lambda task: (
            torch.zeros(128, 16),
            torch.ones(128, dtype=torch.bool),
            Path("unused"),
        ),
    )
    images = {
        "head": np.zeros((720, 720, 3), dtype=np.uint8),
        "left_wrist": np.zeros((480, 480, 3), dtype=np.uint8),
        "right_wrist": np.zeros((480, 480, 3), dtype=np.uint8),
    }
    depths = {
        "head": np.ones((720, 720, 1), dtype=np.float32),
        "left_wrist": np.ones((480, 480), dtype=np.float32),
        "right_wrist": np.ones((1, 480, 480), dtype=np.float32),
    }
    pointmap = build_b1k_pointmap(depths)
    assert pointmap.shape == (3, 384, 256)
    assert torch.isfinite(pointmap).all()

    model = _FlexModel()
    action = policy(
        model,
        observation={
            "images": images,
            "depths": depths,
            "task": "turning_on_radio",
        },
        normalized_proprio=np.zeros(23, dtype=np.float32),
        step=0,
        seed=7,
    )

    assert action.shape == (32, 23)
    assert model.call["input_pointmap"].shape == (1, 3, 384, 256)
    assert model.call["num_video_frames"] == 17


def test_depth_adapter_preserves_positive_infinity_as_invalid_depth():
    from experiments.b1k_2026.env_adapter import _depth_hw

    depth = np.array([[1.0, np.inf]], dtype=np.float32)
    result = _depth_hw(depth, "head")

    assert result.shape == (1, 2)
    assert np.isposinf(result[0, 1])

    for invalid in (np.nan, -np.inf, -1.0):
        with pytest.raises(ValueError, match="invalid metric depth"):
            _depth_hw(np.array([[invalid]], dtype=np.float32), "head")


def test_policy_adapter_requires_depth_only_for_flex(monkeypatch):
    monkeypatch.setattr(
        "experiments.b1k_2026.policy_adapter.load_cached_text_context",
        lambda task: (
            torch.zeros(128, 16),
            torch.ones(128, dtype=torch.bool),
            Path("unused"),
        ),
    )
    images = {
        name: np.zeros((32, 32, 3), dtype=np.uint8)
        for name in ("head", "left_wrist", "right_wrist")
    }
    model = _Model()
    assert policy(
        model,
        observation={"images": images, "task": "turning_on_radio"},
        normalized_proprio=np.zeros(23, dtype=np.float32),
        step=0,
        seed=7,
    ).shape == (32, 23)

    model._fastwam_b1k_requires_depth = True
    with pytest.raises(KeyError, match="must contain a 'depths' mapping"):
        policy(
            model,
            observation={"images": images, "task": "turning_on_radio"},
            normalized_proprio=np.zeros(23, dtype=np.float32),
            step=0,
            seed=7,
        )


def test_model_factory_config_forces_cache_only_text(monkeypatch):
    monkeypatch.setenv("FASTWAM_B1K_MODEL_ID", "/models/wan22")
    monkeypatch.setenv("FASTWAM_B1K_ACTION_DIT_PATH", "/models/action_dit.pt")
    cfg = compose_b1k_config()
    assert cfg.model.load_text_encoder is False
    assert cfg.model.model_id == "/models/wan22"
    assert cfg.model.action_dit_pretrained_path == "/models/action_dit.pt"


def test_model_factory_defaults_to_formal_flex_and_supports_explicit_joint(monkeypatch):
    monkeypatch.delenv("FASTWAM_B1K_HYDRA_TASK", raising=False)
    flex = compose_b1k_config()
    assert flex.model._target_ == "fastwam.runtime.create_fastwam_flex"

    monkeypatch.setenv(
        "FASTWAM_B1K_HYDRA_TASK",
        "b1k_2026_joint_3cam_direct384x256_mb32_gc_off_128gpu",
    )
    joint = compose_b1k_config()
    assert joint.model._target_ == "fastwam.runtime.create_fastwam_joint"


def test_model_factory_reads_device_and_dtype_without_loading_checkpoint(monkeypatch):
    cfg = OmegaConf.create({
        "model": {"_target_": "unused.Factory", "load_text_encoder": False},
        "data": {"val": {"num_frames": 9}},
    })
    captured = {}

    class _FactoryModel:
        text_encoder = None

    def fake_instantiate(model_cfg, **kwargs):
        captured["model_cfg"] = model_cfg
        captured.update(kwargs)
        return _FactoryModel()

    monkeypatch.setattr(model_factory_module, "compose_b1k_config", lambda: cfg)
    monkeypatch.setattr(model_factory_module, "instantiate", fake_instantiate)
    monkeypatch.setenv("FASTWAM_B1K_DEVICE", "cuda:3")
    monkeypatch.setenv("FASTWAM_B1K_DTYPE", "fp16")

    model = model_factory_module.model_factory()

    assert isinstance(model, _FactoryModel)
    assert captured["device"] == "cuda:3"
    assert captured["model_dtype"] is torch.float16
    assert captured["model_cfg"].load_text_encoder is False
    assert model._fastwam_b1k_model_target == "unused.Factory"
    assert model._fastwam_b1k_requires_depth is False


def test_model_factory_derives_flex_temporal_contract_from_train_when_val_is_null(
    monkeypatch,
):
    cfg = OmegaConf.create(
        {
            "model": {
                "_target_": "fastwam.runtime.create_fastwam_flex",
                "load_text_encoder": False,
            },
            "data": {
                "train": {
                    "num_frames": 33,
                    "processor": {"num_obs_steps": 33, "num_image_steps": 9},
                },
                "val": None,
            },
        }
    )

    class _FactoryModel:
        text_encoder = None

    monkeypatch.setattr(model_factory_module, "compose_b1k_config", lambda: cfg)
    monkeypatch.setattr(
        model_factory_module, "instantiate", lambda model_cfg, **kwargs: _FactoryModel()
    )

    model = model_factory_module.model_factory()

    assert model._fastwam_b1k_requires_depth is True
    assert model._fastwam_b1k_action_horizon == 32
    assert model._fastwam_b1k_num_video_frames == 9


def test_checkpoint_probe_is_deterministic_and_covers_both_experts():
    class _TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proprio_encoder = torch.nn.Linear(23, 8)
            self.video_expert = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 2))
            self.action_expert = torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.Linear(5, 2))

    model = _TinyModel()
    first = model_factory_module.checkpoint_probe(model, seed=17)
    second = model_factory_module.checkpoint_probe(model, seed=17)
    assert set(first) == {
        "proprio_output",
        "video_parameter_sample",
        "action_parameter_sample",
    }
    assert first["proprio_output"].shape == (1, 8)
    for key in first:
        assert torch.equal(first[key], second[key])


def test_checkpoint_probe_covers_flex_adapters():
    class _TinyFlex(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proprio_encoder = torch.nn.Linear(23, 8)
            self.video_expert = torch.nn.Linear(4, 4)
            self.action_expert = torch.nn.Linear(3, 3)
            self.dino_embedder = torch.nn.Linear(4, 4)
            self.dino_proj_out = torch.nn.Linear(4, 4)
            self.pt_patch_embedding = torch.nn.Linear(4, 4)
            self.pt_head = torch.nn.Linear(4, 4)

    result = model_factory_module.checkpoint_probe(_TinyFlex(), seed=17)
    assert {
        "dino_embedder_parameter_sample",
        "dino_proj_out_parameter_sample",
        "pt_patch_embedding_parameter_sample",
        "pt_head_parameter_sample",
    } <= set(result)
