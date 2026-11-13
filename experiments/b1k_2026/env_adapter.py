# Copyright (c) 2026 Xiaomi Corporation

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import torch

from fastwam.datasets.b1k.contract import (
    ACTION_DIM,
    ACTION_LABELS,
    RAW_STATE_DIM,
    extract_b1k_proprio,
    extract_r1pro_simulator_proprio,
    validate_b1k_action,
)

DEFAULT_TASK = "turning_on_radio"
DEFAULT_INSTANCE_ID = 198


def _build_available_tasks_from_2025_templates(data_path: Path) -> dict[str, dict[int, dict[str, Any]]]:
    """Reconstruct task configs when the challenge bundle omits available_tasks.yaml."""
    challenge_root = Path(data_path) / "2025-challenge-task-instances"
    room_instances: dict[str, list[str]] = {}
    misc_path = challenge_root / "metadata/B50_task_misc.csv"
    if misc_path.is_file():
        with misc_path.open(newline="", encoding="utf-8") as stream:
            for row in csv.reader(stream):
                if len(row) >= 3 and row[0] != "Task ID":
                    room_instances[row[1]] = [item for item in row[2].splitlines() if item]

    tasks: dict[str, dict[int, dict[str, Any]]] = {}
    scenes_dir = challenge_root / "scenes"
    if not scenes_dir.is_dir():
        return tasks
    for scene_dir in sorted(path for path in scenes_dir.iterdir() if path.is_dir()):
        json_dir = scene_dir / "json"
        if not json_dir.is_dir():
            continue
        prefix = f"{scene_dir.name}_task_"
        suffix = "_0_0_template.json"
        for template_path in sorted(json_dir.glob(f"{prefix}*{suffix}")):
            task_name = template_path.name[len(prefix) : -len(suffix)]
            template = json.loads(template_path.read_text(encoding="utf-8"))
            robot_poses = template.get("metadata", {}).get("task", {}).get("robot_poses", {})
            pose = (robot_poses.get("R1Pro") or [{}])[0]
            task_cfg: dict[str, Any] = {
                "scene_model": scene_dir.name,
                "robot_start_position": pose.get("position", [0.0, 0.0, 0.0]),
                "robot_start_orientation": pose.get("orientation", [0.0, 0.0, 0.0, 1.0]),
            }
            if task_name in room_instances:
                task_cfg["load_room_instances"] = room_instances[task_name]
            tasks.setdefault(task_name, {})[0] = task_cfg
    return tasks


@dataclass(frozen=True)
class B1KSmokeEnvConfig:
    """Explicit, intentionally narrow configuration for a short simulator smoke."""

    behavior_root: Path
    task: str = DEFAULT_TASK
    instance_id: int = DEFAULT_INSTANCE_ID
    max_steps: int = 128
    partial_scene_load: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "behavior_root", Path(self.behavior_root).expanduser().resolve()
        )
        if not self.task or not isinstance(self.task, str):
            raise ValueError("task must be a non-empty BEHAVIOR activity name.")
        if isinstance(self.instance_id, bool) or not isinstance(self.instance_id, int):
            raise TypeError("instance_id must be an integer.")
        if self.instance_id < 0:
            raise ValueError("instance_id must be non-negative.")
        if (
            isinstance(self.max_steps, bool)
            or not isinstance(self.max_steps, int)
            or self.max_steps <= 0
        ):
            raise ValueError("max_steps must be a positive integer.")

    @classmethod
    def from_environment(cls) -> "B1KSmokeEnvConfig":
        root_value = os.environ.get("FASTWAM_B1K_BEHAVIOR_ROOT") or os.environ.get(
            "BEHAVIOR_BASE"
        )
        if not root_value:
            raise RuntimeError(
                "Set FASTWAM_B1K_BEHAVIOR_ROOT (or BEHAVIOR_BASE) to either the "
                "BEHAVIOR-1K checkout or its parent directory."
            )
        return cls(
            behavior_root=Path(root_value),
            task=os.environ.get("FASTWAM_B1K_TASK", DEFAULT_TASK),
            instance_id=int(
                os.environ.get("FASTWAM_B1K_INSTANCE_ID", DEFAULT_INSTANCE_ID)
            ),
            max_steps=int(os.environ.get("FASTWAM_B1K_MAX_STEPS", 128)),
            partial_scene_load=_environment_bool(
                "FASTWAM_B1K_PARTIAL_SCENE_LOAD", False
            ),
        )


def _environment_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value, got {value!r}.")


def _behavior_checkout(root: Path) -> Path:
    candidates = (root, root / "BEHAVIOR-1K")
    for candidate in candidates:
        if (candidate / "OmniGibson" / "omnigibson").is_dir():
            return candidate
    raise FileNotFoundError(
        f"Could not find BEHAVIOR-1K/OmniGibson under {root}. "
        "FASTWAM_B1K_BEHAVIOR_ROOT may point to the checkout or its parent."
    )


def _array(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _flatten_observation(obs: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten nested simulator observations without importing OmniGibson at module load."""

    flattened: dict[str, Any] = {}

    def visit(prefix: str, value: Any) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                child_prefix = f"{prefix}::{key}" if prefix else str(key)
                visit(child_prefix, child)
        else:
            flattened[prefix] = value

    visit("", obs)
    # Already-flattened OmniGibson observations pass through unchanged.
    flattened.update(
        {
            str(key): value
            for key, value in obs.items()
            if not isinstance(value, Mapping)
        }
    )
    return flattened


def _find_unique_suffix(obs: Mapping[str, Any], suffix: str) -> Any:
    matches = [(key, value) for key, value in obs.items() if key.endswith(suffix)]
    if len(matches) != 1:
        raise KeyError(
            f"Expected exactly one observation ending in {suffix!r}, found {[key for key, _ in matches]}."
        )
    return matches[0][1]


def _rgb_hwc(value: Any, name: str) -> np.ndarray:
    image = _array(value)
    if image.ndim != 3:
        raise ValueError(
            f"{name} RGB observation must have three dimensions, got {image.shape}."
        )
    if image.shape[0] in (3, 4) and image.shape[-1] not in (3, 4):
        image = np.moveaxis(image, 0, -1)
    if image.shape[-1] < 3:
        raise ValueError(
            f"{name} RGB observation must have at least three channels, got {image.shape}."
        )
    image = image[..., :3]
    if np.issubdtype(image.dtype, np.floating):
        if not np.isfinite(image).all():
            raise ValueError(f"{name} RGB observation contains non-finite values.")
        if float(image.max(initial=0.0)) <= 1.0 + 1e-6:
            image = np.rint(np.clip(image, 0.0, 1.0) * 255.0)
    return np.clip(image, 0, 255).astype(np.uint8)


def _depth_hw(value: Any, name: str) -> np.ndarray:
    depth = _array(value)
    if depth.ndim == 3 and depth.shape[0] == 1:
        depth = depth[0]
    elif depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2:
        raise ValueError(f"{name} depth observation must be HxW, got {depth.shape}.")
    depth = depth.astype(np.float32, copy=False)
    if np.isnan(depth).any() or np.isneginf(depth).any() or float(depth.min(initial=0.0)) < 0.0:
        raise ValueError(f"{name} depth observation contains invalid metric depth.")
    return depth


def _controller_report(robot: Any) -> dict[str, Any]:
    controllers = getattr(robot, "controllers", {})
    if not isinstance(controllers, Mapping):
        return {"available": False, "repr": repr(controllers)}
    return {
        "available": True,
        "names": sorted(str(name) for name in controllers),
        "types": {
            str(name): type(controller).__name__
            for name, controller in controllers.items()
        },
    }


def _select_robot_action_space(action_space: Any, robot_name: str) -> Any:
    spaces = getattr(action_space, "spaces", None)
    if isinstance(spaces, Mapping) and robot_name in spaces:
        return spaces[robot_name]
    return action_space


def _require_r1pro_identity(robot: Any) -> dict[str, str]:
    """Validate the generic Robot wrapper using its authoritative model metadata."""

    model = getattr(robot, "model", None) or getattr(robot, "model_name", None)
    name = getattr(robot, "name", None)
    identity = {
        "class": type(robot).__name__,
        "model": "" if model is None else str(model),
        "name": "" if name is None else str(name),
    }
    if identity["model"].casefold() != "r1pro":
        raise RuntimeError(
            "Short smoke requires the R1Pro model, got "
            f"class={identity['class']!r} model={identity['model']!r}."
        )
    if identity["name"] != "robot_r1":
        raise RuntimeError(
            f"Short smoke requires robot name 'robot_r1', got {identity['name']!r}."
        )
    return identity


def _select_robot_pose(robot_poses: Mapping[str, Any], robot_model: str) -> Any:
    for key, poses in robot_poses.items():
        if str(key).casefold() == robot_model.casefold():
            if not isinstance(poses, (list, tuple)) or not poses:
                raise ValueError(f"Robot pose list for {key!r} is empty or invalid.")
            return poses[0]
    raise KeyError(
        f"Task instance has no pose for robot model {robot_model!r}; "
        f"available={sorted(str(key) for key in robot_poses)}."
    )


def _task_instance_path(
    path_getter: Any, scene_model: str, activity_name: str, filename: str
) -> Path:
    instance_name = (
        f"{scene_model}_task_{activity_name}_instances/{filename}-tro_state"
    )
    return Path(path_getter(scene_model, instance_name))


def _stabilize_task_entities(
    robot: Any,
    entities: Any,
    base_system_type: type,
    step_physics: Any,
    *,
    steps: int = 25,
) -> None:
    for _ in range(steps):
        step_physics()
        robot.keep_still()
        for entity in entities:
            if entity is not None and not isinstance(entity, base_system_type):
                entity.keep_still()
    robot.keep_still()


class B1KSmokeEnvironment:
    """Thin adapter around the authoritative R1Pro OmniGibson environment.

    It deliberately does not add command clipping, alter controller limits, or
    define task aggregation. Its only purpose is a few short closed-loop steps.
    """

    def __init__(
        self,
        config: B1KSmokeEnvConfig,
        *,
        backend: Any | None = None,
        robot: Any | None = None,
        camera_suffixes: Mapping[str, str] | None = None,
        task_loader: Any | None = None,
        shutdown: Any | None = None,
    ) -> None:
        self.config = config
        self._backend = backend
        self._robot = robot
        self._camera_suffixes = dict(camera_suffixes or {})
        self._task_loader = task_loader
        self._shutdown = shutdown
        self._instance_loaded = False
        self._closed = False
        self._last_observation: dict[str, Any] | None = None
        self._command_count = 0
        self._command_min = np.full(ACTION_DIM, np.inf, dtype=np.float64)
        self._command_max = np.full(ACTION_DIM, -np.inf, dtype=np.float64)
        if backend is None:
            self._build_backend()

    def _build_backend(self) -> None:
        checkout = _behavior_checkout(self.config.behavior_root)
        omni_path = checkout / "OmniGibson"
        for path in (checkout, omni_path):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))

        import omnigibson as og
        from gello.utils.og_teleop_cfg import DISABLED_TRANSITION_RULES
        from gello.utils.og_teleop_utils import augment_rooms
        from gello.utils.og_teleop_utils import generate_robot_config
        from gello.utils.og_teleop_utils import get_task_relevant_room_types
        from gello.utils.og_teleop_utils import load_available_tasks
        from omnigibson.learning.utils.eval_utils import PROPRIOCEPTION_INDICES
        from omnigibson.learning.utils.eval_utils import ROBOT_CAMERA_NAMES
        from omnigibson.learning.utils.eval_utils import (
            generate_basic_environment_config,
        )
        from omnigibson.macros import gm
        from omnigibson.systems.system_base import BaseSystem
        from omnigibson.utils.asset_utils import get_task_instance_path
        from omnigibson.utils.python_utils import recursively_convert_to_torch

        gm.ENABLE_FLATCACHE = True
        gm.USE_GPU_DYNAMICS = False
        gm.ENABLE_TRANSITION_RULES = True
        for rule in DISABLED_TRANSITION_RULES:
            rule.ENABLED = False

        available_tasks = load_available_tasks()
        if not available_tasks:
            available_tasks = _build_available_tasks_from_2025_templates(Path(gm.DATA_PATH))
        if self.config.task not in available_tasks:
            raise ValueError(
                f"Unknown BEHAVIOR task {self.config.task!r}; available task count={len(available_tasks)}."
            )
        task_cfg = available_tasks[self.config.task][0]
        cfg = generate_basic_environment_config(
            task_name=self.config.task, task_cfg=task_cfg
        )
        if self.config.partial_scene_load:
            rooms = get_task_relevant_room_types(activity_name=self.config.task)
            cfg["scene"]["load_room_types"] = augment_rooms(
                rooms, task_cfg["scene_model"], self.config.task
            )
        robot_cfg = generate_robot_config(
            robot_type="r1pro",
            robot_name="robot_r1",
            task_name=self.config.task,
            task_cfg=task_cfg,
        )
        legacy_robot_type = robot_cfg.pop("type", None)
        robot_cfg["model"] = str(legacy_robot_type or "r1pro").lower()
        cfg["robots"] = [robot_cfg]
        include_depth = _environment_bool("FASTWAM_B1K_INCLUDE_DEPTH", False)
        cfg["robots"][0]["obs_modalities"] = (
            ["proprio", "rgb", "depth_linear"] if include_depth else ["proprio", "rgb"]
        )
        cfg["robots"][0]["proprio_obs"] = list(PROPRIOCEPTION_INDICES["R1Pro"].keys())
        cfg["task"]["termination_config"]["max_steps"] = self.config.max_steps
        cfg["task"]["include_obs"] = False

        self._backend = og.Environment(configs=cfg)
        robots = getattr(self._backend, "robots", ())
        self._robot = (
            robots[0]
            if robots
            else self._backend.scene.object_registry("name", "robot_r1")
        )
        self._robot_identity = _require_r1pro_identity(self._robot)
        self._camera_suffixes = {
            name: template.split("robot_r1::", 1)[-1] + "::rgb"
            for name, template in ROBOT_CAMERA_NAMES["R1Pro"].items()
        }
        self._depth_suffixes = {
            name: template.split("robot_r1::", 1)[-1] + "::depth_linear"
            for name, template in ROBOT_CAMERA_NAMES["R1Pro"].items()
        } if include_depth else {}

        def load_instance() -> None:
            scene_model = self._backend.task.scene_name
            filename = self._backend.task.get_cached_activity_scene_filename(
                scene_model=scene_model,
                activity_name=self._backend.task.activity_name,
                activity_definition_id=self._backend.task.activity_definition_id,
                activity_instance_id=self.config.instance_id,
            )
            path = _task_instance_path(
                get_task_instance_path,
                scene_model,
                self._backend.task.activity_name,
                filename,
            )
            with path.open("r", encoding="utf-8") as stream:
                tro_payload = recursively_convert_to_torch(json.load(stream))
            for key, state in tro_payload.items():
                if key == "robot_poses":
                    pose = _select_robot_pose(state, self._robot_identity["model"])
                    self._robot.set_position_orientation(
                        pose["position"], pose["orientation"]
                    )
                    self._backend.scene.write_task_metadata(key=key, data=state)
                else:
                    self._backend.task.object_scope[key].load_state(
                        state, serialized=False
                    )
            _stabilize_task_entities(
                self._robot,
                self._backend.task.object_scope.values(),
                BaseSystem,
                og.sim.step_physics,
            )
            self._backend.scene.update_initial_file()
            self._backend.scene.reset()

        self._task_loader = load_instance
        self._shutdown = og.shutdown

    def _observation(self, raw: Any) -> dict[str, Any]:
        if not isinstance(raw, Mapping):
            raise TypeError(
                f"OmniGibson observation must be a mapping, got {type(raw)!r}."
            )
        flat = _flatten_observation(raw)
        raw_proprio = _array(_find_unique_suffix(flat, "::proprio")).astype(
            np.float32, copy=False
        )
        if raw_proprio.shape == (RAW_STATE_DIM,):
            policy_proprio = extract_b1k_proprio(raw_proprio)
        elif raw_proprio.shape == (256,):
            policy_proprio = extract_r1pro_simulator_proprio(raw_proprio)
        else:
            raise ValueError(
                "R1Pro proprio must be either the 61D challenge state or the "
                f"256D OmniGibson state, got {raw_proprio.shape}."
            )
        images = {
            name: _rgb_hwc(_find_unique_suffix(flat, suffix), name)
            for name, suffix in self._camera_suffixes.items()
        }
        required_cameras = {"head", "left_wrist", "right_wrist"}
        if set(images) != required_cameras:
            raise ValueError(
                f"Expected cameras {sorted(required_cameras)}, got {sorted(images)}."
            )
        observation = {
            "state": raw_proprio,
            "proprio": np.asarray(policy_proprio, dtype=np.float32),
            "images": images,
            "task": self.config.task,
            "instance_id": self.config.instance_id,
        }
        depth_suffixes = getattr(self, "_depth_suffixes", {})
        if depth_suffixes:
            observation["depths"] = {
                name: _depth_hw(_find_unique_suffix(flat, suffix), name)
                for name, suffix in depth_suffixes.items()
            }
        self._last_observation = observation
        return observation

    def reset(self) -> tuple[dict[str, Any], dict[str, Any]]:
        if self._closed:
            raise RuntimeError("Cannot reset a closed B1K smoke environment.")
        raw = self._backend.reset()
        if isinstance(raw, tuple):
            raw = raw[0]
        if not self._instance_loaded:
            if self._task_loader is None:
                raise RuntimeError("No B1K task-instance loader is configured.")
            self._task_loader()
            self._instance_loaded = True
            raw = self._backend.reset()
            if isinstance(raw, tuple):
                raw = raw[0]
        observation = self._observation(raw)
        return observation, {"environment_validation": self.validation_report()}

    def step(
        self, command: Any
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        action = np.asarray(command, dtype=np.float32)
        if action.shape != (ACTION_DIM,):
            raise ValueError(
                f"R1Pro command must have shape ({ACTION_DIM},), got {action.shape}."
            )
        validate_b1k_action(action)
        self._command_count += 1
        self._command_min = np.minimum(self._command_min, action)
        self._command_max = np.maximum(self._command_max, action)
        result = self._backend.step(torch.from_numpy(action), n_render_iterations=1)
        if not isinstance(result, tuple) or len(result) != 5:
            raise TypeError(
                "OmniGibson step must return (obs, reward, terminated, truncated, info)."
            )
        raw, reward, terminated, truncated, info = result
        info = dict(info)
        info["environment_validation"] = self.validation_report()
        return (
            self._observation(raw),
            float(reward),
            bool(terminated),
            bool(truncated),
            info,
        )

    def validation_report(self) -> dict[str, Any]:
        robot_action_dim = getattr(self._robot, "action_dim", None)
        action_space = getattr(self._backend, "action_space", None)
        robot_identity = dict(
            getattr(
                self,
                "_robot_identity",
                {
                    "class": type(self._robot).__name__,
                    "model": str(getattr(self._robot, "model", "")),
                    "name": str(getattr(self._robot, "name", "")),
                },
            )
        )
        robot_action_space = _select_robot_action_space(
            action_space, robot_identity["name"]
        )
        action_space_shape = getattr(robot_action_space, "shape", None)
        report = {
            "schema": "b1k_2026_environment_validation.v1",
            "scope": "short_smoke_only",
            "config": {
                **asdict(self.config),
                "behavior_root": str(self.config.behavior_root),
            },
            "robot_type": type(self._robot).__name__,
            "robot_identity": robot_identity,
            "robot_action_dim": (
                None if robot_action_dim is None else int(robot_action_dim)
            ),
            "environment_action_space": repr(action_space),
            "robot_action_space": repr(robot_action_space),
            "environment_action_space_shape": (
                None if action_space_shape is None else list(action_space_shape)
            ),
            "controllers": _controller_report(self._robot),
            "controller_config_modified_by_adapter": False,
            "physical_limits_modified_by_adapter": False,
            "action_dim": ACTION_DIM,
            "action_labels": list(ACTION_LABELS),
            "command_count": self._command_count,
        }
        if self._last_observation is not None:
            report["observation"] = {
                "raw_state_shape": list(self._last_observation["state"].shape),
                "policy_proprio_shape": list(self._last_observation["proprio"].shape),
                "camera_shapes": {
                    name: list(image.shape)
                    for name, image in self._last_observation["images"].items()
                },
            }
        if self._command_count:
            report["command_min"] = self._command_min.tolist()
            report["command_max"] = self._command_max.tolist()
        if robot_action_dim is not None and int(robot_action_dim) != ACTION_DIM:
            raise RuntimeError(
                f"R1Pro simulator action_dim={robot_action_dim} does not match FastWAM action_dim={ACTION_DIM}."
            )
        if action_space_shape is not None and tuple(action_space_shape) != (
            ACTION_DIM,
        ):
            raise RuntimeError(
                f"R1Pro environment action_space.shape={tuple(action_space_shape)} does not match "
                f"FastWAM action shape ({ACTION_DIM},)."
            )
        return report

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._backend is not None and hasattr(self._backend, "close"):
            self._backend.close()
        if self._shutdown is not None:
            self._shutdown()


def create_environment() -> B1KSmokeEnvironment:
    """Plugin entry point for ``run_short_smoke.py --env-adapter``."""

    return B1KSmokeEnvironment(B1KSmokeEnvConfig.from_environment())


def replay_gt_commands(
    *,
    commands: Any,
    episode_index: int,
    dataset_root: Path,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Replay a short GT prefix through the matching B1K task instance."""

    del dataset_root
    command_array = np.asarray(commands, dtype=np.float32)
    if command_array.ndim != 2 or command_array.shape[1] != ACTION_DIM:
        raise ValueError(
            f"GT replay commands must have shape [T,{ACTION_DIM}], got {command_array.shape}."
        )
    max_ticks = int(os.environ.get("FASTWAM_B1K_GT_REPLAY_MAX_TICKS", "8"))
    if max_ticks <= 0:
        raise ValueError("FASTWAM_B1K_GT_REPLAY_MAX_TICKS must be positive.")
    task = metadata.get("task")
    instance_id = metadata.get("instance_id")
    if not isinstance(task, str) or not task:
        raise ValueError("GT replay metadata must include a non-empty task.")
    if isinstance(instance_id, bool) or not isinstance(instance_id, int):
        raise TypeError("GT replay metadata instance_id must be an integer.")
    root_value = os.environ.get("FASTWAM_B1K_BEHAVIOR_ROOT") or os.environ.get(
        "BEHAVIOR_BASE"
    )
    if not root_value:
        raise RuntimeError(
            "Set FASTWAM_B1K_BEHAVIOR_ROOT (or BEHAVIOR_BASE) before simulator GT replay."
        )
    environment = B1KSmokeEnvironment(
        B1KSmokeEnvConfig(
            behavior_root=Path(root_value),
            task=task,
            instance_id=instance_id,
            max_steps=max(max_ticks, 1),
            partial_scene_load=_environment_bool("FASTWAM_B1K_PARTIAL_SCENE_LOAD", False),
        )
    )
    rewards = []
    terminated = False
    truncated = False
    environment.reset()
    for command in command_array[:max_ticks]:
        _, reward, terminated, truncated, _ = environment.step(command)
        rewards.append(float(reward))
        if terminated or truncated:
            break
    result = {
        "episode_index": int(episode_index),
        "source": {
            key: value for key, value in metadata.items() if not str(key).startswith("_")
        },
        "requested_ticks": min(max_ticks, len(command_array)),
        "executed_ticks": len(rewards),
        "reward_sum": float(sum(rewards)),
        "terminated": terminated,
        "truncated": truncated,
        "environment_validation": environment.validation_report(),
    }
    finalizer = metadata.get("_finalize_environment_report")
    if finalizer is not None:
        if not callable(finalizer):
            raise TypeError("GT replay report finalizer must be callable.")
        finalizer(result)
    environment.close()
    return result
