# Copyright (c) 2026 Xiaomi Corporation

from __future__ import annotations

from typing import Any


def _shape(value: Any):
    """Return an observation shape without importing numpy/torch in the wrapper."""
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return tuple(int(dim) for dim in shape)
    except (TypeError, ValueError):
        return None


def _cameras_ready(sensors: list[Any], expected_modalities: set[str]) -> bool:
    """Check that every configured camera annotator has produced a non-empty frame."""
    for sensor in sensors:
        try:
            observations = sensor.get_obs()[0]
        except Exception:
            return False
        if not isinstance(observations, dict):
            return False
        if not expected_modalities.issubset(observations):
            return False
        height, width = int(sensor.image_height), int(sensor.image_width)
        for modality in expected_modalities:
            shape = _shape(observations[modality])
            if shape is None or len(shape) < 2 or shape[0] == 0 or shape[0] != height or shape[1] != width:
                return False
    return True


def _wait_for_camera_ready(og: Any, sensors: list[Any], modalities: set[str], attempts: int = 120) -> None:
    """Bounded render warmup for annotators recreated by a resolution change."""
    for attempt in range(attempts):
        if _cameras_ready(sensors, modalities):
            return
        if attempt and attempt % 16 == 0:
            og.sim.update_handles()
        og.sim.render()
    raise RuntimeError(
        f"camera annotators did not produce {sorted(modalities)} after {attempts} renders"
    )


def _full_res_wrapper(env: Any, *, include_depth: bool):
    import omnigibson as og
    from omnigibson.envs import EnvironmentWrapper
    from omnigibson.eval.utils.eval_utils import (
        HEAD_RESOLUTION,
        WRIST_RESOLUTION,
        get_robot_camera_names,
        set_sensor_modalities,
    )
    from omnigibson.utils.gym_utils import recursively_generate_compatible_dict

    robot = env.robots[0]
    roles = {
        camera_name.split("::")[1]: camera_id
        for camera_id, camera_name in get_robot_camera_names(
            robot.name, getattr(env, "_eval_robot_config", {})
        ).items()
    }
    modalities = {"rgb", "depth_linear"} if include_depth else {"rgb"}
    camera_sensors = []
    for sensor_name, sensor in robot.sensors.items():
        if not hasattr(sensor, "image_height") or not hasattr(sensor, "image_width"):
            continue
        camera_sensors.append(sensor)
        set_sensor_modalities(sensor, modalities)
        resolution = HEAD_RESOLUTION if roles.get(sensor_name) == "head" else WRIST_RESOLUTION
        sensor.image_height, sensor.image_width = resolution

    og.sim.update_handles()
    # Replicator annotators can lag behind a recreated render product even after
    # handles are refreshed. Warm them up before deriving the expected space.
    _wait_for_camera_ready(og, camera_sensors, modalities)
    env.load_observation_space()

    class CameraReadyEnvironmentWrapper(EnvironmentWrapper):
        def reset(self):
            try:
                return self.env.reset()
            except ValueError as error:
                if str(error) != "Observation space does not match returned observations!":
                    raise
                # The task reset has already completed; only Replicator's
                # camera frame was late. Wait and re-read the same state rather
                # than resetting the task a second time.
                _wait_for_camera_ready(og, camera_sensors, modalities)
                observations, info = self.env.get_obs()
                compatible = recursively_generate_compatible_dict(dic=observations)
                if not self.env.observation_space.contains(compatible):
                    raise RuntimeError(
                        "observation space still mismatches after camera recovery"
                    ) from error
                return observations, {"obs_info": info}

    # EnvironmentWrapper caches ``observation_space`` during construction.
    # Build it only after sensor modalities/resolutions have been applied and
    # the environment has rebuilt its observation space; otherwise the wrapper
    # validates returned observations against the stale pre-resize space.
    return CameraReadyEnvironmentWrapper(env=env)


class RGBFullResWrapper:
    """Build B1K's full-resolution RGB observation space after physics handles exist."""

    def __new__(cls, env: Any):
        return _full_res_wrapper(env, include_depth=False)


class RGBDFullResWrapper:
    """Opt-in full-resolution RGB-D wrapper for Stage-1 pointmap inference."""

    def __new__(cls, env: Any):
        return _full_res_wrapper(env, include_depth=True)
