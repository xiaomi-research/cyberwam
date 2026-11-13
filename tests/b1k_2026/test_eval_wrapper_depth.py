# Copyright (c) 2026 Xiaomi Corporation

import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.b1k_2026.eval_wrapper import RGBDFullResWrapper, RGBFullResWrapper
from experiments.b1k_2026.model_factory import _model_target_requires_depth


class _Sensor:
    image_height = 1
    image_width = 1

    def __init__(self):
        self.ready = True

    def get_obs(self):
        shape = (self.image_height, self.image_width, 3) if self.ready else (0,)
        return {"rgb": _Array(shape), "depth_linear": _Array(shape)}, {}


class _Array:
    def __init__(self, shape):
        self.shape = shape


class _Robot:
    name = "robot_r1"

    def __init__(self):
        self.sensors = {"head_sensor": _Sensor(), "wrist_sensor": _Sensor()}


class _Env:
    _eval_robot_config = {}

    def __init__(self):
        self.robots = [_Robot()]

    def load_observation_space(self):
        self.loaded = True


def _install_fake_omnigibson(monkeypatch):
    calls = []
    og = types.ModuleType("omnigibson")
    og.sim = types.SimpleNamespace(update_handles=lambda: None, render=lambda: None)
    envs = types.ModuleType("omnigibson.envs")
    class FakeEnvironmentWrapper:
        def __init__(self, env):
            assert getattr(env, "loaded", False), "wrapper must be built after observation space"
            self.env = env

    envs.EnvironmentWrapper = FakeEnvironmentWrapper
    eval_utils = types.ModuleType("omnigibson.eval.utils.eval_utils")
    eval_utils.HEAD_RESOLUTION = (720, 720)
    eval_utils.WRIST_RESOLUTION = (480, 480)
    eval_utils.get_robot_camera_names = lambda *_: {
        "head": "robot_r1::head_sensor", "left_wrist": "robot_r1::wrist_sensor"
    }
    eval_utils.set_sensor_modalities = lambda sensor, modalities: calls.append(set(modalities))
    gym_utils = types.ModuleType("omnigibson.utils.gym_utils")
    gym_utils.recursively_generate_compatible_dict = lambda dic: dic
    monkeypatch.setitem(sys.modules, "omnigibson", og)
    monkeypatch.setitem(sys.modules, "omnigibson.envs", envs)
    monkeypatch.setitem(sys.modules, "omnigibson.eval.utils.eval_utils", eval_utils)
    monkeypatch.setitem(sys.modules, "omnigibson.utils.gym_utils", gym_utils)
    return calls


def test_full_resolution_wrappers_preserve_rgb_default_and_opt_in_depth(monkeypatch):
    calls = _install_fake_omnigibson(monkeypatch)
    RGBFullResWrapper(_Env())
    assert calls == [{"rgb"}, {"rgb"}]

    calls.clear()
    RGBDFullResWrapper(_Env())
    assert calls == [{"rgb", "depth_linear"}, {"rgb", "depth_linear"}]


def test_full_resolution_wrapper_warms_up_empty_camera_frames(monkeypatch):
    _install_fake_omnigibson(monkeypatch)
    env = _Env()
    for sensor in env.robots[0].sensors.values():
        sensor.ready = False
    renders = []

    def render():
        renders.append(True)
        for sensor in env.robots[0].sensors.values():
            sensor.ready = True

    monkeypatch.setattr(sys.modules["omnigibson"].sim, "render", render, raising=False)
    RGBDFullResWrapper(env)
    assert renders, "wrapper should render while annotators are empty"


def test_full_resolution_wrapper_rejects_persistently_empty_cameras(monkeypatch):
    _install_fake_omnigibson(monkeypatch)
    env = _Env()
    for sensor in env.robots[0].sensors.values():
        sensor.ready = False

    with pytest.raises(RuntimeError, match="camera annotators did not produce"):
        RGBDFullResWrapper(env)


def test_wrapper_recovers_late_camera_frame_after_reset_validation(monkeypatch):
    _install_fake_omnigibson(monkeypatch)
    env = _Env()
    env.observation_space = types.SimpleNamespace(contains=lambda _: True)
    resets = []

    def reset():
        resets.append(True)
        for sensor in env.robots[0].sensors.values():
            sensor.ready = False
        raise ValueError("Observation space does not match returned observations!")

    def get_obs():
        return {"camera": _Array((1, 1, 3))}, {"recovered": True}

    def render():
        for sensor in env.robots[0].sensors.values():
            sensor.ready = True

    env.reset = reset
    env.get_obs = get_obs
    monkeypatch.setattr(sys.modules["omnigibson"].sim, "render", render, raising=False)
    wrapper = RGBDFullResWrapper(env)

    observations, info = wrapper.reset()

    assert resets == [True]
    assert observations["camera"].shape == (1, 1, 3)
    assert info == {"obs_info": {"recovered": True}}


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        ("fastwam.runtime.create_fastwam", False),
        ("fastwam.runtime.create_fastwam_flex", True),
    ],
)
def test_multimodal_model_targets_request_depth(target, expected):
    assert _model_target_requires_depth(target) is expected
