# Copyright (c) 2026 Xiaomi Corporation

"""FastWAM policy server for official BEHAVIOR-1K Docker evaluation.

Only the policy server runs in this image. OmniGibson and its evaluator remain
outside the container, exactly as required by the 2026 challenge. Checkpoints,
contracts, statistics, and optional policy plugins are runtime inputs.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import hashlib
import functools
from collections import deque
from pathlib import Path
import sys
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from challenge_docker.observation import adapt_observation  # noqa: E402
from challenge_docker.protocol import WebsocketPolicyServer  # noqa: E402
from scripts.b1k_2026.sanity_common import (  # noqa: E402
    assert_stats_match_contract,
    load_contract,
    load_normalizer,
)
from scripts.b1k_2026.contract_checks import (  # noqa: E402
    assert_model_schedulers_match_contract,
    assert_model_target_matches_contract,
    assert_model_temporal_contract_matches_contract,
    inference_metadata,
)
from fastwam.b1k_execution import (  # noqa: E402
    B1KExecutionConfig,
    B1KExecutionState,
    prepare_b1k_execution,
)
from challenge_docker.task_guard import guard_metadata, validate_observation_task  # noqa: E402


# Tasks that are executed at the demonstrated speed (1x) instead of the default 2x:
# 55 re_shelving_library_books, 77 installing_a_modem, 95 cook_brussels_sprouts.
# Every other task uses the server-wide execution settings (default: speed_2x, execute 32).
# Set FASTWAM_DISABLE_TASK_OVERRIDES=1 to run all tasks with the server-wide settings.
ONE_X_TASK_IDS = (55, 77, 95)
ONE_X_EXECUTION = {
    "temporal_strategy": "baseline",
    "resampling": "hybrid_linear",
    "execute_horizon": 16,
}


def _load_callable(spec: str, *, extra_path: str | None = None) -> Callable[..., Any]:
    if extra_path:
        path = str(Path(extra_path).expanduser().resolve())
        if path not in sys.path:
            sys.path.insert(0, path)
    module_name, separator, attribute = spec.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError(f"Callable must use module:attribute syntax, got {spec!r}.")
    value: Any = importlib.import_module(module_name)
    for part in attribute.split("."):
        value = getattr(value, part)
    if not callable(value):
        raise TypeError(f"Configured policy {spec!r} is not callable.")
    return value


def _task_lookup(path: Path) -> dict[int, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    tasks = payload.get("tasks") if isinstance(payload, Mapping) else None
    if not isinstance(tasks, list):
        raise ValueError(f"Task data must contain a tasks list: {path}")
    result: dict[int, str] = {}
    for index, item in enumerate(tasks):
        if not isinstance(item, Mapping) or not isinstance(item.get("id"), str):
            raise ValueError(f"Invalid task entry {index} in {path}")
        result[index] = str(item["id"])
    return result


class ChallengePolicy:
    def __init__(
        self,
        *,
        model: Any,
        policy_fn: Callable[..., Any],
        normalizer: Any,
        task_lookup: Mapping[int, str],
        require_depth: bool,
        seed: int,
        execution_config: B1KExecutionConfig,
        task_execution_configs: Mapping[int, B1KExecutionConfig] | None = None,
    ) -> None:
        self.model = model
        self.policy_fn = policy_fn
        self.normalizer = normalizer
        self.task_lookup = task_lookup
        self.require_depth = bool(require_depth)
        self.seed = int(seed)
        self.step = 0
        self.queue: deque[np.ndarray] = deque()
        self.execution_config = execution_config
        self.task_execution_configs = dict(task_execution_configs or {})
        self.execution_state = B1KExecutionState()
        self.chunk_index = 0

    def reset(self) -> None:
        self.queue.clear()
        self.step = 0
        self.execution_state = B1KExecutionState()
        self.chunk_index = 0

    def act(self, observation: Mapping[str, Any]) -> np.ndarray:
        validate_observation_task(observation)
        if not self.queue:
            adapted, _, normalized, _ = adapt_observation(
                observation,
                task_lookup=self.task_lookup,
                proprio_mean=self.normalizer.proprio_mean.numpy(),
                proprio_std=self.normalizer.proprio_std.numpy(),
                require_depth=self.require_depth,
            )
            with torch.inference_mode():
                normalized_action = self.policy_fn(
                    self.model,
                    observation=adapted,
                    normalized_proprio=normalized,
                    step=self.step,
                    seed=self.seed + self.step,
                )
            normalized_action = torch.as_tensor(normalized_action, dtype=torch.float32)
            if normalized_action.ndim != 2 or normalized_action.shape[1] != 23:
                raise ValueError(
                    "FastWAM policy must return a [action_horizon,23] normalized chunk, "
                    f"got {tuple(normalized_action.shape)}."
                )
            # Match the training postprocess before denormalization.  Keeping
            # this in the container ensures every mounted checkpoint uses the
            # same binary gripper and constant-dimension rules.
            from fastwam.datasets.b1k.contract import postprocess_b1k_action, validate_b1k_action

            canonical = postprocess_b1k_action(normalized_action)
            raw_action = self.normalizer.denormalize_action(canonical)
            validate_b1k_action(raw_action)
            task_id = int(np.asarray(observation["task_id"]).reshape(-1)[0])
            execution = prepare_b1k_execution(
                raw_action,
                self.task_execution_configs.get(task_id, self.execution_config),
                chunk_index=self.chunk_index,
                state=self.execution_state,
            )
            self.execution_state = execution.next_state
            self.chunk_index += 1
            ticks = execution.ticks
            if hasattr(ticks, "detach"):
                ticks = ticks.detach().cpu().numpy()
            self.queue.extend(np.asarray(row, dtype=np.float32) for row in ticks)
            self.step += 1
        action = np.asarray(self.queue.popleft(), dtype=np.float32)
        if action.shape != (23,) or not np.isfinite(action).all():
            raise ValueError(f"Policy produced invalid action {action.shape}.")
        return action


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--contract", type=Path, default=None)
    parser.add_argument("--stats", type=Path, default=None)
    parser.add_argument("--task-data", type=Path, default=None)
    parser.add_argument("--host", default=os.environ.get("FASTWAM_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("FASTWAM_PORT", "8000")))
    parser.add_argument("--policy", default=os.environ.get("FASTWAM_POLICY", "experiments.b1k_2026.policy_adapter:policy"))
    parser.add_argument("--policy-path", default=os.environ.get("FASTWAM_POLICY_PATH"))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("FASTWAM_SEED", "20260719")))
    parser.add_argument(
        "--temporal-strategy",
        default=os.environ.get("FASTWAM_TEMPORAL_STRATEGY", "speed_2x"),
    )
    parser.add_argument(
        "--execute-horizon",
        type=int,
        default=int(os.environ.get("FASTWAM_EXECUTE_HORIZON", "32")),
    )
    parser.add_argument(
        "--resampling",
        default=os.environ.get("FASTWAM_RESAMPLING", "zero_order_hold"),
    )
    parser.add_argument("--telemetry", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = (args.checkpoint or Path(os.environ.get("FASTWAM_CHECKPOINT", ""))).expanduser().resolve()
    contract_path = (args.contract or Path(os.environ.get("FASTWAM_CONTRACT", ""))).expanduser().resolve()
    stats_path = (args.stats or Path(os.environ.get("FASTWAM_STATS", ""))).expanduser().resolve()
    task_data = (args.task_data or Path(os.environ.get("FASTWAM_TASK_DATA", str(PROJECT_ROOT / "challenge_docker" / "task_data.json")))).expanduser().resolve()
    for name, path in (("checkpoint", checkpoint), ("contract", contract_path), ("stats", stats_path), ("task data", task_data)):
        if not path.is_file():
            raise FileNotFoundError(f"Missing runtime {name}: {path}")

    contract = load_contract(contract_path)
    assert_stats_match_contract(stats_path, contract)
    from experiments.b1k_2026.model_factory import model_factory

    os.environ.setdefault("FASTWAM_B1K_INFERENCE_STEPS", "20")
    model = model_factory()
    assert_model_target_matches_contract(model, contract)
    assert_model_schedulers_match_contract(model, contract)
    assert_model_temporal_contract_matches_contract(model, contract)
    model.load_checkpoint(checkpoint, expected_contract=contract, contract_mode="strict")
    model.eval()
    policy_fn = _load_callable(args.policy, extra_path=args.policy_path)
    normalizer, _ = load_normalizer(stats_path)
    lookup = _task_lookup(task_data)
    requires_depth = bool(getattr(model, "_fastwam_b1k_requires_depth", False))
    action_horizon = inference_metadata(model)["action_horizon"]
    execution_config = B1KExecutionConfig(
        prediction_horizon=action_horizon,
        execute_horizon=args.execute_horizon,
        temporal_strategy=args.temporal_strategy,
        resampling=args.resampling,
    )
    task_execution_configs = {}
    if os.environ.get("FASTWAM_DISABLE_TASK_OVERRIDES", "") not in ("1", "true", "True"):
        task_execution_configs = {
            task_id: B1KExecutionConfig(prediction_horizon=action_horizon, **ONE_X_EXECUTION)
            for task_id in ONE_X_TASK_IDS
        }
    policy_factory = functools.partial(ChallengePolicy,
        model=model,
        policy_fn=policy_fn,
        normalizer=normalizer,
        task_lookup=lookup,
        require_depth=requires_depth,
        seed=args.seed,
        execution_config=execution_config,
        task_execution_configs=task_execution_configs,
    )
    metadata = {
        "protocol": "behavior_1k_websocket.v393",
        "checkpoint": str(checkpoint),
        "contract_sha256": contract.contract_sha256,
        "stats_sha256": assert_stats_match_contract(stats_path, contract),
        "requires_depth": requires_depth,
        "action_dim": 23,
        "action_chunking": "official_client_K32",
        "replan_horizon": inference_metadata(model)["action_horizon"],
        "inference_steps": inference_metadata(model)["inference_steps"],
        "execution": execution_config.metadata(),
        "task_execution": {str(k): v.metadata() for k, v in sorted(task_execution_configs.items())},
        "device": str(next(model.parameters()).device),
    }
    def fingerprint(path):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    metadata.update(checkpoint_sha256=fingerprint(checkpoint),
                    config_sha256=contract.contract_sha256, action_chunk_size=32,
                    gpu_name=torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
                    num_video_frames=inference_metadata(model)["num_video_frames"],
                    source_revision=os.environ.get("FASTWAM_SOURCE_REV", "unknown"))
    metadata["task_guard"] = guard_metadata()
    WebsocketPolicyServer(policy_factory, host=args.host, port=args.port, metadata=metadata,
                          telemetry=args.telemetry).serve_forever()


if __name__ == "__main__":
    main()
