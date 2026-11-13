# Copyright (c) 2026 Xiaomi Corporation

"""Checks that a built model matches its checkpoint contract (target, schedulers, temporal settings)."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys
from collections.abc import Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for path in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))



def assert_model_target_matches_contract(model, contract) -> None:
    actual = getattr(model, "_fastwam_b1k_model_target", None)
    try:
        expected = contract.compatibility["model"]["target"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Checkpoint contract is missing compatibility.model.target.") from exc
    if actual != expected:
        raise ValueError(
            "Configured B1K model target does not match checkpoint contract: "
            f"configured {actual!r}, checkpoint requires {expected!r}."
        )


def assert_model_schedulers_match_contract(model, contract) -> None:
    """Reject inference when the instantiated schedulers differ from the checkpoint contract."""
    try:
        schedulers = contract.compatibility["scheduler"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Checkpoint contract is missing compatibility.scheduler.") from exc
    if not isinstance(schedulers, Mapping):
        raise ValueError("Checkpoint contract compatibility.scheduler must be a mapping.")

    differences: list[str] = []
    for stream in ("video", "action", "dino", "pointmap"):
        expected = schedulers.get(stream)
        if expected is None:
            continue
        if not isinstance(expected, Mapping):
            differences.append(f"{stream}: contract entry is not a mapping")
            continue
        for phase in ("train", "infer"):
            scheduler = getattr(model, f"{phase}_{stream}_scheduler", None)
            if scheduler is None:
                differences.append(f"{stream}.{phase}: runtime scheduler is missing")
                continue
            shift_key = f"{phase}_shift"
            if shift_key in expected:
                actual_shift = getattr(scheduler, "shift", None)
                expected_shift = expected[shift_key]
                if (
                    not isinstance(actual_shift, (int, float))
                    or isinstance(actual_shift, bool)
                    or not math.isclose(float(actual_shift), float(expected_shift), rel_tol=0.0, abs_tol=1e-12)
                ):
                    differences.append(
                        f"{stream}.{shift_key}: expected {expected_shift!r}, got {actual_shift!r}"
                    )
            if "num_train_timesteps" in expected:
                actual_steps = getattr(scheduler, "num_train_timesteps", None)
                expected_steps = expected["num_train_timesteps"]
                if actual_steps != expected_steps:
                    differences.append(
                        f"{stream}.num_train_timesteps: expected {expected_steps!r}, got {actual_steps!r}"
                    )

        if "shared_train_timestep" in expected:
            actual_shared = getattr(model, "shared_train_timestep", None)
            if actual_shared != expected["shared_train_timestep"]:
                differences.append(
                    "video.shared_train_timestep: "
                    f"expected {expected['shared_train_timestep']!r}, got {actual_shared!r}"
                )

    if differences:
        detail = "; ".join(differences)
        raise ValueError(f"Runtime scheduler does not match checkpoint contract: {detail}")


def assert_model_temporal_contract_matches_contract(model, contract) -> None:
    """Reject inference when runtime action/video lengths differ from training data."""
    try:
        data = contract.compatibility["data"]
        action = data["action"]
        video = data["video"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "Checkpoint contract is missing compatibility.data action/video metadata."
        ) from exc
    if not isinstance(action, Mapping) or not isinstance(video, Mapping):
        raise ValueError(
            "Checkpoint contract compatibility.data.action/video must be mappings."
        )

    expected_action_horizon = action.get("horizon")
    if isinstance(expected_action_horizon, bool) or not isinstance(
        expected_action_horizon, int
    ) or expected_action_horizon <= 0:
        raise ValueError(
            "Checkpoint contract compatibility.data.action.horizon must be a positive integer."
        )

    frame_sequences = [
        video.get("dataset_image_indices"),
        video.get("raw_frame_indices"),
    ]
    frame_lengths = {
        len(sequence)
        for sequence in frame_sequences
        if isinstance(sequence, (list, tuple))
    }
    if not frame_lengths or len(frame_lengths) != 1:
        raise ValueError(
            "Checkpoint contract video frame metadata must contain consistent frame lists."
        )
    expected_video_frames = frame_lengths.pop()
    if expected_video_frames <= 0:
        raise ValueError("Checkpoint contract video frame metadata must not be empty.")

    actual = inference_metadata(model)
    differences = []
    if actual["action_horizon"] != expected_action_horizon:
        differences.append(
            "action_horizon: "
            f"expected {expected_action_horizon}, got {actual['action_horizon']}"
        )
    if actual["num_video_frames"] != expected_video_frames:
        differences.append(
            "num_video_frames: "
            f"expected {expected_video_frames}, got {actual['num_video_frames']}"
        )
    if differences:
        raise ValueError(
            "Runtime temporal contract does not match checkpoint contract: "
            + "; ".join(differences)
        )


def inference_metadata(model) -> dict[str, int]:
    """Expose the resolved temporal contract for pre-rollout validation."""
    return {
        "action_horizon": int(
            os.environ.get(
                "FASTWAM_B1K_ACTION_HORIZON",
                getattr(model, "_fastwam_b1k_action_horizon", 32),
            )
        ),
        "num_video_frames": int(
            os.environ.get(
                "FASTWAM_B1K_NUM_VIDEO_FRAMES",
                getattr(model, "_fastwam_b1k_num_video_frames", 9),
            )
        ),
        "inference_steps": int(os.environ.get("FASTWAM_B1K_INFERENCE_STEPS", "10")),
    }


def apply_contract_scheduler_overrides(contract) -> None:
    """Compose the inference model with the scheduler settings used in training.

    The evaluation Hydra task is intentionally generic, while a checkpoint
    contract is authoritative for scheduler details such as shared training
    timesteps.  Inject those values before ``model_factory`` instantiates the
    model so strict validation compares like with like.
    """
    compatibility = getattr(contract, "compatibility", None)
    schedulers = compatibility.get("scheduler") if isinstance(compatibility, Mapping) else None
    if not isinstance(schedulers, Mapping):
        raise ValueError("Checkpoint contract compatibility.scheduler must be a mapping.")

    overrides: list[str] = []
    for stream in ("video", "action"):
        expected = schedulers.get(stream)
        if not isinstance(expected, Mapping):
            continue
        for key in ("train_shift", "infer_shift", "num_train_timesteps", "shared_train_timestep"):
            if key in expected:
                overrides.append(
                    f"++model.{stream}_scheduler.{key}={json.dumps(expected[key])}"
                )

    raw = os.environ.get("FASTWAM_B1K_CONFIG_OVERRIDES", "").strip()
    try:
        existing = json.loads(raw) if raw else []
    except json.JSONDecodeError as exc:
        raise ValueError(
            "FASTWAM_B1K_CONFIG_OVERRIDES must be a JSON list of Hydra override strings."
        ) from exc
    if not isinstance(existing, list) or not all(isinstance(item, str) for item in existing):
        raise ValueError(
            "FASTWAM_B1K_CONFIG_OVERRIDES must be a JSON list of Hydra override strings."
        )
    os.environ["FASTWAM_B1K_CONFIG_OVERRIDES"] = json.dumps([*existing, *overrides])
