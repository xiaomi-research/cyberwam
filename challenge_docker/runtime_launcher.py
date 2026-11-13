#!/usr/bin/env python3
# Copyright (c) 2026 Xiaomi Corporation

"""Resolve a mounted model directory, then exec the normal policy server."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

from runtime_contract import validate_bundle


def main() -> int:
    args = list(sys.argv[1:])
    preflight_only = "--fastwam-preflight-only" in args
    args = [arg for arg in args if arg != "--fastwam-preflight-only"]

    root = Path(os.environ.get("FASTWAM_MODEL_DIR", "/runtime/model")).expanduser()
    try:
        info = validate_bundle(root)
    except Exception as exc:
        print(f"FastWAM model bundle preflight failed: {exc}", file=sys.stderr)
        return 2

    os.environ["FASTWAM_CHECKPOINT"] = str(info["checkpoint"])
    os.environ["FASTWAM_CONTRACT"] = str(info["contract"])
    os.environ["FASTWAM_STATS"] = str(info["stats"])
    os.environ["FASTWAM_B1K_HYDRA_TASK"] = str(info["hydra_task"])
    if info.get("model_config_path") is None:
        os.environ.pop("FASTWAM_MODEL_CONFIG", None)
    else:
        os.environ["FASTWAM_MODEL_CONFIG"] = str(info["model_config_path"])
    # The common image keeps the DINO asset path for Flex models.  The model
    # factory applies this environment override generically, but the Joint
    # constructor does not accept any DINO arguments, so remove it before
    # composing a Joint config.
    if info["model_target"] == "fastwam.runtime.create_fastwam_joint":
        os.environ.pop("FASTWAM_B1K_DINO_PATH", None)
    os.environ.setdefault("FASTWAM_TASK_DATA", str(Path(__file__).resolve().with_name("task_data.json")))
    if not Path(os.environ["FASTWAM_TASK_DATA"]).is_file():
        print(
            f"FastWAM task data does not exist: {os.environ['FASTWAM_TASK_DATA']}",
            file=sys.stderr,
        )
        return 2

    if preflight_only:
        print(
            json.dumps(
                {
                    "model_dir": str(root),
                    "checkpoint": str(info["checkpoint"]),
                    "contract_sha256": info["contract_sha256"],
                    "stats_sha256": info["stats_sha256"],
                    "model_target": info["model_target"],
                    "hydra_task": info["hydra_task"],
                    "model_config": info["model_config"],
                },
                sort_keys=True,
            )
        )
        return 0

    python = os.environ.get("FASTWAM_IMAGE_PYTHON", "python")
    os.execvpe(
        python,
        [python, "-m", "challenge_docker.server", *args],
        os.environ,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
