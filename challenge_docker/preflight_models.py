#!/usr/bin/env python3
# Copyright (c) 2026 Xiaomi Corporation

"""Validate all immediate model directories before a GPU startup."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from runtime_contract import validate_bundle


def _bundle_dirs(root: Path) -> list[Path]:
    if (root / "checkpoint_contract.json").is_file() or (root / "norm_stats.json").is_file():
        return [root]
    return sorted(path for path in root.iterdir() if path.is_dir())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        print(f"model root does not exist: {root}", file=sys.stderr)
        return 2
    rows = []
    failed = False
    for bundle in _bundle_dirs(root):
        try:
            info = validate_bundle(bundle)
            rows.append(
                {
                    "name": bundle.name,
                    "checkpoint_bytes": info["checkpoint"].stat().st_size,
                    "contract_sha256": info["contract_sha256"],
                    "stats_sha256": info["stats_sha256"],
                    "hydra_task": info["hydra_task"],
                    "model_config": info["model_config"],
                    "ok": True,
                }
            )
        except Exception as exc:
            failed = True
            rows.append({"name": bundle.name, "ok": False, "error": str(exc)})
    if args.as_json:
        print(json.dumps(rows, indent=2, sort_keys=True))
    else:
        for row in rows:
            if row["ok"]:
                print(
                    f"OK {row['name']}: {row['hydra_task']} "
                    f"contract={row['contract_sha256'][:12]} "
                    f"checkpoint={row['checkpoint_bytes']} bytes"
                )
            else:
                print(f"FAIL {row['name']}: {row['error']}", file=sys.stderr)
    if not rows:
        print(f"no model directories found under {root}", file=sys.stderr)
        return 2
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
