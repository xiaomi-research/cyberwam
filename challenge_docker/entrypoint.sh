#!/usr/bin/env bash
# Copyright (c) 2026 Xiaomi Corporation

set -euo pipefail

# Find the launcher next to this script, so the entrypoint works from /opt/fastwam (the image) and from any checkout.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "${HERE}")"
export PYTHONPATH="${ROOT}:${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

exec "${FASTWAM_IMAGE_PYTHON:-python}" "${HERE}/runtime_launcher.py" "$@"
