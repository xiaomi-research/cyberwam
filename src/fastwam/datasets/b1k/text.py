# Copyright (c) 2026 Xiaomi Corporation

from __future__ import annotations

import re


B1K_PROMPT_TEMPLATE = (
    "A video recorded from a robot's point of view executing the following instruction: {task}"
)


def normalize_b1k_task_text(task: str) -> str:
    if not isinstance(task, str):
        raise TypeError(f"B1K task must be a string, got {type(task)!r}.")
    normalized = re.sub(r"\s+", " ", task.replace("_", " ")).strip()
    if not normalized:
        raise ValueError("B1K task text must not be empty.")
    return normalized


def format_b1k_prompt(task: str) -> str:
    return B1K_PROMPT_TEMPLATE.format(task=normalize_b1k_task_text(task))
