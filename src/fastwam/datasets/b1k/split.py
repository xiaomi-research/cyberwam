# Copyright (c) 2026 Xiaomi Corporation

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Literal


B1KSplit = Literal["train", "validation", "val", "test"]


def _load_task_episode_groups(
    dataset_root: str | Path,
    *,
    expected_tasks: int,
    episodes_per_task: int,
) -> dict[int, list[int]]:
    episodes_path = Path(dataset_root) / "meta" / "episodes.jsonl"
    if not episodes_path.is_file():
        raise FileNotFoundError(f"Missing B1K episode metadata: {episodes_path}")

    groups: dict[int, list[int]] = defaultdict(list)
    seen_episodes: set[int] = set()
    with episodes_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                episode_index = int(record["episode_index"])
                task_index = int(record["task_index"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid B1K episode metadata at {episodes_path}:{line_number}.") from exc
            if episode_index < 0 or task_index < 0:
                raise ValueError(
                    f"Negative episode/task index at {episodes_path}:{line_number}: "
                    f"episode={episode_index}, task={task_index}."
                )
            if episode_index in seen_episodes:
                raise ValueError(f"Duplicate B1K episode_index {episode_index} in {episodes_path}.")
            seen_episodes.add(episode_index)
            groups[task_index].append(episode_index)

    if len(groups) != expected_tasks:
        raise ValueError(f"Expected {expected_tasks} B1K tasks, found {len(groups)} in {episodes_path}.")
    expected_task_ids = set(range(expected_tasks))
    if set(groups) != expected_task_ids:
        raise ValueError(
            f"B1K task_index values must be contiguous 0..{expected_tasks - 1}, "
            f"got {sorted(groups)}."
        )

    for task_index, episode_indices in groups.items():
        episode_indices.sort()
        if len(episode_indices) != episodes_per_task:
            raise ValueError(
                f"B1K task {task_index} must have {episodes_per_task} episodes, "
                f"found {len(episode_indices)}."
            )
    return dict(groups)


def build_b1k_episode_splits(
    dataset_root: str | Path,
    *,
    expected_tasks: int = 100,
    episodes_per_task: int = 200,
) -> dict[str, list[int]]:
    if episodes_per_task < 3:
        raise ValueError("episodes_per_task must be at least 3 for train/validation/test splits.")
    groups = _load_task_episode_groups(
        dataset_root,
        expected_tasks=expected_tasks,
        episodes_per_task=episodes_per_task,
    )
    splits = {"train": [], "validation": [], "test": []}
    for task_index in range(expected_tasks):
        episodes = groups[task_index]
        splits["train"].extend(episodes[:-2])
        splits["validation"].append(episodes[-2])
        splits["test"].append(episodes[-1])

    all_episodes = splits["train"] + splits["validation"] + splits["test"]
    if len(all_episodes) != len(set(all_episodes)):
        raise AssertionError("B1K episode splits overlap.")
    return splits


def build_b1k_episode_split(
    dataset_root: str | Path,
    split: B1KSplit,
    *,
    expected_tasks: int = 100,
    episodes_per_task: int = 200,
) -> list[int]:
    normalized_split = "validation" if split == "val" else split
    if normalized_split not in {"train", "validation", "test"}:
        raise ValueError(f"Unsupported B1K split {split!r}.")
    return build_b1k_episode_splits(
        dataset_root,
        expected_tasks=expected_tasks,
        episodes_per_task=episodes_per_task,
    )[normalized_split]
