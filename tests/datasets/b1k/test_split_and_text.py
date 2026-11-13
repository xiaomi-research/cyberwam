# Copyright (c) 2026 Xiaomi Corporation

import json

import pytest

from fastwam.datasets.b1k.split import build_b1k_episode_split, build_b1k_episode_splits
from fastwam.datasets.b1k.text import format_b1k_prompt, normalize_b1k_task_text


def _write_episodes(root, groups):
    meta = root / "meta"
    meta.mkdir(parents=True)
    with (meta / "episodes.jsonl").open("w", encoding="utf-8") as stream:
        for task_index, episodes in groups.items():
            for episode_index in reversed(episodes):
                stream.write(json.dumps({"episode_index": episode_index, "task_index": task_index}) + "\n")


def test_split_is_per_task_sorted_and_disjoint(tmp_path):
    _write_episodes(tmp_path, {0: [10, 12, 14, 16], 1: [21, 23, 25, 27]})
    splits = build_b1k_episode_splits(tmp_path, expected_tasks=2, episodes_per_task=4)
    assert splits == {
        "train": [10, 12, 21, 23],
        "validation": [14, 25],
        "test": [16, 27],
    }
    assert build_b1k_episode_split(
        tmp_path, "val", expected_tasks=2, episodes_per_task=4
    ) == [14, 25]
    assert set(splits["train"]).isdisjoint(splits["validation"])
    assert set(splits["train"]).isdisjoint(splits["test"])
    assert set(splits["validation"]).isdisjoint(splits["test"])


def test_split_rejects_duplicate_and_wrong_task_size(tmp_path):
    _write_episodes(tmp_path, {0: [0, 1, 2], 1: [2, 3, 4]})
    with pytest.raises(ValueError, match="Duplicate"):
        build_b1k_episode_splits(tmp_path, expected_tasks=2, episodes_per_task=3)

    other = tmp_path / "other"
    _write_episodes(other, {0: [0, 1, 2], 1: [3, 4]})
    with pytest.raises(ValueError, match="must have 3 episodes"):
        build_b1k_episode_splits(other, expected_tasks=2, episodes_per_task=3)


def test_task_text_and_prompt_are_canonical():
    assert normalize_b1k_task_text("  putting__away_Halloween_decorations ") == (
        "putting away Halloween decorations"
    )
    assert format_b1k_prompt("turning_on_radio") == (
        "A video recorded from a robot's point of view executing the following instruction: "
        "turning on radio"
    )
    with pytest.raises(ValueError, match="must not be empty"):
        normalize_b1k_task_text("___")
