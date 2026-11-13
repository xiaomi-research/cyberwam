# Copyright (c) 2026 Xiaomi Corporation

import hashlib
import json

import pytest
import torch
from omegaconf import OmegaConf

from fastwam.datasets.b1k.dataset import B1KRobotVideoDataset
from fastwam.datasets.b1k.text import format_b1k_prompt
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from scripts import precompute_text_embeds


def _write_tasks(dataset_dir, tasks):
    meta_dir = dataset_dir / "meta"
    meta_dir.mkdir(parents=True)
    with (meta_dir / "tasks.jsonl").open("w", encoding="utf-8") as stream:
        for task_index, task in enumerate(tasks):
            stream.write(json.dumps({"task_index": task_index, "task": task}) + "\n")


def test_b1k_dataset_override_and_runtime_prompts_have_identical_hashes(tmp_path):
    tasks = [f"task_{task_index:03d}_with_robot_motion" for task_index in range(100)]
    dataset_dir = tmp_path / "b1k"
    cache_dir = tmp_path / "text_cache"
    _write_tasks(dataset_dir, tasks)
    data_cfg = OmegaConf.create(
        {
            "dataset": {
                "_target_": "fastwam.datasets.b1k.dataset.B1KRobotVideoDataset",
                "dataset_root": str(dataset_dir),
                "text_embedding_cache_dir": str(cache_dir),
                "context_len": 128,
            }
        }
    )

    dataset_dirs, cache_dirs, context_lens, b1k_dirs = (
        precompute_text_embeds._collect_dataset_settings(data_cfg)
    )
    cached_prompts = precompute_text_embeds._read_unique_prompts(dataset_dirs, b1k_dirs)

    assert cache_dirs == [cache_dir]
    assert context_lens == {128}
    assert b1k_dirs == {str(dataset_dir)}
    assert len(cached_prompts) == 100
    for task, cached_prompt in zip(tasks, cached_prompts, strict=True):
        runtime_prompt = B1KRobotVideoDataset._format_prompt(None, task)
        assert runtime_prompt == format_b1k_prompt(task)
        override_prompt = precompute_text_embeds._get_override_prompt(
            task, use_b1k_prompt=True
        )
        expected_hash = hashlib.sha256(runtime_prompt.encode("utf-8")).hexdigest()
        assert cached_prompt == runtime_prompt == override_prompt
        assert precompute_text_embeds._prompt_sha256(cached_prompt) == expected_hash
        assert precompute_text_embeds._prompt_sha256(override_prompt) == expected_hash


def test_b1k_requires_context_len_128(tmp_path):
    data_cfg = OmegaConf.create(
        {
            "_target_": "fastwam.datasets.b1k.dataset.B1KRobotVideoDataset",
            "dataset_root": str(tmp_path / "b1k"),
            "text_embedding_cache_dir": str(tmp_path / "cache"),
            "context_len": 64,
        }
    )

    with pytest.raises(ValueError, match="must set context_len=128"):
        precompute_text_embeds._collect_dataset_settings(data_cfg)


def test_cache_payload_preserves_real_attention_mask():
    context = torch.randn(2, 128, 4)
    mask = torch.zeros(2, 128, dtype=torch.bool)
    mask[0, :17] = True
    mask[1, :29] = True

    payload = precompute_text_embeds._make_cache_payload(context, mask, index=1)

    assert payload["context"].shape == (128, 4)
    assert payload["context"].dtype == torch.bfloat16
    assert payload["mask"].shape == (128,)
    assert payload["mask"].dtype == torch.bool
    assert torch.equal(payload["mask"], mask[1].cpu())
    assert payload["mask"].sum().item() == 29


def test_non_b1k_prompt_behavior_is_unchanged(tmp_path):
    dataset_dir = tmp_path / "libero"
    _write_tasks(dataset_dir, ["pick_up_the_mug"])

    prompts = precompute_text_embeds._read_unique_prompts([str(dataset_dir)])

    assert prompts == [DEFAULT_PROMPT.format(task="pick_up_the_mug")]
    assert precompute_text_embeds._get_override_prompt("pick_up_the_mug") == (
        DEFAULT_PROMPT.format(task="pick_up_the_mug")
    )
