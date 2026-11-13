# Copyright (c) 2026 Xiaomi Corporation

from itertools import islice

import pytest

from fastwam.utils.samplers import ResumableEpochSampler


class SizedDataset:
    def __init__(self, size: int):
        self.size = size

    def __len__(self) -> int:
        return self.size


def test_epoch_permutation_is_deterministic_and_bijective():
    first = ResumableEpochSampler(SizedDataset(101), seed=17, batch_size=1, num_processes=1)
    second = ResumableEpochSampler(SizedDataset(101), seed=17, batch_size=1, num_processes=1)

    epoch_0 = list(first)
    epoch_1 = list(first)

    assert epoch_0 == list(second)
    assert epoch_1 == list(second)
    assert epoch_0 != epoch_1
    assert sorted(epoch_0) == list(range(101))
    assert sorted(epoch_1) == list(range(101))


def test_distributed_batch_shards_are_disjoint_and_tail_is_dropped():
    batch_size = 3
    num_processes = 8
    sampler = ResumableEpochSampler(
        SizedDataset(107), seed=9, batch_size=batch_size, num_processes=num_processes
    )

    indices = list(sampler)
    assert len(indices) == 96
    assert len(indices) == len(set(indices))

    rank_samples = [[] for _ in range(num_processes)]
    global_batch_size = batch_size * num_processes
    for batch_start in range(0, len(indices), global_batch_size):
        global_batch = indices[batch_start : batch_start + global_batch_size]
        for rank in range(num_processes):
            start = rank * batch_size
            rank_samples[rank].extend(global_batch[start : start + batch_size])

    for rank, samples in enumerate(rank_samples):
        other_samples = set().union(
            *(set(values) for other_rank, values in enumerate(rank_samples) if other_rank != rank)
        )
        assert set(samples).isdisjoint(other_samples)


def test_resume_starts_at_exact_next_global_batch_and_len_is_remaining():
    dataset = SizedDataset(103)
    kwargs = dict(seed=123, batch_size=2, num_processes=4)
    uninterrupted = list(ResumableEpochSampler(dataset, **kwargs))

    resumed = ResumableEpochSampler(dataset, **kwargs)
    resumed.set_epoch_offset(5)
    epoch_five = list(resumed)

    resumed = ResumableEpochSampler(dataset, **kwargs)
    resumed.set_epoch_offset(5)
    resumed.set_resume_batch_offset(3)
    assert len(resumed) == 72
    assert list(resumed) == epoch_five[24:]
    assert len(resumed) == 96

    # The offset applies only to the first iterator after restore.
    assert len(list(resumed)) == 96
    assert uninterrupted != epoch_five


def test_large_dataset_starts_in_constant_memory_without_randperm(monkeypatch):
    import torch

    def fail_randperm(*args, **kwargs):
        raise AssertionError("sampler must not materialize torch.randperm")

    monkeypatch.setattr(torch, "randperm", fail_randperm)
    sampler = ResumableEpochSampler(
        SizedDataset(200_000_003), seed=42, batch_size=2, num_processes=8
    )

    assert len(sampler) == 200_000_000
    prefix = list(islice(iter(sampler), 32))
    assert len(prefix) == len(set(prefix)) == 32


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"batch_size": 0, "num_processes": 1}, "batch_size"),
        ({"batch_size": 1, "num_processes": 0}, "num_processes"),
    ],
)
def test_invalid_parallelism_is_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ResumableEpochSampler(SizedDataset(10), seed=0, **kwargs)


def test_invalid_resume_offset_is_rejected():
    sampler = ResumableEpochSampler(SizedDataset(10), seed=0, batch_size=2, num_processes=2)
    with pytest.raises(ValueError, match="exceeds epoch length"):
        sampler.set_resume_batch_offset(3)


def test_preview_indices_matches_resume_without_mutating_sampler():
    dataset = SizedDataset(100)
    sampler = ResumableEpochSampler(dataset, seed=7, batch_size=2, num_processes=4)
    before = (sampler.epoch, sampler.epoch_offset, sampler.resume_batch_offset)
    preview = sampler.preview_indices(epoch=3, batch_in_epoch=2, count=12)
    assert (sampler.epoch, sampler.epoch_offset, sampler.resume_batch_offset) == before

    resumed = ResumableEpochSampler(dataset, seed=7, batch_size=2, num_processes=4)
    resumed.set_epoch_offset(3)
    resumed.set_resume_batch_offset(2)
    assert list(islice(iter(resumed), 12)) == preview
