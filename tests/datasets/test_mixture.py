# Copyright (c) 2026 Xiaomi Corporation

from collections import Counter
from copy import deepcopy

import pytest
from torch.utils.data import Dataset

from fastwam.datasets.mixture import WeightedMixtureDataset
from fastwam.utils.samplers import ResumableEpochSampler


class _Dataset(Dataset):
    def __init__(self, name: str, size: int, contract=None):
        self.name = name
        self.size = size
        self._contract = contract
        self.batch_calls = 0

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.name, index

    def __getitems__(self, indices):
        self.batch_calls += 1
        return [self[index] for index in indices]

    @property
    def checkpoint_data_contract(self):
        return self._contract


def test_weighted_mixture_has_exact_source_quotas_and_batched_reads():
    a = _Dataset("a", 100)
    b = _Dataset("b", 20)
    mixture = WeightedMixtureDataset([a, b], [0.7, 0.3], epoch_size=100, seed=4)

    samples = mixture.__getitems__(list(range(100)))

    assert Counter(name for name, _ in samples) == {"a": 70, "b": 30}
    assert mixture.source_counts == (70, 30)
    assert a.batch_calls == b.batch_calls == 1


def test_sampler_changes_source_samples_by_epoch_and_resumes_exactly():
    mixture = WeightedMixtureDataset(
        [_Dataset("a", 1000), _Dataset("b", 1000)],
        [0.7, 0.3],
        epoch_size=100,
        seed=11,
    )
    kwargs = dict(dataset=mixture, seed=7, batch_size=2, num_processes=2)
    sampler = ResumableEpochSampler(**kwargs)
    epoch_zero = list(sampler)
    epoch_one = list(sampler)
    assert {epoch for epoch, _, _ in epoch_zero} == {0}
    assert {epoch for epoch, _, _ in epoch_one} == {1}
    assert [mixture[index] for index in epoch_zero] != [mixture[index] for index in epoch_one]

    resumed = ResumableEpochSampler(**kwargs)
    resumed.set_epoch_offset(1)
    resumed.set_resume_batch_offset(3)
    assert list(resumed) == epoch_one[12:]
    assert resumed.preview_indices(epoch=1, batch_in_epoch=3, count=8) == epoch_one[12:20]


def test_every_rank_local_batch_tracks_freely_configured_weights():
    mixture = WeightedMixtureDataset(
        [_Dataset("a", 1000), _Dataset("b", 1000)],
        [0.7, 0.3],
        epoch_size=32 * 4 * 10,
        seed=11,
    )
    indices = list(
        ResumableEpochSampler(mixture, seed=7, batch_size=32, num_processes=4)
    )
    counts = Counter()
    local_counts = []
    for start in range(0, len(indices), 32):
        batch = [mixture[index][0] for index in indices[start : start + 32]]
        batch_counts = Counter(batch)
        local_counts.append((batch_counts["a"], batch_counts["b"]))
        counts.update(batch)

    assert set(local_counts) == {(22, 10), (23, 9)}
    assert counts == {"a": 896, "b": 384}


def test_batch_mixing_supports_more_than_two_freely_weighted_sources():
    mixture = WeightedMixtureDataset(
        [_Dataset("a", 100), _Dataset("b", 100), _Dataset("c", 100)],
        [5, 3, 2],
        epoch_size=20,
    )
    indices = list(
        ResumableEpochSampler(mixture, seed=7, batch_size=10, num_processes=1)
    )

    for start in range(0, len(indices), 10):
        assert Counter(mixture[index][0] for index in indices[start : start + 10]) == {
            "a": 5,
            "b": 3,
            "c": 2,
        }


def test_mixture_rejects_incompatible_contracts():
    with pytest.raises(ValueError, match="different checkpoint_data_contract"):
        WeightedMixtureDataset(
            [_Dataset("a", 10, {"version": 1}), _Dataset("b", 10, {"version": 2})],
            [1, 1],
        )


def test_mixture_allows_source_sample_count_to_differ():
    first = {"schema": "v1", "sampling": {"source_fps": 30, "sample_count": 100}}
    second = deepcopy(first)
    second["sampling"]["sample_count"] = 200

    mixture = WeightedMixtureDataset(
        [_Dataset("a", 10, first), _Dataset("b", 10, second)],
        [1, 1],
    )

    assert mixture.checkpoint_data_contract == first


@pytest.mark.parametrize("weights", [[], [1], [1, 0], [1, float("nan")]])
def test_mixture_rejects_invalid_weights(weights):
    with pytest.raises(ValueError, match="weights|length mismatch"):
        WeightedMixtureDataset([_Dataset("a", 10), _Dataset("b", 10)], weights)
