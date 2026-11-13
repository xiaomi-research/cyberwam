# Copyright (c) 2026 Xiaomi Corporation

from __future__ import annotations

import bisect
import copy
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from torch.utils.data import Dataset


_UINT64_MASK = (1 << 64) - 1


def _splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & _UINT64_MASK
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _UINT64_MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _UINT64_MASK
    return value ^ (value >> 31)


def _apportion(size: int, weights: Sequence[float]) -> tuple[int, ...]:
    raw = [size * weight / sum(weights) for weight in weights]
    counts = [math.floor(value) for value in raw]
    remainder = size - sum(counts)
    order = sorted(range(len(weights)), key=lambda i: (-(raw[i] - counts[i]), i))
    for index in order[:remainder]:
        counts[index] += 1
    return tuple(counts)


class WeightedMixtureDataset(Dataset):
    """Deterministic weighted sampling over map-style datasets.

    The virtual epoch has exactly ``epoch_size`` entries. Source quotas use
    largest-remainder apportionment, while each source's samples are permuted
    independently per epoch. Repetition is intentional when a quota exceeds a
    source dataset's length.
    """

    def __init__(
        self,
        datasets: Sequence[Dataset],
        weights: Sequence[float],
        *,
        epoch_size: int | None = None,
        seed: int = 0,
    ) -> None:
        self.datasets = tuple(datasets)
        self.weights = tuple(float(weight) for weight in weights)
        if not self.datasets:
            raise ValueError("WeightedMixtureDataset requires at least one dataset.")
        if len(self.datasets) != len(self.weights):
            raise ValueError(
                f"datasets/weights length mismatch: {len(self.datasets)} != {len(self.weights)}"
            )
        if any(not math.isfinite(weight) or weight <= 0 for weight in self.weights):
            raise ValueError(f"mixture weights must be finite and positive, got {self.weights}")
        self.dataset_lengths = tuple(len(dataset) for dataset in self.datasets)
        if any(length <= 0 for length in self.dataset_lengths):
            raise ValueError(
                f"every mixture source must be non-empty, got lengths={self.dataset_lengths}"
            )
        if epoch_size is None:
            epoch_size = sum(self.dataset_lengths)
        if isinstance(epoch_size, bool) or int(epoch_size) != epoch_size or int(epoch_size) <= 0:
            raise ValueError(f"epoch_size must be a positive integer, got {epoch_size!r}")
        self.epoch_size = int(epoch_size)
        self.seed = int(seed)
        self.source_counts = _apportion(self.epoch_size, self.weights)
        cumulative = []
        total = 0
        for count in self.source_counts:
            total += count
            cumulative.append(total)
        self._cumulative_counts = tuple(cumulative)

        primary_contract = self._read_contract(self.datasets[0])
        comparable_primary_contract = self._comparable_contract(primary_contract)
        for source, dataset in enumerate(self.datasets[1:], start=1):
            contract = self._read_contract(dataset)
            if (
                comparable_primary_contract is not None
                and self._comparable_contract(contract) != comparable_primary_contract
            ):
                raise ValueError(
                    f"mixture source {source} has a different checkpoint_data_contract; "
                    "all sources must use the same action/video/proprio/text contract"
                )
        self._checkpoint_data_contract = primary_contract

    @staticmethod
    def _read_contract(dataset: Dataset) -> Mapping[str, Any] | None:
        contract = getattr(dataset, "checkpoint_data_contract", None)
        if callable(contract):
            contract = contract()
        if contract is not None and not isinstance(contract, Mapping):
            raise TypeError("checkpoint_data_contract must be mapping-valued when present")
        return contract

    @staticmethod
    def _comparable_contract(contract: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
        if contract is None:
            return None
        comparable = copy.deepcopy(dict(contract))
        sampling = comparable.get("sampling")
        if isinstance(sampling, dict):
            sampling.pop("sample_count", None)
        return comparable

    @property
    def checkpoint_data_contract(self) -> Mapping[str, Any]:
        if self._checkpoint_data_contract is None:
            raise AttributeError("mixture sources do not expose checkpoint_data_contract")
        return self._checkpoint_data_contract

    @property
    def mixture_contract(self) -> dict[str, Any]:
        total_weight = sum(self.weights)
        return {
            "weights": [weight / total_weight for weight in self.weights],
            "epoch_size": self.epoch_size,
            "nominal_source_counts": list(self.source_counts),
            "source_lengths": list(self.dataset_lengths),
            "seed": self.seed,
        }

    def __len__(self) -> int:
        return self.epoch_size

    def sampler_index(self, index: int, epoch: int) -> tuple[int, int]:
        return int(epoch), int(index)

    def sampler_indices(
        self,
        *,
        epoch: int,
        start: int,
        stop: int,
        batch_size: int,
        num_processes: int,
    ):
        """Yield a deterministic stream with weighted quotas in every rank-local batch."""
        global_batch_size = batch_size * num_processes
        if start % global_batch_size or stop % global_batch_size:
            raise ValueError("mixture sampler bounds must align to global batches")

        normalized = [weight / sum(self.weights) for weight in self.weights]
        carry = [0.0] * len(self.datasets)
        source_positions = [0] * len(self.datasets)
        first_batch = start // global_batch_size
        last_batch = stop // global_batch_size
        for global_batch in range(last_batch):
            raw = [batch_size * weight + carry[i] for i, weight in enumerate(normalized)]
            counts = [math.floor(value) for value in raw]
            remaining = batch_size - sum(counts)
            order = sorted(
                range(len(counts)),
                key=lambda i: (-(raw[i] - counts[i]), i),
            )
            for source in order[:remaining]:
                counts[source] += 1
            carry = [raw[i] - counts[i] for i in range(len(counts))]

            emit = global_batch >= first_batch
            for rank in range(num_processes):
                local_batch = []
                for source, count in enumerate(counts):
                    for _ in range(count):
                        local_batch.append((epoch, source, source_positions[source]))
                        source_positions[source] += 1
                if not emit:
                    continue
                key = _splitmix64(
                    self.seed
                    ^ _splitmix64(epoch)
                    ^ _splitmix64(global_batch)
                    ^ _splitmix64(rank)
                )
                if len(local_batch) > 1:
                    multiplier = 1 + key % (len(local_batch) - 1)
                    while math.gcd(multiplier, len(local_batch)) != 1:
                        multiplier = multiplier % len(local_batch) + 1
                    offset = _splitmix64(key) % len(local_batch)
                    local_batch = [
                        local_batch[(multiplier * position + offset) % len(local_batch)]
                        for position in range(len(local_batch))
                    ]
                yield from local_batch

    def _resolve_index(self, index: int | tuple[int, ...]) -> tuple[int, int]:
        if isinstance(index, tuple):
            if len(index) == 3:
                epoch, source, source_position = map(int, index)
                if not 0 <= source < len(self.datasets) or source_position < 0:
                    raise IndexError(index)
                source_length = self.dataset_lengths[source]
                key = _splitmix64(self.seed ^ _splitmix64(epoch) ^ _splitmix64(source))
                multiplier = 1 + key % max(source_length - 1, 1)
                while source_length > 1 and math.gcd(multiplier, source_length) != 1:
                    multiplier = multiplier % source_length + 1
                offset = _splitmix64(key ^ 0xD1B54A32D192ED03) % source_length
                return source, (multiplier * source_position + offset) % source_length
            if len(index) != 2:
                raise IndexError(f"invalid mixture sampler index: {index!r}")
            epoch, virtual_index = (int(index[0]), int(index[1]))
        else:
            epoch, virtual_index = 0, int(index)
        if virtual_index < 0:
            virtual_index += self.epoch_size
        if not 0 <= virtual_index < self.epoch_size:
            raise IndexError(virtual_index)

        source = bisect.bisect_right(self._cumulative_counts, virtual_index)
        source_start = 0 if source == 0 else self._cumulative_counts[source - 1]
        source_position = virtual_index - source_start
        source_length = self.dataset_lengths[source]
        key = _splitmix64(self.seed ^ _splitmix64(epoch) ^ _splitmix64(source))
        multiplier = 1 + key % max(source_length - 1, 1)
        while source_length > 1 and math.gcd(multiplier, source_length) != 1:
            multiplier = multiplier % source_length + 1
        offset = _splitmix64(key ^ 0xD1B54A32D192ED03) % source_length
        return source, (multiplier * source_position + offset) % source_length

    def __getitem__(self, index: int | tuple[int, ...]) -> Any:
        source, local_index = self._resolve_index(index)
        return self.datasets[source][local_index]

    def __getitems__(self, indices: Sequence[int | tuple[int, ...]]) -> list[Any]:
        grouped: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for output_position, index in enumerate(indices):
            source, local_index = self._resolve_index(index)
            grouped[source].append((output_position, local_index))

        output: list[Any] = [None] * len(indices)
        for source, requests in grouped.items():
            local_indices = [local_index for _, local_index in requests]
            getitems = getattr(self.datasets[source], "__getitems__", None)
            samples = (
                getitems(local_indices)
                if callable(getitems)
                else [self.datasets[source][index] for index in local_indices]
            )
            if len(samples) != len(requests):
                raise RuntimeError(
                    f"mixture source {source} returned {len(samples)} samples for "
                    f"{len(requests)} requested indices"
                )
            for (output_position, _), sample in zip(requests, samples, strict=True):
                output[output_position] = sample
        return output

    def __getattr__(self, name: str) -> Any:
        # Preserve read-only dataset metadata used by logging and validation.
        if name.startswith("_") or name in {"datasets", "weights"}:
            raise AttributeError(name)
        return getattr(self.datasets[0], name)
