# Copyright (c) 2026 Xiaomi Corporation
# Portions derived from FastWAM (MIT License, https://github.com/yuantianyuan01/FastWAM); see NOTICE.md.

import math
from collections.abc import Iterator, Sized
from itertools import islice

from torch.utils.data import Sampler


_UINT64_MASK = (1 << 64) - 1


def _splitmix64(value: int) -> int:
    """Return a stable 64-bit mix without allocating RNG state."""
    value = (value + 0x9E3779B97F4A7C15) & _UINT64_MASK
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _UINT64_MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _UINT64_MASK
    return value ^ (value >> 31)


class ResumableEpochSampler(Sampler[int]):
    """Deterministic, constant-memory sampler with exact batch-boundary resume.

    Each epoch uses ``f(i) = (a * i + b) mod N`` with ``gcd(a, N) = 1``.
    Multiplication by ``a`` is therefore invertible modulo ``N`` and adding
    ``b`` preserves bijectivity, so every retained sample appears exactly once.

    The sampler emits a shared global stream. Accelerate partitions consecutive
    local batches from that stream between processes. To avoid Accelerate
    padding the last distributed batch with duplicate samples, fewer than
    ``batch_size * num_processes`` trailing samples are dropped each epoch.

    Epochs advance whenever a new iterator is created. This is required by the
    trainer's while-step loop, which recreates its dataloader iterator without
    calling ``set_epoch`` after exhaustion.
    """

    def __init__(self, dataset: Sized, seed: int, batch_size: int, num_processes: int):
        self.dataset = dataset
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self.num_processes = int(num_processes)
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.num_processes <= 0:
            raise ValueError(f"num_processes must be positive, got {self.num_processes}")

        self.epoch = 0
        self.epoch_offset = 0
        self.resume_batch_offset = 0
        self._resume_pending = False

    @property
    def _global_batch_size(self) -> int:
        return self.batch_size * self.num_processes

    @property
    def _samples_per_epoch(self) -> int:
        size = len(self.dataset)
        if size < 0:
            raise ValueError(f"dataset length must be non-negative, got {size}")
        return size - size % self._global_batch_size

    def set_epoch(self, epoch: int):
        epoch = int(epoch)
        if epoch < 0:
            raise ValueError(f"epoch must be non-negative, got {epoch}")
        self.epoch = epoch

    def set_epoch_offset(self, epoch_offset: int):
        epoch_offset = int(epoch_offset)
        if epoch_offset < 0:
            raise ValueError(f"epoch_offset must be non-negative, got {epoch_offset}")
        self.epoch_offset = epoch_offset

    def set_resume_batch_offset(self, batch_in_epoch: int):
        batch_in_epoch = int(batch_in_epoch)
        if batch_in_epoch < 0:
            raise ValueError(f"batch_in_epoch must be non-negative, got {batch_in_epoch}")
        max_batches = self._samples_per_epoch // self._global_batch_size
        if batch_in_epoch > max_batches:
            raise ValueError(
                f"batch_in_epoch {batch_in_epoch} exceeds epoch length {max_batches} "
                "in distributed batches"
            )
        self.resume_batch_offset = batch_in_epoch
        self._resume_pending = batch_in_epoch > 0

    def clear_resume_batch_offset(self):
        self.resume_batch_offset = 0
        self._resume_pending = False

    def _permutation_parameters(self, size: int, effective_epoch: int) -> tuple[int, int]:
        if size <= 1:
            return 0, 0

        epoch_key = _splitmix64(self.seed ^ _splitmix64(effective_epoch))
        multiplier = 1 + epoch_key % (size - 1)
        while math.gcd(multiplier, size) != 1:
            multiplier += 1
            if multiplier == size:
                multiplier = 1

        # Adjacent epochs have different offsets for size > 1, guaranteeing
        # that consecutive epoch permutations cannot be identical.
        base_offset = _splitmix64(self.seed ^ 0xD1B54A32D192ED03) % size
        offset = (base_offset + effective_epoch) % size
        return multiplier, offset

    def preview_indices(
        self,
        *,
        epoch: int,
        batch_in_epoch: int,
        count: int = 32,
    ) -> list[int]:
        """Preview the resumed global sample stream without mutating sampler state."""
        epoch = int(epoch)
        batch_in_epoch = int(batch_in_epoch)
        count = int(count)
        if epoch < 0 or batch_in_epoch < 0 or count < 0:
            raise ValueError("epoch, batch_in_epoch, and count must be non-negative")
        max_batches = self._samples_per_epoch // self._global_batch_size
        if batch_in_epoch > max_batches:
            raise ValueError(
                f"batch_in_epoch {batch_in_epoch} exceeds epoch length {max_batches} "
                "in distributed batches"
            )
        start = batch_in_epoch * self._global_batch_size
        custom_indices = getattr(self.dataset, "sampler_indices", None)
        if callable(custom_indices):
            return list(
                islice(
                    custom_indices(
                        epoch=epoch,
                        start=start,
                        stop=self._samples_per_epoch,
                        batch_size=self.batch_size,
                        num_processes=self.num_processes,
                    ),
                    count,
                )
            )
        size = len(self.dataset)
        multiplier, offset = self._permutation_parameters(size, epoch)
        indices = list(
            islice(
                (
                    (multiplier * position + offset) % size
                    for position in range(start, self._samples_per_epoch)
                ),
                count,
            )
        )
        encode = getattr(self.dataset, "sampler_index", None)
        if callable(encode):
            return [encode(index, epoch) for index in indices]
        return indices

    def __iter__(self) -> Iterator[int]:
        size = len(self.dataset)
        retained_size = self._samples_per_epoch
        effective_epoch = self.epoch_offset + self.epoch
        start = self.resume_batch_offset * self._global_batch_size if self._resume_pending else 0
        multiplier, offset = self._permutation_parameters(size, effective_epoch)

        self.epoch += 1
        self._resume_pending = False

        custom_indices = getattr(self.dataset, "sampler_indices", None)
        if callable(custom_indices):
            return iter(
                custom_indices(
                    epoch=effective_epoch,
                    start=start,
                    stop=retained_size,
                    batch_size=self.batch_size,
                    num_processes=self.num_processes,
                )
            )

        indices = (
            (multiplier * position + offset) % size
            for position in range(start, retained_size)
        )
        encode = getattr(self.dataset, "sampler_index", None)
        if callable(encode):
            return (encode(index, effective_epoch) for index in indices)
        return indices

    def __len__(self) -> int:
        if not self._resume_pending:
            return self._samples_per_epoch
        return self._samples_per_epoch - self.resume_batch_offset * self._global_batch_size
