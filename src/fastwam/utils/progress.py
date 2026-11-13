# Copyright (c) 2026 Xiaomi Corporation

"""Minimal dependency-free progress reporting.

``progress`` covers the small subset of the tqdm interface used in this repository: wrapping an iterable, or use as
a context manager with ``update(n)``. It prints a line to stderr at most every ``interval`` seconds and once at the
end, so it also works in non-interactive logs.
"""

from __future__ import annotations

import sys
import time
from typing import Iterable, Iterator, Optional, TypeVar

T = TypeVar("T")


class progress:
    def __init__(
        self,
        iterable: Optional[Iterable[T]] = None,
        total: Optional[int] = None,
        desc: str = "",
        unit: str = "it",
        disable: bool = False,
        interval: float = 10.0,
        **_: object,
    ) -> None:
        self.iterable = iterable
        if total is None and iterable is not None:
            try:
                total = len(iterable)  # type: ignore[arg-type]
            except TypeError:
                total = None
        self.total = total
        self.desc = desc
        self.unit = unit
        self.disable = disable
        self.interval = interval
        self.n = 0
        self._start = time.monotonic()
        self._last = self._start

    def _print(self, final: bool = False) -> None:
        if self.disable:
            return
        elapsed = time.monotonic() - self._start
        rate = self.n / elapsed if elapsed > 0 else 0.0
        if self.total:
            head = f"{self.n}/{self.total} {self.unit} ({100.0 * self.n / self.total:.0f}%)"
        else:
            head = f"{self.n} {self.unit}"
        prefix = f"{self.desc}: " if self.desc else ""
        tail = " done" if final else ""
        print(f"{prefix}{head}, {elapsed:.0f}s, {rate:.2f} {self.unit}/s{tail}", file=sys.stderr, flush=True)

    def update(self, n: int = 1) -> None:
        self.n += n
        now = time.monotonic()
        if now - self._last >= self.interval:
            self._last = now
            self._print()

    def close(self) -> None:
        self._print(final=True)

    def __iter__(self) -> Iterator[T]:
        if self.iterable is None:
            raise TypeError("progress() needs an iterable to be iterated")
        for item in self.iterable:
            yield item
            self.update(1)
        self.close()

    def __enter__(self) -> "progress":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
