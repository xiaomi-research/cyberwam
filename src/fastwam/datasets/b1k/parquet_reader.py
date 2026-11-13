# Copyright (c) 2026 Xiaomi Corporation

from __future__ import annotations

import os
from collections import OrderedDict
from pathlib import Path
from typing import Any, Sequence


class B1KParquetReader:
    """Small worker-local LRU reader for per-episode B1K parquet files.

    Numeric row groups are cheap enough to retain. Image row groups are always
    transient because retaining them would pin hundreds of JPEG payloads for a
    window that only decodes nine frames per camera.
    """

    def __init__(self, *, max_open_files: int = 8, max_numeric_row_groups: int = 32):
        if max_open_files <= 0 or max_numeric_row_groups < 0:
            raise ValueError("Invalid B1K parquet cache capacity.")
        self.max_open_files = max_open_files
        self.max_numeric_row_groups = max_numeric_row_groups
        self._pid = os.getpid()
        self._files: OrderedDict[str, Any] = OrderedDict()
        self._numeric_row_groups: OrderedDict[tuple[str, int, tuple[str, ...]], Any] = (
            OrderedDict()
        )

    def _ensure_worker_local(self) -> None:
        pid = os.getpid()
        if pid != self._pid:
            self._pid = pid
            self._files.clear()
            self._numeric_row_groups.clear()

    def __getstate__(self) -> dict[str, Any]:
        # ParquetFile objects are process-local and are not reliably picklable.
        return {
            "max_open_files": self.max_open_files,
            "max_numeric_row_groups": self.max_numeric_row_groups,
            "_pid": os.getpid(),
            "_files": OrderedDict(),
            "_numeric_row_groups": OrderedDict(),
        }

    def _parquet_file(self, path: str | Path):
        self._ensure_worker_local()
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError(
                "B1K direct parquet loading requires pyarrow. Install the project "
                "environment declared by pyproject.toml."
            ) from exc

        key = str(path)
        parquet_file = self._files.pop(key, None)
        if parquet_file is None:
            if not Path(key).is_file():
                raise FileNotFoundError(f"Missing B1K episode parquet: {key}")
            parquet_file = pq.ParquetFile(key)
        self._files[key] = parquet_file
        while len(self._files) > self.max_open_files:
            self._files.popitem(last=False)
        return parquet_file

    def num_rows(self, path: str | Path) -> int:
        return int(self._parquet_file(path).metadata.num_rows)

    def read_rows(
        self,
        path: str | Path,
        columns: Sequence[str],
        row_indices: Sequence[int],
        *,
        cache_numeric: bool,
    ) -> dict[str, list[Any]]:
        """Read selected rows while preserving order and duplicate indices."""
        if not columns:
            raise ValueError("At least one parquet column must be requested.")
        if not row_indices:
            return {column: [] for column in columns}

        parquet_file = self._parquet_file(path)
        num_rows = int(parquet_file.metadata.num_rows)
        invalid = [index for index in row_indices if index < 0 or index >= num_rows]
        if invalid:
            raise IndexError(
                f"Parquet row indices out of bounds for {path}: {invalid[:8]} "
                f"(num_rows={num_rows})."
            )

        boundaries: list[tuple[int, int]] = []
        start = 0
        for row_group in range(parquet_file.num_row_groups):
            end = start + int(parquet_file.metadata.row_group(row_group).num_rows)
            boundaries.append((start, end))
            start = end

        locations: dict[int, tuple[int, int]] = {}
        wanted = set(row_indices)
        for row_group, (group_start, group_end) in enumerate(boundaries):
            for index in wanted:
                if group_start <= index < group_end:
                    locations[index] = (row_group, index - group_start)
        if len(locations) != len(wanted):
            missing = sorted(wanted - set(locations))
            raise RuntimeError(f"Failed to locate parquet rows {missing[:8]} in {path}.")

        column_tuple = tuple(columns)
        values_by_row: dict[int, dict[str, Any]] = {}
        groups = sorted({row_group for row_group, _ in locations.values()})
        for row_group in groups:
            cache_key = (str(path), row_group, column_tuple)
            table = None
            if cache_numeric:
                table = self._numeric_row_groups.pop(cache_key, None)
            if table is None:
                table = parquet_file.read_row_group(row_group, columns=list(columns))
            if cache_numeric and self.max_numeric_row_groups:
                self._numeric_row_groups[cache_key] = table
                while len(self._numeric_row_groups) > self.max_numeric_row_groups:
                    self._numeric_row_groups.popitem(last=False)

            group_start, _ = boundaries[row_group]
            local_rows = sorted(
                local for index, (group, local) in locations.items() if group == row_group
            )
            selected = table.take(local_rows).to_pydict()
            for position, local_row in enumerate(local_rows):
                global_row = group_start + local_row
                values_by_row[global_row] = {
                    column: selected[column][position] for column in columns
                }

        return {
            column: [values_by_row[index][column] for index in row_indices]
            for column in columns
        }
