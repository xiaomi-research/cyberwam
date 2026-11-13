#!/usr/bin/env python3
# Copyright (c) 2026 Xiaomi Corporation

"""Build and verify the corrected B1K v3 numeric sidecar and dataset bundle."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SIDECAR_SCHEMA_VERSION = "fastwam.b1k_numeric_sidecar.v1"
BUNDLE_SCHEMA_VERSION = "fastwam.b1k_dataset_bundle.v1"
NUMERIC_COLUMNS = ("action", "observation.state")
NUMERIC_DIMENSIONS = {"action": 23, "observation.state": 61}
SOURCE_METADATA_PATHS = ("meta/info.json", "meta/tasks.jsonl", "meta/tasks.parquet")


class DatasetBundleError(RuntimeError):
    pass


@dataclass(frozen=True)
class SourceEpisode:
    index: int
    task_index: int
    task: str
    length: int
    source_path: Path
    source_relative_path: str
    source_row_start: int


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _write_checksum(path: Path) -> None:
    checksum_path = path.with_suffix(path.suffix + ".sha256")
    checksum_path.write_text(f"{_sha256_file(path)}  {path.name}\n", encoding="ascii")


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    os.close(fd)
    temporary_path = Path(temporary)
    try:
        shutil.copyfile(source, temporary_path)
        os.replace(temporary_path, destination)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _metadata_fingerprint(source_root: Path) -> dict[str, Any]:
    relative_paths = list(SOURCE_METADATA_PATHS)
    relative_paths.extend(
        path.relative_to(source_root).as_posix()
        for path in sorted(source_root.glob("meta/episodes/chunk-*/file-*.parquet"))
        if path.is_file()
    )
    missing = [
        relative
        for relative in relative_paths
        if not (source_root / relative).is_file()
    ]
    if missing:
        raise DatasetBundleError(f"source metadata is incomplete: {missing[:8]}")
    entries = [
        {
            "path": relative,
            "size_bytes": (source_root / relative).stat().st_size,
            "sha256": _sha256_file(source_root / relative),
        }
        for relative in sorted(relative_paths)
    ]
    return {
        "algorithm": "sha256",
        "strategy": "lerobot_v3_metadata_content_v1",
        "sha256": _canonical_sha256(entries),
        "files": entries,
    }


def _load_source_episodes(
    source_root: Path, *, expected_episodes: int | None
) -> tuple[dict[str, Any], list[SourceEpisode]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise DatasetBundleError(
            "building the numeric sidecar requires pyarrow"
        ) from exc

    info = json.loads((source_root / "meta" / "info.json").read_text(encoding="utf-8"))
    if not str(info.get("codebase_version", "")).startswith("v3."):
        raise DatasetBundleError("numeric sidecar source must use LeRobot v3 metadata")
    paths = sorted(source_root.glob("meta/episodes/chunk-*/file-*.parquet"))
    if not paths:
        raise DatasetBundleError("source contains no episode metadata parquet files")
    columns = [
        "episode_index",
        "tasks",
        "length",
        "data/chunk_index",
        "data/file_index",
        "dataset_from_index",
        "dataset_to_index",
    ]
    has_task_index = all("task_index" in pq.read_schema(path).names for path in paths)
    if has_task_index:
        columns.append("task_index")
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(pq.read_table(path, columns=columns).to_pylist())
    rows.sort(key=lambda row: int(row["episode_index"]))
    if expected_episodes is not None and len(rows) != expected_episodes:
        raise DatasetBundleError(
            f"expected {expected_episodes} episodes, found {len(rows)}"
        )
    indices = [int(row["episode_index"]) for row in rows]
    if indices != list(range(len(rows))):
        raise DatasetBundleError("source episode indices must be contiguous from zero")

    file_starts: dict[Path, int] = {}
    rendered: list[tuple[dict[str, Any], Path]] = []
    for row in rows:
        path = source_root / info["data_path"].format(
            chunk_index=int(row["data/chunk_index"]),
            file_index=int(row["data/file_index"]),
        )
        rendered.append((row, path))
        start = int(row["dataset_from_index"])
        file_starts[path] = min(start, file_starts.get(path, start))

    if has_task_index:
        task_indices = {str(row["tasks"][0]): int(row["task_index"]) for row in rows}
    else:
        task_indices = {
            task: index
            for index, task in enumerate(sorted({str(row["tasks"][0]) for row in rows}))
        }
    episodes = []
    for row, path in rendered:
        length = int(row["length"])
        start = int(row["dataset_from_index"])
        if int(row["dataset_to_index"]) - start != length:
            raise DatasetBundleError(
                f"invalid range for episode {row['episode_index']}"
            )
        tasks = row["tasks"]
        if not isinstance(tasks, list) or len(tasks) != 1:
            raise DatasetBundleError(
                f"episode {row['episode_index']} must contain exactly one task"
            )
        episodes.append(
            SourceEpisode(
                index=int(row["episode_index"]),
                task_index=(
                    int(row["task_index"])
                    if has_task_index
                    else task_indices[str(tasks[0])]
                ),
                task=str(tasks[0]),
                length=length,
                source_path=path,
                source_relative_path=path.relative_to(source_root).as_posix(),
                source_row_start=start - file_starts[path],
            )
        )
    return info, episodes


def _fixed_size_column(column: Any, dimension: int, name: str) -> Any:
    import pyarrow as pa

    column = column.combine_chunks()
    if column.null_count:
        raise DatasetBundleError(f"source column {name!r} contains nulls")
    lengths = column.value_lengths().to_numpy(zero_copy_only=False)
    if len(lengths) and not (lengths == dimension).all():
        raise DatasetBundleError(
            f"source column {name!r} is not consistently {dimension}D"
        )
    values = column.flatten()
    return pa.FixedSizeListArray.from_arrays(values, dimension)


def _numeric_table(source_table: Any, start: int, length: int) -> Any:
    import pyarrow as pa

    sliced = source_table.slice(start, length)
    if sliced.num_rows != length:
        raise DatasetBundleError(
            f"source slice [{start}, {start + length}) returned {sliced.num_rows} rows"
        )
    return pa.table(
        {
            name: _fixed_size_column(sliced[name], dimension, name)
            for name, dimension in NUMERIC_DIMENSIONS.items()
        }
    )


def _tables_equal(actual: Any, expected: Any) -> bool:
    return actual.equals(expected, check_metadata=False)


def _episode_output_path(
    output_root: Path, episode_index: int, chunk_size: int
) -> Path:
    return (
        output_root
        / "data"
        / f"chunk-{episode_index // chunk_size:03d}"
        / f"episode_{episode_index:06d}.parquet"
    )


def _atomic_write_parquet(table: Any, path: Path) -> None:
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(fd)
    temporary_path = Path(temporary)
    try:
        pq.write_table(
            table,
            temporary_path,
            compression="zstd",
            use_dictionary=False,
            row_group_size=512,
        )
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _process_source_file(
    source_path: Path,
    episodes: list[SourceEpisode],
    output_root: Path,
    chunk_size: int,
    resume: bool,
) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    source = pq.read_table(source_path, columns=list(NUMERIC_COLUMNS), use_threads=True)
    records = []
    for episode in episodes:
        expected = _numeric_table(source, episode.source_row_start, episode.length)
        output = _episode_output_path(output_root, episode.index, chunk_size)
        reused = False
        if output.exists() and resume:
            try:
                reused = _tables_equal(
                    pq.read_table(output, columns=list(NUMERIC_COLUMNS)), expected
                )
            except Exception:
                reused = False
        if not reused:
            _atomic_write_parquet(expected, output)
        records.append(
            {
                "episode_index": episode.index,
                "length": episode.length,
                "path": output.relative_to(output_root).as_posix(),
                "sha256": _sha256_file(output),
                "size_bytes": output.stat().st_size,
                "source_path": episode.source_relative_path,
                "source_row_start": episode.source_row_start,
            }
        )
    return records


def build_numeric_sidecar(
    source_root: Path,
    output_root: Path,
    *,
    source_revision: str,
    expected_episodes: int | None = 20_000,
    chunk_size: int = 200,
    workers: int = 8,
    resume: bool = True,
) -> dict[str, Any]:
    source_root = source_root.expanduser().resolve(strict=True)
    output_root = output_root.expanduser().resolve()
    if output_root == source_root or output_root.is_relative_to(source_root):
        raise DatasetBundleError("numeric sidecar output must not be inside its source")
    if workers <= 0 or chunk_size <= 0:
        raise DatasetBundleError("workers and chunk_size must be positive")
    source_info, episodes = _load_source_episodes(
        source_root, expected_episodes=expected_episodes
    )
    source_fingerprint = _metadata_fingerprint(source_root)
    identity = {
        "schema_version": SIDECAR_SCHEMA_VERSION,
        "source_root": str(source_root),
        "source_revision": source_revision,
        "source_metadata_sha256": source_fingerprint["sha256"],
        "episode_count": len(episodes),
        "chunk_size": chunk_size,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    identity_path = output_root / ".build_identity.json"
    if identity_path.exists():
        existing = json.loads(identity_path.read_text(encoding="utf-8"))
        if existing != identity:
            raise DatasetBundleError(
                "refusing to resume a sidecar built from a different source or layout"
            )
    else:
        if any(output_root.iterdir()):
            raise DatasetBundleError(
                f"non-empty sidecar output has no build identity: {output_root}"
            )
        _atomic_json(identity_path, identity)

    grouped: dict[Path, list[SourceEpisode]] = {}
    for episode in episodes:
        grouped.setdefault(episode.source_path, []).append(episode)
    records: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _process_source_file,
                source_path,
                source_episodes,
                output_root,
                chunk_size,
                resume,
            )
            for source_path, source_episodes in grouped.items()
        ]
        for future in concurrent.futures.as_completed(futures):
            records.extend(future.result())
    records.sort(key=lambda record: record["episode_index"])

    episode_records = [
        {
            "episode_index": ep.index,
            "task_index": ep.task_index,
            "tasks": [ep.task],
            "length": ep.length,
        }
        for ep in episodes
    ]
    episodes_path = output_root / "meta" / "episodes.jsonl"
    episodes_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".episodes.", dir=episodes_path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            for record in episode_records:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, episodes_path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise

    info = {
        "codebase_version": "v2.1",
        "fps": int(source_info["fps"]),
        "robot_type": source_info.get("robot_type"),
        "chunks_size": chunk_size,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": None,
        "features": {
            name: {
                "dtype": "float32",
                "shape": [dimension],
                "names": [name.split(".")[-1] if name == "action" else None],
            }
            for name, dimension in NUMERIC_DIMENSIONS.items()
        },
        "total_episodes": len(episodes),
        "total_frames": sum(ep.length for ep in episodes),
        "total_chunks": (len(episodes) + chunk_size - 1) // chunk_size,
        "total_tasks": len({ep.task for ep in episodes}),
        "source_revision": source_revision,
        "source_metadata_sha256": source_fingerprint["sha256"],
    }
    _atomic_json(output_root / "meta" / "info.json", info)
    prior_manifest_path = output_root / "manifest.json"
    created_at_utc = datetime.now(timezone.utc).isoformat()
    if prior_manifest_path.is_file():
        try:
            created_at_utc = json.loads(
                prior_manifest_path.read_text(encoding="utf-8")
            )["created_at_utc"]
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            pass
    manifest = {
        "schema_version": SIDECAR_SCHEMA_VERSION,
        "created_at_utc": created_at_utc,
        "source": {
            "path": str(source_root),
            "revision": source_revision,
            "metadata_fingerprint": source_fingerprint,
        },
        "layout": {
            "episode_count": len(episodes),
            "total_rows": sum(ep.length for ep in episodes),
            "chunk_size": chunk_size,
            "columns": list(NUMERIC_COLUMNS),
        },
        "metadata": {
            "info_json_sha256": _sha256_file(output_root / "meta" / "info.json"),
            "episodes_jsonl_sha256": _sha256_file(episodes_path),
        },
        "files": records,
        "files_sha256": _canonical_sha256(
            [
                {
                    k: record[k]
                    for k in ("episode_index", "length", "path", "sha256", "size_bytes")
                }
                for record in records
            ]
        ),
    }
    manifest_path = output_root / "manifest.json"
    _atomic_json(manifest_path, manifest)
    _write_checksum(manifest_path)
    return manifest


def verify_numeric_sidecar(
    sidecar_root: Path,
    *,
    source_root: Path | None = None,
    compare_source: bool = True,
    verify_files: bool = True,
) -> dict[str, Any]:
    import pyarrow.parquet as pq

    sidecar_root = sidecar_root.expanduser().resolve(strict=True)
    manifest_path = sidecar_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SIDECAR_SCHEMA_VERSION:
        raise DatasetBundleError("unsupported numeric sidecar manifest schema")
    checksum_text = manifest_path.with_suffix(".json.sha256").read_text(
        encoding="ascii"
    )
    if checksum_text.split()[0] != _sha256_file(manifest_path):
        raise DatasetBundleError("numeric sidecar manifest checksum mismatch")
    root = (
        Path(manifest["source"]["path"])
        if source_root is None
        else source_root.expanduser()
    ).resolve(strict=True)
    if _metadata_fingerprint(root) != manifest["source"]["metadata_fingerprint"]:
        raise DatasetBundleError("numeric sidecar source metadata fingerprint mismatch")
    _, episodes = _load_source_episodes(
        root, expected_episodes=int(manifest["layout"]["episode_count"])
    )
    records = manifest["files"]
    if len(records) != len(episodes):
        raise DatasetBundleError("numeric sidecar manifest episode count mismatch")
    stable_records = [
        {
            k: record[k]
            for k in ("episode_index", "length", "path", "sha256", "size_bytes")
        }
        for record in records
    ]
    if _canonical_sha256(stable_records) != manifest.get("files_sha256"):
        raise DatasetBundleError("numeric sidecar aggregate file fingerprint mismatch")
    metadata = manifest.get("metadata", {})
    if metadata.get("info_json_sha256") != _sha256_file(
        sidecar_root / "meta/info.json"
    ):
        raise DatasetBundleError("numeric sidecar info.json fingerprint mismatch")
    if metadata.get("episodes_jsonl_sha256") != _sha256_file(
        sidecar_root / "meta/episodes.jsonl"
    ):
        raise DatasetBundleError("numeric sidecar episodes.jsonl fingerprint mismatch")
    if compare_source and not verify_files:
        raise DatasetBundleError("compare_source requires verify_files")
    if not verify_files:
        return manifest

    by_source: dict[Path, list[SourceEpisode]] = {}
    for episode in episodes:
        by_source.setdefault(episode.source_path, []).append(episode)
    record_by_index = {int(record["episode_index"]): record for record in records}
    for source_path, source_episodes in by_source.items():
        source = None
        if compare_source:
            source = pq.read_table(
                source_path, columns=list(NUMERIC_COLUMNS), use_threads=True
            )
        for episode in source_episodes:
            record = record_by_index.get(episode.index)
            if record is None:
                raise DatasetBundleError(
                    f"missing sidecar record for episode {episode.index}"
                )
            path = sidecar_root / record["path"]
            if not path.is_file() or _sha256_file(path) != record["sha256"]:
                raise DatasetBundleError(f"sidecar file fingerprint mismatch: {path}")
            actual = pq.read_table(path, columns=list(NUMERIC_COLUMNS))
            if actual.num_rows != episode.length:
                raise DatasetBundleError(
                    f"sidecar row count mismatch for episode {episode.index}"
                )
            if source is not None and not _tables_equal(
                actual, _numeric_table(source, episode.source_row_start, episode.length)
            ):
                raise DatasetBundleError(
                    f"sidecar values differ from v3 source for episode {episode.index}"
                )
    return manifest


def refresh_sidecar_metadata(
    sidecar_root: Path, *, source_root: Path
) -> dict[str, Any]:
    """Rewrite only derived sidecar metadata after a schema-compatible fix.

    Numeric parquet files and their manifest hashes are left untouched; the
    source episode table is reread to regenerate the split-loader JSONL.
    """
    sidecar_root = sidecar_root.expanduser().resolve(strict=True)
    source_root = source_root.expanduser().resolve(strict=True)
    manifest_path = sidecar_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SIDECAR_SCHEMA_VERSION:
        raise DatasetBundleError("unsupported numeric sidecar manifest schema")
    _, episodes = _load_source_episodes(
        source_root, expected_episodes=int(manifest["layout"]["episode_count"])
    )
    episodes_path = sidecar_root / "meta" / "episodes.jsonl"
    episodes_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".episodes.", dir=episodes_path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            for episode in episodes:
                stream.write(
                    json.dumps(
                        {
                            "episode_index": episode.index,
                            "task_index": episode.task_index,
                            "tasks": [episode.task],
                            "length": episode.length,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, episodes_path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    manifest["metadata"]["episodes_jsonl_sha256"] = _sha256_file(episodes_path)
    _atomic_json(manifest_path, manifest)
    _write_checksum(manifest_path)
    return manifest


def _file_fingerprint(path: Path) -> dict[str, Any]:
    return {
        "algorithm": "sha256",
        "strategy": "full_file_content_v1",
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _directory_fingerprint(path: Path) -> dict[str, Any]:
    entries = [
        {
            "path": candidate.relative_to(path).as_posix(),
            "size_bytes": candidate.stat().st_size,
            "sha256": _sha256_file(candidate),
        }
        for candidate in sorted(item for item in path.rglob("*") if item.is_file())
    ]
    if not entries:
        raise DatasetBundleError(f"directory contains no files: {path}")
    return {
        "algorithm": "sha256",
        "strategy": "full_directory_content_v1",
        "sha256": _canonical_sha256(entries),
        "file_count": len(entries),
        "files": entries,
    }


def _norm_source_metadata_fingerprint(dataset_root: Path) -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    aggregate = hashlib.sha256()
    paths = [
        path for path in sorted((dataset_root / "meta").glob("*")) if path.is_file()
    ]
    paths.extend(
        path
        for path in (
            dataset_root / "manifest.json",
            dataset_root / "manifest.json.sha256",
        )
        if path.is_file()
    )
    for path in sorted(paths):
        relative = path.relative_to(dataset_root).as_posix()
        file_hash = _sha256_file(path)
        files[relative] = {"sha256": file_hash, "size_bytes": path.stat().st_size}
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(file_hash.encode("ascii"))
        aggregate.update(b"\n")
    if not files:
        raise DatasetBundleError(
            f"numeric dataset has no metadata files: {dataset_root}"
        )
    return {"sha256": aggregate.hexdigest(), "files": files}


def _validate_norm_source(norm_path: Path, numeric_root: Path) -> None:
    try:
        document = json.loads(norm_path.read_text(encoding="utf-8"))
        dataset = document["dataset"]
        recorded_root = Path(dataset["root"]).expanduser().resolve(strict=True)
        recorded_fingerprint = dataset["source_metadata_fingerprint"]
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise DatasetBundleError(
            "norm stats do not contain a valid dataset source contract"
        ) from exc
    numeric_root = numeric_root.resolve(strict=True)
    if recorded_fingerprint != _norm_source_metadata_fingerprint(numeric_root):
        raise DatasetBundleError("norm stats numeric metadata fingerprint mismatch")
    if recorded_root == numeric_root:
        return
    try:
        manifest = json.loads(
            (numeric_root / "manifest.json").read_text(encoding="utf-8")
        )
        relative_paths = [Path(record["path"]) for record in manifest["files"]]
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise DatasetBundleError(
            "numeric hardlink clone has an invalid file manifest"
        ) from exc
    for relative_path in relative_paths:
        source_path = recorded_root / relative_path
        clone_path = numeric_root / relative_path
        try:
            identical = source_path.samefile(clone_path)
        except OSError as exc:
            raise DatasetBundleError(
                f"numeric hardlink clone file is missing: {relative_path}"
            ) from exc
        if not identical:
            raise DatasetBundleError(
                f"numeric clone is not hardlinked to the norm source: {relative_path}"
            )


def _ensure_symlink(path: Path, target: Path) -> None:
    target = target.resolve(strict=True)
    if path.is_symlink():
        if path.resolve(strict=True) != target:
            raise DatasetBundleError(f"existing symlink has a different target: {path}")
        return
    if path.exists():
        raise DatasetBundleError(
            f"bundle link path already exists and is not a symlink: {path}"
        )
    path.symlink_to(target, target_is_directory=target.is_dir())


def build_bundle(
    bundle_root: Path,
    *,
    source_root: Path,
    norm_stats: Path,
    subtask_manifest: Path,
    text_cache: Path,
    deep_numeric: bool = True,
) -> dict[str, Any]:
    bundle_root = bundle_root.expanduser().resolve()
    bundle_root.mkdir(parents=True, exist_ok=True)
    numeric_root = bundle_root / "numeric_v3"
    if not numeric_root.is_dir() or numeric_root.is_symlink():
        raise DatasetBundleError(
            "build the sidecar directly at <bundle>/numeric_v3 before assembling the bundle"
        )
    numeric_manifest = verify_numeric_sidecar(
        numeric_root,
        source_root=source_root,
        compare_source=deep_numeric,
        verify_files=deep_numeric,
    )
    _ensure_symlink(bundle_root / "v3_source", source_root)
    _ensure_symlink(bundle_root / "subtask_manifest.jsonl", subtask_manifest)
    _ensure_symlink(bundle_root / "text_cache", text_cache)
    norm_target = bundle_root / "norm_stats.json"
    norm_stats = norm_stats.expanduser().resolve(strict=True)
    _validate_norm_source(norm_stats, numeric_root)
    if norm_target.is_symlink():
        raise DatasetBundleError(
            "bundle norm_stats.json must be a copied file, not a symlink"
        )
    if norm_target.exists():
        if _sha256_file(norm_target) != _sha256_file(norm_stats):
            raise DatasetBundleError(
                "bundle norm_stats.json already differs from requested norm"
            )
    else:
        _atomic_copy(norm_stats, norm_target)

    inputs = {
        "video_dataset": {
            "path": "v3_source",
            "kind": "symlink",
            "fingerprint": numeric_manifest["source"]["metadata_fingerprint"],
        },
        "numeric_dataset": {
            "path": "numeric_v3",
            "kind": "directory",
            "fingerprint": {
                "algorithm": "sha256",
                "strategy": "numeric_sidecar_manifest_v1",
                "sha256": _sha256_file(numeric_root / "manifest.json"),
                "source_metadata_sha256": numeric_manifest["source"][
                    "metadata_fingerprint"
                ]["sha256"],
                "episode_count": numeric_manifest["layout"]["episode_count"],
                "total_rows": numeric_manifest["layout"]["total_rows"],
            },
        },
        "norm_stats": {
            "path": "norm_stats.json",
            "kind": "file",
            "fingerprint": _file_fingerprint(norm_target),
        },
        "subtask_manifest": {
            "path": "subtask_manifest.jsonl",
            "kind": "symlink",
            "fingerprint": _file_fingerprint(subtask_manifest.resolve(strict=True)),
        },
        "text_cache": {
            "path": "text_cache",
            "kind": "symlink",
            "fingerprint": _directory_fingerprint(text_cache.resolve(strict=True)),
        },
    }
    manifest_path = bundle_root / "bundle_manifest.json"
    created_at_utc = datetime.now(timezone.utc).isoformat()
    if manifest_path.is_file():
        try:
            created_at_utc = json.loads(manifest_path.read_text(encoding="utf-8"))[
                "created_at_utc"
            ]
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            pass
    manifest = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "created_at_utc": created_at_utc,
        "source_revision": numeric_manifest["source"]["revision"],
        "inputs": inputs,
        "bundle_sha256": _canonical_sha256(inputs),
    }
    _atomic_json(manifest_path, manifest)
    _write_checksum(manifest_path)
    return manifest


def verify_bundle(bundle_root: Path, *, deep_numeric: bool = True) -> dict[str, Any]:
    bundle_root = bundle_root.expanduser().resolve(strict=True)
    manifest_path = bundle_root / "bundle_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        raise DatasetBundleError("unsupported dataset bundle manifest schema")
    if manifest.get("bundle_sha256") != _canonical_sha256(manifest.get("inputs")):
        raise DatasetBundleError("dataset bundle aggregate fingerprint mismatch")
    if manifest_path.with_suffix(".json.sha256").read_text(encoding="ascii").split()[
        0
    ] != _sha256_file(manifest_path):
        raise DatasetBundleError("dataset bundle manifest checksum mismatch")
    inputs = manifest["inputs"]
    for name in ("video_dataset", "subtask_manifest", "text_cache"):
        if not (bundle_root / inputs[name]["path"]).is_symlink():
            raise DatasetBundleError(f"bundle input must remain a symlink: {name}")
    if (
        _metadata_fingerprint(bundle_root / inputs["video_dataset"]["path"])
        != inputs["video_dataset"]["fingerprint"]
    ):
        raise DatasetBundleError("bundle video dataset fingerprint mismatch")
    numeric_path = bundle_root / inputs["numeric_dataset"]["path"]
    norm_path = bundle_root / inputs["norm_stats"]["path"]
    if numeric_path.is_symlink():
        raise DatasetBundleError("bundle numeric dataset must not be a symlink")
    if norm_path.is_symlink():
        raise DatasetBundleError("bundle norm stats must not be a symlink")
    numeric_manifest = verify_numeric_sidecar(
        numeric_path,
        source_root=bundle_root / inputs["video_dataset"]["path"],
        compare_source=False,
        verify_files=deep_numeric,
    )
    numeric_expected = inputs["numeric_dataset"]["fingerprint"]
    if _sha256_file(numeric_path / "manifest.json") != numeric_expected["sha256"]:
        raise DatasetBundleError("bundle numeric dataset fingerprint mismatch")
    _validate_norm_source(norm_path, numeric_path)
    checks = {
        "norm_stats": _file_fingerprint,
        "subtask_manifest": _file_fingerprint,
        "text_cache": _directory_fingerprint,
    }
    for name, fingerprint in checks.items():
        actual = fingerprint((bundle_root / inputs[name]["path"]).resolve(strict=True))
        if actual != inputs[name]["fingerprint"]:
            raise DatasetBundleError(f"bundle input fingerprint mismatch: {name}")
    if numeric_manifest["source"]["revision"] != manifest["source_revision"]:
        raise DatasetBundleError("bundle source revision mismatch")
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-sidecar")
    build.add_argument("--source-root", type=Path, required=True)
    build.add_argument("--output-root", type=Path, required=True)
    build.add_argument("--source-revision", required=True)
    build.add_argument("--expected-episodes", type=int, default=20_000)
    build.add_argument("--chunk-size", type=int, default=200)
    build.add_argument("--workers", type=int, default=8)
    build.add_argument("--no-resume", action="store_true")
    verify = commands.add_parser("verify-sidecar")
    verify.add_argument("--sidecar-root", type=Path, required=True)
    verify.add_argument("--source-root", type=Path)
    verify.add_argument("--no-compare-source", action="store_true")
    refresh = commands.add_parser("refresh-metadata")
    refresh.add_argument("--sidecar-root", type=Path, required=True)
    refresh.add_argument("--source-root", type=Path, required=True)
    bundle = commands.add_parser("build-bundle")
    bundle.add_argument("--bundle-root", type=Path, required=True)
    bundle.add_argument("--source-root", type=Path, required=True)
    bundle.add_argument("--norm-stats", type=Path, required=True)
    bundle.add_argument("--subtask-manifest", type=Path, required=True)
    bundle.add_argument("--text-cache", type=Path, required=True)
    bundle.add_argument(
        "--skip-deep-numeric-verify",
        action="store_true",
        help="skip rereading every numeric parquet when assembling a verified hardlink clone",
    )
    verify_bundle_parser = commands.add_parser("verify-bundle")
    verify_bundle_parser.add_argument("--bundle-root", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "build-sidecar":
        result = build_numeric_sidecar(
            args.source_root,
            args.output_root,
            source_revision=args.source_revision,
            expected_episodes=args.expected_episodes,
            chunk_size=args.chunk_size,
            workers=args.workers,
            resume=not args.no_resume,
        )
    elif args.command == "verify-sidecar":
        result = verify_numeric_sidecar(
            args.sidecar_root,
            source_root=args.source_root,
            compare_source=not args.no_compare_source,
        )
    elif args.command == "refresh-metadata":
        result = refresh_sidecar_metadata(
            args.sidecar_root,
            source_root=args.source_root,
        )
    elif args.command == "build-bundle":
        result = build_bundle(
            args.bundle_root,
            source_root=args.source_root,
            norm_stats=args.norm_stats,
            subtask_manifest=args.subtask_manifest,
            text_cache=args.text_cache,
            deep_numeric=not args.skip_deep_numeric_verify,
        )
    else:
        result = verify_bundle(args.bundle_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
