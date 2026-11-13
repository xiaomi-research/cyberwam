# Copyright (c) 2026 Xiaomi Corporation

"""Small dependency-light implementation of the official B1K websocket wire format.

The message layout (msgpack with the ``__ndarray__`` / ``__npgeneric__`` numpy encoding, a metadata message
on connect, and a ``/healthz`` check) has to match the official BEHAVIOR-1K evaluator
(``omnigibson.eval.utils.network_utils``), which is adapted from openpi
(https://github.com/Physical-Intelligence/openpi, Apache-2.0). The numpy encoding originates from
msgpack-numpy (https://github.com/lebedov/msgpack-numpy, BSD-3-Clause). The code below is our own
implementation of that format; see ``NOTICE.md``.
"""

from __future__ import annotations

import asyncio
import functools
import http
import logging
import time
import traceback
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import uuid
from typing import Any

import msgpack
import numpy as np
import websockets
import websockets.asyncio.server as websocket_server


LOGGER = logging.getLogger(__name__)


def _pack_data(value: Any) -> Any:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        if value.dtype.kind in ("V", "O", "c"):
            raise ValueError(f"Unsupported ndarray dtype {value.dtype}.")
        return {
            b"__ndarray__": True,
            b"data": value.tobytes(),
            b"dtype": value.dtype.str,
            b"shape": value.shape,
        }
    if isinstance(value, np.generic):
        return {b"__npgeneric__": True, b"data": value.item(), b"dtype": value.dtype.str}
    return value


def _unpack_data(value: Any) -> Any:
    if isinstance(value, dict) and value.get(b"__ndarray__"):
        return np.ndarray(
            buffer=value[b"data"], dtype=np.dtype(value[b"dtype"]), shape=value[b"shape"]
        )
    if isinstance(value, dict) and value.get(b"__npgeneric__"):
        return np.dtype(value[b"dtype"]).type(value[b"data"])
    return value


Packer = functools.partial(msgpack.Packer, default=_pack_data)
packb = functools.partial(msgpack.packb, default=_pack_data)
unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_data)


class WebsocketPolicyServer:
    """Serve a policy using the protocol used by ``omnigibson.eval``."""

    def __init__(self, policy_factory: Any, *, host: str, port: int, metadata: Mapping[str, Any], telemetry: Path | None = None):
        self.policy_factory = policy_factory
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="flex-inference")
        self.telemetry = telemetry
        self.host = host
        self.port = int(port)
        self.metadata = dict(metadata)

    def serve_forever(self) -> None:
        asyncio.run(self._run())

    async def _run(self) -> None:
        LOGGER.info("Starting official B1K websocket server on %s:%d", self.host, self.port)
        async with websocket_server.serve(
            self._handler,
            self.host,
            self.port,
            compression=None,
            max_size=None,
            process_request=self._health_check,
            ping_interval=None,
        ):
            await asyncio.Future()

    async def _handler(self, websocket: Any) -> None:
        LOGGER.info("Official evaluator connected from %s", websocket.remote_address)
        from challenge_docker.session import PolicySession
        session = PolicySession(self.policy_factory)
        connection_id = uuid.uuid4().hex
        packer = Packer()
        await websocket.send(packer.pack(self.metadata))
        previous_total = None
        try:
            while True:
                started = time.monotonic()
                result = unpackb(await websocket.recv(), strict_map_key=False)
                if isinstance(result, Mapping) and result.get("reset"):
                    session.reset()
                    continue
                infer_started = time.monotonic()
                queued_ns = time.time_ns()
                def infer():
                    began = time.time_ns()
                    response = session.infer(result)
                    ended = time.time_ns()
                    model_timing = getattr(session.slots[-1].model, "_fastwam_last_timing", {})
                    response["server_timing"] = {
                        **model_timing,
                        "infer_ms": (ended - began) / 1e6,
                        "lock_wait_ms": (began - queued_ns) / 1e6,
                        "model_start_unix_ns": began, "model_end_unix_ns": ended,
                        "batch_size": len(session.slots),
                        "action_chunk_size": session.chunk_size,
                    }
                    return response
                response = await asyncio.get_running_loop().run_in_executor(self.executor, infer)
                if self.telemetry:
                    self.telemetry.parent.mkdir(parents=True, exist_ok=True)
                    with self.telemetry.open("a") as log:
                        log.write(json.dumps(dict(connection_id=connection_id, **response["server_timing"])) + "\n")
                if previous_total is not None:
                    response["server_timing"]["prev_total_ms"] = previous_total * 1000.0
                await websocket.send(packer.pack(response))
                previous_total = time.monotonic() - started
        except websockets.ConnectionClosed:
            LOGGER.info("Official evaluator connection closed")
        except Exception:
            LOGGER.error("Policy websocket handler failed:\n%s", traceback.format_exc())
            try:
                await websocket.close(code=1011, reason="policy inference failed")
            except Exception:
                pass
            raise


    def _health_check(self, connection: Any, request: Any) -> Any:
        if getattr(request, "path", None) != "/healthz":
            return None
        return connection.respond(http.HTTPStatus.OK, json.dumps({"ready": True, "metadata": self.metadata}) + "\n")
