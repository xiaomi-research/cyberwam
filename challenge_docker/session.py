# Copyright (c) 2026 Xiaomi Corporation

"""Official v3.9.3 batched protocol over the per-slot policy execution."""
from collections.abc import Mapping
import numpy as np

from challenge_docker.task_guard import expected_batch_size


class PolicySession:
    def __init__(self, factory):
        self.factory = factory
        self.reset()

    def reset(self):
        self.slots = []
        self.tasks = None
        self.chunk_size = None

    def infer(self, request):
        if not isinstance(request, Mapping):
            raise TypeError('observation must be a mapping')
        tasks = np.asarray(request.get('task_id'))
        if tasks.ndim != 2 or tasks.shape[1] != 1 or tasks.dtype.kind not in 'iu' or not len(tasks):
            raise ValueError('task_id must be integer [B,1]')
        batch = len(tasks)
        configured_batch = expected_batch_size()
        if configured_batch is not None and batch != configured_batch:
            raise ValueError(
                f"model task guard requires batch_size={configured_batch}, got {batch}"
            )
        k = request.get('__action_chunk_size__', 0)
        if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k not in (0, 32):
            raise ValueError('FLEX supports K=0 or K=32; K32 is required for deployment')
        if self.tasks is not None and (not np.array_equal(tasks, self.tasks) or k != self.chunk_size):
            raise ValueError('batch/tasks/K changed without reset')
        # Validate every field before advancing any slot's RNG/execution state.
        observations = [{} for _ in range(batch)]
        for key, value in request.items():
            if key.startswith('__'):
                continue
            array = np.asarray(value)
            if array.ndim == 0 or array.shape[0] != batch:
                raise ValueError(f'{key} must have a leading batch dimension {batch}: {array.shape}')
            for slot, row in zip(observations, array):
                slot[key] = row
        if self.tasks is None:
            self.slots = [self.factory() for _ in range(batch)]
            self.tasks = tasks.copy()
            self.chunk_size = k
        chunks = []
        for policy, observation in zip(self.slots, observations):
            if k and policy.queue:
                raise RuntimeError('chunk request encountered an undrained queue')
            # act() calls the model once; the remaining ticks drain the original
            # execution queue. This preserves full-horizon speed2x boundaries.
            chunk = np.stack([policy.act(observation) for _ in range(k or 1)])
            if k and policy.queue:
                raise RuntimeError('execution did not produce exactly 32 native ticks')
            chunks.append(chunk)
        output = np.stack(chunks).astype(np.float32)
        if output.shape != (batch, k or 1, 23) or not np.isfinite(output).all():
            raise ValueError('invalid action chunk')
        response = {'action': output[:, 0]}
        if k:
            response['action_chunk'] = output
        return response
