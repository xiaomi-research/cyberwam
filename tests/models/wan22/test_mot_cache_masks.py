# Copyright (c) 2026 Xiaomi Corporation

import types

import torch
import torch.nn as nn

from fastwam.models.wan22.mot import MoT


class _Expert(nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        self.blocks = nn.ModuleList(nn.Identity() for _ in range(num_layers))
        self.num_heads = 1
        self.attn_head_dim = 1


def _make_mot(num_layers=3):
    mot = MoT(
        {"video": _Expert(num_layers), "action": _Expert(num_layers)},
        mot_checkpoint_mixed_attn=False,
    )

    def build_io(self, expert, block, x, freqs, t_mod):
        del expert, block, freqs, t_mod
        unused = torch.zeros_like(x)
        return x, x, x, x, unused, unused, unused, unused, False

    def apply_post(self, **kwargs):
        return kwargs["residual_x"]

    mot._build_expert_attention_io = types.MethodType(build_io, mot)
    mot._apply_post_with_optional_checkpoint = types.MethodType(apply_post, mot)
    return mot


def test_prefill_video_cache_routes_layer_and_batched_masks():
    mot = _make_mot()
    seen = []

    def mixed(self, q_cat, k_cat, v_cat, attention_mask):
        del k_cat, v_cat
        seen.append(attention_mask)
        return q_cat

    mot._mixed_attention = types.MethodType(mixed, mot)
    tokens = torch.zeros(2, 2, 1)
    base = torch.ones(2, 2, dtype=torch.bool)
    layers = [torch.full((2, 2), i + 1, dtype=torch.bool) for i in range(3)]
    mot.prefill_video_cache(
        tokens,
        torch.empty(2),
        torch.empty(1),
        None,
        base,
        layer_video_attention_masks=layers,
    )
    for actual, expected in zip(seen, layers):
        torch.testing.assert_close(actual, expected)

    seen.clear()
    batched_layers = [torch.full((2, 2, 2), i + 1, dtype=torch.bool) for i in range(3)]
    mot.prefill_video_cache(tokens, torch.empty(2), torch.empty(1), None, base, batched_layers)
    for actual, expected in zip(seen, batched_layers):
        torch.testing.assert_close(actual, expected)


def test_cached_action_uses_each_layer_action_query_rows_and_batch_mask():
    mot = _make_mot()
    seen = []

    def mixed(self, q_cat, k_cat, v_cat, attention_mask):
        del k_cat, v_cat
        seen.append(attention_mask)
        return q_cat

    mot._mixed_attention = types.MethodType(mixed, mot)
    video_len, action_len, batch = 2, 1, 2
    total = video_len + action_len
    cache = [{"k": torch.zeros(batch, video_len, 1), "v": torch.zeros(batch, video_len, 1)} for _ in range(3)]
    base = torch.ones(total, total, dtype=torch.bool)
    layers = []
    expected = []
    for layer_idx in range(3):
        mask = torch.zeros(batch, total, total, dtype=torch.bool)
        mask[:, video_len:, :] = layer_idx % 2 == 0
        layers.append(mask)
        expected.append(mask[:, video_len:, :])
    mot.forward_action_with_video_cache(
        torch.zeros(batch, action_len, 1),
        torch.empty(action_len),
        torch.empty(1),
        None,
        cache,
        base,
        video_len,
        layer_attention_masks=layers,
    )
    assert len(seen) == 3
    for actual, wanted in zip(seen, expected):
        torch.testing.assert_close(actual, wanted)


def test_cache_mask_legacy_2d_interface_is_preserved():
    mot = _make_mot(num_layers=1)
    seen = []

    def mixed(self, q_cat, k_cat, v_cat, attention_mask):
        del k_cat, v_cat
        seen.append(attention_mask)
        return q_cat

    mot._mixed_attention = types.MethodType(mixed, mot)
    mot.prefill_video_cache(torch.zeros(1, 2, 1), torch.empty(2), torch.empty(1), None, torch.eye(2, dtype=torch.bool))
    cache = [{"k": torch.zeros(1, 2, 1), "v": torch.zeros(1, 2, 1)}]
    mot.forward_action_with_video_cache(
        torch.zeros(1, 1, 1), torch.empty(1), torch.empty(1), None, cache, torch.eye(3, dtype=torch.bool), 2
    )
    torch.testing.assert_close(seen[0], torch.eye(2, dtype=torch.bool))
    torch.testing.assert_close(seen[1], torch.eye(3, dtype=torch.bool)[2:, :])
