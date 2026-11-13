# Copyright (c) 2026 Xiaomi Corporation

import types

import pytest
import torch
import torch.nn as nn

from fastwam.models.wan22.fastwam_flex import FastWAMFlex
from fastwam.models.wan22.mot import MoT


STREAM_NAMES = ("rgb", "dino", "pointmap")


def _build_masks(
    *,
    present=None,
    outputs=None,
    cross_modal=None,
    num_layers=30,
    bottom_local_layers=7,
    top_local_layers=7,
):
    def flags(value, default=True):
        if value is None:
            value = dict.fromkeys(STREAM_NAMES, default)
        if isinstance(value, dict):
            value = [[value[name] for name in STREAM_NAMES]]
        return torch.tensor(value, dtype=torch.bool)

    if cross_modal is None:
        cross_modal = (True, True, True)
    elif isinstance(cross_modal, dict):
        cross_modal = tuple(cross_modal[name] for name in STREAM_NAMES)
    return FastWAMFlex.build_flex_attention_masks(
        stream_lengths=(3, 3, 3),
        anchor_lengths=(1, 1, 1),
        action_seq_len=2,
        num_layers=num_layers,
        bottom_local_layers=bottom_local_layers,
        top_local_layers=top_local_layers,
        present_modalities=flags(present),
        output_modalities=flags(outputs),
        cross_modal_predict=cross_modal,
        device=torch.device("cpu"),
    )


def _allowed(mask, row):
    if mask.ndim == 3:
        mask = mask[0]
    return set(mask[row].nonzero().flatten().tolist())


def test_flex_hbridge_routes_seven_local_sixteen_mixed_seven_local_layers():
    masks = _build_masks()

    assert len(masks) == 30
    local_layers = set(range(7)) | set(range(23, 30))
    mixed_layers = set(range(7, 23))

    for layer in local_layers:
        mask = masks[layer][0]
        # Outer HBridge layers are strictly stream-local, including action.
        assert _allowed(mask, 1) == {0, 1, 2}
        assert _allowed(mask, 4) == {3, 4, 5}
        assert _allowed(mask, 7) == {6, 7, 8}
        assert _allowed(mask, 9) == {9, 10}

    for layer in mixed_layers:
        mask = masks[layer]
        # Middle layers fuse visual futures, and action reads every visual stream.
        assert _allowed(mask, 1) == set(range(9))
        assert _allowed(mask, 9) == set(range(11))


def test_flex_local_anchor_queries_never_read_future_tokens():
    masks = _build_masks()

    for layer in (*range(7), *range(23, 30)):
        assert _allowed(masks[layer], 0) == {0}
        assert _allowed(masks[layer], 3) == {3}
        assert _allowed(masks[layer], 6) == {6}


def test_flex_m_out_controls_middle_action_and_future_visibility_only():
    masks = _build_masks(
        outputs={"rgb": True, "dino": False, "pointmap": False},
    )
    middle = masks[7][0]
    outer = masks[6][0]

    # Action always sees present anchors and its own stream. It sees only the
    # RGB future selected by m_out, not DINO/pointmap futures.
    assert _allowed(middle, 9) == {0, 1, 2, 3, 6, 9, 10}
    # Future streams with equal m_out values remain mutually visible. RGB is
    # isolated from the two unselected future streams.
    assert _allowed(middle, 1) == {0, 1, 2, 3, 6}
    assert _allowed(middle, 4) == {0, 3, 4, 5, 6, 7, 8}
    assert _allowed(middle, 7) == {0, 3, 4, 5, 6, 7, 8}
    # m_out does not alter HBridge's strictly local outer layers.
    assert _allowed(outer, 9) == {9, 10}


def test_flex_missing_input_with_cross_modal_forcing_masks_only_its_anchor():
    masks = _build_masks(
        present={"rgb": True, "dino": False, "pointmap": True},
        cross_modal={"rgb": True, "dino": True, "pointmap": True},
    )
    middle = masks[7][0]

    # The absent DINO anchor is neither a query nor a key.
    assert not middle[3].any()
    assert not middle[:, 3].any()
    # Its future tokens remain active and can be synthesized from other
    # modalities; cross-modal forcing therefore keeps their denoising loss valid.
    assert middle[4].any()
    assert middle[:, 4].any()
    assert 0 in _allowed(middle, 4)
    assert 6 in _allowed(middle, 4)


def test_flex_missing_input_without_cross_modal_forcing_disables_whole_stream():
    masks = _build_masks(
        present={"rgb": True, "dino": False, "pointmap": True},
        cross_modal={"rgb": True, "dino": False, "pointmap": True},
    )

    for batched_mask in masks:
        mask = batched_mask[0]
        assert not mask[3:6].any()
        assert not mask[:, 3:6].any()


def test_flex_hbridge_layer_counts_are_configurable():
    masks = _build_masks(num_layers=6, bottom_local_layers=1, top_local_layers=2)

    assert len(masks) == 6
    assert _allowed(masks[0], 9) == {9, 10}
    assert _allowed(masks[1], 9) == set(range(11))
    assert _allowed(masks[3], 9) == set(range(11))
    assert _allowed(masks[4], 9) == {9, 10}
    assert _allowed(masks[5], 9) == {9, 10}


def test_flex_masks_apply_m_in_and_m_out_independently_per_sample():
    masks = _build_masks(
        present=[[True, True, True], [True, False, True]],
        outputs=[[True, True, True], [False, True, False]],
    )

    assert masks[7].shape == (2, 11, 11)
    # Sample 0 is full joint. Sample 1 hides the DINO anchor and lets action
    # read only the DINO future selected by its independently sampled m_out.
    assert _allowed(masks[7][0], 9) == set(range(11))
    assert _allowed(masks[7][1], 9) == {0, 4, 5, 6, 9, 10}
    assert not masks[7][1, 3].any()
    assert not masks[7][1, :, 3].any()


def test_flex_modality_sampling_is_reproducible_and_never_drops_all_inputs():
    generator_a = torch.Generator().manual_seed(123)
    generator_b = torch.Generator().manual_seed(123)

    sampled_a = FastWAMFlex.sample_modality_masks(
        batch_size=256,
        present_probability=(0.1, 0.1, 0.1),
        output_probability=(0.2, 0.5, 0.8),
        device=torch.device("cpu"),
        generator=generator_a,
    )
    sampled_b = FastWAMFlex.sample_modality_masks(
        batch_size=256,
        present_probability=(0.1, 0.1, 0.1),
        output_probability=(0.2, 0.5, 0.8),
        device=torch.device("cpu"),
        generator=generator_b,
    )

    m_in, m_out = sampled_a
    assert m_in.shape == (256, 3)
    assert m_out.shape == (256, 3)
    assert m_in.dtype == torch.bool
    assert m_out.dtype == torch.bool
    assert m_in.any(dim=1).all()
    torch.testing.assert_close(m_in, sampled_b[0])
    torch.testing.assert_close(m_out, sampled_b[1])


def test_flex_modality_sampling_respects_deterministic_probabilities():
    m_in, m_out = FastWAMFlex.sample_modality_masks(
        batch_size=4,
        present_probability=(0.0, 0.0, 1.0),
        output_probability=(1.0, 0.0, 1.0),
        device=torch.device("cpu"),
    )

    assert torch.equal(m_in, torch.tensor([[False, False, True]]).expand(4, -1))
    assert torch.equal(m_out, torch.tensor([[True, False, True]]).expand(4, -1))


def test_flex_per_batch_modality_sampling_shares_one_route():
    m_in, m_out = FastWAMFlex.sample_modality_masks(
        batch_size=32,
        present_probability=(0.5, 0.5, 0.5),
        output_probability=(0.5, 0.5, 0.5),
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(123),
        sampling_scope="per_batch",
    )

    assert torch.equal(m_in, m_in[:1].expand_as(m_in))
    assert torch.equal(m_out, m_out[:1].expand_as(m_out))
    assert m_in.any(dim=1).all()


def test_flex_fixed_all_modalities_ignores_sampling_probabilities():
    m_in, m_out = FastWAMFlex.sample_modality_masks(
        batch_size=4,
        present_probability=(0.0, 0.0, 0.0),
        output_probability=(0.0, 0.0, 0.0),
        device=torch.device("cpu"),
        sampling_scope="fixed_all_modalities",
    )

    assert m_in.all()
    assert m_out.all()


def test_flex_rejects_unknown_routing_sampling_scope():
    with pytest.raises(ValueError, match="routing_sampling_scope"):
        FastWAMFlex.sample_modality_masks(
            batch_size=1,
            present_probability=(0.5, 0.5, 0.5),
            output_probability=(0.5, 0.5, 0.5),
            device=torch.device("cpu"),
            sampling_scope="per_worker",
        )


def test_flex_shared_routing_builds_one_2d_attention_mask_per_layer():
    model = FastWAMFlex.__new__(FastWAMFlex)
    model.mot = types.SimpleNamespace(num_layers=4)
    model.hbridge_enabled = True
    model.hbridge_bottom_ratio = 0.25
    model.hbridge_top_ratio = 0.25
    model.cross_modal_predict = (True, True, True)
    routes = torch.tensor([[True, False, True], [True, False, True]])

    base, layers = model._build_routed_attention_masks(
        stream_lengths=(3, 3, 3),
        anchor_lengths=(1, 1, 1),
        action_seq_len=2,
        device=torch.device("cpu"),
        routing={"present": routes, "output": routes, "shared": True},
    )

    assert base.shape == (11, 11)
    assert all(mask.shape == (11, 11) for mask in layers)


class _RoutingExpert(nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        self.blocks = nn.ModuleList(nn.Identity() for _ in range(num_layers))
        self.num_heads = 1
        self.attn_head_dim = 1


def _make_routing_mot(num_layers=3):
    mot = MoT(
        {"video": _RoutingExpert(num_layers), "action": _RoutingExpert(num_layers)},
        mot_checkpoint_mixed_attn=False,
    )
    seen_masks = []

    def build_io(self, expert, block, x, freqs, t_mod):
        del expert, block, freqs, t_mod
        unused = torch.zeros_like(x)
        return x, x, x, x, unused, unused, unused, unused, False

    def mixed(self, q_cat, k_cat, v_cat, attention_mask):
        del k_cat, v_cat
        seen_masks.append(attention_mask.clone())
        return q_cat

    def apply_post(self, **kwargs):
        return kwargs["residual_x"]

    mot._build_expert_attention_io = types.MethodType(build_io, mot)
    mot._mixed_attention = types.MethodType(mixed, mot)
    mot._apply_post_with_optional_checkpoint = types.MethodType(apply_post, mot)
    return mot, seen_masks


def _run_routing_mot(mot, *, base_mask, layer_masks=None):
    tokens = {"video": torch.zeros(1, 2, 1), "action": torch.zeros(1, 1, 1)}
    return mot(
        embeds_all=tokens,
        attention_mask=base_mask,
        freqs_all={"video": torch.empty(2), "action": torch.empty(1)},
        context_all={"video": None, "action": None},
        t_mod_all={"video": torch.empty(1), "action": torch.empty(1)},
        layer_attention_masks=layer_masks,
    )


def test_mot_uses_the_corresponding_attention_mask_at_each_layer():
    mot, seen = _make_routing_mot(num_layers=3)
    base = torch.ones(3, 3, dtype=torch.bool)
    layer_masks = [torch.eye(3, dtype=torch.bool)]
    layer_masks += [torch.tril(base), torch.triu(base)]

    output = _run_routing_mot(mot, base_mask=base, layer_masks=layer_masks)

    assert len(seen) == 3
    for actual, expected in zip(seen, layer_masks):
        torch.testing.assert_close(actual, expected)
    assert output["video"].shape == (1, 2, 1)
    assert output["action"].shape == (1, 1, 1)


def test_mot_without_layer_masks_keeps_the_legacy_mask_on_every_layer():
    mot, seen = _make_routing_mot(num_layers=3)
    base = torch.tensor(
        [[True, False, False], [True, True, False], [True, True, True]]
    )

    _run_routing_mot(mot, base_mask=base)

    assert len(seen) == 3
    assert all(torch.equal(mask, base) for mask in seen)


@pytest.mark.parametrize(
    ("layer_masks", "error", "match"),
    [
        ([torch.ones(3, 3, dtype=torch.bool)], ValueError, "one mask per MoT layer"),
        (
            [torch.ones(3, 3, dtype=torch.bool), object()],
            TypeError,
            r"layer_attention_masks\[1\].*tensor",
        ),
        (
            [torch.ones(3, 3, dtype=torch.bool), torch.ones(3, 3, 1)],
            ValueError,
            r"layer_attention_masks\[1\].*shape",
        ),
        (
            [torch.ones(3, 3, dtype=torch.bool), torch.ones(2, 2)],
            ValueError,
            r"layer_attention_masks\[1\].*shape",
        ),
    ],
)
def test_mot_rejects_invalid_per_layer_attention_masks(layer_masks, error, match):
    mot, _ = _make_routing_mot(num_layers=2)
    with pytest.raises(error, match=match):
        _run_routing_mot(
            mot,
            base_mask=torch.ones(3, 3, dtype=torch.bool),
            layer_masks=layer_masks,
        )
