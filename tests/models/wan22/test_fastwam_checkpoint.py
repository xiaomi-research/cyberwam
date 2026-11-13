# Copyright (c) 2026 Xiaomi Corporation

import torch
import torch.nn as nn
import pytest

from fastwam.models.wan22.fastwam import (
    FASTWAM_CHECKPOINT_FORMAT_VERSION,
    FastWAM,
)
from fastwam.utils.checkpoint_contract import (
    CheckpointContract,
    CheckpointContractMismatchError,
)


def _contract(*, action_dim: int = 23) -> CheckpointContract:
    return CheckpointContract(
        compatibility={
            "model": {"variant": "fastwam_uncond", "action_dim": action_dim},
            "data": {"action_horizon": 32, "sample_stride": 2},
        },
        provenance={"git_commit": "test"},
    )


def _model(*, proprio_dim: int | None = 23) -> FastWAM:
    model = FastWAM.__new__(FastWAM)
    nn.Module.__init__(model)
    model.mot = nn.Linear(3, 2)
    model.video_expert = nn.Linear(3, 2)
    model.proprio_dim = proprio_dim
    model.proprio_encoder = None if proprio_dim is None else nn.Linear(proprio_dim, 4)
    model.torch_dtype = torch.float32
    return model


def test_resize_contract_accounts_for_vae_and_dit_spatial_factors():
    assert FastWAM._check_resize_height_width(352, 224, 9) == (352, 224, 9)
    assert FastWAM._check_resize_height_width(336, 224, 9) == (352, 224, 9)


def test_checkpoint_with_contract_round_trip_strict(tmp_path):
    source = _model()
    contract = _contract()
    path = tmp_path / "model.pt"
    source.save_checkpoint(path, step=12, checkpoint_contract=contract)

    payload = torch.load(path, map_location="cpu")
    assert payload["format_version"] == FASTWAM_CHECKPOINT_FORMAT_VERSION
    assert payload["contract"] == contract.to_dict()

    target = _model()
    target.load_checkpoint(
        path,
        expected_contract=contract,
        contract_mode="strict",
    )
    assert target.mot.state_dict().keys() == source.mot.state_dict().keys()
    for key, value in source.mot.state_dict().items():
        torch.testing.assert_close(target.mot.state_dict()[key], value)
    for key, value in source.proprio_encoder.state_dict().items():
        torch.testing.assert_close(target.proprio_encoder.state_dict()[key], value)


@pytest.mark.parametrize("mutation", ["missing", "unexpected"])
def test_strict_load_rejects_missing_or_unexpected_mot_tensor(tmp_path, mutation):
    model = _model()
    contract = _contract()
    path = tmp_path / "model.pt"
    model.save_checkpoint(path, checkpoint_contract=contract)
    payload = torch.load(path, map_location="cpu")
    if mutation == "missing":
        payload["mot"].pop("bias")
    else:
        payload["mot"]["unexpected"] = torch.zeros(1)
    torch.save(payload, path)

    with pytest.raises(RuntimeError, match="Missing key|Unexpected key"):
        model.load_checkpoint(
            path,
            expected_contract=contract,
            contract_mode="strict",
        )


def test_strict_load_rejects_missing_proprio_encoder(tmp_path):
    model = _model()
    contract = _contract()
    path = tmp_path / "model.pt"
    model.save_checkpoint(path, checkpoint_contract=contract)
    payload = torch.load(path, map_location="cpu")
    del payload["proprio_encoder"]
    torch.save(payload, path)

    with pytest.raises(ValueError, match="missing `proprio_encoder`.*proprio_dim=23"):
        model.load_checkpoint(
            path,
            expected_contract=contract,
            contract_mode="strict",
        )


def test_strict_load_rejects_unexpected_proprio_encoder(tmp_path):
    source = _model(proprio_dim=None)
    contract = _contract()
    path = tmp_path / "model.pt"
    source.save_checkpoint(path, checkpoint_contract=contract)
    payload = torch.load(path, map_location="cpu")
    payload["proprio_encoder"] = nn.Linear(23, 4).state_dict()
    torch.save(payload, path)

    with pytest.raises(ValueError, match="unexpected `proprio_encoder`"):
        source.load_checkpoint(
            path,
            expected_contract=contract,
            contract_mode="strict",
        )


def test_strict_load_validates_contract_before_loading_weights(tmp_path):
    source = _model()
    path = tmp_path / "model.pt"
    source.save_checkpoint(path, checkpoint_contract=_contract(action_dim=14))

    target = _model()
    weights_before = {key: value.clone() for key, value in target.mot.state_dict().items()}
    with pytest.raises(CheckpointContractMismatchError, match="action_dim"):
        target.load_checkpoint(
            path,
            expected_contract=_contract(action_dim=23),
            contract_mode="strict",
        )
    for key, value in weights_before.items():
        torch.testing.assert_close(target.mot.state_dict()[key], value)


def test_legacy_mode_loads_official_checkpoint_without_contract(tmp_path):
    source = _model(proprio_dim=None)
    path = tmp_path / "official_legacy.pt"
    torch.save({"mot": source.mot.state_dict(), "step": 3}, path)

    target = _model(proprio_dim=None)
    target.load_checkpoint(path, contract_mode="legacy")
    for key, value in source.mot.state_dict().items():
        torch.testing.assert_close(target.mot.state_dict()[key], value)


def test_strict_mode_rejects_legacy_checkpoint_without_contract(tmp_path):
    model = _model(proprio_dim=None)
    path = tmp_path / "official_legacy.pt"
    torch.save({"mot": model.mot.state_dict()}, path)

    with pytest.raises(ValueError, match="format_version"):
        model.load_checkpoint(
            path,
            expected_contract=_contract(),
            contract_mode="strict",
        )
