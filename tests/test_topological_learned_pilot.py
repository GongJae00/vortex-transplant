from __future__ import annotations

import torch
import pytest

from topological.learned_pilot import (
    checkpoint_bytes,
    checkpoint_roundtrip_exact,
    canonical_contract_path,
    hash_chain,
    initialize_cuda_device,
    payload_sha256,
)
from topological.training import make_model


def test_hash_chain_is_order_sensitive_and_repeatable() -> None:
    first = "00" * 32
    second = "11" * 32

    assert hash_chain([first, second]) == hash_chain([first, second])
    assert hash_chain([first, second]) != hash_chain([second, first])
    assert payload_sha256({"b": 2, "a": 1}) == payload_sha256({"a": 1, "b": 2})
    assert canonical_contract_path().is_file()


def test_checkpoint_roundtrip_preserves_every_tensor_exactly() -> None:
    model = make_model(3, device="cpu")
    payload = checkpoint_bytes(model, {"seed": 3})

    assert checkpoint_roundtrip_exact(payload, 3)
    saved = torch.load(__import__("io").BytesIO(payload), map_location="cpu", weights_only=True)
    assert saved["metadata"] == {"seed": 3}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_canonical_cuda_bootstrap_uses_explicit_device_zero() -> None:
    device = initialize_cuda_device()

    assert device == torch.device("cuda:0")
    assert torch.cuda.current_device() == 0
