from __future__ import annotations

import torch

from topological.model import ModelSpec, U1ConvRNN
from topological.task import (
    continue_copy,
    donor_sequences,
    generate_copy_batch,
    generate_reverse_copy_batch,
    run_copy,
    write_copy,
)


def test_copy_batches_are_hash_separated_and_repeatable() -> None:
    first = generate_copy_batch(0, "train/7", 8, 5)
    repeat = generate_copy_batch(0, "train/7", 8, 5)
    validation = generate_copy_batch(0, "validation", 8, 5)

    assert torch.equal(first.symbols, repeat.symbols)
    assert first.content_sha256 == repeat.content_sha256
    assert first.content_sha256 != validation.content_sha256


def test_copy_trace_and_post_write_continuation_agree() -> None:
    model = U1ConvRNN(
        ModelSpec(height=5, width=5, channels=2),
        generator=torch.Generator().manual_seed(9),
    )
    batch = generate_copy_batch(2, "unit", 3, 6)

    trace = run_copy(model, batch.symbols, batch.delay)
    logits, pre_go = continue_copy(model, trace.post_write, batch.delay)

    assert trace.logits.shape == (3, 4, 10)
    assert torch.equal(trace.logits, logits)
    assert torch.equal(trace.pre_go, pre_go)
    assert torch.isfinite(trace.final_state).all()
    assert torch.equal(trace.post_write, write_copy(model, batch.symbols))


def test_donor_catalog_has_eight_unique_fully_mismatched_sequences() -> None:
    recipients = torch.tensor([[1, 2, 3, 4], [8, 7, 6, 5]])

    donors = donor_sequences(recipients)

    assert donors.shape == (2, 8, 4)
    assert torch.all(donors != recipients[:, None, :])
    assert all(len(torch.unique(row, dim=0)) == 8 for row in donors)


def test_reverse_copy_batch_is_frozen_and_isolated_from_copy() -> None:
    reverse = generate_reverse_copy_batch(0, "unit/reverse", 4, 3)
    copy = generate_copy_batch(0, "unit/reverse", 4, 3)

    assert reverse.symbols.shape == (4, 4)
    assert reverse.copy_length == 4
    assert not torch.equal(reverse.symbols, copy.symbols)


def test_run_copy_with_variable_length_produces_matching_logits_shape() -> None:
    model = U1ConvRNN(
        ModelSpec(height=4, width=4, channels=2),
        generator=torch.Generator().manual_seed(9),
    )
    for length in (2, 4, 6, 8):
        batch = generate_copy_batch(3, f"unit/length-{length}", 2, 3, copy_length=length)

        trace = run_copy(model, batch.symbols, batch.delay, copy_length=length)

        assert trace.logits.shape == (2, length, 10)
        assert torch.isfinite(trace.logits).all()
        assert trace.post_write.shape == (2, 2, 2, 4, 4)
