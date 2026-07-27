from __future__ import annotations

import torch

from topological.model import ModelSpec
from topological.training import (
    TrainingSpec,
    TrainingSnapshot,
    _is_better,
    make_model,
    one_training_step,
    train_seed,
    training_batch,
    training_delay,
    validate_model,
)


def _training_spec() -> TrainingSpec:
    return TrainingSpec(
        updates=2,
        batch_size=2,
        train_delay_min=1,
        train_delay_max=2,
        validation_examples=4,
        validation_interval=1,
    )


def test_training_delay_and_batches_are_frozen_by_namespace() -> None:
    spec = _training_spec()

    assert training_delay(3, 7, spec) == training_delay(3, 7, spec)
    first = training_batch(3, 7, spec, device="cpu")
    repeat = training_batch(3, 7, spec, device="cpu")
    other = training_batch(3, 8, spec, device="cpu")
    assert torch.equal(first.symbols, repeat.symbols)
    assert first.content_sha256 == repeat.content_sha256
    assert first.content_sha256 != other.content_sha256


def test_one_training_step_is_finite_and_keeps_blank_embedding_zero() -> None:
    model = make_model(4, model_spec=ModelSpec(height=4, width=4, channels=2))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch = training_batch(4, 1, _training_spec(), device="cpu")

    loss, gradient_norm = one_training_step(model, optimizer, batch, 1.0)

    assert loss > 0.0
    assert gradient_norm >= 0.0
    assert torch.count_nonzero(model.token_embedding.weight[0]) == 0


def test_validation_is_repeatable_and_checkpoint_order_is_lexicographic() -> None:
    spec = _training_spec()
    model = make_model(5, model_spec=ModelSpec(height=4, width=4, channels=2))

    first = validate_model(model, 5, 1, spec)
    repeat = validate_model(model, 5, 1, spec)
    later = type(first)(
        update=2,
        accuracy=first.accuracy,
        cross_entropy=first.cross_entropy,
        example_count=first.example_count,
        delay_accuracies=first.delay_accuracies,
        amplitude_saturation=first.amplitude_saturation,
        content_sha256=first.content_sha256,
    )

    assert first == repeat
    assert _is_better(first, None)
    assert not _is_better(later, first)


def test_tiny_training_selects_one_complete_checkpoint_deterministically() -> None:
    model_spec = ModelSpec(height=4, width=4, channels=2)
    first = train_seed(6, model_spec=model_spec, training_spec=_training_spec())
    repeat = train_seed(6, model_spec=model_spec, training_spec=_training_spec())

    assert first.selected_update == repeat.selected_update
    assert first.selected_accuracy == repeat.selected_accuracy
    assert first.selected_cross_entropy == repeat.selected_cross_entropy
    assert first.update_count == first.finite_gradient_steps == 2
    assert first.train_hashes == repeat.train_hashes
    assert len(first.history) == 2


def test_training_log_records_loss_gradient_and_learning_rate() -> None:
    spec = _training_spec()
    model_spec = ModelSpec(height=4, width=4, channels=2)
    result = train_seed(9, model_spec=model_spec, training_spec=spec)

    assert len(result.training_log) == len(result.history) == 2
    for snapshot in result.training_log:
        assert isinstance(snapshot, TrainingSnapshot)
        assert snapshot.loss > 0.0
        assert snapshot.gradient_norm >= 0.0
        assert snapshot.learning_rate == 0.001
    assert result.training_log[-1].update == 2


def test_reverse_training_executes_finite_loss_and_deterministically() -> None:
    spec = _training_spec()
    model_spec = ModelSpec(height=4, width=4, channels=2)

    first = train_seed(10, model_spec=model_spec, training_spec=spec, task_type="reverse")
    repeat = train_seed(10, model_spec=model_spec, training_spec=spec, task_type="reverse")

    assert first.selected_accuracy >= 0.0
    assert first.selected_accuracy == repeat.selected_accuracy
    assert len(first.history) == 2


def test_variable_length_training_executes_finite_loss() -> None:
    spec = _training_spec()
    model_spec = ModelSpec(height=4, width=4, channels=2)

    result = train_seed(11, model_spec=model_spec, training_spec=spec, variable_length=True)

    assert result.selected_accuracy >= 0.0
    assert len(result.history) == 2
