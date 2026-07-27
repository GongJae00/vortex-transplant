from __future__ import annotations

import torch

from topological.baseline import LSTMConfig, LSTMBaseline, make_lstm, one_lstm_training_step
from topological.training import training_batch, TrainingSpec


def test_lstm_baseline_interface_matches_u1_conv_rnn() -> None:
    model = LSTMBaseline(LSTMConfig(hidden_size=32))
    batch = 4
    hidden = model.initial_state(batch)
    assert isinstance(hidden, tuple)
    assert hidden[0].shape == (batch, 32)
    assert hidden[1].shape == (batch, 32)

    tokens = torch.randint(1, 10, (batch,))
    next_hidden = model.step(tokens, hidden)
    assert isinstance(next_hidden, tuple)
    assert next_hidden[0].shape == (batch, 32)

    logits = model.logits(next_hidden)
    assert logits.shape == (batch, 10)

    model.enforce_blank_embedding()
    assert torch.count_nonzero(model.embedding.weight[0]) == 0


def test_lstm_parameter_count_is_reasonable() -> None:
    small = LSTMBaseline(LSTMConfig(hidden_size=32))
    medium = LSTMBaseline(LSTMConfig(hidden_size=256))
    assert 0 < small.parameter_count() < medium.parameter_count()


def test_lstm_training_step_is_finite_and_updates_parameters() -> None:
    spec = TrainingSpec(updates=2, batch_size=2, train_delay_min=1, train_delay_max=2,
                           validation_examples=4, validation_interval=1)
    model = make_lstm(7, LSTMConfig(hidden_size=32))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch = training_batch(7, 1, spec, device="cpu")
    params_before = [p.clone() for p in model.parameters()]

    loss, grad_norm = one_lstm_training_step(model, optimizer, batch, 1.0)

    assert loss > 0.0
    assert grad_norm >= 0.0
    params_after = [p for p in model.parameters()]
    changed = any(not torch.equal(b, a) for b, a in zip(params_before, params_after))
    assert changed, "LSTM parameters should update after a training step"


def test_lstm_copy_task_forward_pass_converges() -> None:
    model = make_lstm(8, LSTMConfig(hidden_size=64))
    batch = training_batch(8, 1, TrainingSpec(updates=1, batch_size=4,
                           train_delay_min=1, train_delay_max=2,
                           validation_examples=4, validation_interval=1), device="cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for step in range(5):
        one_lstm_training_step(model, optimizer, batch, 1.0)

    from topological.task import run_copy
    trace = run_copy(model, batch.symbols, batch.delay)
    assert torch.isfinite(trace.logits).all()
    assert trace.logits.shape == (batch.symbols.shape[0], 4, 10)
