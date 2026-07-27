from __future__ import annotations

import torch

from topological.model import ModelSpec, PlainConvRNN, U1ConvRNN


def _u1() -> U1ConvRNN:
    generator = torch.Generator().manual_seed(17)
    return U1ConvRNN(ModelSpec(height=6, width=6, channels=3), generator=generator)


def _plain() -> PlainConvRNN:
    generator = torch.Generator().manual_seed(17)
    return PlainConvRNN(ModelSpec(height=6, width=6, channels=3), generator=generator)


def test_recurrent_update_is_u1_equivariant() -> None:
    model = _u1()
    hidden = torch.randn(2, 2, 3, 6, 6)
    angle = torch.tensor(0.73)
    cosine, sine = torch.cos(angle), torch.sin(angle)
    rotated = torch.stack(
        (
            cosine * hidden[:, 0] - sine * hidden[:, 1],
            sine * hidden[:, 0] + cosine * hidden[:, 1],
        ),
        dim=1,
    )
    expected = model.radial_tanh(model.recurrent_linear(hidden))
    expected_rotated = torch.stack(
        (
            cosine * expected[:, 0] - sine * expected[:, 1],
            sine * expected[:, 0] + cosine * expected[:, 1],
        ),
        dim=1,
    )

    actual = model.radial_tanh(model.recurrent_linear(rotated))

    relative = torch.linalg.vector_norm(actual - expected_rotated) / torch.linalg.vector_norm(expected_rotated)
    assert float(relative.detach()) <= 1e-6


def test_blank_embedding_is_exactly_zero_and_has_no_gradient() -> None:
    model = _u1()
    tokens = torch.zeros(4, dtype=torch.long)
    embedded = model.embedded_input(tokens)
    embedded.sum().backward()

    assert torch.count_nonzero(embedded) == 0
    assert torch.count_nonzero(model.token_embedding.weight.grad[0]) == 0


def test_frozen_model_parameter_and_state_accounting() -> None:
    model = U1ConvRNN(ModelSpec(), generator=torch.Generator().manual_seed(3))

    assert model.parameter_count() == 83_082
    assert model.real_state_bytes() == 16_384
    assert model.initial_state(5).shape == (5, 2, 8, 16, 16)
    assert model.kernel_real.shape == (8, 8, 3, 3)
    assert model.kernel_imag.shape == (8, 8, 3, 3)


def test_sigmoid_gate_nonlinearity_produces_finite_output() -> None:
    spec = ModelSpec(height=6, width=6, channels=3, nonlinearity="sigmoid_gate")
    model = U1ConvRNN(spec, generator=torch.Generator().manual_seed(17))
    hidden = torch.randn(2, 2, 3, 6, 6)
    tokens = torch.randint(1, 10, (2,))

    next_hidden = model.step(tokens, hidden)

    assert torch.isfinite(next_hidden).all()
    assert next_hidden.shape == hidden.shape


def test_sigmoid_gate_nonlinearity_is_also_u1_equivariant() -> None:
    spec = ModelSpec(height=6, width=6, channels=3, nonlinearity="sigmoid_gate")
    model = U1ConvRNN(spec, generator=torch.Generator().manual_seed(17))
    hidden = torch.randn(2, 2, 3, 6, 6)
    angle = torch.tensor(0.73)
    cosine, sine = torch.cos(angle), torch.sin(angle)
    rotated = torch.stack(
        (
            cosine * hidden[:, 0] - sine * hidden[:, 1],
            sine * hidden[:, 0] + cosine * hidden[:, 1],
        ),
        dim=1,
    )
    expected = model.radial_sigmoid_gate(model.recurrent_linear(hidden))
    expected_rotated = torch.stack(
        (
            cosine * expected[:, 0] - sine * expected[:, 1],
            sine * expected[:, 0] + cosine * expected[:, 1],
        ),
        dim=1,
    )

    actual = model.radial_sigmoid_gate(model.recurrent_linear(rotated))

    relative = torch.linalg.vector_norm(actual - expected_rotated) / torch.linalg.vector_norm(expected_rotated)
    assert float(relative.detach()) <= 1e-6


def test_invalid_nonlinearity_raises() -> None:
    try:
        ModelSpec(nonlinearity="invalid").validate()
        assert False, "should have raised"
    except ValueError as error:
        assert "nonlinearity" in str(error).lower()


def test_plain_conv_rnn_interface_matches_u1() -> None:
    model = _plain()
    batch = 4
    hidden = model.initial_state(batch)
    assert hidden.shape == (batch, 2, 3, 6, 6)
    tokens = torch.randint(1, 10, (batch,))
    next_hidden = model.step(tokens, hidden)
    assert next_hidden.shape == (batch, 2, 3, 6, 6)
    logits = model.logits(next_hidden)
    assert logits.shape == (batch, 10)
    model.enforce_blank_embedding()
    assert torch.count_nonzero(model.token_embedding.weight[0]) == 0
    assert model.parameter_count() > 0


def test_plain_conv_rnn_does_not_equivary_with_global_rotation() -> None:
    model = _plain()
    hidden = torch.randn(2, 2, 3, 6, 6)
    angle = torch.tensor(0.73)
    cosine, sine = torch.cos(angle), torch.sin(angle)
    rotated = torch.stack(
        (
            cosine * hidden[:, 0] - sine * hidden[:, 1],
            sine * hidden[:, 0] + cosine * hidden[:, 1],
        ),
        dim=1,
    )
    tokens = torch.randint(1, 10, (2,))
    actual = model.step(tokens, rotated)
    expected = model.step(tokens, hidden)
    expected_rotated = torch.stack(
        (
            cosine * expected[:, 0] - sine * expected[:, 1],
            sine * expected[:, 0] + cosine * expected[:, 1],
        ),
        dim=1,
    )
    relative = torch.linalg.vector_norm(actual - expected_rotated) / torch.linalg.vector_norm(
        expected_rotated
    )
    assert float(relative.detach()) > 1e-3
