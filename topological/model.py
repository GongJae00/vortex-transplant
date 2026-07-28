"""U(1)-equivariant convolutional recurrent network."""

from __future__ import annotations

from dataclasses import dataclass

import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelSpec:
    height: int = 16
    width: int = 16
    channels: int = 8
    vocabulary: int = 10
    kernel_size: int = 3
    radial_epsilon: float = 1e-8
    nonlinearity: str = "tanh"

    def validate(self) -> None:
        if min(self.height, self.width, self.channels) <= 0:
            raise ValueError("model dimensions must be positive")
        if self.vocabulary != 10 or self.kernel_size != 3:
            raise ValueError("vocabulary and kernel size are frozen vocabulary 10 and kernel size 3")
        if self.nonlinearity not in ("tanh", "sigmoid_gate"):
            raise ValueError("nonlinearity must be 'tanh' or 'sigmoid_gate'")

    @property
    def real_state_size(self) -> int:
        return 2 * self.channels * self.height * self.width


class U1ConvRNN(nn.Module):
    def __init__(
        self,
        spec: ModelSpec = ModelSpec(),
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__()
        spec.validate()
        self.spec = spec
        self.token_embedding = nn.Embedding(
            spec.vocabulary,
            spec.real_state_size,
            padding_idx=0,
        )
        kernel_shape = (spec.channels, spec.channels, spec.kernel_size, spec.kernel_size)
        self.kernel_real = nn.Parameter(torch.empty(kernel_shape))
        self.kernel_imag = nn.Parameter(torch.empty(kernel_shape))
        self.readout = nn.Linear(spec.real_state_size, spec.vocabulary)
        self.reset_parameters(generator=generator)

    def reset_parameters(self, *, generator: torch.Generator | None = None) -> None:
        with torch.no_grad():
            fan = self.spec.channels * self.spec.kernel_size * self.spec.kernel_size
            xavier_std = math.sqrt(2.0 / (2 * fan))
            scale = 0.1
            self.kernel_real.normal_(mean=0.0, std=xavier_std * scale, generator=generator)
            self.kernel_imag.normal_(mean=0.0, std=xavier_std * scale, generator=generator)
            diagonal = torch.arange(self.spec.channels)
            self.kernel_real[diagonal, diagonal, 1, 1] += 1.0
            self.token_embedding.weight.normal_(mean=0.0, std=0.001, generator=generator)
            self.token_embedding.weight[0].zero_()
        self.readout.reset_parameters()

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        parameter = self.kernel_real
        return torch.zeros(
            batch_size,
            2,
            self.spec.channels,
            self.spec.height,
            self.spec.width,
            device=device or parameter.device,
            dtype=dtype or parameter.dtype,
        )

    def recurrent_linear(self, hidden: torch.Tensor) -> torch.Tensor:
        expected = (2, self.spec.channels, self.spec.height, self.spec.width)
        if hidden.ndim != 5 or tuple(hidden.shape[1:]) != expected:
            raise ValueError(f"hidden state must have trailing shape {expected}")
        real = F.pad(hidden[:, 0], (1, 1, 1, 1), mode="circular")
        imag = F.pad(hidden[:, 1], (1, 1, 1, 1), mode="circular")
        real_out = F.conv2d(real, self.kernel_real) - F.conv2d(imag, self.kernel_imag)
        imag_out = F.conv2d(real, self.kernel_imag) + F.conv2d(imag, self.kernel_real)
        return torch.stack((real_out, imag_out), dim=1)

    def radial_tanh(self, field: torch.Tensor) -> torch.Tensor:
        radius = torch.sqrt(torch.sum(field.square(), dim=1, keepdim=True))
        scale = torch.tanh(radius) / (radius + self.spec.radial_epsilon)
        return field * scale

    def radial_sigmoid_gate(self, field: torch.Tensor) -> torch.Tensor:
        radius = torch.sqrt(torch.sum(field.square(), dim=1, keepdim=True))
        scale = radius * torch.sigmoid(radius) / (radius + self.spec.radial_epsilon)
        return field * scale

    def embedded_input(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 1:
            raise ValueError("tokens must be a one-dimensional batch")
        embedded = self.token_embedding(tokens)
        return embedded.reshape(
            len(tokens),
            2,
            self.spec.channels,
            self.spec.height,
            self.spec.width,
        )

    def step(self, tokens: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        pre_activation = self.recurrent_linear(hidden) + self.embedded_input(tokens)
        if self.spec.nonlinearity == "tanh":
            return self.radial_tanh(pre_activation)
        return self.radial_sigmoid_gate(pre_activation)

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.readout(hidden.flatten(start_dim=1))

    def enforce_blank_embedding(self) -> None:
        with torch.no_grad():
            self.token_embedding.weight[0].zero_()

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def real_state_bytes(self, dtype: torch.dtype = torch.float32) -> int:
        element_size = torch.empty((), dtype=dtype).element_size()
        return self.spec.real_state_size * element_size


class PlainConvRNN(nn.Module):
    """Non-equivariant convolutional RNN baseline.

    Same hidden shape (2, channels, height, width) as U1ConvRNN but with
    a standard real-valued convolution — no complex cross-coupling, no U(1)
    equivariance.  Uses tanh + LayerNorm instead of U1ConvRNN's radial_tanh
    (which provides inherent amplitude normalization through its radial
    scaling).  This asymmetry is intentional: radial_tanh is part of the
    U(1)-equivariant design; PlainConvRNN compensates with a standard
    normalization layer that is functionally analogous but architecturally
    non-equivariant.

    Kernel initialization matches U1ConvRNN (scaled normal with identity
    offset) to ensure a fair comparison baseline.
    """

    def __init__(
        self,
        spec: ModelSpec = ModelSpec(),
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__()
        spec.validate()
        self.spec = spec
        self.token_embedding = nn.Embedding(
            spec.vocabulary,
            spec.real_state_size,
            padding_idx=0,
        )
        in_features = spec.channels * 2
        self.conv = nn.Conv2d(
            in_features, in_features, spec.kernel_size,
            padding=spec.kernel_size // 2, padding_mode="circular", bias=False,
        )
        self.norm = nn.LayerNorm([in_features, spec.height, spec.width])
        self.readout = nn.Linear(spec.real_state_size, spec.vocabulary)
        self.reset_parameters(generator=generator)

    def reset_parameters(self, *, generator: torch.Generator | None = None) -> None:
        with torch.no_grad():
            fan = self.spec.channels * 2 * self.spec.kernel_size * self.spec.kernel_size
            xavier_std = math.sqrt(2.0 / (2 * fan))
            scale = 0.1
            nn.init.normal_(self.conv.weight, mean=0.0, std=xavier_std * scale)
            diagonal = torch.arange(self.spec.channels * 2)
            self.conv.weight[diagonal, diagonal, 1, 1] += 1.0
            self.token_embedding.weight.normal_(mean=0.0, std=0.001, generator=generator)
            self.token_embedding.weight[0].zero_()
        self.readout.reset_parameters()

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        parameter = next(self.parameters())
        return torch.zeros(
            batch_size,
            2,
            self.spec.channels,
            self.spec.height,
            self.spec.width,
            device=device or parameter.device,
            dtype=dtype or parameter.dtype,
        )

    def embedded_input(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 1:
            raise ValueError("tokens must be a one-dimensional batch")
        embedded = self.token_embedding(tokens)
        return embedded.reshape(
            len(tokens),
            2,
            self.spec.channels,
            self.spec.height,
            self.spec.width,
        )

    def step(self, tokens: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        batch = len(tokens)
        flat = hidden.reshape(batch, self.spec.channels * 2, self.spec.height, self.spec.width)
        convolved = self.conv(flat)
        normed = self.norm(convolved)
        embedded = self.embedded_input(tokens)
        pre = normed + embedded.reshape(batch, self.spec.channels * 2, self.spec.height, self.spec.width)
        activated = torch.tanh(pre)
        return activated.reshape(batch, 2, self.spec.channels, self.spec.height, self.spec.width)

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.readout(hidden.flatten(start_dim=1))

    def enforce_blank_embedding(self) -> None:
        with torch.no_grad():
            self.token_embedding.weight[0].zero_()

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def real_state_bytes(self, dtype: torch.dtype = torch.float32) -> int:
        element_size = torch.empty((), dtype=dtype).element_size()
        return self.spec.real_state_size * element_size


# ── V2 extensions: C=1 gateway, factorial 2x2 baseline variants ──

import types
import torch.nn.functional as F


def make_scalar_u1_model(model_spec=None, generator=None, device=None):
    """C=1 U1ConvRNN."""
    from dataclasses import replace
    if model_spec is None:
        model_spec = ModelSpec(channels=1)
    else:
        model_spec = replace(model_spec, channels=1)
    model = U1ConvRNN(model_spec, generator=generator)
    if device is not None:
        model = model.to(device)
    return model


def make_factorial_model(variant, model_spec=None, generator=None, device=None):
    """Factory for factorial 2x2 baseline variants."""
    if model_spec is None:
        model_spec = ModelSpec()
    if variant == "U1CommutingLinear_RadialNonlinear":
        model = U1ConvRNN(model_spec, generator=generator)
    elif variant == "U1CommutingLinear_ElementwiseNonlinear":
        model = U1ConvRNN(model_spec, generator=generator)
        _override_step_to_elementwise(model)
    elif variant == "UnrestrictedLinear_RadialNonlinear":
        model = PlainConvRNN(model_spec, generator=generator)
        _override_step_to_radial(model)
    elif variant == "UnrestrictedLinear_ElementwiseNonlinear":
        model = PlainConvRNN(model_spec, generator=generator)
    else:
        raise ValueError(f"Unknown variant '{variant}'")
    if device is not None:
        model = model.to(device)
    return model


def _override_step_to_elementwise(model):
    def step_el(self, tokens, hidden):
        pre = self.recurrent_linear(hidden) + self.embedded_input(tokens)
        return torch.tanh(pre)
    model.step = types.MethodType(step_el, model)


def _override_step_to_radial(model):
    def step_rad(self, tokens, hidden):
        B, _, C, H, W = hidden.shape
        flat = hidden.reshape(B, C * 2, H, W)
        pre = self.conv(flat) + self.embedded_input(tokens).reshape(B, C * 2, H, W)
        pre_reshaped = pre.reshape(B, C, 2, H, W)
        real = pre_reshaped[:, :, 0]
        imag = pre_reshaped[:, :, 1]
        r = torch.sqrt(real**2 + imag**2 + 1e-8)
        rt = torch.tanh(r) / (r + 1e-8)
        out = torch.zeros_like(pre_reshaped)
        out[:, :, 0] = rt * real
        out[:, :, 1] = rt * imag
        return self.norm(out.reshape(B, C * 2, H, W)).reshape(B, 2, C, H, W)
    model.step = types.MethodType(step_rad, model)
