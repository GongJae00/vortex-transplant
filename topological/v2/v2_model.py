"""V2 model extensions: C=1 gateway, factorial 2×2 baseline variants."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import types
from ..model import ModelSpec, U1ConvRNN, PlainConvRNN


def make_scalar_u1_model(
    model_spec: ModelSpec | None = None,
    generator: torch.Generator | None = None,
    device: torch.device | None = None,
) -> U1ConvRNN:
    """C=1 U1ConvRNN — scalar complex order parameter."""
    if model_spec is None:
        model_spec = ModelSpec(channels=1)
    else:
        from dataclasses import replace
        model_spec = replace(model_spec, channels=1)
    model = U1ConvRNN(model_spec, generator=generator)
    if device is not None:
        model = model.to(device)
    return model


def make_factorial_model(
    variant: str,
    model_spec: ModelSpec | None = None,
    generator: torch.Generator | None = None,
    device: torch.device | None = None,
):
    """Factory for factorial 2×2 baseline variants.

    Variants:
    - U1CommutingLinear_RadialNonlinear: full U1ConvRNN
    - U1CommutingLinear_ElementwiseNonlinear: U1ConvRNN.step → elementwise tanh
    - UnrestrictedLinear_RadialNonlinear: PlainConvRNN.step → radial tanh
    - UnrestrictedLinear_ElementwiseNonlinear: full PlainConvRNN
    """
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
    """Override U1ConvRNN.step to use elementwise tanh (breaks Factor B)."""
    def step_el(self, tokens, hidden):
        pre = self.recurrent_linear(hidden) + self.embedded_input(tokens)
        return torch.tanh(pre)
    model.step = types.MethodType(step_el, model)


def _override_step_to_radial(model):
    """Override PlainConvRNN.step to use radial_tanh (breaks Factor A)."""
    def step_rad(self, tokens, hidden):
        B, _, C, H, W = hidden.shape  # (B, 2, C, H, W)
        flat = hidden.reshape(B, C * 2, H, W)
        pre = self.conv(flat) + self.embedded_input(tokens).reshape(B, C * 2, H, W)
        out = torch.zeros_like(pre)
        for ch in range(0, pre.shape[1], 2):
            real = pre[:, ch]
            imag = pre[:, ch + 1]
            r = torch.sqrt(real**2 + imag**2 + 1e-8)
            rt = torch.tanh(r) / (r + 1e-8)
            out[:, ch] = rt * real
            out[:, ch + 1] = rt * imag
        return self.norm(out).reshape(B, 2, C, H, W)  # back to (B, 2, C, H, W)
    model.step = types.MethodType(step_rad, model)
