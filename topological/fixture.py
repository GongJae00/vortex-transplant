"""Deterministic generated compact fields."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np

from .topology import canonical_vortex_field, phase_links


SEPARATIONS = ((5, 0), (0, 5), (4, 3), (-3, 4))
TEMPLATE_SHIFT = (7, 11)


@dataclass(frozen=True)
class FeasibilitySpec:
    height: int = 32
    width: int = 32
    field_count: int = 32
    smooth_link_amplitude: float = 0.10
    magnitude_amplitude: float = 0.20

    def validate(self) -> None:
        if (self.height, self.width, self.field_count) != (32, 32, 32):
            raise ValueError("dimensions are frozen at 32 x 32 x 32")
        if self.smooth_link_amplitude != 0.10 or self.magnitude_amplitude != 0.20:
            raise ValueError("amplitudes are frozen")


@dataclass(frozen=True)
class GeneratedCompactField:
    seed: int
    field_index: int
    group_index: int
    template_index: int
    separation: tuple[int, int]
    charge: np.ndarray
    magnitude: np.ndarray
    smooth: np.ndarray
    field: np.ndarray


def namespace_rng(seed: int, namespace: str) -> np.random.Generator:
    digest = hashlib.sha256(f"PM1-FEASIBILITY-V1|{seed}|{namespace}".encode()).digest()
    words = np.frombuffer(digest[:16], dtype=np.uint32)
    return np.random.default_rng(np.random.SeedSequence(words.tolist()))


def charge_template(template_index: int, spec: FeasibilitySpec = FeasibilitySpec()) -> np.ndarray:
    spec.validate()
    if template_index not in range(8):
        raise ValueError("template index must be in 0..7")
    family = template_index % 4
    translation = (0, 0) if template_index < 4 else TEMPLATE_SHIFT
    positive = (
        (5 + 3 * family + translation[0]) % spec.height,
        (6 + 4 * family + translation[1]) % spec.width,
    )
    separation = SEPARATIONS[family]
    negative = (
        (positive[0] + separation[0]) % spec.height,
        (positive[1] + separation[1]) % spec.width,
    )
    charge = np.zeros((spec.height, spec.width), dtype=np.int64)
    charge[positive] = 1
    charge[negative] = -1
    return charge


def _band_limited_real(rng: np.random.Generator, shape: tuple[int, int]) -> np.ndarray:
    noise = rng.normal(size=shape)
    transform = np.fft.fft2(noise)
    frequency_x = np.abs(np.fft.fftfreq(shape[0]) * shape[0])
    frequency_y = np.abs(np.fft.fftfreq(shape[1]) * shape[1])
    mask = (frequency_x[:, None] <= 2) & (frequency_y[None, :] <= 2)
    mask[0, 0] = False
    result = np.fft.ifft2(transform * mask).real
    result -= float(result.mean())
    return result


def _smooth_component(seed: int, group_index: int, spec: FeasibilitySpec) -> np.ndarray:
    phase = _band_limited_real(namespace_rng(seed, f"smooth/{group_index}"), (32, 32))
    trial = np.exp(1j * phase)
    link_x, link_y = phase_links(trial)
    maximum = float(max(np.max(np.abs(link_x)), np.max(np.abs(link_y))))
    if maximum <= 1e-15:
        raise RuntimeError("smooth fixture has zero link amplitude")
    phase = phase * (spec.smooth_link_amplitude / maximum)
    return np.exp(1j * phase)


def _magnitude(seed: int, group_index: int, spec: FeasibilitySpec) -> np.ndarray:
    pattern = _band_limited_real(namespace_rng(seed, f"magnitude/{group_index}"), (32, 32))
    maximum = float(np.max(np.abs(pattern)))
    if maximum <= 1e-15:
        raise RuntimeError("magnitude fixture is constant")
    magnitude = 1.0 + spec.magnitude_amplitude * pattern / maximum
    if float(magnitude.min()) < 0.75 or float(magnitude.max()) > 1.25:
        raise RuntimeError("magnitude fixture violates frozen bounds")
    return magnitude


def generate_fields(
    seed: int,
    spec: FeasibilitySpec = FeasibilitySpec(),
) -> tuple[GeneratedCompactField, ...]:
    spec.validate()
    groups = spec.field_count // 8
    records: list[GeneratedCompactField] = []
    for group_index in range(groups):
        smooth = _smooth_component(seed, group_index, spec)
        magnitude = _magnitude(seed, group_index, spec)
        for template_index in range(8):
            charge = charge_template(template_index, spec)
            vortex = canonical_vortex_field(charge).field
            field = magnitude * vortex * smooth
            records.append(
                GeneratedCompactField(
                    seed=seed,
                    field_index=len(records),
                    group_index=group_index,
                    template_index=template_index,
                    separation=SEPARATIONS[template_index % 4],
                    charge=charge,
                    magnitude=magnitude.copy(),
                    smooth=smooth.copy(),
                    field=field,
                )
            )
    if len(records) != spec.field_count:
        raise RuntimeError("fixture emitted the wrong field count")
    return tuple(records)
