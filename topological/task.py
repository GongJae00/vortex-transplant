"""Hash-separated variable-delay copy task."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import torch
from torch import nn


COPY_LENGTH = 4
SYMBOL_COUNT = 8
BLANK_TOKEN = 0
GO_TOKEN = 9


def namespace_seed(seed: int, namespace: str) -> int:
    digest = hashlib.sha256(f"PM1-LEARNED-V1|{seed}|{namespace}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


@dataclass(frozen=True)
class CopyBatch:
    symbols: torch.Tensor
    delay: int
    copy_length: int
    content_sha256: str


@dataclass(frozen=True)
class CopyTrace:
    logits: torch.Tensor
    post_write: torch.Tensor
    pre_go: torch.Tensor
    final_state: torch.Tensor


def generate_reverse_copy_batch(
    seed: int,
    namespace: str,
    batch_size: int,
    delay: int,
    *,
    copy_length: int = COPY_LENGTH,
    symbol_count: int = SYMBOL_COUNT,
    device: torch.device | str = "cpu",
) -> CopyBatch:
    if batch_size <= 0 or delay < 0:
        raise ValueError("batch size must be positive and delay nonnegative")
    if symbol_count != 8:
        raise ValueError("vocabulary and kernel size are frozen symbol count to 8")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(namespace_seed(seed, f"reverse/{namespace}"))
    symbols = torch.randint(
        1,
        symbol_count + 1,
        (batch_size, copy_length),
        generator=generator,
        dtype=torch.long,
    )
    payload = symbols.numpy().tobytes() + int(delay).to_bytes(8, "little", signed=False)
    payload += b"reverse"
    content_sha256 = hashlib.sha256(payload).hexdigest()
    return CopyBatch(symbols=symbols.to(device), delay=int(delay), copy_length=int(copy_length), content_sha256=content_sha256)


def generate_copy_batch(
    seed: int,
    namespace: str,
    batch_size: int,
    delay: int,
    *,
    copy_length: int = COPY_LENGTH,
    symbol_count: int = SYMBOL_COUNT,
    device: torch.device | str = "cpu",
) -> CopyBatch:
    if batch_size <= 0 or delay < 0:
        raise ValueError("batch size must be positive and delay nonnegative")
    if symbol_count != 8:
        raise ValueError("vocabulary and kernel size are frozen symbol count to 8")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(namespace_seed(seed, namespace))
    symbols = torch.randint(
        1,
        symbol_count + 1,
        (batch_size, copy_length),
        generator=generator,
        dtype=torch.long,
    )
    payload = symbols.numpy().tobytes() + int(delay).to_bytes(8, "little", signed=False)
    content_sha256 = hashlib.sha256(payload).hexdigest()
    return CopyBatch(symbols=symbols.to(device), delay=int(delay), copy_length=int(copy_length), content_sha256=content_sha256)


def donor_sequences(
    symbols: torch.Tensor,
    *,
    copy_length: int = COPY_LENGTH,
    symbol_count: int = SYMBOL_COUNT,
) -> torch.Tensor:
    """Return the eight frozen, unique, position-wise mismatched donors."""

    if symbols.ndim != 2 or symbols.shape[1] != copy_length:
        raise ValueError(f"symbols must have shape (batch, {copy_length})")
    if torch.any((symbols < 1) | (symbols > symbol_count)):
        raise ValueError(f"donor construction requires symbols in 1..{symbol_count}")
    constant = torch.arange(1, symbol_count, device=symbols.device).view(1, -1, 1)
    shifted = 1 + torch.remainder(
        symbols[:, None, :] - 1 + constant,
        symbol_count,
    )
    position_shift = torch.arange(1, copy_length + 1, device=symbols.device).view(1, 1, -1)
    position_donor = 1 + torch.remainder(
        symbols[:, None, :] - 1 + position_shift,
        symbol_count,
    )
    donors = torch.cat((shifted, position_donor), dim=1)
    if donors.shape[1] != symbol_count:
        raise RuntimeError(f"donor catalog must contain exactly {symbol_count} candidates")
    return donors


def write_copy(
    model: nn.Module,
    symbols: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
    copy_length: int = COPY_LENGTH,
) -> torch.Tensor:
    if symbols.ndim != 2 or symbols.shape[1] != copy_length:
        raise ValueError(f"symbols must have shape (batch, {copy_length})")
    hidden = (
        initial_state
        if initial_state is not None
        else model.initial_state(len(symbols), device=symbols.device)
    )
    for position in range(copy_length):
        hidden = model.step(symbols[:, position], hidden)
    return hidden


def run_copy(
    model: nn.Module,
    symbols: torch.Tensor,
    delay: int,
    *,
    initial_state: torch.Tensor | None = None,
    copy_length: int = COPY_LENGTH,
) -> CopyTrace:
    if symbols.ndim != 2 or symbols.shape[1] != copy_length:
        raise ValueError(f"symbols must have shape (batch, {copy_length})")
    if delay < 0:
        raise ValueError("delay must be nonnegative")
    batch_size = len(symbols)
    hidden = write_copy(model, symbols, initial_state=initial_state, copy_length=copy_length)
    post_write = hidden
    blank = torch.full((batch_size,), BLANK_TOKEN, dtype=torch.long, device=symbols.device)
    for _ in range(delay):
        hidden = model.step(blank, hidden)
    pre_go = hidden
    go = torch.full((batch_size,), GO_TOKEN, dtype=torch.long, device=symbols.device)
    hidden = model.step(go, hidden)
    outputs = [model.logits(hidden)]
    for _ in range(copy_length - 1):
        hidden = model.step(blank, hidden)
        outputs.append(model.logits(hidden))
    return CopyTrace(
        logits=torch.stack(outputs, dim=1),
        post_write=post_write,
        pre_go=pre_go,
        final_state=hidden,
    )


def continue_copy(
    model: nn.Module,
    post_write: torch.Tensor,
    delay: int,
    *,
    copy_length: int = COPY_LENGTH,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = len(post_write)
    device = post_write.device
    blank = torch.full((batch_size,), BLANK_TOKEN, dtype=torch.long, device=device)
    hidden = post_write
    for _ in range(delay):
        hidden = model.step(blank, hidden)
    pre_go = hidden
    go = torch.full((batch_size,), GO_TOKEN, dtype=torch.long, device=device)
    hidden = model.step(go, hidden)
    outputs = [model.logits(hidden)]
    for _ in range(copy_length - 1):
        hidden = model.step(blank, hidden)
        outputs.append(model.logits(hidden))
    return torch.stack(outputs, dim=1), pre_go
