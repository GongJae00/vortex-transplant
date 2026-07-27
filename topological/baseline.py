"""LSTM baseline model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn

from .task import run_copy
from .training import (
    TrainingSpec,
    configure_determinism,
    ValidationRecord,
)


@dataclass(frozen=True)
class LSTMConfig:
    hidden_size: int = 256
    vocabulary: int = 10

    @property
    def real_state_size(self) -> int:
        return self.hidden_size


class LSTMBaseline(nn.Module):
    def __init__(self, config: LSTMConfig = LSTMConfig()) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocabulary, config.hidden_size, padding_idx=0)
        self.lstm_cell = nn.LSTMCell(config.hidden_size, config.hidden_size)
        self.readout = nn.Linear(config.hidden_size, config.vocabulary)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        with torch.no_grad():
            self.embedding.weight.normal_(mean=0.0, std=0.001)
            self.embedding.weight[0].zero_()
        for name, parameter in self.lstm_cell.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(parameter)
            elif "bias" in name:
                nn.init.zeros_(parameter)
        self.readout.reset_parameters()

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        parameter = next(self.parameters())
        device = device or parameter.device
        dtype = dtype or parameter.dtype
        h = torch.zeros(batch_size, self.config.hidden_size, device=device, dtype=dtype)
        c = torch.zeros(batch_size, self.config.hidden_size, device=device, dtype=dtype)
        return (h, c)

    def step(
        self,
        tokens: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h, c = hidden
        embedded = self.embedding(tokens)
        h, c = self.lstm_cell(embedded, (h, c))
        return (h, c)

    def logits(self, hidden: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        h, _ = hidden
        return self.readout(h)

    def enforce_blank_embedding(self) -> None:
        with torch.no_grad():
            self.embedding.weight[0].zero_()

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def make_lstm(
    seed: int,
    config: LSTMConfig = LSTMConfig(),
    *,
    device: torch.device | str = "cpu",
) -> LSTMBaseline:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    model = LSTMBaseline(config)
    return model.to(device)


def one_lstm_training_step(
    model: LSTMBaseline,
    optimizer: torch.optim.Optimizer,
    batch,
    gradient_clip: float,
) -> tuple[float, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    trace = run_copy(model, batch.symbols, batch.delay, copy_length=batch.copy_length)
    loss = torch.nn.functional.cross_entropy(
        trace.logits.flatten(0, 1), batch.symbols.flatten()
    )
    if not torch.isfinite(loss):
        raise RuntimeError("LSTM baseline training loss is non-finite")
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
    if not torch.isfinite(gradient_norm):
        raise RuntimeError("LSTM baseline gradient norm is non-finite")
    optimizer.step()
    model.enforce_blank_embedding()
    return float(loss.detach().cpu()), float(gradient_norm.detach().cpu())


@dataclass
class LSTMResult:
    model: LSTMBaseline
    selected_update: int
    selected_accuracy: float
    selected_cross_entropy: float
    history: list[ValidationRecord]
    train_hashes: list[str]
    update_count: int
    finite_gradient_steps: int


def train_lstm_seed(
    seed: int,
    config: LSTMConfig = LSTMConfig(),
    training_spec: TrainingSpec = TrainingSpec(),
    *,
    device: torch.device | str = "cpu",
    progress: Callable[[str], None] | None = None,
) -> LSTMResult:
    from .training import training_batch, validate_model

    training_spec.validate()
    configure_determinism(seed)
    model = make_lstm(seed, config, device=device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_spec.learning_rate,
        betas=(0.9, 0.999),
    )
    best: ValidationRecord | None = None
    best_state: dict[str, torch.Tensor] | None = None
    history: list[ValidationRecord] = []
    train_hashes: list[str] = []
    finite_gradient_steps = 0
    for update in range(1, training_spec.updates + 1):
        batch = training_batch(seed, update, training_spec, device=device)
        loss, gradient_norm = one_lstm_training_step(model, optimizer, batch, training_spec.gradient_clip)
        if not (float(loss) > 0 and float(gradient_norm) >= 0):
            raise RuntimeError("LSTM baseline scalar is invalid")
        finite_gradient_steps += 1
        train_hashes.append(batch.content_sha256)
        if update % training_spec.validation_interval == 0 or update == training_spec.updates:
            record = validate_model(model, seed, update, training_spec)
            history.append(record)
            if (best is None
                or (-record.accuracy, record.cross_entropy, record.update)
                < (-best.accuracy, best.cross_entropy, best.update)):
                best = record
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
            if progress is not None:
                progress(f"LSTM seed {seed} update {update}/{training_spec.updates}")
    if best is None or best_state is None:
        raise RuntimeError("LSTM training did not produce a validation checkpoint")
    model.load_state_dict(best_state)
    model.enforce_blank_embedding()
    return LSTMResult(
        model=model,
        selected_update=best.update,
        selected_accuracy=best.accuracy,
        selected_cross_entropy=best.cross_entropy,
        history=history,
        train_hashes=train_hashes,
        update_count=training_spec.updates,
        finite_gradient_steps=finite_gradient_steps,
    )



