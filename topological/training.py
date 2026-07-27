"""Deterministic training and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Callable

import numpy as np

import torch
from torch.nn import functional as F

from .model import ModelSpec, PlainConvRNN, U1ConvRNN
from .task import CopyBatch, generate_copy_batch, generate_reverse_copy_batch, namespace_seed, run_copy


@dataclass(frozen=True)
class TrainingSpec:
    updates: int = 30_000
    batch_size: int = 64
    copy_length: int = 4
    copy_length_min: int = 3
    copy_length_max: int = 7
    train_delay_min: int = 16
    train_delay_max: int = 32
    validation_examples: int = 512
    validation_interval: int = 2_000
    learning_rate: float = 1e-3
    gradient_clip: float = 1.0

    def validate(self) -> None:
        if self.updates <= 0 or self.batch_size <= 0 or self.validation_examples <= 0:
            raise ValueError("training counts must be positive")
        if self.copy_length <= 0:
            raise ValueError("copy length must be positive")
        if self.copy_length_min <= 0 or self.copy_length_max < self.copy_length_min:
            raise ValueError("copy length range is invalid")
        if self.train_delay_min < 0 or self.train_delay_max < self.train_delay_min:
            raise ValueError("train delay range is invalid")
        if self.validation_interval <= 0 or self.learning_rate <= 0 or self.gradient_clip <= 0:
            raise ValueError("optimizer settings must be positive")


@dataclass(frozen=True)
class ValidationRecord:
    update: int
    accuracy: float
    cross_entropy: float
    example_count: int
    delay_accuracies: tuple[tuple[int, float], ...]
    amplitude_saturation: float
    content_sha256: tuple[str, ...]


@dataclass(frozen=True)
class TrainingSnapshot:
    update: int
    loss: float
    gradient_norm: float
    learning_rate: float


@dataclass
class TrainingResult:
    model: nn.Module
    selected_update: int
    selected_accuracy: float
    selected_cross_entropy: float
    history: list[ValidationRecord]
    train_hashes: list[str]
    update_count: int
    finite_gradient_steps: int
    training_log: list[TrainingSnapshot] = field(default_factory=list)


def configure_determinism(seed: int) -> None:
    torch.manual_seed(namespace_seed(seed, "torch"))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(namespace_seed(seed, "cuda"))
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def make_model(
    seed: int,
    *,
    model_type: str = "u1",
    model_spec: ModelSpec = ModelSpec(),
    device: torch.device | str = "cpu",
) -> nn.Module:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(namespace_seed(seed, "model-init"))
    if model_type == "u1":
        return U1ConvRNN(model_spec, generator=generator).to(device)
    if model_type == "plain":
        return PlainConvRNN(model_spec, generator=generator).to(device)
    raise ValueError(f"Unknown model type: {model_type}")


def training_delay(seed: int, update: int, spec: TrainingSpec) -> int:
    width = spec.train_delay_max - spec.train_delay_min + 1
    return spec.train_delay_min + namespace_seed(seed, f"delay/{update}") % width


def training_batch(
    seed: int,
    update: int,
    spec: TrainingSpec,
    *,
    task_type: str = "copy",
    copy_length: int | None = None,
    device: torch.device | str,
) -> CopyBatch:
    if task_type == "reverse":
        gen = generate_reverse_copy_batch
    else:
        gen = generate_copy_batch
    return gen(
        seed,
        f"train/{update}",
        spec.batch_size,
        training_delay(seed, update, spec),
        copy_length=copy_length or spec.copy_length,
        device=device,
    )


def one_training_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: CopyBatch,
    gradient_clip: float,
    *,
    task_type: str = "copy",
) -> tuple[float, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    trace = run_copy(model, batch.symbols, batch.delay, copy_length=batch.copy_length)
    targets = torch.flip(batch.symbols, dims=[1]) if task_type == "reverse" else batch.symbols
    loss = F.cross_entropy(trace.logits.flatten(0, 1), targets.flatten())
    if not torch.isfinite(loss):
        raise RuntimeError("training loss is non-finite")
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
    if not torch.isfinite(gradient_norm):
        raise RuntimeError("gradient norm is non-finite")
    optimizer.step()
    model.enforce_blank_embedding()
    return float(loss.detach().cpu()), float(gradient_norm.detach().cpu())


def _validation_counts(spec: TrainingSpec) -> list[tuple[int, int]]:
    delays = list(range(spec.train_delay_min, spec.train_delay_max + 1))
    base, remainder = divmod(spec.validation_examples, len(delays))
    return [(delay, base + int(index < remainder)) for index, delay in enumerate(delays)]


AMPLITUDE_SATURATION_THRESHOLD = 0.95


def _amplitude_saturation(hidden: torch.Tensor) -> float:
    radius = torch.sqrt(
        hidden[:, 0].square() + hidden[:, 1].square()
    )
    return float((radius > AMPLITUDE_SATURATION_THRESHOLD).float().mean().cpu())


@torch.no_grad()
def validate_model(
    model: nn.Module,
    seed: int,
    update: int,
    spec: TrainingSpec,
    *,
    task_type: str = "copy",
) -> ValidationRecord:
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    hashes: list[str] = []
    delay_accuracies: list[tuple[int, float]] = []
    saturations: list[float] = []
    device = next(model.parameters()).device
    for delay, count in _validation_counts(spec):
        batch = generate_copy_batch(
            seed,
            f"validation/delay-{delay}",
            count,
            delay,
            copy_length=spec.copy_length,
            device=device,
        )
        trace = run_copy(model, batch.symbols, delay, copy_length=batch.copy_length)
        targets = torch.flip(batch.symbols, dims=[1]) if task_type == "reverse" else batch.symbols
        loss = F.cross_entropy(
            trace.logits.flatten(0, 1),
            targets.flatten(),
            reduction="sum",
        )
        delay_correct = int((trace.logits.argmax(dim=-1) == targets).sum().cpu())
        delay_total = int(targets.numel())
        correct += delay_correct
        total += delay_total
        loss_sum += float(loss.cpu())
        delay_accuracies.append((delay, delay_correct / delay_total))
        saturations.append(_amplitude_saturation(trace.post_write))
        hashes.append(batch.content_sha256)
    if total != spec.validation_examples * spec.copy_length:
        raise RuntimeError("validation emitted the wrong token count")
    return ValidationRecord(
        update=update,
        accuracy=correct / total,
        cross_entropy=loss_sum / total,
        example_count=spec.validation_examples,
        delay_accuracies=tuple(delay_accuracies),
        amplitude_saturation=float(np.mean(saturations)),
        content_sha256=tuple(hashes),
    )


def _is_better(candidate: ValidationRecord, incumbent: ValidationRecord | None) -> bool:
    if incumbent is None:
        return True
    return (-candidate.accuracy, candidate.cross_entropy, candidate.update) < (
        -incumbent.accuracy,
        incumbent.cross_entropy,
        incumbent.update,
    )


def train_seed(
    seed: int,
    *,
    model_type: str = "u1",
    task_type: str = "copy",
    variable_length: bool = False,
    model_spec: ModelSpec = ModelSpec(),
    training_spec: TrainingSpec = TrainingSpec(),
    device: torch.device | str = "cpu",
    progress: Callable[[str], None] | None = None,
) -> TrainingResult:
    training_spec.validate()
    configure_determinism(seed)
    model = make_model(seed, model_type=model_type, model_spec=model_spec, device=device)
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
    training_log: list[TrainingSnapshot] = []
    _accum_loss = 0.0
    _accum_grad = 0.0
    _accum_count = 0
    for update in range(1, training_spec.updates + 1):
        copy_length = training_spec.copy_length
        if variable_length:
            copy_length = training_spec.copy_length_min + namespace_seed(
                seed, f"copy_length/{update}"
            ) % (training_spec.copy_length_max - training_spec.copy_length_min + 1)
        batch = training_batch(
            seed, update, training_spec,
            task_type=task_type, copy_length=copy_length, device=device,
        )
        loss, gradient_norm = one_training_step(
            model, optimizer, batch, training_spec.gradient_clip, task_type=task_type,
        )
        if not math.isfinite(loss) or not math.isfinite(gradient_norm):
            raise RuntimeError("training scalar is non-finite")
        finite_gradient_steps += 1
        train_hashes.append(batch.content_sha256)
        _accum_loss += loss
        _accum_grad += gradient_norm
        _accum_count += 1
        if update % training_spec.validation_interval == 0 or update == training_spec.updates:
            record = validate_model(model, seed, update, training_spec, task_type=task_type)
            history.append(record)
            lr = float(optimizer.param_groups[0]["lr"])
            training_log.append(
                TrainingSnapshot(
                    update=update,
                    loss=_accum_loss / _accum_count,
                    gradient_norm=_accum_grad / _accum_count,
                    learning_rate=lr,
                )
            )
            _accum_loss = 0.0
            _accum_grad = 0.0
            _accum_count = 0
            if _is_better(record, best):
                best = record
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
            if progress is not None:
                progress(
                    f"seed {seed} update {update}/{training_spec.updates} "
                    f"{task_type} {copy_length} validation complete"
                )
    if best is None or best_state is None:
        raise RuntimeError("training did not produce a validation checkpoint")
    model.load_state_dict(best_state)
    model.enforce_blank_embedding()
    return TrainingResult(
        model=model,
        selected_update=best.update,
        selected_accuracy=best.accuracy,
        selected_cross_entropy=best.cross_entropy,
        history=history,
        train_hashes=train_hashes,
        update_count=training_spec.updates,
        finite_gradient_steps=finite_gradient_steps,
        training_log=training_log,
    )
