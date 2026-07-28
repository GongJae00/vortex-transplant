# vortex-transplant

**Topological defect causal intervention for U(1)-equivariant neural networks.**

## Overview

When a U(1)-equivariant convolutional RNN is trained on a delayed copy task, its
hidden state develops quantized vortex defects — point-like topological charges
on a spatial field. This project tests whether those vortices are causally
responsible for the computation, or merely correlated with it.

The experiment:
1. Train a U(1)-equivariant ConvRNN on variable-delay symbol copy
2. Decompose the hidden state into vortex (topological charge), smooth
   (charge-free phase), and magnitude (radial amplitude) components
3. Select donor-recipient pairs with differing vortex charge patterns
4. Transplant only the vortex component from donor to recipient
5. Measure whether the network output follows the donor sequence

If vortex transplant shifts the output, topological charge is a causal
computation unit. If not, the observed vortices are epiphenomenal.

## Causal Intervention Arms

Every donor-recipient pair is tested against 10 comparable null families using
an intersection-union test (IUT): the vortex effect must surpass ALL null
families simultaneously. Additional validity gates verify representative
invariance and manifold validity.

| Family | Description | Role |
|--------|-------------|------|
| `vortex` | Vortex transplant | Primary hypothesis |
| `smooth` | Smooth component swap | Component control |
| `magnitude` | Magnitude component swap | Component control |
| `global_phase` | Displacement-matched global rotation | Perturbation control |
| `zero_charge_phase` | Displacement-matched zero-charge rotation | Perturbation control |
| `fourier_low` | Low-frequency Fourier transplant | Frequency baseline |
| `fourier_high` | High-frequency Fourier transplant | Frequency baseline |
| `pca` | PCA component transplant | Variance baseline |
| `random_direction` | Same-norm random perturbation | Null baseline |
| `harmonic` | Harmonic sector swap | Competing topology |
| `charge_arrangement_shuffle` | Charge-count-matched random arrangement | Spatial baseline |

Additional arms: `natural_recipient`, `whole_state`, sign flip, vortex removal,
minimal topological surgery, same-charge representative sampling.

## Architecture

| Component | U1ConvRNN | PlainConvRNN (baseline) |
|-----------|-----------|-------------------------|
| Convolution | Complex cross-coupled (U(1)-commuting) | Standard real Conv2d |
| Nonlinearity | radial_tanh (U(1)-equivariant) | tanh + LayerNorm |
| Equivariance | Global U(1) phase rotation commutes with dynamics | None |

Both models share identical hidden-state shape `(2, C, H, W)`. A factorial
2×2 baseline design isolates the linear equivariance and nonlinear equivariance
factors independently.

## Install

```bash
pip install -e ".[dev]"
```

Requires Python ≥ 3.12, PyTorch ≥ 2.12, NumPy ≥ 2.0, SciPy ≥ 1.14, scikit-learn ≥ 1.5.

## Run

```bash
# Smoke gate — verifies pipeline integrity
RESEARCH_DATA_ROOT=/tmp python -m topological.learned_smoke

# Full pilot — 10 seeds × 2 models
bash scripts/run_topological_pilot.sh
```

## Test

```bash
pytest tests/ -q
```

## Package Map

```
topological/
├── model.py              # U1ConvRNN, PlainConvRNN, C=1, factorial 2×2
├── task.py               # CopyBatch, run_copy, continue_copy
├── training.py           # train_seed, validate_model, TrainingSpec
├── topology.py           # extract_charge, branch margins, defect tracking
├── decomposition.py      # decompose (vortex/smooth/magnitude), Fourier
├── interventions.py      # 13-arm intervention suite
├── learned_evaluation.py # evaluate_seed_model, select_donor_pair, decision gates
├── learned_smoke.py      # Smoke gate
├── learned_pilot.py      # Canonical pilot
├── baseline.py           # LSTM baseline
├── hodge.py              # Compact Hodge decomposition
├── statistics.py         # IUT, wild bootstrap, SESOI
├── surgery.py            # Minimal topological surgery
├── manifold.py           # Manifold diagnostics (PCA, kNN)
├── representatives.py    # Same-charge representative sampling
├── protocol.py           # Split enforcement, contract loading
├── types.py              # Dataclass specifications
├── evaluation.py         # Feasibility evaluator
├── fixture.py            # Synthetic field generators
├── pilot.py              # Feasibility pilot
├── smoke.py              # Feasibility smoke gate
└── _artifacts.py         # WriteOnceArtifact
```

## Citation

```bibtex
@misc{vortex-transplant,
  author = {GongJae},
  title = {vortex-transplant: Topological Defect Causal Intervention},
  year = {2026},
  url = {https://github.com/GongJae00/vortex-transplant},
}
```

## License

MIT
