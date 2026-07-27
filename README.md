# vortex-transplant

**Topological defect causal intervention for U(1)-equivariant neural networks.**

## Overview

When a U(1)-equivariant convolutional RNN is trained on a delayed copy task, its
hidden state develops a quantized structure: vortex defects — point-like
topological charges on a spatial field.  This project tests whether those
vortices are causally responsible for the computation, or merely correlated
with it.

The experiment:
1. **Train** a U(1)-equivariant ConvRNN on variable-delay symbol copy
2. **Decompose** the hidden state into three components:
   - *vortex* — topological charge (integer-valued, quantized)
   - *smooth* — charge-free continuous phase flow
   - *magnitude* — radial amplitude
3. **Select** a donor-recipient pair with differing vortex charge patterns
4. **Transplant** only the vortex component from donor to recipient
5. **Measure** whether the network's output follows the donor sequence

If vortex transplant shifts the output, topological charge is a causal
computation unit. If not, the observed vortices are epiphenomenal.

## Architecture

| Component | U1ConvRNN | PlainConvRNN (baseline) |
|-----------|-----------|-------------------------|
| Convolution | Complex cross-coupled (real + imag → real + imag) | Standard real Conv2d |
| Nonlinearity | radial_tanh (U(1)-equivariant) | tanh + LayerNorm |
| Equivariance | Global U(1) phase rotation commutes with dynamics | None |

Both models share identical hidden-state shape `(2, C, H, W)`, allowing the
same decomposition pipeline to operate on both.  The comparison tests whether
U(1)-equivariant dynamics produce *causally stronger* topological structure
than generic convolutional dynamics.

## Causal Intervention Arms (13 total)

Every donor-recipient pair is tested against 13 intervention conditions:

| Arm | Description | Role |
|-----|-------------|------|
| `natural_recipient` | No intervention | Negative control |
| `natural_donor` | Full donor state | Upper bound |
| `vortex` | **Vortex transplant** | Primary hypothesis |
| `smooth` | Smooth component swap | Nuisance control |
| `magnitude` | Magnitude component swap | Nuisance control |
| `global_phase` | Displacement-matched global rotation | Nuisance control |
| `zero_phase` | Displacement-matched zero-charge rotation | Nuisance control |
| `whole_phase` | Full phase replacement | Upper bound |
| `whole_state` | Full state replacement | Upper bound |
| `fourier_low` | Low-frequency Fourier transplant | Frequency baseline |
| `fourier_high` | High-frequency Fourier transplant | Frequency baseline |
| `pca` | Top-$k$ PCA component transplant | Variance baseline |
| `random_direction` | Same-norm random perturbation | Null baseline |

**Mechanism advantage** = vortex margin − max(nuisance margins).  Positive
advantage means vortex carries causal signal beyond what any generic
intervention of comparable magnitude achieves.

## Decision Gate

A learned pilot survives (`PM1_SURVIVE_LEARNED_PILOT`) when:

1. **U1ConvRNN passes** all single-model gates:
   - Held-out accuracy ≥ 95% on delay 64
   - Defect prevalence ≥ 50%
   - Signed topology persistence ≥ 50%
   - All 13 arms complete
   - Vortex margin > 0, natural_recipient < 0, whole_state > 0
   - Component transplantation valid
   - Nuisance guards ≥ 90% joint pass
   - ≥80% seeds show positive mechanism advantage
   - Bootstrap 2.5% CI > 0
2. **Plus**: either PlainConvRNN shows no vortex phenomenon, or U1ConvRNN
   advantage exceeds PlainConvRNN with paired-bootstrap significance.

## Install

```bash
pip install -e ".[dev]"
```

Requires Python ≥ 3.11, PyTorch ≥ 2.12, NumPy ≥ 2.0, SciPy ≥ 1.14.

## Run

```bash
# Smoke gate — verifies pipeline integrity without CUDA
RESEARCH_DATA_ROOT=/tmp python -m topological.learned_smoke

# Full pilot — 10 seeds × 2 models, ~10h on RTX 5080
bash scripts/run_topological_pilot.sh
```

## Test

```bash
pytest tests/ -q       # 71 tests, ~6s without slow marker
```

## Package Map

```
topological/
├── model.py              # U1ConvRNN, PlainConvRNN, ModelSpec
├── task.py               # CopyBatch, run_copy, continue_copy, reverse_copy
├── training.py           # train_seed, validate_model, TrainingSpec
├── topology.py           # extract_charge, canonical_vortex_field
├── decomposition.py      # decompose (vortex/smooth/magnitude), fourier
├── interventions.py      # component_intervention, PCA, random_direction, fit_pca
├── learned_evaluation.py # evaluate_seed_model, select_donor_pair, decide_learned_pilot
├── learned_smoke.py      # Resource & path smoke gate (CPU)
├── learned_pilot.py      # Canonical 10-seed × 2-model pilot (CUDA)
├── baseline.py           # LSTM baseline
├── evaluation.py         # Feasibility evaluator
├── fixture.py            # Synthetic field generators
├── pilot.py              # Feasibility pilot
├── smoke.py              # Feasibility smoke gate
└── _artifacts.py         # WriteOnceArtifact, verify_manifest
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
