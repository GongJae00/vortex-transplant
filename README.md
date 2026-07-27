# vortex-transplant

Vortex defect transplant: decompose neural hidden states into topological
components and test whether vortex charge is a causal computation unit.

## Install

```bash
pip install -e ".[dev]"
```

## Experiment

```bash
# Resource smoke (CPU)
RESEARCH_DATA_ROOT=/tmp python -m topological.learned_smoke

# Full pilot (CUDA)
bash scripts/run_topological_pilot.sh
```

## Test

```bash
pytest tests/ -q
```

## Package

```python
from topological.model import U1ConvRNN, PlainConvRNN, ModelSpec
from topological.task import generate_copy_batch, run_copy
from topological.interventions import decompose_hidden, component_intervention
from topological.learned_evaluation import evaluate_seed_model, decide_learned_pilot
```

## Structure

```
.
├── topological/          # source package
├── tests/                # pytest suite
├── configs/              # frozen experiment contracts
├── scripts/              # entry-point shells
└── pyproject.toml
```

## License

MIT
