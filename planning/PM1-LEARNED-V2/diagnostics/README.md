# Diagnostic README

## Contents

| File | Description |
|------|-------------|
| `metric_definitions.py` | Canonical metric definitions (site count vs charge units) |
| `run_untrained_topology.py` | Exact V1 gate replication (128 examples, 10 seeds, 2 models) |
| `run_random_phase_null.py` | Random phase field topology baseline |
| `run_embedding_topology.py` | Token embedding topology analysis |
| `raw/` | Raw JSON output files |
| `sha256.json` | SHA-256 hashes of raw outputs |
| `environment.json` | Environment (PyTorch version, numpy, etc.) |
| `commands.txt` | Exact commands used |

## Invariants

- `positive_site_count + negative_site_count == signed_site_count`
- `net_charge == positive_charge_units - negative_charge_units`
- `0 <= signed_site_count <= C * H * W (2048)`
- `0 <= site_density <= 1.0`
- `pairs per channel` cannot exceed 256 (only 16×16 plaquettes per channel)
