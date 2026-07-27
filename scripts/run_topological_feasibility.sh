#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
from topological.smoke import canonical_smoke_root, run_smoke

print(run_smoke(canonical_smoke_root()))
PY

python -m topological.pilot
