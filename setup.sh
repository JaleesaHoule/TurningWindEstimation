#!/usr/bin/env bash
# One-shot environment setup with uv (https://docs.astral.sh/uv/).
# Creates a Python 3.11 virtualenv in .venv and installs this package + all deps
# (numpy/scipy/pandas/matplotlib, casadi/do-mpc, keras/tensorflow, pybounds).
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Install it first:  curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi

uv sync                     # creates .venv (Python 3.11) and installs everything editable
echo
echo "Done. Run an analysis, e.g.:"
echo "  uv run python analysis_scripts/ann_papernet_constant_minev.py"
