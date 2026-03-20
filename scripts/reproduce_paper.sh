#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ ! -d ".venv" ]]; then
  echo "[reproduce] creating .venv"
  python -m venv .venv
fi

echo "[reproduce] activating .venv"
# shellcheck disable=SC1091
source .venv/bin/activate

echo "[reproduce] installing python deps"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "[reproduce] running SPMA paper suite (reuse existing runs)"
python scripts/run_spma_multiseed.py --fast-gpu --reuse-existing --num-workers 4

echo "[reproduce] generating paper figures/tables from outputs"
python paper/build_figures.py

echo "[reproduce] done"

