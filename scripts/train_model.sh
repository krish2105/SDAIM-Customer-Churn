#!/usr/bin/env bash
# Train, compare, select and export the churn pipeline and its artifacts.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PY="${ROOT_DIR}/.venv/bin/python"
[ -x "$PY" ] || PY="python3"
exec "$PY" -m src.train "$@"
