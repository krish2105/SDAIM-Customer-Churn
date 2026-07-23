#!/usr/bin/env bash
# Regenerate every EDA figure, table and the observations document.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PY="${ROOT_DIR}/.venv/bin/python"
[ -x "$PY" ] || PY="python3"
exec "$PY" -m src.eda "$@"
