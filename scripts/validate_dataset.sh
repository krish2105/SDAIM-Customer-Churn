#!/usr/bin/env bash
# Validate the raw IBM Telco dataset against the documented contract.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PY="${ROOT_DIR}/.venv/bin/python"
[ -x "$PY" ] || PY="python3"
exec "$PY" -m src.data_validation "$@"
