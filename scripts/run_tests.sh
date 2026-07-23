#!/usr/bin/env bash
# Run the full pytest suite from the repository root.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PY="${ROOT_DIR}/.venv/bin/python"
[ -x "$PY" ] || PY="python3"
exec "$PY" -m pytest "$@"
