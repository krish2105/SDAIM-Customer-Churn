#!/usr/bin/env bash
# Run the Streamlit application locally against the exported artifacts.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR/deploy"
PY="${ROOT_DIR}/.venv/bin/python"
[ -x "$PY" ] || PY="python3"

if [ ! -f artifacts/model_pipeline.joblib ]; then
  echo "ERROR: deploy/artifacts/model_pipeline.joblib is missing." >&2
  echo "Run 'make train' first." >&2
  exit 1
fi

echo "Starting Streamlit at http://localhost:8501 (Ctrl+C to stop)"
exec "$PY" -m streamlit run app.py --server.port=8501 "$@"
