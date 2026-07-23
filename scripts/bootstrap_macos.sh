#!/usr/bin/env bash
# Prepare a local macOS development environment for this project.
# Installs nothing globally and never touches Homebrew or the system Python.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${ROOT_DIR}/.venv"
REQUIRED_MAJOR_MINOR="3.11"

echo "=================================================================="
echo " Customer Churn Intelligence — local bootstrap"
echo " Project root: ${ROOT_DIR}"
echo "=================================================================="

# ---------------------------------------------------------------------------
# 1. Locate a Python 3.11 interpreter
# ---------------------------------------------------------------------------
PYTHON_BIN=""
for candidate in python3.11 python3 python; do
  if command -v "$candidate" >/dev/null 2>&1; then
    version="$("$candidate" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    if [ "$version" = "$REQUIRED_MAJOR_MINOR" ]; then
      PYTHON_BIN="$(command -v "$candidate")"
      break
    fi
  fi
done

if [ -z "$PYTHON_BIN" ]; then
  echo "ERROR: Python ${REQUIRED_MAJOR_MINOR} was not found on PATH." >&2
  echo "Interpreters detected:" >&2
  for candidate in python3.11 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
      echo "  - $(command -v "$candidate"): $("$candidate" --version 2>&1)" >&2
    fi
  done
  echo "" >&2
  echo "This project targets Python ${REQUIRED_MAJOR_MINOR}, which matches the runtime" >&2
  echo "in deploy/Dockerfile. Install it yourself, for example:" >&2
  echo "  brew install python@3.11" >&2
  echo "This script will not install Homebrew or modify your global Python." >&2
  exit 1
fi

echo "[1/7] Using Python: ${PYTHON_BIN} ($("$PYTHON_BIN" --version 2>&1))"

# ---------------------------------------------------------------------------
# 2. Create the virtual environment
# ---------------------------------------------------------------------------
if [ -d "$VENV_DIR" ]; then
  echo "[2/7] Reusing existing virtual environment at .venv"
else
  echo "[2/7] Creating virtual environment at .venv"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# ---------------------------------------------------------------------------
# 3. Activate it for the remainder of this script
# ---------------------------------------------------------------------------
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
echo "[3/7] Activated: $(python --version 2>&1) at $(command -v python)"

# ---------------------------------------------------------------------------
# 4. Upgrade pip
# ---------------------------------------------------------------------------
echo "[4/7] Upgrading pip"
python -m pip install --upgrade pip --quiet

# ---------------------------------------------------------------------------
# 5. Install development dependencies
# ---------------------------------------------------------------------------
echo "[5/7] Installing requirements-dev.txt"
python -m pip install -r requirements-dev.txt --quiet

# ---------------------------------------------------------------------------
# 6. Create generated-output directories (data/raw is never touched)
# ---------------------------------------------------------------------------
echo "[6/7] Creating output directories"
mkdir -p data/processed reports/figures reports/tables deploy/artifacts
touch data/processed/.gitkeep

# ---------------------------------------------------------------------------
# 7. Validate the raw dataset
# ---------------------------------------------------------------------------
echo "[7/7] Validating the raw dataset"
if ! python -m src.data_validation; then
  echo "" >&2
  echo "ERROR: dataset validation failed. Do not repair the raw file by hand." >&2
  echo "Re-download the official file from the URL in SOURCE_MANIFEST.json." >&2
  exit 1
fi

cat <<'NEXT'

==================================================================
 Bootstrap complete.
==================================================================

Activate the environment in your own shell:

    source .venv/bin/activate

Then run, from the repository root:

    make validate      # revalidate the raw dataset
    make eda           # figures, tables and observations
    make train         # train, compare, select and export the pipeline
    make test          # full pytest suite
    make app           # run the Streamlit application locally
    make docker-build  # build the deployment image (Docker must be running)
    make secret-scan   # scan the project for credential patterns
    make verify        # every non-interactive local quality gate

NEXT
