#!/usr/bin/env bash
# Run every non-interactive local quality gate and print a pass/fail summary.
#
# Gates that cannot run in the current environment are reported as NOT RUN.
# They are never reported as passed. External deployment gates (GitHub Actions,
# Hugging Face) are outside the scope of this script by design.
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PY="${ROOT_DIR}/.venv/bin/python"
[ -x "$PY" ] || PY="python3"

PASS=0
FAIL=0
SKIP=0
RESULTS=()

started_at="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

run_gate() {
  local name="$1"
  shift
  echo ""
  echo "=================================================================="
  echo "GATE: ${name}"
  echo "COMMAND: $*"
  echo "=================================================================="
  if "$@"; then
    echo "--> PASS: ${name}"
    RESULTS+=("PASS | ${name}")
    PASS=$((PASS + 1))
    return 0
  fi
  echo "--> FAIL: ${name}" >&2
  RESULTS+=("FAIL | ${name}")
  FAIL=$((FAIL + 1))
  return 1
}

skip_gate() {
  local name="$1"
  local reason="$2"
  echo ""
  echo "=================================================================="
  echo "GATE: ${name}"
  echo "--> NOT RUN: ${reason}"
  echo "=================================================================="
  RESULTS+=("NOT RUN | ${name} (${reason})")
  SKIP=$((SKIP + 1))
}

echo "Release verification started at ${started_at} (UTC)"
echo "Python: $("$PY" --version 2>&1)"

# 1. Dataset validation ------------------------------------------------------
run_gate "Dataset validation" "$PY" -m src.data_validation --strict-sha || true

# 2. Python syntax compilation ----------------------------------------------
run_gate "Python syntax compilation" "$PY" -m compileall -q src tests deploy || true

# 3. Artifact reload in a fresh process --------------------------------------
if [ -f deploy/artifacts/model_pipeline.joblib ]; then
  run_gate "Artifact reload in a fresh process" "$PY" - <<'PYCODE' || true
import json
from pathlib import Path

import joblib
import pandas as pd

artifacts = Path("deploy/artifacts")
pipeline = joblib.load(artifacts / "model_pipeline.joblib")
schema = json.loads((artifacts / "feature_schema.json").read_text(encoding="utf-8"))
metadata = json.loads((artifacts / "model_metadata.json").read_text(encoding="utf-8"))

row = {feature["name"]: feature["default"] for feature in schema["features"]}
frame = pd.DataFrame([row])[schema["feature_order"]]
probability = float(pipeline.predict_proba(frame)[0, 1])
assert 0.0 <= probability <= 1.0, probability

for name, value in metadata["metrics"].items():
    assert isinstance(value, (int, float)), f"{name} is not numeric"

print(f"Loaded {metadata['model_name']} v{metadata['model_version']}")
print(f"Scored the schema default record: probability={probability:.6f}")
PYCODE
else
  skip_gate "Artifact reload in a fresh process" "model artifact missing — run make train"
fi

# 4. Test suite --------------------------------------------------------------
run_gate "Test suite (pytest)" "$PY" -m pytest -q || true

# 5. Streamlit smoke test ----------------------------------------------------
streamlit_smoke() {
  local port=8599
  local log
  log="$(mktemp)"

  # Reclaim the port first. Killing the launching subshell does not kill the
  # Streamlit process it spawned, so an interrupted earlier run can leave an
  # orphan holding the port — which would otherwise make this gate answer
  # healthy from the wrong process, or fail with "Port is not available".
  pkill -f "server.port=${port}" >/dev/null 2>&1 || true
  sleep 1

  local start_dir="${ROOT_DIR}/deploy"
  ( cd "$start_dir" && exec "$PY" -m streamlit run app.py \
      --server.headless=true --server.port="$port" --server.address=127.0.0.1 \
      >"$log" 2>&1 ) &
  local pid=$!

  local ok=1
  for _ in $(seq 1 45); do
    if curl -fsS "http://127.0.0.1:${port}/_stcore/health" >/dev/null 2>&1; then
      ok=0
      break
    fi
    # Stop waiting if the server died rather than polling for the full period.
    if ! kill -0 "$pid" 2>/dev/null; then
      break
    fi
    sleep 1
  done

  if [ "$ok" -eq 0 ]; then
    echo "Streamlit answered /_stcore/health with 200 on port ${port}"
  else
    echo "Streamlit did not become healthy. Server output:" >&2
    cat "$log" >&2
  fi

  kill "$pid" >/dev/null 2>&1 || true
  pkill -f "server.port=${port}" >/dev/null 2>&1 || true
  wait "$pid" 2>/dev/null || true
  rm -f "$log"
  return "$ok"
}
run_gate "Streamlit smoke test" streamlit_smoke || true

# 6. Secret scan -------------------------------------------------------------
run_gate "Secret scan" bash scripts/scan_secrets.sh || true

# 7. Required-file verification ---------------------------------------------
verify_required_files() {
  local missing=0
  local required=(
    README.md LICENSE_NOTICE.md .gitignore Makefile
    requirements-dev.txt SOURCE_MANIFEST.json PROJECT_INPUTS.md
    data/raw/Telco-Customer-Churn.csv
    src/config.py src/data_validation.py src/eda.py src/train.py src/evaluate.py src/schemas.py
    src/analysis_base.py src/fairness.py src/calibration.py src/threshold.py
    src/drift.py src/tracking.py
    notebooks/01_eda_and_modeling.ipynb
    reports/model_comparison.csv reports/executive_model_summary.md reports/eda_observations.md
    reports/fairness_report.md reports/calibration_report.md reports/threshold_analysis.md
    reports/drift_report.md reports/tracking_report.md
    deploy/app.py deploy/theme.py deploy/charts.py deploy/explain.py deploy/batch.py
    deploy/rationale.py deploy/Dockerfile
    deploy/requirements.txt deploy/README.md deploy/.dockerignore
    deploy/artifacts/model_pipeline.joblib deploy/artifacts/model_metadata.json
    deploy/artifacts/feature_schema.json deploy/artifacts/model_card.md
    tests/conftest.py tests/test_data_validation.py tests/test_model_artifact.py
    tests/test_prediction.py tests/test_deployment_files.py
    tests/test_analysis.py tests/test_app_features.py
    .github/workflows/ci.yml .github/workflows/deploy.yml
    docs/INPUT_AUDIT.md docs/IMPLEMENTATION_PLAN.md docs/IMPLEMENTATION_LOG.md
    docs/DECISIONS.md docs/ARCHITECTURE.md docs/SECURITY.md docs/TEST_PLAN.md
    docs/TROUBLESHOOTING.md docs/SCREENSHOT_CHECKLIST.md docs/DELIVERABLES_CHECKLIST.md
    docs/REPORT_TEMPLATE.md docs/DEMONSTRATION_SCRIPT.md
  )
  for path in "${required[@]}"; do
    if [ ! -f "$path" ]; then
      echo "MISSING: $path"
      missing=$((missing + 1))
    fi
  done
  local figures
  figures="$(find reports/figures -name '*.png' 2>/dev/null | wc -l | tr -d ' ')"
  echo "Figures present: ${figures}"
  [ "$figures" -ge 18 ] || { echo "Expected at least 18 figures"; missing=$((missing + 1)); }
  if [ "$missing" -ne 0 ]; then
    echo "${missing} required file(s) missing." >&2
    return 1
  fi
  echo "All required files present."
  return 0
}
run_gate "Required-file verification" verify_required_files || true

# 8. Docker build and run ----------------------------------------------------
if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  run_gate "Docker build" docker build -t churn-intelligence:verify deploy/ || true
  docker_run_check() {
    docker rm -f churn-intelligence-verify >/dev/null 2>&1 || true
    docker run -d --name churn-intelligence-verify -p 7861:7860 churn-intelligence:verify >/dev/null
    local ok=1
    for _ in $(seq 1 45); do
      if curl -fsS http://localhost:7861/_stcore/health >/dev/null 2>&1; then
        ok=0
        break
      fi
      sleep 2
    done
    if [ "$ok" -ne 0 ]; then
      echo "Container did not become healthy. Logs:" >&2
      docker logs churn-intelligence-verify >&2 || true
    else
      echo "Container answered /_stcore/health with 200 on port 7861"
    fi
    docker rm -f churn-intelligence-verify >/dev/null 2>&1 || true
    return "$ok"
  }
  run_gate "Docker run and health check" docker_run_check || true
else
  skip_gate "Docker build" "Docker daemon not available"
  skip_gate "Docker run and health check" "Docker daemon not available"
fi

# 9. Git status --------------------------------------------------------------
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  run_gate "Git status and staged-file review" bash -c '
    echo "--- git status --short ---"
    git status --short
    echo "--- staged files ---"
    git diff --cached --name-only
    echo "--- tracked files matching sensitive names (expect none) ---"
    git ls-files | grep -E "(^|/)(\.env|credentials|.*\.pem|id_rsa|id_ed25519)$" && exit 1
    echo "none"
  ' || true
else
  skip_gate "Git status and staged-file review" "not a Git repository"
fi

# Summary --------------------------------------------------------------------
finished_at="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo ""
echo "=================================================================="
echo " QUALITY GATE SUMMARY"
echo " started : ${started_at}"
echo " finished: ${finished_at}"
echo "=================================================================="
for line in "${RESULTS[@]}"; do
  echo "  ${line}"
done
echo "------------------------------------------------------------------"
echo "  PASS: ${PASS}   FAIL: ${FAIL}   NOT RUN: ${SKIP}"
echo ""
echo "  External gates NOT covered here and still pending real evidence:"
echo "    - GitHub Actions CI run"
echo "    - GitHub Actions deployment run"
echo "    - Hugging Face Space build and live application"
echo "    - Visible-change redeployment test"
echo "=================================================================="

[ "$FAIL" -eq 0 ]
