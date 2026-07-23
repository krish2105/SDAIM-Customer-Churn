#!/usr/bin/env bash
# Build the deployment image and, optionally, run it locally.
#   scripts/build_docker.sh          # build only
#   scripts/build_docker.sh --run    # build, run and health-check on port 7860
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

IMAGE_NAME="churn-intelligence:local"
CONTAINER_NAME="churn-intelligence-local"

if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: docker is not installed or not on PATH." >&2
  exit 1
fi
if ! docker info >/dev/null 2>&1; then
  echo "ERROR: the Docker daemon is not running. Start Docker Desktop and retry." >&2
  exit 1
fi

echo "Building ${IMAGE_NAME} from deploy/"
docker build -t "$IMAGE_NAME" deploy/

if [ "${1:-}" != "--run" ]; then
  echo "Build complete. Run it with: scripts/build_docker.sh --run"
  exit 0
fi

docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
echo "Starting container on http://localhost:7860"
docker run -d --name "$CONTAINER_NAME" -p 7860:7860 "$IMAGE_NAME" >/dev/null

echo "Waiting for the application to become healthy..."
for attempt in $(seq 1 40); do
  if curl -fsS http://localhost:7860/_stcore/health >/dev/null 2>&1; then
    echo "HEALTH CHECK PASSED after ${attempt} attempt(s): http://localhost:7860"
    echo "Stop it with: docker rm -f ${CONTAINER_NAME}"
    exit 0
  fi
  sleep 2
done

echo "HEALTH CHECK FAILED. Container logs:" >&2
docker logs "$CONTAINER_NAME" >&2 || true
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
exit 1
