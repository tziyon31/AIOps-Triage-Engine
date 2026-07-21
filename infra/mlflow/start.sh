#!/usr/bin/env sh
set -eu

: "${MLFLOW_BACKEND_STORE_URI:?MLFLOW_BACKEND_STORE_URI is required}"
: "${MLFLOW_ARTIFACT_ROOT:?MLFLOW_ARTIFACT_ROOT is required}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-5000}"

echo "Starting MLflow Tracking Server"
echo "Host: ${HOST}"
echo "Port: ${PORT}"
echo "Backend store: configured"
echo "Artifact root: ${MLFLOW_ARTIFACT_ROOT}"

exec mlflow server \
  --host "${HOST}" \
  --port "${PORT}" \
  --backend-store-uri "${MLFLOW_BACKEND_STORE_URI}" \
  --default-artifact-root "${MLFLOW_ARTIFACT_ROOT}"
