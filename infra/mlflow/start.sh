#!/usr/bin/env sh
set -eu

: "${MLFLOW_BACKEND_STORE_URI:?MLFLOW_BACKEND_STORE_URI is required}"
: "${MLFLOW_ARTIFACT_ROOT:?MLFLOW_ARTIFACT_ROOT is required}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-5000}"
MLFLOW_ENABLE_BASIC_AUTH="${MLFLOW_ENABLE_BASIC_AUTH:-false}"

AUTH_ARGS=""

echo "Starting MLflow Tracking Server"
echo "Host: ${HOST}"
echo "Port: ${PORT}"
echo "Backend store: configured"
echo "Artifact root: ${MLFLOW_ARTIFACT_ROOT}"

if [ "${MLFLOW_ENABLE_BASIC_AUTH}" = "true" ]; then
  : "${MLFLOW_FLASK_SERVER_SECRET_KEY:?MLFLOW_FLASK_SERVER_SECRET_KEY is required when auth is enabled}"
  : "${MLFLOW_AUTH_ADMIN_USERNAME:?MLFLOW_AUTH_ADMIN_USERNAME is required when auth is enabled}"
  : "${MLFLOW_AUTH_ADMIN_PASSWORD:?MLFLOW_AUTH_ADMIN_PASSWORD is required when auth is enabled}"

  AUTH_DATABASE_URI="${MLFLOW_AUTH_DATABASE_URI:-${MLFLOW_BACKEND_STORE_URI}}"

  cat > /tmp/mlflow_auth.ini <<EOF
[mlflow]
database_uri = ${AUTH_DATABASE_URI}
admin_username = ${MLFLOW_AUTH_ADMIN_USERNAME}
admin_password = ${MLFLOW_AUTH_ADMIN_PASSWORD}
authorization_function = mlflow.server.auth:authenticate_request_basic_auth
default_permission = READ
EOF

  export MLFLOW_AUTH_CONFIG_PATH=/tmp/mlflow_auth.ini
  AUTH_ARGS="--app-name basic-auth"

  echo "Basic auth: enabled"
  echo "Auth database: configured"
else
  echo "Basic auth: disabled"
fi

exec mlflow server \
  ${AUTH_ARGS} \
  --host "${HOST}" \
  --port "${PORT}" \
  --backend-store-uri "${MLFLOW_BACKEND_STORE_URI}" \
  --default-artifact-root "${MLFLOW_ARTIFACT_ROOT}"
