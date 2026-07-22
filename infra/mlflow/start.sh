#!/usr/bin/env sh
set -eu

: "${MLFLOW_BACKEND_STORE_URI:?MLFLOW_BACKEND_STORE_URI is required}"
: "${MLFLOW_ARTIFACT_ROOT:?MLFLOW_ARTIFACT_ROOT is required}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-5000}"
MLFLOW_ENABLE_BASIC_AUTH="${MLFLOW_ENABLE_BASIC_AUTH:-false}"
MLFLOW_SERVER_ALLOWED_HOSTS="${MLFLOW_SERVER_ALLOWED_HOSTS:-127.0.0.1,localhost}"
MLFLOW_SERVER_CORS_ALLOWED_ORIGINS="${MLFLOW_SERVER_CORS_ALLOWED_ORIGINS:-http://127.0.0.1:5000}"

AUTH_ARGS=""

echo "Starting MLflow Tracking Server"
echo "Host: ${HOST}"
echo "Port: ${PORT}"
echo "Backend store: configured"
echo "Artifact destination: ${MLFLOW_ARTIFACT_ROOT}"
echo "Artifact mode: server proxy (--serve-artifacts)"

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

# Clients talk only to the tracking server. The server holds S3 credentials and
# proxies artifact uploads/downloads (--serve-artifacts).
#
# --default-artifact-root must be mlflow-artifacts:/ (proxy URI for new experiments).
# Do NOT set it to s3://... — that makes clients upload directly to S3 and need boto3.
# --artifacts-destination is where the server stores files (S3).
#
# Already-created experiments keep their old artifact root. Use a new experiment name
# after enabling proxy mode.
exec mlflow server \
  ${AUTH_ARGS} \
  --host "${HOST}" \
  --port "${PORT}" \
  --backend-store-uri "${MLFLOW_BACKEND_STORE_URI}" \
  --serve-artifacts \
  --artifacts-destination "${MLFLOW_ARTIFACT_ROOT}" \
  --default-artifact-root mlflow-artifacts:/ \
  --allowed-hosts "${MLFLOW_SERVER_ALLOWED_HOSTS}" \
  --cors-allowed-origins "${MLFLOW_SERVER_CORS_ALLOWED_ORIGINS}"
