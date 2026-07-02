#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

.venv/bin/python -m src.log_triage.predict \
  "2026-05-03 09:12:11 ERROR payments db timeout after retries cpu 93 memory 84"

.venv/bin/python -m src.log_triage.predict \
  --file data/test_logs.txt
