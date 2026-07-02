"""Bitwarden-backed secret access with explicit, fast failures."""

from __future__ import annotations

import json
import os
import shutil
import subprocess

from openai import OpenAI

from src.log_triage.config import BITWARDEN_SECRET_NAME

BW_UNLOCK_HINT = (
    'Unlock Bitwarden and export a session first: '
    'export BW_SESSION="$(bw unlock --raw)"'
)


def require_bitwarden_session() -> None:
    if shutil.which("bw") is None:
        raise RuntimeError(
            "Bitwarden CLI (bw) not found in PATH. "
            f"{BW_UNLOCK_HINT}"
        )

    if not os.environ.get("BW_SESSION"):
        raise RuntimeError(
            "BW_SESSION is not set. "
            f"{BW_UNLOCK_HINT}"
        )

    try:
        completed = subprocess.run(
            ["bw", "status"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            "Timed out checking Bitwarden status. "
            f"{BW_UNLOCK_HINT}"
        ) from error

    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        raise RuntimeError(
            "Failed to check Bitwarden status. "
            f"{stderr or 'bw status returned non-zero exit code'}. "
            f"{BW_UNLOCK_HINT}"
        )

    try:
        status_payload = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            "Bitwarden status returned invalid JSON. "
            f"{BW_UNLOCK_HINT}"
        ) from error

    vault_status = status_payload.get("status")
    if vault_status != "unlocked":
        raise RuntimeError(
            f"Bitwarden vault is {vault_status or 'unknown'}. "
            f"{BW_UNLOCK_HINT}"
        )


def get_openai_api_key_from_bitwarden(
    secret_name: str,
    *,
    timeout_seconds: int = 30,
) -> str:
    require_bitwarden_session()

    try:
        completed = subprocess.run(
            ["bw", "get", "password", secret_name],
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            f"Timed out fetching Bitwarden secret '{secret_name}' "
            f"after {timeout_seconds}s. "
            f"{BW_UNLOCK_HINT}"
        ) from error
    except subprocess.CalledProcessError as error:
        stderr = (error.stderr or "").strip()
        raise RuntimeError(
            f"Failed to fetch Bitwarden secret '{secret_name}'. "
            f"{stderr or 'bw get returned non-zero exit code'}. "
            f"{BW_UNLOCK_HINT}"
        ) from error

    api_key = completed.stdout.strip()
    if not api_key:
        raise RuntimeError(f"Empty secret value: {secret_name}")

    return api_key


def create_openai_client() -> OpenAI:
    api_key = get_openai_api_key_from_bitwarden(BITWARDEN_SECRET_NAME)
    return OpenAI(api_key=api_key)
