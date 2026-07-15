from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


DEFAULT_PROMOTION_EVIDENCE_CONTRACT_PATH = (
    "config/promotion_evidence_contract.yaml"
)


def load_promotion_evidence_contract(
    path: str | Path = DEFAULT_PROMOTION_EVIDENCE_CONTRACT_PATH,
) -> dict[str, Any]:
    contract_path = Path(path)

    if not contract_path.exists():
        raise FileNotFoundError(
            f"Promotion evidence contract not found: {contract_path}"
        )

    contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))

    if not isinstance(contract, dict):
        raise ValueError(
            f"Promotion evidence contract must be a YAML mapping: {contract_path}"
        )

    required_fields = [
        "contract_name",
        "contract_version",
        "required_params",
        "required_metrics",
        "required_tags",
        "required_artifacts",
        "candidate_statuses",
    ]

    missing_fields = [
        field for field in required_fields
        if field not in contract
    ]

    if missing_fields:
        raise ValueError(
            f"Promotion evidence contract missing required fields: {missing_fields}"
        )

    contract["contract_path"] = str(contract_path)

    return contract


def _missing_keys(
    *,
    required_keys: list[str],
    values: dict[str, Any],
) -> list[str]:
    return [
        key for key in required_keys
        if values.get(key) in {None, ""}
    ]


def validate_promotion_evidence(
    *,
    run: dict[str, Any],
    artifact_paths: list[str],
    contract: dict[str, Any],
) -> dict[str, Any]:
    params = run.get("params", {})
    metrics = run.get("metrics", {})
    tags = run.get("tags", {})

    missing_params = _missing_keys(
        required_keys=contract["required_params"],
        values=params,
    )

    missing_metrics = _missing_keys(
        required_keys=contract["required_metrics"],
        values=metrics,
    )

    missing_tags = _missing_keys(
        required_keys=contract["required_tags"],
        values=tags,
    )

    artifact_path_set = set(artifact_paths)
    missing_artifacts = [
        artifact
        for artifact in contract["required_artifacts"]
        if artifact not in artifact_path_set
    ]

    status = (
        "passed"
        if not missing_params
        and not missing_metrics
        and not missing_tags
        and not missing_artifacts
        else "failed"
    )

    return {
        "status": status,
        "contract_name": contract["contract_name"],
        "contract_version": contract["contract_version"],
        "missing_params": missing_params,
        "missing_metrics": missing_metrics,
        "missing_tags": missing_tags,
        "missing_artifacts": missing_artifacts,
    }


def build_default_promotion_report_text(
    *,
    run_id: str,
    variant_name: str,
    candidate_status: str,
    promotion_reason: str,
    run_owner: str,
    contract: dict[str, Any],
) -> str:
    return "\n".join(
        [
            "# Promotion Report",
            "",
            "## Status",
            "",
            f"- Run ID: `{run_id}`",
            f"- Variant: `{variant_name}`",
            f"- Candidate status: `{candidate_status}`",
            f"- Promotion reason: `{promotion_reason}`",
            f"- Run owner: `{run_owner}`",
            "",
            "## Contract",
            "",
            f"- Contract name: `{contract['contract_name']}`",
            f"- Contract version: `{contract['contract_version']}`",
            f"- Contract path: `{contract['contract_path']}`",
            "",
            "## Boundary",
            "",
            "This report is created during training as the initial promotion evidence placeholder.",
            "Candidate selection still requires `scripts/promote.py` and explicit operator review.",
            "",
        ]
    )
