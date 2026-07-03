"""Generate Quality Gate evidence reports for the Decision Artifact pipeline."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

QUALITY_GATE_REPORT_VERSION = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_quality_gate_report(
    *,
    pipeline_run_id: str,
    artifact_dir: Path,
    checks: dict[str, str],
    sample_decision: dict[str, Any] | None = None,
    status: str = "passed",
) -> dict[str, Any]:
    manifest_path = artifact_dir / "manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found at: {manifest_path}")

    manifest = load_json(manifest_path)

    return {
        "quality_gate_report_version": QUALITY_GATE_REPORT_VERSION,
        "report_generated_at": utc_now(),
        "pipeline_run_id": pipeline_run_id,
        "git_sha": get_git_sha(),
        "status": status,
        "artifact": {
            "artifact_dir": str(artifact_dir),
            "artifact_id": artifact_dir.name,
            "model_version": manifest.get("model_version"),
            "created_at": manifest.get("created_at"),
            "run_id": manifest.get("run_id"),
            "manifest_path": str(manifest_path),
            "hashes": manifest.get("hashes", {}),
        },
        "checks": checks,
        "sample_decision": sample_decision,
        "limitations": [
            "Training data is small and learning-oriented.",
            "Model quality is not proven on broad real-world production data.",
            "No production-scale drift detection yet.",
            "No load, latency, or cost testing yet.",
            "No external immutable model registry such as MLflow or S3 yet.",
            "LLM integration only proves the path can run; it does not prove answer quality.",
            "No full security hardening yet.",
        ],
    }


def render_markdown(report: dict[str, Any]) -> str:
    artifact = report["artifact"]
    checks = report["checks"]
    hashes = artifact.get("hashes", {})

    check_rows = "\n".join(
        f"| {name} | {status} |" for name, status in checks.items()
    )

    limitations = "\n".join(
        f"- {item}" for item in report.get("limitations", [])
    )

    sample_decision = report.get("sample_decision")
    sample_decision_json = (
        json.dumps(sample_decision, indent=2, sort_keys=True)
        if sample_decision is not None
        else "null"
    )

    hashes_json = json.dumps(hashes, indent=2, sort_keys=True)

    return f"""# Quality Gate Report

## Run Identity

| Field | Value |
|---|---|
| Report version | {report["quality_gate_report_version"]} |
| Report generated at | {report["report_generated_at"]} |
| Pipeline run id | {report["pipeline_run_id"]} |
| Git SHA | {report["git_sha"]} |
| Status | {report["status"]} |

## Artifact

| Field | Value |
|---|---|
| Artifact id | {artifact["artifact_id"]} |
| Artifact dir | `{artifact["artifact_dir"]}` |
| Model version | {artifact["model_version"]} |
| Artifact created at | {artifact["created_at"]} |
| Training run id | {artifact["run_id"]} |
| Manifest path | `{artifact["manifest_path"]}` |

## Checks

| Check | Status |
|---|---|
{check_rows}

## Manifest Hashes

```json
{hashes_json}
```

## Sample Decision Object

```json
{sample_decision_json}
```

## What This Quality Gate Proves

This Quality Gate proves that the artifact is packaged, identifiable, loadable,
contract-compatible, policy-checked, smoke-tested, and traceable.

It does not prove that the entire AI system is production-complete.

## What Is Still Not Proven

{limitations}
"""


def write_quality_gate_report(
    *,
    report: dict[str, Any],
    output_dir: Path = Path("evidence/quality_gate"),
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "quality_gate_report.json"
    md_path = output_dir / "quality_gate_report.md"

    write_json(json_path, report)
    md_path.write_text(render_markdown(report), encoding="utf-8")

    sample_decision = report.get("sample_decision")
    if sample_decision is not None:
        write_json(output_dir / "sample_decision.json", sample_decision)

    artifact = report.get("artifact", {})
    hashes = artifact.get("hashes", {})
    if hashes:
        write_json(output_dir / "manifest_hashes.json", hashes)
