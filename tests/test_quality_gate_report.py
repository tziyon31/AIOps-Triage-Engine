import json
from pathlib import Path

from src.log_triage.quality_gate_report import (
    build_quality_gate_report,
    write_quality_gate_report,
)


def test_build_quality_gate_report_contains_required_identity_fields(tmp_path: Path):
    artifact_dir = tmp_path / "log-triage-v1.0.0-20260703-070000"
    artifact_dir.mkdir()

    manifest = {
        "model_version": "v1.0.0",
        "created_at": "2026-07-03T07:00:00+00:00",
        "run_id": "training-run-123",
        "hashes": {
            "model_sha256": "a" * 64,
            "vectorizer_sha256": "b" * 64,
            "known_actions_sha256": "c" * 64,
            "config_sha256": "d" * 64,
            "training_data_sha256": "e" * 64,
        },
    }

    (artifact_dir / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    sample_decision = {
        "strategy_used": "manual_features_plus_tfidf",
        "predicted_action": "open_ticket",
        "confidence": 0.91,
        "risk_level": "low",
        "requires_approval": False,
        "reason": "Smoke prediction.",
        "similar_incidents": [],
        "trace": {
            "decision_id": "decision-123",
            "artifact_id": artifact_dir.name,
        },
    }

    report = build_quality_gate_report(
        pipeline_run_id="pipeline-run-123",
        artifact_dir=artifact_dir,
        checks={
            "policy_tests": "passed",
            "artifact_tests": "passed",
            "prediction_contract_tests": "passed",
            "smoke_prediction": "passed",
            "traceability_integration": "passed",
        },
        sample_decision=sample_decision,
        status="passed",
    )

    assert report["quality_gate_report_version"] == 1
    assert report["report_generated_at"]
    assert report["pipeline_run_id"] == "pipeline-run-123"
    assert report["git_sha"]
    assert report["status"] == "passed"

    assert report["artifact"]["artifact_id"] == artifact_dir.name
    assert report["artifact"]["model_version"] == "v1.0.0"
    assert report["artifact"]["created_at"] == "2026-07-03T07:00:00+00:00"
    assert report["artifact"]["run_id"] == "training-run-123"
    assert report["artifact"]["hashes"]["training_data_sha256"] == "e" * 64

    assert report["checks"]["policy_tests"] == "passed"
    assert report["sample_decision"]["predicted_action"] == "open_ticket"
    assert report["limitations"]


def test_write_quality_gate_report_writes_human_and_machine_readable_evidence(
    tmp_path: Path,
):
    output_dir = tmp_path / "evidence" / "quality_gate"

    report = {
        "quality_gate_report_version": 1,
        "report_generated_at": "2026-07-03T07:00:00+00:00",
        "pipeline_run_id": "pipeline-run-123",
        "git_sha": "abc123",
        "status": "passed",
        "artifact": {
            "artifact_dir": "artifacts/log-triage-v1.0.0-20260703-070000",
            "artifact_id": "log-triage-v1.0.0-20260703-070000",
            "model_version": "v1.0.0",
            "created_at": "2026-07-03T07:00:00+00:00",
            "run_id": "training-run-123",
            "manifest_path": "artifacts/log-triage-v1.0.0-20260703-070000/manifest.json",
            "hashes": {
                "model_sha256": "a" * 64,
                "training_data_sha256": "e" * 64,
            },
        },
        "checks": {
            "policy_tests": "passed",
            "smoke_prediction": "passed",
        },
        "sample_decision": {
            "predicted_action": "open_ticket",
            "confidence": 0.91,
        },
        "limitations": [
            "Training data is small and learning-oriented.",
        ],
    }

    write_quality_gate_report(report=report, output_dir=output_dir)

    assert (output_dir / "quality_gate_report.json").exists()
    assert (output_dir / "quality_gate_report.md").exists()
    assert (output_dir / "sample_decision.json").exists()
    assert (output_dir / "manifest_hashes.json").exists()

    written_report = json.loads(
        (output_dir / "quality_gate_report.json").read_text(encoding="utf-8")
    )
    assert written_report["pipeline_run_id"] == "pipeline-run-123"

    sample_decision = json.loads(
        (output_dir / "sample_decision.json").read_text(encoding="utf-8")
    )
    assert sample_decision["predicted_action"] == "open_ticket"

    manifest_hashes = json.loads(
        (output_dir / "manifest_hashes.json").read_text(encoding="utf-8")
    )
    assert manifest_hashes["training_data_sha256"] == "e" * 64

    markdown = (output_dir / "quality_gate_report.md").read_text(encoding="utf-8")
    assert "# Quality Gate Report" in markdown
    assert "pipeline-run-123" in markdown
    assert "log-triage-v1.0.0-20260703-070000" in markdown
