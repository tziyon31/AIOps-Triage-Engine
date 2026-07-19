from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class BoundaryRule:
    name: str
    path_prefixes: tuple[str, ...]
    forbidden_imports: tuple[str, ...]
    reason: str


BOUNDARY_RULES = (
    BoundaryRule(
        name="api_must_not_import_training_or_infra_internals",
        path_prefixes=(
            "src/log_triage/api/",
            "src/log_triage/adapters/api/",
        ),
        forbidden_imports=(
            "src.log_triage.train",
            "mlflow",
            "sklearn",
            "scipy",
            "joblib",
            "scripts.promote",
            "scripts.compare_runs",
            "scripts.build_experiment_history_report",
            "scripts.build_promotion_status_report",
        ),
        reason=(
            "API adapters must call DecisionService instead of training, "
            "MLflow, sklearn, or operator scripts."
        ),
    ),
    BoundaryRule(
        name="application_must_not_import_external_infra_sdks",
        path_prefixes=(
            "src/log_triage/application/",
        ),
        forbidden_imports=(
            "fastapi",
            "mlflow",
            "kubernetes",
            "prometheus_api_client",
            "boto3",
            "botocore",
            "psycopg",
            "psycopg2",
            "sqlalchemy",
            "openai",
        ),
        reason=(
            "Application/Core code must depend on stable interfaces, "
            "not external SDKs or HTTP framework packages."
        ),
    ),
    BoundaryRule(
        name="schemas_must_stay_sdk_free",
        path_prefixes=(
            "src/log_triage/schemas.py",
        ),
        forbidden_imports=(
            "fastapi",
            "mlflow",
            "kubernetes",
            "prometheus_api_client",
            "boto3",
            "botocore",
            "psycopg",
            "psycopg2",
            "sqlalchemy",
            "openai",
            "sklearn",
            "scipy",
            "joblib",
        ),
        reason=(
            "Schemas define stable contracts and must not depend on infra, "
            "provider, training, or persistence SDKs."
        ),
    ),
    BoundaryRule(
        name="policy_must_not_query_infrastructure",
        path_prefixes=(
            "src/log_triage/policy.py",
        ),
        forbidden_imports=(
            "kubernetes",
            "prometheus_api_client",
            "boto3",
            "botocore",
            "psycopg",
            "psycopg2",
            "sqlalchemy",
            "mlflow",
            "openai",
        ),
        reason=(
            "Policy may validate a decision, but must not query infrastructure "
            "or providers directly."
        ),
    ),
)


def normalize_repo_path(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def iter_python_source_files() -> list[Path]:
    return sorted((PROJECT_ROOT / "src" / "log_triage").rglob("*.py"))


def imported_modules_from_source(source: str) -> set[str]:
    tree = ast.parse(source)
    imports: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)

        if isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module)

    return imports


def import_matches(imported_module: str, forbidden_module: str) -> bool:
    return (
        imported_module == forbidden_module
        or imported_module.startswith(f"{forbidden_module}.")
    )


def path_matches_rule(repo_path: str, rule: BoundaryRule) -> bool:
    return any(
        repo_path == prefix.rstrip("/")
        or repo_path.startswith(prefix)
        for prefix in rule.path_prefixes
    )


def find_boundary_violations_for_source(
    *,
    repo_path: str,
    source: str,
    rules: tuple[BoundaryRule, ...] = BOUNDARY_RULES,
) -> list[str]:
    imports = imported_modules_from_source(source)
    violations: list[str] = []

    for rule in rules:
        if not path_matches_rule(repo_path, rule):
            continue

        for imported_module in sorted(imports):
            for forbidden_module in rule.forbidden_imports:
                if import_matches(imported_module, forbidden_module):
                    violations.append(
                        f"{repo_path}: forbidden import '{imported_module}' "
                        f"matched rule '{rule.name}'. Reason: {rule.reason}"
                    )

    return violations


def find_repo_boundary_violations() -> list[str]:
    violations: list[str] = []

    for path in iter_python_source_files():
        repo_path = normalize_repo_path(path)
        source = path.read_text(encoding="utf-8")
        violations.extend(
            find_boundary_violations_for_source(
                repo_path=repo_path,
                source=source,
            )
        )

    return violations


def test_architecture_boundary_rules_are_not_violated():
    violations = find_repo_boundary_violations()

    assert violations == [], (
        "Architecture boundary violations found:\n"
        + "\n".join(f"- {violation}" for violation in violations)
    )


def test_boundary_rule_catches_future_api_importing_train_py():
    source = "from src.log_triage.train import build_decision_artifact\n"

    violations = find_boundary_violations_for_source(
        repo_path="src/log_triage/api/routes.py",
        source=source,
    )

    assert violations
    assert "src.log_triage.train" in violations[0]


def test_boundary_rule_catches_application_importing_mlflow():
    source = "import mlflow\n"

    violations = find_boundary_violations_for_source(
        repo_path="src/log_triage/application/decision_service.py",
        source=source,
    )

    assert violations
    assert "mlflow" in violations[0]


def test_boundary_rule_catches_schema_importing_provider_sdk():
    source = "from openai import OpenAI\n"

    violations = find_boundary_violations_for_source(
        repo_path="src/log_triage/schemas.py",
        source=source,
    )

    assert violations
    assert "openai" in violations[0]


def test_boundary_rule_catches_policy_querying_kubernetes():
    source = "from kubernetes import client\n"

    violations = find_boundary_violations_for_source(
        repo_path="src/log_triage/policy.py",
        source=source,
    )

    assert violations
    assert "kubernetes" in violations[0]
