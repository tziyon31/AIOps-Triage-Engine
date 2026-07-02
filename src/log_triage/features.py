"""Feature extraction; thin layer over pipeline until split is complete."""

from src.log_triage.pipeline import (
    FEATURE_NAMES,
    build_features_from_log,
    explain_features,
    extract_features,
    extract_label,
)

MANUAL_FEATURE_NAMES = FEATURE_NAMES

__all__ = [
    "FEATURE_NAMES",
    "MANUAL_FEATURE_NAMES",
    "build_features_from_log",
    "explain_features",
    "extract_features",
    "extract_label",
]
