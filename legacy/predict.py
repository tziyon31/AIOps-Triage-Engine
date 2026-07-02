"""Prediction API (reusable from FastAPI, tests, notebooks). CLI is only in main()."""

import sys  # type: ignore[import-untyped]
from pathlib import Path

import joblib  # type: ignore[import-not-found]

from src.log_triage.pipeline import FEATURE_NAMES, build_features_from_log, explain_features

_LEGACY_DIR = Path(__file__).resolve().parent
_DEFAULT_MODEL = _LEGACY_DIR / "model.pkl"

DEFAULT_LOG = "2026-05-03 09:20:11 ERROR payments db timeout cpu 93 memory 84"


def decode_label(encoded_label, known_actions):
    return known_actions[int(encoded_label)]


def predict_logs(model, raw_logs, known_actions):
    feature_rows = [
        build_features_from_log(raw_log)
        for raw_log in raw_logs
    ]

    predictions = model.predict(feature_rows)

    results = []

    for raw_log, features, prediction in zip(raw_logs, feature_rows, predictions):
        results.append({
            "raw_log": raw_log,
            "features": explain_features(features),
            "predicted_action": decode_label(prediction, known_actions),
        })

    return results


def predict_log(model, raw_log, known_actions):
    return predict_logs(model, [raw_log], known_actions)[0]


def load_artifact(path=None):
    if path is None:
        path = _DEFAULT_MODEL
    artifact = joblib.load(path)
    model = artifact["model"]
    feature_names = artifact["feature_names"]
    known_actions = artifact["known_actions"]

    if FEATURE_NAMES != feature_names:
        raise RuntimeError(
            "Feature schema mismatch between training artifact and current pipeline"
        )

    return model, feature_names, known_actions


def main():
    model, _feature_names, known_actions = load_artifact()

    raw_log = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_LOG

    result = predict_log(model, raw_log, known_actions)

    print(result)


if __name__ == "__main__":
    main()
