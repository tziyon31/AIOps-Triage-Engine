from src.log_triage.pipeline import (
    FEATURE_NAMES,
    build_features_from_log,
    build_dataset,
    explain_features,
    parse_log_line,
)


def test_parse_log_line_extracts_basic_fields():
    line = "2026-05-03 09:20:11 ERROR payments db timeout cpu 93 memory 84 action open_ticket"

    event = parse_log_line(line)

    assert event["date"] == "2026-05-03"
    assert event["time"] == "09:20:11"
    assert event["level"] == "ERROR"
    assert event["service"] == "payments"
    assert event["message"] == "db timeout"
    assert event["cpu"] == "93"
    assert event["memory"] == "84"
    assert event["action"] == "open_ticket"


def test_build_features_from_log_matches_feature_names_length():
    line = "2026-05-03 09:20:11 ERROR payments db timeout cpu 93 memory 84 action open_ticket"

    features = build_features_from_log(line)

    assert len(features) == len(FEATURE_NAMES)


def test_missing_cpu_gets_missing_flag():
    line = "2026-05-03 09:20:11 ERROR payments db timeout memory 84 action open_ticket"

    features = build_features_from_log(line)
    readable = explain_features(features)

    assert readable["cpu"] == -1
    assert readable["cpu_missing"] == 1
    assert readable["memory"] == 84
    assert readable["memory_missing"] == 0


def test_build_dataset_skips_missing_label():
    raw_logs = [
        "2026-05-03 09:20:11 ERROR payments db timeout cpu 93 memory 84 action open_ticket",
        "2026-05-03 09:21:11 ERROR payments db timeout cpu 91 memory 80",
    ]

    X, y, skipped = build_dataset(raw_logs)

    assert len(X) == 1
    assert len(y) == 1
    assert len(skipped) == 1
    assert "missing label" in skipped[0]["error"]
