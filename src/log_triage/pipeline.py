"""Parse log lines and build numeric feature rows + class indices."""

NUMERIC_FEATURES = {
    "cpu": {"default": -1, "min": 0, "max": 100, "add_missing_flag": True},
    "memory": {"default": -1, "min": 0, "max": 100, "add_missing_flag": True},
}

KNOWN_SERVICES = ["payments", "web", "api"]
KNOWN_LEVELS = ["ERROR", "INFO", "WARNING"]
KNOWN_ACTIONS = ["open_ticket", "ignore", "scale_up"]


def build_feature_names():
    names = []

    for field_name, config in NUMERIC_FEATURES.items():
        names.append(field_name)

        if config["add_missing_flag"]:
            names.append(f"{field_name}_missing")

    names.extend(f"service_is_{service}" for service in KNOWN_SERVICES)
    names.extend(f"level_is_{level}" for level in KNOWN_LEVELS)
    names.extend(["has_timeout", "has_latency"])

    return names


FEATURE_NAMES = build_feature_names()


def get_value_after_token(parts, token):
    if token not in parts:
        return None

    token_index = parts.index(token)

    if token_index + 1 >= len(parts):
        return None

    return parts[token_index + 1]


def parse_log_line(line):
    parts = line.split()

    event = {
        "date": parts[0],
        "time": parts[1],
        "level": parts[2],
        "service": parts[3],
    }

    cpu_index = parts.index("cpu") if "cpu" in parts else None
    memory_index = parts.index("memory") if "memory" in parts else None
    action_index = parts.index("action") if "action" in parts else None

    possible_end_indexes = [
        index for index in [cpu_index, memory_index, action_index]
        if index is not None
    ]

    message_end_index = min(possible_end_indexes) if possible_end_indexes else len(parts)

    event["message"] = " ".join(parts[4:message_end_index])
    event["cpu"] = get_value_after_token(parts, "cpu")
    event["memory"] = get_value_after_token(parts, "memory")
    event["action"] = get_value_after_token(parts, "action")

    event["service"] = str(event["service"]).lower()
    event["level"] = str(event["level"]).upper()
    event["message"] = str(event["message"]).lower()

    return event


def clean_numeric_value(value, config):
    if value is None:
        return config["default"], 1

    try:
        numeric_value = int(value)
    except ValueError:
        return config["default"], 1

    if numeric_value < config["min"] or numeric_value > config["max"]:
        return config["default"], 1

    return numeric_value, 0


def one_hot(value, known_values):
    return [int(value == item) for item in known_values]


def extract_features(event):
    features = []

    for field_name, config in NUMERIC_FEATURES.items():
        raw_value = event.get(field_name)
        value, missing_flag = clean_numeric_value(raw_value, config)

        features.append(value)

        if config["add_missing_flag"]:
            features.append(missing_flag)

    message = event["message"]

    features.extend(one_hot(event["service"], KNOWN_SERVICES))
    features.extend(one_hot(event["level"], KNOWN_LEVELS))

    features.extend([
        int("timeout" in message),
        int("latency" in message),
    ])

    return features


def extract_label(event):
    action = event.get("action")

    if action is None:
        raise ValueError("missing label: action")

    if action not in KNOWN_ACTIONS:
        raise ValueError(f"unknown action: {action!r}")

    return KNOWN_ACTIONS.index(action)


def explain_features(features):
    return dict(zip(FEATURE_NAMES, features))


def _event_and_features_from_line(line):
    """Single parse + feature vector (shared by build_dataset and build_features_from_log)."""
    event = parse_log_line(line)
    features = extract_features(event)
    return event, features


def build_features_from_log(line):
    """Parse one raw log line and return its numeric feature row (inference / reuse)."""
    _, features = _event_and_features_from_line(line)
    return features


def build_dataset(raw_logs):
    X = []
    y = []
    skipped = []

    for line in raw_logs:
        try:
            event, features = _event_and_features_from_line(line)
            label = extract_label(event)

            X.append(features)
            y.append(label)

        except ValueError as error:
            skipped.append({
                "line": line,
                "error": str(error),
            })

    assert all(len(row) == len(FEATURE_NAMES) for row in X), "Feature length mismatch"

    return X, y, skipped
