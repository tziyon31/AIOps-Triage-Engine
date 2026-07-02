from pathlib import Path

from src.log_triage.config import load_training_config


def load_raw_logs(path=None):
    if path is None:
        path = load_training_config()["raw_logs_path"]

    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"data file not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as file:
        lines = file.readlines()

    raw_logs = []

    for line in lines:
        line = line.strip()

        if not line:
            continue

        raw_logs.append(line)

    return raw_logs
