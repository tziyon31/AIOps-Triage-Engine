import json
from pathlib import Path

import pytest

from src.log_triage.artifact_version import (
    build_versioned_artifact_dir_name,
    find_latest_artifact_for_version,
    parse_versioned_artifact_dir,
    resolve_patch,
)


def test_parse_versioned_artifact_dir():
    parsed = parse_versioned_artifact_dir("log-triage-v1.0.3-20260702-090000")

    assert parsed == {
        "name": "log-triage",
        "major": 1,
        "minor": 0,
        "patch": 3,
        "timestamp": "20260702-090000",
    }


def test_resolve_patch_first_artifact():
    assert resolve_patch({"model_sha256": "abc"}, last_patch=-1, last_hashes=None) == 0


def test_resolve_patch_same_hashes_keeps_patch():
    hashes = {"model_sha256": "abc", "vectorizer_sha256": "def"}
    assert resolve_patch(hashes, last_patch=3, last_hashes=hashes) == 3


def test_resolve_patch_changed_hashes_increments_patch():
    old_hashes = {"model_sha256": "abc"}
    new_hashes = {"model_sha256": "xyz"}
    assert resolve_patch(new_hashes, last_patch=3, last_hashes=old_hashes) == 4


def test_find_latest_artifact_for_version_uses_highest_patch(tmp_path: Path):
    name = "log-triage"
    major = 1
    minor = 0

    first = tmp_path / build_versioned_artifact_dir_name(name, major, minor, 0, "20260702-090000")
    second = tmp_path / build_versioned_artifact_dir_name(name, major, minor, 2, "20260702-100000")
    first.mkdir()
    second.mkdir()

    manifest = {
        "hashes": {
            "model_sha256": "latest",
            "vectorizer_sha256": "latest",
            "known_actions_sha256": "latest",
            "config_sha256": "latest",
        }
    }
    with (second / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file)

    last_patch, last_hashes = find_latest_artifact_for_version(tmp_path, name, major, minor)

    assert last_patch == 2
    assert last_hashes == manifest["hashes"]
