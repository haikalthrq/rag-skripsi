"""Tests for the faculty source-code archive policy."""

import json
import sys
from pathlib import Path
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import create_faculty_submission as submission


def test_allowlist_contains_source_and_excludes_runtime_assets():
    relative_paths = {
        path.relative_to(ROOT).as_posix()
        for path in submission.collect_submission_files()
    }

    assert "README.md" in relative_paths
    assert "src/rag/pipeline.py" in relative_paths
    assert "scripts/create_faculty_submission.py" in relative_paths
    assert "data/ground_truth/qa_pairs_binary.json" in relative_paths
    assert "results/final/analysis10/figures/f1_at_k_top1_10.png" in relative_paths
    assert not any(path.startswith("models/") for path in relative_paths)
    assert not any(path.startswith("data/embeddings/") for path in relative_paths)
    assert not any(path.startswith("results/chat_history/") for path in relative_paths)
    assert not any("__pycache__" in path for path in relative_paths)


def test_sanitize_notebook_removes_outputs_and_execution_metadata():
    raw = json.dumps({
        "cells": [{
            "cell_type": "code",
            "execution_count": 7,
            "metadata": {"execution": {"started": "now"}},
            "outputs": [{"output_type": "stream", "text": ["secret path"]}],
            "source": ["print('ok')"],
        }],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11.9"},
            "widgets": {"state": {}},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }).encode("utf-8")

    notebook = json.loads(submission.sanitize_notebook(raw))

    assert notebook["cells"][0]["execution_count"] is None
    assert notebook["cells"][0]["outputs"] == []
    assert notebook["cells"][0]["metadata"] == {}
    assert "widgets" not in notebook["metadata"]
    assert "version" not in notebook["metadata"]["language_info"]


def test_build_submission_writes_manifest_and_external_checksum(tmp_path):
    zip_path, checksum_path = submission.build_submission(tmp_path / "submission.zip")

    assert zip_path.is_file()
    assert checksum_path.is_file()
    with ZipFile(zip_path) as archive:
        names = set(archive.namelist())
        prefix = f"{submission.PACKAGE_ROOT}/"
        assert f"{prefix}PACKAGE_INFO.md" in names
        assert f"{prefix}SHA256SUMS.txt" in names
        assert f"{prefix}README.md" in names
        assert not any("models/" in name for name in names)
        assert not any("chat_history" in name for name in names)
