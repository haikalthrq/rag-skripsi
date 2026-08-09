"""Build a curated faculty source-code submission ZIP.

The archive is allowlist-based. It intentionally excludes runtime assets,
personal chat history, Git metadata, local tooling, caches, and source
publications. Notebook outputs are stripped before packaging.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = "rag-skripsi-faculty-source"
MAX_FILE_SIZE = 25 * 1024 * 1024

ROOT_FILES = {
    ".gitignore",
    ".python-version",
    "DISTRIBUTION_NOTES.md",
    "README.md",
    "requirements.txt",
    "requirements-dev.txt",
    "requirements-gguf.txt",
    "requirements-visualization.txt",
}

NOTEBOOKS = {
    "notebooks/eval_analysis_top1_10.ipynb",
    "notebooks/rag_inference.ipynb",
}

GROUND_TRUTH_FILES = {
    "data/ground_truth/qa_gold_standard_rag_bps_30qa_question_newest.xlsx",
    "data/ground_truth/qa_pairs_binary.json",
}

ANALYSIS_FILES = {
    "results/final/analysis10/bab6_data_notes_top1_10.md",
    "results/final/analysis10/bab6_tables_top1_10.md",
    "results/final/analysis10/top1_10_audit_notes.md",
    "results/final/analysis10/top1_10_metric_summary_by_method.csv",
    "results/final/analysis10/top1_10_metric_winners.csv",
    "results/final/analysis10/top1_10_metrics_by_k.csv",
    "results/final/analysis10/top1_10_overall_average.csv",
    "results/final/generation/summary_20260725_195535_full_top1-10.csv",
}

FORBIDDEN_TOP_LEVEL = {
    ".codex",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ua",
    ".venv",
    ".vscode",
    "backup",
    "graphify-out",
    "logs",
    "models",
}

FORBIDDEN_ANYWHERE = {"__pycache__"}

FORBIDDEN_PREFIXES = {
    "data/chroma",
    "data/embeddings",
    "results/chat_history",
}

TEXT_SUFFIXES = {
    ".csv",
    ".json",
    ".md",
    ".py",
    ".txt",
}

SECRET_PATTERNS = {
    "private key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "AWS access key": re.compile(rb"\bAKIA[0-9A-Z]{16}\b"),
    "GitHub token": re.compile(rb"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    "Hugging Face token": re.compile(rb"\bhf_[A-Za-z0-9]{20,}\b"),
    "OpenAI-style key": re.compile(rb"\bsk-[A-Za-z0-9_-]{20,}\b"),
}


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def collect_submission_files() -> list[Path]:
    """Return the sorted allowlist of source-submission files."""
    paths = {ROOT / relative for relative in ROOT_FILES}
    paths.update(ROOT / relative for relative in NOTEBOOKS)
    paths.update(ROOT / relative for relative in GROUND_TRUTH_FILES)
    paths.update(ROOT / relative for relative in ANALYSIS_FILES)
    paths.add(ROOT / "docs" / "DEVELOPER_HANDOFF.md")

    for path in (ROOT / "src").rglob("*"):
        if path.is_file() and path.suffix.lower() in {".py", ".md"}:
            paths.add(path)
    for path in (ROOT / "scripts").glob("*.py"):
        paths.add(path)
    for path in (ROOT / "tests").glob("*.py"):
        paths.add(path)
    for path in (ROOT / "results/final/analysis10/figures").glob("*.png"):
        paths.add(path)

    missing = sorted(_relative(path) for path in paths if not path.is_file())
    if missing:
        raise FileNotFoundError(f"Required submission files are missing: {missing}")

    result = sorted(paths, key=_relative)
    validate_allowlist(result)
    return result


def validate_allowlist(paths: list[Path]) -> None:
    """Reject unsafe paths, oversized files, or unexpected data directories."""
    for path in paths:
        relative = Path(_relative(path))
        if relative.parts and relative.parts[0] in FORBIDDEN_TOP_LEVEL:
            raise ValueError(f"Forbidden path selected for submission: {relative}")
        if FORBIDDEN_ANYWHERE.intersection(relative.parts):
            raise ValueError(f"Generated path selected for submission: {relative}")
        if any(
            relative.as_posix() == prefix
            or relative.as_posix().startswith(f"{prefix}/")
            for prefix in FORBIDDEN_PREFIXES
        ):
            raise ValueError(f"Runtime path selected for submission: {relative}")
        if relative.parts[:2] == ("data", "ground_truth"):
            if relative.as_posix() not in GROUND_TRUTH_FILES:
                raise ValueError(f"Ground-truth file is not allowlisted: {relative}")
        elif relative.parts and relative.parts[0] == "data":
            raise ValueError(f"Data directory is not allowed in submission: {relative}")
        if path.stat().st_size > MAX_FILE_SIZE:
            raise ValueError(f"Submission file exceeds {MAX_FILE_SIZE} bytes: {relative}")


def sanitize_notebook(raw: bytes) -> bytes:
    """Remove execution output, counters, widgets, and environment versions."""
    notebook = json.loads(raw.decode("utf-8"))
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        cell["metadata"] = {}

    metadata = notebook.get("metadata", {})
    clean_metadata = {}
    if isinstance(metadata.get("kernelspec"), dict):
        clean_metadata["kernelspec"] = {
            key: value
            for key, value in metadata["kernelspec"].items()
            if key in {"display_name", "language", "name"}
        }
    if isinstance(metadata.get("language_info"), dict):
        clean_metadata["language_info"] = {
            key: value
            for key, value in metadata["language_info"].items()
            if key in {"codemirror_mode", "file_extension", "mimetype", "name", "pygments_lexer"}
        }
    notebook["metadata"] = clean_metadata
    return (json.dumps(notebook, ensure_ascii=False, indent=1) + "\n").encode("utf-8")


def load_payloads(paths: list[Path]) -> dict[str, bytes]:
    """Load and sanitize all selected files, then scan text for common secrets."""
    payloads: dict[str, bytes] = {}
    for path in paths:
        relative = _relative(path)
        raw = path.read_bytes()
        if path.suffix.lower() == ".ipynb":
            raw = sanitize_notebook(raw)
        if path.suffix.lower() in TEXT_SUFFIXES or path.suffix.lower() == ".ipynb":
            for label, pattern in SECRET_PATTERNS.items():
                if pattern.search(raw):
                    raise ValueError(f"Potential {label} found in {relative}")
        payloads[relative] = raw
    return payloads


def get_source_commit() -> str:
    """Return the current commit ID when Git is available."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unavailable"


def build_package_info(source_commit: str, file_count: int) -> bytes:
    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    text = f"""# Package Information

- Purpose: faculty thesis source-code submission
- Generated UTC: {generated}
- Source commit: `{source_commit}`
- Included project files: {file_count}
- Archive policy: explicit allowlist
- Notebook policy: outputs, execution counters, and cell metadata removed

Large runtime assets and source publications are intentionally excluded. Read
`README.md`, `DISTRIBUTION_NOTES.md`, and `docs/DEVELOPER_HANDOFF.md` before
running the project.
"""
    return text.encode("utf-8")


def _zip_info(name: str) -> ZipInfo:
    info = ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
    info.compress_type = ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    return info


def build_submission(output_path: Path) -> tuple[Path, Path]:
    """Build the ZIP and adjacent SHA-256 checksum file."""
    paths = collect_submission_files()
    payloads = load_payloads(paths)
    source_commit = get_source_commit()
    payloads["PACKAGE_INFO.md"] = build_package_info(source_commit, len(paths))

    checksum_lines = [
        f"{hashlib.sha256(payloads[name]).hexdigest()}  {name}"
        for name in sorted(payloads)
    ]
    payloads["SHA256SUMS.txt"] = ("\n".join(checksum_lines) + "\n").encode("ascii")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output_path, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(payloads):
            archive.writestr(_zip_info(f"{PACKAGE_ROOT}/{name}"), payloads[name])

    archive_hash = hashlib.sha256(output_path.read_bytes()).hexdigest()
    checksum_path = output_path.with_suffix(output_path.suffix + ".sha256")
    checksum_path.write_text(f"{archive_hash}  {output_path.name}\n", encoding="ascii")
    return output_path, checksum_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the curated faculty source-code submission ZIP"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output ZIP path (default: dist/rag-skripsi-faculty-source-<commit>.zip)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    commit = get_source_commit()
    default_name = f"rag-skripsi-faculty-source-{commit[:8]}.zip"
    output = args.output or ROOT / "dist" / default_name
    if not output.is_absolute():
        output = ROOT / output
    zip_path, checksum_path = build_submission(output)
    print(f"ZIP: {zip_path}")
    print(f"SHA256: {checksum_path.read_text(encoding='ascii').strip()}")


if __name__ == "__main__":
    main()
