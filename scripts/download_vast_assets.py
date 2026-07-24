"""Download Vast.ai assets from public Google Drive folders.

This script uses only Python's standard library. It does not require gdown,
the Google Drive API client, OAuth credentials, or any plugin.

Assets:
  models     -> models/
  chroma     -> data/chroma/
  embeddings -> data/embeddings/

Downloads are resumable through .part files and can be repeated safely on a
persistent Vast.ai volume.

Usage:
  python scripts/download_vast_assets.py
  python scripts/download_vast_assets.py --asset models
  python scripts/download_vast_assets.py --asset all --dry-run
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from html import unescape
from pathlib import Path
import re
import sys
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parent.parent
USER_AGENT = "rag-skripsi-drive-downloader/1.0"
CHUNK_SIZE = 8 * 1024 * 1024

FOLDERS = {
    "models": {
        "id": "1yvLAXCMKhDEizkQXDq_WmkUBxrJui4GJ",
        "target": ROOT / "models",
    },
    "chroma": {
        "id": "19Abmg-di7A1i4zLE5JuVm8b7ez8Pez_U",
        "target": ROOT / "data" / "chroma",
    },
    "embeddings": {
        "id": "13KHF1CaRx-vUBVcBR8Sl7WhCrmVkBmfW",
        "target": ROOT / "data" / "embeddings",
    },
}

FOLDER_URL = "https://drive.google.com/drive/folders/{folder_id}?usp=sharing"
FILE_URL = (
    "https://drive.usercontent.google.com/download"
    "?id={file_id}&export=download&confirm=t"
)


@dataclass(frozen=True)
class DriveItem:
    file_id: str
    name: str
    is_folder: bool


def _request(url: str, headers: dict[str, str] | None = None):
    request_headers = {"User-Agent": USER_AGENT}
    if headers:
        request_headers.update(headers)
    return urlopen(Request(url, headers=request_headers), timeout=120)


def _clean_name(value: str) -> str:
    value = unescape(re.sub(r"<[^>]+>", "", value)).strip()
    value = value.replace("/", "_").replace("\\", "_")
    return value or "unnamed"


def list_folder(folder_id: str) -> list[DriveItem]:
    """Parse public Drive folder rows without using the Drive API."""
    try:
        with _request(FOLDER_URL.format(folder_id=folder_id)) as response:
            html = response.read().decode("utf-8", "replace")
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"Cannot read public Drive folder {folder_id}: {exc}") from exc

    pattern = re.compile(
        r'<div\b(?=[^>]*\bdata-id="([^"]+)")'
        r'(?=[^>]*\bjsname="vtaz5c")'
        r'(?=[^>]*\bdata-tooltip="([^"]*)")[^>]*>',
        re.IGNORECASE,
    )
    items: list[DriveItem] = []
    seen_ids: set[str] = set()
    matches = list(pattern.finditer(html))
    for index, match in enumerate(matches):
        file_id, tooltip = match.groups()
        if file_id in seen_ids:
            continue
        seen_ids.add(file_id)

        next_start = matches[index + 1].start() if index + 1 < len(matches) else len(html)
        snippet = html[match.end():next_start]
        name_match = re.search(r"<strong[^>]*>(.*?)</strong>", snippet, re.IGNORECASE | re.DOTALL)
        name = _clean_name(name_match.group(1) if name_match else tooltip)
        is_folder = "shared folder" in unescape(tooltip).lower()
        items.append(DriveItem(file_id=file_id, name=name, is_folder=is_folder))

    if not items:
        raise RuntimeError(
            f"No items found in Drive folder {folder_id}. "
            "The folder may no longer be public or Google changed its HTML."
        )
    return items


def collect_files(
    folder_id: str,
    relative_dir: Path = Path(),
    visited: set[str] | None = None,
) -> list[tuple[DriveItem, Path]]:
    """Recursively collect files and their relative output paths."""
    if visited is None:
        visited = set()
    if folder_id in visited:
        raise RuntimeError(f"Drive folder cycle detected at {folder_id}")
    visited.add(folder_id)

    files: list[tuple[DriveItem, Path]] = []
    for item in list_folder(folder_id):
        # Drive model folders contain an internal .cache directory that is not
        # needed to load the models and can contain inaccessible metadata.
        if item.name.startswith("."):
            continue
        output_path = relative_dir / item.name
        if item.is_folder:
            files.extend(collect_files(item.file_id, output_path, visited))
        else:
            files.append((item, output_path))
    return files


def _remote_size(file_id: str) -> int | None:
    request = Request(
        FILE_URL.format(file_id=file_id),
        headers={"User-Agent": USER_AGENT, "Range": "bytes=0-0"},
    )
    try:
        with urlopen(request, timeout=120) as response:
            content_range = response.headers.get("Content-Range", "")
            match = re.search(r"/(\d+)$", content_range)
            if match:
                return int(match.group(1))
            content_length = response.headers.get("Content-Length")
            return int(content_length) if content_length else None
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"Cannot inspect Drive file {file_id}: {exc}") from exc


def download_file(item: DriveItem, target: Path, dry_run: bool = False) -> str:
    """Download one public Drive file atomically and resume partial files."""
    total = _remote_size(item.file_id)
    if dry_run:
        size_text = f" ({total / (1024 ** 3):.2f} GiB)" if total else ""
        return f"DRY-RUN {item.name}{size_text} -> {target}"

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(f"{target}.part")
    if target.exists() and total is not None and target.stat().st_size == total:
        return f"SKIP {target}"
    if target.exists() and not partial.exists():
        target.replace(partial)

    start = partial.stat().st_size if partial.exists() else 0
    if total is not None and start > total:
        partial.unlink()
        start = 0

    headers = {"User-Agent": USER_AGENT}
    if start:
        headers["Range"] = f"bytes={start}-"
    request = Request(FILE_URL.format(file_id=item.file_id), headers=headers)
    try:
        with urlopen(request, timeout=120) as response:
            append = start > 0 and response.status == 206
            if not append:
                start = 0
            mode = "ab" if append else "wb"
            with partial.open(mode) as output:
                while True:
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    output.write(chunk)
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"Download failed for {item.name}: {exc}") from exc

    actual = partial.stat().st_size
    if total is not None and actual != total:
        raise RuntimeError(
            f"Incomplete download for {item.name}: {actual} of {total} bytes. "
            f"Rerun to resume from {partial}."
        )
    partial.replace(target)
    return f"OK {target} ({actual / (1024 ** 3):.2f} GiB)"


def _print_plan(files: Iterable[tuple[DriveItem, Path]], dry_run: bool) -> None:
    prefix = "DRY-RUN" if dry_run else "PLAN"
    for item, target in files:
        print(f"{prefix} {item.name} -> {target}")


def download_asset(asset: str, dry_run: bool, workers: int) -> None:
    config = FOLDERS[asset]
    print(f"\n[{asset}] {config['id']} -> {config['target']}")
    files = collect_files(config["id"])
    print(f"Found {len(files)} files")
    if dry_run:
        _print_plan(files, dry_run=True)
        return

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(download_file, item, config["target"] / relative): item
            for item, relative in files
        }
        for future in as_completed(futures):
            print(future.result())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download public Google Drive assets for Vast.ai RAG"
    )
    parser.add_argument(
        "--asset",
        choices=["all", *FOLDERS],
        default="all",
        help="Asset group to download (default: all)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel file downloads (default: 4)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List public files and targets without downloading data",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    assets = list(FOLDERS) if args.asset == "all" else [args.asset]
    print("Vast.ai Google Drive asset setup")
    print("Only Python standard library is used.")
    for asset in assets:
        download_asset(asset, dry_run=args.dry_run, workers=args.workers)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
