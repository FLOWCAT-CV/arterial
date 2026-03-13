#!/usr/bin/env python3

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


DEFAULT_CHUNK_SIZE_MIB = 48
MANIFEST_VERSION = 1
PRESERVED_OUTPUT_FILES = {".gitkeep", "README.md"}


@dataclass(frozen=True)
class ExportTarget:
    source_path: Path
    relative_path: str


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Split repo-local .pth files into deterministic Git-friendly chunks."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=repo_root,
        help="Repository root. Defaults to the parent of this script.",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=repo_root / "arterial",
        help="Directory to scan for .pth files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repo_root / "model_chunks",
        help="Directory where chunk files and the manifest will be written.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help="Manifest path. Defaults to <output-root>/manifest.json.",
    )
    parser.add_argument(
        "--chunk-size-mib",
        type=int,
        default=DEFAULT_CHUNK_SIZE_MIB,
        help=f"Chunk size in MiB. Defaults to {DEFAULT_CHUNK_SIZE_MIB}.",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="GLOB",
        help="Limit export to repo-relative file globs like 'arterial/segmentation/**/*.pth'.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove previously generated chunk files under the output directory before exporting.",
    )
    return parser.parse_args()


def normalize_path(path: Path) -> Path:
    return path.expanduser().resolve()


def ensure_relative_to(path: Path, root: Path, label: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} must live inside {root}") from exc


def should_include(relative_path: str, patterns: list[str]) -> bool:
    if not patterns:
        return True
    return any(fnmatch.fnmatch(relative_path, pattern) for pattern in patterns)


def discover_targets(repo_root: Path, source_root: Path, patterns: list[str]) -> list[ExportTarget]:
    targets: list[ExportTarget] = []
    for source_path in sorted(source_root.rglob("*.pth")):
        if not source_path.is_file():
            continue
        relative_path = source_path.relative_to(repo_root).as_posix()
        if should_include(relative_path, patterns):
            targets.append(ExportTarget(source_path=source_path, relative_path=relative_path))
    return targets


def clear_output_root(output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    for child in output_root.iterdir():
        if child.name in PRESERVED_OUTPUT_FILES:
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def remove_existing_parts(output_root: Path, relative_path: str) -> None:
    target_prefix = output_root / relative_path
    for existing_part in sorted(target_prefix.parent.glob(f"{target_prefix.name}.part*")):
        existing_part.unlink()


def chunk_file(
    source_path: Path,
    output_root: Path,
    relative_path: str,
    chunk_size_bytes: int,
    repo_root: Path,
) -> dict:
    remove_existing_parts(output_root, relative_path)

    source_sha256 = hashlib.sha256()
    part_entries: list[dict] = []
    size_bytes = 0
    part_index = 0
    target_prefix = output_root / relative_path
    target_prefix.parent.mkdir(parents=True, exist_ok=True)

    with source_path.open("rb") as source_handle:
        while True:
            chunk = source_handle.read(chunk_size_bytes)
            if not chunk:
                break

            part_path = Path(f"{target_prefix}.part{part_index:03d}")
            part_path.write_bytes(chunk)

            source_sha256.update(chunk)
            size_bytes += len(chunk)
            part_entries.append(
                {
                    "path": part_path.relative_to(repo_root).as_posix(),
                    "size_bytes": len(chunk),
                    "sha256": hashlib.sha256(chunk).hexdigest(),
                }
            )
            part_index += 1

    return {
        "relative_path": relative_path,
        "size_bytes": size_bytes,
        "sha256": source_sha256.hexdigest(),
        "chunk_count": len(part_entries),
        "chunks": part_entries,
    }


def write_manifest(manifest_path: Path, manifest: dict) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = manifest_path.with_suffix(".json.tmp")
    temp_path.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    temp_path.replace(manifest_path)


def build_manifest(
    files: Iterable[dict],
    source_root: Path,
    output_root: Path,
    chunk_size_bytes: int,
    repo_root: Path,
) -> dict:
    file_entries = list(files)
    total_bytes = sum(entry["size_bytes"] for entry in file_entries)
    return {
        "version": MANIFEST_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": ".",
        "source_root": source_root.relative_to(repo_root).as_posix(),
        "output_root": output_root.relative_to(repo_root).as_posix(),
        "chunk_size_bytes": chunk_size_bytes,
        "file_count": len(file_entries),
        "total_bytes": total_bytes,
        "files": file_entries,
    }


def main() -> int:
    args = parse_args()
    repo_root = normalize_path(args.repo_root)
    source_root = normalize_path(args.source_root)
    output_root = normalize_path(args.output_root)
    manifest_path = normalize_path(args.manifest_path) if args.manifest_path else output_root / "manifest.json"
    chunk_size_bytes = args.chunk_size_mib * 1024 * 1024

    ensure_relative_to(source_root, repo_root, "--source-root")
    ensure_relative_to(output_root, repo_root, "--output-root")
    ensure_relative_to(manifest_path, repo_root, "--manifest-path")

    if chunk_size_bytes <= 0:
        raise ValueError("--chunk-size-mib must be positive")

    if args.clean:
        clear_output_root(output_root)
    else:
        output_root.mkdir(parents=True, exist_ok=True)

    targets = discover_targets(repo_root, source_root, args.only)
    if not targets:
        print("No matching .pth files found.")
        return 0

    file_entries = [
        chunk_file(target.source_path, output_root, target.relative_path, chunk_size_bytes, repo_root)
        for target in targets
    ]
    manifest = build_manifest(file_entries, source_root, output_root, chunk_size_bytes, repo_root)
    write_manifest(manifest_path, manifest)

    print(f"Exported {manifest['file_count']} files into {output_root.relative_to(repo_root).as_posix()}")
    print(f"Manifest: {manifest_path.relative_to(repo_root).as_posix()}")
    print(f"Total bytes: {manifest['total_bytes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
