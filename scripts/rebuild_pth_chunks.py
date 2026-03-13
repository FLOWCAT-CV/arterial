#!/usr/bin/env python3

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Rebuild .pth files from deterministic Git-stored chunks."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=repo_root,
        help="Repository root. Defaults to the parent of this script.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=repo_root / "model_chunks" / "manifest.json",
        help="Manifest path. Defaults to model_chunks/manifest.json.",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="GLOB",
        help="Limit rebuild to repo-relative file globs like 'arterial/segmentation/**/*.pth'.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files that already exist locally with the expected size and SHA256.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify existing reconstructed files without writing anything.",
    )
    return parser.parse_args()


def normalize_path(path: Path) -> Path:
    return path.expanduser().resolve()


def should_include(relative_path: str, patterns: list[str]) -> bool:
    if not patterns:
        return True
    return any(fnmatch.fnmatch(relative_path, pattern) for pattern in patterns)


def sha256_of_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def load_manifest(manifest_path: Path) -> dict:
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("version") != 1:
        raise ValueError(f"Unsupported manifest version: {manifest.get('version')}")
    return manifest


def verify_existing_file(destination: Path, file_entry: dict) -> tuple[bool, str]:
    if not destination.exists():
        return False, "missing"
    if destination.stat().st_size != file_entry["size_bytes"]:
        return False, "size mismatch"
    if sha256_of_file(destination) != file_entry["sha256"]:
        return False, "sha256 mismatch"
    return True, "ok"


def verify_chunk(repo_root: Path, chunk_entry: dict) -> bytes:
    chunk_path = repo_root / chunk_entry["path"]
    if not chunk_path.exists():
        raise FileNotFoundError(f"Missing chunk: {chunk_entry['path']}")
    data = chunk_path.read_bytes()
    if len(data) != chunk_entry["size_bytes"]:
        raise ValueError(f"Chunk size mismatch for {chunk_entry['path']}")
    if hashlib.sha256(data).hexdigest() != chunk_entry["sha256"]:
        raise ValueError(f"Chunk hash mismatch for {chunk_entry['path']}")
    return data


def rebuild_file(repo_root: Path, file_entry: dict) -> None:
    destination = repo_root / file_entry["relative_path"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_destination = destination.with_suffix(destination.suffix + ".tmp")

    hasher = hashlib.sha256()
    total_size = 0
    with temp_destination.open("wb") as output_handle:
        for chunk_entry in file_entry["chunks"]:
            data = verify_chunk(repo_root, chunk_entry)
            output_handle.write(data)
            hasher.update(data)
            total_size += len(data)

    if total_size != file_entry["size_bytes"]:
        temp_destination.unlink(missing_ok=True)
        raise ValueError(f"Rebuilt size mismatch for {file_entry['relative_path']}")
    if hasher.hexdigest() != file_entry["sha256"]:
        temp_destination.unlink(missing_ok=True)
        raise ValueError(f"Rebuilt hash mismatch for {file_entry['relative_path']}")

    temp_destination.replace(destination)


def main() -> int:
    args = parse_args()
    repo_root = normalize_path(args.repo_root)
    manifest_path = normalize_path(args.manifest_path)
    manifest = load_manifest(manifest_path)

    rebuilt = 0
    skipped = 0
    verified = 0
    files = manifest.get("files", [])

    for file_entry in files:
        relative_path = file_entry["relative_path"]
        if not should_include(relative_path, args.only):
            continue

        destination = repo_root / relative_path
        if args.verify_only:
            ok, reason = verify_existing_file(destination, file_entry)
            if not ok:
                raise ValueError(f"Verification failed for {relative_path}: {reason}")
            verified += 1
            print(f"Verified {relative_path}")
            continue

        if args.skip_existing:
            ok, _ = verify_existing_file(destination, file_entry)
            if ok:
                skipped += 1
                print(f"Skipped {relative_path}")
                continue

        rebuild_file(repo_root, file_entry)
        rebuilt += 1
        print(f"Rebuilt {relative_path}")

    print(f"Rebuilt: {rebuilt}, skipped: {skipped}, verified: {verified}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
