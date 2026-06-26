#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def default_roots() -> list[Path]:
    roots = [ROOT]
    override = os.environ.get("WBGBIF_DATA_PACKAGE")
    if override:
        roots.append(Path(override).expanduser().resolve())
    else:
        sibling = ROOT.parent / "data_package"
        if (sibling / "checksums.sha256").exists():
            roots.append(sibling.resolve())
    unique: list[Path] = []
    for root in roots:
        resolved = root.resolve()
        if resolved not in unique:
            unique.append(resolved)
    return unique


def verify(root: Path) -> tuple[int, int]:
    manifest = root / "checksums.sha256"
    if not manifest.exists():
        print(f"checksum manifest missing: {manifest}", file=sys.stderr)
        return 1, 0
    checked = 0
    with manifest.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                expected, rel = line.rstrip("\n").split(None, 1)
            except ValueError:
                print(f"bad checksum line {manifest}:{line_number}", file=sys.stderr)
                return 1, checked
            path = root / rel.strip()
            if not path.exists():
                print(f"checksummed file missing: {path}", file=sys.stderr)
                return 1, checked
            actual = sha256_file(path)
            if actual != expected:
                print(f"checksum mismatch: {path}", file=sys.stderr)
                return 1, checked
            checked += 1
    return 0, checked


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify generated release checksum manifests.")
    parser.add_argument(
        "--root",
        action="append",
        type=Path,
        help="Root containing checksums.sha256. May be supplied more than once.",
    )
    args = parser.parse_args()
    roots = [path.expanduser().resolve() for path in args.root] if args.root else default_roots()
    total = 0
    for root in roots:
        code, checked = verify(root)
        if code:
            return code
        total += checked
    print(f"checksum check passed; roots={len(roots)}, files={total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
