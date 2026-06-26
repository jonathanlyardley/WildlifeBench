#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
from pathlib import Path

try:
    from PIL import Image
except Exception:
    Image = None


ROOT = Path(__file__).resolve().parents[1]


def default_data_root() -> Path:
    override = os.environ.get("WBGBIF_DATA_PACKAGE")
    if override:
        return Path(override).expanduser().resolve()
    sibling = ROOT.parent / "data_package"
    if (sibling / "image_hashes.csv").exists():
        return sibling.resolve()
    return ROOT / "data"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate WildlifeBench GBIF award media metadata.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=default_data_root(),
        help="Path to the data package root. Defaults to ../data_package when present, then ./data.",
    )
    args = parser.parse_args()
    data_root = args.data_root.expanduser().resolve()
    rows = list(csv.DictReader((data_root / "image_hashes.csv").open(encoding="utf-8")))
    missing_required = []
    checked = 0
    for row in rows:
        for field in ("event_id", "media_id", "licence", "licence_url", "redistribution_decision", "source_export_sha256"):
            if not row.get(field):
                missing_required.append((row.get("media_id"), field))
        if row.get("file_present") != "true":
            continue
        path = data_root / row["relative_path"]
        if not path.exists():
            print(f"declared media file missing: {path}", file=sys.stderr)
            return 1
        if row.get("sha256") and sha256_file(path) != row["sha256"]:
            print(f"sha256 mismatch: {path}", file=sys.stderr)
            return 1
        if Image is not None:
            with Image.open(path) as img:
                if len(img.getexif()):
                    print(f"EXIF metadata remains: {path}", file=sys.stderr)
                    return 1
        checked += 1
    if missing_required:
        print(f"media metadata missing required fields: {missing_required[:5]}", file=sys.stderr)
        return 1
    print(f"media metadata check passed; data_root={data_root}; files checked={checked}, rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
