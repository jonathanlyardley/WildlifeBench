#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def default_data_root() -> Path:
    override = os.environ.get("WBGBIF_DATA_PACKAGE")
    if override:
        return Path(override).expanduser().resolve()
    sibling = ROOT.parent / "data_package"
    if (sibling / "croissant" / "gbif_ebbe_2026_croissant.jsonld").exists():
        return sibling.resolve()
    return ROOT / "data"


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_csv(path: Path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def fail(message: str) -> int:
    print(message, file=sys.stderr)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Croissant and CamtrapDP release metadata.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=default_data_root(),
        help="Path to the data package root. Defaults to ../data_package when present, then ./data.",
    )
    args = parser.parse_args()
    data_root = args.data_root.expanduser().resolve()
    events = read_jsonl(data_root / "model_inputs" / "events.jsonl")
    answer_key = read_jsonl(data_root / "ground_truth" / "answer_key.jsonl")
    media_hashes = read_csv(data_root / "image_hashes.csv")

    croissant_path = data_root / "croissant" / "gbif_ebbe_2026_croissant.jsonld"
    camtrap_path = data_root / "camtrapdp" / "datapackage.json"
    croissant = json.loads(croissant_path.read_text(encoding="utf-8"))
    camtrap = json.loads(camtrap_path.read_text(encoding="utf-8"))

    if "Dataset" not in str(croissant.get("@type", "")):
        return fail("Croissant JSON-LD does not declare a Dataset type")
    if not croissant.get("distribution"):
        return fail("Croissant JSON-LD has no distributions")
    if camtrap.get("profile") != "camtrap-dp":
        return fail("CamtrapDP datapackage profile is not camtrap-dp")
    resource_names = {resource.get("name") for resource in camtrap.get("resources", [])}
    if not {"deployments", "media", "observations"}.issubset(resource_names):
        return fail("CamtrapDP resources must include deployments, media and observations")

    deployments = read_csv(data_root / "camtrapdp" / "deployments.csv")
    camtrap_media = read_csv(data_root / "camtrapdp" / "media.csv")
    observations = read_csv(data_root / "camtrapdp" / "observations.csv")
    if len(answer_key) != len(events):
        return fail("ground truth row count does not match model input events")
    if len(observations) != len(answer_key):
        return fail("CamtrapDP observations row count does not match answer key")
    if len(camtrap_media) != len(media_hashes):
        return fail("CamtrapDP media row count does not match image hashes")
    if not deployments:
        return fail("CamtrapDP deployments table is empty")

    print(
        "standards metadata check passed; "
        f"data_root={data_root}; events={len(events)}, media_rows={len(media_hashes)}, deployments={len(deployments)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
