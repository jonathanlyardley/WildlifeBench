#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVENT_SET_ID = "gbif_award_speciesnet_overlap_20260626"
FILTER_SCOPE = "speciesnet_v403b_geo_supported_overlap"
SANITISED_SOURCE = "not_redistributed_public_staging_speciesnet_overlap_only"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=False) + "\n")


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_file_paths(root: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and path.name != "checksums.sha256" and "__pycache__" not in path.parts
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )


def repo_file_paths(root: Path) -> list[Path]:
    excluded_dirs = {".git", "__pycache__"}
    paths: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.name == "checksums.sha256":
            continue
        if excluded_dirs.intersection(path.relative_to(root).parts):
            continue
        paths.append(path)
    return sorted(paths, key=lambda path: path.relative_to(root).as_posix())


def write_checksums(root: Path, paths: list[Path]) -> None:
    lines = []
    for path in paths:
        rel = path.relative_to(root).as_posix()
        lines.append(f"{sha256_file(path)}  {rel}")
    (root / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_release_rows(source: Path) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    events = read_jsonl(source / "model_inputs" / "events.jsonl")
    answers = read_jsonl(source / "ground_truth" / "answer_key.jsonl")
    image_hashes = read_csv(source / "image_hashes.csv")
    attribution = read_jsonl(source / "attribution.jsonl")
    return events, answers, image_hashes, attribution


def included_answer_rows(answers: list[dict]) -> list[dict]:
    included = [row for row in answers if (row.get("speciesnet_overlap") or {}).get("included") is True]
    if not included:
        raise RuntimeError("No SpeciesNet-overlap rows found.")
    for row in included:
        overlap = row.get("speciesnet_overlap") or {}
        required = ("on_speciesnet_taxonomy", "on_speciesnet_classifier_label", "geofence_allowed")
        missing = [key for key in required if overlap.get(key) is not True]
        if missing:
            raise RuntimeError(f"Included row {row.get('event_id')} is missing required SpeciesNet flags: {missing}")
    return included


def normalise_events(events: list[dict], included_ids: set[str]) -> list[dict]:
    filtered = []
    for event in events:
        if event["event_id"] not in included_ids:
            continue
        event = dict(event)
        event["event_set_id"] = EVENT_SET_ID
        event["speciesnet_overlap"] = {
            "included": True,
            "reason": "included_speciesnet_classifier_label_and_country_geofence",
        }
        filtered.append(event)
    return filtered


def normalise_answers(answers: list[dict], included_ids: set[str]) -> list[dict]:
    filtered = []
    for row in answers:
        if row["event_id"] not in included_ids:
            continue
        row = dict(row)
        row["event_set_id"] = EVENT_SET_ID
        filtered.append(row)
    return filtered


def filter_source_datasets(source: Path, answers: list[dict], attribution: list[dict]) -> list[dict]:
    rows = read_csv(source / "source_datasets.csv")
    event_to_dataset = {
        row["event_id"]: (row.get("source") or {}).get("dataset_key")
        for row in answers
        if (row.get("source") or {}).get("dataset_key")
    }
    event_to_licence: dict[str, str] = {}
    for row in attribution:
        event_to_licence.setdefault(row["event_id"], row.get("licence_url") or row.get("licence") or "")

    dataset_events: dict[str, set[str]] = {}
    dataset_licences: dict[str, Counter] = {}
    for event_id, dataset_key in event_to_dataset.items():
        dataset_events.setdefault(dataset_key, set()).add(event_id)
        dataset_licences.setdefault(dataset_key, Counter())[event_to_licence.get(event_id, "")] += 1

    filtered = []
    for row in rows:
        dataset_key = row["dataset_key"]
        if dataset_key not in dataset_events:
            continue
        row = dict(row)
        row["event_count"] = str(len(dataset_events[dataset_key]))
        row["licence_counts"] = json.dumps(dict(dataset_licences[dataset_key]), ensure_ascii=False, sort_keys=True)
        filtered.append(row)
    return filtered


def write_data_tree(target_data: Path, source: Path, copy_media: bool, media_rows: list[dict]) -> None:
    target_data.mkdir(parents=True, exist_ok=True)
    for subdir in ("model_inputs", "ground_truth", "camtrapdp", "croissant"):
        (target_data / subdir).mkdir(parents=True, exist_ok=True)
    if copy_media:
        for row in media_rows:
            if row.get("file_present") != "true":
                continue
            rel = Path(row["relative_path"])
            src = source / rel
            dst = target_data / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def write_camtrapdp(target_data: Path, source: Path, event_ids: set[str], media_ids: set[str]) -> None:
    camtrap_source = source / "camtrapdp"
    media_rows = [row for row in read_csv(camtrap_source / "media.csv") if row["mediaID"] in media_ids]
    observation_rows = [row for row in read_csv(camtrap_source / "observations.csv") if row["eventID"] in event_ids]
    deployment_ids = {row["deploymentID"] for row in media_rows} | {row["deploymentID"] for row in observation_rows}
    deployment_rows = [row for row in read_csv(camtrap_source / "deployments.csv") if row["deploymentID"] in deployment_ids]
    write_csv(target_data / "camtrapdp" / "media.csv", media_rows)
    write_csv(target_data / "camtrapdp" / "observations.csv", observation_rows)
    write_csv(target_data / "camtrapdp" / "deployments.csv", deployment_rows)

    datapackage = json.loads((camtrap_source / "datapackage.json").read_text(encoding="utf-8"))
    datapackage["name"] = "wildlifebench-gbif-ebbe-2026-speciesnet-overlap"
    datapackage["id"] = EVENT_SET_ID
    datapackage["title"] = "WildlifeBench GBIF Ebbe Nielsen 2026 SpeciesNet-overlap camera-trap package"
    datapackage["description"] = (
        "Camera-trap event package filtered to the final SpeciesNet v4.0.3b classifier-label "
        "and country-geofence overlap used for award-facing comparison. Coordinates are omitted; "
        "media are EXIF-stripped derivatives."
    )
    (target_data / "camtrapdp" / "datapackage.json").write_text(
        json.dumps(datapackage, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def write_croissant(target_data: Path, source: Path, event_count: int, species_count: int, media_count: int) -> None:
    croissant_path = source / "croissant" / "gbif_ebbe_2026_croissant.jsonld"
    croissant = json.loads(croissant_path.read_text(encoding="utf-8"))
    croissant["name"] = "wildlifebench-gbif-ebbe-2026-speciesnet-overlap"
    croissant["description"] = (
        "WildlifeBench GBIF Ebbe Nielsen 2026 camera-trap package filtered to the final "
        "SpeciesNet v4.0.3b classifier-label and country-geofence overlap, with neutral "
        "model input IDs, isolated ground truth, CamtrapDP companion tables and per-media attribution."
    )
    croissant["version"] = EVENT_SET_ID
    croissant["url"] = "https://github.com/jonathanlyardley/WildlifeBench"
    croissant["releaseSummary"] = {
        "included_events": event_count,
        "included_species": species_count,
        "media_rows": media_count,
        "headline_scope": FILTER_SCOPE,
    }
    (target_data / "croissant" / "gbif_ebbe_2026_croissant.jsonld").write_text(
        json.dumps(croissant, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def write_manifest(target: Path, events: list[dict], answers: list[dict], media_rows: list[dict]) -> None:
    species = {row["taxonomy"]["species"] for row in answers}
    bird_species = {row["taxonomy"]["species"] for row in answers if row["taxonomy"].get("class") == "Aves"}
    manifest = {
        "schema_version": "wildlifebench.gbif_ebbe_2026_release_manifest.v1",
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "event_set_id": EVENT_SET_ID,
        "public_prefix": "WBGBIF2026",
        "release_layout": {"github_repo": "github_repo", "data_package": "data_package"},
        "counts": {
            "source_events": len(events),
            "source_species": len(species),
            "speciesnet_overlap_events": len(events),
            "speciesnet_overlap_species": len(species),
            "excluded_events": 0,
            "media_rows": len(media_rows),
            "copied_media_files": sum(1 for row in media_rows if row.get("file_present") == "true"),
            "copied_result_files": 9,
            "included_bird_species": len(bird_species),
        },
        "media_mode": "speciesnet_overlap_cropped_and_uncropped",
        "include_uncropped_media": True,
        "standards": {
            "croissant": "data/croissant/gbif_ebbe_2026_croissant.jsonld",
            "camtrapdp": "data/camtrapdp/datapackage.json",
        },
        "links": {
            "github": "https://github.com/jonathanlyardley/WildlifeBench",
            "huggingface_dataset": "https://huggingface.co/datasets/jonathanlyardley/wildlifebench-gbif-ebbe-2026-data",
            "staging_status": "private staging - to be made public after final release checks",
        },
        "notes": [
            "Public model input IDs and filenames are neutral.",
            "Ground truth is isolated from model_inputs.",
            "This release copy contains only SpeciesNet v4.0.3b classifier-label and country-geofence overlap events.",
            "Excluded non-overlap events from the 330-event development snapshot are not redistributed in this staging package.",
        ],
    }
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_readmes(target_data: Path, event_count: int, species_count: int, media_count: int) -> None:
    readme = f"""# WildlifeBench GBIF Ebbe Nielsen 2026 SpeciesNet-Overlap Data Package

This is the private-staging upload package for Hugging Face Dataset and archive-style review.

- Staging status: private staging - to be made public after final release checks.
- Scope: final SpeciesNet v4.0.3b classifier-label and country-geofence overlap used for award-facing comparison.
- Included bursts/events: {event_count}.
- Included species: {species_count}.
- Copied media files: {media_count} cropped/uncropped JPEG derivatives.
- The model-facing events use neutral IDs and do not contain ground-truth species labels.
- The isolated answer key under `ground_truth/` is retained for no-cost review scoring.
- Media are EXIF-stripped and retain per-media GBIF attribution, rights-holder, licence URL and checksum metadata.
"""
    target_data.joinpath("README.md").write_text(readme, encoding="utf-8")
    notice = """# Private Data Notice

Do not upload private source mappings, local filesystem paths, `.env` files, raw provider logs or workspace state.

This staging copy contains only the SpeciesNet-overlap events used for the final GBIF award-facing dataset. Source event IDs and taxonomy labels are confined to the answer-key/audit layer, not model-facing inputs.
Publication-ready copy note: private source-package paths were removed from attribution metadata on 2026-06-26 before public upload.
"""
    target_data.joinpath("PRIVATE_DATA_NOTICE.md").write_text(notice, encoding="utf-8")


def update_results(repo_root: Path, event_ids: set[str]) -> None:
    results_dir = repo_root / "results" / "speciesnet_overlap"
    excluded_files = [
        results_dir / "application_safe_speciesnet_overlap_excluded_events.csv",
        results_dir / "application_safe_speciesnet_overlap_excluded_species.csv",
    ]
    for path in excluded_files:
        if path.exists():
            path.unlink()

    results_csv = results_dir / "application_safe_results.csv"
    rows = read_csv(results_csv)
    for row in rows:
        if "source_event_count_full" in row:
            row["source_event_count_full"] = str(len(event_ids))
        row["excluded_event_count"] = "0"
        row["excluded_species_count"] = "0"
        row["source_result_paths"] = SANITISED_SOURCE
        note = row.get("notes") or ""
        suffix = " Raw full-run outputs are not redistributed in this SpeciesNet-overlap-only staging package."
        if suffix.strip() not in note:
            row["notes"] = (note + suffix).strip()
    write_csv(results_csv, rows)

    event_csv = results_dir / "application_safe_results_events.csv"
    event_rows = [row for row in read_csv(event_csv) if row["event_id"] in event_ids]
    for row in event_rows:
        row["source_result_path"] = SANITISED_SOURCE
    write_csv(event_csv, event_rows)

    manifest_path = results_dir / "application_safe_results_manifest_public.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_event_count"] = len(event_ids)
    manifest["source_species_count"] = 82
    manifest["included_event_count"] = len(event_ids)
    manifest["included_species_count"] = 82
    manifest["excluded_event_count"] = 0
    manifest["excluded_species_count"] = 0
    manifest["exclusion_reason_counts"] = {}
    manifest["source_records"] = []
    manifest["source_records_note"] = (
        "Raw full-run output records are not redistributed in this SpeciesNet-overlap-only public staging package; "
        "aggregate and event-level SpeciesNet-overlap result rows are retained."
    )
    outputs = manifest.get("outputs") or {}
    outputs.pop("excluded_events_csv", None)
    outputs.pop("excluded_species_csv", None)
    manifest["outputs"] = outputs
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build(source: Path, hf_data_package: Path, repo_root: Path) -> None:
    events, answers, image_hashes, attribution = load_release_rows(source)
    included_answers = included_answer_rows(answers)
    included_ids = {row["event_id"] for row in included_answers}
    filtered_events = normalise_events(events, included_ids)
    filtered_answers = normalise_answers(answers, included_ids)
    media_rows = [row for row in image_hashes if row["event_id"] in included_ids]
    attr_rows = [row for row in attribution if row["event_id"] in included_ids]
    media_ids = {row["media_id"] for row in media_rows}
    source_dataset_rows = filter_source_datasets(source, filtered_answers, attr_rows)
    species_count = len({row["taxonomy"]["species"] for row in filtered_answers})

    if len(filtered_events) != 246 or species_count != 82:
        raise RuntimeError(f"Unexpected SpeciesNet-overlap size: events={len(filtered_events)}, species={species_count}")
    if any((row.get("speciesnet_overlap") or {}).get("included") is not True for row in filtered_answers):
        raise RuntimeError("Filtered answer key still contains a non-included event.")

    write_data_tree(hf_data_package, source, copy_media=True, media_rows=media_rows)
    write_data_tree(repo_root / "data", source, copy_media=False, media_rows=media_rows)

    for target_data in (hf_data_package, repo_root / "data"):
        write_jsonl(target_data / "model_inputs" / "events.jsonl", filtered_events)
        write_jsonl(target_data / "ground_truth" / "answer_key.jsonl", filtered_answers)
        write_jsonl(target_data / "attribution.jsonl", attr_rows)
        write_csv(target_data / "image_hashes.csv", media_rows, fieldnames=list(image_hashes[0].keys()))
        write_csv(target_data / "source_datasets.csv", source_dataset_rows)
        write_camtrapdp(target_data, source, included_ids, media_ids)
        write_croissant(target_data, source, len(filtered_events), species_count, len(media_rows))
        write_manifest(target_data / "release_manifest.json", filtered_events, filtered_answers, media_rows)
        write_readmes(target_data, len(filtered_events), species_count, len(media_rows))

    shutil.copy2(hf_data_package / "release_manifest.json", repo_root / "release" / "release_manifest.json")
    update_results(repo_root, included_ids)

    subprocess.run([sys.executable, str(repo_root / "scripts" / "score_cached_outputs.py")], cwd=repo_root, check=True)
    subprocess.run([sys.executable, str(repo_root / "scripts" / "build_submission_table.py")], cwd=repo_root, check=True)

    write_checksums(hf_data_package, package_file_paths(hf_data_package))
    write_checksums(repo_root, repo_file_paths(repo_root))

    print(f"wrote SpeciesNet-overlap repo metadata: {repo_root}")
    print(f"wrote SpeciesNet-overlap HF data package: {hf_data_package}")
    print(f"events={len(filtered_events)} species={species_count} media_files={len(media_rows)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Restrict the GBIF Ebbe release to the SpeciesNet-overlap award slice.")
    parser.add_argument("--source-data-package", required=True, type=Path)
    parser.add_argument("--hf-data-package", required=True, type=Path)
    parser.add_argument("--repo-root", default=ROOT, type=Path)
    args = parser.parse_args()
    source = args.source_data_package.expanduser().resolve()
    hf_data_package = args.hf_data_package.expanduser().resolve()
    repo_root = args.repo_root.expanduser().resolve()
    if hf_data_package.exists():
        raise RuntimeError(f"HF data-package output already exists: {hf_data_package}")
    build(source, hf_data_package, repo_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
