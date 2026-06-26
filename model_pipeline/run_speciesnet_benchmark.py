#!/usr/bin/env python3
"""
SpeciesNet camera-trap benchmark runner.

For the GBIF award prototype-330 work, pass an explicit flat runner package:

    uv run python run_speciesnet_benchmark.py ^
        --image_dir track_b/data/gbif_award/prototype330_uncropped_speciesnet_runner ^
        --runs geo_ensemble ^
        --model kaggle:google/speciesnet/pyTorch/v4.0.3b/1 ^
        --run_tag v403b_uncropped_proto330

Current useful upstream model examples:
    v4.0.3a default/always-crop:
        kaggle:google/speciesnet/pyTorch/v4.0.3a/1
    v4.0.3b full-image:
        kaggle:google/speciesnet/pyTorch/v4.0.3b/1

Use --dry_run for configuration checks. Dry-runs do not import or execute
SpeciesNet inference.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from run_manifest import utc_now_iso, write_speciesnet_run_manifest


IMAGE_DIR = Path("track_b/data/prototype_original")
RESULTS_DIR = Path("track_b/results/speciesnet")
TEMP_DIR = Path("track_b/results/speciesnet/_tmp")

SPECIESNET_PACKAGE_EXAMPLE_VERSION = "5.0.5"
SPECIESNET_MODEL_EXAMPLES = {
    "v4.0.3a": {
        "model_id": "kaggle:google/speciesnet/pyTorch/v4.0.3a/1",
        "role": "default always-crop model",
        "recommended_input": "cropped or standard SpeciesNet default route",
    },
    "v4.0.3b": {
        "model_id": "kaggle:google/speciesnet/pyTorch/v4.0.3b/1",
        "role": "full-image model",
        "recommended_input": "uncropped/full-frame camera-trap images",
    },
}

COUNTRY_MAP = {
    "Ecuador": "ECU",
    "Belgium": "BEL",
    "Luxembourg": "LUX",
    "France": "FRA",
    "Germany": "DEU",
}

RUNS_CONFIG = [
    {"label": "blind_ensemble", "geo": False, "no_ensemble": False},
    {"label": "geo_ensemble", "geo": True, "no_ensemble": False},
    {"label": "blind_classifier", "geo": False, "no_ensemble": True},
    {"label": "geo_classifier", "geo": True, "no_ensemble": True},
]


def safe_label(value: str | None) -> str | None:
    """Return a filename-safe label, or None when the label is blank."""
    if value is None:
        return None
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("._-")
    return cleaned or None


def speciesnet_python_executable() -> str:
    """Interpreter used for SpeciesNet CLI calls."""
    return os.environ.get("SPECIESNET_PYTHON", sys.executable)


def speciesnet_runtime_record() -> dict[str, Any]:
    """Record the external Python runtime that executes SpeciesNet inference."""
    executable = speciesnet_python_executable()
    code = (
        "import json, sys, platform\n"
        "from importlib import metadata as importlib_metadata\n"
        "try:\n"
        "    speciesnet_version = importlib_metadata.version('speciesnet')\n"
        "except importlib_metadata.PackageNotFoundError:\n"
        "    speciesnet_version = None\n"
        "print(json.dumps({\n"
        "    'python_executable': sys.executable,\n"
        "    'python_version': sys.version.split()[0],\n"
        "    'platform': platform.platform(),\n"
        "    'speciesnet_version': speciesnet_version,\n"
        "}, sort_keys=True))\n"
    )
    try:
        result = subprocess.run(
            [executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except Exception as exc:
        return {
            "python_executable": executable,
            "query_error": str(exc),
        }

    if result.returncode != 0:
        return {
            "python_executable": executable,
            "query_error": (result.stderr or result.stdout).strip(),
            "returncode": result.returncode,
        }

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {
            "python_executable": executable,
            "query_error": "Could not parse runtime query output as JSON.",
            "stdout": result.stdout.strip(),
        }
    payload["query_method"] = "external_python_importlib_metadata"
    return payload


def output_stem(label: str, timestamp: str, run_tag: str | None = None) -> str:
    parts = ["speciesnet"]
    if run_tag:
        parts.append(run_tag)
    parts.extend([safe_label(label) or label, timestamp])
    return "_".join(parts)


def summary_name(timestamp: str, run_tag: str | None = None) -> str:
    parts = ["speciesnet", "summary"]
    if run_tag:
        parts.append(run_tag)
    parts.append(timestamp)
    return "_".join(parts) + ".txt"


def raw_output_prefix(label: str, timestamp: str, run_tag: str | None = None) -> str:
    return output_stem(label, timestamp, run_tag)


def model_policy(model_id: str | None) -> dict[str, Any]:
    """Describe the selected SpeciesNet model policy for results/manifests."""
    selected = model_id or "installed SpeciesNet default"
    detected_version = None
    if model_id:
        match = re.search(r"/(v\d+\.\d+\.\d+[a-z]?)/", model_id)
        detected_version = match.group(1) if match else None

    if detected_version in SPECIESNET_MODEL_EXAMPLES:
        example = SPECIESNET_MODEL_EXAMPLES[detected_version]
        variant_role = example["role"]
        recommended_input = example["recommended_input"]
        upstream_note = "Pinned explicit SpeciesNet model ID."
    elif model_id:
        variant_role = "explicit model ID, unrecognised local role"
        recommended_input = "check upstream SpeciesNet model documentation"
        upstream_note = "Model ID is explicit but not one of the current runner examples."
    else:
        variant_role = "installed package default"
        recommended_input = "avoid relying on this for award rows; pass --model explicitly"
        upstream_note = "The installed SpeciesNet package decides the default model."

    return {
        "selected_model_id": selected,
        "detected_model_version": detected_version,
        "variant_role": variant_role,
        "recommended_input": recommended_input,
        "package_version_to_prepare": SPECIESNET_PACKAGE_EXAMPLE_VERSION,
        "examples": SPECIESNET_MODEL_EXAMPLES,
        "upstream_note": upstream_note,
    }


def image_package_policy(image_dir: Path) -> dict[str, Any]:
    """Describe the flat image package being scored."""
    manifest_path = image_dir / "manifest.json"
    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as f:
            manifest = json.load(f)

    path_text = image_dir.as_posix()
    if "prototype330_uncropped_speciesnet_runner" in path_text:
        input_policy = (
            "uncropped/full-frame EXIF-stripped SpeciesNet runner package; "
            "private provenance stores source paths and source/export hashes"
        )
    elif "prototype330_cropped_runner" in path_text:
        input_policy = (
            "timestamp-cropped EXIF-stripped same-image runner package; "
            "closest image-policy match to VLM prototype rows"
        )
    elif image_dir == IMAGE_DIR:
        input_policy = "historical default flat package; not the GBIF award prototype-330 input"
    else:
        input_policy = "custom flat SpeciesNet runner package"

    return {
        "image_dir": str(image_dir),
        "input_policy": input_policy,
        "manifest_path": str(manifest_path) if manifest_path.exists() else None,
        "event_set_id": manifest.get("event_set_id"),
        "event_count": manifest.get("event_count"),
        "species_count": manifest.get("species_count"),
        "frame_count": manifest.get("frame_count"),
        "country_counts": manifest.get("country_counts"),
        "manifest_image_policy": manifest.get("image_policy"),
    }


def build_country_policy(
    events: list[dict[str, Any]],
    *,
    geo: bool,
    geo_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Describe how country hints will be passed to SpeciesNet."""
    original_counts = Counter(e.get("country_code") or "NONE" for e in events)
    effective_counts: Counter[str] = Counter()
    for event in events:
        country_code = event.get("country_code") or ""
        if geo_overrides and event["event_id"] in geo_overrides:
            country_code = geo_overrides[event["event_id"]]
        effective_counts[country_code or "NONE"] += 1

    return {
        "country_code_standard": "ISO 3166-1 alpha-3",
        "runner_country_map": COUNTRY_MAP,
        "country_hint_sent": bool(geo),
        "raw_metadata_country_code_counts": dict(sorted(original_counts.items())),
        "effective_country_code_counts": dict(sorted(effective_counts.items())),
        "geo_overrides": geo_overrides or {},
        "note": (
            "For geo_* runs the runner groups frames by country and passes "
            "--country to SpeciesNet. The country hint affects the ensemble/"
            "geofencing final prediction policy, not the underlying image bytes."
        ),
    }


def speciesnet_supports_classifier_only() -> bool:
    """Check whether the selected SpeciesNet environment exposes --classifier_only."""
    try:
        result = subprocess.run(
            [speciesnet_python_executable(), "-m", "speciesnet.scripts.run_model", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        return "--classifier_only" in result.stdout or "--classifier_only" in result.stderr
    except Exception:
        return False


def load_events(image_dir: Path) -> list[dict[str, Any]]:
    """Load events from a flat runner package of *_meta.json and *_frame*.jpg files."""
    events: list[dict[str, Any]] = []
    for meta_path in sorted(image_dir.glob("*_meta.json")):
        with meta_path.open(encoding="utf-8") as f:
            meta = json.load(f)

        stem = meta_path.stem.replace("_meta", "")
        frames = sorted(image_dir.glob(f"{stem}_frame*.jpg"))
        if not frames:
            frames = sorted(image_dir.glob(f"{stem}*.jpg"))
        if not frames:
            print(f"  WARNING: No frames found for {stem}")
            continue

        country = meta.get("country", "")
        country_code = COUNTRY_MAP.get(country, "")
        if not country_code and country:
            print(f"  WARNING: Unknown country '{country}' for {stem}; no geo hint will be applied")

        events.append(
            {
                "event_id": stem,
                "species": meta.get("species", ""),
                "genus": meta.get("genus", ""),
                "family": meta.get("family", ""),
                "order": meta.get("order", ""),
                "class": meta.get("class", ""),
                "country": country,
                "country_code": country_code,
                "difficulty": meta.get("_review", {}).get("difficulty", ""),
                "time_of_day": meta.get("_review", {}).get("time_of_day", ""),
                "frames": [str(f.resolve()) for f in frames],
            }
        )

    print(f"Loaded {len(events)} events, {sum(len(e['frames']) for e in events)} frames total")
    return events


def apply_event_limit(events: list[dict[str, Any]], limit_events: int | None) -> list[dict[str, Any]]:
    if limit_events is None:
        return events
    if limit_events <= 0:
        raise ValueError("--limit_events must be a positive integer")
    limited = sorted(events, key=lambda event: event["event_id"])[:limit_events]
    print(f"Event limit applied: {len(limited)} of {len(events)} events")
    return limited


def run_speciesnet_on_paths(
    frame_paths: list[str],
    country: str | None,
    output_json: Path,
    no_ensemble: bool = False,
    model_id: str | None = None,
) -> dict[str, Any]:
    """
    Call SpeciesNet CLI on a specific list of image paths.

    model_id examples:
      - kaggle:google/speciesnet/pyTorch/v4.0.3a/1
      - kaggle:google/speciesnet/pyTorch/v4.0.3b/1
    """
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    fp_label = country if country else "blind"
    filepaths_txt = TEMP_DIR / f"filepaths_{fp_label}.txt"
    filepaths_txt.write_text("\n".join(frame_paths), encoding="utf-8")

    cmd = [
        speciesnet_python_executable(),
        "-m",
        "speciesnet.scripts.run_model",
        "--filepaths_txt",
        str(filepaths_txt),
        "--predictions_json",
        str(output_json),
        "--noprogress_bars",
    ]
    if model_id:
        cmd += ["--model", model_id]
    if country:
        cmd += ["--country", country]
    if no_ensemble:
        cmd += ["--classifier_only"]

    mode_str = f"country={country or 'none'}, ensemble={'off' if no_ensemble else 'on'}"
    print(f"    Running SpeciesNet ({mode_str}) on {len(frame_paths)} frames...")

    start = time.time()
    result = subprocess.run(cmd, text=True, check=False)
    elapsed = round(time.time() - start, 1)

    if result.returncode != 0:
        raise RuntimeError(f"SpeciesNet exited with code {result.returncode}")

    print(f"    Done in {elapsed}s")

    if not output_json.exists():
        raise RuntimeError(f"Output file not created: {output_json}")

    with output_json.open(encoding="utf-8") as f:
        return json.load(f)


def parse_raw_predictions(raw: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """
    Parse SpeciesNet output JSON into {filepath_stem: parsed_prediction}.

    Expected ensemble prediction string:
    uuid;class;order;family;genus;epithet;common_name
    """
    lookup: dict[str, dict[str, Any]] = {}
    for entry in raw.get("predictions", []):
        filepath = entry.get("filepath", "")
        stem = Path(filepath.replace("\\", "/")).stem

        pred_str = entry.get("prediction", "")
        score = entry.get("prediction_score", None)
        source = entry.get("prediction_source", "")
        model_ver = entry.get("model_version", "")

        parts = [p.strip() for p in pred_str.split(";")][1:]

        def gp(index: int) -> str | None:
            if index >= len(parts):
                return None
            value = parts[index].strip()
            return None if value.lower() in {"", "no cv result", "blank"} else value

        pred_class = gp(0)
        pred_order = gp(1)
        pred_family = gp(2)
        pred_genus = gp(3)
        pred_epithet = gp(4)
        pred_common = gp(5)

        is_blank = "blank" in pred_str.lower() and not pred_genus
        is_no_result = "no cv result" in pred_str.lower()

        species = None
        if pred_genus and pred_epithet:
            species = pred_genus.capitalize() + " " + pred_epithet.lower()

        if is_blank:
            outcome = "no_animal"
        elif is_no_result:
            outcome = "uncertain"
        elif not pred_genus:
            outcome = "class_only"
        elif not pred_epithet:
            outcome = "genus_only"
        else:
            outcome = "species"

        lookup[stem] = {
            "species": species,
            "genus": pred_genus.capitalize() if pred_genus else None,
            "family": pred_family.capitalize() if pred_family else None,
            "order": pred_order.capitalize() if pred_order else None,
            "class": pred_class.capitalize() if pred_class else None,
            "common_name": pred_common,
            "outcome": outcome,
            "score": round(score, 4) if score is not None else None,
            "source": source,
            "model_version": model_ver,
            "raw_label": pred_str,
        }
    return lookup


def aggregate_burst(frame_preds: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-frame predictions into event-level best-frame and majority-vote outputs."""
    valid_species = [p for p in frame_preds if p.get("species")]

    if not valid_species:
        best_class = max(
            [p for p in frame_preds if p.get("class")] or [{}],
            key=lambda p: p.get("score") or 0,
            default={},
        )
        return {
            "best_frame": best_class,
            "majority_vote": best_class,
            "n_frames_with_species": 0,
            "n_frames_total": len(frame_preds),
            "frame_predictions": frame_preds,
        }

    best = max(valid_species, key=lambda p: p.get("score") or 0)
    species_counts = Counter(p["species"] for p in valid_species)
    top_species = species_counts.most_common(1)[0][0]
    top_species_frames = [p for p in valid_species if p["species"] == top_species]
    majority = dict(max(top_species_frames, key=lambda p: p.get("score") or 0))
    majority["vote_count"] = species_counts[top_species]
    majority["vote_fraction"] = round(species_counts[top_species] / len(valid_species), 2)

    return {
        "best_frame": best,
        "majority_vote": majority,
        "n_frames_with_species": len(valid_species),
        "n_frames_total": len(frame_preds),
        "frame_predictions": frame_preds,
    }


def compute_tis(prediction: dict[str, Any], ground_truth: dict[str, Any]):
    """Taxonomic Identification Score on the same 0 to 5 scale used in the LLM pipeline."""

    def match(a: str | None, b: str | None) -> bool:
        if not a or not b:
            return False
        return a.strip().lower() == b.strip().lower()

    if match(prediction.get("species"), ground_truth.get("species")):
        return 5.0, "species"
    if match(prediction.get("genus"), ground_truth.get("genus")):
        return 3.0, "genus"
    if match(prediction.get("family"), ground_truth.get("family")):
        return 2.0, "family"
    if match(prediction.get("order"), ground_truth.get("order")):
        return 1.0, "order"
    if match(prediction.get("class"), ground_truth.get("class")):
        return 0.5, "class"
    return 0.0, "none"


def execute_run(
    run_cfg: dict[str, Any],
    events: list[dict[str, Any]],
    timestamp: str,
    *,
    model_id: str | None = None,
    run_tag: str | None = None,
    limit_events: int | None = None,
    geo_overrides: dict[str, str] | None = None,
    model_policy_payload: dict[str, Any] | None = None,
    image_policy_payload: dict[str, Any] | None = None,
    speciesnet_runtime_payload: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Run one SpeciesNet configuration and return scored results."""
    label = run_cfg["label"]
    geo = run_cfg["geo"]
    no_ensemble = run_cfg["no_ensemble"]

    print(f"\n{'=' * 60}")
    print(f"  RUN: {label.upper()}")
    print(f"{'=' * 60}")

    all_predictions: dict[str, dict[str, Any]] = {}

    try:
        if geo:

            def effective_country(event: dict[str, Any]) -> str:
                if geo_overrides and event["event_id"] in geo_overrides:
                    return geo_overrides[event["event_id"]]
                return event["country_code"]

            grouped: dict[str, list[dict[str, Any]]] = {}
            for event in events:
                grouped.setdefault(effective_country(event), []).append(event)

            for country_code, country_events in grouped.items():
                cc_label = country_code or "blind_fallback"
                sn_country = country_code or None
                frame_paths = [frame for event in country_events for frame in event["frames"]]
                out_json = TEMP_DIR / f"{raw_output_prefix(label, timestamp, run_tag)}_{cc_label}_raw.json"
                raw = run_speciesnet_on_paths(frame_paths, sn_country, out_json, no_ensemble, model_id)
                all_predictions.update(parse_raw_predictions(raw))
        else:
            all_frame_paths = [frame for event in events for frame in event["frames"]]
            out_json = TEMP_DIR / f"{raw_output_prefix(label, timestamp, run_tag)}_all_raw.json"
            raw = run_speciesnet_on_paths(all_frame_paths, None, out_json, no_ensemble, model_id)
            all_predictions = parse_raw_predictions(raw)

    except RuntimeError as exc:
        msg = str(exc).lower()
        if no_ensemble and ("unrecognized" in msg or "classifier_only" in msg or "error" in msg):
            print("  SKIPPED: --classifier_only not supported by this SpeciesNet version.")
            print("  Install a newer version or remove classifier-only runs.")
            return None
        print(f"  ERROR: {exc}")
        return None

    scored_events: list[dict[str, Any]] = []
    tis_best_list: list[float] = []
    tis_vote_list: list[float] = []

    for event in events:
        frame_preds = []
        for frame_path in event["frames"]:
            stem = Path(frame_path).stem
            frame_preds.append(
                all_predictions.get(
                    stem,
                    {"outcome": "missing", "species": None, "score": None},
                )
            )

        burst = aggregate_burst(frame_preds)
        gt = {key: event[key] for key in ("species", "genus", "family", "order", "class")}

        tis_best, lvl_best = compute_tis(burst["best_frame"], gt)
        tis_vote, lvl_vote = compute_tis(burst["majority_vote"], gt)

        tis_best_list.append(tis_best)
        tis_vote_list.append(tis_vote)

        scored_events.append(
            {
                "event_id": event["event_id"],
                "ground_truth": gt,
                "country": event["country"],
                "country_code": event["country_code"],
                "difficulty": event["difficulty"],
                "time_of_day": event["time_of_day"],
                "n_frames": len(event["frames"]),
                "n_frames_with_species": burst["n_frames_with_species"],
                "best_frame_prediction": burst["best_frame"],
                "majority_vote_prediction": burst["majority_vote"],
                "frame_predictions": burst.get("frame_predictions", []),
                "tis_best_frame": tis_best,
                "tis_best_frame_level": lvl_best,
                "tis_majority_vote": tis_vote,
                "tis_majority_vote_level": lvl_vote,
            }
        )

    n = len(scored_events)

    def pct_at(scores: list[float], value: float) -> float:
        return round(100 * scores.count(value) / n, 1) if n else 0

    summary = {
        "n_events": n,
        "mean_tis_best_frame": round(sum(tis_best_list) / n, 3) if n else 0,
        "mean_tis_majority_vote": round(sum(tis_vote_list) / n, 3) if n else 0,
        "best_frame": {
            "species_pct": pct_at(tis_best_list, 5.0),
            "genus_pct": pct_at(tis_best_list, 3.0),
            "family_pct": pct_at(tis_best_list, 2.0),
            "order_pct": pct_at(tis_best_list, 1.0),
            "class_pct": pct_at(tis_best_list, 0.5),
            "none_pct": pct_at(tis_best_list, 0.0),
        },
        "majority_vote": {
            "species_pct": pct_at(tis_vote_list, 5.0),
            "genus_pct": pct_at(tis_vote_list, 3.0),
            "family_pct": pct_at(tis_vote_list, 2.0),
            "order_pct": pct_at(tis_vote_list, 1.0),
            "class_pct": pct_at(tis_vote_list, 0.5),
            "none_pct": pct_at(tis_vote_list, 0.0),
        },
    }

    print(f"\n  [{label}] Results:")
    print(f"    Mean TIS (best frame):    {summary['mean_tis_best_frame']:.3f}")
    print(f"    Mean TIS (majority vote): {summary['mean_tis_majority_vote']:.3f}")
    print(
        f"    Species %: {summary['best_frame']['species_pct']}% (best frame)  "
        f"{summary['majority_vote']['species_pct']}% (majority vote)"
    )

    return {
        "run_label": label,
        "run_tag": run_tag,
        "run_date": timestamp,
        "mode": {"geo": geo, "ensemble": not no_ensemble},
        "image_source": str(run_cfg.get("image_dir") or IMAGE_DIR),
        "model_id": model_id,
        "event_limit": limit_events,
        "speciesnet_model_policy": model_policy_payload or model_policy(model_id),
        "speciesnet_runtime": speciesnet_runtime_payload,
        "image_package_policy": image_policy_payload,
        "country_policy": build_country_policy(events, geo=geo, geo_overrides=geo_overrides),
        "summary": summary,
        "events": scored_events,
    }


def print_and_save_summary(
    all_results: dict[str, dict[str, Any] | None],
    timestamp: str,
    *,
    image_dir: Path | None = None,
    run_tag: str | None = None,
    model_policy_payload: dict[str, Any] | None = None,
) -> Path:
    img_src = image_dir or IMAGE_DIR
    lines = [
        f"SpeciesNet Camera Trap Benchmark - {timestamp}",
        f"Run tag      : {run_tag or 'none'}",
        f"Image source : {img_src}",
    ]
    if model_policy_payload:
        lines.append(f"Model        : {model_policy_payload['selected_model_id']}")
        lines.append(f"Model role   : {model_policy_payload['variant_role']}")
    lines.extend(
        [
            "",
            f"{'Run Label':<26} {'TIS best frame':<17} {'TIS majority vote':<20} Species% (best)",
            f"{'-' * 75}",
        ]
    )
    for label, result in all_results.items():
        if result is None:
            lines.append(f"{label:<26} SKIPPED (unsupported or failed)")
        else:
            summary = result["summary"]
            lines.append(
                f"{label:<26} {summary['mean_tis_best_frame']:<17.3f} "
                f"{summary['mean_tis_majority_vote']:<20.3f} {summary['best_frame']['species_pct']}%"
            )

    for line in lines:
        print(line)

    summary_path = RESULTS_DIR / summary_name(timestamp, run_tag)
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSummary saved to: {summary_path}")
    return summary_path


def select_runs(labels: list[str] | None) -> list[dict[str, Any]]:
    if not labels:
        return RUNS_CONFIG
    return [run_cfg for run_cfg in RUNS_CONFIG if run_cfg["label"] in labels]


def print_dry_run(
    *,
    timestamp: str,
    image_dir: Path,
    run_tag: str | None,
    limit_events: int | None,
    runs_to_execute: list[dict[str, Any]],
    events: list[dict[str, Any]],
    model_policy_payload: dict[str, Any],
    image_policy_payload: dict[str, Any],
    geo_overrides: dict[str, str] | None,
) -> None:
    print("\nDRY RUN - configuration only")
    print("No SpeciesNet module or inference command is executed in dry-run mode.")
    print(f"Timestamp    : {timestamp}")
    print(f"Run tag      : {run_tag or 'none'}")
    print(f"Event limit  : {limit_events or 'none'}")
    print(f"Loaded events: {len(events)}")
    print(f"Loaded frames: {sum(len(event['frames']) for event in events)}")
    print(f"Image policy : {image_policy_payload['input_policy']}")
    print(f"Model        : {model_policy_payload['selected_model_id']}")
    print(f"Model role   : {model_policy_payload['variant_role']}")
    print("\nCurrent model examples:")
    for version, example in SPECIESNET_MODEL_EXAMPLES.items():
        print(f"  {version:<8} {example['model_id']} - {example['role']}")
    print("\nRuns scheduled:")
    for cfg in runs_to_execute:
        policy = build_country_policy(events, geo=cfg["geo"], geo_overrides=geo_overrides)
        print(
            f"  {cfg['label']:<25} geo={cfg['geo']}, "
            f"ensemble={not cfg['no_ensemble']}, "
            f"countries={policy['effective_country_code_counts']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SpeciesNet Camera Trap Benchmark",
        epilog=(
            "Model examples: "
            "v4.0.3a=kaggle:google/speciesnet/pyTorch/v4.0.3a/1; "
            "v4.0.3b=kaggle:google/speciesnet/pyTorch/v4.0.3b/1. "
            "Use --run_tag v403b_uncropped_proto330 for clear output names."
        ),
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        choices=[r["label"] for r in RUNS_CONFIG],
        help="Specific runs to execute (default: all)",
    )
    parser.add_argument(
        "--image_dir",
        type=Path,
        default=None,
        help="Flat image package directory (default: historical track_b/data/prototype_original)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "SpeciesNet model ID, e.g. "
            "kaggle:google/speciesnet/pyTorch/v4.0.3b/1 for the full-image model"
        ),
    )
    parser.add_argument(
        "--run_tag",
        type=str,
        default=None,
        help="Optional filename/results label, e.g. v403b_uncropped_proto330",
    )
    parser.add_argument(
        "--limit_events",
        type=int,
        default=None,
        help="Limit to the first N sorted events for a later tiny smoke run",
    )
    parser.add_argument(
        "--eu_geo",
        type=str,
        default=None,
        help="Override geo hint for European BEL/LUX events, e.g. GBR",
    )
    parser.add_argument(
        "--species_geo",
        type=str,
        nargs=2,
        action="append",
        default=[],
        metavar=("SPECIES", "COUNTRY"),
        help="Override geo hint for a specific species, e.g. --species_geo 'Martes foina' BEL",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print configuration and exit without importing or running SpeciesNet inference",
    )
    args = parser.parse_args()

    image_dir = args.image_dir or IMAGE_DIR
    model_id = args.model
    run_tag = safe_label(args.run_tag)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_started_at = utc_now_iso()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    runs_to_execute = select_runs(args.runs)
    model_policy_payload = model_policy(model_id)

    print("SpeciesNet Camera Trap Benchmark")
    print(f"Timestamp  : {timestamp}")
    print(f"Image dir  : {image_dir}")
    print(f"Results dir: {RESULTS_DIR}")
    print(f"Run tag    : {run_tag or 'none'}")
    print(f"Model      : {model_policy_payload['selected_model_id']}")

    if not image_dir.exists():
        print(f"\nERROR: Image directory not found: {image_dir}")
        print("  Expected: directory with *_meta.json and *_frame*.jpg files")
        sys.exit(1)

    image_policy_payload = image_package_policy(image_dir)
    events = load_events(image_dir)
    if not events:
        print("ERROR: No events loaded. Check image directory.")
        sys.exit(1)

    try:
        events = apply_event_limit(events, args.limit_events)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    geo_overrides: dict[str, str] = {}
    species_geo_map = {species: country_code for species, country_code in args.species_geo}
    if args.eu_geo or species_geo_map:
        for event in events:
            species = event.get("species", "")
            if species in species_geo_map:
                geo_overrides[event["event_id"]] = species_geo_map[species]
            elif args.eu_geo and event["country_code"] in {"BEL", "LUX"}:
                geo_overrides[event["event_id"]] = args.eu_geo
        if geo_overrides:
            cc_counts = Counter(geo_overrides.values())
            print(f"Geo overrides: {dict(cc_counts)} across {len(geo_overrides)} events")
    geo_overrides_payload = geo_overrides or None

    if args.dry_run:
        print_dry_run(
            timestamp=timestamp,
            image_dir=image_dir,
            run_tag=run_tag,
            limit_events=args.limit_events,
            runs_to_execute=runs_to_execute,
            events=events,
            model_policy_payload=model_policy_payload,
            image_policy_payload=image_policy_payload,
            geo_overrides=geo_overrides_payload,
        )
        return

    speciesnet_runtime_payload = speciesnet_runtime_record()
    print(
        "SpeciesNet runtime: "
        f"{speciesnet_runtime_payload.get('python_executable')} "
        f"(speciesnet={speciesnet_runtime_payload.get('speciesnet_version')})"
    )

    needs_classifier_only = any(run_cfg["no_ensemble"] for run_cfg in runs_to_execute)
    has_classifier_only = True
    if needs_classifier_only:
        has_classifier_only = speciesnet_supports_classifier_only()
        if not has_classifier_only:
            print("\nNOTE: --classifier_only not detected in this SpeciesNet environment.")
            print("  classifier-only runs will be skipped.")

    runs_with_dir = [{**cfg, "image_dir": image_dir} for cfg in runs_to_execute]
    print(f"\nRuns scheduled: {[run['label'] for run in runs_to_execute]}")

    all_results: dict[str, dict[str, Any] | None] = {}
    saved_paths: dict[str, Path] = {}

    for run_cfg in runs_with_dir:
        if run_cfg["no_ensemble"] and not has_classifier_only:
            print(f"\n  SKIPPING {run_cfg['label']} - --classifier_only not supported")
            all_results[run_cfg["label"]] = None
            continue

        result = execute_run(
            run_cfg,
            events,
            timestamp,
            model_id=model_id,
            run_tag=run_tag,
            limit_events=args.limit_events,
            geo_overrides=geo_overrides_payload,
            model_policy_payload=model_policy_payload,
            image_policy_payload=image_policy_payload,
            speciesnet_runtime_payload=speciesnet_runtime_payload,
        )
        all_results[run_cfg["label"]] = result

        if result is not None:
            out_path = RESULTS_DIR / f"{output_stem(run_cfg['label'], timestamp, run_tag)}.json"
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            saved_paths[run_cfg["label"]] = out_path
            print(f"  Saved: {out_path}")

    print(f"\n{'=' * 75}")
    print("  FINAL SUMMARY")
    print(f"{'=' * 75}")
    summary_path = print_and_save_summary(
        all_results,
        timestamp,
        image_dir=image_dir,
        run_tag=run_tag,
        model_policy_payload=model_policy_payload,
    )

    raw_dir = RESULTS_DIR / "raw"
    raw_dir.mkdir(exist_ok=True)
    for tmp in TEMP_DIR.glob("*_raw.json"):
        dest = raw_dir / tmp.name
        try:
            tmp.rename(dest)
        except FileExistsError:
            try:
                dest.unlink()
            except FileNotFoundError:
                pass
            tmp.rename(dest)

    for label, result in all_results.items():
        if result is None or label not in saved_paths:
            continue
        raw_paths = sorted(raw_dir.glob(f"{raw_output_prefix(label, timestamp, run_tag)}_*_raw.json"))
        manifest_path = write_speciesnet_run_manifest(
            result_json_path=saved_paths[label],
            raw_output_paths=raw_paths,
            result=result,
            image_dir=image_dir,
            events=events,
            started_at=run_started_at,
            completed_at=utc_now_iso(),
            run_parameters={
                "run_label": label,
                "run_tag": run_tag,
                "timestamp": timestamp,
                "model_id_override": model_id,
                "image_dir": str(image_dir),
                "limit_events": args.limit_events,
                "geo": result.get("mode", {}).get("geo"),
                "ensemble": result.get("mode", {}).get("ensemble"),
                "species_geo_override": args.species_geo,
                "eu_geo_override": args.eu_geo,
                "speciesnet_python": speciesnet_python_executable(),
                "speciesnet_runtime": speciesnet_runtime_payload,
                "speciesnet_model_policy": model_policy_payload,
                "image_package_policy": image_policy_payload,
                "country_policy": result.get("country_policy"),
            },
            extra={
                "raw_cli_output_count": len(raw_paths),
                "summary_file": str(summary_path),
                "speciesnet_runtime": speciesnet_runtime_payload,
                "dry_run": False,
            },
        )
        print(f"  Manifest: {manifest_path}")

    print("Done.")


if __name__ == "__main__":
    main()
