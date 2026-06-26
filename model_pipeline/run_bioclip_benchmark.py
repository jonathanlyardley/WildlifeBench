#!/usr/bin/env python3
"""BioCLIP comparator runner for WildlifeBench Track B."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from run_manifest import utc_now_iso, write_bioclip_run_manifest


IMAGE_DIR = Path("track_b/data/gbif_award/prototype330_cropped_runner")
RESULTS_DIR = Path("track_b/results/bioclip")
RAW_DIR = RESULTS_DIR / "raw"
TEMP_DIR = RESULTS_DIR / "_tmp"
DEFAULT_MODEL = "hf-hub:imageomics/bioclip-2"
COUNTRY_MAP = {
    "Belgium": "BEL",
    "Ecuador": "ECU",
    "France": "FRA",
    "Germany": "DEU",
    "Luxembourg": "LUX",
}
TAXONOMIC_LEVELS = ("species", "genus", "family", "order", "class")


BIOCLIP_WORKER = r"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from importlib import metadata as importlib_metadata

import open_clip as oc
import torch
import torch.nn.functional as F
from PIL import Image

from bioclip.predict import OPENA_AI_IMAGENET_TEMPLATE, Rank, TreeOfLifeClassifier, create_bioclip_tokenizer


def now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def species_epithet(label):
    parts = str(label or "").split()
    return parts[1] if len(parts) > 1 else None


def enrich_custom_predictions(predictions, label_taxonomy):
    enriched = []
    for row in predictions:
        label = row.get("classification")
        taxonomy = label_taxonomy.get(label, {})
        enriched_row = {
            "file_name": row.get("file_name"),
            "kingdom": taxonomy.get("kingdom"),
            "phylum": taxonomy.get("phylum"),
            "class": taxonomy.get("class"),
            "order": taxonomy.get("order"),
            "family": taxonomy.get("family"),
            "genus": taxonomy.get("genus") or (str(label).split()[0] if label else None),
            "species_epithet": taxonomy.get("species_epithet") or species_epithet(label),
            "species": taxonomy.get("species") or label,
            "common_name": taxonomy.get("common_name"),
            "score": row.get("score"),
            "classification": label,
        }
        enriched.append(enriched_row)
    return enriched


def ensure_rgb_image(image_path):
    return Image.open(image_path).convert("RGB")


def label_templates(policy):
    if policy == "single_photo":
        return ["a photo of a {}."]
    return OPENA_AI_IMAGENET_TEMPLATE


def build_text_embeddings(model, tokenizer, labels, device, text_batch_size, template_policy):
    templates = label_templates(template_policy)
    texts = []
    spans = []
    for label in labels:
        start = len(texts)
        texts.extend(template.format(label) for template in templates)
        spans.append((start, len(texts)))

    encoded_chunks = []
    with torch.inference_mode():
        for start in range(0, len(texts), text_batch_size):
            chunk = texts[start:start + text_batch_size]
            tokens = tokenizer(chunk).to(device)
            features = model.encode_text(tokens)
            encoded_chunks.append(F.normalize(features, dim=-1).detach().cpu())
    all_features = torch.cat(encoded_chunks, dim=0)
    label_features = []
    for start, end in spans:
        features = all_features[start:end].mean(dim=0)
        features = features / features.norm()
        label_features.append(features)
    return torch.stack(label_features, dim=1)


def label_cache_key(model_name, pretrained, labels, pybioclip_version, template_policy):
    payload = json.dumps(
        {
            "model": model_name,
            "pretrained": pretrained,
            "labels": labels,
            "pybioclip_version": pybioclip_version,
            "template_policy": template_policy,
            "templates": len(label_templates(template_policy)),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_or_create_text_embeddings(model, tokenizer, request, pybioclip_version):
    labels = request["labels"]
    template_policy = request.get("custom_label_template_policy") or "openai_imagenet"
    cache_path = Path(request["embedding_cache_path"])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    expected_key = label_cache_key(request["model"], request.get("pretrained"), labels, pybioclip_version, template_policy)
    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        if cached.get("cache_key") == expected_key and cached.get("labels") == labels:
            return cached["text_embeddings"], True, expected_key
    text_embeddings = build_text_embeddings(
        model=model,
        tokenizer=tokenizer,
        labels=labels,
        device=request["device"],
        text_batch_size=request.get("text_batch_size") or 512,
        template_policy=template_policy,
    )
    torch.save(
        {
            "cache_key": expected_key,
            "labels": labels,
            "model": request["model"],
            "pretrained": request.get("pretrained"),
            "pybioclip_version": pybioclip_version,
            "custom_label_template_policy": template_policy,
            "text_embeddings": text_embeddings,
        },
        cache_path,
    )
    return text_embeddings, False, expected_key


def predict_custom_labels(request, pybioclip_version):
    model, preprocess = oc.create_model_from_pretrained(
        request["model"],
        pretrained=request.get("pretrained"),
        device=request["device"],
        return_transform=True,
    )
    model = model.to(request["device"]).eval()
    tokenizer = create_bioclip_tokenizer(request["model"])
    text_embeddings, cache_hit, cache_key = load_or_create_text_embeddings(model, tokenizer, request, pybioclip_version)
    text_embeddings = text_embeddings.to(request["device"])
    rows = []
    images = request["images"]
    batch_size = request["batch_size"] or len(images)
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            image_paths = images[start:start + batch_size]
            image_tensors = [preprocess(ensure_rgb_image(image_path)).to(request["device"]) for image_path in image_paths]
            image_tensor = torch.stack(image_tensors)
            image_features = F.normalize(model.encode_image(image_tensor), dim=-1)
            logits = model.logit_scale.exp() * image_features @ text_embeddings
            probs = F.softmax(logits, dim=1).detach().cpu()
            k = min(request["k"], len(request["labels"]))
            for image_path, image_probs in zip(image_paths, probs):
                topk = image_probs.topk(k)
                for index, score in zip(topk.indices.tolist(), topk.values.tolist()):
                    rows.append(
                        {
                            "file_name": image_path,
                            "classification": request["labels"][index],
                            "score": score,
                        }
                    )
    return enrich_custom_predictions(rows, request.get("label_taxonomy", {})), {
        "embedding_cache_path": str(request["embedding_cache_path"]),
        "embedding_cache_hit": cache_hit,
        "embedding_cache_key": cache_key,
        "custom_label_template_policy": request.get("custom_label_template_policy") or "openai_imagenet",
    }


def main():
    request_path = Path(sys.argv[1])
    request = json.loads(request_path.read_text(encoding="utf-8"))
    started_at = now()
    classifier_mode = request.get("classifier_mode", "treeoflife")
    pybioclip_version = importlib_metadata.version("pybioclip")
    custom_label_log = None
    if classifier_mode == "custom_labels":
        predictions, custom_label_log = predict_custom_labels(request, pybioclip_version)
    else:
        classifier = TreeOfLifeClassifier(
            device=request["device"],
            model_str=request["model"],
            pretrained_str=request.get("pretrained"),
        )
        if request.get("subset_csv"):
            keep = classifier.create_taxa_filter_from_csv(request["subset_csv"])
            classifier.apply_filter(keep)
        predictions = classifier.predict(
            images=request["images"],
            rank=Rank.SPECIES,
            k=request["k"],
            batch_size=request["batch_size"],
        )
    output_path = Path(request["output_csv"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preferred = ["file_name", "kingdom", "phylum", "class", "order", "family", "genus", "species_epithet", "species", "common_name", "score"]
    seen = {key for row in predictions for key in row}
    fieldnames = [key for key in preferred if key in seen] + sorted(seen - set(preferred))
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(predictions)
    completed_at = now()
    log_payload = {
        "schema_version": "wildlifebench.bioclip_worker_log.v1",
        "started_at": started_at,
        "completed_at": completed_at,
        "request_file": str(request_path),
        "output_csv": str(output_path),
        "model": request["model"],
        "pretrained": request.get("pretrained"),
        "device": request["device"],
        "classifier_mode": classifier_mode,
        "subset_csv": request.get("subset_csv"),
        "label_count": len(request.get("labels") or []),
        "image_count": len(request["images"]),
        "prediction_count": len(predictions),
        "k": request["k"],
        "batch_size": request["batch_size"],
        "pybioclip_version": pybioclip_version,
        "custom_label_inference": custom_label_log,
    }
    Path(request["log_json"]).write_text(json.dumps(log_payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
"""


def safe_label(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("._-")
    return cleaned or None


def model_key(model: str) -> str:
    if model == "hf-hub:imageomics/bioclip-2":
        return "bioclip2"
    if model == "hf-hub:imageomics/bioclip-2.5-vith14":
        return "bioclip25"
    if model == "hf-hub:imageomics/bioclip":
        return "bioclip"
    return safe_label(model) or "bioclip"


def supports_treeoflife_classifier(model: str) -> bool:
    return model in {"hf-hub:imageomics/bioclip", "hf-hub:imageomics/bioclip-2"}


def classifier_mode_for_model(model: str, candidate_policy: str) -> str:
    if supports_treeoflife_classifier(model):
        return "treeoflife"
    if candidate_policy == "tol_rank":
        raise ValueError(
            f"{model} is not supported by pybioclip TreeOfLifeClassifier; "
            "use a candidate-list policy such as geo or blind."
        )
    return "custom_labels"


def output_stem(model: str, candidate_policy: str, aggregation: str, timestamp: str, run_tag: str | None) -> str:
    parts = [model_key(model)]
    if run_tag:
        parts.append(run_tag)
    parts.extend([safe_label(candidate_policy) or candidate_policy, safe_label(aggregation) or aggregation, timestamp])
    return "_".join(parts)


def custom_embedding_cache_path(model: str, pretrained: str | None, labels: list[str], template_policy: str) -> Path:
    payload = json.dumps(
        {
            "model": model,
            "pretrained": pretrained,
            "labels": labels,
            "custom_label_template_policy": template_policy,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
    return TEMP_DIR / "custom_label_embeddings" / f"{model_key(model)}_{digest}.pt"


def bioclip_python_executable() -> str:
    return os.environ.get("BIOCLIP_PYTHON", sys.executable)


def bioclip_runtime_record() -> dict[str, Any]:
    executable = bioclip_python_executable()
    code = (
        "import json, sys, platform\n"
        "from importlib import metadata as m\n"
        "def version(name):\n"
        "    try:\n"
        "        return m.version(name)\n"
        "    except m.PackageNotFoundError:\n"
        "        return None\n"
        "print(json.dumps({\n"
        "    'python_executable': sys.executable,\n"
        "    'python_version': sys.version.split()[0],\n"
        "    'platform': platform.platform(),\n"
        "    'pybioclip_version': version('pybioclip'),\n"
        "    'torch_version': version('torch'),\n"
        "    'open_clip_torch_version': version('open-clip-torch'),\n"
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
        return {"python_executable": executable, "query_error": str(exc)}
    if result.returncode != 0:
        return {
            "python_executable": executable,
            "query_error": (result.stderr or result.stdout).strip(),
            "returncode": result.returncode,
        }
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"python_executable": executable, "query_error": "Could not parse runtime JSON.", "stdout": result.stdout}
    payload["query_method"] = "external_python_importlib_metadata"
    return payload


def load_events(image_dir: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for meta_path in sorted(image_dir.glob("*_meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        event_id = meta.get("event_id") or meta_path.stem.replace("_meta", "")
        frames = sorted(image_dir.glob(f"{event_id}_frame*.jpg"))
        if not frames:
            frames = sorted(image_dir.glob(f"{event_id}*.jpg"))
        if not frames:
            print(f"  WARNING: No frames found for {event_id}")
            continue
        country = meta.get("country", "")
        country_code = meta.get("country_code") or COUNTRY_MAP.get(country, "")
        events.append(
            {
                "event_id": event_id,
                "species": meta.get("species"),
                "genus": meta.get("genus"),
                "family": meta.get("family"),
                "order": meta.get("order"),
                "class": meta.get("class"),
                "country": country,
                "country_code": country_code,
                "difficulty": (meta.get("_review") or {}).get("difficulty"),
                "time_of_day": (meta.get("_review") or {}).get("time_of_day"),
                "frames": [str(frame.resolve()) for frame in frames],
            }
        )
    return events


def apply_event_limit(events: list[dict[str, Any]], limit_events: int | None) -> list[dict[str, Any]]:
    if limit_events is None:
        return events
    if limit_events <= 0:
        raise ValueError("--limit_events must be a positive integer")
    return sorted(events, key=lambda event: event["event_id"])[:limit_events]


def image_package_policy(image_dir: Path) -> dict[str, Any]:
    manifest_path = image_dir / "manifest.json"
    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "image_dir": str(image_dir),
        "input_policy": "timestamp-cropped EXIF-stripped flat BioCLIP runner package",
        "manifest_path": str(manifest_path) if manifest_path.exists() else None,
        "event_set_id": manifest.get("event_set_id"),
        "event_count": manifest.get("event_count"),
        "species_count": manifest.get("species_count"),
        "frame_count": manifest.get("frame_count"),
        "country_counts": manifest.get("country_counts"),
        "manifest_image_policy": manifest.get("image_policy"),
    }


def latest_candidate_file(candidate_dir: Path, policy: str, country_code: str | None = None) -> Path:
    if policy == "blind":
        pattern = "blind_candidates_*.json"
    elif policy == "geo":
        if not country_code:
            raise ValueError("geo candidate policy requires a country code")
        pattern = f"geo_candidates_{country_code}_*.json"
    elif policy == "closed_target_list":
        pattern = "closed_target_list_candidates_*.json"
    else:
        raise ValueError(f"Unsupported candidate policy: {policy}")
    matches = sorted(candidate_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No candidate list found for policy={policy}, country={country_code}")
    return matches[-1]


def load_candidate_payload(path: Path, *, allow_diagnostic: bool = False) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    provenance = payload.get("provenance") or {}
    if provenance.get("uses_benchmark_target_species_as_filter") and not allow_diagnostic:
        raise ValueError(f"Candidate list uses benchmark target species and is not public-safe: {path}")
    if not payload.get("public_comparator_eligible") and not allow_diagnostic:
        raise ValueError(f"Candidate list is not public-comparator eligible: {path}")
    subset_csv = Path(payload["subset_csv"])
    if not subset_csv.exists():
        candidate = path.parent / subset_csv.name
        if candidate.exists():
            subset_csv = candidate
        else:
            raise FileNotFoundError(f"Subset CSV not found for {path}: {payload['subset_csv']}")
    payload["_path"] = str(path)
    payload["_subset_csv_path"] = str(subset_csv)
    return payload


def load_taxonomy_by_species(tol_taxa_path: Path) -> dict[str, dict[str, Any]]:
    taxonomy: dict[str, dict[str, Any]] = {}
    with tol_taxa_path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            species = (row.get("species") or "").strip()
            if not species:
                continue
            taxonomy[species] = {
                rank: (row.get(rank) or None)
                for rank in ("kingdom", "phylum", "class", "order", "family", "genus", "species_epithet", "species", "common_name")
            }
    return taxonomy


def taxonomy_for_labels(labels: list[str], taxonomy_by_species: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        label: taxonomy_by_species[label]
        for label in labels
        if label in taxonomy_by_species
    }


def candidate_groups(
    events: list[dict[str, Any]],
    *,
    candidate_policy: str,
    candidate_list_dir: Path | None,
) -> list[dict[str, Any]]:
    if candidate_policy == "tol_rank":
        return [
            {
                "label": "full_tol",
                "events": events,
                "subset_csv": None,
                "candidate_payload": None,
            }
        ]
    if candidate_list_dir is None:
        raise ValueError(f"{candidate_policy} policy requires --candidate_list_dir")

    allow_diagnostic = candidate_policy == "closed_target_list"
    if candidate_policy in {"blind", "closed_target_list"}:
        path = latest_candidate_file(candidate_list_dir, candidate_policy)
        payload = load_candidate_payload(path, allow_diagnostic=allow_diagnostic)
        if not payload.get("labels"):
            raise ValueError(f"Candidate list is empty: {path}")
        return [
            {
                "label": candidate_policy,
                "events": events,
                "subset_csv": payload["_subset_csv_path"],
                "candidate_payload": payload,
            }
        ]

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        grouped[event.get("country_code") or "NONE"].append(event)
    groups = []
    for country_code, country_events in sorted(grouped.items()):
        if country_code == "NONE":
            raise ValueError("Geo policy cannot run events without country_code")
        path = latest_candidate_file(candidate_list_dir, "geo", country_code)
        payload = load_candidate_payload(path)
        if not payload.get("labels"):
            raise ValueError(f"Geo candidate list is empty: {path}")
        groups.append(
            {
                "label": country_code,
                "events": country_events,
                "subset_csv": payload["_subset_csv_path"],
                "candidate_payload": payload,
            }
        )
    return groups


def write_worker_script(timestamp: str) -> Path:
    path = TEMP_DIR / f"bioclip_worker_{timestamp}.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(BIOCLIP_WORKER, encoding="utf-8")
    return path


def run_bioclip_group(
    *,
    worker_script: Path,
    group: dict[str, Any],
    timestamp: str,
    output_prefix: str,
    model: str,
    pretrained: str | None,
    device: str,
    k: int,
    batch_size: int,
    classifier_mode: str,
    taxonomy_by_species: dict[str, dict[str, Any]],
    custom_label_template_policy: str,
) -> tuple[Path, Path, list[dict[str, Any]], float]:
    frames = [frame for event in group["events"] for frame in event["frames"]]
    csv_path = RAW_DIR / f"{output_prefix}_{group['label']}_frame_predictions.csv"
    log_path = RAW_DIR / f"{output_prefix}_{group['label']}_worker_log.json"
    request_path = TEMP_DIR / f"{output_prefix}_{group['label']}_request.json"
    for path in (csv_path, log_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing BioCLIP raw output: {path}")
    request = {
        "schema_version": "wildlifebench.bioclip_worker_request.v1",
        "timestamp": timestamp,
        "images": frames,
        "output_csv": str(csv_path),
        "log_json": str(log_path),
        "subset_csv": group["subset_csv"],
        "classifier_mode": classifier_mode,
        "labels": (group.get("candidate_payload") or {}).get("labels") or [],
        "label_taxonomy": taxonomy_for_labels((group.get("candidate_payload") or {}).get("labels") or [], taxonomy_by_species),
        "embedding_cache_path": str(
            custom_embedding_cache_path(
                model,
                pretrained,
                (group.get("candidate_payload") or {}).get("labels") or [],
                custom_label_template_policy,
            )
        ),
        "text_batch_size": 512,
        "custom_label_template_policy": custom_label_template_policy,
        "model": model,
        "pretrained": pretrained,
        "device": device,
        "k": k,
        "batch_size": batch_size,
    }
    request_path.write_text(json.dumps(request, indent=2), encoding="utf-8")
    print(f"  Running BioCLIP group {group['label']} on {len(frames)} frames...")
    start = time.time()
    result = subprocess.run(
        [bioclip_python_executable(), str(worker_script), str(request_path)],
        cwd=Path.cwd(),
        text=True,
        check=False,
    )
    elapsed = round(time.time() - start, 1)
    if result.returncode != 0:
        raise RuntimeError(f"BioCLIP group {group['label']} failed with exit code {result.returncode}")
    rows = read_prediction_csv(csv_path)
    print(f"  Finished {group['label']} in {elapsed}s")
    return csv_path, log_path, rows, elapsed


def read_prediction_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        try:
            row["score"] = float(row["score"])
        except (TypeError, ValueError, KeyError):
            row["score"] = None
    return rows


def confidence_from_score(score: float | None) -> str | None:
    if score is None:
        return None
    if score >= 0.5:
        return "high"
    if score >= 0.2:
        return "medium"
    return "low"


def prediction_from_row(row: dict[str, Any] | None) -> dict[str, Any]:
    if not row:
        return {
            "species": None,
            "genus": None,
            "family": None,
            "order": None,
            "class": None,
            "confidence": None,
            "score": None,
            "source": "bioclip_missing",
        }
    score = row.get("score")
    return {
        "species": row.get("species") or None,
        "genus": row.get("genus") or None,
        "family": row.get("family") or None,
        "order": row.get("order") or None,
        "class": row.get("class") or None,
        "common_name": row.get("common_name") or None,
        "confidence": confidence_from_score(score),
        "score": round(score, 6) if isinstance(score, float) else None,
        "source": "bioclip_topk",
    }


def group_rows_by_frame(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(Path(row.get("file_name", "")).resolve())].append(row)
    for frame_rows in grouped.values():
        frame_rows.sort(key=lambda item: item.get("score") or 0.0, reverse=True)
    return grouped


def aggregate_event(frame_rows: list[list[dict[str, Any]]], aggregation: str) -> dict[str, Any]:
    top_rows = [rows[0] for rows in frame_rows if rows]
    if not top_rows:
        missing = prediction_from_row(None)
        return {
            "selected": missing,
            "best_frame": missing,
            "majority_vote": missing,
            "mean_score": missing,
            "n_frames_with_prediction": 0,
        }

    best_row = max(top_rows, key=lambda row: row.get("score") or 0.0)
    species_counts = Counter(row.get("species") for row in top_rows if row.get("species"))
    if species_counts:
        top_species = species_counts.most_common(1)[0][0]
        majority_rows = [row for row in top_rows if row.get("species") == top_species]
        majority_row = max(majority_rows, key=lambda row: row.get("score") or 0.0)
        majority = prediction_from_row(majority_row)
        majority["vote_count"] = species_counts[top_species]
        majority["vote_fraction"] = round(species_counts[top_species] / len(top_rows), 3)
    else:
        majority = prediction_from_row(best_row)
        majority["vote_count"] = 0
        majority["vote_fraction"] = 0.0

    score_by_species: dict[str, list[float]] = defaultdict(list)
    row_by_species: dict[str, dict[str, Any]] = {}
    for rows in frame_rows:
        for row in rows:
            species = row.get("species")
            score = row.get("score")
            if species and isinstance(score, float):
                score_by_species[species].append(score)
                row_by_species.setdefault(species, row)
    if score_by_species:
        mean_species = max(score_by_species, key=lambda species: sum(score_by_species[species]) / len(score_by_species[species]))
        mean_score = sum(score_by_species[mean_species]) / len(score_by_species[mean_species])
        mean_prediction = prediction_from_row(row_by_species[mean_species])
        mean_prediction["score"] = round(mean_score, 6)
        mean_prediction["mean_score_frame_count"] = len(score_by_species[mean_species])
    else:
        mean_prediction = prediction_from_row(best_row)

    best = prediction_from_row(best_row)
    choices = {
        "best_frame": best,
        "majority_vote": majority,
        "mean_score": mean_prediction,
    }
    return {
        "selected": choices[aggregation],
        "best_frame": best,
        "majority_vote": majority,
        "mean_score": mean_prediction,
        "n_frames_with_prediction": len(top_rows),
    }


def build_result(
    *,
    events: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    model: str,
    timestamp: str,
    run_tag: str | None,
    candidate_policy: str,
    aggregation: str,
    image_dir: Path,
    candidate_payloads: list[dict[str, Any]],
    raw_outputs: list[Path],
    elapsed_by_group: dict[str, float],
    classifier_mode: str = "treeoflife",
    custom_label_template_policy: str | None = None,
) -> dict[str, Any]:
    rows_by_frame = group_rows_by_frame(prediction_rows)
    result_rows: list[dict[str, Any]] = []
    for event in events:
        frame_prediction_rows = [
            rows_by_frame.get(str(Path(frame).resolve()), [])
            for frame in event["frames"]
        ]
        aggregated = aggregate_event(frame_prediction_rows, aggregation)
        ground_truth = {
            rank: event.get(rank)
            for rank in TAXONOMIC_LEVELS
        }
        ground_truth["country"] = event.get("country")
        result_rows.append(
            {
                "event_id": event["event_id"],
                "ground_truth": ground_truth,
                "prediction": aggregated["selected"],
                "country": event.get("country"),
                "country_code": event.get("country_code"),
                "difficulty": event.get("difficulty"),
                "time_of_day": event.get("time_of_day"),
                "n_frames": len(event["frames"]),
                "benchmark_frames": len(event["frames"]),
                "n_frames_with_prediction": aggregated["n_frames_with_prediction"],
                "best_frame_prediction": aggregated["best_frame"],
                "majority_vote_prediction": aggregated["majority_vote"],
                "mean_score_prediction": aggregated["mean_score"],
                "frame_predictions": [
                    [prediction_from_row(row) for row in rows]
                    for rows in frame_prediction_rows
                ],
            }
        )

    return {
        "schema_version": "wildlifebench.bioclip_results.v1",
        "model": model_key(model),
        "model_id": model,
        "run_label": f"{candidate_policy}_{aggregation}",
        "run_tag": run_tag,
        "run_date": timestamp,
        "image_source": str(image_dir),
        "candidate_policy": candidate_policy,
        "aggregation": aggregation,
        "classifier_mode": classifier_mode,
        "custom_label_template_policy": custom_label_template_policy,
        "n_events": len(result_rows),
        "n_frames": sum(len(event["frames"]) for event in events),
        "candidate_lists": [
            {
                "path": payload.get("_path"),
                "subset_csv": payload.get("_subset_csv_path"),
                "policy": payload.get("policy"),
                "label_count": payload.get("label_count"),
                "public_comparator_eligible": payload.get("public_comparator_eligible"),
                "uses_benchmark_target_species_as_filter": (payload.get("provenance") or {}).get("uses_benchmark_target_species_as_filter"),
            }
            for payload in candidate_payloads
            if payload
        ],
        "raw_outputs": [str(path) for path in raw_outputs],
        "elapsed_seconds_by_group": elapsed_by_group,
        "results": result_rows,
    }


def print_dry_run(args: argparse.Namespace, events: list[dict[str, Any]], groups: list[dict[str, Any]], image_policy: dict[str, Any]) -> None:
    print("DRY RUN - configuration only")
    print("No BioCLIP model or inference command is executed.")
    print(f"Image dir       : {args.image_dir}")
    print(f"Image policy    : {image_policy['input_policy']}")
    print(f"Model           : {args.model}")
    print(f"Candidate policy: {args.candidate_policy}")
    print(f"Aggregation     : {args.aggregation}")
    print(f"Template policy : {getattr(args, 'custom_label_template_policy', None)}")
    print(f"Events          : {len(events)}")
    print(f"Frames          : {sum(len(event['frames']) for event in events)}")
    for group in groups:
        payload = group.get("candidate_payload") or {}
        print(
            f"  group={group['label']} events={len(group['events'])} "
            f"subset={group.get('subset_csv') or 'full TOL'} labels={payload.get('label_count', 'all')}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BioCLIP on a WildlifeBench flat runner package.")
    parser.add_argument("--image_dir", type=Path, default=IMAGE_DIR)
    parser.add_argument("--candidate_policy", choices=["tol_rank", "blind", "geo", "closed_target_list"], default="blind")
    parser.add_argument("--candidate_list_dir", type=Path)
    parser.add_argument("--aggregation", choices=["best_frame", "majority_vote", "mean_score"], default="majority_vote")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--pretrained")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--custom_label_template_policy", choices=["openai_imagenet", "single_photo"], default="openai_imagenet")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--run_tag")
    parser.add_argument("--limit_events", type=int)
    parser.add_argument("--tol_taxa", type=Path)
    parser.add_argument("--coverage_report", type=Path, action="append", default=[])
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    started_at = utc_now_iso()
    run_tag = safe_label(args.run_tag)
    output_prefix = output_stem(args.model, args.candidate_policy, args.aggregation, timestamp, run_tag)

    if not args.image_dir.exists():
        raise SystemExit(f"Image directory not found: {args.image_dir}")
    events = load_events(args.image_dir)
    if not events:
        raise SystemExit(f"No events loaded from {args.image_dir}")
    events = apply_event_limit(events, args.limit_events)
    image_policy = image_package_policy(args.image_dir)
    groups = candidate_groups(events, candidate_policy=args.candidate_policy, candidate_list_dir=args.candidate_list_dir)
    try:
        classifier_mode = classifier_mode_for_model(args.model, args.candidate_policy)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    taxonomy_by_species: dict[str, dict[str, Any]] = {}
    if classifier_mode == "custom_labels":
        if not args.tol_taxa:
            raise SystemExit(
                f"{args.model} uses custom-label inference in this runner; pass --tol_taxa so species labels can be mapped to genus/family/order/class."
            )
        if not args.tol_taxa.exists():
            raise SystemExit(f"TOL taxonomy file not found: {args.tol_taxa}")
        taxonomy_by_species = load_taxonomy_by_species(args.tol_taxa)

    if args.dry_run:
        print_dry_run(args, events, groups, image_policy)
        print(f"Classifier mode : {classifier_mode}")
        return

    runtime = bioclip_runtime_record()
    if not runtime.get("pybioclip_version"):
        raise SystemExit(f"BIOCLIP_PYTHON does not expose pybioclip: {runtime}")
    worker_script = write_worker_script(timestamp)
    raw_paths: list[Path] = []
    all_rows: list[dict[str, Any]] = []
    elapsed_by_group: dict[str, float] = {}
    candidate_payloads: list[dict[str, Any]] = []

    for group in groups:
        payload = group.get("candidate_payload")
        if payload:
            candidate_payloads.append(payload)
        csv_path, log_path, rows, elapsed = run_bioclip_group(
            worker_script=worker_script,
            group=group,
            timestamp=timestamp,
            output_prefix=output_prefix,
            model=args.model,
            pretrained=args.pretrained,
            device=args.device,
            k=args.k,
            batch_size=args.batch_size,
            classifier_mode=classifier_mode,
            taxonomy_by_species=taxonomy_by_species,
            custom_label_template_policy=args.custom_label_template_policy,
        )
        raw_paths.extend([csv_path, log_path])
        all_rows.extend(rows)
        elapsed_by_group[group["label"]] = elapsed

    result = build_result(
        events=events,
        prediction_rows=all_rows,
        model=args.model,
        timestamp=timestamp,
        run_tag=run_tag,
        candidate_policy=args.candidate_policy,
        aggregation=args.aggregation,
        image_dir=args.image_dir,
        candidate_payloads=candidate_payloads,
        raw_outputs=raw_paths,
        elapsed_by_group=elapsed_by_group,
        classifier_mode=classifier_mode,
        custom_label_template_policy=args.custom_label_template_policy if classifier_mode == "custom_labels" else None,
    )

    out_path = RESULTS_DIR / f"{output_prefix}.json"
    if out_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing BioCLIP result: {out_path}")
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved result: {out_path}")

    candidate_paths = []
    for payload in candidate_payloads:
        candidate_paths.append(Path(payload["_path"]))
        candidate_paths.append(Path(payload["_subset_csv_path"]))
    manifest_path = write_bioclip_run_manifest(
        result_json_path=out_path,
        raw_output_paths=raw_paths,
        result=result,
        image_dir=args.image_dir,
        events=events,
        started_at=started_at,
        completed_at=utc_now_iso(),
        run_parameters={
            "run_tag": run_tag,
            "timestamp": timestamp,
            "candidate_policy": args.candidate_policy,
            "aggregation": args.aggregation,
            "model": args.model,
            "pretrained": args.pretrained,
            "device": args.device,
            "batch_size": args.batch_size,
            "custom_label_template_policy": args.custom_label_template_policy,
            "k": args.k,
            "limit_events": args.limit_events,
            "classifier_mode": classifier_mode,
            "bioclip_python": bioclip_python_executable(),
            "bioclip_runtime": runtime,
            "image_package_policy": image_policy,
        },
        candidate_list_paths=candidate_paths,
        tol_taxa_path=args.tol_taxa,
        coverage_report_paths=args.coverage_report,
        extra={
            "worker_script": str(worker_script),
            "raw_output_count": len(raw_paths),
            "candidate_payload_count": len(candidate_payloads),
            "source_overlap_policy": {
                "status": "not_assessed_by_runner",
                "note": "Candidate-list provenance is recorded here; BioCLIP training-data overlap requires separate methodology notes before public claims.",
            },
            "dry_run": False,
        },
    )
    print(f"Manifest: {manifest_path}")
    print("Done.")


if __name__ == "__main__":
    main()
