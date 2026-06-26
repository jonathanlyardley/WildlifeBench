
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
