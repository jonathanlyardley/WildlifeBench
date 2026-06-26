#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_KEYS = {"species", "genus", "family", "order", "class", "source_event_id", "gbif_key", "dataset_key"}


def load_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def walk_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from walk_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from walk_keys(item)


def main() -> int:
    model_path = ROOT / "data" / "model_inputs" / "events.jsonl"
    model_events = load_jsonl(model_path)
    answer_key = load_jsonl(ROOT / "data" / "ground_truth" / "answer_key.jsonl")
    model_text = model_path.read_text(encoding="utf-8").lower()
    for event in model_events:
        overlap = FORBIDDEN_KEYS.intersection(set(walk_keys(event)))
        if overlap:
            print(f"forbidden keys in model input {event.get('event_id')}: {sorted(overlap)}", file=sys.stderr)
            return 1
    for row in answer_key:
        source_id = str(row.get("source_event_id") or "").lower()
        if source_id and source_id in model_text:
            print(f"source event id leaked into model inputs: {source_id}", file=sys.stderr)
            return 1
        taxonomy = row.get("taxonomy") or {}
        for value in taxonomy.values():
            text = str(value or "").strip().lower()
            if text and text in model_text:
                print(f"taxon label leaked into model inputs: {text}", file=sys.stderr)
                return 1
    print("answer-key isolation check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
