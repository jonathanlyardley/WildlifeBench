#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def run(name: str) -> int:
    result = subprocess.run([sys.executable, str(ROOT / "scripts" / name)], cwd=ROOT)
    return result.returncode


def main() -> int:
    manifest = json.loads((ROOT / "release" / "release_manifest.json").read_text(encoding="utf-8"))
    events = load_jsonl(ROOT / "data" / "model_inputs" / "events.jsonl")
    answer_key = load_jsonl(ROOT / "data" / "ground_truth" / "answer_key.jsonl")
    if len(events) != manifest["counts"]["source_events"]:
        print("event count mismatch", file=sys.stderr)
        return 1
    if len(answer_key) != len(events):
        print("answer key count mismatch", file=sys.stderr)
        return 1
    for helper in (
        "check_answer_key_isolation.py",
        "check_media_metadata.py",
        "validate_standards_metadata.py",
        "check_checksums.py",
        "score_cached_outputs.py",
        "build_submission_table.py",
    ):
        code = run(helper)
        if code:
            return code
    print("release validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
