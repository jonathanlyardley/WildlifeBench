"""
run_prototype_inference.py
--------------------------
Runs inference on the flat prototype_cropped dataset for one council model.

Directory layout expected (flat, no region sub-folders):
    track_b/data/prototype_cropped/
        ct_<species>_NNN_frame0..N.jpg
        ct_<species>_NNN_meta.json
        ...

Each meta.json supplies ground-truth species, genus, family, order, class,
country, and the reviewed frame list.

Returns the path to the saved results JSON.
"""

import json
import os
import time
import glob
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import config
from providers.google_provider import GoogleProvider
from providers.groq_provider import GroqProvider
try:
    from providers.openrouter_provider import OpenRouterProvider
except ModuleNotFoundError:
    OpenRouterProvider = None
from run_manifest import flush_langfuse, utc_now_iso, write_llm_run_manifest

# Langfuse @observe — graceful no-op if package missing or LANGFUSE_PUBLIC_KEY
# unset. Wrapping the top-level entry point creates a parent trace under which
# all child `generate_response` (generation) and `compute_adjusted_tis` (span)
# calls nest automatically — gives a per-run timeline view in the dashboard.
try:
    from langfuse import observe as _lf_observe, get_client as _lf_get_client
    _LANGFUSE_AVAILABLE = True
except ImportError:
    _LANGFUSE_AVAILABLE = False
    def _lf_observe(*args, **kwargs):
        def _decorator(fn):
            return fn
        return _decorator
    def _lf_get_client():
        return None

# Default points at the post-2026-04-17 canonical timestamp-cropped tree
# (nested class/species/burst_NNN/ layout). The legacy flat
# `track_b/data/prototype_cropped/` tree is frozen and not the live target.
# Override per call via `dataset_dir=Path(...)` if needed.
PROTOTYPE_DIR = Path("track_b/data/prototype_cropped_timestamp")
RESULTS_DIR   = Path("track_b/results/council")


# ── helpers ───────────────────────────────────────────────────

RATE_LIMIT_ERROR_MARKERS = (
    "429",
    "resource_exhausted",
    "too many requests",
    "rate limit",
    "quota",
)


def is_rate_limit_error(error: object) -> bool:
    """Return True when a provider error is a rate/quota exhaustion signal."""
    if not error:
        return False
    text = str(error).lower()
    return any(marker in text for marker in RATE_LIMIT_ERROR_MARKERS)


def non_groq_delay_seconds() -> float:
    """Inter-call delay for non-Groq API routes."""
    raw = os.getenv("WILDLIFEBENCH_API_DELAY_SECONDS")
    if raw is None or raw.strip() == "":
        return 0.5
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(
            "WILDLIFEBENCH_API_DELAY_SECONDS must be a non-negative number"
        ) from exc
    if value < 0:
        raise ValueError("WILDLIFEBENCH_API_DELAY_SECONDS must be non-negative")
    return value


def get_provider(model_name: str):
    cfg = config.MODELS[model_name]
    if cfg["provider"] == "google":
        return GoogleProvider(model_name)
    elif cfg["provider"] == "groq":
        return GroqProvider(model_name)
    elif cfg["provider"] == "openrouter":
        if OpenRouterProvider is None:
            raise ModuleNotFoundError(
                "providers.openrouter_provider is unavailable; cannot run OpenRouter models"
            )
        return OpenRouterProvider(model_name)
    raise ValueError(f"Unknown provider for {model_name}")


def load_events(
    dataset_dir: Path,
    per_species_limit: int | None = None,
    difficulty: str | None = None,
) -> list[dict]:
    """
    Scan dataset_dir for *_meta.json files, read ground truth, and resolve
    frame paths.  Optionally cap the number of bursts taken per species, and
    optionally filter to a single _review.difficulty stratum (e.g. "medium"
    for Loop A / Loop B quick-eval).

    Events are sorted by event_id for deterministic ordering across runs.

    Uses rglob (recursive), but EXCLUDES paths under any `.trash/` directory.
    Curated-out bursts are kept on disk under `.trash/` for archival; they
    must not appear in any live evaluation. Without this filter, rglob
    silently inflates the prototype count from the true 330 (110 species x
    3 bursts) to 353 by including 23 trashed bursts — see DECISIONS.md
    2026-05-06.

    Layout supported: legacy flat `prototype_cropped/<stem>_meta.json` AND
    the post-2026-04-17 nested
    `prototype_cropped_timestamp/<class>/<species>/burst_NNN/<stem>_meta.json`
    that `crop_bursts.py --method timestamp` writes.
    """
    meta_files = sorted(
        p for p in dataset_dir.rglob("*_meta.json") if ".trash" not in p.parts
    )
    if not meta_files:
        raise FileNotFoundError(f"No *_meta.json files found in {dataset_dir}")

    species_counts: dict[str, int] = defaultdict(int)
    events = []

    for mf in meta_files:
        try:
            with open(mf, encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            continue

        species = meta.get("species", "Unknown")
        event_difficulty = meta.get("_review", {}).get("difficulty", "unset")

        if difficulty and event_difficulty != difficulty:
            continue

        if per_species_limit and species_counts[species] >= per_species_limit:
            continue

        # Resolve frame paths from sampled_frames indices (authoritative 5-frame set).
        # _review.frames is NOT used — it can contain extra frames due to a curation
        # bug in 46/160 events. sampled_frames is the canonical curated selection.
        stem = mf.stem.removesuffix("_meta")
        sampled = meta.get("sampled_frames", [])
        frames = []
        for idx in sampled[:5]:  # hard cap: never send more than 5 frames
            p = mf.parent / f"{stem}_frame{idx}.jpg"
            if p.exists():
                frames.append(str(p))

        # Fallback: discover frames from sibling files if sampled_frames absent
        if not frames:
            frames = sorted(str(x) for x in mf.parent.glob(f"{stem}_frame*.jpg"))[:5]

        if not frames:
            continue

        events.append({
            "event_id": mf.stem.removesuffix("_meta"),
            "species":  species,
            "genus":    meta.get("genus", ""),
            "family":   meta.get("family", ""),
            "order":    meta.get("order", ""),
            "class":    meta.get("class", ""),
            "country":  meta.get("country", "Unknown"),
            "time_of_day": meta.get("_review", {}).get("time_of_day", "day"),
            "difficulty":  meta.get("_review", {}).get("difficulty", "unset"),
            "frames":   frames,
        })
        species_counts[species] += 1

    events.sort(key=lambda e: e["event_id"])
    return events


def load_events_from_roots(
    dataset_dir: Path,
    *,
    extra_dataset_dirs: list[Path] | tuple[Path, ...] | None = None,
    per_species_limit: int | None = None,
    difficulty: str | None = None,
) -> list[dict]:
    """Load one evaluation event set from a primary root plus optional side roots."""
    roots = [Path(dataset_dir), *(Path(p) for p in (extra_dataset_dirs or []))]
    events: list[dict] = []
    seen: dict[str, Path] = {}

    for root in roots:
        for event in load_events(
            root,
            per_species_limit=per_species_limit,
            difficulty=difficulty,
        ):
            event_id = event["event_id"]
            if event_id in seen:
                raise ValueError(
                    f"Duplicate event_id {event_id!r} in {root} and {seen[event_id]}"
                )
            event["source_dataset_dir"] = str(root)
            seen[event_id] = root
            events.append(event)

    events.sort(key=lambda e: e["event_id"])
    return events


# ── main inference function ───────────────────────────────────

@_lf_observe(name="prototype_run")
def run_prototype_inference(
    dataset_dir=PROTOTYPE_DIR,
    model_name: str = "gemini-3-1-flash-lite",
    per_species_limit: int | None = None,
    difficulty: str | None = None,
    iter_num=None,
    verbose: bool = True,
    skip_ids: set | None = None,
    resume_results: list | None = None,
    extra_dataset_dirs: list[Path] | tuple[Path, ...] | None = None,
) -> Path:
    """
    Run inference for one model against the prototype dataset.

    Args:
        dataset_dir:        Path to flat prototype_final directory.
        model_name:         Key from config.MODELS.
        per_species_limit:  Max bursts per species (None = all).
        difficulty:         Optional _review.difficulty filter (e.g. "medium"
                            for the Loop A / Loop B quick-eval stratum).
                            None = include all strata (variance gate / full eval).
        iter_num:           Optional label appended to output filename.
        verbose:            Print per-event progress.
        skip_ids:           Set of event_ids to skip (already have valid responses).
        resume_results:     Previous results list to merge into the output file.
        extra_dataset_dirs: Optional side roots to append to the primary dataset.

    Returns:
        Path to the saved results JSON.
    """
    dataset_dir = Path(dataset_dir)
    extra_dataset_dirs = [Path(p) for p in (extra_dataset_dirs or [])]
    run_started_at = utc_now_iso()
    all_events = load_events_from_roots(
        dataset_dir,
        extra_dataset_dirs=extra_dataset_dirs,
        per_species_limit=per_species_limit,
        difficulty=difficulty,
    )
    dataset_roots = [dataset_dir, *extra_dataset_dirs]

    # Filter out already-completed events when resuming
    if skip_ids:
        events = [e for e in all_events if e["event_id"] not in skip_ids]
        if verbose:
            print(f"  [{model_name}]  {len(events)} events to run  "
                  f"({len(skip_ids)} skipped from previous run, "
                  f"{len(all_events)} total)")
    else:
        events = all_events
        if verbose:
            print(f"  [{model_name}]  {len(events)} events  "
                  f"(limit={per_species_limit or 'all'} per species)")

    provider = get_provider(model_name)
    is_groq = config.MODELS[model_name]["provider"] == "groq"

    # Tag the Langfuse parent trace with run-level context. Child
    # generate_response + compute_adjusted_tis spans nest under this trace
    # automatically. No-op if Langfuse inactive.
    if _LANGFUSE_AVAILABLE:
        try:
            lf = _lf_get_client()
            if lf is not None:
                lf.update_current_trace(
                    name=f"prototype_run.{model_name}",
                    input={
                        "dataset_dir": str(dataset_dir),
                        "dataset_roots": [str(p) for p in dataset_roots],
                        "model_name": model_name,
                        "per_species_limit": per_species_limit,
                        "n_events": len(events),
                        "n_skipped": len(skip_ids) if skip_ids else 0,
                    },
                    metadata={
                        "model": model_name,
                        "prompt_version": getattr(config, "PROMPT_VERSION", "unknown"),
                        "dataset": str(dataset_dir),
                        "dataset_roots": [str(p) for p in dataset_roots],
                        "iter_num": iter_num,
                        "is_resume": bool(skip_ids),
                    },
                    tags=[
                        "prototype_run",
                        f"model:{model_name}",
                        f"prompt:{getattr(config, 'PROMPT_VERSION', 'unknown')}",
                    ],
                )
        except Exception:
            pass

    results = []
    total_start = time.time()
    interrupted = False
    stop_reason = None

    # Resolve locked prompt-suffix policy once (constant within a run).
    model_cfg = config.MODELS[model_name]
    family = model_cfg.get("family", "")
    abstention_suffix = config.get_abstention_suffix_for_model(model_name)
    prompt_suffix_policy = config.get_prompt_suffix_policy_for_model(model_name)
    effective_max_output_tokens = config.get_effective_max_output_tokens_for_model(
        model_name
    )

    # Captured at end-of-loop into the output JSON for audit. Initialised so
    # an empty events list still yields a well-formed output.
    full_prompt_example = ""

    try:
        for i, ev in enumerate(events):
            prompt = config.PROMPT_TEMPLATE.format(
                n_frames=len(ev["frames"]),
                country=ev["country"],
            )
            if abstention_suffix:
                prompt = prompt + "\n\n" + abstention_suffix
            full_prompt_example = prompt

            if is_groq:
                time.sleep(2.5)

            t0 = time.time()
            resp = provider.generate_response(prompt, ev["frames"])
            latency = round(time.time() - t0, 2)
            rate_limit_error = is_rate_limit_error(resp.get("error"))

            # Normalise parsed_json: some models occasionally wrap their
            # response in a single-element JSON array `[{...}]` instead of
            # the expected bare object `{...}`. Coerce list-of-one to its
            # element. Anything else weird becomes None (treated downstream
            # as a model abstention rather than a parser crash).
            raw_parsed = resp.get("parsed_json")
            if isinstance(raw_parsed, list):
                raw_parsed = raw_parsed[0] if raw_parsed and isinstance(raw_parsed[0], dict) else None
            elif raw_parsed is not None and not isinstance(raw_parsed, dict):
                raw_parsed = None
            normalised_pred = raw_parsed

            if verbose:
                pred = normalised_pred or {}
                pred_sp = pred.get("species") or "null"
                hit = "HIT" if (pred_sp or "").lower() == ev["species"].lower() else "---"
                if rate_limit_error:
                    print(f"    [{i+1:3d}/{len(events)}]  {ev['event_id']}  ERR  "
                          f"rate_limit={resp.get('error')!r}  [{latency}s]", flush=True)
                else:
                    print(f"    [{i+1:3d}/{len(events)}]  {ev['event_id']}  {hit}  "
                          f"pred={pred_sp!r}  [{latency}s]", flush=True)

            results.append({
                "event_id":   ev["event_id"],
                "ground_truth": {
                    "species": ev["species"],
                    "genus":   ev["genus"],
                    "family":  ev["family"],
                    "order":   ev["order"],
                    "class":   ev["class"],
                    "country": ev["country"],
                },
                "prediction":   normalised_pred,
                "raw_response": resp.get("raw_response"),
                "error":        resp.get("error"),
                "latency":      latency,
                "tokens": {
                    "input":  resp.get("input_tokens"),
                    "output": resp.get("output_tokens"),
                },
                "telemetry": resp.get("telemetry"),
            })

            if rate_limit_error:
                interrupted = True
                stop_reason = "rate_limit"
                if verbose:
                    print(
                        f"\n  Rate limit detected after {len(results)}/{len(events)} events; "
                        "saving partial results for a throttled/resumed rerun.",
                        flush=True,
                    )
                break

            if not is_groq:
                time.sleep(non_groq_delay_seconds())

    except KeyboardInterrupt:
        interrupted = True
        if verbose:
            print(f"\n  Interrupted after {len(results)}/{len(events)} events — saving partial results.")

    total_latency = time.time() - total_start

    # Merge with previous results if resuming
    if resume_results:
        # Build lookup of new results by event_id
        new_by_id = {r["event_id"]: r for r in results}
        # Start from previous results: replace quota-error entries with new results
        # where available; keep previous genuine responses untouched
        merged = []
        for prev in resume_results:
            eid = prev["event_id"]
            if eid in new_by_id:
                merged.append(new_by_id[eid])
            else:
                merged.append(prev)
        # Append any new events that weren't in the previous file at all
        prev_ids = {r["event_id"] for r in resume_results}
        for r in results:
            if r["event_id"] not in prev_ids:
                merged.append(r)
        # Re-sort by event_id to maintain deterministic ordering
        merged.sort(key=lambda r: r["event_id"])
        final_results = merged
    else:
        final_results = results

    n_total_events = len(all_events)
    n_complete = sum(
        1 for r in final_results
        if r.get("prediction") is not None and not r.get("error")
    )
    is_partial = interrupted or (len(final_results) < n_total_events)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    iter_tag = f"_iter{iter_num}" if iter_num is not None else ""
    out_path = RESULTS_DIR / f"proto_{model_name}_{config.PROMPT_VERSION}{iter_tag}_{ts}.json"

    output = {
        "model":           model_name,
        "family":          family,
        "prompt_version":  config.PROMPT_VERSION,
        "prompt_template": config.PROMPT_TEMPLATE,
        "abstention_suffix":   abstention_suffix,
        "prompt_suffix_policy": prompt_suffix_policy,
        "full_prompt_example": full_prompt_example,
        "quick_eval_difficulty": difficulty,
        "dataset":         str(dataset_dir),
        "dataset_extra_dirs": [str(p) for p in extra_dataset_dirs],
        "dataset_roots":   [str(p) for p in dataset_roots],
        "run_date":        datetime.now().isoformat(),
        "n_events":        len(final_results),
        "n_events_total":  n_total_events,
        "n_complete":      n_complete,
        "partial":         is_partial,
        "stop_reason":     stop_reason,
        "resumed_from":    str(resume_results[0].get("event_id", "")) if resume_results else None,
        "per_species_limit": per_species_limit,
        "new_events_latency_seconds": round(total_latency, 2),
        "avg_latency_seconds":   round(total_latency / len(results), 2) if results else 0,
        "results": final_results,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    manifest_path = write_llm_run_manifest(
        result_json_path=out_path,
        model_name=model_name,
        model_config=model_cfg,
        provider_obj=provider,
        dataset_dir=dataset_dir,
        events=all_events,
        results=final_results,
        prompt_version=config.PROMPT_VERSION,
        prompt_template=config.PROMPT_TEMPLATE,
        full_prompt_example=full_prompt_example,
        abstention_suffix=abstention_suffix,
        started_at=run_started_at,
        completed_at=utc_now_iso(),
        run_parameters={
            "per_species_limit": per_species_limit,
            "difficulty": difficulty,
            "iter_num": iter_num,
            "dataset_extra_dirs": [str(p) for p in extra_dataset_dirs],
            "skip_ids_count": len(skip_ids) if skip_ids else 0,
            "temperature": config.TEMPERATURE,
            "top_p": config.TOP_P,
            "seed": config.SEED,
            "decoding_mode": "temperature_0_top_p_1_seed_42",
            "max_tokens": effective_max_output_tokens,
            "max_output_tokens_omitted": effective_max_output_tokens is None,
            "truncation_retry_max": getattr(config, "TRUNCATION_RETRY_MAX", None),
            "truncation_token_threshold": getattr(config, "TRUNCATION_TOKEN_THRESHOLD", None),
            "partial": is_partial,
            "stop_reason": stop_reason,
            "n_complete": n_complete,
            "n_events_total": n_total_events,
            "non_groq_delay_seconds": None if is_groq else non_groq_delay_seconds(),
        },
        extra={
            "output_json_contains_prompt_template": True,
            "output_json_contains_full_prompt_example": True,
            "prompt_suffix_policy": prompt_suffix_policy,
            "dataset_roots": [str(p) for p in dataset_roots],
        },
    )
    flush_langfuse()

    if verbose:
        status = "COMPLETE" if not is_partial else f"partial ({n_complete}/{n_total_events} complete)"
        print(f"  Saved: {out_path}  [{status}]")
        print(f"  Manifest: {manifest_path}")

    return out_path


if __name__ == "__main__":
    # Quick smoke-test: one model, 1 burst per species
    out = run_prototype_inference(
        model_name="gemini-3-1-flash-lite",
        per_species_limit=1,
    )
    print(f"Done: {out}")
