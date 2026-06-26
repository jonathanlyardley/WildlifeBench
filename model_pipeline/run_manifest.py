"""Run manifest helpers for WildlifeBench Track B.

The manifest is the durable audit record for a benchmark run. Hosted tracing
systems such as Langfuse are useful cross-references, but every publishable
number should also point to a local JSON file with enough metadata to reproduce
or challenge the run.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "wildlifebench.run_manifest.v1"
REPO_ROOT = Path(__file__).resolve().parent


def utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp with a stable trailing Z."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stable_json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def repo_relative(path: str | Path) -> str:
    p = Path(path)
    if not p.is_absolute():
        p = (REPO_ROOT / p).resolve()
    try:
        return p.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(p)


def file_record(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    exists = p.exists()
    record: dict[str, Any] = {
        "path": repo_relative(p),
        "exists": exists,
    }
    if exists and p.is_file():
        stat = p.stat()
        record.update({
            "bytes": stat.st_size,
            "sha256": sha256_file(p),
        })
    return record


def _run_git(args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def git_state() -> dict[str, Any]:
    status = _run_git(["status", "--short"]) or ""
    return {
        "commit": _run_git(["rev-parse", "HEAD"]),
        "branch": _run_git(["branch", "--show-current"]),
        "dirty": bool(status),
        "status_short": status.splitlines(),
    }


def package_version(distribution_name: str) -> str | None:
    try:
        return importlib_metadata.version(distribution_name)
    except importlib_metadata.PackageNotFoundError:
        return None


def runtime_versions(extra_packages: Iterable[str] = ()) -> dict[str, Any]:
    package_names = [
        "google-genai",
        "inspect-ai",
        "langfuse",
        "requests",
        "Pillow",
        "numpy",
        "scipy",
        *extra_packages,
    ]
    return {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": {
            name: package_version(name)
            for name in package_names
        },
    }


def uv_lock_record() -> dict[str, Any]:
    return file_record(REPO_ROOT / "uv.lock")


def dataset_fingerprint_from_events(events: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    n_frames = 0
    for event in events:
        frames = []
        for frame in event.get("frames", []):
            frame_path = Path(frame)
            if not frame_path.is_absolute():
                frame_path = REPO_ROOT / frame_path
            frame_exists = frame_path.exists()
            frames.append({
                "path": repo_relative(frame_path),
                "exists": frame_exists,
                "bytes": frame_path.stat().st_size if frame_exists and frame_path.is_file() else None,
            })
        n_frames += len(frames)
        rows.append({
            "event_id": event.get("event_id"),
            "species": event.get("species"),
            "country": event.get("country"),
            "difficulty": event.get("difficulty"),
            "frames": frames,
        })
    rows.sort(key=lambda row: row.get("event_id") or "")
    return {
        "n_events": len(rows),
        "n_frames": n_frames,
        "fingerprint_sha256": sha256_text(stable_json_dumps(rows)),
    }


def token_usage_from_results(results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    input_tokens = 0
    output_tokens = 0
    total_tokens = 0
    request_ids: list[dict[str, str]] = []
    local_cost_values: list[float] = []

    for row in results:
        tokens = row.get("tokens") or {}
        in_tok = tokens.get("input") or 0
        out_tok = tokens.get("output") or 0
        input_tokens += in_tok
        output_tokens += out_tok
        total_tokens += in_tok + out_tok

        telemetry = row.get("telemetry") or {}
        for key, value in telemetry.items():
            if key.endswith("_request_id") and value:
                request_ids.append({"type": key, "value": str(value)})
            if key in {"cost_usd", "total_cost_usd", "calculated_total_cost_usd"}:
                try:
                    local_cost_values.append(float(value))
                except (TypeError, ValueError):
                    pass

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "provider_request_ids": request_ids,
        "local_cost_usd": round(sum(local_cost_values), 6) if local_cost_values else None,
    }


def langfuse_trace_reference() -> dict[str, Any]:
    try:
        from langfuse import get_client
    except Exception:
        return {"enabled": False, "trace_id": None, "trace_url": None}

    try:
        client = get_client()
        trace_id = client.get_current_trace_id()
        trace_url = client.get_trace_url(trace_id=trace_id) if trace_id else None
        return {
            "enabled": bool(trace_id),
            "trace_id": trace_id,
            "trace_url": trace_url,
        }
    except Exception:
        return {"enabled": False, "trace_id": None, "trace_url": None}


def flush_langfuse() -> None:
    try:
        from langfuse import get_client

        client = get_client()
        client.flush()
    except Exception:
        pass


def describe_provider_route(provider_obj: Any, model_config: dict[str, Any]) -> dict[str, Any]:
    provider_name = model_config.get("provider")
    api_model = model_config.get("api_model")

    if provider_name == "google":
        use_vertex_rest = bool(getattr(provider_obj, "use_vertex_rest", False))
        use_vertex = bool(getattr(provider_obj, "use_vertex", False))
        backend = "vertex_rest_api_key" if use_vertex_rest else "vertex_ai" if use_vertex else "ai_studio"
        if use_vertex_rest:
            endpoint = "https://aiplatform.googleapis.com/v1/{project}/locations/{location}/publishers/google/models/{model}:generateContent"
        elif use_vertex:
            endpoint = "google-genai SDK Vertex AI generate_content"
        else:
            endpoint = "google-genai SDK AI Studio generate_content"
        return {
            "provider": provider_name,
            "backend": backend,
            "api_model": api_model,
            "endpoint": endpoint,
            "gcp_location": getattr(provider_obj, "gcp_location", None),
            "transport": "inline image bytes/PIL images",
        }

    if provider_name == "openrouter":
        return {
            "provider": provider_name,
            "backend": "openrouter_chat_completions",
            "api_model": api_model,
            "endpoint": getattr(provider_obj, "api_url", "https://openrouter.ai/api/v1/chat/completions"),
            "transport": "OpenAI-compatible chat completions with base64 image URLs",
            "provider_pin": model_config.get("openrouter_provider_pin"),
        }

    if provider_name == "openai":
        return {
            "provider": provider_name,
            "backend": "openai_chat_completions",
            "api_model": api_model,
            "endpoint": getattr(provider_obj, "api_url", "https://api.openai.com/v1/chat/completions"),
            "transport": "OpenAI chat completions with high-detail base64 image URLs",
        }

    if provider_name == "groq":
        return {
            "provider": provider_name,
            "backend": "groq_openai_compatible",
            "api_model": api_model,
            "endpoint": getattr(provider_obj, "api_url", "https://api.groq.com/openai/v1/chat/completions"),
            "transport": "OpenAI-compatible chat completions with base64 image URLs",
        }

    return {
        "provider": provider_name,
        "backend": "unknown",
        "api_model": api_model,
        "endpoint": None,
        "transport": None,
    }


def image_resolution_policy(model_config: dict[str, Any], run_type: str, image_source: str | Path | None = None) -> dict[str, Any]:
    provider_name = model_config.get("provider")
    if provider_name == "openrouter":
        return {
            "policy": "1280px long-edge resize before OpenRouter upload",
            "scope": "prototype iteration / OpenRouter Qwen path",
            "details": "providers/openrouter_provider.py resizes to JPEG quality 90 before base64 encoding.",
        }
    if run_type.startswith("speciesnet"):
        return {
            "policy": "native original camera-trap frames",
            "scope": "SpeciesNet specialist baseline",
            "details": "Timestamp overlays retained; SpeciesNet performs its own internal preprocessing.",
            "image_source": repo_relative(image_source) if image_source else None,
        }
    if run_type.startswith("bioclip"):
        return {
            "policy": "native timestamp-cropped runner frames",
            "scope": "BioCLIP specialist comparator",
            "details": "WildlifeBench passes cropped JPEG paths; pybioclip applies its own 224 px model transform.",
            "image_source": repo_relative(image_source) if image_source else None,
        }
    return {
        "policy": "native post-crop resolution",
        "scope": "LLM API call",
        "details": "No pre-resize in WildlifeBench; provider-internal downsampling, if any, is part of the measured deployment.",
    }


def decoding_record(run_parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        "mode": run_parameters.get("decoding_mode") or "not_recorded",
        "temperature": run_parameters.get("temperature"),
        "top_p": run_parameters.get("top_p"),
        "top_k": run_parameters.get("top_k"),
        "max_output_tokens": run_parameters.get("max_output_tokens")
        or run_parameters.get("max_tokens")
        or run_parameters.get("max_completion_tokens"),
        "candidate_count": run_parameters.get("candidate_count")
        or run_parameters.get("n")
        or run_parameters.get("num_generations"),
    }


def _present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (dict, list, tuple, set)):
        return bool(value)
    return True


def _check(present: bool, detail: str, *, required: bool = True) -> dict[str, Any]:
    return {
        "required": required,
        "present": bool(present),
        "detail": detail,
    }


def api_row_reporting_checklist(manifest: dict[str, Any]) -> dict[str, Any]:
    """Return a compact audit checklist for publishable API-system rows."""
    model = manifest.get("model") or {}
    api_route = manifest.get("api_route") or {}
    prompt = manifest.get("prompt") or {}
    dataset = manifest.get("dataset") or {}
    decoding = manifest.get("decoding") or {}
    environment = manifest.get("environment") or {}
    git = environment.get("git") or {}
    raw_files = manifest.get("raw_result_files") or []
    first_raw = raw_files[0] if raw_files else {}
    usage = manifest.get("usage") or {}
    cost = manifest.get("cost") or {}

    checks = {
        "run_id": _check(_present(manifest.get("run_id")), "Stable local run identifier."),
        "timestamps": _check(
            all(_present(manifest.get(key)) for key in ("started_at", "completed_at", "created_at")),
            "Started/completed/manifest-created timestamps are recorded.",
        ),
        "local_result_file_hash": _check(
            bool(first_raw.get("exists") and first_raw.get("sha256")),
            "Local result JSON path, byte size, and SHA-256 are recorded.",
        ),
        "dataset_fingerprint": _check(
            _present((dataset.get("dataset_fingerprint") or {}).get("fingerprint_sha256")),
            "Dataset event/frame fingerprint is recorded.",
        ),
        "model_identifier": _check(
            _present(model.get("model_key")) and _present(model.get("api_model")),
            "Model key and provider API model identifier are recorded.",
        ),
        "model_version_or_alias_pin": _check(
            _present(model.get("provider_version_pin")) or _present(model.get("reported_model_identifier")),
            "Provider-specific immutable pin when available, otherwise the exact reported API model alias.",
        ),
        "provider_route": _check(
            all(_present(api_route.get(key)) for key in ("provider", "backend", "endpoint", "transport")),
            "Provider, backend, endpoint, and image/text transport are recorded.",
        ),
        "prompt_hashes": _check(
            prompt is not None
            and all(
                _present(prompt.get(key))
                for key in (
                    "template_sha256",
                    "full_prompt_example_sha256",
                    "abstention_suffix_sha256",
                )
            ),
            "Prompt template, concrete example, and suffix hashes are recorded.",
        ),
        "generation_parameters": _check(
            _present(manifest.get("run_parameters")),
            "Generation/run parameters are recorded.",
        ),
        "decoding_mode": _check(
            decoding.get("mode") != "not_recorded",
            "Decoding mode is explicitly named rather than inferred later.",
        ),
        "image_policy": _check(
            _present((manifest.get("image_resolution") or {}).get("policy")),
            "Image resolution/preprocessing policy is recorded.",
        ),
        "token_usage": _check(
            any(usage.get(key) is not None for key in ("input_tokens", "output_tokens", "total_tokens")),
            "Token usage is recorded where provider telemetry exposes it.",
            required=False,
        ),
        "cost_slot": _check(
            _present(cost.get("source")),
            "Cost source is recorded, including explicit not-available status.",
            required=False,
        ),
        "trace_reference": _check(
            "langfuse" in (manifest.get("trace") or {}),
            "Langfuse trace reference slot is recorded even when tracing is disabled.",
            required=False,
        ),
        "environment_git": _check(
            _present(git.get("commit")) and _present(git.get("branch")),
            "Git commit and branch are recorded.",
        ),
        "environment_runtime": _check(
            _present((environment.get("runtime") or {}).get("python")),
            "Python/runtime package environment is recorded.",
        ),
        "uv_lock": _check(
            _present((environment.get("uv_lock") or {}).get("path")),
            "uv.lock provenance slot is recorded.",
            required=False,
        ),
        "error_retry_policy": _check(
            _present((manifest.get("extra") or {}).get("retry_policy"))
            or _present((manifest.get("extra") or {}).get("error_policy"))
            or _present((manifest.get("extra") or {}).get("retry_summary")),
            "Retry/error policy or retry summary is recorded when available.",
            required=False,
        ),
    }

    missing_required = [
        name
        for name, item in checks.items()
        if item["required"] and not item["present"]
    ]
    return {
        "schema_version": "wildlifebench.api_row_reporting_checklist.v1",
        "scope": "api_system_row",
        "audit_grade": "audit_grade" if not missing_required else "provisional",
        "missing_required": missing_required,
        "checks": checks,
        "note": "This checklist supports API-row reporting only; agent-system and specialist-baseline rows remain separate measurands.",
    }


def write_manifest(manifest: dict[str, Any], manifest_path: str | Path) -> Path:
    path = Path(manifest_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return path


def write_llm_run_manifest(
    *,
    result_json_path: str | Path,
    model_name: str,
    model_config: dict[str, Any],
    provider_obj: Any,
    dataset_dir: str | Path,
    events: Iterable[dict[str, Any]],
    results: Iterable[dict[str, Any]],
    prompt_version: str,
    prompt_template: str,
    full_prompt_example: str,
    abstention_suffix: str,
    started_at: str,
    completed_at: str,
    run_parameters: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> Path:
    result_path = Path(result_json_path)
    manifest_path = result_path.with_suffix(".run_manifest.json")
    results_list = list(results)
    events_list = list(events)
    run_parameters_payload = dict(run_parameters)
    usage = token_usage_from_results(results_list)
    local_cost = usage.pop("local_cost_usd")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": result_path.stem,
        "run_type": "llm_inference",
        "created_at": utc_now_iso(),
        "started_at": started_at,
        "completed_at": completed_at,
        "model": {
            "model_key": model_name,
            "api_model": model_config.get("api_model"),
            "reported_model_identifier": model_config.get("api_model"),
            "provider_version_pin": model_config.get("api_model_version")
            or model_config.get("version_pin")
            or model_config.get("openrouter_provider_pin"),
            "provider": model_config.get("provider"),
            "family": model_config.get("family"),
            "description": model_config.get("description"),
        },
        "api_route": describe_provider_route(provider_obj, model_config),
        "prompt": {
            "version": prompt_version,
            "template_sha256": sha256_text(prompt_template),
            "full_prompt_example_sha256": sha256_text(full_prompt_example or ""),
            "abstention_suffix_sha256": sha256_text(abstention_suffix or ""),
        },
        "dataset": {
            "dataset_dir": repo_relative(dataset_dir),
            "dataset_fingerprint": dataset_fingerprint_from_events(events_list),
        },
        "image_resolution": image_resolution_policy(model_config, "llm_inference", dataset_dir),
        "decoding": decoding_record(run_parameters_payload),
        "run_parameters": run_parameters_payload,
        "usage": usage,
        "cost": {
            "total_usd": local_cost,
            "source": "provider_telemetry" if local_cost is not None else "not_available_locally",
            "note": "Langfuse may infer cost from trace usage when configured; local manifest does not invent prices.",
        },
        "trace": {
            "langfuse": langfuse_trace_reference(),
        },
        "environment": {
            "git": git_state(),
            "uv_lock": uv_lock_record(),
            "runtime": runtime_versions(),
        },
        "raw_result_files": [
            file_record(result_path),
        ],
        "extra": extra or {},
    }
    manifest["reporting_checklist"] = api_row_reporting_checklist(manifest)
    return write_manifest(manifest, manifest_path)


def write_speciesnet_run_manifest(
    *,
    result_json_path: str | Path,
    raw_output_paths: Iterable[str | Path],
    result: dict[str, Any],
    image_dir: str | Path,
    events: Iterable[dict[str, Any]],
    run_parameters: dict[str, Any],
    started_at: str,
    completed_at: str,
    extra: dict[str, Any] | None = None,
) -> Path:
    result_path = Path(result_json_path)
    manifest_path = result_path.with_suffix(".run_manifest.json")
    raw_records = [file_record(result_path)]
    raw_records.extend(file_record(path) for path in raw_output_paths)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": result_path.stem,
        "run_type": "speciesnet_benchmark",
        "created_at": utc_now_iso(),
        "started_at": started_at,
        "completed_at": completed_at,
        "model": {
            "model_key": "speciesnet",
            "api_model": result.get("model_id") or "installed SpeciesNet default",
            "provider": "local_cli",
            "family": "speciesnet",
        },
        "api_route": {
            "provider": "local_cli",
            "backend": "speciesnet.scripts.run_model",
            "endpoint": None,
            "transport": "file paths passed to local SpeciesNet CLI",
        },
        "prompt": None,
        "dataset": {
            "dataset_dir": repo_relative(image_dir),
            "dataset_fingerprint": dataset_fingerprint_from_events(events),
        },
        "image_resolution": image_resolution_policy({"provider": "speciesnet"}, "speciesnet_benchmark", image_dir),
        "run_parameters": run_parameters,
        "usage": {
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "provider_request_ids": [],
        },
        "cost": {
            "total_usd": 0.0,
            "source": "local_colab_or_local_cli",
            "note": "SpeciesNet run has no LLM API token billing.",
        },
        "trace": {
            "langfuse": langfuse_trace_reference(),
        },
        "environment": {
            "git": git_state(),
            "uv_lock": uv_lock_record(),
            "runtime": runtime_versions(extra_packages=["speciesnet"]),
        },
        "raw_result_files": raw_records,
        "extra": extra or {},
    }
    return write_manifest(manifest, manifest_path)


def write_bioclip_run_manifest(
    *,
    result_json_path: str | Path,
    raw_output_paths: Iterable[str | Path],
    result: dict[str, Any],
    image_dir: str | Path,
    events: Iterable[dict[str, Any]],
    run_parameters: dict[str, Any],
    started_at: str,
    completed_at: str,
    candidate_list_paths: Iterable[str | Path] = (),
    tol_taxa_path: str | Path | None = None,
    coverage_report_paths: Iterable[str | Path] = (),
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write an audit manifest for a local BioCLIP comparator run."""
    result_path = Path(result_json_path)
    manifest_path = result_path.with_suffix(".run_manifest.json")
    raw_records = [file_record(result_path)]
    raw_records.extend(file_record(path) for path in raw_output_paths)
    candidate_records = [file_record(path) for path in candidate_list_paths]
    coverage_records = [file_record(path) for path in coverage_report_paths]

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": result_path.stem,
        "run_type": "bioclip_benchmark",
        "created_at": utc_now_iso(),
        "started_at": started_at,
        "completed_at": completed_at,
        "model": {
            "model_key": result.get("model") or "bioclip",
            "api_model": result.get("model_id") or result.get("model"),
            "reported_model_identifier": result.get("model_id") or result.get("model"),
            "provider": "local_cli",
            "family": "bioclip",
            "description": "Local pybioclip specialist/open-foundation-model comparator.",
        },
        "api_route": {
            "provider": "local_cli",
            "backend": "pybioclip TreeOfLifeClassifier",
            "endpoint": None,
            "transport": "file paths passed to a separate local pybioclip Python runtime",
        },
        "prompt": None,
        "dataset": {
            "dataset_dir": repo_relative(image_dir),
            "dataset_fingerprint": dataset_fingerprint_from_events(events),
        },
        "image_resolution": image_resolution_policy({"provider": "bioclip"}, "bioclip_benchmark", image_dir),
        "run_parameters": run_parameters,
        "usage": {
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "provider_request_ids": [],
        },
        "cost": {
            "total_usd": 0.0,
            "source": "local_cli",
            "note": "BioCLIP run has no LLM API token billing.",
        },
        "trace": {
            "langfuse": langfuse_trace_reference(),
        },
        "environment": {
            "git": git_state(),
            "uv_lock": uv_lock_record(),
            "runtime": runtime_versions(extra_packages=["pybioclip", "open-clip-torch", "torch", "torchvision"]),
        },
        "raw_result_files": raw_records,
        "candidate_list_files": candidate_records,
        "tol_taxa_file": file_record(tol_taxa_path) if tol_taxa_path else None,
        "coverage_report_files": coverage_records,
        "extra": extra or {},
    }
    return write_manifest(manifest, manifest_path)
