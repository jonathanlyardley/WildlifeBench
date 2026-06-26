import os
import re
import time
import json
import base64
from pathlib import Path
from PIL import Image
import requests
from google import genai
from google.genai import types
from dotenv import load_dotenv, find_dotenv

import config

load_dotenv(find_dotenv())

# Langfuse @observe — graceful no-op if package missing or LANGFUSE_PUBLIC_KEY
# unset. Decoration is additive: zero behaviour change for existing callers.
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


class GoogleProvider:
    def __init__(self, model_key):
        self.model_config = config.MODELS.get(model_key)
        if not self.model_config:
            raise ValueError(f"Model key '{model_key}' not found in config.py")

        # Two backends supported:
        # (a) AI Studio (default) — uses GOOGLE_API_KEY; observed Gemini 3.1 Pro
        #     rate limit ~150 RPD on new accounts (Pro is paid-only since
        #     2026-04-01).
        # (b) Vertex AI — uses GOOGLE_CLOUD_PROJECT + GOOGLE_CLOUD_LOCATION via
        #     Application Default Credentials (run `gcloud auth application-
        #     default login` once). Same Gemini models, same $260 trial credits,
        #     but ~1,000 RPD default for Pro and ~15× faster latency. Recommended
        #     for full-prototype + full-final runs that can't fit in 150 RPD
        #     before credits expire.
        # Selection logic: if GOOGLE_CLOUD_PROJECT is set, use Vertex for Gemini
        # models; else fall back to AI Studio. A one-run backend override can
        # force either AI Studio or Vertex REST, useful when an evidence row must
        # prove it used a specific Google route. Gemma models ALWAYS use AI Studio
        # because Vertex's `aiplatform.googleapis.com` endpoint serves only
        # Google's commercial Gemini family — Gemma (open-weights) returns 404
        # on `publishers/google/models/gemma-3-27b-it` (verified 2026-05-06 on
        # gen-lang-client-* project). Self-hosted Gemma deployments via Vertex
        # Model Garden are a separate flow not used by this provider.
        is_gemma_model = self.model_config["api_model"].startswith("gemma")
        backend_override = (os.getenv("WILDLIFEBENCH_GOOGLE_BACKEND") or "").strip().lower()
        backend_override = backend_override.replace("_", "-")
        force_vertex_rest = backend_override in {"vertex-rest", "vertex-rest-api-key"}
        force_vertex_adc = backend_override in {"vertex", "vertex-ai", "vertex-adc"}
        if backend_override not in {
            "",
            "ai-studio",
            "vertex",
            "vertex-ai",
            "vertex-adc",
            "vertex-rest",
            "vertex-rest-api-key",
        }:
            raise ValueError(
                "Unsupported WILDLIFEBENCH_GOOGLE_BACKEND value "
                f"{backend_override!r}; expected 'ai-studio', 'vertex-ai', "
                "'vertex-rest-api-key', or unset"
            )
        force_ai_studio = backend_override == "ai-studio"
        if is_gemma_model and (force_vertex_rest or force_vertex_adc):
            raise ValueError(
                "Gemma models cannot use Vertex in this provider; use ai-studio "
                "or a Gemini model key."
            )
        if (force_vertex_rest or force_vertex_adc) and not os.getenv("GOOGLE_CLOUD_PROJECT"):
            raise ValueError(
                "WILDLIFEBENCH_GOOGLE_BACKEND requested Vertex but "
                "GOOGLE_CLOUD_PROJECT is not set."
            )
        if force_vertex_rest and not os.getenv("GOOGLE_CLOUD_AGENT_KEY"):
            raise ValueError(
                "WILDLIFEBENCH_GOOGLE_BACKEND=vertex-rest-api-key requires "
                "GOOGLE_CLOUD_AGENT_KEY."
            )
        self.use_vertex = (
            (bool(os.getenv("GOOGLE_CLOUD_PROJECT")) or force_vertex_rest or force_vertex_adc)
            and not is_gemma_model
            and not force_ai_studio
        )

        if self.use_vertex:
            self.api_key = os.getenv("GOOGLE_CLOUD_AGENT_KEY")
            if force_vertex_adc:
                self.use_vertex_rest = False
            elif force_vertex_rest:
                self.use_vertex_rest = True
            else:
                self.use_vertex_rest = bool(self.api_key)
            self.gcp_project = os.getenv("GOOGLE_CLOUD_PROJECT")
            # Default location = "global" because Gemini 3.x preview models
            # are only published to the global publisher endpoint as of
            # 2026-05; all regional locations (us-central1, us-east5,
            # europe-west4, etc.) 404 for `gemini-3.1-pro-preview`. Verified
            # 2026-05-05 via 8-region probe. Regional endpoints work for
            # older Gemini 2.5 family and GA-released models — override via
            # GOOGLE_CLOUD_LOCATION if you switch model families.
            self.gcp_location = os.getenv("GOOGLE_CLOUD_LOCATION", "global")
        else:
            self.use_vertex_rest = False
            self.api_key = os.getenv("GOOGLE_API_KEY")
            if not self.api_key:
                raise ValueError(
                    "Neither GOOGLE_CLOUD_PROJECT (Vertex AI) nor "
                    "GOOGLE_API_KEY (AI Studio) set in environment"
                )
            self.gcp_project = None
            self.gcp_location = None

        # Per-call timeout (120s). Without this, the SDK retries 503s silently
        # with exponential backoff and a single call can hang for tens of
        # minutes. Observed 2026-04-17 evening when Google's Gemma serving tier
        # had transient 503s - our outer retry loop never got a chance to run
        # because the SDK call never returned. 120s is generous for even slow
        # Gemma-4-31b (41s/call baseline, max observed ~69s) while still
        # bailing fast on a genuine server stall.
        if self.use_vertex_rest:
            self.client = None
        elif self.use_vertex:
            self.client = genai.Client(
                vertexai=True,
                project=self.gcp_project,
                location=self.gcp_location,
                http_options=types.HttpOptions(timeout=300_000),  # 5 min — Gemini 3 Pro on Vertex/AI-Studio multi-image bursts can spike to 100-200s under load; was 120s but produced 499 CANCELLED on slow calls (verified 2026-05-06)
            )
        else:
            self.client = genai.Client(
                api_key=self.api_key,
                http_options=types.HttpOptions(timeout=300_000),  # 5 min — Gemini 3 Pro on Vertex/AI-Studio multi-image bursts can spike to 100-200s under load; was 120s but produced 499 CANCELLED on slow calls (verified 2026-05-06)
            )
        self.model_id = self.model_config["api_model"]
        self.is_gemma = self.model_id.startswith("gemma")

    @staticmethod
    def _mime_type(path):
        suffix = Path(path).suffix.lower()
        if suffix in {".jpg", ".jpeg"}:
            return "image/jpeg"
        if suffix == ".png":
            return "image/png"
        if suffix == ".webp":
            return "image/webp"
        return "application/octet-stream"

    def _build_vertex_rest_contents(self, prompt, image_paths):
        parts = []
        for path in image_paths:
            with open(path, "rb") as fh:
                parts.append({
                    "inlineData": {
                        "mimeType": self._mime_type(path),
                        "data": base64.b64encode(fh.read()).decode("ascii"),
                    }
                })
        parts.append({"text": prompt})
        return [{"role": "user", "parts": parts}]

    @staticmethod
    def _vertex_rest_generation_config(gen_kwargs):
        key_map = {
            "temperature": "temperature",
            "top_p": "topP",
            "seed": "seed",
            "max_output_tokens": "maxOutputTokens",
            "response_mime_type": "responseMimeType",
            "candidate_count": "candidateCount",
        }
        return {
            rest_key: gen_kwargs[src_key]
            for src_key, rest_key in key_map.items()
            if src_key in gen_kwargs
        }

    @staticmethod
    def _vertex_rest_safety_settings(gen_kwargs):
        settings = gen_kwargs.get("safety_settings", [])
        return [
            {"category": setting.category, "threshold": setting.threshold}
            for setting in settings
        ]

    def _generate_content_vertex_rest(self, contents, gen_kwargs):
        url = (
            "https://aiplatform.googleapis.com/v1/"
            f"projects/{self.gcp_project}/locations/{self.gcp_location}/"
            f"publishers/google/models/{self.model_id}:generateContent"
        )
        body = {
            "contents": contents,
            "generationConfig": self._vertex_rest_generation_config(gen_kwargs),
            "safetySettings": self._vertex_rest_safety_settings(gen_kwargs),
        }
        response = requests.post(
            url,
            params={"key": self.api_key},
            json=body,
            timeout=300,
        )
        try:
            data = response.json()
        except ValueError as exc:
            raise RuntimeError(
                f"Vertex REST returned non-JSON response ({response.status_code})"
            ) from exc
        if not response.ok:
            message = data.get("error", {}).get("message") or json.dumps(data)[:1000]
            raise RuntimeError(
                f"Vertex REST generateContent failed ({response.status_code}): {message}"
            )

        raw_text = ""
        candidates = data.get("candidates") or []
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", []) or []
            raw_text = "".join(
                part.get("text", "")
                for part in parts
                if not part.get("thought", False) and part.get("text")
            ).strip()

        usage = data.get("usageMetadata") or {}
        return {
            "raw_text": raw_text,
            "input_tokens": usage.get("promptTokenCount", 0),
            "output_tokens": usage.get("candidatesTokenCount", 0),
        }

    @_lf_observe(name="google_provider.generate_response", as_type="generation")
    def generate_response(self, prompt, image_paths):
        """
        Send a fully-filled prompt + images to Gemini or Gemma.

        `prompt` is the complete, pre-formatted string from config.PROMPT_TEMPLATE.
        No separate system_instruction is used — the same content reaches every
        model in the same way, making cross-model comparisons fair.

        Auto-traced via Langfuse `@observe` (no-ops if Langfuse not configured —
        existing callers see zero behaviour change). Each call appears as a
        `generation` in the Langfuse dashboard with model, backend, token counts,
        and latency captured automatically. Custom metadata (model_id, backend,
        gcp_location, n_frames, prompt_version, retries) attached below.

        Transport:
          - Gemini models: PIL.Image.open(path) (SDK-native inline path)
          - Gemma models:  types.Part.from_bytes(raw JPEG bytes) - inline raw
            bytes via the SDK Part API.  Verified bit-identical to the old
            client.files.upload() path for scored fields (species/genus/
            family/order/class) on 2026-04-17.  Switched to avoid the ~10 s
            file-upload overhead per frame (~50 s per burst).
        """
        if self.use_vertex_rest:
            contents = self._build_vertex_rest_contents(prompt, image_paths)
        else:
            contents = []

            for path in image_paths:
                if self.is_gemma:
                    # Inline raw JPEG bytes - same bytes as client.files.upload()
                    # would send, just packaged in the request body instead of a
                    # separate upload round-trip.
                    with open(path, "rb") as fh:
                        contents.append(types.Part.from_bytes(
                            data=fh.read(), mime_type="image/jpeg"))
                else:
                    contents.append(Image.open(path))

            contents.append(prompt)

        # Gemini supports response_mime_type for guaranteed JSON output;
        # Gemma does not — omit it to avoid 400 errors.
        # Gemma 4 entries with supports_json_mime=True are allowed below.
        # Some models (e.g. gemini-3.1-pro-preview) truncate to ~7 tokens with
        # this flag — they should set no_json_mime=True in config.MODELS.
        gen_kwargs = dict(
            temperature=config.TEMPERATURE,
            top_p=config.TOP_P,
            seed=config.SEED,
        )
        if not self.model_config.get("no_max_tokens"):
            # Per-model override (e.g. Flash-Preview needs more headroom for
            # extended thinking under reasoning-first schema). Defaults to
            # config.MAX_TOKENS = 1000 unless model entry sets max_output_tokens.
            gen_kwargs["max_output_tokens"] = self.model_config.get(
                "max_output_tokens", config.MAX_TOKENS
            )
        supports_json_mime = (
            not self.is_gemma or self.model_config.get("supports_json_mime", False)
        )
        if supports_json_mime and not self.model_config.get("no_json_mime"):
            gen_kwargs["response_mime_type"] = "application/json"
            gen_kwargs["candidate_count"] = 1

        # Disable safety filters for research benchmarking
        safety_settings = [
            types.SafetySetting(category=cat, threshold="BLOCK_NONE")
            for cat in [
                "HARM_CATEGORY_HATE_SPEECH",
                "HARM_CATEGORY_HARASSMENT",
                "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "HARM_CATEGORY_DANGEROUS_CONTENT",
            ]
        ]
        gen_kwargs["safety_settings"] = safety_settings

        gen_config = types.GenerateContentConfig(**gen_kwargs)

        # Truncation retry: some models (e.g. gemini-3.1-pro-preview with
        # no_json_mime=True) occasionally cut responses at ~40 tokens mid-JSON.
        # Retry when output_tokens < threshold AND JSON parsing failed.
        max_retries = getattr(config, "TRUNCATION_RETRY_MAX", 2)
        trunc_threshold = getattr(config, "TRUNCATION_TOKEN_THRESHOLD", 60)

        total_start = time.time()
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                call_start = time.time()
                if self.use_vertex_rest:
                    response = self._generate_content_vertex_rest(contents, gen_kwargs)
                else:
                    response = self.client.models.generate_content(
                        model=self.model_id,
                        contents=contents,
                        config=gen_config,
                    )
                latency = time.time() - call_start

                # Extract text from response.  IMPORTANT: Gemma occasionally
                # emits a 2-part response: a first `Part(text=..., thought=True)`
                # containing chain-of-thought reasoning, and a second
                # `Part(text=...)` containing the actual answer (JSON in our
                # case).  The SDK's `response.text` accessor returns empty
                # string in this scenario.  We observed this 2026-04-17 on
                # burst `branta_canadensis_041` — a tricky image with a
                # Canada Goose decoy + real partridge in background — and it
                # silently hung v4 at event 30.  Fix: walk all parts, skip
                # any marked `thought=True`, concatenate the rest.
                if self.use_vertex_rest:
                    raw_text = response["raw_text"]
                    input_tokens = response["input_tokens"]
                    output_tokens = response["output_tokens"]
                else:
                    raw_text = ""
                    try:
                        if hasattr(response, "candidates") and response.candidates:
                            cand = response.candidates[0]
                            if cand.content and cand.content.parts:
                                answer_parts = [
                                    (p.text or "") for p in cand.content.parts
                                    if not getattr(p, "thought", False) and getattr(p, "text", None)
                                ]
                                raw_text = "".join(answer_parts).strip()
                        # Fallback: SDK shortcut (only reliable when there's no thought part)
                        if not raw_text:
                            try:
                                raw_text = (response.text or "").strip()
                            except Exception:
                                raw_text = ""
                    except Exception:
                        raw_text = ""

                    output_tokens = (
                        response.usage_metadata.candidates_token_count
                        if response.usage_metadata else 0
                    ) or 0
                    input_tokens = (
                        response.usage_metadata.prompt_token_count
                        if response.usage_metadata else 0
                    ) or 0

                # 3-level JSON parsing
                parsed_json = None
                if raw_text:
                    try:
                        parsed_json = json.loads(raw_text)
                    except json.JSONDecodeError:
                        # Level 2: strip markdown fences
                        for fence in ("```json", "```"):
                            if fence in raw_text:
                                inner = raw_text.split(fence)[1].split("```")[0].strip()
                                try:
                                    parsed_json = json.loads(inner)
                                    break
                                except json.JSONDecodeError:
                                    pass
                        # Level 3: regex — find outermost {...} block
                        if parsed_json is None:
                            m = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', raw_text, re.DOTALL)
                            if m:
                                try:
                                    parsed_json = json.loads(m.group())
                                except json.JSONDecodeError:
                                    pass

                        if parsed_json is None:
                            parsed_json = {"error": "JSON parse error", "raw": raw_text}

                # Check if parse succeeded
                parse_failed = (
                    parsed_json is None
                    or (isinstance(parsed_json, dict) and "error" in parsed_json)
                )

                if not parse_failed:
                    # Clean response — done
                    break

                # Only retry if this looks like a truncation (low token count),
                # not a genuine null or safety block
                is_truncation = output_tokens < trunc_threshold
                if is_truncation and attempt < max_retries:
                    time.sleep(1.5)
                    continue  # retry the API call

                # Genuine parse error or max retries exhausted — keep result as-is
                break

            except Exception as e:
                last_error = str(e)
                latency = time.time() - call_start
                parsed_json = None
                raw_text = None
                input_tokens = 0
                output_tokens = 0
                if attempt < max_retries:
                    time.sleep(1.5)
                    continue
                break

        # Build the result dict first so we can attach it to Langfuse, then return
        if last_error and parsed_json is None and raw_text is None:
            result = {
                "parsed_json": None,
                "raw_response": None,
                "error": last_error,
                "latency_seconds": round(time.time() - total_start, 2),
                "input_tokens": 0,
                "output_tokens": 0,
                "retries": attempt,
            }
        else:
            result = {
                "parsed_json": parsed_json,
                "raw_response": raw_text,
                "latency_seconds": round(time.time() - total_start, 2),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "retries": attempt,  # 0 = no retry needed; >0 = truncation recovery
            }

        # Attach Langfuse metadata + usage if Langfuse is active. No-op if not.
        if _LANGFUSE_AVAILABLE:
            try:
                lf = _lf_get_client()
                if lf is not None:
                    lf.update_current_generation(
                        model=self.model_id,
                        input={"prompt": prompt[:2000], "n_images": len(image_paths)},
                        output=result.get("parsed_json") or {"error": result.get("error", "")[:500]},
                        metadata={
                            "backend": (
                                "vertex_rest_api_key"
                                if self.use_vertex_rest
                                else "vertex_ai"
                                if self.use_vertex
                                else "ai_studio"
                            ),
                            "gcp_project": self.gcp_project,
                            "gcp_location": self.gcp_location,
                            "model_id": self.model_id,
                            "is_gemma": self.is_gemma,
                            "n_frames": len(image_paths),
                            "retries": result.get("retries", 0),
                            "prompt_version": getattr(config, "PROMPT_VERSION", "unknown"),
                        },
                        usage_details={
                            "input": result.get("input_tokens", 0) or 0,
                            "output": result.get("output_tokens", 0) or 0,
                        },
                    )
            except Exception:
                # Never let telemetry failure break inference
                pass

        return result
