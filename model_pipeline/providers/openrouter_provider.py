"""OpenRouter provider — Qwen VL / Kimi / open-weights VLMs via OpenRouter.

OpenRouter aggregates many VLM providers behind one OpenAI-compatible API.
Qwen3-VL-30B-A3B-Instruct supports 10+ images per request and costs
~$0.13/$0.52 per M tokens — well inside our budget.

Key implementation notes:

- Prototype Qwen prompt-iteration images are resized to 1280 px long edge
  before base64 encoding. This is not a hard OpenRouter API cap. It is a
  deliberate Qwen-route experimental control: it keeps request payload,
  visual-token cost, and latency low while preserving comparability with
  the existing qwen01-qwen14/BP rows. Native-resolution Qwen runs must be
  treated as a separate image-transport condition and re-baselined.
- response_format={"type":"json_object"} is forwarded; not all upstream
  partners enforce strictly but Qwen3-VL is good about it.
- HTTP-Referer + X-Title headers are OpenRouter's attribution convention.
"""
from __future__ import annotations
import os
import io
import time
import json
import base64
from email.utils import parsedate_to_datetime
import requests
from PIL import Image
from dotenv import load_dotenv, find_dotenv

import config

load_dotenv(find_dotenv())

# Qwen/Alibaba expose model-side pixel budgets, and downstream providers may
# downscale larger images anyway. We pre-resize this prototype route to keep
# Qwen prompt rows cheap, fast, and comparable; this is not an OpenRouter hard
# rejection limit. Final/native Qwen runs need a separate explicit transport.
_OPENROUTER_MAX_EDGE_PX = 1280
_OPENROUTER_JPEG_QUALITY = 90
_OPENROUTER_MAX_ATTEMPTS = 4
_OPENROUTER_RETRYABLE_STATUS = {429, 500, 502, 503, 504, 520}
_OPENROUTER_RETRY_BASE_SECONDS = 5
_OPENROUTER_RETRY_MAX_SECONDS = 30


def _retry_delay_seconds(response, attempt_number: int) -> float:
    """Return Retry-After if supplied, otherwise bounded exponential backoff."""
    retry_after = response.headers.get("retry-after") if response is not None else None
    if retry_after:
        try:
            return min(float(retry_after), _OPENROUTER_RETRY_MAX_SECONDS)
        except ValueError:
            try:
                dt = parsedate_to_datetime(retry_after)
                return min(max(dt.timestamp() - time.time(), 0), _OPENROUTER_RETRY_MAX_SECONDS)
            except Exception:
                pass
    return min(
        _OPENROUTER_RETRY_BASE_SECONDS * (2 ** max(attempt_number - 1, 0)),
        _OPENROUTER_RETRY_MAX_SECONDS,
    )


def _encode_image_for_openrouter(path):
    """Resize to 1280 long-edge JPEG q=90, return base64 string."""
    with Image.open(path) as im_src:
        im = im_src.convert("RGB")
    max_edge = max(im.size)
    if max_edge > _OPENROUTER_MAX_EDGE_PX:
        scale = _OPENROUTER_MAX_EDGE_PX / float(max_edge)
        new_size = (int(im.size[0] * scale), int(im.size[1] * scale))
        im = im.resize(new_size, Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=_OPENROUTER_JPEG_QUALITY, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class OpenRouterProvider:
    def __init__(self, model_key):
        self.model_config = config.MODELS.get(model_key)
        if not self.model_config:
            raise ValueError(f"Model key '{model_key}' not found in config.py")
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found in .env or environment")
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.model_id = self.model_config["api_model"]

    def generate_response(self, prompt, image_paths):
        content_parts = [{"type": "text", "text": prompt}]
        for path in image_paths:
            encoded = _encode_image_for_openrouter(path)
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
            })

        payload = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": content_parts}],
            "temperature": config.TEMPERATURE,
            "top_p": config.TOP_P,
            "max_tokens": config.MAX_TOKENS,
            "response_format": {"type": "json_object"},
        }

        # Optional per-model upstream-provider pin. Some models (e.g. Mistral
        # Small 3.2 via OpenRouter) have uneven quality across upstream routes —
        # Parasail returned garbage tokens, Mistral-direct + Venice returned 400,
        # DeepInfra returned clean JSON. Config entry sets the pin:
        #   "openrouter_provider_pin": ["DeepInfra"]
        pin = self.model_config.get("openrouter_provider_pin")
        if pin:
            payload["provider"] = {"order": pin, "allow_fallbacks": False}
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://wildlifebench.local",
            "X-Title": "WildlifeBench-TrackB",
        }

        start_time = time.time()
        last_error = None
        for attempt in range(1, _OPENROUTER_MAX_ATTEMPTS + 1):
            try:
                response = requests.post(self.api_url, json=payload, headers=headers, timeout=180)
                if (
                    response.status_code in _OPENROUTER_RETRYABLE_STATUS
                    and attempt < _OPENROUTER_MAX_ATTEMPTS
                ):
                    time.sleep(_retry_delay_seconds(response, attempt))
                    continue
                response.raise_for_status()
                res_json = response.json()
                break
            except Exception as e:
                last_error = e
                status_code = getattr(getattr(e, "response", None), "status_code", None)
                if status_code in _OPENROUTER_RETRYABLE_STATUS and attempt < _OPENROUTER_MAX_ATTEMPTS:
                    time.sleep(_retry_delay_seconds(getattr(e, "response", None), attempt))
                    continue
                return self._error_response(e, start_time, attempt)
        else:
            return self._error_response(last_error, start_time, _OPENROUTER_MAX_ATTEMPTS)

        try:
            latency = time.time() - start_time

            usage = res_json.get("usage", {}) or {}
            telemetry = {
                "model": self.model_id,
                "provider": "openrouter",
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
                "upstream_provider": res_json.get("provider", ""),
                "openrouter_request_id": response.headers.get("x-request-id", ""),
                "openrouter_attempts": attempt,
            }

            raw_text = res_json["choices"][0]["message"]["content"] or ""
            try:
                parsed_json = json.loads(raw_text)
            except json.JSONDecodeError:
                parsed_json = {"error": "JSON parse error", "raw": raw_text}

            return {
                "parsed_json": parsed_json,
                "raw_response": raw_text,
                "latency_seconds": round(latency, 2),
                "telemetry": telemetry,
                "input_tokens": telemetry["prompt_tokens"],
                "output_tokens": telemetry["completion_tokens"],
            }

        except Exception as e:
            return self._error_response(e, start_time, attempt)

    def _error_response(self, error, start_time, attempts):
        # Capture response body for HTTP errors (Together/OpenRouter error
        # bodies are more useful than the bare exception string).
        error_str = str(error)
        try:
            if hasattr(error, "response") and error.response is not None:
                error_str = f"{error_str} | body: {error.response.text[:400]}"
        except Exception:
            pass
        return {
            "parsed_json": None,
            "raw_response": None,
            "error": error_str,
            "latency_seconds": round(time.time() - start_time, 2),
            "telemetry": {
                "model": self.model_id,
                "provider": "openrouter",
                "error": error_str,
                "openrouter_attempts": attempts,
            },
        }
