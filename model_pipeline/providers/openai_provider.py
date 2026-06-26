"""OpenAI provider — serves GPT-4o-mini / GPT-5-mini / GPT-5-nano and
similar via the OpenAI Chat Completions API.  Uses the same OpenAI-compatible
schema as the Groq provider, with two GPT-5-family quirks:

1. GPT-5 models use `max_completion_tokens` (NOT `max_tokens`).  Using
   `max_tokens` triggers an API error on GPT-5-family models.
2. GPT-5 family burns internal "reasoning" tokens that count against
   `max_completion_tokens` before the visible answer is emitted.  Cap
   too low and the model returns null (reasoning hit the cap before the
   JSON answer).  We use a generous cap (config.MAX_COMPLETION_TOKENS,
   default 4000) to give it room.

Image payloads use OpenAI's object form with `detail: "high"` so the
vision encoder uses maximum fidelity on our camera-trap frames.
"""
from __future__ import annotations
import os
import time
import json
import base64
import requests
from dotenv import load_dotenv, find_dotenv

import config

load_dotenv(find_dotenv())


class OpenAIProvider:
    def __init__(self, model_key):
        self.model_config = config.MODELS.get(model_key)
        if not self.model_config:
            raise ValueError(f"Model key '{model_key}' not found in config.py")

        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in .env or environment")

        self.api_url = "https://api.openai.com/v1/chat/completions"
        self.model_id = self.model_config["api_model"]

        # GPT-5 family (nano/mini/main) and o-series use reasoning tokens.
        # Others (GPT-4o, GPT-4.1) use max_tokens.
        self.is_reasoning_model = any(
            self.model_id.startswith(p) for p in ("gpt-5", "o1", "o3", "o4")
        )

    def generate_response(self, prompt, image_paths):
        """Send prompt + images to OpenAI chat completions.

        `prompt` is the complete, pre-formatted string from config.PROMPT_TEMPLATE.
        """
        content_parts = [{"type": "text", "text": prompt}]
        for path in image_paths:
            with open(path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
            content_parts.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded}",
                    "detail": "high",
                },
            })

        payload = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": content_parts}],
            "response_format": {"type": "json_object"},
        }

        if self.is_reasoning_model:
            # GPT-5 family: need big completion budget for internal reasoning +
            # output.  Don't pass temperature/top_p; reasoning models ignore or
            # reject them (GPT-5 uses reasoning_effort instead, defaults fine).
            payload["max_completion_tokens"] = getattr(
                config, "MAX_COMPLETION_TOKENS", 4000
            )
        else:
            # GPT-4o / GPT-4.1 family: standard OpenAI chat completions knobs
            payload["max_tokens"] = config.MAX_TOKENS
            payload["temperature"] = config.TEMPERATURE
            payload["top_p"] = config.TOP_P

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        start_time = time.time()
        try:
            # Generous timeout — GPT-5-nano with reasoning can take 15-25s/call
            response = requests.post(self.api_url, json=payload, headers=headers, timeout=120)
            response.raise_for_status()
            res_json = response.json()
            latency = time.time() - start_time

            usage = res_json.get("usage", {}) or {}
            telemetry = {
                "model": self.model_id,
                "provider": "openai",
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
                "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0),
                "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
                "system_fingerprint": res_json.get("system_fingerprint", ""),
                "openai_request_id": response.headers.get("x-request-id", ""),
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
            return {
                "parsed_json": None,
                "raw_response": None,
                "error": str(e),
                "latency_seconds": round(time.time() - start_time, 2),
                "telemetry": {"model": self.model_id, "provider": "openai", "error": str(e)},
            }
