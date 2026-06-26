import os
import time
import json
import base64
import requests
from dotenv import load_dotenv, find_dotenv

import config

load_dotenv(find_dotenv())


class GroqProvider:
    def __init__(self, model_key):
        self.model_config = config.MODELS.get(model_key)
        if not self.model_config:
            raise ValueError(f"Model key '{model_key}' not found in config.py")

        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in .env or environment")

        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        self.model_id = self.model_config["api_model"]

    def generate_response(self, prompt, image_paths):
        """
        Send a fully-filled prompt + images to Llama 4 Scout via Groq.

        `prompt` is the complete, pre-formatted string from config.PROMPT_TEMPLATE.
        Everything goes in the user role so all three council models receive
        identical content.
        """
        content_parts = [{"type": "text", "text": prompt}]

        for path in image_paths:
            with open(path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
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

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        start_time = time.time()
        try:
            response = requests.post(self.api_url, json=payload, headers=headers)
            response.raise_for_status()
            res_json = response.json()
            latency = time.time() - start_time

            telemetry = {
                "model": self.model_id,
                "provider": "groq",
                "prompt_tokens": res_json.get("usage", {}).get("prompt_tokens", 0),
                "completion_tokens": res_json.get("usage", {}).get("completion_tokens", 0),
                "total_tokens": res_json.get("usage", {}).get("total_tokens", 0),
                "queue_time_s": res_json.get("usage", {}).get("queue_time", 0.0),
                "prompt_time_s": res_json.get("usage", {}).get("prompt_time", 0.0),
                "completion_time_s": res_json.get("usage", {}).get("completion_time", 0.0),
                "total_time_s": res_json.get("usage", {}).get("total_time", 0.0),
                "system_fingerprint": res_json.get("system_fingerprint", ""),
                "groq_request_id": response.headers.get("x-request-id", ""),
            }

            raw_text = res_json["choices"][0]["message"]["content"]
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
                "telemetry": {"model": self.model_id, "provider": "groq", "error": str(e)},
            }
