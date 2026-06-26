# model_pipeline - exact run code

This folder contains the exact Python code used to produce the model outputs in the
GBIF Ebbe Nielsen 2026 award comparison. It is provided for audit and reproduction.
See `../REPRODUCE.md` for end-to-end instructions and `../PROMPTS.md` for the locked
prompt and its SHA-256 verification.

## Contents

| File | Role |
|---|---|
| `config.py` | Single source of truth: prompt template, prompt version, family abstention suffixes, model definitions. The `PROMPT_TEMPLATE` and `GEMINI_ABSTENTION_SUFFIX` reproduce the SHA-256 values in `../PROMPTS.md`. |
| `run_prototype_inference.py` | Vision-language model runner (Gemini, Qwen). Writes a run manifest and cached-output JSON per run. |
| `run_bioclip_benchmark.py` | BioCLIP 2.5 local runner (CLIP-style open foundation model over a GBIF-derived geo candidate-label set). |
| `bioclip_worker_20260626_005320.py` | The exact generated BioCLIP worker for the released run (`run_id` timestamp `20260626_005320`). |
| `run_speciesnet_benchmark.py` | SpeciesNet v4.0.3b local-CLI runner with geofence + ensemble. |
| `run_manifest.py` | Builds the per-run audit manifests (model, route, prompt SHA, decoding, usage, cost, Langfuse trace). |
| `providers/` | API provider adapters: `google_provider.py`, `openrouter_provider.py`, `openai_provider.py`, `groq_provider.py`. |

## Credentials

No credentials are stored in this repository. The provider adapters read keys from a
local `.env` via `python-dotenv`:

- `GOOGLE_API_KEY` (or the Vertex variables `GOOGLE_CLOUD_PROJECT`,
  `GOOGLE_CLOUD_LOCATION`, `GOOGLE_CLOUD_AGENT_KEY`) for Gemini routes.
- `OPENROUTER_API_KEY` for the Qwen route.

BioCLIP and SpeciesNet are local models and need no API key.

## Import note

These modules import each other by top-level name (`import config`,
`from providers... import ...`, `from run_manifest import ...`). Run them with
`model_pipeline/` as the working directory, e.g. `cd model_pipeline` first, or add it
to `PYTHONPATH`. The runner stack also depends on the wider WildlifeBench data
loaders and scoring helpers from the private working repository; this folder is the
model-facing core of that stack, sufficient to read and audit exactly how each model
was called.
