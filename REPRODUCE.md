# Reproducing the WildlifeBench GBIF Ebbe Nielsen 2026 run

This guide explains how to (a) re-verify the released numbers with no API cost, and
(b) re-run the models live. It is written for a careful reviewer who is comfortable
with a terminal but not necessarily with the internals of the pipeline.

The award comparison is a single, frozen result set: **246 camera-trap bursts / 82
species** in the SpeciesNet v4.0.3b classifier-label and country-geofence overlap.
All scripts referenced below are in this repository.

## What is included for audit

| Artefact | Path |
|---|---|
| Exact prompt text + SHA-256 | `PROMPTS.md` |
| Per-run manifests (model, route, Langfuse trace, cost, prompt SHA) | `results/run_outputs/run_manifests/` |
| Raw per-event model outputs (246 award events) | `results/run_outputs/cached_outputs/` |
| Headline scored table | `results/speciesnet_overlap/application_safe_results.md` |
| Per-event scored predictions | `results/speciesnet_overlap/application_safe_results_events.csv` |
| Exact run code | `model_pipeline/` |
| Static visual preview | `gbif_award_preview.html` |
| Dataset media (cropped + uncropped JPEGs) | Hugging Face dataset (see README) |

The 84 non-overlap prototype events are deliberately withheld; every released
per-event file is restricted to the 246 award events (see each file's
`release_scope` block).

## Part A - No-cost verification (no credentials, no GPU)

This recomputes the displayed table from the cached, frozen outputs.

1. Obtain the data package (Hugging Face dataset) and point the environment at it:

   PowerShell:
   ```powershell
   $env:WBGBIF_DATA_PACKAGE = "path\to\speciesnet_overlap_data_package"
   python scripts\validate_release.py
   python scripts\score_cached_outputs.py
   python scripts\build_submission_table.py
   ```

   Bash:
   ```bash
   export WBGBIF_DATA_PACKAGE=/path/to/speciesnet_overlap_data_package
   python scripts/validate_release.py
   python scripts/score_cached_outputs.py
   python scripts/build_submission_table.py
   ```

2. `validate_release.py` checks event/answer-key isolation, media metadata,
   Croissant + CamtrapDP standards metadata, checksums, and the cached score table.
3. Confirm the rebuilt table in `results/speciesnet_overlap/rebuilt_submission_table.md`
   matches `application_safe_results.md`.

## Part B - Live model re-run (optional, costs apply)

Live reruns require the model runtimes and, for the LLM routes, provider API
credentials. Code is in `model_pipeline/`. See `model_pipeline/README.md` for the
dependency and credential details. Secrets are read from a local `.env` and are
never stored in this repository.

The exact run configurations below are reconstructed from the `run_parameters` block
in each manifest under `results/run_outputs/run_manifests/`.

### Vision-language models (Gemini, Qwen)

The VLM route is config-driven. The prompt, version and family suffix are locked in
`model_pipeline/config.py` (`PROMPT_TEMPLATE`, `PROMPT_VERSION`,
`GEMINI_ABSTENTION_SUFFIX`) and match the SHA-256 values in `PROMPTS.md`. Each model
is run through `model_pipeline/run_prototype_inference.py`, which writes a new run
manifest and a cached-output JSON. Provider routing:

- Gemini routes: Google (Vertex REST / AI Studio), key from `GOOGLE_API_KEY` or the
  Vertex variables in `.env`.
- Qwen3-VL-30B: OpenRouter, key from `OPENROUTER_API_KEY`.

### BioCLIP 2.5 (local, CPU)

```bash
python model_pipeline/run_bioclip_benchmark.py \
  --model hf-hub:imageomics/bioclip-2.5-vith14 \
  --candidate_policy geo \
  --aggregation majority_vote \
  --custom_label_template_policy single_photo \
  --k 5 --batch_size 10 --device cpu \
  --image_dir <cropped_runner_image_dir> \
  --run_tag bioclip25_geo_cropped_proto330_single
```

BioCLIP runs locally with `open_clip` + `torch`; no internet or API key is needed
after the model weights are cached. The exact per-run worker is preserved as
`model_pipeline/bioclip_worker_20260626_005320.py`.

### SpeciesNet v4.0.3b (local CLI)

```bash
python model_pipeline/run_speciesnet_benchmark.py \
  --runs geo_ensemble \
  --model kaggle:google/speciesnet/pyTorch/v4.0.3b/1 \
  --image_dir data_package/model_inputs \
  --run_tag v403b_uncropped_proto330
```

SpeciesNet runs in its own Python environment; point at it with the
`SPECIESNET_PYTHON` environment variable if it differs from the current interpreter.

## Responsible-use and data-rights notes

- The media derive from GBIF-mediated camera-trap datasets. Preserve the GBIF dataset
  citations, licence URLs, and creator/rights-holder attribution in
  `DATA_AND_MEDIA_RIGHTS.md`, `data/source_datasets.csv`, and `data/attribution.jsonl`.
- Public images are EXIF-stripped; checksums are provided in `checksums.sha256` and
  the data-package `checksums.sha256`.
- Model outputs are research artefacts, not verified identifications. Treat species
  labels - especially abstentions and low-confidence calls - as benchmark signal, not
  ground-truth determinations. See `MODEL_OUTPUT_TERMS.md`.
- This package is not a sealed holdout and is not endorsed by GBIF. No DOI or GBIF
  derived-dataset registration is claimed.
