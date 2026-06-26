# Operating Instructions

## No-cost review

Set `WBGBIF_DATA_PACKAGE` to the local SpeciesNet-overlap data-package folder, or place that folder beside this repo as `data_package`.

PowerShell example:

```powershell
$env:WBGBIF_DATA_PACKAGE = "path\to\speciesnet_overlap_data_package"
python scripts\validate_release.py
python scripts\score_cached_outputs.py
python scripts\build_submission_table.py
```

Bash example:

```bash
export WBGBIF_DATA_PACKAGE=/path/to/speciesnet_overlap_data_package
python scripts/validate_release.py
python scripts/score_cached_outputs.py
python scripts/build_submission_table.py
```

This route checks model-input isolation, media metadata, Croissant/CamtrapDP metadata, checksums, cached result files and table rebuilds. It does not call any paid model API.

## Audit the frozen run

- `PROMPTS.md` - exact prompt text and SHA-256 digests (match the `prompt` block in every run manifest).
- `results/run_outputs/run_manifests/` - per-run manifests with model, route, cost, prompt SHA-256 and Langfuse trace IDs.
- `results/run_outputs/cached_outputs/` - raw per-event model responses, restricted to the 246 award events.
- `model_pipeline/` - the exact run code (Gemini, Qwen, BioCLIP, SpeciesNet).

## Optional live reruns

Live reruns are intentionally outside the no-cost review path. They require the original WildlifeBench runner stack, provider credentials or a SpeciesNet runtime, and must write new run manifests before any new score is reported. Step-by-step commands are in `REPRODUCE.md` (Part B); the model-facing code is in `model_pipeline/`. Secrets are read from a local `.env` and are never stored in this repository.
