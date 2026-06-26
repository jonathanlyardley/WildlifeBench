# Submission Summary

WildlifeBench tests complete AI-assisted camera-trap labelling routes on GBIF-mediated media. This private staging copy contains only the final award-facing SpeciesNet v4.0.3b classifier-label and country-geofence overlap.

Staged links:

- GitHub: https://github.com/jonathanlyardley/WildlifeBench (private staging - to be made public after final release checks).
- Hugging Face image/dataset repository: https://huggingface.co/datasets/JonathanYardley/wildlifebench-gbif-ebbe-2026-data (private staging - to be made public after final release checks). Holds the 2,010 cropped/uncropped JPEG derivatives.

## Evidence layers

| Layer | Events | Species | Purpose |
|---|---:|---:|---|
| SpeciesNet-overlap award slice | 246 | 82 | Headline comparison against SpeciesNet v4.0.3b support. |

Use the generated table in `results/speciesnet_overlap/application_safe_results.md` for award-facing numerical claims.

## Audit trail

- `PROMPTS.md` - the exact prompt text and SHA-256 digests sent to the VLM routes.
- `results/run_outputs/run_manifests/` - one manifest per run: model, API route, decoding, token usage, cost, prompt SHA-256, and Langfuse trace IDs.
- `results/run_outputs/cached_outputs/` - raw per-event model responses for the 246 award events.
- `model_pipeline/` - the exact run code for the Gemini, Qwen, BioCLIP and SpeciesNet routes.
- `REPRODUCE.md` - no-cost re-scoring and full live re-run instructions.

For a downloadable visual preview of the same displayed comparison, use `gbif_award_preview.html` in the repo root (also at `docs/gbif_award_preview.html`). It is a self-contained static HTML file included in this GitHub review package.

No DOI or GBIF derived-dataset registration is claimed in this staging copy.
