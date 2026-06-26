# Run outputs - per-run audit bundle

This folder is the machine-checkable audit trail for the model runs behind the
246-event SpeciesNet-overlap award table in
`results/speciesnet_overlap/application_safe_results.md`.

## Scope

- Per-event model outputs in `cached_outputs/` are **restricted to the 246**
  SpeciesNet v4.0.3b classifier-label + country-geofence overlap event_ids. The 84
  non-overlap prototype events are withheld (see each file's `release_scope` block).
- `run_manifests/` holds the unmodified per-run manifests: model, API route, prompt
  SHA-256, decoding parameters, token usage, cost, and Langfuse trace IDs.
- The exact prompt text and SHA-256 values are in `../../PROMPTS.md`.
- BioCLIP 2.5 and SpeciesNet v4.0.3b are local-CLI models, so they have no LLM API
  Langfuse trace (none is expected).

## Langfuse traces

Langfuse project: `cmoua1qeo006qad07hsgo5cuv` (private cloud project; trace IDs are recorded for
reviewer audit and require project access to open).

| Model | Provider | Prompt version | Langfuse trace ID |
|---|---|---|---|
| hf-hub:imageomics/bioclip-2.5-vith14 | local_cli | - | local CLI - no trace |
| gemini-3.1-flash-lite-preview | google | 1.2-conf-cal-bp47-no-descriptive-taxon | 44f2405234e8dcc390a79ec6528f4975 |
| gemini-3.1-pro-preview | google | 1.2-conf-cal-bp47-no-descriptive-taxon | 8b42dbf384190cc68b43c6282e77c7fe |
| gemini-3.5-flash | google | 1.2-conf-cal-bp47-no-descriptive-taxon | 0d0c3fccc494b0e9d1d9ae8326a4d1ee |
| gemini-3.5-flash | google | 1.2-conf-cal-bp47-no-descriptive-taxon | bc3578d7eebd86db055c7860a30edd45 |
| gemini-3.5-flash | google | 1.2-conf-cal-bp47-no-descriptive-taxon | 79b3459d5ef563c51579f5b1aaa580df |
| qwen/qwen3-vl-30b-a3b-instruct | openrouter | 1.2-conf-cal-bp47-no-descriptive-taxon | c48feb9a5b99e22894fe6cab46332760 |
| qwen/qwen3-vl-30b-a3b-instruct | openrouter | 1.2-conf-cal-bp47-no-descriptive-taxon | 6463ed63704ffee79956a00f5a61c988 |
| speciesnet | local_cli | - | local CLI - no trace |

Full trace URLs are stored per run in each manifest's `trace.langfuse.trace_url`.

## Files

- `run_manifests/<run_id>.run_manifest.json` - one per run; the run-level audit record.
- `cached_outputs/<run_id>.json` - raw per-event model responses (`raw_response`),
  predictions, per-event ground truth and TIS for the 246 award events.

