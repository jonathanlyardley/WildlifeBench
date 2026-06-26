# GBIF Award Application-safe Result Table

Generated: 2026-06-26

## Practical summary

- Headline comparison scope: `speciesnet_v403b_geo_supported_overlap`.
- Included in the presentation slice: 246 events / 82 species.
- Inclusion rule: the ground-truth species must be in the local SpeciesNet v4.0.3b classifier labels and allowed by the SpeciesNet country geofence for that event.
- Excluded non-overlap events from the earlier development snapshot are not redistributed in this staging package.
- Raw per-event model outputs restricted to the 246-event SpeciesNet-overlap slice are included under `results/run_outputs/`; the 84 non-overlap prototype events are withheld. Per-run manifests (with Langfuse trace IDs) and the exact prompt are in `results/run_outputs/run_manifests/` and `PROMPTS.md`.

## Headline rows

| Model route | Runs | Events | Species | Species accuracy | Raw TIS | Adjusted TIS | Macro F1 | Hallucinations | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| SpeciesNet v4.0.3b geo ensemble | 1 | 246 | 82 | 26.0% | 35.3% | n/a | 0.27 | n/a | Canonical full-frame SpeciesNet v4.0.3b geo ensemble row. |
| BioCLIP 2.5 geo custom labels | 1 | 246 | 82 | 29.3% | 39.4% | -31.3% | 0.216 | 174.0 | Caveated BioCLIP 2.5 row using WildlifeBench GBIF-derived geo custom labels. |
| Gemini 3.1 Flash-Lite BP47 | 1 | 246 | 82 | 43.1% | 56.8% | 13.3% | 0.3888 | 107.0 | Current BP47 prompt plus Gemini abstention suffix. |
| Gemini 3.1 Pro BP47 | 1 | 246 | 82 | 43.5% | 56.7% | 35.5% | 0.4705 | 52.0 | Current BP47 prompt plus Gemini abstention suffix; repaired scored copy. |
| Gemini 3.5 Flash BP47 | 3 | 246 | 82 | 44.4% (43.5-45.5%) | 60.5% (60.0-61.0%) | 43.9% (42.9-44.8%) | 0.4916 (0.4807-0.5073) | 41.0 (40.0-42.0) | Three retained BP47 repeats summarised as mean with min/max range. |
| Qwen3-VL-30B BP47 | 2 | 246 | 82 | 27.4% (27.2-27.6%) | 40.6% | 3.2% (2.0-4.4%) | 0.2375 (0.2268-0.2482) | 92.0 (89.0-95.0) | Two retained BP47 repeats summarised as mean with min/max range. |

## Output files

- `headline_csv`: `results/speciesnet_overlap/application_safe_results.csv`
- `headline_md`: `results/speciesnet_overlap/application_safe_results.md`
- `event_predictions_csv`: `results/speciesnet_overlap/application_safe_results_events.csv`
- `included_events_csv`: `results/speciesnet_overlap/application_safe_speciesnet_overlap_included_events.csv`
- `included_species_csv`: `results/speciesnet_overlap/application_safe_speciesnet_overlap_included_species.csv`
- `included_birds_csv`: `results/speciesnet_overlap/application_safe_speciesnet_overlap_included_birds.csv`
- `manifest_json`: `results/speciesnet_overlap/application_safe_results_manifest_public.json`
