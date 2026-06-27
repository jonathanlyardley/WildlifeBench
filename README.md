## Practical summary
- Hugging Face image/dataset repository: https://huggingface.co/datasets/JonathanYardley/wildlifebench-gbif-ebbe-2026-data
- The cropped/uncropped 246 camera-trap bursts / 82 species JPEG media live here, not in this GitHub repo.
- Media package: 2,010 cropped/uncropped JPEG derivatives with per-media attribution, licence URLs, EXIF-stripping notes and checksums, hosted on the Hugging Face dataset above.
- Audit trail for the run: exact prompt (`PROMPTS.md`), per-run manifests with Langfuse trace IDs and the exact run code (`results/run_outputs/`, `model_pipeline/`), and reproduction steps (`REPRODUCE.md`).
- Per-event model outputs are restricted to the 246 events;
- No DOI or GBIF derived-dataset registration is claimed here.

## Review route

1. Read `SUBMISSION.md` and `docs/METHODOLOGY.md`.
2. Set `WBGBIF_DATA_PACKAGE` to the local data-package folder, or place that folder beside this repo as `data_package`.
3. Validate the release with `python scripts/validate_release.py`.
4. Rebuild the displayed table with `python scripts/build_submission_table.py`.
5. Inspect `results/speciesnet_overlap/application_safe_results.md`.
6. For deeper audit, read `PROMPTS.md`, the per-run manifests and raw outputs in `results/run_outputs/`, and the exact run code in `model_pipeline/`. Full reproduction steps are in `REPRODUCE.md`.
7. Open or download `gbif_award_preview.html` (repo root) for the self-contained visual preview of the same displayed comparison.
8. Review data rights in `DATA_AND_MEDIA_RIGHTS.md` and the data package files.

The no-cost route uses cached outputs and does not require paid API credentials.
