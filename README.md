# WildlifeBench GBIF Ebbe Nielsen 2026 Review Package

This is a clean review scaffold for the GBIF Ebbe Nielsen Challenge submission. It is deliberately separate from the private WildlifeBench working repository.

## Practical summary

- GitHub staging URL: https://github.com/jonathanlyardley/WildlifeBench (private staging - to be made public after final release checks).
- Hugging Face dataset staging URL: https://huggingface.co/datasets/jonathanlyardley/wildlifebench-gbif-ebbe-2026-data (private staging - to be made public after final release checks).
- Final award-facing dataset in this release copy: 246 camera-trap bursts / 82 species in the SpeciesNet v4.0.3b classifier-label and country-geofence overlap.
- Media package: 2,010 cropped/uncropped JPEG derivatives with per-media attribution, licence URLs, EXIF-stripping notes and checksums.
- Excluded non-overlap events from the earlier 330-event development snapshot are not redistributed in this staging package.
- Heavy image files belong in the data package, not in the GitHub repo.
- This package is not a sealed holdout and is not endorsed by GBIF.
- No DOI or GBIF derived-dataset registration is claimed here.

## Review route

1. Read `SUBMISSION.md` and `docs/METHODOLOGY.md`.
2. Set `WBGBIF_DATA_PACKAGE` to the local data-package folder, or place that folder beside this repo as `data_package`.
3. Validate the release with `python scripts/validate_release.py`.
4. Rebuild the displayed table with `python scripts/build_submission_table.py`.
5. Inspect `results/speciesnet_overlap/application_safe_results.md`.
6. Review data rights in `DATA_AND_MEDIA_RIGHTS.md` and the data package files.

The no-cost route uses cached outputs and does not require paid API credentials.
