# Data And Media Rights

The data package records per-media licence URLs, GBIF source dataset links, occurrence links where available, derivative notes and redistribution decisions.

Important caveats:

- Media are GBIF-mediated, not GBIF-owned.
- The current package records `2010` copied media files under media mode `speciesnet_overlap_cropped_and_uncropped`.
- The release copy contains only the 246 camera-trap bursts / 82 species in the SpeciesNet v4.0.3b classifier-label and country-geofence overlap.
- Uncropped SpeciesNet media copied: `True`.
- Rights-holder fields were refreshed from GBIF occurrence metadata where present; creator fields remain `unknown_not_in_local_meta` where source metadata does not carry a recorder/creator value.
- A publication rights review was completed on 2026-06-26 for this release copy. Media are redistributed only where the recorded source licence is CC0-1.0 or CC-BY-4.0; per-media attribution is retained in `data/attribution.jsonl` and `data/image_hashes.csv`.

Primary data-package files:

- `data/model_inputs/events.jsonl`
- `data/ground_truth/answer_key.jsonl`
- `data/attribution.jsonl`
- `data/image_hashes.csv`
- `data/camtrapdp/datapackage.json`
- `data/croissant/gbif_ebbe_2026_croissant.jsonld`
