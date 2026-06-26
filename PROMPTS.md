# Prompts used in the GBIF Ebbe Nielsen 2026 run

This file is the single source of truth for the text-prompt instruction sent to the
vision-language models in the 246-event SpeciesNet-overlap award comparison.

The SHA-256 values below match the `prompt.template_sha256` and
`prompt.abstention_suffix_sha256` fields recorded in every run manifest under
`results/run_outputs/run_manifests/`, so reviewers can confirm that this is exactly
the prompt that produced the released outputs.

- Prompt version: `1.2-conf-cal-bp47-no-descriptive-taxon`
- Template SHA-256: `ca6a168ce409fc10786b2095705640a7f84a1ab4bd87789a4670f1bf48f2a1f1`
- Gemini abstention-suffix SHA-256: `b7fdbb3eb3bd0aff5d60c94f82cbc462c4dff4bdb8f440f03c96b9b52a299447`

The two placeholders `{n_frames}` and `{country}` are filled per event at inference
time. `{n_frames}` is the number of frames in the burst; `{country}` is the
deployment country derived from the camera-trap metadata.

## Base prompt template (sent to all VLM routes)

```
You are a taxonomic identification expert. Here is a sequence of {n_frames} images from a single camera trap trigger event. This camera trap is deployed in: {country}. Identify the focal animal that is in one or more of the images to the most specific taxonomic level you can with confidence. Reason and do not guess.

Use the confidence levels as follows:
- high: clear diagnostic features visible across multiple frames; you would stake your reputation on this identification.
- medium: most diagnostic features visible but some ambiguity remains; roughly 70-90 percent sure.
- low: limited features visible; you are guessing within a narrow set of candidates. Prefer abstention to a low-confidence species call.

Taxon fields must contain scientific taxon names or null, not descriptions.

You MUST respond ONLY with valid JSON in this exact format:

{
  "reasoning": "Key diagnostic features used (1-2 sentences maximum)",
  "species": "Scientific binomial name, or null if uncertain",
  "genus": "Genus name, or null if uncertain",
  "family": "Family name, or null if uncertain",
  "order": "Order name, or null if uncertain",
  "class": "Class name, or null if uncertain",
  "confidence": "high, medium, or low"
}
```

## Family abstention suffix

A single locked abstention suffix is appended to the base template for
**Gemini-family routes only**. Open-weight routes (Qwen3-VL-30B) and the local
specialist models receive an empty suffix.

Gemini abstention suffix (appended verbatim after the base template):

```
Balance caution with evidence: abstain when features are missing, but commit when the visible features are diagnostic.
```

Suffix policy applied in this run:

| Model route | Family | Abstention suffix |
|---|---|---|
| Gemini 3.1 Flash-Lite | gemini | Gemini balance-caution suffix |
| Gemini 3.1 Pro | gemini | Gemini balance-caution suffix |
| Gemini 3.5 Flash | gemini | Gemini balance-caution suffix |
| Qwen3-VL-30B | qwen | none (empty) |

## Specialist / open-foundation models (no text prompt)

- **SpeciesNet v4.0.3b** (`kaggle:google/speciesnet/pyTorch/v4.0.3b/1`) is a camera-trap
  classifier with a built-in geofence; it takes images and country, not a text prompt.
- **BioCLIP 2.5** (`hf-hub:imageomics/bioclip-2.5-vith14`) is a CLIP-style open
  foundation model scored over a GBIF-derived geo candidate-label set; it has no
  free-text instruction prompt. Its candidate-label construction is documented in the
  BioCLIP run manifest and `docs/METHODOLOGY.md`.

## Verifying the SHAs

```bash
python - <<'PY'
import hashlib
template = open("PROMPTS_TEMPLATE.txt").read()  # or paste the block above
print(hashlib.sha256(template.encode()).hexdigest())
PY
```

The canonical template string lives in the run code at
`model_pipeline/config.py` (`PROMPT_TEMPLATE`, `PROMPT_VERSION`,
`GEMINI_ABSTENTION_SUFFIX`), and that file's values reproduce the SHA-256 digests above.
