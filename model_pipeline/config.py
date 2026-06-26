# config.py
# ============================================================
# WildlifeBenchmark Pipeline Configuration
# ============================================================

TEMPERATURE = 0
TOP_P = 1.0
SEED = 42
MAX_TOKENS = 1000

# ── TRUNCATION RETRY ──────────────────────────────────────────
# Some models (e.g. gemini-3.1-pro-preview with no_json_mime=True) occasionally
# truncate responses mid-JSON at very low token counts.  The provider will
# retry up to TRUNCATION_RETRY_MAX times when output_tokens < threshold AND
# JSON parsing fails (indicating truncation, not a genuine null prediction).
TRUNCATION_RETRY_MAX       = 2   # max extra attempts after first truncation
TRUNCATION_TOKEN_THRESHOLD = 60  # output tokens below this = suspected truncation

# ── PROMPT VERSION ────────────────────────────────────────────
# Bump this every time you change PROMPT_TEMPLATE.
# Format: "<major>.<minor>-<tag>"  e.g. "1.1-geo-cue", "2.0-cot"
PROMPT_VERSION = "1.2-conf-cal-bp47-no-descriptive-taxon"

# ── PROMPT TEMPLATE ───────────────────────────────────────────
# Single source of truth sent to ALL models.
# {n_frames} and {country} are filled at inference time.
#
# AGENT INSTRUCTIONS (for optimization iterations):
#   - Edit only this string between runs.
#   - Bump PROMPT_VERSION to match.
#   - Change ONE word, phrase, or sentence per iteration —
#     isolate what drives each score change.
#   - Do NOT add species-specific hints or lists; keep it
#     structural/instructional only.
#
# v1.2 changes (2026-05-06):
#   - Wording: "in the images or at times just one of the images"
#     -> "in one or more of the images" (clarity).
#   - JSON schema: "reasoning" moved to the top of the object so the
#     model writes diagnostic features before committing to a species
#     name (reason-first chain-of-thought baked into the schema).
#
# v1.2-conf-cal champion-locked 2026-05-07:
#   - Adds three explicit confidence-level definitions (high / medium /
#     low) before the JSON schema. Tells the model what each tier
#     means and (critically for the −5 hallucination penalty) instructs
#     it to "Prefer abstention to a low-confidence species call."
#   - Won the overnight 5-variant medium-stratum quick-eval 2026-05-07
#     by Adj TIS = 0.6849 (vs B0 v1.1 reasoning-LAST 0.4212, +62.6%
#     relative; vs B1 abstention-cue 0.5697, +20.2%; vs B2 country-
#     prior 0.5030; vs B5 stacked 0.4379 — stacking regresses below
#     B4 alone; cleaner instruction wins). Variance gate (full 330,
#     all difficulty strata) confirmed: Adj TIS = 0.7525 (even higher
#     than the medium-stratum number, since easy/hard bursts boost the
#     mean). All 3 council members improved: Flash-Lite 0.7424 Adj TIS,
#     Flash-Preview 1.4803, Qwen 0.0348 (Qwen's biggest jump — from
#     -1.03 under v1.1 to +0.03 under v1.2-conf-cal). Source: Tian
#     2023 / Lin 2022 verbal confidence calibration. Audit:
#     `track_b/results/optimization_runs/overnight_progress.json`.
#     Champion file: `track_b/results/optimization_runs/champion_state.json`.
#
# BP47 lock 2026-06-13:
#   - Adds one schema hygiene sentence before the JSON schema:
#     "Taxon fields must contain scientific taxon names or null, not descriptions."
#   - Locked after 3x full-330 checks on Gemini 3.5 Flash and Qwen3-VL-30B.
#     BP47 tied BP29 on equal-model mean Adjusted TIS but was the cleaner
#     safety-weighted challenger, with slightly better stability and Qwen
#     raw/species accuracy. Future prompt tests use BP47 as the shared base.
PROMPT_TEMPLATE = """\
You are a taxonomic identification expert. Here is a sequence of {n_frames} \
images from a single camera trap trigger event. This camera trap is deployed \
in: {country}. Identify the focal animal that is in one or more of the \
images to the most specific taxonomic level you can with confidence. Reason \
and do not guess.

Use the confidence levels as follows:
- high: clear diagnostic features visible across multiple frames; you would stake your reputation on this identification.
- medium: most diagnostic features visible but some ambiguity remains; roughly 70-90 percent sure.
- low: limited features visible; you are guessing within a narrow set of candidates. Prefer abstention to a low-confidence species call.

Taxon fields must contain scientific taxon names or null, not descriptions.

You MUST respond ONLY with valid JSON in this exact format:

{{
  "reasoning": "Key diagnostic features used (1-2 sentences maximum)",
  "species": "Scientific binomial name, or null if uncertain",
  "genus": "Genus name, or null if uncertain",
  "family": "Family name, or null if uncertain",
  "order": "Order name, or null if uncertain",
  "class": "Class name, or null if uncertain",
  "confidence": "high, medium, or low"
}}\
"""

# ── OPTIMIZATION LOOP CONSTANTS ───────────────────────────────
# Quick-eval stratum — Loop A (base-prompt iteration) and Loop B (per-family
# abstention iteration) run on this difficulty stratum only. Variance gate
# uses the full set across all strata (set difficulty=None to override).
# Medium = 110 bursts post Buteo buteo relabel = exactly 1 burst/species,
# all 110 species covered. Item-discrimination rationale documented in
# _context/METHODOLOGY.md "Quick-eval stratification".
QUICK_EVAL_DIFFICULTY = "medium"

QUICK_EVAL_PER_SPECIES = 5        # Per-species cap (mostly inactive — most species have <=2 bursts in medium)
VARIANCE_RUNS = 3                 # Number of full runs for the variance gate
SIGNIFICANT_IMPROVEMENT_THRESHOLD = 0.5   # Min Adj TIS gain (vs champion) to trigger variance test

# ── MODEL DEFINITIONS ─────────────────────────────────────────
MODELS = {
    "gemini-3-1-flash-lite": {
        "provider": "google",
        "family": "gemini",
        "api_model": "gemini-3.1-flash-lite-preview",
        "description": "Council Member - High Speed/Logic",
    },
    "gemma-3-27b": {
        "provider": "google",
        "family": "gemma",
        "api_model": "gemma-3-27b-it",
        "description": "DEPRECATED 2026-05-06 - sunsetted by Google (404 on AI Studio "
                       "v1beta/generateContent). Kept for historical result attribution; "
                       "do NOT re-add to COUNCIL_MEMBERS.",
    },
    "gemma-4-31b-google": {
        "provider": "google",
        "family": "gemma",
        "api_model": "gemma-4-31b-it",
        "supports_json_mime": True,
        "description": "Route-qualification candidate - Gemma 4 31B Dense via "
                       "Google AI Studio / Gemini API. Added 2026-05-29 as the "
                       "highest-quality Gemma 4 open-model alternate for Qwen "
                       "replacement and open-model benchmark coverage. Not a "
                       "council member; test with the locked base prompt and "
                       "empty Gemma suffix before any prompt work.",
    },
    "gemma-4-26b-a4b-google": {
        "provider": "google",
        "family": "gemma",
        "api_model": "gemma-4-26b-a4b-it",
        "supports_json_mime": True,
        "description": "Route-qualification candidate - Gemma 4 26B A4B MoE "
                       "via Google AI Studio / Gemini API. Added 2026-05-29 as "
                       "the efficiency alternate after the Gemma 4 31B route; "
                       "not a council member and not a prompt-optimisation target.",
    },
    "gemini-3-flash-preview": {
        "provider": "google",
        "family": "gemini",
        "api_model": "gemini-3-flash-preview",
        "description": "Council Member - Advanced Vision",
        # 2026-05-06: under v1.2-reason-first prompt (reasoning at top of JSON
        # schema), Flash-Preview's extended-thinking budget eats into the 1000
        # output-token cap and produces empty raw_response on ~50% of bursts
        # with output_tokens stuck at ~28. Diagnostic confirmed:
        #   - v1.1 prompt (reasoning-last) at 1000 tokens: clean JSON in 11s.
        #   - v1.2 + no_max_tokens=True (uncapped): also clean BUT calls take
        #     5+ minutes each as the model uses its full thinking budget; not
        #     workable at 110 burst x 5 min = 9h per model.
        # Resolution: bounded headroom — set max_output_tokens explicitly to
        # 8000 (8x the default cap), enough for thinking + JSON in <60s without
        # uncapped runaway. See DECISIONS.md 2026-05-06.
        "max_output_tokens": 8000,
    },
    "gemini-3-1-pro": {
        "provider": "google",
        "family": "gemini",
        "api_model": "gemini-3.1-pro-preview",
        "description": "Benchmark - Flagship Pro (reference, NOT council)",
        "no_json_mime":   True,  # response_mime_type truncates to ~7 tokens on this model
        "no_max_tokens":  True,  # max_output_tokens causes truncation; use model default
    },
    "gemini-3-5-flash": {
        "provider": "google",
        "family": "gemini",
        "api_model": "gemini-3.5-flash",
        "description": "Gemini 3.5 Flash (stable, released 2026-05-07). Same family as "
                       "gemini-3-flash-preview but newer stable. Knowledge cutoff Jan 2025; "
                       "final rows must still record row-level contamination disclosure "
                       "because a knowledge cutoff is not full training-pipeline proof. "
                       "Added 2026-05-07 for head-to-head test on the locked champion prompt "
                       "(1.2-conf-cal). Mirroring Flash-Preview's max_output_tokens=8000 "
                       "as a conservative prior in case the same extended-thinking-budget "
                       "interaction with reasoning-first JSON schema applies; revisit "
                       "after first scale run.",
        "max_output_tokens": 8000,
    },
    "qwen3-vl-30b": {
        "provider": "openrouter",
        "family": "qwen",
        "api_model": "qwen/qwen3-vl-30b-a3b-instruct",
        "description": "Council Member - Qwen 3 VL 30B-A3B via OpenRouter. Promoted "
                       "from observer to voting council member 2026-05-06 (replaces "
                       "gemma-3-27b after Google sunsetted it). MoE (30B total / 3B "
                       "active), 131K context, multi-image OpenRouter route. "
                       "$0.13 in / $0.52 out per M tokens. Prototype OpenRouter "
                       "route pre-resizes images to 1280 long-edge for cost, "
                       "latency, and comparability; this is not a hard API cap. "
                       "Final scored rows need frozen endpoint metadata and 2/5/10-image smoke.",
    },
    "llama-4-scout-openrouter": {
        "provider": "openrouter",
        "family": "llama",
        "api_model": "meta-llama/llama-4-scout",
        "openrouter_provider_pin": ["DeepInfra"],
        "description": "Route-qualification candidate - Meta Llama 4 Scout via "
                       "OpenRouter pinned to DeepInfra. Added 2026-05-27 as a "
                       "low-active-parameter open-model replacement candidate for "
                       "unstable Qwen prompt-testing rows. Uses the shared "
                       "OpenRouter 1280 long-edge JPEG image transport; official "
                       "Llama 4 docs report testing up to 5 input images, so any "
                       "10-frame row needs an explicit image-count policy. Has no "
                       "family-specific abstention suffix until scored Loop B "
                       "evidence exists. Not a council member.",
    },
    "mistral-medium-3-5-openrouter": {
        "provider": "openrouter",
        "family": "mistral",
        "api_model": "mistralai/mistral-medium-3-5",
        "openrouter_provider_pin": ["Mistral"],
        "description": "Route-qualification candidate - Mistral Medium 3.5 "
                       "via OpenRouter pinned to the Mistral upstream endpoint. "
                       "Added 2026-05-27 as the strongest current Mistral VLM "
                       "route for replacement-model testing. Dense 128B, "
                       "text/image/file input, 262K context, structured-output "
                       "support. Not a council member.",
    },
}

# ── COUNCIL & PROMPT-SUFFIX TABLE ─────────────────────────────
# Explicit prompt-optimisation panel (single source of truth for
# run_optimization_loop.py). Flash-Lite remains in MODELS for historical result
# attribution, but current prompt work compares Gemini 3.5 Flash with Qwen.
COUNCIL_MEMBERS = [
    "gemini-3-5-flash",
    "qwen3-vl-30b",
]

# Family-level abstention suffixes - appended to PROMPT_TEMPLATE per model
# during inference. Empty string = no suffix. Values are populated only from
# scored Loop B experiments; do NOT hand-edit these to invented wording.
# 2026-05-24: Gemini suffix locked after medium-110 lift and full-330
# confirmation on gemini-3-5-flash; Qwen remains empty after medium-110
# candidates failed to show a robust adjusted-TIS gain.
GEMINI_ABSTENTION_SUFFIX = (
    "Balance caution with evidence: abstain when features are missing, "
    "but commit when the visible features are diagnostic."
)

FAMILY_ABSTENTION_SUFFIXES = {
    "gemini":  GEMINI_ABSTENTION_SUFFIX,
    "gemma":   "",
    "qwen":    "",
    "claude":  "",   # placeholder for v1.1+
    "gpt":     "",   # placeholder for v1.1+
    "mistral": "",   # placeholder for v1.1+
    "pixtral": "",   # placeholder for v1.1+
    "llama":   "",   # placeholder for v1.1+
}

# Prompt-suffix policy lock, added 2026-06-22 after a Gemini 3.1 Pro/Antigravity
# run dropped the Gemini suffix by mistake. Current evidence supports applying
# the locked balance-caution suffix to Gemini-family frontier rows only. Small
# open-source/open-weight rows stay suffix-free unless a scored family-specific
# suffix round later promotes wording for that family.
FRONTIER_SUFFIX_REQUIRED_FAMILIES = {"gemini"}
SMALL_OPEN_SOURCE_NO_SUFFIX_FAMILIES = {"gemma", "qwen", "llama", "mistral", "pixtral"}

PROMPT_SUFFIX_POLICY = {
    "gemini": {
        "mode": "required_gemini_balance_caution",
        "suffix": GEMINI_ABSTENTION_SUFFIX,
        "evidence": "2026-05-24 Gemini medium-110 lift plus full-330 confirmation on gemini-3-5-flash",
    },
    "gemma": {
        "mode": "no_suffix",
        "suffix": "",
        "evidence": "No scored Gemma family-specific suffix promotion; use locked base prompt only",
    },
    "qwen": {
        "mode": "no_suffix",
        "suffix": "",
        "evidence": "Qwen suffix candidates failed medium-110/pair-delta gates; keep empty suffix",
    },
    "llama": {
        "mode": "no_suffix",
        "suffix": "",
        "evidence": "No scored Llama family-specific suffix promotion; route-quality evidence was poor",
    },
    "mistral": {
        "mode": "no_suffix",
        "suffix": "",
        "evidence": "No scored Mistral family-specific suffix promotion",
    },
    "pixtral": {
        "mode": "no_suffix",
        "suffix": "",
        "evidence": "No scored Pixtral family-specific suffix promotion",
    },
}


def get_prompt_suffix_policy_for_family(family: str) -> dict:
    """Return the locked suffix policy for a model family."""
    key = (family or "").strip().lower()
    return PROMPT_SUFFIX_POLICY.get(
        key,
        {
            "mode": "no_suffix_unpromoted_family",
            "suffix": "",
            "evidence": "No scored family-specific suffix promotion",
        },
    )


def requires_abstention_suffix_for_model(model_name: str) -> bool:
    """Return True for model routes where dropping the suffix is a policy error."""
    family = MODELS[model_name].get("family", "")
    return (family or "").strip().lower() in FRONTIER_SUFFIX_REQUIRED_FAMILIES


def get_abstention_suffix_for_model(model_name: str) -> str:
    """Return the locked suffix for a model route.

    Gemini-family routes always receive the locked Gemini balance-caution suffix.
    Small open-source/open-weight routes remain empty unless a later scored
    suffix round updates PROMPT_SUFFIX_POLICY.
    """
    family = MODELS[model_name].get("family", "")
    policy = get_prompt_suffix_policy_for_family(family)
    if requires_abstention_suffix_for_model(model_name):
        return GEMINI_ABSTENTION_SUFFIX
    return str(policy.get("suffix") or "")


def get_prompt_suffix_policy_for_model(model_name: str) -> dict:
    """Return audit metadata for the suffix policy used by a model route."""
    family = MODELS[model_name].get("family", "")
    policy = dict(get_prompt_suffix_policy_for_family(family))
    policy["family"] = family
    policy["model_name"] = model_name
    policy["suffix_required"] = requires_abstention_suffix_for_model(model_name)
    return policy


def get_effective_max_output_tokens_for_model(model_name: str) -> int | None:
    """Return the max-output-token cap actually sent for a model route.

    Some Gemini routes, especially Pro preview routes, deliberately omit
    max_output_tokens because sending a cap can trigger response truncation.
    Manifests should record that omission rather than the project default.
    """
    model_cfg = MODELS[model_name]
    if model_cfg.get("no_max_tokens"):
        return None
    return int(model_cfg.get("max_output_tokens", MAX_TOKENS))


def omits_max_output_tokens_for_model(model_name: str) -> bool:
    """Return True when the provider request should omit max_output_tokens."""
    return get_effective_max_output_tokens_for_model(model_name) is None
