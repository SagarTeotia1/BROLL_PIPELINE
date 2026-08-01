"""Level-6 Compositing Agent — background/B-roll SELECTION system prompt (A3).

Grounded in CLAUDE.md "PIPELINE ADDENDUM — Motion Graphics/Compositing +
Hardening" -> A3 ("Background/B-roll selection (the judgment call)"): the
retrieval half (embedding + `search_stock_assets` top-k) is cheap and
deterministic and lives in `pipeline/level6/background_selector.py`. The
ONE LLM-worthy decision is the actual PICK — which of the top-k retrieved
candidates best fits this scene, plus how to time it (loop it for the full
scene duration, or trim/offset into a longer stock clip).

Grounding discipline (same closed-set rule as L4's Grounding Agent / L5's
Pass A, and the same "text must come from real upstream data" discipline
`caption_overlay.py` already enforces for `dialogue_subtitle`): the model
may ONLY pick an `asset_id` that appears in the `candidates` list handed to
it for that scene. It must never invent an asset_id, and it must say so
(return `asset_id: null`) when none of the retrieved candidates are a
plausible fit for the scene — a bad forced pick is worse than no pick, and
a `null` result is a legitimate, expected answer here, not a failure.

Provider: Groq, `groq` Python SDK, OpenAI-compatible `chat.completions`
function-calling shape (`{"type": "function", "function": {...}}`). Model:
`settings.L6_COMPOSITING_MODEL` (default `qwen/qwen3.6-27b`) — same
single-tier Groq Qwen model already used at L4/L5/L6-color/L6-caption.
Tool-call arguments come back as a JSON STRING
(`response.choices[0].message.tool_calls[0].function.arguments`) —
`json.loads()`-ed before pydantic validation, same pattern as every other
Groq call site in this pipeline.
"""
from __future__ import annotations

L6_BACKGROUND_SELECTION_SYSTEM_PROMPT = """\
You pick a background/B-roll stock asset for a BATCH of video scenes. For \
each scene you are given a short list of CANDIDATE stock assets already \
retrieved by embedding similarity (real search results — not something you \
generate) plus the scene's own grounding context (summary, mood signals, \
transcript excerpt). Your job has two parts, for EACH scene:

  1. PICK exactly one candidate's `asset_id` that best matches the scene's \
setting, mood, and content — or decide NONE of the candidates fit well \
enough and return `asset_id: null`. A forced mediocre pick is worse than \
no pick; null is a normal, expected, non-failure answer when nothing in \
the candidate list is genuinely appropriate (e.g. the scene is an indoor \
interview and every candidate is an outdoor nature clip).

     You may ONLY return an `asset_id` that is literally present in this \
scene's own `candidates` list. Never invent an asset_id, never reuse an \
asset_id from a DIFFERENT scene's candidate list, never guess an id that \
"seems like it would exist." This is a hard, closed-set constraint — \
equivalent to Level-4/5's rule that an LLM here operates only on real \
retrieved data, never invents facts.

  2. TIME it, only when you picked a non-null asset_id:
     - `loop`: true if the stock clip is meant to tile/loop under the \
scene for its full duration (typical for short ambient B-roll/texture \
clips, or any clip shorter than the scene). false if the clip should play \
once, trimmed to the scene's length.
     - `start_offset`: seconds into the stock asset's own footage to start \
playback from (0 if the clip should be used from its beginning — most \
common; a non-zero offset is for skipping past an unwanted intro portion \
of a longer stock clip, when you have reason to believe the candidate's \
description suggests one, e.g. a "compilation" or "montage" tagged clip). \
Default to 0 unless you have a specific reason not to.

  3. `rationale`: one short sentence — for a non-null pick, cite what in \
the scene's own grounding context (mood, tags, transcript content) matched \
the chosen candidate's description/tags. For a null pick, say briefly why \
no candidate fit (e.g. "no outdoor/nature candidate available for this \
indoor interview scene").

Input:
  scenes: [
    {
      scene_id,
      query_text: the text used to retrieve these candidates — scene \
summary + emotional_arc + aggregated frame-level scene_mood/tags +
transcript excerpt for this scene's time range,
      candidates: [
        {asset_id, description, tags, license_type}, ...
        (already license-filtered upstream if a license_type was
        required for this call — you do not need to re-check license_type
        yourself, just judge content/mood fit)
      ]
    },
    ...
  ]

Return one result per scene_id via the select_background tool call. Never \
output a scene_id not present in this batch — closed-set discipline, same \
as every other reasoning stage in this pipeline.
"""

# Groq's chat.completions API is OpenAI-compatible: tools are
# {"type": "function", "function": {"name", "description", "parameters"}} —
# not Anthropic's top-level "input_schema" shape. Same convention as
# prompts/l6_color_grading.py / prompts/l6_caption_style.py.
SELECT_BACKGROUND_TOOL = {
    "type": "function",
    "function": {
        "name": "select_background",
        "description": (
            "Return one background asset pick (or null if no candidate "
            "fits) + timing + rationale per scene_id in the input scenes "
            "batch. asset_id must be one of that scene's own candidates, "
            "or null — never an invented or cross-scene id."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "picks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "scene_id": {"type": "string"},
                            "asset_id": {
                                "type": ["string", "null"],
                                "description": (
                                    "Must be an asset_id literally present "
                                    "in this scene's own candidates list, "
                                    "or null if none are a good fit."
                                ),
                            },
                            "start_offset": {"type": "number"},
                            "loop": {"type": "boolean"},
                            "rationale": {"type": "string"},
                        },
                        "required": ["scene_id", "asset_id", "rationale"],
                    },
                }
            },
            "required": ["picks"],
        },
    },
}
