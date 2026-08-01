"""Level-4 Grounding Agent — Relation Canonicalization system prompt.

Verbatim (structure preserved) from CLAUDE.md "LEVEL 4 — REASONING" ->
"Agent 1 -- Grounding Agent" -> "1b. Relation canonicalization" ->
"SYSTEM PROMPT -- Grounding Agent: Relation Canonicalization".

Do not edit this string without updating CLAUDE.md first -- it is the
locked, reviewed spec. The `ontology` list injected into the user message
at call time is read from the `ontology_relations` table at runtime (see
pipeline/level4/grounding_runner.py) — NOT hardcoded here — so a future
ontology version bump never requires a code change.
"""
from __future__ import annotations

GROUNDING_RELATION_SYSTEM_PROMPT = """\
You assign ONE canonical relation type to each cluster of near-duplicate
free-text relation strings extracted from one video's frame analysis.

Input:
  clusters: [{cluster_id, raw_relations: [...] (a representative sample of
    up to 8 near-duplicate strings from this cluster, NOT the exhaustive
    list — every member still gets mapped to your answer, only what's shown
    to you is capped), total_count (the real number of members in this
    cluster)}, ...]
  ontology: the current allowed canonical relations (below) — you may ONLY
    choose from this list, never invent a new one.

Canonical ontology (v1):
  SPEAKS_TO, ADDRESSES, DISCUSSES, EXPLAINS, REACTS_TO, AGREES_WITH,
  DISAGREES_WITH, CRITICIZES, PRAISES, ASKS, ANSWERS, HAS_EMOTION,
  GESTURES_TOWARD, LOOKS_AT, IS_SEATED_NEAR, IS_POSITIONED_NEAR, WEARS,
  HOLDS, CAUSES, FOLLOWS, PART_OF_SCENE, APPEARS_IN, CONTAINS_OBJECT,
  IN_LOCATION, ILLUSTRATES_THEME, MENTIONS, OTHER

Return one result per cluster_id via the canonicalize_relations tool call.
If nothing fits, use OTHER — do not force a bad match.
"""

# Tool/function schema (structured output). Not spelled out verbatim in
# CLAUDE.md for 1b (only 1a and Agent-2 schemas are given in full) — shaped
# to match the same "one result per input item" contract as
# resolve_speaker_turns / write_scene_beats.
#
# Groq's chat.completions API is OpenAI-compatible: tools are
# {"type": "function", "function": {"name", "description", "parameters"}} —
# NOT Anthropic's top-level "input_schema" shape. Field structure unchanged.
CANONICALIZE_RELATIONS_TOOL = {
    "type": "function",
    "function": {
        "name": "canonicalize_relations",
        "description": (
            "Return one canonical_relation per cluster_id, chosen only from the "
            "ontology list supplied in the input."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "cluster_id": {"type": "string"},
                            "canonical_relation": {"type": "string"},
                            "reasoning": {"type": "string"},
                        },
                        "required": ["cluster_id", "canonical_relation", "reasoning"],
                    },
                }
            },
            "required": ["results"],
        },
    },
}
