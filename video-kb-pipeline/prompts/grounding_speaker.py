"""Level-4 Grounding Agent — Speaker Turn Resolution system prompt.

Verbatim (structure preserved) from CLAUDE.md "LEVEL 4 — REASONING" ->
"Agent 1 -- Grounding Agent" -> "1a. Speaker turn resolution" ->
"SYSTEM PROMPT -- Grounding Agent: Speaker Resolution".

Do not edit this string without updating CLAUDE.md first -- it is the
locked, reviewed spec.  See pipeline/level4/grounding_runner.py for the
batching/escalation logic that surrounds this prompt and the
resolve_speaker_turns tool schema used for structured output.
"""
from __future__ import annotations

GROUNDING_SPEAKER_SYSTEM_PROMPT = """\
You resolve a BATCH of ambiguous speaker turns for one video's knowledge
base in a single pass. You do not re-derive anything already certain — you
only decide what deterministic fusion could not.

Input:
  cast: [{pid, display_name}, ...]   — the ONLY valid non-null answers
  turns: [
    {
      turn_id, cluster_label, start_time, end_time,
      transcript_snippet: transcript text overlapping [start-2s, end+2s],
      visual_context: Qwen frame people[] entries in-window
        (pid, gaze, pose, action, story_role) — may be empty,
      face_presence: {pid: frame_count} strictly inside [start, end] — may be
        empty (that is WHY this turn is unresolved),
      face_presence_widened: {pid: frame_count} inside [start-3s, end+3s] on
        a continuous track_id — identity evidence just outside the strict
        window, may still be empty
    },
    ...
  ]

For EACH turn, decide independently (do not let one turn's answer bias
another) using this priority order:
  1. Explicit self-reference, or being addressed by name, in transcript_snippet
  2. Turn-taking pattern — who was addressed/asked a question in the prior turn
  3. face_presence_widened — same track_id continuous across the turn boundary
  4. Visual cues — gaze direction toward camera/mic, "speaking"/"addressing"
     action tags in visual_context
  5. If signals conflict or none are present, return null for that turn. Do
     not guess.

Return one result per turn_id, via the resolve_speaker_turns tool call.
Never output a pid not present in `cast`. Never invent a new speaker.
"""

# Tool/function schema (structured output, not parsed free text) — see
# CLAUDE.md "Tool/function schema (structured output, not parsed free text)"
# under 1a. Kept alongside the prompt so callers never hand-roll it.
#
# Groq's chat.completions API is OpenAI-compatible: tools are
# {"type": "function", "function": {"name", "description", "parameters"}} —
# NOT Anthropic's top-level "input_schema" shape. The schema's actual field
# structure (properties/required) is unchanged from the original Anthropic
# version; only the wrapper differs.
RESOLVE_SPEAKER_TURNS_TOOL = {
    "type": "function",
    "function": {
        "name": "resolve_speaker_turns",
        "description": (
            "Return one resolution per turn_id in the input batch, choosing "
            "person_id only from the provided cast (or null when genuinely "
            "ambiguous)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "resolutions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "turn_id": {"type": "string"},
                            "person_id": {"type": ["string", "null"]},
                            "confidence": {"type": "number"},
                            "reasoning": {"type": "string"},
                        },
                        "required": ["turn_id", "person_id", "confidence", "reasoning"],
                    },
                }
            },
            "required": ["resolutions"],
        },
    },
}
