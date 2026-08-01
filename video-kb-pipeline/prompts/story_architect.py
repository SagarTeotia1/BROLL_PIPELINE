"""Level-4 Story Architect Agent — system prompt (Agent 2).

Verbatim (structure preserved) from CLAUDE.md "LEVEL 4 -- REASONING" ->
"Agent 2 -- Story Architect Agent" -> "SYSTEM PROMPT -- Story Architect
Agent".

Do not edit this string without updating CLAUDE.md first -- it is the
locked, reviewed spec.  See pipeline/level4/story_architect_runner.py for
the windowed-batching logic (3-5 consecutive scenes per call, rolling
summary synced once per batch) and the write_scene_beats tool schema used
for structured output.

Windowed-batch version: batches of 3-5 consecutive scenes, rolling summary
synced once per batch (not per scene), PRUNED frame representation only
(caption/causality/continuity/people[pid+story_role+action] -- NEVER full
raw qwen_output JSON, per the explicit ~3x token cost warning in CLAUDE.md).
"""
from __future__ import annotations

STORY_ARCHITECT_SYSTEM_PROMPT = """\
You are the final reasoning pass before planning agents consume this video's
knowledge base. You write FOR a planner, not for a human reader — be
structured and grounded, not literary.

Input (one batch of 3-5 consecutive scenes, chronological):
  rolling_summary: 1-3 sentence summary of everything established before
    this batch (empty on the first batch)
  scenes: [
    {
      scene_index: 0-based position of this scene within THIS batch — echo
        this back unchanged on the matching scene_beat so your beat can be
        matched to the correct input scene. Every scene_index in this list
        must appear exactly once in your scene_beats output.
      scene_frames: PRUNED frame_analyses for this scene — caption,
        causality, continuity, people[] (pid+story_role+action only, NOT
        full gaze/pose/clothing/apparel detail), beat_type, scene_mood,
        tension_level. NEVER feed full raw qwen_output JSON here — a dense
        video can have 30+ frames/scene at ~300 tokens/frame raw; the pruned
        fields above run ~80-100 tokens/frame, a ~3x cut that matters a lot
        at scene-batch scale.
      speaker_turns: resolved dialogue for this scene's time range
        [{person_id, start_time, end_time, transcript_text}, ...]
    },
    ...
  ]
  cast: {pid: display_name}
  client_style_examples: OPTIONAL. Present only when this video's client has \
enough reward-scored history (CLAUDE.md "PIPELINE ADDENDUM 3" -> "LEVEL 9 \
-- REWARD & PUNISHMENT", item 9b) -- a short list of that client's past \
scene summaries that scored well. Usually an empty list, which must change \
nothing about how you write. THIS IS A SOFT STYLE PRIOR, NEVER GROUNDED \
CONTENT (rule 26): it may nudge phrasing/tone toward what has read well for \
this client before, but it is NEVER a source of facts -- every scene_beat \
you write must still be grounded ONLY in that scene's own scene_frames + \
speaker_turns, exactly as before this field existed.

For EACH scene in the batch:
  1. Merge that scene's frame-level scene_id variants into ONE canonical
     scene record. Trust the majority scene_id string across its
     scene_frames; list discarded aliases.
  2. Write one scene beat, grounded ONLY in that scene's scene_frames +
     speaker_turns — never invent a detail absent from the input. Use the
     OTHER scenes in this same batch as context for causal_link_to_next
     where relevant (they are already in front of you, no need to guess).

Then, once for the whole batch:
  3. Update rolling_summary for the NEXT batch. Keep it to 3 sentences max —
     this is what gets carried forward, not the full history. Do not let it
     grow unbounded across batches.

Return one scene_beat per input scene, via the write_scene_beats tool call.
Every scene_beat MUST include the scene_index of the input scene it answers,
copied verbatim from that scene's scene_index — this is how your beat is
matched back to the correct scene, not the order you return them in.
"""

# Tool/function schema (structured output) — see CLAUDE.md "Tool/function
# schema (structured output)" under Agent 2. Kept alongside the prompt so
# callers never hand-roll it.
#
# Groq's chat.completions API is OpenAI-compatible: tools are
# {"type": "function", "function": {"name", "description", "parameters"}} —
# NOT Anthropic's top-level "input_schema" shape. Field structure unchanged.
WRITE_SCENE_BEATS_TOOL = {
    "type": "function",
    "function": {
        "name": "write_scene_beats",
        "description": (
            "Return one scene_beat per input scene in this batch, plus an "
            "updated_rolling_summary (<=3 sentences) to carry into the next "
            "batch."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "scene_beats": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "scene_index": {"type": "integer"},
                            "canonical_scene_id": {"type": "string"},
                            "discarded_aliases": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "participants": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "summary": {"type": "string"},
                            "emotional_arc": {"type": "string"},
                            "causal_link_to_next": {"type": ["string", "null"]},
                        },
                        "required": [
                            "scene_index",
                            "canonical_scene_id",
                            "discarded_aliases",
                            "participants",
                            "summary",
                            "emotional_arc",
                            "causal_link_to_next",
                        ],
                    },
                },
                "updated_rolling_summary": {"type": "string"},
            },
            "required": ["scene_beats", "updated_rolling_summary"],
        },
    },
}
