"""Level-5 Planning — Pass A (Selection & Scoring) system prompt.

Grounded in CLAUDE.md "LEVEL 5 — PLANNING (Editing Director's Plan)" ->
"Two-pass structure" -> "Pass A — Selection & scoring". Same closed-set
discipline as Level-4's Grounding Agent: score only the candidate scenes
given in the batch, never invent a scene_id, ground every rationale only in
the beat/scene data present in the input.

Batch cap (CLAUDE.md rule 23 / "Batch cap, same reasoning as the L4
Grounding Agent"): callers must sub-batch candidate_scenes at ~30-40 per
call themselves — this prompt/tool pair is per-batch, not per-video. See
`pipeline/level5/planner_runner.py::run_selection_pass`.

Provider: Groq, `groq` Python SDK, OpenAI-compatible `chat.completions`
function-calling shape (`{"type": "function", "function": {...}}`) — NOT
Anthropic's `input_schema` shape. Model: `settings.L5_SELECTION_MODEL`
(default `qwen/qwen3.6-27b`).
"""
from __future__ import annotations

L5_SELECTION_SYSTEM_PROMPT = """\
You score a BATCH of candidate scenes for one video against a user's \
editing intent, in a single pass. You do not invent scenes — score only \
the scenes given to you in candidate_scenes, and ground every rationale \
only in that scene's own data. This is closed-set scoring, the same \
discipline used elsewhere in this pipeline's reasoning stages: never \
reference a scene_id, person, or detail absent from the input.

Input:
  user_prompt: the user's editing intent in natural language (e.g. "cut \
this into a 60s reel focused on Person X's commentary", "give me the \
highlights", "full cut, tighter pacing")
  target_duration_s: target output duration in seconds, or null if \
unspecified
  platform: reel | full_cut | youtube | ... or null
  must_include: list of persons/topics/pids that MUST appear if at all \
possible, may be empty
  must_exclude: list of persons/topics/content that MUST NOT appear, may \
be empty — a hard constraint, not a preference
  candidate_scenes: [
    {
      scene_id, canonical_scene_id, start_time, end_time,
      participants: [pid, ...],
      summary, emotional_arc, causal_link_to_next,
      usability_score: 0.0-1.0 take/technical quality (audio clarity, \
exposure/focus, transcript confidence) — higher is better, use it to break \
ties between near-duplicate scenes covering the same moment
    },
    ...
  ]

For EACH candidate scene, assign:
  relevance_score: 0.0-1.0 — how well this scene serves user_prompt and the \
hard constraints.
    - A scene that contains must_exclude content scores near 0 regardless \
of otherwise-strong narrative fit — must_exclude is a hard constraint, not \
a soft preference.
    - A scene whose participants/summary clearly satisfy must_include \
scores higher, all else equal.
    - When two or more scenes cover the same beat/moment (similar summary, \
overlapping participants, adjacent causal_link_to_next), prefer the one \
with the higher usability_score — that is precisely what usability_score \
exists for.
  rationale: one short sentence, grounded ONLY in this scene's own \
summary/participants/emotional_arc/causal_link_to_next — never invent a \
detail absent from the input, never reference another scene.

Return one result per scene_id in candidate_scenes, via the \
score_candidate_scenes tool call. Never output a scene_id that was not in \
candidate_scenes — closed-set discipline, same as Level-4's Grounding \
Agent.
"""

# Groq's chat.completions API is OpenAI-compatible: tools are
# {"type": "function", "function": {"name", "description", "parameters"}} —
# not Anthropic's top-level "input_schema" shape. See
# prompts/grounding_speaker.py for the same convention already used at L4.
SCORE_CANDIDATE_SCENES_TOOL = {
    "type": "function",
    "function": {
        "name": "score_candidate_scenes",
        "description": (
            "Return one relevance score + rationale per scene_id in the "
            "input candidate_scenes batch, grounded only in that scene's "
            "own data."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "candidates": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "scene_id": {"type": "string"},
                            "relevance_score": {"type": "number"},
                            "rationale": {"type": "string"},
                        },
                        "required": ["scene_id", "relevance_score", "rationale"],
                    },
                }
            },
            "required": ["candidates"],
        },
    },
}
