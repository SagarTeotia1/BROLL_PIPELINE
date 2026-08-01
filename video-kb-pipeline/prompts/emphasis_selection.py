"""Level-6 Compositing Agent — emphasis DECISION system prompt (A5, decision
half only).

Grounded in CLAUDE.md "PIPELINE ADDENDUM — Motion Graphics/Compositing +
Hardening" -> A5 ("Zoom in/out and highlight/emphasis effects"): the
mechanics (`zoompan`/`crop` Ken-Burns motion, overlay shapes) are a
SEPARATE, later phase in `pipeline/level6/editing_director.py` — not built
here. This prompt is only ever invoked as the FALLBACK path in
`pipeline/level6/emphasis_selector.py`, for the subset of `cut_list_items`
the deterministic rule pass (`beat_type`/`tension_level` spikes + a single
dominant face) could NOT confidently decide on its own — most cut_list_items
never reach this prompt at all (CLAUDE.md: "Rule-first ... LLM only if a
rule can't decide").

Grounding discipline: `target_pid` may only be one of the `candidate_faces`
actually observed (from real `face_appearances` rows) inside this clip's
time window — the model may never invent a person or point at someone not
actually detected in this specific clip. `effect_type: "none"` is a fully
legitimate answer (skip this clip — most ambiguous clips genuinely don't
need an emphasis effect at all; forcing one on every clip would be visual
noise, not judgment).

Provider: Groq, `groq` Python SDK, OpenAI-compatible `chat.completions`
function-calling shape. Model: `settings.L6_COMPOSITING_MODEL` (default
`qwen/qwen3.6-27b`) — same single-tier Groq Qwen model already used
throughout L4/L5/L6.
"""
from __future__ import annotations

L6_EMPHASIS_DECISION_SYSTEM_PROMPT = """\
You decide, for a BATCH of already-cut video clips, whether an EMPHASIS \
effect (a zoom-in, or a highlight callout) is warranted — and if so, on \
WHOM. You do NOT decide the FFmpeg mechanics (crop rectangles, easing \
curves, callout shapes) — only the two things a downstream mechanics stage \
needs: whether to emphasize at all, and which detected face (if any) to \
target.

Every clip you are given here already failed a deterministic rule pass \
(no single-dominant-face-plus-tension-spike pattern was obviously true) — \
you are the tie-breaker for genuinely ambiguous cases, not a second pass \
over everything. Expect that MOST clips in front of you will legitimately \
warrant `effect_type: "none"` — do not manufacture an emphasis effect for \
every clip just because you were asked to look at it.

For EACH clip, decide:

  effect_type: "zoom" | "highlight" | "none"
    - "zoom": push in on a specific person's face/region because they are \
the clear narrative focus of this moment even though the deterministic \
rule didn't catch it cleanly (e.g. tension_level is "medium" but the \
transcript/caption content and beat_type together make this person clearly \
the emotional center of the clip).
    - "highlight": call out a person or moment WITHOUT zooming — e.g. a \
reaction shot where drawing attention via a subtle marker fits better than \
a full push-in (rare; only when zoom would feel like the wrong register \
for the moment).
    - "none": nothing here rises to the level of a deliberate emphasis \
choice. This is the correct answer for the majority of ambiguous clips — \
choose it whenever you are not genuinely confident.

  target_pid: the person to emphasize, or null. If effect_type is "none", \
target_pid must be null. Otherwise it MUST be one of the pids listed in \
this clip's own `candidate_faces` — never a pid you have not been shown \
for this specific clip, never a guess at who "probably" is in frame. If no \
candidate_faces are given for a clip, effect_type must be "none" (there is \
nothing real to target).

  rationale: one short sentence. For "none", state briefly why nothing \
warranted emphasis (e.g. "multiple faces with similar screen time, no \
single clear focus"). For zoom/highlight, cite the specific signal \
(beat_type, tension_level, scene_mood, or transcript content) that tipped \
the decision.

Input, per clip in `clips`:
  cut_list_item_id: opaque id — echo back unchanged, never invent one
  beat_type, tension_level, scene_mood: aggregated from frame_analyses for \
this clip's time window (may be null if no frame_analyses overlap it)
  caption: a representative frame caption from this clip's window, for \
context
  transcript_excerpt: any transcript text overlapping this clip's window
  candidate_faces: [{pid, frame_count, dominant_emotion}, ...] — every \
person actually detected during this clip's window, with how many sampled \
frames they appeared in (higher frame_count = more screen presence) and \
their most common detected emotion. May be empty.

Return one decision per cut_list_item_id via the decide_emphasis tool \
call. Never output a cut_list_item_id not present in this batch, and never \
a target_pid not present in that clip's own candidate_faces — closed-set \
discipline, same as every other reasoning stage in this pipeline.
"""

# Groq's chat.completions API is OpenAI-compatible: tools are
# {"type": "function", "function": {"name", "description", "parameters"}} —
# not Anthropic's top-level "input_schema" shape. Same convention as
# prompts/background_selection.py / prompts/l6_color_grading.py.
DECIDE_EMPHASIS_TOOL = {
    "type": "function",
    "function": {
        "name": "decide_emphasis",
        "description": (
            "Return one emphasis decision (zoom | highlight | none) + "
            "optional target_pid + rationale per cut_list_item_id in the "
            "input clips batch. target_pid must be one of that clip's own "
            "candidate_faces, or null."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "decisions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "cut_list_item_id": {"type": "string"},
                            "effect_type": {
                                "type": "string",
                                "enum": ["zoom", "highlight", "none"],
                            },
                            "target_pid": {
                                "type": ["string", "null"],
                                "description": (
                                    "Must be a pid from this clip's own "
                                    "candidate_faces, or null. Must be "
                                    "null when effect_type is 'none'."
                                ),
                            },
                            "rationale": {"type": "string"},
                        },
                        "required": ["cut_list_item_id", "effect_type", "rationale"],
                    },
                }
            },
            "required": ["decisions"],
        },
    },
}
