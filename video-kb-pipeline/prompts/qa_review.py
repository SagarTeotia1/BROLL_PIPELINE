"""Level-6 QA/Validation Agent — content-intent match system prompt.

Grounded in CLAUDE.md "PIPELINE ADDENDUM 2" -> "1. QA / Validation Agent":
deterministic checks (black-frame, silence, loudness, caption-drift, color
clipping) run first, no LLM. This is the ONE LLM-worthy QA judgment call —
does the delivered cut's assembled content actually match what the user
asked for (`edit_plan.user_prompt`), e.g. "Person X's commentary only" but
the assembled scenes' participants show someone else dominating.

Grounding discipline (same closed-set rule as every other reasoning stage
in this pipeline — L4's Grounding Agent, L5's Pass A, the Compositing
Agent's background pick): the model is given ONLY the real
`storylines`/`scenes` text actually selected into this edit_plan's
`cut_list_items` (via each clip's scene) — it never sees raw frame_analyses
or invents content that isn't in the input. It reports a verdict; it never
edits the plan, the cut list, or anything else (rule 24 — QA reports, it
doesn't edit).

Provider: Groq, `groq` Python SDK, OpenAI-compatible `chat.completions`
function-calling shape. Model: `settings.L6_QA_MODEL` (default
`qwen/qwen3.6-27b`) — same single-tier Groq Qwen model already used at
L4/L5/L6-color/L6-caption/L6-compositing.
"""
from __future__ import annotations

L6_QA_INTENT_REVIEW_SYSTEM_PROMPT = """\
You are a QA reviewer for one finished video edit. You check whether the \
ASSEMBLED CONTENT actually delivered matches what the requester ASKED FOR — \
nothing else. You do not judge technical quality (that is handled by \
separate deterministic checks you never see). You do not suggest edits or \
rewrite anything — you only report a verdict and a short reason.

Input:
  user_prompt: the requester's original editing intent (e.g. "cut this into \
a 60s reel focused on Person X's commentary", "give me the highlights only \
of the football discussion").
  target_duration_s: requested duration, or null if unspecified.
  platform: requested platform (reel | full_cut | youtube | ...), or null.
  selected_scenes: the REAL scene records actually assembled into the \
delivered cut, ALREADY in final playback order — element 0 in this list \
plays first, the last element plays last. Each entry carries its own \
explicit `playback_position` (1-based) for exactly this reason: do not \
re-derive order by scanning or counting, read `playback_position` directly.
    [
      {
        playback_position (1 = plays first, 2 = plays second, ...),
        canonical_scene_id, summary, emotional_arc, participants: [pid, ...],
        cut_start_time, cut_end_time (this clip's own trimmed window,
          NOT the scene's full original bounds)
      },
      ...
    ]
  cast: {pid: display_name} — resolve participant pids to names using this,
    never guess a name not present here.

Judge whether `selected_scenes` (the content actually delivered, in the \
`playback_position` order given) is consistent with `user_prompt`'s stated \
intent — BOTH what content was included AND, when `user_prompt` specifies \
an order, whether that order was honored. Examples of a mismatch: the \
prompt asked for one person's commentary only but other people's scenes \
dominate the selection; the prompt asked for "highlights" but the selection \
reads as a near-complete chronological dump; the prompt named a topic and \
none of the selected scenes' summaries relate to it; the prompt gave an \
explicit sequence ("start with X, then Y, end with Z") and the \
`playback_position` values show a different order than requested — when \
checking this, match by `playback_position` number, never by re-scanning \
the list start-to-end from memory.

When checking an explicit sequence request, judge ONLY the RELATIVE ORDER \
of the named items against each other — do not require any named item to \
be immediately adjacent to the next one, and do not treat "X, then skit, \
end with Y" as broken just because Y is not the very next entry after the \
skit. "end with Z" is satisfied whenever Z is the LAST entry, full stop — \
it is never violated by what happens earlier in the list. Likewise "start \
with X" is satisfied whenever X is the FIRST entry. A named item being \
selected/repeated more than once (e.g. the same brand logo used at both \
the start and the end) is normal editing practice, not a defect, unless \
`user_prompt` explicitly objected to repetition.

Return exactly one verdict for the whole edit_plan via the \
`review_intent_match` tool call:
  - status "pass": the selection is clearly consistent with the stated intent.
  - status "warn": mostly consistent, but something is off enough to flag \
for a human to glance at before delivery (e.g. one borderline scene, a \
partial topic match).
  - status "fail": the selection clearly does not deliver what was asked \
for.
  - reasoning: one or two sentences, grounded ONLY in `selected_scenes` and \
`user_prompt` — never invent a scene or participant not present in the input.

You do not have access to target_duration_s enforcement details (that's a \
separate hard-constraint check elsewhere) — do not flag duration alone as a \
mismatch unless it's wildly inconsistent with an explicit duration-shaping \
request in user_prompt itself (e.g. prompt says "highlights only" but the \
selection is clearly the entire video).
"""

# Groq's chat.completions API is OpenAI-compatible: tools are
# {"type": "function", "function": {"name", "description", "parameters"}} —
# not Anthropic's top-level "input_schema" shape. Same convention as
# prompts/l6_color_grading.py / prompts/l6_caption_style.py /
# prompts/background_selection.py.
REVIEW_INTENT_MATCH_TOOL = {
    "type": "function",
    "function": {
        "name": "review_intent_match",
        "description": (
            "Return one pass/warn/fail verdict + short reasoning for whether "
            "the delivered edit's assembled scene selection matches the "
            "requester's stated editing intent (edit_plan.user_prompt)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["pass", "warn", "fail"],
                },
                "reasoning": {"type": "string"},
            },
            "required": ["status", "reasoning"],
        },
    },
}


# ---------------------------------------------------------------------------
# LEVEL 7 -- EVALUATION (CLAUDE.md "PIPELINE ADDENDUM 3" -> "LEVEL 7 --
# EVALUATION" -> 7b, "Rubric scoring"). Extends this module's existing
# intent-match pass with a second, independent LLM call scoring FOUR
# dimensions -- the existing `review_intent_match` call above is UNCHANGED,
# this is purely additive (rule 27: additive to qa_status, never a new
# blocking gate on its own).
# ---------------------------------------------------------------------------

L6_QA_RUBRIC_REVIEW_SYSTEM_PROMPT = """\
Score this delivered edit against FOUR dimensions, 0-10 each, grounded ONLY \
in the storylines/scenes/edit_plan data already provided (same closed-set \
discipline as every other agent in this pipeline — never invent a judgment \
about footage you weren't shown):

  intent_match      — does the assembled content match user_prompt \
                       (existing check, unchanged)
  narrative_coherence — do the selected scenes in this order tell a \
                       coherent story per their causal_link_to_next chains
  pacing_consistency  — do cut durations follow a defensible rhythm, or \
                       does the sequence feel arbitrary (grounded in \
                       sequence_color_adjustments/cut_list_items durations \
                       already computed, not re-analyzing pixels)
  technical_cleanliness — cross-check against the deterministic checks \
                       already run (blackdetect/silencedetect/caption-drift/ \
                       clipping) — does the LLM's read agree with what the \
                       formulas already found, flag if not (a real \
                       disagreement here is itself a signal worth logging)

Return one score + one-sentence rationale per dimension via the \
score_edit_rubric tool call. Never invent a score dimension not listed.

Input:
  user_prompt: the requester's original editing intent, or null.
  target_duration_s / achieved_duration_s: requested vs. actual duration.
  platform: requested platform (reel | full_cut | youtube | ...), or null.
  selected_scenes: the REAL scene records actually assembled into the \
delivered cut, in cut order — canonical_scene_id, summary, emotional_arc, \
causal_link_to_next, participants, cut_start_time, cut_end_time (this \
clip's own trimmed window).
  clip_durations: the actual (end - start) duration of each SELECT_CLIP in \
cut order — the real numbers to judge pacing_consistency against, not a \
guess.
  deterministic_checks: the SAME black_frames/silences/loudness_range/ \
caption_drift/clipping results the deterministic pass already computed for \
this delivered render — use this to ground technical_cleanliness, do not \
re-derive it from anything else.
  cast: {pid: display_name} — resolve participant pids to names using this, \
never guess a name not present here.

Ground every rationale sentence in the input actually given — never invent a \
scene, participant, or technical finding not present above.
"""

SCORE_EDIT_RUBRIC_TOOL = {
    "type": "function",
    "function": {
        "name": "score_edit_rubric",
        "description": (
            "Return four 0-10 dimension scores + one-sentence rationale each "
            "for one delivered edit_plan's rendered output."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "intent_match": {"type": "number"},
                "intent_match_rationale": {"type": "string"},
                "narrative_coherence": {"type": "number"},
                "narrative_coherence_rationale": {"type": "string"},
                "pacing_consistency": {"type": "number"},
                "pacing_consistency_rationale": {"type": "string"},
                "technical_cleanliness": {"type": "number"},
                "technical_cleanliness_rationale": {"type": "string"},
            },
            "required": [
                "intent_match", "intent_match_rationale",
                "narrative_coherence", "narrative_coherence_rationale",
                "pacing_consistency", "pacing_consistency_rationale",
                "technical_cleanliness", "technical_cleanliness_rationale",
            ],
        },
    },
}
