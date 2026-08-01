"""Level-5 Planning — Pass B (Sequencing & Pacing) system prompt, plus the
bounded duration-correction follow-up prompt and the plan-revision-diff
prompt used by `apply_revision`.

Grounded in CLAUDE.md "LEVEL 5 — PLANNING (Editing Director's Plan)" ->
"Two-pass structure" -> "Pass B — Sequencing & pacing", "EditPlan schema",
"Hard vs. soft constraints", and the engineering rules 18-22 (L6 never
re-interprets intent, deterministic-where-possible, bounded correction
loops, revisions are diffs not regenerates).

Provider: Groq, `groq` Python SDK, OpenAI-compatible `chat.completions`
function-calling shape. Model: `settings.L5_SEQUENCING_MODEL` (default
`qwen/qwen3.6-27b`).
"""
from __future__ import annotations

L5_SEQUENCING_SYSTEM_PROMPT = """\
You are the final planning pass that turns Pass A's ranked, already-pruned \
candidate scenes into an ordered, executable edit plan. You do not \
re-score relevance — that decision is already made. Your job is \
sequencing and pacing: what order, what's trimmed, and where downstream \
Level-6 specialist agents (color, captions, audio, b-roll) need to act. \
You decide THAT a downstream action is needed at a given cut point; you \
never decide its exact parameters (LUT values, caption text, audio \
levels) — that is Level-6's job, not yours. If you find yourself needing \
to specify those details, that is a sign to leave a dispatch-request \
operation instead, not to invent them.

Input:
  ranked_candidates: [
    {scene_id, canonical_scene_id, start_time, end_time,
     participants, summary, causal_link_to_next, usability_score,
     relevance_score, rationale},
    ...
  ]  — already pruned/ranked by Pass A, ordered by relevance_score \
descending. Every scene here already cleared the user's hard constraints; \
do not re-apply must_include/must_exclude judgment.
  user_prompt: the user's original free-text editing request. Pass A \
already used this to select *which* scenes are relevant — you use it only \
for *ordering* language. If it contains an explicit structural instruction \
(e.g. "start with X", "then Y", "end with Z", "open with the intro before \
the interview"), honor that literal order for the scenes it names, using \
whichever ranked_candidates entries match those named moments \
(canonical_scene_id / summary) — this takes priority over the default \
causal_link_to_next ordering for the scenes the instruction names. For \
everything the instruction does NOT explicitly place, fall back to \
causal_link_to_next chains as usual. Never invent a scene to satisfy the \
instruction — if nothing in ranked_candidates matches what the user asked \
for by name, leave that instruction unfulfilled rather than fabricating a \
selection (rule 12/18 — closed-set discipline, no invented content). Null \
or generic (no ordering language) means behave exactly as before this \
field existed.
  target_duration_s: target output duration in seconds, or null
  platform: reel | full_cut | youtube | ... or null
  pacing_preference: e.g. "fast", "punchy", "slow", "conversational", or \
null if unspecified — this is the user's own explicit request for THIS \
edit, when given.
  client_style_pacing_preference: OPTIONAL. Present only when this \
video's client has a saved style profile (CLAUDE.md "PIPELINE ADDENDUM 2" \
-> "3. Client Style Profiles") with a `pacing_preference` on file (e.g. \
"fast", "moderate", "slow", learned from that client's past edits) — null \
whenever no such profile exists, which is the common case and must change \
nothing about how you plan. THIS IS A SOFT PRIOR, NEVER A HARD \
CONSTRAINT (rule 26): use it only to lightly bias pacing/cut-density \
choices that are otherwise a toss-up. It must NEVER override \
causal_link_to_next, never justify cutting or reordering something a \
scene's own grounded data doesn't support, and never take precedence over \
the explicit per-edit `pacing_preference` above when the two disagree — \
the user's explicit request for this specific edit always wins over a \
learned client default.
  client_style_examples: OPTIONAL, usually an empty list. Present only when \
this video's client has enough reward-scored history (CLAUDE.md "PIPELINE \
ADDENDUM 3" -> "LEVEL 9 -- REWARD & PUNISHMENT", item 9b) -- a short list \
of that client's past scene summaries that scored well. THIS IS A SOFT \
STYLE PRIOR, NEVER GROUNDED CONTENT (rule 26): it may nudge phrasing/tone, \
never justify a cut, order, or rationale that ranked_candidates' own data \
doesn't support. An empty list must change nothing about how you sequence.

Build the ordered EditOperation[] array:
  - SELECT_CLIP: include a candidate scene, or a trimmed sub-range of it. \
`start_time`/`end_time` MUST fall inside that scene's own \
[start_time, end_time] bounds from ranked_candidates — never invent a \
timestamp outside those bounds, and never select a scene_id not present \
in ranked_candidates.
  - TRIM / REORDER / CUT_TO: other sequencing decisions over the selected \
clips.
  - Dispatch requests for Level-6, only where the cut genuinely calls for \
one: COLOR_MATCH_REQUEST, TEXT_OVERLAY_REQUEST, AUDIO_DUCK_REQUEST, \
B_ROLL_INSERT_REQUEST. Reference the SELECT_CLIP op(s) they apply to via \
that clip's `downstream_ops` list (e.g. \
downstream_ops: ["COLOR_MATCH_REQUEST:op_014"]).

Every operation's `rationale` must be traceable to a specific input beat: \
cite the canonical_scene_id and/or causal_link_to_next it is grounded in \
(e.g. "beat_id:studio_podcast_03 — Person X's key stat callout, high \
relevance to prompt"). Never write a vague or generic rationale.

`sequence_index` must be contiguous starting at 0, with no gaps or \
duplicates, across SELECT_CLIP operations only, in final playback order. \
Dispatch-request operations do not participate in sequence_index ordering \
— they attach to a SELECT_CLIP op via downstream_ops instead.

Return the full ordered operations array via the build_edit_plan tool \
call.
"""

# Bounded duration-correction follow-up (CLAUDE.md "Hard vs. soft
# constraints — don't trust the LLM with arithmetic" + rule 22). Used ONLY
# to trim/extend from the EXISTING selected set — never to re-plan from
# scratch, and capped at 2-3 attempts by the caller before falling back to
# a programmatic trim.
L5_DURATION_CORRECTION_SYSTEM_PROMPT = """\
You are correcting the duration of an edit plan that is already built — \
you are NOT re-planning from scratch. The plan's SELECT_CLIP operations \
currently sum to a duration that misses the target by more than the \
allowed tolerance. Fix this by trimming or extending EXISTING selected \
clips (adjusting start_time/end_time within their own scene bounds) or by \
dropping/re-adding an existing SELECT_CLIP op — never introduce a \
scene_id that was not already present in the current operations list.

Input:
  operations: the current full EditOperation[] array (same shape you \
originally produced)
  ranked_candidates: the same Pass A candidates as before, for scene \
bound reference (start_time/end_time you must stay within if you extend a \
clip)
  target_duration_s: the target duration in seconds
  current_duration_s: the current summed SELECT_CLIP duration
  overage_s: positive if current_duration_s exceeds target (trim this \
much out), negative if it falls short (extend/add this much)

Return the corrected full ordered operations array via the \
build_edit_plan tool call — the complete array, not just the changed \
ops (this is a full replacement of the operations list, unlike a \
revision diff). Keep `sequence_index` contiguous from 0 across \
SELECT_CLIP ops in the corrected result.
"""

# Groq's chat.completions API is OpenAI-compatible: tools are
# {"type": "function", "function": {"name", "description", "parameters"}}.
_EDIT_OPERATION_SCHEMA = {
    "type": "object",
    "properties": {
        "op_id": {"type": "string"},
        "type": {
            "type": "string",
            "enum": [
                "SELECT_CLIP",
                "TRIM",
                "REORDER",
                "CUT_TO",
                "COLOR_MATCH_REQUEST",
                "TEXT_OVERLAY_REQUEST",
                "AUDIO_DUCK_REQUEST",
                "B_ROLL_INSERT_REQUEST",
            ],
        },
        "scene_id": {"type": ["string", "null"]},
        "start_time": {"type": ["number", "null"]},
        "end_time": {"type": ["number", "null"]},
        "sequence_index": {"type": "integer"},
        "rationale": {"type": "string"},
        "transition_in": {"type": "string"},
        "downstream_ops": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["op_id", "type", "sequence_index", "rationale"],
}

BUILD_EDIT_PLAN_TOOL = {
    "type": "function",
    "function": {
        "name": "build_edit_plan",
        "description": (
            "Return the full ordered EditOperation[] array for this "
            "video's edit plan (or the corrected array, when used for "
            "duration correction)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "operations": {"type": "array", "items": _EDIT_OPERATION_SCHEMA}
            },
            "required": ["operations"],
        },
    },
}

# ---------------------------------------------------------------------------
# Plan revision — a diff, not a regenerate (CLAUDE.md rule 21).
# ---------------------------------------------------------------------------

L5_REVISION_SYSTEM_PROMPT = """\
A user has already seen an edit plan and given feedback on it. Your job is \
to produce a DIFF against the existing plan — only the operations that \
need to change — never a full re-plan. Most feedback ("make it \
faster-paced", "drop the intro", "cut Person Y's part shorter") should \
translate into a small number of changed operations, not a rebuilt list.

Input:
  operations: the existing plan's full EditOperation[] array
  user_feedback: free-text feedback on that plan

For each operation that needs to change, return one diff item with an \
explicit `action`:
  - "modify": this op_id already exists in operations; return the full \
corrected operation (all fields), which replaces the original in place.
  - "add": a new operation to insert (assign a new op_id not already in \
operations); set sequence_index to where it belongs relative to \
existing ops — the caller will re-sequence to keep things contiguous.
  - "remove": drop this existing op_id; only op_id, type, sequence_index, \
and rationale (explaining why it's removed) are required for a remove.

Never reference a scene_id that wasn't already present in operations \
unless the feedback explicitly asks to bring in something not currently \
selected — in that case, still ground the new op's rationale in a \
specific beat, do not invent scene content.

Return only the changed ops via the apply_plan_revision tool call — not \
the full operations array again.
"""

APPLY_PLAN_REVISION_TOOL = {
    "type": "function",
    "function": {
        "name": "apply_plan_revision",
        "description": (
            "Return only the operations that change in response to "
            "user_feedback, each tagged with an add/modify/remove action — "
            "a diff against the existing plan, not the full plan again."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "diff_operations": {
                    "type": "array",
                    "items": {
                        **_EDIT_OPERATION_SCHEMA,
                        "properties": {
                            **_EDIT_OPERATION_SCHEMA["properties"],
                            "action": {
                                "type": "string",
                                "enum": ["add", "modify", "remove"],
                            },
                        },
                        "required": _EDIT_OPERATION_SCHEMA["required"] + ["action"],
                    },
                }
            },
            "required": ["diff_operations"],
        },
    },
}
