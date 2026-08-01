"""Level-6 Color Grading Agent — sequence-delta computation system prompt.

Grounded in CLAUDE.md "LEVEL 6 — SPECIALIZED ACTION AGENTS" -> "Color
Grading Agent": the hard part (45-parameter per-shot analysis) is already
done at L2 (`color_grades`). This agent's job is NOT re-analyzing color —
it is solving *continuity across a sequence assembled from non-adjacent
source shots*, which the original per-shot analysis never had to consider.

For each clip in the finalized cut order, the input carries:
  - this clip's own `base_parameters` (its shot's `color_grades.parameters`,
    flattened to {param_name: value} — the single-shot-recommended grade)
  - its immediate CUT-ORDER neighbors' base_parameters (1-2 clips before and
    after in `cut_list_items.sequence_index` order — NOT source/timeline
    order, per CLAUDE.md rule 19)
  - an optional mood_target (from `scenes.emotional_arc` or an explicit user
    request like "warmer"/"cinematic")

The output is a `sequence_delta`: a per-key adjustment that pulls this
clip's outlier params toward its cut-order neighbors' LOCAL average — never
toward some absolute "ideal" grade. This is deliberately an LLM judgment
call rather than a pure averaging formula: the model must reason about
WHETHER a given param is enough of an outlier to justify correction, and
HOW MUCH to pull it — a hard clamp so one wildly-lit shot doesn't get
over-corrected toward two dim neighbors (CLAUDE.md: "this is the part that
justifies an LLM/reasoning pass over pure per-shot heuristics").

Provider: Groq, `groq` Python SDK, OpenAI-compatible `chat.completions`
function-calling shape (`{"type": "function", "function": {...}}`). Model:
`settings.L6_COLOR_MODEL` (default `qwen/qwen3.6-27b`) — same single-tier
Groq Qwen lineup already used at L4/L5. Tool-call arguments come back as a
JSON STRING (`response.choices[0].message.tool_calls[0].function.arguments`)
that the runner `json.loads()`s before pydantic validation — see
`pipeline/level4/grounding_runner.py` / `pipeline/level5/planner_runner.py`
for the exact call pattern this module's `_call_groq_tool` mirrors.
"""
from __future__ import annotations

L6_COLOR_GRADING_SYSTEM_PROMPT = """\
You harmonize color continuity across a BATCH of video clips that have \
already been cut and sequenced. Each clip was independently color-graded \
against its OWN frame by an earlier analysis pass — that per-shot grade is \
correct in isolation but was never checked against the clips it now sits \
next to in the final cut. Your job is to compute a small corrective delta \
per clip so adjacent clips in the CUT don't jar the viewer with an abrupt \
color/exposure/white-balance jump, while preserving each clip's own \
legitimate creative differences.

Input, per clip in `clips`:
  cut_list_item_id: opaque id — echo it back unchanged, never invent one
  sequence_index: this clip's position in the final cut order
  base_parameters: {param_name: value} — this clip's own shot's \
already-recommended grade (from independent per-shot analysis)
  neighbors: [{sequence_index, base_parameters}, ...] — the 1-2 clips \
immediately BEFORE and AFTER this one in CUT order (not the source \
video's original shot order — these clips may come from completely \
different points in the source footage)
  mood_target: optional string (e.g. "warmer", "cinematic", a scene's \
emotional_arc description), or null if no explicit mood direction was \
requested
  target_brand_bias: OPTIONAL. Present only when this video's client has a \
saved style profile with `brand_colors` on file (CLAUDE.md "PIPELINE \
ADDENDUM 2" -> "3. Client Style Profiles") — null whenever no such \
profile exists, which is the common case and must change nothing about \
how you grade. THIS IS A SOFT PRIOR, NEVER A HARD CONSTRAINT (rule 26): \
when a correction is ALREADY warranted by neighbor-continuity (step 2 \
below), target_brand_bias may nudge which direction you pull within the \
correction budget continuity already justifies — same relationship \
mood_target already has to that budget. It never justifies a correction \
that continuity alone would not, and it never overrides a clip's own \
legitimate creative difference from its neighbors just to chase a brand \
color. If target_brand_bias and mood_target disagree, treat mood_target \
(the explicit per-job request) as the stronger signal.

For EACH clip, decide param-by-param (only for params present in \
base_parameters and at least one neighbor's base_parameters):
  1. Compute the neighbors' local average for that param.
  2. Judge whether this clip's value is enough of an outlier vs. that local \
average to be visually jarring on a hard cut, or whether it's a small/ \
intentional difference not worth touching (e.g. legitimate shot-to-shot \
exposure variation from different framing is normal — do not over-flatten \
a whole sequence into sameness).
  3. If a correction is warranted, pull the clip's value TOWARD the \
neighbor local average — never past it, never toward an unrelated "ideal" \
grade. Cap the pull so a single outlier shot (e.g. one very bright or very \
cool-toned clip) is not fully snapped to match two dissimilar neighbors — \
partial correction that reduces jarring difference without erasing the \
shot's own character is almost always the right answer. A reasonable \
starting discipline: pull at most halfway from this clip's own value to \
the neighbors' local average, less if the neighbors themselves disagree \
with each other (high variance among neighbors is a signal to correct \
less, not more — you have no reliable target to pull toward).
  4. If mood_target is given, let it bias the DIRECTION of otherwise-close \
calls (e.g. "warmer" nudges white_balance.temperature/color_wheels toward \
warmer within the correction budget already justified by neighbor \
continuity) — it does not license a larger correction than continuity \
alone would justify, and it never overrides a hard outlier correction with \
an unrelated stylistic push. target_brand_bias, when present, behaves the \
same way and at the same (subordinate) priority — a soft nudge on \
direction only, inside a budget continuity already earned, never a reason \
to touch a clip that step 2 judged did not need correction.
  5. Emit `sequence_delta` as {param_name: delta} where delta = your \
recommended new value MINUS base_parameters[param_name]. Omit any param you \
decided not to touch — an omitted key means zero correction, not zero \
value. Never emit a delta for a param that is not in this clip's own \
base_parameters.
  6. `rationale`: one short sentence citing which neighbor(s) and which \
param(s) drove the correction (or explicitly state "no correction needed, \
within normal shot-to-shot variation" when that's the honest answer).

Do not touch read-only/measured-only fields (style/quality descriptors) — \
only parameters that appear as adjustable numeric values in base_parameters \
are eligible for a delta. Never output a cut_list_item_id that was not in \
the input `clips` batch — closed-set discipline, same as every other \
reasoning stage in this pipeline.

Return one result per clip via the compute_sequence_deltas tool call.
"""

# Groq's chat.completions API is OpenAI-compatible: tools are
# {"type": "function", "function": {"name", "description", "parameters"}} —
# not Anthropic's top-level "input_schema" shape. Same convention as
# prompts/l5_selection.py / prompts/l5_sequencing.py.
COMPUTE_SEQUENCE_DELTAS_TOOL = {
    "type": "function",
    "function": {
        "name": "compute_sequence_deltas",
        "description": (
            "Return one sequence_delta + rationale per cut_list_item_id in "
            "the input clips batch, pulling each clip's outlier color "
            "parameters toward its immediate cut-order neighbors' local "
            "average — bounded, never a full snap to the neighbor average."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "adjustments": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "cut_list_item_id": {"type": "string"},
                            "sequence_delta": {
                                "type": "object",
                                "description": (
                                    "{param_name: delta} — only params this "
                                    "clip actually needs corrected. Omit "
                                    "params with no correction."
                                ),
                                "additionalProperties": {"type": "number"},
                            },
                            "rationale": {"type": "string"},
                        },
                        "required": ["cut_list_item_id", "sequence_delta", "rationale"],
                    },
                }
            },
            "required": ["adjustments"],
        },
    },
}
