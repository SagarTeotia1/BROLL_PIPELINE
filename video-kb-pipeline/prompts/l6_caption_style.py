"""Level-6 Caption/Text Overlay Agent — caption STYLE system prompt.

Grounded in CLAUDE.md "LEVEL 6 — SPECIALIZED ACTION AGENTS" ->
"Caption/Text Overlay Agent" and engineering rule 18 (L6 never
re-interprets intent). The ONE LLM-worthy decision at this stage is
STYLE — word-by-word karaoke timing/emphasis for reels vs. a static
lower-third block for a full cut — never the text content. The caption
text itself must be real transcript data (from `transcript_segments.text`
/ `frame_analyses.qwen_output.dialogue_subtitle`), passed in unchanged;
the model is told explicitly to echo it back, and
`pipeline/level6/caption_overlay.py` cross-checks the returned text
against the source and overrides it with the original if the model
drifted (never trusts the LLM's rewrite of the words themselves).

Provider: Groq, `groq` Python SDK, OpenAI-compatible `chat.completions`
function-calling shape (`{"type": "function", "function": {...}}`).
Model: `settings.L6_CAPTION_MODEL` (default `qwen/qwen3.6-27b`) — same
single-tier Groq Qwen model already used at L4/L5.
"""
from __future__ import annotations

L6_CAPTION_STYLE_SYSTEM_PROMPT = """\
You choose the on-screen STYLE for a batch of caption/text-overlay \
requests attached to an edit plan's cut points. You do NOT write or \
rewrite the caption text — the `transcript_text` given to you for each \
request is already real, grounded transcript data (from Whisper \
word-level transcription or Level-3's Qwen `dialogue_subtitle` field). \
Your `text` field in the response must be that same text, character-for- \
character unchanged. If you find yourself wanting to shorten, clean up, \
retranslate, or paraphrase it, don't — that is not your job here; leave \
it exactly as given. Inventing or altering dialogue text is a hard \
violation of this pipeline's grounding discipline (equivalent to \
Level-4/5's closed-set rule: never output content absent from the input).

What IS your job — pick the presentation style that fits the platform and \
the line itself:

  font_size_tier: "small" | "medium" | "large" — larger for short, \
punchy, high-emphasis lines on a reel; smaller/medium for longer, \
denser lines that need to fit without wrapping awkwardly (e.g. full_cut \
lower-thirds).

  position: "top" | "center" | "lower_third" | "bottom" — where on frame. \
"lower_third" is the conventional choice for a full_cut / interview-style \
deliverable. "center" or "top" suit short high-energy reel captions where \
the speaker's face may already occupy the lower frame.

  timing_mode: "karaoke_word_by_word" | "static_block" —
    - "karaoke_word_by_word": each word (or short phrase) appears/ \
highlights in sync with speech, common on reels/shorts for retention. \
Prefer this for platform="reel" and short (~<15 word) lines.
    - "static_block": the full line appears as one static block for its \
duration, standard for full_cut / longer-form / podcast-style captions, \
or for long lines where word-by-word would be visually noisy.

  emphasis_words: only populated when timing_mode is \
"karaoke_word_by_word" — a short list of words FROM transcript_text \
(exact substrings, not paraphrases) worth visually emphasizing (bigger/ \
bolder/different color at render time) because they carry the line's key \
stat, name, or punchline. Leave empty for static_block, and leave empty \
if no word in the line is meaningfully more emphasis-worthy than the \
rest — do not force an emphasis word onto every line.

  rationale: one short sentence tying your choice to platform + the \
line's own content/length — never generic ("looks good"), always \
specific to this line.

Input:
  platform: reel | full_cut | youtube | ... or null
  captions: [
    {op_id, transcript_text, start_time, end_time}, ...
  ]

Return one style item per op_id in captions, via the choose_caption_style \
tool call. Never output an op_id not present in the input batch — \
closed-set discipline, same as every other reasoning stage in this \
pipeline.
"""

# Groq's chat.completions API is OpenAI-compatible: tools are
# {"type": "function", "function": {"name", "description", "parameters"}}.
CHOOSE_CAPTION_STYLE_TOOL = {
    "type": "function",
    "function": {
        "name": "choose_caption_style",
        "description": (
            "Return one caption style choice per op_id in the input "
            "captions batch. `text` must echo the input transcript_text "
            "unchanged — style parameters only, never rewritten wording."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "styles": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "op_id": {"type": "string"},
                            "text": {
                                "type": "string",
                                "description": (
                                    "Must be an unchanged echo of the "
                                    "transcript_text given for this op_id."
                                ),
                            },
                            "font_size_tier": {
                                "type": "string",
                                "enum": ["small", "medium", "large"],
                            },
                            "position": {
                                "type": "string",
                                "enum": ["top", "center", "lower_third", "bottom"],
                            },
                            "timing_mode": {
                                "type": "string",
                                "enum": ["karaoke_word_by_word", "static_block"],
                            },
                            "emphasis_words": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "rationale": {"type": "string"},
                        },
                        "required": [
                            "op_id",
                            "text",
                            "font_size_tier",
                            "position",
                            "timing_mode",
                            "rationale",
                        ],
                    },
                }
            },
            "required": ["styles"],
        },
    },
}
