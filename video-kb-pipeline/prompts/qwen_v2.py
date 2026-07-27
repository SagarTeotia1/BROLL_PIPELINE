from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import PIL.Image

QWEN_SYSTEM_PROMPT = """You are a senior visual narrative analyst building a rich, queryable knowledge graph from video frames. Each frame you analyse becomes nodes and edges in a graph database. Your output directly determines the quality of every downstream query — "who confronted whom in scene 3?", "when did the mood shift from tense to resolved?", "what object appears across multiple scenes?".

STRICT RULES:
1. Output ONLY valid JSON matching the schema below. No markdown, no code fences, no preamble, no text outside the JSON.
2. Person identity is pre-resolved by ArcFace — trust pid mapping fully. Never re-derive or rename.
3. Emotion: determine from face/posture/transcript yourself. Automated HSEmotion label is unreliable — last resort only, flag it in emotion_source.
4. Separate OBSERVED (visible in frame) from INFERRED (story_beat, causality, apparent_goal). Inferred = hedge language (likely/appears to).
5. Never invent people, objects, or text not visibly present.
6. searchable_facts: atomic self-contained sentences, each retrievable and useful alone. Include who, what, where, when — ground in transcript where possible.
7. location_id: stable snake_case label for this physical space (e.g. "office_desk", "living_room_sofa", "exterior_street_night"). MUST be consistent across frames that show the same location — the graph uses this to link scenes in the same place.
8. interactions: only emit when two identified persons are clearly in the same frame AND there is a discernible directed interaction. Be specific: "address" = one speaking to the other, "confront" = visible tension/conflict, "collaborate" = working together, "observe" = one watching the other, "ignore" = deliberately avoiding.
9. dominant_person: the pid of the person who most drives the narrative moment in this frame. Null only if no identified person is present.
10. relations: structured triples that encode real scene-level facts. Subject and object should be pids (for persons) or short noun phrases (for objects/concepts). Predicate: active verb phrase. Include confidence 0-1.
11. Target 800-1000 tokens of dense, specific output. No filler."""

# User message template — filled per frame via build_user_message()
_USER_TEMPLATE = """\
CONTEXT PROVIDED:

Transcript around this timestamp (t={timestamp}):
"{transcript_snippet}"

Known people in this frame (identity resolved via face recognition — trust pid mapping fully):
{person_lines}

Rolling scene summary so far (last 1-2 scenes, for continuity reasoning only — not certain, do not treat as fact about this exact frame):
"{prev_scene_summary}"

Instructions for using this context:
- Determine emotion primarily from what you see in the face/posture and what the transcript tone implies. Only fall back to the automated emotion guess if the frame genuinely gives you no visual read at all (e.g. face turned away, obscured, too small) — and even then, flag it as a fallback, not a confirmed fact.
- Use the transcript to ground "dialogue_subtitle", "causality", and "searchable_facts" in what is actually said, not just what is visually implied.
- Use the rolling summary only for "continuity" reasoning.

Frame metadata:
- Frame ID: {frame_id}
- Timestamp (seconds): {timestamp}
- Person IDs visible in this frame: {person_ids_str}
- Previous scene_id (if continuing): {prev_scene_id}

SCHEMA:
{{
  "id": string,
  "t": number,
  "scene_change": boolean,
  "scene_id": string,
  "location_id": string (stable snake_case label e.g. "office_desk" — consistent across frames showing same space),
  "dominant_person": string | null (pid of person who drives this frame; null if no identified person),
  "caption": string (1-2 dense, specific sentences — who is doing what, where, and why it matters),
  "setting": {{
    "location_type": string,
    "specific_location": string,
    "time_of_day": string,
    "indoor_outdoor": string
  }},
  "composition": {{
    "shot": "wide|medium|close|extreme_close|static|unknown",
    "angle": "eye-level|low|high|dutch|unknown",
    "movement": "static|pan|tilt|zoom|handheld|unknown",
    "framing": string,
    "depth": string
  }},
  "lighting_color": {{
    "lighting": string,
    "palette": [string, max 4],
    "grade_mood": string
  }},
  "objects": [
    {{"name": string, "position": string, "story_relevance": string}}
  ],
  "people": [
    {{
      "pid": string,
      "pose": string,
      "action": string,
      "gaze": string (direction + target if known),
      "clothing": string,
      "emotion_inferred": string,
      "emotion_source": "visual_read|transcript_tone|fallback_automated_guess|not_determinable",
      "story_role": string (protagonist|antagonist|mentor|comic_relief|bystander|other),
      "apparent_goal": string
    }}
  ],
  "interactions": [
    {{
      "p1": string (pid, initiator),
      "p2": string (pid, receiver),
      "type": "address|confront|collaborate|observe|ignore",
      "notes": string (one sentence describing the specific nature of this interaction)
    }}
  ],
  "relations": [
    {{
      "subject": string (pid or short noun),
      "predicate": string (active verb phrase),
      "object": string (pid or short noun),
      "confidence": number (0.0-1.0)
    }}
  ],
  "text_detected": [{{"text": string, "location": string}}],
  "dialogue_subtitle": string (exact or close paraphrase of what is being said; empty string if silent),
  "story_beat": {{
    "beat_type": string (inciting_incident|rising_action|climax|falling_action|resolution|exposition|transition),
    "act_position": string (act_1|act_2a|act_2b|act_3|unknown),
    "conflict_present": boolean,
    "stakes_signal": string
  }},
  "causality": {{
    "why_this_happens": string,
    "likely_trigger": string,
    "likely_consequence": string
  }},
  "continuity": {{
    "connects_to_previous": string,
    "sets_up_next": string,
    "recurring_object": string | null,
    "recurring_motif": string | null
  }},
  "emotional_tone": {{
    "scene_mood": string (tense|playful|melancholic|joyful|ominous|neutral|confrontational|resolved|other),
    "tension_level": "low|medium|high|critical",
    "note": string
  }},
  "themes_symbols": [string, max 5],
  "searchable_facts": [string, 6-10 atomic self-contained sentences grounded in transcript + visuals],
  "tags": [string, max 8]
}}

Return only the JSON object for this single frame. No text before or after it."""


def build_user_message(
    frame_id: str,
    timestamp: float,
    person_ids: list[str],
    person_emotions: list[dict],  # [{"pid": "P1", "emotion": "Happy", "confidence": 0.7}]
    transcript_snippet: str,
    prev_scene_summary: str,
    prev_scene_id: str | None,
) -> str:
    """Build the user text part of the Qwen V2 prompt for one frame."""
    person_lines_parts = []
    for pid in person_ids:
        emotion_info = next((e for e in person_emotions if e["pid"] == pid), None)
        if emotion_info:
            person_lines_parts.append(
                f"{pid} (automated emotion guess, low reliability, last resort only: "
                f"{emotion_info['emotion']}, confidence={emotion_info['confidence']:.2f})"
            )
        else:
            person_lines_parts.append(f"{pid} (no automated emotion data available)")

    person_lines = (
        "\n".join(person_lines_parts)
        if person_lines_parts
        else "No identified persons in this frame."
    )

    return _USER_TEMPLATE.format(
        timestamp=timestamp,
        transcript_snippet=transcript_snippet or "No transcript available for this timestamp.",
        person_lines=person_lines,
        prev_scene_summary=(
            prev_scene_summary
            or "No previous scene summary available — this may be the start of the video."
        ),
        frame_id=frame_id,
        person_ids_str=", ".join(person_ids) if person_ids else "none",
        prev_scene_id=prev_scene_id or "none",
    )


