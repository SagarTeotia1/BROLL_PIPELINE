from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class LevelStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Core dataclasses
# ---------------------------------------------------------------------------


@dataclass
class VideoMeta:
    id: str
    path: str
    r2_key: str | None
    duration_s: float | None
    fps: float | None
    width: int | None
    height: int | None


@dataclass
class ChunkRecord:
    id: str
    video_id: str
    chunk_index: int
    start_frame: int | None
    end_frame: int | None
    start_time: float | None
    end_time: float | None


@dataclass
class ShotRecord:
    id: str
    chunk_id: str
    video_id: str
    shot_index: int
    start_frame: int | None
    end_frame: int | None
    start_time: float | None
    end_time: float | None
    shot_type: str | None
    complexity: float | None


@dataclass
class KeyframeRecord:
    id: str
    shot_id: str
    video_id: str
    frame_index: int
    timestamp_s: float
    r2_key: str | None = None
    selection_reason: str | None = None
    dino_embedding: list[float] | None = None
    siglip_embedding: list[float] | None = None


@dataclass
class TranscriptSegment:
    id: str
    video_id: str
    chunk_id: str | None
    text: str
    start_time: float
    end_time: float
    confidence: float | None = None
    words: list[dict] = field(default_factory=list)


@dataclass
class PersonRecord:
    id: str
    video_id: str
    pid: str
    display_name: str | None = None
    arcface_embedding: list[float] | None = None


@dataclass
class FaceAppearanceRecord:
    id: str
    video_id: str
    frame_index: int
    timestamp_s: float
    person_id: str | None
    track_id: int | None
    bbox: dict | None
    emotion: str | None
    emotion_conf: float | None


@dataclass
class FaceTimelineEvent:
    id: str
    video_id: str
    person_id: str | None
    emotion: str
    start_time: float
    end_time: float
    confidence: float | None


@dataclass
class SpeakerTurnRecord:
    id: str
    video_id: str
    cluster_label: str
    person_id: str | None
    start_time: float
    end_time: float
    confidence: float | None
    resolution_method: str = "unresolved"  # face_majority | single_candidate | llm_tiebreak | llm_unresolved_final | unresolved


@dataclass
class SceneRecord:
    id: str
    video_id: str
    canonical_scene_id: str
    discarded_aliases: list[str]
    start_time: float
    end_time: float
    participants: list[str]          # list of pid
    summary: str | None
    emotional_arc: str | None
    causal_link_to_next: str | None
    usability_score: float | None = None


@dataclass
class StorylineRecord:
    id: str
    video_id: str
    version: int
    status: str                      # draft | final
    title: str | None
    synopsis: str | None
    cast_members: dict
    beats: list[dict]


@dataclass
class EditPlanRecord:
    id: str
    video_id: str
    storyline_id: str
    user_prompt: str
    target_duration_s: float | None
    platform: str | None
    status: str                      # draft | reviewed | applied | superseded
    version: int
    operations: list[dict]           # ordered EditOperation[]
    achieved_duration_s: float | None = None


@dataclass
class EditPlanRevisionRecord:
    id: str
    edit_plan_id: str
    user_feedback: str
    diff_operations: list[dict]


@dataclass
class CutListItemRecord:
    id: str
    edit_plan_id: str
    op_id: str
    sequence_index: int
    source_start: float
    source_end: float
    audio_lead_ms: int = 0
    video_lead_ms: int = 0
    transition: str = "cut"


@dataclass
class SequenceColorAdjustmentRecord:
    id: str
    edit_plan_id: str
    cut_list_item_id: str
    base_parameters: dict
    sequence_delta: dict
    rationale: str | None = None


@dataclass
class ColorGradeRecord:
    id: str
    video_id: str
    shot_id: str | None
    frame_index: int | None
    timestamp_s: float | None
    parameters: dict
    style_tags: list[str]


@dataclass
class FrameAnalysisRecord:
    id: str
    keyframe_id: str
    video_id: str
    scene_id: str | None
    scene_change: bool | None
    qwen_output: dict
    caption: str | None
    beat_type: str | None
    scene_mood: str | None
    tension_level: str | None
    tags: list[str]


@dataclass
class SearchableFactRecord:
    id: str
    video_id: str
    fact_text: str
    frame_id: str | None = None
    timestamp_s: float | None = None
    embedding: list[float] | None = None
    pinecone_id: str | None = None


@dataclass
class KGNode:
    id: str
    video_id: str
    node_type: str
    ref_id: str
    label: str | None = None
    properties: dict = field(default_factory=dict)
    embedding: list[float] | None = None


@dataclass
class KGEdge:
    id: str
    video_id: str
    source_id: str
    target_id: str
    relation: str
    weight: float
    properties: dict


@dataclass
class ProcessingJob:
    id: str
    video_id: str
    level: int
    status: LevelStatus
    error_msg: str | None
    meta: dict


# ---------------------------------------------------------------------------
# Pydantic models for Qwen V2 structured output parsing
# ---------------------------------------------------------------------------


class QwenSetting(BaseModel):
    model_config = ConfigDict(extra="ignore")

    location_type: str
    specific_location: str
    time_of_day: str
    indoor_outdoor: str


_VALID_SHOTS = {"wide", "medium", "close", "extreme_close", "static", "unknown"}
_VALID_ANGLES = {"eye-level", "low", "high", "dutch", "unknown"}
_VALID_MOVEMENTS = {"static", "pan", "tilt", "zoom", "handheld", "unknown"}


class QwenComposition(BaseModel):
    model_config = ConfigDict(extra="ignore")

    shot: str = "unknown"
    angle: str = "unknown"
    movement: str = "static"
    framing: str
    depth: str

    @field_validator("shot", mode="before")
    @classmethod
    def coerce_shot(cls, v: object) -> str:
        return v if isinstance(v, str) and v in _VALID_SHOTS else "unknown"

    @field_validator("angle", mode="before")
    @classmethod
    def coerce_angle(cls, v: object) -> str:
        return v if isinstance(v, str) and v in _VALID_ANGLES else "unknown"

    @field_validator("movement", mode="before")
    @classmethod
    def coerce_movement(cls, v: object) -> str:
        return v if isinstance(v, str) and v in _VALID_MOVEMENTS else "static"


class QwenLightingColor(BaseModel):
    model_config = ConfigDict(extra="ignore")

    lighting: str
    palette: list[str]
    grade_mood: str


class QwenObject(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str
    position: str
    story_relevance: str


class QwenPersonEntry(BaseModel):
    model_config = ConfigDict(extra="ignore")

    pid: str
    pose: str
    action: str
    gaze: str
    clothing: str
    emotion_inferred: str
    emotion_source: Literal[
        "visual_read",
        "transcript_tone",
        "fallback_automated_guess",
        "not_determinable",
    ]
    story_role: str
    apparent_goal: str


_VALID_INTERACTION_TYPES = {"address", "confront", "collaborate", "ignore", "observe"}


class QwenPersonInteraction(BaseModel):
    """Directed interaction between two identified persons in the frame."""
    model_config = ConfigDict(extra="ignore")

    p1: str | None = None
    p2: str | None = None
    type: str = "observe"
    notes: str = ""

    @field_validator("type", mode="before")
    @classmethod
    def coerce_type(cls, v: object) -> str:
        if isinstance(v, str) and v in _VALID_INTERACTION_TYPES:
            return v
        return "observe"  # comfort, support, challenge, etc. → generic

    @property
    def is_valid(self) -> bool:
        return bool(self.p1 and self.p2 and self.p1 != self.p2)


class QwenRelation(BaseModel):
    """Structured subject-predicate-object triple with confidence."""
    model_config = ConfigDict(extra="ignore")

    subject: str    # pid or object name
    predicate: str  # verb phrase
    object: str     # pid or object name
    confidence: float = 1.0


class QwenTextDetected(BaseModel):
    model_config = ConfigDict(extra="ignore")

    text: str
    location: str


class QwenStoryBeat(BaseModel):
    model_config = ConfigDict(extra="ignore")

    beat_type: str
    act_position: str
    conflict_present: bool
    stakes_signal: str


class QwenCausality(BaseModel):
    model_config = ConfigDict(extra="ignore")

    why_this_happens: str
    likely_trigger: str
    likely_consequence: str


class QwenContinuity(BaseModel):
    model_config = ConfigDict(extra="ignore")

    connects_to_previous: str
    sets_up_next: str
    recurring_object: str | None
    recurring_motif: str | None


class QwenEmotionalTone(BaseModel):
    model_config = ConfigDict(extra="ignore")

    scene_mood: str
    tension_level: str
    note: str


class QwenFrameOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    t: float
    scene_change: bool
    scene_id: str
    location_id: str = ""        # stable normalized location label (e.g. "office_desk")
    dominant_person: str | None = None   # pid of person who drives this frame narratively
    caption: str
    setting: QwenSetting
    composition: QwenComposition
    lighting_color: QwenLightingColor
    objects: list[QwenObject]
    people: list[QwenPersonEntry]
    interactions: list[QwenPersonInteraction] = []   # person-person interactions
    relations: list[QwenRelation] | list[list[str]] | None = None  # accept both formats
    text_detected: list[QwenTextDetected] | None = None
    dialogue_subtitle: str = ""
    story_beat: QwenStoryBeat | None = None
    causality: QwenCausality | None = None
    continuity: QwenContinuity | None = None
    emotional_tone: QwenEmotionalTone | None = None
    themes_symbols: list[str] = []
    searchable_facts: list[str] = []
    tags: list[str] = []

    def get_relations_structured(self) -> list[QwenRelation]:
        """Normalize relations to QwenRelation regardless of format Qwen returns."""
        if not self.relations:
            return []
        result = []
        for r in self.relations:
            if isinstance(r, QwenRelation):
                result.append(r)
            elif isinstance(r, (list, tuple)) and len(r) >= 3:
                result.append(QwenRelation(subject=str(r[0]), predicate=str(r[1]), object=str(r[2])))
        return result


# ---------------------------------------------------------------------------
# Pydantic models for Level-4 Grounding Agent structured tool output
# (mirror the QwenFrameOutput validation pattern: extra="ignore", strict
# about required fields — every LLM response is validated before any DB
# write, never written as raw text to a typed column).
# ---------------------------------------------------------------------------


class SpeakerResolutionItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    turn_id: str
    person_id: str | None = None   # a `pid` from the cast list (e.g. "P1"), or null
    confidence: float
    reasoning: str = ""


class SpeakerResolutionBatch(BaseModel):
    model_config = ConfigDict(extra="ignore")

    resolutions: list[SpeakerResolutionItem] = []


class RelationClusterResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    cluster_id: str
    canonical_relation: str
    reasoning: str = ""


class RelationCanonicalizationBatch(BaseModel):
    model_config = ConfigDict(extra="ignore")

    results: list[RelationClusterResult] = []


# ---------------------------------------------------------------------------
# Pydantic models for Level-4 Story Architect Agent structured tool output
# (mirror the QwenFrameOutput validation pattern — every LLM response is
# validated with these models before any DB write, never written as raw
# `response["field"]` text to a typed column).
# ---------------------------------------------------------------------------


class SceneBeatItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    canonical_scene_id: str
    discarded_aliases: list[str] = []
    start_time: float
    end_time: float
    participants: list[str] = []          # list of pid
    summary: str
    emotional_arc: str
    causal_link_to_next: str | None = None


class WriteSceneBeatsOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    scene_beats: list[SceneBeatItem] = []
    updated_rolling_summary: str = ""


class StorylineSummaryOutput(BaseModel):
    """Response shape for the single extra title/synopsis LLM call the
    Story Architect runner makes after all scene beats are written."""

    model_config = ConfigDict(extra="ignore")

    title: str
    synopsis: str


# ---------------------------------------------------------------------------
# Pydantic models for Level-5 Planning (Selection & Scoring / Sequencing &
# Pacing) structured tool output. Same discipline as L4: every Groq tool-call
# response is validated with these models before any DB write, never written
# as raw `response["field"]` text to a typed column (CLAUDE.md L5 read
# contract + "Validation before a plan is usable").
# ---------------------------------------------------------------------------


class CandidateSceneScore(BaseModel):
    model_config = ConfigDict(extra="ignore")

    scene_id: str
    relevance_score: float
    rationale: str = ""


class ScoreCandidateScenesOutput(BaseModel):
    """Response shape for L5 Pass A (Selection & Scoring) —
    `score_candidate_scenes` tool call."""

    model_config = ConfigDict(extra="ignore")

    candidates: list[CandidateSceneScore] = []


class EditOperationItem(BaseModel):
    """One element of the `EditOperation[]` array per CLAUDE.md's exact JSON
    shape under "LEVEL 5 — PLANNING" -> "EditPlan schema". `scene_id`,
    `start_time`, `end_time` are optional because dispatch-request op types
    (COLOR_MATCH_REQUEST, TEXT_OVERLAY_REQUEST, AUDIO_DUCK_REQUEST,
    B_ROLL_INSERT_REQUEST) don't necessarily carry their own scene/time
    range — they attach to a `SELECT_CLIP` op via `downstream_ops`."""

    model_config = ConfigDict(extra="ignore")

    op_id: str
    type: str
    scene_id: str | None = None
    start_time: float | None = None
    end_time: float | None = None
    sequence_index: int
    rationale: str
    transition_in: str = "cut"
    downstream_ops: list[str] = []


class BuildEditPlanOutput(BaseModel):
    """Response shape for L5 Pass B (Sequencing & Pacing) —
    `build_edit_plan` tool call."""

    model_config = ConfigDict(extra="ignore")

    operations: list[EditOperationItem] = []


class PlanRevisionDiffItem(EditOperationItem):
    """One diff element for `edit_plan_revisions.diff_operations` — an
    `EditOperationItem` plus an explicit `action` so the runner knows
    whether to add, replace, or drop the op with this `op_id` when applying
    the diff onto the existing plan (CLAUDE.md rule 21 — revisions are
    diffs, never full-plan regenerates)."""

    action: str = "modify"  # add | modify | remove


class ApplyPlanRevisionOutput(BaseModel):
    """Response shape for L5's revision call — `apply_plan_revision` tool
    call."""

    model_config = ConfigDict(extra="ignore")

    diff_operations: list[PlanRevisionDiffItem] = []


# ---------------------------------------------------------------------------
# Pydantic models for Level-6 Color Grading Agent structured tool output
# (mirror the L4/L5 pattern: every Groq tool-call response is validated with
# these models before any DB write, never written as raw `response["field"]`
# text to a typed column). `sequence_delta` values are further clamped
# against `color-grading`'s `schema.py` per-parameter bounds by
# `pipeline/level6/color_grading_runner.py` — this model only enforces
# *shape* (which keys/types came back), not the numeric range; range
# clamping needs the live PARAM_BY_NAME bounds, which this module
# deliberately does not import (keeps shared/types.py free of the
# color-grading sys.path dependency).
# ---------------------------------------------------------------------------


class SequenceDeltaItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    cut_list_item_id: str
    sequence_delta: dict[str, float] = {}
    rationale: str = ""


class ComputeSequenceDeltaOutput(BaseModel):
    """Response shape for L6 Color Grading Agent's `compute_sequence_deltas`
    tool call — one batch of `cut_list_items`, each with its own delta."""

    model_config = ConfigDict(extra="ignore")

    adjustments: list[SequenceDeltaItem] = []


# ---------------------------------------------------------------------------
# Pydantic models for Level-6 Caption/Text Overlay Agent structured tool
# output (mirror the L4/L5/L6-color pattern: every Groq tool-call response
# is validated with these models before use — never written as raw
# `response["field"]` text). The ONE LLM-worthy decision here is STYLE only
# (CLAUDE.md "Caption/Text Overlay Agent") — `text` is expected to be an
# unchanged echo of the real transcript text passed in; `caption_overlay.py`
# additionally cross-checks `text` against the source transcript and
# overrides it with the original if the LLM drifted (rule 18 — L6 never
# re-interprets intent, including "improving" the wording).
# ---------------------------------------------------------------------------


class CaptionStyleItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    op_id: str
    text: str
    font_size_tier: str = "medium"        # small | medium | large
    position: str = "lower_third"         # top | center | lower_third | bottom
    timing_mode: str = "static_block"     # karaoke_word_by_word | static_block
    emphasis_words: list[str] = []        # only meaningful for karaoke_word_by_word
    rationale: str = ""


class ChooseCaptionStyleOutput(BaseModel):
    """Response shape for L6 Caption Agent's `choose_caption_style` tool
    call — one batch of TEXT_OVERLAY_REQUEST ops, each with its own style."""

    model_config = ConfigDict(extra="ignore")

    styles: list[CaptionStyleItem] = []
