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


class QwenComposition(BaseModel):
    model_config = ConfigDict(extra="ignore")

    shot: Literal["wide", "medium", "close", "extreme_close", "static", "unknown"]
    angle: Literal["eye-level", "low", "high", "dutch", "unknown"]
    movement: Literal["static", "pan", "tilt", "zoom", "handheld", "unknown"]
    framing: str
    depth: str


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
