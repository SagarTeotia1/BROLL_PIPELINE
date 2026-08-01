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
    client_id: str | None = None  # external client identifier — see migration 016_client_style_profiles.sql


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
    embedding: list[float] | None = None   # 1024-dim, BAAI/bge-large-en-v1.5 of `summary` (pgvector; B7)


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
    embedding: list[float] | None = None   # 1024-dim, BAAI/bge-large-en-v1.5 of `synopsis` (pgvector; B7)


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
class SceneOverrideRecord:
    id: str
    scene_id: str
    field: str
    new_value: dict | list | str | float | int
    reason: str | None
    created_by: str | None


@dataclass
class StorylineOverrideRecord:
    id: str
    storyline_id: str
    field: str
    new_value: dict | list | str | float | int
    reason: str | None
    created_by: str | None


@dataclass
class CorrectionEventRecord:
    id: str
    video_id: str
    level: int                       # which level's output was corrected (2, 4, 5, 6)
    entity_type: str                 # speaker_turn | canonical_relation | scene | storyline | edit_plan | qa_report
    entity_id: str
    field: str
    original_value: dict | list | str | float | int
    corrected_value: dict | list | str | float | int
    correction_source: str           # client | internal_editor | qa_agent_flag
    reason: str | None = None


@dataclass
class ClientStyleProfileRecord:
    """CLAUDE.md "PIPELINE ADDENDUM 2" -> "3. Client Style Profiles" — one
    row of `client_style_profiles` (migration 016). Read as a SOFT PRIOR
    only (rule 26) by L5 Pass B (`pacing_preference`) and L6's Color
    Grading Agent (`brand_colors`) / Caption-Text Overlay Agent
    (`caption_style`) — never a hard constraint, never overrides
    scene-grounded reasoning or causal continuity. One profile per
    `client_id` (v1, no versioning — see migration comment)."""

    id: str
    client_id: str
    caption_style: dict = field(default_factory=dict)
    pacing_preference: str | None = None
    brand_colors: dict = field(default_factory=dict)
    default_platform: str | None = None
    notes: str | None = None


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
class QAReportRecord:
    """Level-6 QA/Validation Agent report (PIPELINE ADDENDUM 2, item 1) —
    one row per QA pass over a finalized edit_plan's rendered output.
    `deterministic_checks` is the aggregate of every no-LLM check
    (black_frames, silences, loudness_range, caption_drift, clipping);
    `llm_review`/`llm_status` are the one Groq intent-match pass's output
    (both None if the LLM pass didn't run, e.g. no GROQ_API_KEY configured
    — deterministic checks alone still produce a `status`)."""

    id: str
    edit_plan_id: str
    video_id: str
    status: str                      # pass | warn | fail — overall gate status
    deterministic_checks: dict
    llm_review: str | None = None
    llm_status: str | None = None    # pass | warn | fail | None


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
class ShotMatteRecord:
    """Level-2 background matte for one shot (see CLAUDE.md PIPELINE ADDENDUM A1).

    ``r2_key`` points at the alpha-matte artifact (mp4 video, one shot-length
    clip) uploaded to R2 by ``pipeline.level2.matting_runner``. ``model_version``
    identifies the matting model/variant used (e.g. ``"rvm_mobilenetv3"``) so
    shots can be re-processed with a newer model without ambiguity about which
    matte came from which model.
    """

    id: str
    video_id: str
    shot_id: str
    r2_key: str
    model_version: str


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
class StockAssetRecord:
    """A2 — one row of `stock_assets` (knowledge_base/stock_assets/).

    Standalone infrastructure, built once by the indexer, queried later by
    L6's Compositing Agent (built separately). `license_type` is present and
    filterable now even though enforcement is deferred — see CLAUDE.md
    "Before building A3: stock licensing".
    """

    id: str
    source: str                          # pexels | storyblocks | internal
    license_type: str
    external_id: str | None = None
    description: str | None = None
    tags: list[str] = field(default_factory=list)
    embedding: list[float] | None = None  # 1024-dim, BAAI/bge-large-en-v1.5
    r2_cache_key: str | None = None


@dataclass
class BackgroundAssignmentRecord:
    """A3 — one row of `background_assignments` (L6 Compositing Agent,
    `pipeline/level6/background_selector.py`).

    Scene-level, not cut-order-level (CLAUDE.md PIPELINE ADDENDUM A3's table
    has no `edit_plan_id`/`cut_list_item_id` — a background pick is a
    property of the canonical `scenes` row, reusable across any edit plan
    built from that scene). ``asset_id`` must always reference a real
    `stock_assets` row returned by `search_stock_assets` — never invented
    (rule 12 closed-set discipline).
    """

    id: str
    scene_id: str
    asset_id: str
    start_offset: float = 0.0
    loop: bool = False
    rationale: str | None = None


@dataclass
class EmphasisEffectRecord:
    """A5 (decision half only) — one row of `emphasis_effects` (L6
    Compositing Agent, `pipeline/level6/emphasis_selector.py`).

    Cut-order-level (FK to `cut_list_items`, per CLAUDE.md PIPELINE
    ADDENDUM A5's table) — decides WHAT to zoom/highlight on, never the
    FFmpeg mechanics (`zoompan`/`crop`/overlay-shape parameters), which is
    a separate, later phase in `pipeline/level6/editing_director.py`.
    ``parameters`` is decision-level only: which person/region drove the
    choice and why, not render parameters like start_rect/end_rect/easing.
    """

    id: str
    cut_list_item_id: str
    effect_type: str  # zoom | highlight
    parameters: dict
    rationale: str | None = None


@dataclass
class LayerCompositeRecord:
    """A4 — one row of `layer_composites` (L6 Editing Director,
    `pipeline/level6/editing_director.py::materialize_layer_composite_ops`).

    Matches CLAUDE.md PIPELINE ADDENDUM A4's exact field set. This is the
    codebase's existing style for an `EditOperation` "type" — there is no
    Python `Enum`/`Literal` union for operation types anywhere in this
    pipeline (`EditOperationItem.type` is a plain `str`, matched by string
    comparison e.g. `op.get("type") == "SELECT_CLIP"` in
    `caption_overlay.py`/`editing_director.py`); a new operation "type" is
    represented the same way every other DB-backed L6 record is: one
    dataclass mirroring its table, with `layer_type` as the free-text
    discriminator (mirrors `emphasis_effects.effect_type`). Never authored
    by L5 — `edit_plan.operations` has no LAYER_COMPOSITE entries; L6
    materializes these directly from `background_assignments` (via the
    clip's own SELECT_CLIP op `scene_id`) at render time, per CLAUDE.md A4's
    "L6 pulls compositing decisions in at render time, not pre-populated in
    the plan" note.
    """

    id: str
    cut_list_item_id: str
    layer_type: str  # background_swap | pip | overlay
    source_ref: str | None = None  # background_assignments.id or another cut_list_item_id
    position: dict | None = None
    opacity: float = 1.0
    z_index: int = 0


@dataclass
class ZoomEmphasisMechanics:
    """A5 mechanics half — in-memory only, never persisted (see
    `layer_composites` docstring above for why: unlike A4's background swap,
    a zoom/highlight's render-time parameters are cheap to recompute
    deterministically from `emphasis_effects.parameters` +
    `cut_list_items` timing + `videos.width/height` every render, so there is
    no separate mechanics table for A5 — persisting a redundant, always-
    derivable copy would just be another place for it to drift stale).
    Built by `editing_director.py::_build_zoom_emphasis_mechanics` from one
    `emphasis_effects` row (`effect_type='zoom'`) and consumed immediately by
    `_build_zoom_emphasis_filter` in the same render pass.
    """

    cut_list_item_id: str
    start_rect: dict   # {x, y, w, h} in source pixel units — wide/establishing framing
    end_rect: dict     # {x, y, w, h} in source pixel units — tight framing on target_bbox
    easing: str        # linear | ease_in_out
    duration: float    # seconds, == this clip's own (source_end - source_start)


@dataclass
class HighlightCalloutMechanics:
    """A5 mechanics half — in-memory only, same rationale as
    `ZoomEmphasisMechanics`. Built from one `emphasis_effects` row
    (`effect_type='highlight'`)."""

    cut_list_item_id: str
    shape: str          # circle | arrow | underline
    target_bbox: dict   # {x, y, w, h} in source pixel units
    start_time: float   # seconds, relative to the clip's OWN trimmed timeline (0-based)
    duration: float     # seconds


@dataclass
class SearchableFactRecord:
    id: str
    video_id: str
    fact_text: str
    frame_id: str | None = None
    timestamp_s: float | None = None
    embedding: list[float] | None = None
    legacy_vector_id: str | None = None    # was `pinecone_id` — see migration 012 (B7)


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

    # No start_time/end_time here on purpose — pipeline/level4/
    # story_architect_runner.py's write-back always uses the input group's
    # own start_time/end_time (real frame timestamps), never the LLM's,
    # per the "ground truth never trusted from the LLM's own arithmetic"
    # rule. Real-video finding: requiring these as LLM-output fields caused
    # validation failures (the model returned null for values it doesn't
    # actually know) that dropped entire otherwise-valid windows — asking
    # for data that's discarded anyway was pure downside, no upside.
    # Position of the input scene this beat answers, 0-based within the
    # batch's `scenes` array — required so the runner can match beats back
    # to input scenes by identity instead of positional zip(). Real-video
    # finding: the model sometimes returns fewer beats than input scenes
    # (e.g. 3 beats for 4 scenes); without scene_index there is no way to
    # tell WHICH scene was dropped, so positional zip silently drops the
    # wrong one and leaves a gap in scene time coverage.
    scene_index: int
    canonical_scene_id: str
    discarded_aliases: list[str] = []
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
    # Optional, not required: real-video finding, first live L5 run — the
    # system prompt itself tells the model dispatch-request ops "do not
    # participate in sequence_index ordering... they attach to a SELECT_CLIP
    # op via downstream_ops instead," so the model reasonably omits it on
    # those ops. Making this a hard-required field meant a single dispatch
    # op omitting it (correctly, per the prompt's own instruction) failed
    # pydantic validation for the ENTIRE operations array — one op sank an
    # otherwise-valid 18-op plan. None is only meaningful for non-SELECT_CLIP
    # ops; run_sequencing_pass still enforces contiguous 0..N indices across
    # SELECT_CLIP ops specifically.
    sequence_index: int | None = None
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


# ---------------------------------------------------------------------------
# Pydantic models for Level-6 Compositing Agent structured tool output (A3
# background_selector.py / A5-decision emphasis_selector.py). Same discipline
# as every other Groq structured-output stage in this pipeline: validate
# shape here, then the runner enforces the CLOSED-SET constraint (asset_id
# must be one of the candidates actually offered; target_pid must be one of
# the faces actually observed in the window) before any DB write — this
# model only enforces field shape/types, not membership in that per-call
# candidate set (which isn't knowable at model-definition time).
# ---------------------------------------------------------------------------


class BackgroundPickItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    scene_id: str
    asset_id: str | None = None  # null = "no candidate was a good fit, skip"
    start_offset: float = 0.0
    loop: bool = False
    rationale: str = ""


class SelectBackgroundOutput(BaseModel):
    """Response shape for L6 Compositing Agent's `select_background` tool
    call — one batch of scenes, each with its own pick (or null asset_id)."""

    model_config = ConfigDict(extra="ignore")

    picks: list[BackgroundPickItem] = []


class EmphasisDecisionItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    cut_list_item_id: str
    effect_type: str = "none"  # zoom | highlight | none
    target_pid: str | None = None  # must be one of the candidate faces given, or null
    rationale: str = ""


class DecideEmphasisOutput(BaseModel):
    """Response shape for L6 Compositing Agent's `decide_emphasis` tool
    call (emphasis_selector.py's LLM-fallback path only — used solely for
    cut_list_items the deterministic rules couldn't confidently decide)."""

    model_config = ConfigDict(extra="ignore")

    decisions: list[EmphasisDecisionItem] = []


# ---------------------------------------------------------------------------
# Pydantic model for Level-6 QA/Validation Agent's one LLM pass (PIPELINE
# ADDENDUM 2, item 1 — `pipeline/level6/qa_agent.py`). Text-only, one call
# per delivered plan: compares `edit_plan.user_prompt` against the assembled
# storylines/scenes text actually selected. Same discipline as every other
# Groq structured-output stage — validate shape here before any DB write
# (rule 12), QA reports never edit anything (rule 24).
# ---------------------------------------------------------------------------


class QAIntentReviewOutput(BaseModel):
    """Response shape for the QA Agent's `review_intent_match` tool call —
    one call, one verdict, for the whole edit_plan."""

    model_config = ConfigDict(extra="ignore")

    status: str = "warn"   # pass | warn | fail
    reasoning: str = ""


# ---------------------------------------------------------------------------
# PIPELINE ADDENDUM 3, LEVEL 8: Human Feedback (item 8b — migration 018).
# Holistic/qualitative feedback, distinct from `correction_events` (which is
# field-level: "this exact value was X, should be Y"). `human_feedback`
# captures feedback that can't be pinned to one specific field — "the pacing
# always feels too slow for this client's reels" — via a closed `sentiment` +
# `category` pair so it's aggregatable (rule 28) for L9's `reward_signals`
# (not built in this migration; L8 only produces the raw signal L9 would
# later aggregate).
# ---------------------------------------------------------------------------


@dataclass
class HumanFeedbackRecord:
    id: str
    video_id: str
    sentiment: str                    # positive | negative | neutral
    category: str                     # pacing | color | caption | speaker_attribution |
                                       # narrative | music_audio | b_roll | other
    free_text: str
    edit_plan_id: str | None = None   # nullable — feedback can be on the video generally
    scene_id: str | None = None       # nullable — feedback can be scene-scoped or general
    rating: int | None = None         # optional, 1-5
    source: str = "client"            # client | internal_editor


# ---------------------------------------------------------------------------
# LEVEL 7 -- EVALUATION (CLAUDE.md "PIPELINE ADDENDUM 3" -> "LEVEL 7 --
# EVALUATION"). 7b: rubric scoring, extends qa_agent.py's existing
# intent-match pass with a 4-dimension score. Same closed-set/grounded
# discipline as QAIntentReviewOutput above -- validate shape before any DB
# write (rule 12), additive to `qa_status`, never a new blocking gate
# (rule 27).
# ---------------------------------------------------------------------------


class RubricScoreOutput(BaseModel):
    """Response shape for the QA Agent's `score_edit_rubric` tool call
    (7b) -- one call, four 0-10 dimension scores + one-sentence rationale
    each, for the whole edit_plan. `extra="ignore"` + permissive defaults
    match every other Groq/OpenRouter structured-output model in this
    codebase (a malformed/partial response degrades to defaults rather than
    raising — the caller treats a low-confidence/default-only response the
    same as "the rubric call didn't run", never as gate-blocking)."""

    model_config = ConfigDict(extra="ignore")

    intent_match: float = 0.0
    intent_match_rationale: str = ""
    narrative_coherence: float = 0.0
    narrative_coherence_rationale: str = ""
    pacing_consistency: float = 0.0
    pacing_consistency_rationale: str = ""
    technical_cleanliness: float = 0.0
    technical_cleanliness_rationale: str = ""


@dataclass
class EvaluationScoreRecord:
    """Level-7 rubric score (7b) -- one row per `qa_reports` row, 1:1
    extension. `rationale` is `{dimension: sentence}` — see
    `RubricScoreOutput` above for the LLM response shape this is built
    from. Additive signal for L9's `reward_signals` rollup (out of scope
    for this L7 implementation) -- never read by `qa_agent.py`'s own
    pass/warn/fail gate (rule 27)."""

    id: str
    qa_report_id: str
    edit_plan_id: str
    intent_match: float | None = None
    narrative_coherence: float | None = None
    pacing_consistency: float | None = None
    technical_cleanliness: float | None = None
    rationale: dict = field(default_factory=dict)


@dataclass
class LLMCallLogRecord:
    """One durable row per LLM call across L4-L6 (7c, closes audit gap #9
    — "only logger.info of token counts, no durable table"). `cost_usd` is
    a best-effort estimate from a small static per-model pricing table
    (see `shared/llm_client.py::estimate_cost_usd`) — `None` when the model
    isn't in that table, never a guessed number for an unknown model."""

    id: str
    video_id: str | None
    level: int
    stage: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost_usd: float | None = None
    latency_ms: int | None = None


# ---------------------------------------------------------------------------
# LEVEL 9 -- REWARD & PUNISHMENT (CLAUDE.md "PIPELINE ADDENDUM 3" -> "LEVEL
# 9 -- REWARD & PUNISHMENT", 9a). Appended after LLMCallLogRecord per this
# file's existing append-at-end convention for parallel in-flight work.
# ---------------------------------------------------------------------------


@dataclass
class RewardSignalRecord:
    """One computed rollup row (9a) -- a confidence-shrunk average of
    `evaluation_scores` dimensions + `human_feedback` sentiment, scoped by
    `scope_type`/`scope_key` (e.g. scope_type='client', scope_key=<client_id>,
    or scope_type='canonical_relation', scope_key='SPEAKS_TO'). NOT raw
    storage -- `evaluation_scores` and `human_feedback` remain the source of
    truth; this is a recomputed-on-demand rollup written by
    `scripts/compute_reward_signals.py` (UPSERT on (scope_type, scope_key),
    safe to rerun -- rule 15's idempotency discipline).

    `reward_score` is already confidence-shrunk toward 0 (see
    `scripts/compute_reward_signals.py::_shrink` for the exact formula) --
    consumers (9b few-shot injection, 9c alerting) additionally gate on
    `sample_count` before trusting a row for anything more than reporting
    (rule 29 -- this table never mutates pipeline behavior directly; it is
    read-only input to human-reviewed downstream steps)."""

    id: str
    scope_type: str      # 'canonical_relation' | 'pacing_style' | 'color_style' | 'client'
    scope_key: str
    reward_score: float  # -1..1, confidence-shrunk
    sample_count: int    # raw (pre-shrinkage) contributing row count
