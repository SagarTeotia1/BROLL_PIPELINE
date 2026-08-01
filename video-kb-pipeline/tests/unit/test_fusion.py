"""Unit tests for pipeline/level2/fusion.py::fuse_diarization_with_faces.

This is the pure, DB-free half of Level-2's identity resolution: mapping a
raw pyannote diarization turn (cluster_label, start_time, end_time) to a
`person_id` via face co-presence majority vote. No DB/network calls — a
plain function of `raw_turns` + `appearances` lists, so it belongs in
unit tests per CLAUDE.md B3 ("pure functions per the 'deterministic where
possible' design principle").

The `actor_id_to_pid` (P1/PU1 numbering) logic in
`pipeline/level2/face_runner.py::run_face_analysis` is NOT tested here: it
is embedded inline inside a large function that also drives the ONNX face
pipeline and reads a SQLite cast DB, so it cannot be exercised without
those dependencies — that is integration-test (or refactor) territory, not
unit-test territory. Flagged rather than silently skipped.
"""
from __future__ import annotations

from pipeline.level2.fusion import fuse_diarization_with_faces
from shared.types import FaceAppearanceRecord


def _appearance(person_id: str | None, timestamp_s: float) -> FaceAppearanceRecord:
    return FaceAppearanceRecord(
        id=f"app-{timestamp_s}-{person_id}",
        video_id="vid-1",
        frame_index=int(timestamp_s * 30),
        timestamp_s=timestamp_s,
        person_id=person_id,
        track_id=1,
        bbox={"x": 0, "y": 0, "w": 10, "h": 10},
        emotion=None,
        emotion_conf=None,
    )


def test_empty_raw_turns_returns_empty_list():
    assert fuse_diarization_with_faces("vid-1", [], []) == []


def test_single_candidate_resolves_with_full_confidence():
    """Exactly one distinct person present in-window -> single_candidate,
    confidence 1.0."""
    raw_turns = [{"cluster_label": "SPEAKER_00", "start_time": 1.0, "end_time": 3.0}]
    appearances = [
        _appearance("P1", 1.5),
        _appearance("P1", 2.0),
        _appearance("P1", 2.5),
    ]
    turns = fuse_diarization_with_faces("vid-1", raw_turns, appearances)
    assert len(turns) == 1
    t = turns[0]
    assert t.person_id == "P1"
    assert t.resolution_method == "single_candidate"
    assert t.confidence == 1.0
    assert t.cluster_label == "SPEAKER_00"
    assert t.video_id == "vid-1"


def test_face_majority_picks_the_most_frequent_person():
    """Two distinct persons in-window -> face_majority, person_id is the
    one with more frames, confidence is that share."""
    raw_turns = [{"cluster_label": "SPEAKER_01", "start_time": 0.0, "end_time": 4.0}]
    appearances = [
        _appearance("P1", 0.5),
        _appearance("P1", 1.0),
        _appearance("P1", 1.5),
        _appearance("P2", 2.0),
    ]
    turns = fuse_diarization_with_faces("vid-1", raw_turns, appearances)
    t = turns[0]
    assert t.person_id == "P1"
    assert t.resolution_method == "face_majority"
    assert t.confidence == 0.75  # 3/4


def test_unresolved_when_no_faces_in_window():
    raw_turns = [{"cluster_label": "SPEAKER_02", "start_time": 10.0, "end_time": 12.0}]
    appearances = [_appearance("P1", 50.0)]  # far outside the turn window
    turns = fuse_diarization_with_faces("vid-1", raw_turns, appearances)
    t = turns[0]
    assert t.person_id is None
    assert t.confidence is None
    assert t.resolution_method == "unresolved"


def test_unresolved_when_appearances_have_no_person_id():
    """face_appearances rows with person_id=None (detected but unidentified
    face) must never count as evidence for a candidate."""
    raw_turns = [{"cluster_label": "SPEAKER_00", "start_time": 0.0, "end_time": 2.0}]
    appearances = [_appearance(None, 1.0), _appearance(None, 1.2)]
    turns = fuse_diarization_with_faces("vid-1", raw_turns, appearances)
    assert turns[0].resolution_method == "unresolved"
    assert turns[0].person_id is None


def test_boundary_timestamps_are_inclusive():
    """A face appearance exactly AT start_time or end_time counts as
    in-window (the implementation uses <= on both ends)."""
    raw_turns = [{"cluster_label": "SPEAKER_00", "start_time": 5.0, "end_time": 7.0}]
    appearances = [_appearance("P1", 5.0), _appearance("P1", 7.0)]
    turns = fuse_diarization_with_faces("vid-1", raw_turns, appearances)
    assert turns[0].resolution_method == "single_candidate"
    assert turns[0].person_id == "P1"


def test_multiple_turns_processed_independently_and_in_order():
    raw_turns = [
        {"cluster_label": "SPEAKER_00", "start_time": 0.0, "end_time": 2.0},
        {"cluster_label": "SPEAKER_01", "start_time": 10.0, "end_time": 12.0},
        {"cluster_label": "SPEAKER_02", "start_time": 20.0, "end_time": 22.0},
    ]
    appearances = [
        _appearance("P1", 1.0),          # resolves turn 0 -> single_candidate
        _appearance("P1", 11.0),
        _appearance("P2", 11.5),         # resolves turn 1 -> face_majority
        # nothing in [20, 22] -> turn 2 unresolved
    ]
    turns = fuse_diarization_with_faces("vid-1", raw_turns, appearances)
    assert [t.resolution_method for t in turns] == [
        "single_candidate", "face_majority", "unresolved",
    ]
    # Order of output must mirror order of raw_turns input.
    assert [t.cluster_label for t in turns] == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"]


def test_result_ids_are_unique_and_generated():
    raw_turns = [
        {"cluster_label": "SPEAKER_00", "start_time": 0.0, "end_time": 2.0},
        {"cluster_label": "SPEAKER_01", "start_time": 2.0, "end_time": 4.0},
    ]
    turns = fuse_diarization_with_faces("vid-1", raw_turns, [])
    ids = {t.id for t in turns}
    assert len(ids) == 2
    assert all(isinstance(i, str) and i for i in ids)
