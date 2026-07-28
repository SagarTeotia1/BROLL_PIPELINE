"""Level-2 Stage 0: fuse raw diarization turns with ArcFace identity.

Cluster labels coming out of pyannote (``SPEAKER_00``, ...) carry no identity
— they are just consistent-per-run voice clusters. This module maps each
diarization turn to a real cast ``person_id`` using face co-presence
(majority vote over ``face_appearances`` rows falling inside the turn's time
range), which is a stronger identity signal than acoustic clustering alone.

Turns that cannot be confidently resolved from the closed cast set are kept
with ``person_id=None`` and ``resolution_method="unresolved"`` — L5's
Grounding Agent picks those up later using transcript content + Qwen visual
cues, rather than this deterministic fusion step guessing.
"""
from __future__ import annotations

import logging
from collections import Counter

from shared.types import FaceAppearanceRecord, SpeakerTurnRecord
from shared.utils import gen_id

logger = logging.getLogger(__name__)


def fuse_diarization_with_faces(
    video_id: str,
    raw_turns: list[dict],
    appearances: list[FaceAppearanceRecord],
) -> list[SpeakerTurnRecord]:
    """Map each raw diarization turn to a person_id via face co-presence.

    Parameters
    ----------
    video_id:
        Identifier of the video these turns belong to.
    raw_turns:
        Output of :func:`pipeline.level2.diarization_runner.run_diarization` —
        ``[{"cluster_label", "start_time", "end_time"}, ...]``.
    appearances:
        Face appearance records for the same video (from
        :func:`pipeline.level2.face_runner.run_face_analysis`). Used as the
        identity evidence source.

    Returns
    -------
    list[SpeakerTurnRecord]
        One record per raw turn, in the same order, each resolved (or left
        unresolved) against the closed cast set present in *appearances*.
    """
    if not raw_turns:
        return []

    # Sort appearances by timestamp once so per-turn lookups are cheap.
    sorted_apps = sorted(appearances, key=lambda a: a.timestamp_s)

    turns: list[SpeakerTurnRecord] = []
    n_unresolved = 0
    n_single = 0
    n_majority = 0

    for raw in raw_turns:
        start = raw["start_time"]
        end = raw["end_time"]

        # Faces present anywhere inside [start, end].
        in_window = [
            a for a in sorted_apps
            if a.person_id is not None and start <= a.timestamp_s <= end
        ]

        person_id: str | None = None
        confidence: float | None = None
        resolution_method = "unresolved"

        if in_window:
            counts = Counter(a.person_id for a in in_window)
            distinct_persons = len(counts)
            top_person, top_count = counts.most_common(1)[0]
            total = sum(counts.values())

            person_id = top_person
            confidence = round(top_count / total, 4)

            if distinct_persons == 1:
                resolution_method = "single_candidate"
                n_single += 1
            else:
                resolution_method = "face_majority"
                n_majority += 1
        else:
            n_unresolved += 1

        turns.append(
            SpeakerTurnRecord(
                id=gen_id(),
                video_id=video_id,
                cluster_label=raw["cluster_label"],
                person_id=person_id,
                start_time=start,
                end_time=end,
                confidence=confidence,
                resolution_method=resolution_method,
            )
        )

    logger.info(
        "Speaker fusion complete for video_id=%s: %d turns "
        "(single_candidate=%d, face_majority=%d, unresolved=%d)",
        video_id, len(turns), n_single, n_majority, n_unresolved,
    )
    return turns
