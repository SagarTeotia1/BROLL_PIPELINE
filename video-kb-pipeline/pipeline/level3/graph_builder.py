from __future__ import annotations

import logging
from collections import defaultdict

from shared.types import (
    KGEdge,
    KGNode,
    KeyframeRecord,
    PersonRecord,
    QwenFrameOutput,
    ShotRecord,
)
from shared.utils import gen_id

logger = logging.getLogger(__name__)


def _make_node(
    video_id: str,
    node_type: str,
    ref_id: str,
    label: str | None = None,
    properties: dict | None = None,
) -> KGNode:
    return KGNode(
        id=gen_id(),
        video_id=video_id,
        node_type=node_type,
        ref_id=ref_id,
        label=label,
        properties=properties or {},
        embedding=None,
    )


def _make_edge(
    video_id: str,
    source_id: str,
    target_id: str,
    relation: str,
    weight: float = 1.0,
    properties: dict | None = None,
) -> KGEdge:
    return KGEdge(
        id=gen_id(),
        video_id=video_id,
        source_id=source_id,
        target_id=target_id,
        relation=relation,
        weight=weight,
        properties=properties or {},
    )


def build_knowledge_graph(
    video_id: str,
    frame_analyses: list[tuple[KeyframeRecord, QwenFrameOutput]],
    persons: list[PersonRecord],
    shots: list[ShotRecord],
) -> tuple[list[KGNode], list[KGEdge]]:
    """Build a knowledge graph from Qwen frame analysis outputs.

    Creates nodes for: Video, Shot, Scene, Frame, Person, Theme, and notable Objects.
    Creates edges for containment, appearance, causality, and relation triples
    extracted from the model output.

    Nodes are deduplicated by (node_type, ref_id). Edges are not deduplicated —
    duplicate edges from different frames are kept with their individual weights.

    Args:
        video_id: The video's unique identifier.
        frame_analyses: List of (KeyframeRecord, QwenFrameOutput) pairs (None outputs
            already filtered out by the caller).
        persons: All PersonRecord instances for this video.
        shots: All ShotRecord instances for this video.

    Returns:
        A tuple of (nodes, edges) — both as flat lists.
    """
    # -----------------------------------------------------------------------
    # Deduplication registry: (node_type, ref_id) → KGNode
    # -----------------------------------------------------------------------
    node_registry: dict[tuple[str, str], KGNode] = {}
    edges: list[KGEdge] = []

    def get_or_create_node(
        node_type: str,
        ref_id: str,
        label: str | None = None,
        properties: dict | None = None,
    ) -> KGNode:
        key = (node_type, ref_id)
        if key not in node_registry:
            node_registry[key] = _make_node(
                video_id=video_id,
                node_type=node_type,
                ref_id=ref_id,
                label=label,
                properties=properties or {},
            )
        else:
            # Merge properties if provided
            if properties:
                node_registry[key].properties.update(properties)
            if label and not node_registry[key].label:
                node_registry[key].label = label
        return node_registry[key]

    # -----------------------------------------------------------------------
    # 1. Video node
    # -----------------------------------------------------------------------
    video_node = get_or_create_node(
        node_type="Video",
        ref_id=video_id,
        label=video_id,
        properties={"video_id": video_id},
    )

    # -----------------------------------------------------------------------
    # 2. Shot nodes + Video → CONTAINS → Shot edges
    # -----------------------------------------------------------------------
    shots_sorted = sorted(shots, key=lambda s: s.shot_index)
    shot_nodes: dict[str, KGNode] = {}  # shot.id → KGNode

    for shot in shots_sorted:
        shot_node = get_or_create_node(
            node_type="Shot",
            ref_id=shot.id,
            label=f"Shot {shot.shot_index}",
            properties={
                "shot_index": shot.shot_index,
                "start_time": shot.start_time,
                "end_time": shot.end_time,
                "shot_type": shot.shot_type,
                "complexity": shot.complexity,
            },
        )
        shot_nodes[shot.id] = shot_node
        edges.append(
            _make_edge(
                video_id=video_id,
                source_id=video_node.id,
                target_id=shot_node.id,
                relation="CONTAINS",
            )
        )

    # Build a shot time-range index for matching keyframes to shots
    # shot.id → (start_time, end_time)
    shot_time_index: list[tuple[float, float, str]] = []
    for shot in shots_sorted:
        st = shot.start_time if shot.start_time is not None else 0.0
        et = shot.end_time if shot.end_time is not None else float("inf")
        shot_time_index.append((st, et, shot.id))

    def find_shot_id_for_timestamp(ts: float) -> str | None:
        for st, et, sid in shot_time_index:
            if st <= ts <= et:
                return sid
        return None

    # -----------------------------------------------------------------------
    # 3. Person nodes
    # -----------------------------------------------------------------------
    person_nodes: dict[str, KGNode] = {}  # person.pid → KGNode

    for person in persons:
        person_node = get_or_create_node(
            node_type="Person",
            ref_id=person.id,
            label=person.pid,
            properties={
                "pid": person.pid,
                "display_name": person.display_name or person.pid,
            },
        )
        person_nodes[person.pid] = person_node

    # -----------------------------------------------------------------------
    # 4. Scene, Frame, Theme, Object nodes + all relational edges
    # -----------------------------------------------------------------------

    # Track scene ordering: scene_id → first seen timestamp (for FOLLOWS edges)
    scene_first_seen: dict[str, float] = {}
    # scene_id → KGNode
    scene_nodes: dict[str, KGNode] = {}

    # Frame nodes in timestamp order (for consecutive CAUSES edges)
    frame_nodes_ordered: list[KGNode] = []

    for kf, output in sorted(frame_analyses, key=lambda x: x[0].timestamp_s):
        # -- Scene node --
        scene_id = output.scene_id
        if scene_id not in scene_nodes:
            scene_node = get_or_create_node(
                node_type="Scene",
                ref_id=scene_id,
                label=scene_id,
                properties={
                    "beat_type": output.story_beat.beat_type if output.story_beat else "",
                    "act_position": output.story_beat.act_position if output.story_beat else "",
                    "conflict_present": (
                        str(output.story_beat.conflict_present)
                        if output.story_beat
                        else ""
                    ),
                },
            )
            scene_nodes[scene_id] = scene_node
            scene_first_seen[scene_id] = kf.timestamp_s
        else:
            scene_node = scene_nodes[scene_id]

        # -- Frame node --
        frame_node = get_or_create_node(
            node_type="Frame",
            ref_id=kf.id,
            label=f"Frame @{kf.timestamp_s:.1f}s",
            properties={
                "caption": output.caption,
                "beat_type": output.story_beat.beat_type if output.story_beat else "",
                "scene_mood": (
                    output.emotional_tone.scene_mood if output.emotional_tone else ""
                ),
                "tension_level": (
                    output.emotional_tone.tension_level if output.emotional_tone else ""
                ),
                "timestamp_s": kf.timestamp_s,
            },
        )
        frame_nodes_ordered.append(frame_node)

        # Shot → CONTAINS → Frame
        shot_id = find_shot_id_for_timestamp(kf.timestamp_s)
        if shot_id and shot_id in shot_nodes:
            edges.append(
                _make_edge(
                    video_id=video_id,
                    source_id=shot_nodes[shot_id].id,
                    target_id=frame_node.id,
                    relation="CONTAINS",
                )
            )

        # Frame → HAS_ANALYSIS → Scene
        edges.append(
            _make_edge(
                video_id=video_id,
                source_id=frame_node.id,
                target_id=scene_node.id,
                relation="HAS_ANALYSIS",
            )
        )

        # Person → APPEARS_IN → Frame (from Qwen people list)
        for person_entry in output.people:
            pid = person_entry.pid
            if pid in person_nodes:
                edges.append(
                    _make_edge(
                        video_id=video_id,
                        source_id=person_nodes[pid].id,
                        target_id=frame_node.id,
                        relation="APPEARS_IN",
                        properties={
                            "emotion_inferred": person_entry.emotion_inferred,
                            "action": person_entry.action,
                        },
                    )
                )
            else:
                logger.debug(
                    "Person pid=%s from Qwen output not found in persons list — skipping edge.",
                    pid,
                )

        # -- Theme nodes + Frame → HAS_THEME → Theme edges --
        for theme_text in output.themes_symbols:
            if not theme_text:
                continue
            theme_node = get_or_create_node(
                node_type="Theme",
                ref_id=theme_text,
                label=theme_text,
                properties={"theme_text": theme_text},
            )
            edges.append(
                _make_edge(
                    video_id=video_id,
                    source_id=frame_node.id,
                    target_id=theme_node.id,
                    relation="HAS_THEME",
                )
            )

        # -- Important Object nodes (only those with a non-empty story_relevance) --
        for obj_entry in output.objects:
            if not obj_entry.story_relevance:
                continue
            # Use name as ref_id scoped to video — collisions mean same object type
            obj_ref_id = f"obj:{obj_entry.name.lower().strip()}"
            obj_node = get_or_create_node(
                node_type="Object",
                ref_id=obj_ref_id,
                label=obj_entry.name,
                properties={
                    "name": obj_entry.name,
                    "story_relevance": obj_entry.story_relevance,
                },
            )
            edges.append(
                _make_edge(
                    video_id=video_id,
                    source_id=frame_node.id,
                    target_id=obj_node.id,
                    relation="CONTAINS_OBJECT",
                    properties={"position": obj_entry.position},
                )
            )

        # -- Qwen relation triples: [subject, predicate, object] --
        for triple in (output.relations or []):
            if len(triple) != 3:
                continue
            subj, pred, obj = triple[0], triple[1], triple[2]

            # Resolve subject to a node
            if subj in person_nodes:
                src_node = person_nodes[subj]
            else:
                obj_ref = f"obj:{subj.lower().strip()}"
                src_node = get_or_create_node(
                    node_type="Object",
                    ref_id=obj_ref,
                    label=subj,
                    properties={"name": subj},
                )

            # Resolve object to a node
            if obj in person_nodes:
                tgt_node = person_nodes[obj]
            else:
                obj_ref = f"obj:{obj.lower().strip()}"
                tgt_node = get_or_create_node(
                    node_type="Object",
                    ref_id=obj_ref,
                    label=obj,
                    properties={"name": obj},
                )

            # Normalise predicate: uppercase with underscores
            relation_type = pred.upper().replace(" ", "_")

            edges.append(
                _make_edge(
                    video_id=video_id,
                    source_id=src_node.id,
                    target_id=tgt_node.id,
                    relation=relation_type,
                    properties={"source_frame": kf.id},
                )
            )

    # -----------------------------------------------------------------------
    # 5. Scene → FOLLOWS → Scene (sequential ordering by first-seen timestamp)
    # -----------------------------------------------------------------------
    scene_order = sorted(scene_first_seen.items(), key=lambda x: x[1])
    for i in range(len(scene_order) - 1):
        curr_scene_id = scene_order[i][0]
        next_scene_id = scene_order[i + 1][0]
        if curr_scene_id in scene_nodes and next_scene_id in scene_nodes:
            edges.append(
                _make_edge(
                    video_id=video_id,
                    source_id=scene_nodes[curr_scene_id].id,
                    target_id=scene_nodes[next_scene_id].id,
                    relation="FOLLOWS",
                )
            )

    # -----------------------------------------------------------------------
    # 6. Frame → CAUSES → Frame (consecutive frames, lightweight heuristic)
    #    Connect each frame to the next with weight 0.5 to represent implied
    #    causality. This is intentionally shallow — deeper causality is captured
    #    in the searchable_facts and Qwen causality fields.
    # -----------------------------------------------------------------------
    for i in range(len(frame_nodes_ordered) - 1):
        curr_frame_node = frame_nodes_ordered[i]
        next_frame_node = frame_nodes_ordered[i + 1]
        edges.append(
            _make_edge(
                video_id=video_id,
                source_id=curr_frame_node.id,
                target_id=next_frame_node.id,
                relation="CAUSES",
                weight=0.5,
            )
        )

    # -----------------------------------------------------------------------
    # Final assembly
    # -----------------------------------------------------------------------
    all_nodes = list(node_registry.values())

    logger.info(
        "KG built for video %s: %d nodes, %d edges",
        video_id,
        len(all_nodes),
        len(edges),
    )
    return all_nodes, edges
