"""Level-3 knowledge-base finalizer.

Converts raw Qwen VL frame outputs into the full set of Level-3 KB records:
  - FrameAnalysisRecord  → postgres frame_analyses table
  - SearchableFactRecord → postgres searchable_facts + Pinecone facts namespace
  - KGNode / KGEdge      → postgres kg_nodes / kg_edges tables (structured cache)
  - Neo4j graph          → primary graph traversal store (Cypher queries)

Text embeddings for searchable facts and scene captions are generated locally
via the BAAI/bge-large-en-v1.5 model (1024-dim) loaded by LocalEmbedder on the
same GPU as Qwen. Embeddings are computed before DB insert so no per-fact update
round-trips are needed.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import re

import asyncpg

from knowledge_base.postgres.queries import (
    bulk_insert_frame_analyses,
    bulk_insert_kg_edges,
    bulk_insert_searchable_facts,
    bulk_upsert_kg_nodes,
)
from shared.types import (
    FrameAnalysisRecord,
    KGEdge,
    KGNode,
    KeyframeRecord,
    PersonRecord,
    QwenFrameOutput,
    SearchableFactRecord,
    ShotRecord,
)
from shared.utils import gen_id

logger = logging.getLogger(__name__)

_EMBED_BATCH_SIZE = 128


# ---------------------------------------------------------------------------
# Embedding helper — local GPU model (BAAI/bge-large-en-v1.5, 1024-dim)
# ---------------------------------------------------------------------------


async def _embed_texts_batched(
    texts: list[str],
    batch_size: int = _EMBED_BATCH_SIZE,
) -> list[list[float] | None]:
    """Embed texts using the local GPU embedder, returning results in order.

    Wraps the synchronous LocalEmbedder.embed() in asyncio.to_thread so the
    event loop is not blocked during GPU inference.  Returns None for each
    position if the embedder is unavailable or raises.
    """
    if not texts:
        return []
    try:
        from models.embed_model import LocalEmbedder  # type: ignore[import]

        embedder = LocalEmbedder.get()
        embeddings: list[list[float]] = await asyncio.to_thread(
            embedder.embed, texts, batch_size
        )
        return embeddings  # type: ignore[return-value]
    except Exception as exc:
        logger.warning("Local embedding failed (%d texts): %s", len(texts), exc)
        return [None] * len(texts)


# ---------------------------------------------------------------------------
# KG node / edge builders — rich semantic graph
# ---------------------------------------------------------------------------

_INTERACTION_RELATION = {
    "address":     "ADDRESSES",
    "confront":    "CONFRONTS",
    "collaborate": "COLLABORATES",
    "observe":     "OBSERVES",
    "ignore":      "IGNORES",
}

_TENSION_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}


def _build_kg_nodes_for_frame(
    output: QwenFrameOutput,
    keyframe: KeyframeRecord,
    video_id: str,
    persons: list[PersonRecord],
) -> list[KGNode]:
    pid_to_person: dict[str, PersonRecord] = {p.pid: p for p in persons}
    nodes: list[KGNode] = []

    mood    = output.emotional_tone.scene_mood    if output.emotional_tone else ""
    tension = output.emotional_tone.tension_level if output.emotional_tone else ""
    beat    = output.story_beat.beat_type         if output.story_beat     else ""

    nodes.append(KGNode(
        id=gen_id(), video_id=video_id,
        node_type="Frame", ref_id=keyframe.id,
        label=output.caption[:120] if output.caption else f"Frame@{keyframe.timestamp_s:.1f}s",
        properties={
            "timestamp_s":     keyframe.timestamp_s,
            "scene_id":        output.scene_id,
            "location_id":     output.location_id or "",
            "dominant_person": output.dominant_person or "",
            "beat_type":       beat,
            "scene_mood":      mood,
            "tension_level":   tension,
            "dialogue":        output.dialogue_subtitle or "",
            "tags":            output.tags or [],
        },
        embedding=None,
    ))

    if output.scene_id:
        nodes.append(KGNode(
            id=gen_id(), video_id=video_id,
            node_type="Scene", ref_id=output.scene_id,
            label=output.scene_id,
            properties={
                "first_timestamp_s": keyframe.timestamp_s,
                "beat_type":         beat,
                "location_id":       output.location_id or "",
                "mood":              mood,
                "tension_level":     tension,
                "conflict_present":  str(output.story_beat.conflict_present) if output.story_beat else "false",
            },
            embedding=None,
        ))

    loc_id = (output.location_id or "").strip()
    if loc_id:
        nodes.append(KGNode(
            id=gen_id(), video_id=video_id,
            node_type="Location", ref_id=f"loc:{loc_id}",
            label=loc_id.replace("_", " ").title(),
            properties={
                "location_type":     output.setting.location_type    if output.setting else "",
                "specific_location": output.setting.specific_location if output.setting else "",
                "indoor_outdoor":    output.setting.indoor_outdoor   if output.setting else "",
            },
            embedding=None,
        ))

    if mood:
        nodes.append(KGNode(
            id=gen_id(), video_id=video_id,
            node_type="Emotion", ref_id=f"emotion:{mood.lower()}",
            label=mood,
            properties={"tension_level": tension},
            embedding=None,
        ))

    for pe in output.people or []:
        person = pid_to_person.get(pe.pid)
        if person is None:
            logger.debug("Qwen pid=%s not in PersonRecord list — skipping node", pe.pid)
            continue
        nodes.append(KGNode(
            id=gen_id(), video_id=video_id,
            node_type="Person", ref_id=person.id,
            label=person.display_name or person.pid,
            properties={
                "pid":              person.pid,
                "display_name":     person.display_name or person.pid,
                "story_role":       pe.story_role,
                "emotion_inferred": pe.emotion_inferred,
                "emotion_source":   pe.emotion_source,
                "apparent_goal":    pe.apparent_goal,
                "action":           pe.action,
                "pose":             pe.pose,
                "gaze":             pe.gaze,
                "clothing":         pe.clothing,
            },
            embedding=None,
        ))
        if pe.emotion_inferred:
            nodes.append(KGNode(
                id=gen_id(), video_id=video_id,
                node_type="Emotion", ref_id=f"emotion:{pe.emotion_inferred.lower()}",
                label=pe.emotion_inferred,
                properties={},
                embedding=None,
            ))

    for obj in output.objects or []:
        if not obj.story_relevance:
            continue
        nodes.append(KGNode(
            id=gen_id(), video_id=video_id,
            node_type="Object", ref_id=f"obj:{obj.name.lower().strip()}",
            label=obj.name,
            properties={"story_relevance": obj.story_relevance},
            embedding=None,
        ))

    for theme in output.themes_symbols or []:
        if not theme:
            continue
        nodes.append(KGNode(
            id=gen_id(), video_id=video_id,
            node_type="Theme", ref_id=f"theme:{theme.lower().strip()}",
            label=theme,
            properties={},
            embedding=None,
        ))

    return nodes


def _build_kg_edges_for_frame(
    output: QwenFrameOutput,
    keyframe: KeyframeRecord,
    video_id: str,
    node_ref_to_db_id: dict[tuple[str, str], str],
    object_first_seen: dict[str, str],
    person_first_seen: dict[str, str],
) -> list[KGEdge]:
    edges: list[KGEdge] = []
    ts = keyframe.timestamp_s

    frame_db_id = node_ref_to_db_id.get(("Frame",    keyframe.id))
    scene_db_id = node_ref_to_db_id.get(("Scene",    output.scene_id))    if output.scene_id        else None
    loc_ref     = f"loc:{output.location_id.strip()}"                      if output.location_id      else None
    loc_db_id   = node_ref_to_db_id.get(("Location", loc_ref))             if loc_ref                 else None
    mood        = (output.emotional_tone.scene_mood or "").lower()          if output.emotional_tone   else ""
    emo_db_id   = node_ref_to_db_id.get(("Emotion", f"emotion:{mood}"))    if mood                    else None

    def edge(src, tgt, rel, weight=1.0, props=None):
        if src and tgt and src != tgt:
            edges.append(KGEdge(
                id=gen_id(), video_id=video_id,
                source_id=src, target_id=tgt,
                relation=rel, weight=weight,
                properties=props or {},
            ))

    edge(frame_db_id, scene_db_id, "PART_OF_SCENE", props={"timestamp_s": ts})
    edge(frame_db_id, loc_db_id,   "IN_LOCATION",   props={"timestamp_s": ts})
    edge(scene_db_id, emo_db_id,   "HAS_MOOD",
         props={"tension_level": output.emotional_tone.tension_level if output.emotional_tone else ""})

    pid_to_person_ref: dict[str, str] = {}

    for pe in output.people or []:
        p_db_id = node_ref_to_db_id.get(("person_pid", pe.pid))
        if not p_db_id:
            continue

        p_ref = next((k[1] for k in node_ref_to_db_id if k[0] == "Person" and node_ref_to_db_id[k] == p_db_id), None)
        if p_ref:
            pid_to_person_ref[pe.pid] = p_ref
            if p_ref not in person_first_seen and frame_db_id:
                person_first_seen[p_ref] = frame_db_id
                edge(p_db_id, frame_db_id, "FIRST_SEEN_IN", props={"timestamp_s": ts})

        edge(p_db_id, frame_db_id, "APPEARS_IN",
             props={"emotion": pe.emotion_inferred, "emotion_source": pe.emotion_source,
                    "action": pe.action, "story_role": pe.story_role, "timestamp_s": ts})
        edge(p_db_id, scene_db_id, "PRESENT_IN_SCENE",
             props={"timestamp_s": ts, "action": pe.action})

        if output.dialogue_subtitle and output.dominant_person == pe.pid:
            edge(p_db_id, frame_db_id, "SPEAKS_IN",
                 props={"dialogue": output.dialogue_subtitle[:200], "timestamp_s": ts})

        if pe.emotion_inferred:
            emo_p_db_id = node_ref_to_db_id.get(("Emotion", f"emotion:{pe.emotion_inferred.lower()}"))
            edge(p_db_id, emo_p_db_id, "EXPRESSES",
                 props={"emotion_source": pe.emotion_source,
                        "confidence": 1.0 if pe.emotion_source == "visual_read" else 0.6,
                        "timestamp_s": ts})

    if output.dominant_person:
        dom_db_id = node_ref_to_db_id.get(("person_pid", output.dominant_person))
        edge(dom_db_id, scene_db_id, "DRIVES_SCENE",
             props={"timestamp_s": ts,
                    "beat_type": output.story_beat.beat_type if output.story_beat else ""})

    for ix in output.interactions or []:
        if not ix.is_valid:
            continue
        p1_db_id = node_ref_to_db_id.get(("person_pid", ix.p1))
        p2_db_id = node_ref_to_db_id.get(("person_pid", ix.p2))
        rel = _INTERACTION_RELATION.get(ix.type, "INTERACTS_WITH")
        edge(p1_db_id, p2_db_id, rel,
             props={"notes": ix.notes, "timestamp_s": ts, "frame_id": keyframe.id})
        edge(p1_db_id, p2_db_id, "CO_OCCURS_WITH", weight=1.0,
             props={"timestamp_s": ts})

    visible_pids = [pe.pid for pe in (output.people or [])]
    for i in range(len(visible_pids)):
        for j in range(i + 1, len(visible_pids)):
            pa_db_id = node_ref_to_db_id.get(("person_pid", visible_pids[i]))
            pb_db_id = node_ref_to_db_id.get(("person_pid", visible_pids[j]))
            edge(pa_db_id, pb_db_id, "CO_OCCURS_WITH", weight=1.0,
                 props={"timestamp_s": ts})

    for obj in output.objects or []:
        if not obj.story_relevance:
            continue
        obj_ref   = f"obj:{obj.name.lower().strip()}"
        obj_db_id = node_ref_to_db_id.get(("Object", obj_ref))
        if not obj_db_id or not frame_db_id:
            continue
        edge(frame_db_id, obj_db_id, "CONTAINS_OBJECT",
             props={"position": obj.position, "story_relevance": obj.story_relevance,
                    "timestamp_s": ts})
        if obj_ref not in object_first_seen:
            object_first_seen[obj_ref] = frame_db_id
            edge(obj_db_id, frame_db_id, "INTRODUCED_IN", props={"timestamp_s": ts})
        for pe in output.people or []:
            if obj.name.lower() in (pe.action or "").lower():
                p_db_id = node_ref_to_db_id.get(("person_pid", pe.pid))
                edge(obj_db_id, p_db_id, "HELD_BY",
                     props={"frame_id": keyframe.id, "timestamp_s": ts})

    for theme in output.themes_symbols or []:
        if not theme:
            continue
        theme_db_id = node_ref_to_db_id.get(("Theme", f"theme:{theme.lower().strip()}"))
        edge(frame_db_id, theme_db_id, "ILLUSTRATES_THEME", props={"timestamp_s": ts})
        edge(scene_db_id, theme_db_id, "THEME_PRESENT",     props={"timestamp_s": ts})

    for rel in output.get_relations_structured():
        subj_id = (node_ref_to_db_id.get(("person_pid", rel.subject))
                   or node_ref_to_db_id.get(("Object", f"obj:{rel.subject.lower().strip()}"))
                   or frame_db_id)
        obj_id  = (node_ref_to_db_id.get(("person_pid", rel.object))
                   or node_ref_to_db_id.get(("Object", f"obj:{rel.object.lower().strip()}"))
                   or scene_db_id)
        if subj_id and obj_id and subj_id != obj_id:
            rel_name = re.sub(r"[^A-Z0-9_]", "_", rel.predicate.upper())[:64].strip("_") or "RELATES_TO"
            edge(subj_id, obj_id, rel_name,
                 weight=float(rel.confidence),
                 props={"subject": rel.subject, "predicate": rel.predicate,
                        "object": rel.object, "frame_id": keyframe.id, "timestamp_s": ts})

    return edges


def _make_frame_analysis_record(
    kf: KeyframeRecord,
    output: QwenFrameOutput,
    video_id: str,
) -> FrameAnalysisRecord:
    return FrameAnalysisRecord(
        id=gen_id(),
        keyframe_id=kf.id,
        video_id=video_id,
        scene_id=output.scene_id,
        scene_change=output.scene_change,
        qwen_output=output.model_dump(),
        caption=output.caption,
        beat_type=output.story_beat.beat_type if output.story_beat else None,
        scene_mood=(output.emotional_tone.scene_mood if output.emotional_tone else None),
        tension_level=(output.emotional_tone.tension_level if output.emotional_tone else None),
        tags=output.tags or [],
    )


def _build_frame_analysis_records(
    frame_results: list[tuple[KeyframeRecord, QwenFrameOutput | None]],
) -> list[FrameAnalysisRecord]:
    """Build FrameAnalysisRecord objects from frame results, skipping None outputs.

    Used for per-chunk DB checkpointing during long-video Qwen inference so that
    progress is not lost if the container is interrupted mid-run.
    """
    records: list[FrameAnalysisRecord] = []
    for kf, output in frame_results:
        if output is None:
            continue
        try:
            records.append(_make_frame_analysis_record(kf, output, kf.video_id))
        except Exception as exc:
            logger.error("_build_frame_analysis_records: failed for keyframe %s: %s", kf.id, exc)
    return records


async def finalize_level3_kb(
    pool: asyncpg.Pool,
    video_id: str,
    frame_results: list[tuple[KeyframeRecord, QwenFrameOutput | None]],
    persons: list[PersonRecord],
    shots: list[ShotRecord],
) -> None:
    """Persist all Level-3 outputs to Postgres, Pinecone, and Neo4j.

    Processing order:
    1. Collect all analyses, facts, scene captions, KG nodes from all frames
    2. Bulk-upsert frame analyses to Postgres
    3. Embed facts + scene captions in parallel (single GPU pass each)
    4. Bulk-insert searchable_facts with embeddings included (no update loop)
    5. Upsert facts + scenes to Pinecone
    6. Bulk-upsert KG nodes → build edges → bulk-insert KG edges
    7. Write Neo4j graph (background, non-blocking to caller)
    """
    valid_results: list[tuple[KeyframeRecord, QwenFrameOutput]] = [
        (kf, out) for kf, out in frame_results if out is not None
    ]

    logger.info(
        "finalize_level3_kb: %d/%d valid frames for video_id=%s",
        len(valid_results),
        len(frame_results),
        video_id,
    )

    if not valid_results:
        logger.warning("finalize_level3_kb: no valid frame results — nothing to write.")
        return

    all_facts: list[SearchableFactRecord] = []
    scene_pinecone_records: dict[str, dict] = {}
    all_nodes: list[KGNode] = []
    all_analyses: list[FrameAnalysisRecord] = []
    scene_meta: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Pass 1: collect analyses, facts, scene captions, KG nodes
    # ------------------------------------------------------------------
    sorted_results = sorted(valid_results, key=lambda x: x[0].timestamp_s)

    for kf, output in sorted_results:
        try:
            all_analyses.append(_make_frame_analysis_record(kf, output, video_id))
        except Exception as exc:
            logger.error("_make_frame_analysis_record failed for keyframe %s: %s", kf.id, exc)

        for fact_text in output.searchable_facts or []:
            stripped = (fact_text or "").strip()
            if not stripped:
                continue
            fact_id = hashlib.sha256(f"{video_id}:{kf.id}:{stripped}".encode()).hexdigest()[:32]
            all_facts.append(SearchableFactRecord(
                id=fact_id, video_id=video_id, frame_id=kf.id,
                fact_text=stripped, timestamp_s=kf.timestamp_s,
                embedding=None, pinecone_id=fact_id,
            ))

        if output.scene_id and output.scene_id not in scene_pinecone_records:
            scene_pinecone_records[output.scene_id] = {
                "id": f"{video_id}::{output.scene_id}",
                "video_id": video_id, "scene_id": output.scene_id,
                "caption": output.caption or "", "embedding": None,
                "metadata": {
                    "first_timestamp_s": kf.timestamp_s,
                    "beat_type": output.story_beat.beat_type if output.story_beat else "",
                    "location_id": output.location_id or "",
                    "mood": output.emotional_tone.scene_mood if output.emotional_tone else "",
                    "tension_level": output.emotional_tone.tension_level if output.emotional_tone else "",
                },
            }

        if output.scene_id and output.scene_id not in scene_meta:
            scene_meta[output.scene_id] = {
                "first_ts": kf.timestamp_s,
                "location_id": output.location_id or "",
                "mood": output.emotional_tone.scene_mood if output.emotional_tone else "",
                "tension": output.emotional_tone.tension_level if output.emotional_tone else "",
            }

        try:
            all_nodes.extend(_build_kg_nodes_for_frame(output, kf, video_id, persons))
        except Exception as exc:
            logger.error("_build_kg_nodes_for_frame failed for keyframe %s: %s", kf.id, exc)

    # ------------------------------------------------------------------
    # Step 1: Bulk-upsert frame analyses
    # ------------------------------------------------------------------
    if all_analyses:
        try:
            await bulk_insert_frame_analyses(pool, all_analyses)
            logger.info("Bulk-inserted %d frame analyses", len(all_analyses))
        except Exception as exc:
            logger.error("bulk_insert_frame_analyses failed: %s", exc)

    # ------------------------------------------------------------------
    # Step 2: Embed facts + scene captions in parallel, THEN insert
    # This eliminates the per-fact update loop: embed once, insert with vectors.
    # ------------------------------------------------------------------
    fact_texts    = [f.fact_text for f in all_facts]
    scene_list    = list(scene_pinecone_records.values())
    scene_captions = [s["caption"] for s in scene_list]

    logger.info(
        "Embedding %d facts + %d scene captions in parallel", len(fact_texts), len(scene_captions)
    )

    fact_embeddings, scene_embeddings = await asyncio.gather(
        _embed_texts_batched(fact_texts),
        _embed_texts_batched(scene_captions),
    )

    # Attach embeddings to facts before DB insert
    for fact, emb in zip(all_facts, fact_embeddings):
        if emb is not None:
            fact.embedding = emb

    # ------------------------------------------------------------------
    # Step 3: Single bulk insert — facts already carry embeddings
    # ------------------------------------------------------------------
    if all_facts:
        try:
            await bulk_insert_searchable_facts(pool, all_facts)
            logger.info("Bulk-inserted %d searchable facts (with embeddings)", len(all_facts))
        except Exception as exc:
            logger.error("bulk_insert_searchable_facts failed: %s", exc)

    # ------------------------------------------------------------------
    # Step 4: Pinecone upserts — facts + scenes
    # ------------------------------------------------------------------
    from knowledge_base.pinecone_kb.indexer import upsert_facts, upsert_scenes

    embedded_facts = [f for f in all_facts if f.embedding is not None]
    if embedded_facts:
        try:
            upserted = await asyncio.to_thread(upsert_facts, embedded_facts)
            logger.info("Upserted %d fact vectors to Pinecone", upserted)
        except Exception as exc:
            logger.error("Pinecone upsert_facts failed: %s", exc)

    embedded_scene_records = [
        {**rec, "embedding": emb}
        for rec, emb in zip(scene_list, scene_embeddings)
        if emb is not None
    ]
    if embedded_scene_records:
        try:
            upserted = await asyncio.to_thread(upsert_scenes, embedded_scene_records)
            logger.info("Upserted %d scene vectors to Pinecone", upserted)
        except Exception as exc:
            logger.error("Pinecone upsert_scenes failed: %s", exc)

    # ------------------------------------------------------------------
    # Step 5: KG nodes → edges → Postgres
    # ------------------------------------------------------------------
    node_ref_to_db_id: dict[tuple[str, str], str] = {}
    if all_nodes:
        try:
            node_ref_to_db_id = await bulk_upsert_kg_nodes(pool, all_nodes)
            for node in all_nodes:
                db_id = node_ref_to_db_id.get((node.node_type, node.ref_id))
                if db_id:
                    node.id = db_id
            for node in all_nodes:
                if node.node_type == "Person":
                    pid_val = node.properties.get("pid", "")
                    if pid_val:
                        db_id = node_ref_to_db_id.get((node.node_type, node.ref_id))
                        if db_id:
                            node_ref_to_db_id[("person_pid", pid_val)] = db_id
            logger.info("Bulk-upserted %d KG nodes", len(all_nodes))
        except Exception as exc:
            logger.error("bulk_upsert_kg_nodes failed: %s", exc)

    all_edges: list[KGEdge] = []
    object_first_seen: dict[str, str] = {}
    person_first_seen: dict[str, str] = {}

    for kf, output in sorted_results:
        try:
            edges = _build_kg_edges_for_frame(
                output, kf, video_id, node_ref_to_db_id,
                object_first_seen, person_first_seen,
            )
            all_edges.extend(edges)
        except Exception as exc:
            logger.error("_build_kg_edges_for_frame failed for keyframe %s: %s", kf.id, exc)

    # Cross-scene edges: FOLLOWS, EMOTIONAL_SHIFT, SAME_LOCATION
    scenes_by_time = sorted(scene_meta.items(), key=lambda x: x[1]["first_ts"])
    for i in range(len(scenes_by_time) - 1):
        s_id, s_info = scenes_by_time[i]
        n_id, n_info = scenes_by_time[i + 1]
        s_db = node_ref_to_db_id.get(("Scene", s_id))
        n_db = node_ref_to_db_id.get(("Scene", n_id))
        if not s_db or not n_db:
            continue

        gap_s = n_info["first_ts"] - s_info["first_ts"]
        all_edges.append(KGEdge(
            id=gen_id(), video_id=video_id,
            source_id=s_db, target_id=n_db,
            relation="FOLLOWS",
            weight=round(1.0 / max(gap_s, 1.0), 4),
            properties={"gap_s": round(gap_s, 2),
                        "from_ts": s_info["first_ts"], "to_ts": n_info["first_ts"]},
        ))

        mood_changed = s_info["mood"] != n_info["mood"] and s_info["mood"] and n_info["mood"]
        s_t = _TENSION_ORDER.get(s_info["tension"], -1)
        n_t = _TENSION_ORDER.get(n_info["tension"], -1)
        tension_delta = n_t - s_t if s_t >= 0 and n_t >= 0 else 0
        if mood_changed or abs(tension_delta) >= 1:
            all_edges.append(KGEdge(
                id=gen_id(), video_id=video_id,
                source_id=s_db, target_id=n_db,
                relation="EMOTIONAL_SHIFT",
                weight=abs(tension_delta) / 3.0 if tension_delta else 0.5,
                properties={"from_mood": s_info["mood"], "to_mood": n_info["mood"],
                            "tension_delta": tension_delta,
                            "from_tension": s_info["tension"], "to_tension": n_info["tension"]},
            ))

        if s_info["location_id"] and s_info["location_id"] == n_info["location_id"]:
            all_edges.append(KGEdge(
                id=gen_id(), video_id=video_id,
                source_id=s_db, target_id=n_db,
                relation="SAME_LOCATION",
                weight=1.0,
                properties={"location_id": s_info["location_id"]},
            ))

    logger.info("Built %d cross-scene edges", len(scenes_by_time))

    if all_edges:
        try:
            await bulk_insert_kg_edges(pool, all_edges)
            logger.info("Bulk-inserted %d KG edges", len(all_edges))
        except Exception as exc:
            logger.error("bulk_insert_kg_edges failed: %s", exc)

    # ------------------------------------------------------------------
    # Step 6: Neo4j graph — awaited so container doesn't exit before write
    # ------------------------------------------------------------------
    if all_nodes:
        try:
            from knowledge_base.neo4j.graph_writer import write_graph  # type: ignore[import]
            await write_graph(video_id, all_nodes, all_edges)
            logger.info(
                "Neo4j graph written: %d nodes, %d edges", len(all_nodes), len(all_edges)
            )
        except Exception as exc:
            logger.error("Neo4j write failed (non-fatal): %s", exc)

    logger.info(
        "finalize_level3_kb complete for video_id=%s — "
        "frames=%d facts=%d (pinecone=%d) scenes=%d (pinecone=%d) "
        "kg_nodes=%d kg_edges=%d",
        video_id,
        len(valid_results),
        len(all_facts),
        len(embedded_facts),
        len(scene_list),
        len(embedded_scene_records),
        len(all_nodes),
        len(all_edges),
    )
