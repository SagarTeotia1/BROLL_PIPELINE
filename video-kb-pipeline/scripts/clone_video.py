"""Clone a video's full KB record (Postgres + Neo4j + Pinecone) under a new video_id.

Usage:
    python scripts/clone_video.py SOURCE_VIDEO_ID [--count N] [--dry-run]

For each new UUID generated:
  1. Postgres — deep-copy every row FK-chained to the source video_id, with
     fresh row ids, into the same tables under the new video_id.
  2. Neo4j — re-derive KGNode/KGEdge objects from the freshly-cloned Postgres
     kg_nodes/kg_edges rows (ref_id already remapped) and write via the
     existing graph_writer.write_graph (same code path production uses).
  3. Pinecone — re-upsert the cloned searchable_facts (facts namespace) and
     copy the source video's scene vectors (scenes namespace, read directly
     from Pinecone since there's no scenes table in Postgres).
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncpg

from knowledge_base.postgres.client import get_pool, close_pool
from knowledge_base.postgres.queries import get_kg_nodes_for_video, get_kg_edges_for_video
from knowledge_base.pinecone_kb.client import get_index
from knowledge_base.neo4j.graph_writer import write_graph
from knowledge_base.neo4j.client import close_driver
from shared.types import KGNode, KGEdge

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("clone_video")


# ---------------------------------------------------------------------------
# Postgres deep copy
# ---------------------------------------------------------------------------

async def clone_postgres(pool: asyncpg.Pool, source_id: str, new_id: str) -> dict:
    """Deep-copy all rows for source_id into new_id inside one transaction.

    Uses per-connection temp tables to map old row id -> new row id so FK
    chains (chunk -> shot -> keyframe -> frame_analyses / searchable_facts,
    person -> face_*, kg_nodes -> kg_edges) stay internally consistent.
    Returns a dict of row counts copied per table for verification.
    """
    counts: dict[str, int] = {}
    async with pool.acquire() as conn:
        async with conn.transaction():
            # videos
            await conn.execute(
                """
                INSERT INTO videos (id, path, r2_key, duration_s, fps, width, height)
                SELECT $2, path, r2_key, duration_s, fps, width, height
                FROM videos WHERE id = $1
                """,
                uuid.UUID(source_id), uuid.UUID(new_id),
            )

            # chunks — map old chunk id -> new chunk id
            await conn.execute("CREATE TEMP TABLE map_chunks (old uuid, new uuid) ON COMMIT DROP")
            rows = await conn.fetch("SELECT id FROM chunks WHERE video_id = $1", uuid.UUID(source_id))
            chunk_map = {r["id"]: uuid.uuid4() for r in rows}
            if chunk_map:
                await conn.executemany(
                    "INSERT INTO map_chunks (old, new) VALUES ($1, $2)",
                    list(chunk_map.items()),
                )
            await conn.execute(
                """
                INSERT INTO chunks (id, video_id, chunk_index, start_frame, end_frame, start_time, end_time)
                SELECT m.new, $2, c.chunk_index, c.start_frame, c.end_frame, c.start_time, c.end_time
                FROM chunks c JOIN map_chunks m ON m.old = c.id
                WHERE c.video_id = $1
                """,
                uuid.UUID(source_id), uuid.UUID(new_id),
            )
            counts["chunks"] = len(chunk_map)

            # shots — map old shot id -> new shot id, chunk_id remapped via map_chunks
            await conn.execute("CREATE TEMP TABLE map_shots (old uuid, new uuid) ON COMMIT DROP")
            rows = await conn.fetch("SELECT id FROM shots WHERE video_id = $1", uuid.UUID(source_id))
            shot_map = {r["id"]: uuid.uuid4() for r in rows}
            if shot_map:
                await conn.executemany(
                    "INSERT INTO map_shots (old, new) VALUES ($1, $2)",
                    list(shot_map.items()),
                )
            await conn.execute(
                """
                INSERT INTO shots (id, chunk_id, video_id, shot_index, start_frame, end_frame,
                                   start_time, end_time, shot_type, complexity)
                SELECT ms.new, mc.new, $2, s.shot_index, s.start_frame, s.end_frame,
                       s.start_time, s.end_time, s.shot_type, s.complexity
                FROM shots s
                JOIN map_shots ms ON ms.old = s.id
                LEFT JOIN map_chunks mc ON mc.old = s.chunk_id
                WHERE s.video_id = $1
                """,
                uuid.UUID(source_id), uuid.UUID(new_id),
            )
            counts["shots"] = len(shot_map)

            # keyframes — map old keyframe id -> new keyframe id, shot_id remapped
            await conn.execute("CREATE TEMP TABLE map_keyframes (old uuid, new uuid) ON COMMIT DROP")
            rows = await conn.fetch("SELECT id FROM keyframes WHERE video_id = $1", uuid.UUID(source_id))
            kf_map = {r["id"]: uuid.uuid4() for r in rows}
            if kf_map:
                await conn.executemany(
                    "INSERT INTO map_keyframes (old, new) VALUES ($1, $2)",
                    list(kf_map.items()),
                )
            await conn.execute(
                """
                INSERT INTO keyframes (id, shot_id, video_id, frame_index, timestamp_s,
                                       r2_key, selection_reason, dino_embedding, siglip_embedding)
                SELECT mk.new, ms.new, $2, k.frame_index, k.timestamp_s,
                       k.r2_key, k.selection_reason, k.dino_embedding, k.siglip_embedding
                FROM keyframes k
                JOIN map_keyframes mk ON mk.old = k.id
                LEFT JOIN map_shots ms ON ms.old = k.shot_id
                WHERE k.video_id = $1
                """,
                uuid.UUID(source_id), uuid.UUID(new_id),
            )
            counts["keyframes"] = len(kf_map)

            # transcript_segments — chunk_id remapped, nullable
            r = await conn.execute(
                """
                INSERT INTO transcript_segments (id, video_id, chunk_id, text, start_time, end_time, confidence, words)
                SELECT gen_random_uuid(), $2, mc.new, t.text, t.start_time, t.end_time, t.confidence, t.words
                FROM transcript_segments t
                LEFT JOIN map_chunks mc ON mc.old = t.chunk_id
                WHERE t.video_id = $1
                """,
                uuid.UUID(source_id), uuid.UUID(new_id),
            )
            counts["transcript_segments"] = _exec_count(r)

            # persons — map old person id -> new person id
            await conn.execute("CREATE TEMP TABLE map_persons (old uuid, new uuid) ON COMMIT DROP")
            rows = await conn.fetch("SELECT id FROM persons WHERE video_id = $1", uuid.UUID(source_id))
            person_map = {r["id"]: uuid.uuid4() for r in rows}
            if person_map:
                await conn.executemany(
                    "INSERT INTO map_persons (old, new) VALUES ($1, $2)",
                    list(person_map.items()),
                )
            await conn.execute(
                """
                INSERT INTO persons (id, video_id, pid, display_name, arcface_embedding)
                SELECT mp.new, $2, p.pid, p.display_name, p.arcface_embedding
                FROM persons p JOIN map_persons mp ON mp.old = p.id
                WHERE p.video_id = $1
                """,
                uuid.UUID(source_id), uuid.UUID(new_id),
            )
            counts["persons"] = len(person_map)

            # face_appearances — person_id remapped (nullable)
            r = await conn.execute(
                """
                INSERT INTO face_appearances (id, video_id, frame_index, timestamp_s, person_id, track_id, bbox, emotion, emotion_conf)
                SELECT gen_random_uuid(), $2, f.frame_index, f.timestamp_s, mp.new, f.track_id, f.bbox, f.emotion, f.emotion_conf
                FROM face_appearances f
                LEFT JOIN map_persons mp ON mp.old = f.person_id
                WHERE f.video_id = $1
                """,
                uuid.UUID(source_id), uuid.UUID(new_id),
            )
            counts["face_appearances"] = _exec_count(r)

            # face_timeline_events — person_id remapped (nullable)
            r = await conn.execute(
                """
                INSERT INTO face_timeline_events (id, video_id, person_id, emotion, start_time, end_time, confidence)
                SELECT gen_random_uuid(), $2, mp.new, e.emotion, e.start_time, e.end_time, e.confidence
                FROM face_timeline_events e
                LEFT JOIN map_persons mp ON mp.old = e.person_id
                WHERE e.video_id = $1
                """,
                uuid.UUID(source_id), uuid.UUID(new_id),
            )
            counts["face_timeline_events"] = _exec_count(r)

            # color_grades — shot_id remapped (nullable)
            r = await conn.execute(
                """
                INSERT INTO color_grades (id, video_id, shot_id, frame_index, timestamp_s, parameters, style_tags)
                SELECT gen_random_uuid(), $2, ms.new, c.frame_index, c.timestamp_s, c.parameters, c.style_tags
                FROM color_grades c
                LEFT JOIN map_shots ms ON ms.old = c.shot_id
                WHERE c.video_id = $1
                """,
                uuid.UUID(source_id), uuid.UUID(new_id),
            )
            counts["color_grades"] = _exec_count(r)

            # frame_analyses — keyframe_id remapped
            await conn.execute("CREATE TEMP TABLE map_frame_analyses (old uuid, new uuid) ON COMMIT DROP")
            rows = await conn.fetch("SELECT id FROM frame_analyses WHERE video_id = $1", uuid.UUID(source_id))
            fa_map = {r["id"]: uuid.uuid4() for r in rows}
            if fa_map:
                await conn.executemany(
                    "INSERT INTO map_frame_analyses (old, new) VALUES ($1, $2)",
                    list(fa_map.items()),
                )
            await conn.execute(
                """
                INSERT INTO frame_analyses (id, keyframe_id, video_id, scene_id, scene_change,
                                            qwen_output, caption, beat_type, scene_mood, tension_level, tags)
                SELECT mfa.new, mk.new, $2, a.scene_id, a.scene_change,
                       a.qwen_output, a.caption, a.beat_type, a.scene_mood, a.tension_level, a.tags
                FROM frame_analyses a
                JOIN map_frame_analyses mfa ON mfa.old = a.id
                LEFT JOIN map_keyframes mk ON mk.old = a.keyframe_id
                WHERE a.video_id = $1
                """,
                uuid.UUID(source_id), uuid.UUID(new_id),
            )
            counts["frame_analyses"] = len(fa_map)

            # searchable_facts — frame_id (keyframe) remapped; new pinecone_id = new fact id
            await conn.execute("CREATE TEMP TABLE map_facts (old uuid, new uuid) ON COMMIT DROP")
            rows = await conn.fetch("SELECT id FROM searchable_facts WHERE video_id = $1", uuid.UUID(source_id))
            fact_map = {r["id"]: uuid.uuid4() for r in rows}
            if fact_map:
                await conn.executemany(
                    "INSERT INTO map_facts (old, new) VALUES ($1, $2)",
                    list(fact_map.items()),
                )
            await conn.execute(
                """
                INSERT INTO searchable_facts (id, video_id, frame_id, fact_text, timestamp_s, embedding, pinecone_id)
                SELECT mf.new, $2, mk.new, sf.fact_text, sf.timestamp_s, sf.embedding, mf.new::text
                FROM searchable_facts sf
                JOIN map_facts mf ON mf.old = sf.id
                LEFT JOIN map_keyframes mk ON mk.old = sf.frame_id
                WHERE sf.video_id = $1
                """,
                uuid.UUID(source_id), uuid.UUID(new_id),
            )
            counts["searchable_facts"] = len(fact_map)

            # kg_nodes — map old node id -> new node id, ref_id remapped per node_type
            await conn.execute("CREATE TEMP TABLE map_kg_nodes (old uuid, new uuid) ON COMMIT DROP")
            rows = await conn.fetch("SELECT id, node_type, ref_id FROM kg_nodes WHERE video_id = $1", uuid.UUID(source_id))
            node_map = {r["id"]: uuid.uuid4() for r in rows}
            if node_map:
                await conn.executemany(
                    "INSERT INTO map_kg_nodes (old, new) VALUES ($1, $2)",
                    list(node_map.items()),
                )

            def remap_ref_id(node_type: str, ref_id: str) -> str:
                if node_type == "Video":
                    return new_id
                if node_type == "Shot":
                    return str(shot_map.get(uuid.UUID(ref_id), ref_id))
                if node_type == "Person":
                    return str(person_map.get(uuid.UUID(ref_id), ref_id))
                if node_type == "Frame":
                    return str(kf_map.get(uuid.UUID(ref_id), ref_id))
                return ref_id  # Scene / Object / Theme — text refs, scoped by video_id already

            for r in rows:
                new_ref = remap_ref_id(r["node_type"], r["ref_id"])
                await conn.execute(
                    """
                    INSERT INTO kg_nodes (id, video_id, node_type, ref_id, label, properties, embedding)
                    SELECT $1, $2, node_type, $3, label, properties, embedding
                    FROM kg_nodes WHERE id = $4
                    """,
                    node_map[r["id"]], uuid.UUID(new_id), new_ref, r["id"],
                )
            counts["kg_nodes"] = len(node_map)

            # kg_edges — source_id/target_id remapped via map_kg_nodes
            r = await conn.execute(
                """
                INSERT INTO kg_edges (id, video_id, source_id, target_id, relation, weight, properties)
                SELECT gen_random_uuid(), $2, ms.new, mt.new, e.relation, e.weight, e.properties
                FROM kg_edges e
                JOIN map_kg_nodes ms ON ms.old = e.source_id
                JOIN map_kg_nodes mt ON mt.old = e.target_id
                WHERE e.video_id = $1
                """,
                uuid.UUID(source_id), uuid.UUID(new_id),
            )
            counts["kg_edges"] = _exec_count(r)

    return counts


def _exec_count(status: str) -> int:
    # asyncpg execute() returns "INSERT 0 N"
    try:
        return int(status.split()[-1])
    except Exception:
        return -1


# ---------------------------------------------------------------------------
# Neo4j clone — reuse the production graph_writer via freshly-cloned kg_nodes/edges
# ---------------------------------------------------------------------------

async def clone_neo4j(pool: asyncpg.Pool, new_id: str) -> tuple[int, int]:
    nodes = await get_kg_nodes_for_video(pool, new_id)
    edges = await get_kg_edges_for_video(pool, new_id)
    if nodes:
        await write_graph(new_id, nodes, edges)
    return len(nodes), len(edges)


# ---------------------------------------------------------------------------
# Pinecone clone — facts from cloned Postgres rows, scenes copied directly from Pinecone
# ---------------------------------------------------------------------------

async def clone_pinecone_facts(pool: asyncpg.Pool, new_id: str) -> int:
    from knowledge_base.pinecone_kb.indexer import upsert_facts
    from knowledge_base.postgres.queries import _vec_to_list
    from shared.types import SearchableFactRecord

    rows = await pool.fetch(
        "SELECT id, video_id, frame_id, fact_text, timestamp_s, embedding, pinecone_id "
        "FROM searchable_facts WHERE video_id = $1",
        uuid.UUID(new_id),
    )
    facts = [
        SearchableFactRecord(
            id=str(r["id"]),
            video_id=str(r["video_id"]),
            frame_id=str(r["frame_id"]) if r["frame_id"] else None,
            fact_text=r["fact_text"],
            timestamp_s=r["timestamp_s"],
            embedding=_vec_to_list(r["embedding"]),
            pinecone_id=r["pinecone_id"],
        )
        for r in rows
    ]
    if not facts:
        return 0
    return await asyncio.to_thread(upsert_facts, facts)


def clone_pinecone_scenes(source_id: str, new_id: str) -> int:
    """Copy the 'scenes' namespace vectors for source_id -> new_id, new vector ids."""
    index = get_index()
    matches = []
    # Pinecone has no "list by metadata" primitive on pod-free serverless without a
    # query vector, so query with a zero vector + metadata filter and a large top_k.
    resp = index.query(
        vector=[0.0] * 1024,
        filter={"video_id": {"$eq": source_id}},
        top_k=10000,
        include_values=True,
        include_metadata=True,
        namespace="scenes",
    )
    matches = resp.get("matches", []) if isinstance(resp, dict) else resp.matches
    if not matches:
        return 0
    vectors = []
    for m in matches:
        md = dict(m["metadata"] if isinstance(m, dict) else m.metadata)
        md["video_id"] = new_id
        vectors.append({
            "id": str(uuid.uuid4()),
            "values": m["values"] if isinstance(m, dict) else m.values,
            "metadata": md,
        })
    index.upsert(vectors=vectors, namespace="scenes")
    return len(vectors)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

async def clone_one(pool: asyncpg.Pool, source_id: str, new_id: str, dry_run: bool) -> None:
    if dry_run:
        logger.info("[dry-run] would clone %s -> %s", source_id, new_id)
        return
    pg_counts = await clone_postgres(pool, source_id, new_id)
    logger.info("Postgres clone %s -> %s: %s", source_id, new_id, pg_counts)

    n_nodes, n_edges = await clone_neo4j(pool, new_id)
    logger.info("Neo4j clone %s: %d nodes, %d edges", new_id, n_nodes, n_edges)

    n_facts = await clone_pinecone_facts(pool, new_id)
    n_scenes = await asyncio.to_thread(clone_pinecone_scenes, source_id, new_id)
    logger.info("Pinecone clone %s: %d fact vectors, %d scene vectors", new_id, n_facts, n_scenes)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source_video_id")
    ap.add_argument("--count", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--resume-pinecone-only", metavar="NEW_VIDEO_ID", default=None,
                     help="Skip Postgres/Neo4j clone; a target video_id already cloned in Postgres — just finish the Pinecone step for it.")
    args = ap.parse_args()

    pool = await get_pool()
    try:
        exists = await pool.fetchval("SELECT 1 FROM videos WHERE id = $1", uuid.UUID(args.source_video_id))
        if not exists:
            logger.error("source video_id %s not found in Postgres — aborting", args.source_video_id)
            return

        if args.resume_pinecone_only:
            new_id = args.resume_pinecone_only
            n_facts = await clone_pinecone_facts(pool, new_id)
            n_scenes = await asyncio.to_thread(clone_pinecone_scenes, args.source_video_id, new_id)
            logger.info("Pinecone clone %s: %d fact vectors, %d scene vectors", new_id, n_facts, n_scenes)
            return

        new_ids = [str(uuid.uuid4()) for _ in range(args.count)]
        logger.info("Cloning %s into %d new video_id(s): %s", args.source_video_id, args.count, new_ids)

        for new_id in new_ids:
            await clone_one(pool, args.source_video_id, new_id, args.dry_run)

        if not args.dry_run:
            print("\nNew video_id(s):")
            for nid in new_ids:
                print(nid)
    finally:
        await close_pool()
        await close_driver()


if __name__ == "__main__":
    asyncio.run(main())
