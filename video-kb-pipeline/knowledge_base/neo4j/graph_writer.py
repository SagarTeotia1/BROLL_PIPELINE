from __future__ import annotations

import asyncio
import logging
import re
from collections import defaultdict

from shared.types import KGNode, KGEdge
from knowledge_base.neo4j.client import neo4j_session, run_cypher
from shared.config import settings

logger = logging.getLogger(__name__)

_NODE_BATCH_SIZE = 500
_EDGE_BATCH_SIZE = 500


async def ensure_constraints(database: str | None = None) -> None:
    """Create uniqueness constraints for each node type. Run once at startup."""
    node_types = ["Video", "Shot", "Frame", "Scene", "Person", "Object", "Theme", "Chunk", "Location", "Emotion"]
    async with neo4j_session(database) as session:
        for node_type in node_types:
            # Unique constraint on (node_type, ref_id) — idempotent
            try:
                await session.run(
                    f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{node_type}) REQUIRE (n.ref_id, n.video_id) IS UNIQUE"
                )
            except Exception as e:
                logger.debug("Constraint for %s: %s", node_type, e)


async def write_nodes(nodes: list[KGNode], database: str | None = None) -> dict[tuple[str, str], str]:
    """Upsert all nodes using UNWIND batches (100 nodes per transaction, grouped by label).

    Returns map of (node_type, ref_id) -> neo4j element_id string.
    Falls back to individual MERGE on batch failure so no nodes are silently dropped.
    """
    node_ref_to_element_id: dict[tuple[str, str], str] = {}

    # Group nodes by label — UNWIND requires a uniform label per batch.
    nodes_by_type: dict[str, list[KGNode]] = defaultdict(list)
    for node in nodes:
        nodes_by_type[node.node_type].append(node)

    async def _write_node_type(node_type: str, typed_nodes: list[KGNode]) -> dict[tuple[str, str], str]:
        result_map: dict[tuple[str, str], str] = {}
        async with neo4j_session(database) as session:
            for i in range(0, len(typed_nodes), _NODE_BATCH_SIZE):
                batch = typed_nodes[i : i + _NODE_BATCH_SIZE]
                batch_props: list[dict] = []
                for node in batch:
                    props: dict = {
                        "ref_id": node.ref_id,
                        "video_id": node.video_id,
                        "db_id": node.id,
                        "label": node.label or "",
                    }
                    for k, v in node.properties.items():
                        if isinstance(v, (str, int, float, bool)):
                            props[k] = v
                        elif isinstance(v, list) and all(isinstance(i2, str) for i2 in v):
                            props[k] = v
                    batch_props.append(props)

                try:
                    result = await session.run(
                        f"""
                        UNWIND $batch AS props
                        MERGE (n:{node_type} {{ref_id: props.ref_id, video_id: props.video_id}})
                        SET n += props
                        RETURN props.ref_id AS ref_id, elementId(n) AS element_id
                        """,
                        batch=batch_props,
                    )
                    async for record in result:
                        result_map[(node_type, record["ref_id"])] = record["element_id"]
                except Exception as e:
                    logger.warning(
                        "UNWIND batch failed for %s (offset %d): %s — falling back to individual merges",
                        node_type, i, e,
                    )
                    for node in batch:
                        try:
                            r = await session.run(
                                f"""
                                MERGE (n:{node_type} {{ref_id: $ref_id, video_id: $video_id}})
                                SET n += $props
                                RETURN elementId(n) AS element_id
                                """,
                                ref_id=node.ref_id,
                                video_id=node.video_id,
                                props={
                                    "ref_id": node.ref_id,
                                    "video_id": node.video_id,
                                    "db_id": node.id,
                                    "label": node.label or "",
                                },
                            )
                            rec = await r.single()
                            if rec:
                                result_map[(node_type, node.ref_id)] = rec["element_id"]
                        except Exception as e2:
                            logger.warning(
                                "Individual node merge failed %s/%s: %s", node_type, node.ref_id, e2
                            )
        return result_map

    # Write all node types in parallel — different labels, no write conflicts
    type_results = await asyncio.gather(
        *[_write_node_type(nt, ns) for nt, ns in nodes_by_type.items()],
        return_exceptions=True,
    )
    for res in type_results:
        if isinstance(res, dict):
            node_ref_to_element_id.update(res)
        elif isinstance(res, Exception):
            logger.warning("Node type write task failed: %s", res)

    logger.info(
        "write_nodes complete: %d nodes written across %d label types",
        len(node_ref_to_element_id), len(nodes_by_type),
    )
    return node_ref_to_element_id


async def write_edges(
    edges: list[KGEdge],
    nodes: list[KGNode],
    node_ref_map: dict[tuple[str, str], str],  # (node_type, ref_id) -> element_id
    database: str | None = None,
) -> None:
    """Upsert all edges using UNWIND batches grouped by (src_type, tgt_type, relation).

    Edges within each group are written 100 at a time to avoid N round-trips to AuraDB.
    Falls back to individual MERGE on batch failure.
    """
    # Build reverse map: KGNode.id (db UUID) -> (node_type, ref_id)
    node_db_id_to_ref: dict[str, tuple[str, str]] = {
        n.id: (n.node_type, n.ref_id) for n in nodes
    }

    # Group edges by (src_type, tgt_type, relation) — UNWIND requires a fixed Cypher
    # pattern, and the label names must be literals in the query.
    edge_groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    skipped = 0

    for edge in edges:
        src_ref = node_db_id_to_ref.get(edge.source_id)
        tgt_ref = node_db_id_to_ref.get(edge.target_id)
        if not src_ref or not tgt_ref:
            logger.debug("Skipping edge %s — endpoints not found", edge.id)
            skipped += 1
            continue

        src_type, src_ref_id = src_ref
        tgt_type, tgt_ref_id = tgt_ref

        entry: dict = {
            "src_ref_id": src_ref_id,
            "tgt_ref_id": tgt_ref_id,
            "video_id": edge.video_id,
            "weight": edge.weight,
            "db_id": edge.id,
        }
        entry.update({
            k: v for k, v in edge.properties.items()
            if isinstance(v, (str, int, float, bool))
        })
        safe_rel = re.sub(r"[^A-Z0-9_]", "_", edge.relation.upper())[:64].strip("_") or "RELATES_TO"
        edge_groups[(src_type, tgt_type, safe_rel)].append(entry)

    if skipped:
        logger.debug("write_edges: skipped %d edges with unresolved endpoints", skipped)

    async with neo4j_session(database) as session:
        for (src_type, tgt_type, relation), group in edge_groups.items():
            for i in range(0, len(group), _EDGE_BATCH_SIZE):
                batch = group[i : i + _EDGE_BATCH_SIZE]
                try:
                    await session.run(
                        f"""
                        UNWIND $batch AS e
                        MATCH (a:{src_type} {{ref_id: e.src_ref_id, video_id: e.video_id}})
                        MATCH (b:{tgt_type} {{ref_id: e.tgt_ref_id, video_id: e.video_id}})
                        MERGE (a)-[r:{relation}]->(b)
                        SET r.weight = e.weight, r.db_id = e.db_id
                        """,
                        batch=batch,
                    )
                except Exception as e:
                    logger.warning(
                        "Edge UNWIND failed %s->%s [%s] (offset %d): %s — falling back to individual merges",
                        src_type, tgt_type, relation, i, e,
                    )
                    # Fallback: individual MERGE for each edge in this batch
                    for entry in batch:
                        try:
                            await session.run(
                                f"""
                                MATCH (a:{src_type} {{ref_id: $src_ref_id, video_id: $video_id}})
                                MATCH (b:{tgt_type} {{ref_id: $tgt_ref_id, video_id: $video_id}})
                                MERGE (a)-[r:{relation}]->(b)
                                SET r.weight = $weight, r.db_id = $db_id
                                """,
                                src_ref_id=entry["src_ref_id"],
                                tgt_ref_id=entry["tgt_ref_id"],
                                video_id=entry["video_id"],
                                weight=entry["weight"],
                                db_id=entry["db_id"],
                            )
                        except Exception as e2:
                            logger.warning(
                                "Individual edge merge failed %s->%s [%s]: %s",
                                entry["src_ref_id"], entry["tgt_ref_id"], relation, e2,
                            )

    logger.info(
        "write_edges complete: %d edge groups, %d total edges (%d skipped)",
        len(edge_groups), sum(len(g) for g in edge_groups.values()), skipped,
    )


async def write_graph(
    video_id: str,
    nodes: list[KGNode],
    edges: list[KGEdge],
    database: str | None = None,
) -> None:
    """Full graph write: ensure constraints, upsert nodes, upsert edges."""
    await ensure_constraints(database)
    node_ref_map = await write_nodes(nodes, database)
    await write_edges(edges, nodes, node_ref_map, database)
    logger.info(
        "Neo4j graph written for video %s: %d nodes, %d edges",
        video_id, len(nodes), len(edges)
    )


async def query_graph(cypher: str, params: dict | None = None, database: str | None = None) -> list[dict]:
    """Run arbitrary Cypher query. Returns list of record dicts."""
    return await run_cypher(cypher, params, database)
