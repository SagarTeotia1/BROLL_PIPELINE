"""Hierarchical diversity planner for ASH-FS pipeline.

Builds a 4-level tree (shot → chunk → scene → sequence → video) over the
selected keyframes for a video and uses per-level diversity thresholds to
decide whether each non-shot node should expand (recurse into children) or
summarise (keep the single medoid keyframe).
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from .config import Config
from .types import SelectedKeyframes, TreeNode

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Level ordering (bottom-up index)
# ---------------------------------------------------------------------------
_LEVEL_ORDER = ("shot", "chunk", "scene", "sequence", "video")


class HierarchicalDiversityPlanner:
    """Build a diversity-aware keyframe tree and prune it top-down.

    Parameters
    ----------
    cfg:
        Fully-loaded :class:`~ashfs.config.Config` instance.
    """

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        hier = cfg.hierarchy
        vertical = cfg.complexity.content_vertical
        _vertical_chunk_threshold_map = {
            "talking_head": hier.expansion_threshold_talking_head,
            "fast_cut":     hier.expansion_threshold_fast_cut,
            "general":      hier.expansion_threshold_general,
        }
        chunk_threshold = _vertical_chunk_threshold_map.get(
            vertical, hier.expansion_threshold_chunk
        )
        # expansion thresholds keyed by the *parent* level that owns children
        self._expansion_thresholds: dict[str, float] = {
            "chunk":    chunk_threshold,
            "scene":    hier.expansion_threshold_scene,
            "sequence": hier.expansion_threshold_sequence,
            # video node uses the sequence threshold as a conservative default
            "video":    hier.expansion_threshold_sequence,
        }
        # Per-instance walk counters (reset at the start of each get_final_keyframes call)
        self._stats_expanded: int = 0
        self._stats_summarized: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_tree(
        self,
        selected_keyframes: list[SelectedKeyframes],
    ) -> TreeNode:
        """Build the full 4-level tree bottom-up from per-shot keyframes.

        Parameters
        ----------
        selected_keyframes:
            One :class:`~ashfs.types.SelectedKeyframes` per shot, ordered by
            shot position in the video.

        Returns
        -------
        TreeNode
            Root node at level ``'video'``.
        """
        if not selected_keyframes:
            return TreeNode(
                level="video",
                node_id="video_0",
                children=[],
                keyframes=[],
                aggregate_embedding=None,
                expand=True,
                diversity_score=0.0,
            )

        chunk_size = self._cfg.hierarchy.chunk_size_shots

        # ----------------------------------------------------------------
        # Level 0: shot nodes (one per SelectedKeyframes)
        # ----------------------------------------------------------------
        shot_nodes: list[TreeNode] = []
        for i, kf in enumerate(selected_keyframes):
            agg = self._mean_pool(kf.embeddings) if kf.embeddings.size > 0 else None
            shot_nodes.append(
                TreeNode(
                    level="shot",
                    node_id=f"shot_{i}",
                    children=[],
                    keyframes=[kf],
                    aggregate_embedding=agg,
                    expand=True,       # shot-level nodes always expand
                    diversity_score=0.0,
                )
            )

        # ----------------------------------------------------------------
        # Level 1: chunk nodes  (every chunk_size_shots shots)
        # ----------------------------------------------------------------
        chunk_nodes = self._group_nodes(
            shot_nodes,
            group_size=chunk_size,
            level="chunk",
            id_prefix="chunk",
        )

        # ----------------------------------------------------------------
        # Level 2: scene nodes  (every 3 chunks)
        # ----------------------------------------------------------------
        scene_nodes = self._group_nodes(
            chunk_nodes,
            group_size=3,
            level="scene",
            id_prefix="scene",
        )

        # ----------------------------------------------------------------
        # Level 3: sequence nodes  (every 3 scenes)
        # ----------------------------------------------------------------
        seq_nodes = self._group_nodes(
            scene_nodes,
            group_size=3,
            level="sequence",
            id_prefix="seq",
        )

        # ----------------------------------------------------------------
        # Level 4: video root
        # ----------------------------------------------------------------
        root = self._make_parent_node(seq_nodes, level="video", node_id="video_0")

        return root

    def get_final_keyframes(
        self,
        tree_root: TreeNode,
    ) -> list[SelectedKeyframes]:
        """Walk tree top-down and collect the keyframes that survive pruning.

        Rules
        -----
        * Shot-level nodes always return their keyframes as-is.
        * For any non-shot node:
          - ``expand=True``  → recurse into children
          - ``expand=False`` → return the single medoid keyframe for this node

        Parameters
        ----------
        tree_root:
            Root :class:`~ashfs.types.TreeNode` as returned by
            :meth:`build_tree`.

        Returns
        -------
        list[SelectedKeyframes]
            Final keyframes in chronological order (sorted by first frame
            index).
        """
        # Reset walk counters so stats reflect only this call, not accumulated totals
        self._stats_expanded = 0
        self._stats_summarized = 0

        collected: list[SelectedKeyframes] = []
        self._walk(tree_root, collected)
        # Sort by the first selected frame index to preserve temporal order
        collected.sort(key=lambda kf: kf.frame_indices[0] if kf.frame_indices else 0)
        return collected

    def get_compression_stats(
        self,
        original: list[SelectedKeyframes],
        final: list[SelectedKeyframes],
    ) -> dict:
        """Summarise compression achieved by the diversity planner.

        Parameters
        ----------
        original:
            Input to :meth:`build_tree` (one per shot, pre-hierarchy).
        final:
            Output of :meth:`get_final_keyframes`.

        Returns
        -------
        dict with keys
            ``total_original_frames``, ``total_final_frames``,
            ``compression_ratio``, ``nodes_expanded``, ``nodes_summarized``.
        """
        total_orig = sum(len(kf.frame_indices) for kf in original)
        total_final = sum(len(kf.frame_indices) for kf in final)
        ratio = total_orig / total_final if total_final > 0 else float("inf")
        return {
            "total_original_frames": total_orig,
            "total_final_frames": total_final,
            "compression_ratio": round(ratio, 4),
            "nodes_expanded": self._stats_expanded,
            "nodes_summarized": self._stats_summarized,
        }

    # ------------------------------------------------------------------
    # Internal tree construction helpers
    # ------------------------------------------------------------------

    def _group_nodes(
        self,
        nodes: list[TreeNode],
        group_size: int,
        level: str,
        id_prefix: str,
    ) -> list[TreeNode]:
        """Partition *nodes* into groups of *group_size* and build parent nodes."""
        parents: list[TreeNode] = []
        for g_idx in range(0, len(nodes), group_size):
            group = nodes[g_idx : g_idx + group_size]
            node_id = f"{id_prefix}_{g_idx // group_size}"
            parent = self._make_parent_node(group, level=level, node_id=node_id)
            parents.append(parent)
        return parents

    def _make_parent_node(
        self,
        children: list[TreeNode],
        level: str,
        node_id: str,
    ) -> TreeNode:
        """Create a non-shot parent node, compute aggregate embedding and
        diversity score, then set ``expand`` according to the threshold."""
        if not children:
            return TreeNode(
                level=level,
                node_id=node_id,
                children=[],
                keyframes=[],
                aggregate_embedding=None,
                expand=True,
                diversity_score=0.0,
            )

        # Gather all keyframes from children
        all_kf: list[SelectedKeyframes] = []
        for child in children:
            all_kf.extend(child.keyframes)

        # Aggregate embedding = mean-pool of all child aggregate embeddings
        child_agg_embs: list[np.ndarray] = [
            c.aggregate_embedding
            for c in children
            if c.aggregate_embedding is not None
        ]

        if child_agg_embs:
            agg = np.mean(np.stack(child_agg_embs, axis=0), axis=0)
            # L2-normalise
            norm = np.linalg.norm(agg)
            if norm > 1e-8:
                agg = agg / norm
        else:
            agg = None

        # Diversity score = mean pairwise cosine distance between children's
        # aggregate embeddings (cosine distance = 1 - cosine similarity)
        diversity = self._pairwise_mean_cosine_distance(child_agg_embs)

        # Decide expand/summarise.
        # Only chunk-level nodes are permitted to summarise — scene, sequence,
        # and video nodes always expand.  Allowing higher-level nodes to summarise
        # cascades a talking-head video (near-zero diversity everywhere) all the
        # way down to 1 keyframe, which is never useful.
        # Special case: a node with a single child (or zero valid aggregate
        # embeddings) has no meaningful pairwise diversity — always expand.
        threshold = self._expansion_thresholds.get(level, 0.10)
        if level != "chunk":
            expand = True
        elif len(children) <= 1 or len(child_agg_embs) < 2:
            expand = True
        else:
            expand = diversity > threshold

        # If not expanding, choose the medoid among children
        medoid_kf: Optional[SelectedKeyframes] = None
        if not expand and child_agg_embs:
            # BUG-HDP-1: medoid_idx is computed on the *filtered* child_agg_embs
            # list (children with non-None aggregate_embedding).  Indexing the
            # full `children` list with that index is an off-by-N whenever any
            # earlier child has a None aggregate_embedding.  Build a parallel
            # `valid_children` list so index maps back correctly.
            valid_children: list[TreeNode] = [
                c for c in children if c.aggregate_embedding is not None
            ]
            medoid_idx = self._medoid_index(child_agg_embs)
            medoid_child = valid_children[medoid_idx]
            medoid_kf = self._medoid_keyframe_for_node(medoid_child)

        node = TreeNode(
            level=level,
            node_id=node_id,
            children=children,
            keyframes=all_kf,
            aggregate_embedding=agg,
            expand=expand,
            diversity_score=diversity,
        )
        # Attach the medoid keyframe as a convenience attribute for the walker
        node._medoid_kf = medoid_kf  # type: ignore[attr-defined]
        return node

    # ------------------------------------------------------------------
    # Internal tree walking
    # ------------------------------------------------------------------

    def _walk(
        self,
        node: TreeNode,
        collected: list[SelectedKeyframes],
    ) -> None:
        """Recursive top-down walker."""
        if node.level == "shot":
            # Shot nodes always yield their keyframes
            collected.extend(node.keyframes)
            return

        if node.expand:
            self._stats_expanded += 1
            for child in node.children:
                self._walk(child, collected)
        else:
            self._stats_summarized += 1
            medoid_kf: Optional[SelectedKeyframes] = getattr(
                node, "_medoid_kf", None
            )
            if medoid_kf is not None:
                collected.append(medoid_kf)
            elif node.keyframes:
                # Fallback: use the first keyframe
                collected.append(node.keyframes[0])

    # ------------------------------------------------------------------
    # Numpy / embedding helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _mean_pool(embeddings: np.ndarray) -> np.ndarray:
        """Mean-pool row embeddings and L2-normalise the result."""
        if embeddings.ndim == 1:
            vec = embeddings
        else:
            vec = embeddings.mean(axis=0)
        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            vec = vec / norm
        return vec.astype(np.float32)

    @staticmethod
    def _pairwise_mean_cosine_distance(embs: list[np.ndarray]) -> float:
        """Return mean pairwise cosine distance (1 - cos_sim) for *embs*.

        Returns 0.0 if there are fewer than 2 embeddings.
        """
        n = len(embs)
        if n < 2:
            return 0.0

        mat = np.stack(embs, axis=0)  # (n, D)
        # Dot products — embeddings are already L2-normalised
        sims = mat @ mat.T            # (n, n)
        # Extract upper-triangle indices (excluding diagonal)
        idx_i, idx_j = np.triu_indices(n, k=1)
        cos_distances = 1.0 - sims[idx_i, idx_j]
        return float(cos_distances.mean())

    @staticmethod
    def _medoid_index(embs: list[np.ndarray]) -> int:
        """Return index of the medoid: the embedding with the highest mean
        cosine similarity to all others (argmax of row means in sim matrix)."""
        if len(embs) == 1:
            return 0
        mat = np.stack(embs, axis=0)  # (n, D)
        sims = mat @ mat.T            # (n, n)
        # Mean similarity for each candidate (excluding self by masking diagonal)
        np.fill_diagonal(sims, 0.0)
        mean_sims = sims.sum(axis=1) / (len(embs) - 1)
        return int(np.argmax(mean_sims))

    def _medoid_keyframe_for_node(self, node: TreeNode) -> Optional[SelectedKeyframes]:
        """Drill down into *node* to find its representative (medoid) shot-level
        keyframe using the medoid-of-medoids strategy."""
        if node.level == "shot":
            # Shot node: pick the frame whose embedding is the medoid
            kf = node.keyframes[0] if node.keyframes else None
            if kf is None:
                return None
            if len(kf.frame_indices) <= 1:
                return kf
            # Pick medoid frame within the shot
            medoid_local = self._medoid_index(list(kf.embeddings))
            return SelectedKeyframes(
                shot=kf.shot,
                frame_indices=[kf.frame_indices[medoid_local]],
                embeddings=kf.embeddings[medoid_local : medoid_local + 1],
                siglip_embeddings=(
                    kf.siglip_embeddings[medoid_local : medoid_local + 1]
                    if kf.siglip_embeddings is not None
                    else None
                ),
                frame_roles=["medoid"],
            )

        # Non-shot: recurse to find the medoid child.
        # Build parallel lists so that the index returned by _medoid_index
        # (which operates on the filtered embedding list) maps back to the
        # correct child node — using a single filtered list for both would
        # produce an off-by-N index when some children have None embeddings.
        valid_children: list[TreeNode] = [
            c for c in node.children if c.aggregate_embedding is not None
        ]
        child_agg_embs: list[np.ndarray] = [
            c.aggregate_embedding for c in valid_children  # type: ignore[misc]
        ]
        if not child_agg_embs:
            return node.keyframes[0] if node.keyframes else None

        medoid_child_idx = self._medoid_index(child_agg_embs)
        return self._medoid_keyframe_for_node(valid_children[medoid_child_idx])
