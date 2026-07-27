"""Main orchestrator for the ASH-FS keyframe sampling pipeline.

Wires together all modules in the correct order and returns a rich manifest
dict suitable for serialisation to JSON.  Any unhandled exception causes a
graceful fallback to uniform-FPS sampling so the caller always receives
usable keyframes.
"""
from __future__ import annotations

import logging
import random
import time
from typing import Any, Optional

import numpy as np

from .config import Config, load_config, get_complexity_threshold
from .types import (
    BudgetedShot,
    SelectedKeyframes,
    Shot,
    ShotEmbeddings,
    ScoredShot,
)
from .embedding_manager import EmbeddingManager
from .budget_planner import BudgetPlanner
from .hierarchical_diversity_planner import HierarchicalDiversityPlanner

logger = logging.getLogger(__name__)


class SamplerPipeline:
    """End-to-end ASH-FS keyframe sampling pipeline.

    Parameters
    ----------
    config_path:
        Path to ``config.yaml``.
    """

    def __init__(self, config_path: str) -> None:
        self._cfg: Config = load_config(config_path)
        self._config_path = config_path

        # Initialise modules that exist
        self._embedding_manager = EmbeddingManager(self._cfg)
        self._budget_planner = BudgetPlanner(self._cfg)
        self._diversity_planner = HierarchicalDiversityPlanner(self._cfg)

        # Lazily import modules that may not exist yet — imported inside run()
        # so that import errors surface with a helpful message rather than at
        # construction time.

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, video_path: str) -> dict:
        """Run the full pipeline on *video_path* and return a manifest dict.

        On any unhandled exception the pipeline falls back to uniform
        sampling.  The manifest always has the same schema.

        Parameters
        ----------
        video_path:
            Absolute or relative path to the input video file.

        Returns
        -------
        dict
            Rich manifest — see module docstring for schema.
        """
        timing: dict[str, float] = {}

        try:
            return self._run_pipeline(video_path, timing)
        except Exception as exc:  # pylint: disable=broad-except
            logger.error(
                "ASH-FS pipeline failed (%s: %s) — falling back to uniform sampling.",
                type(exc).__name__,
                exc,
                exc_info=True,
            )
            return self._fallback(video_path, timing, error=str(exc))

    # ------------------------------------------------------------------
    # Pipeline implementation
    # ------------------------------------------------------------------

    def _run_pipeline(self, video_path: str, timing: dict[str, float]) -> dict:
        """Execute all pipeline stages in order."""
        cfg = self._cfg

        # ----------------------------------------------------------------
        # Lazy-import optional modules (shot_segmenter, complexity_scorer,
        # dual_keyframe_selector) — they are built in separate files.
        # ----------------------------------------------------------------
        shot_segmenter = self._load_shot_segmenter()
        complexity_scorer = self._load_complexity_scorer()
        dual_keyframe_selector = self._load_dual_keyframe_selector()

        # ----------------------------------------------------------------
        # Stage 1: Sample candidate frames
        # ----------------------------------------------------------------
        t0 = time.perf_counter()
        frame_indices, frame_loader = shot_segmenter.sample_candidate_frames(
            video_path
        )
        timing["candidate_sampling"] = time.perf_counter() - t0
        logger.info("Sampled %d candidate frames.", len(frame_indices))

        # ----------------------------------------------------------------
        # Stage 2: Extract DINOv2 embeddings for all candidate frames
        # ----------------------------------------------------------------
        t0 = time.perf_counter()
        embeddings_cache: dict[int, np.ndarray] = (
            self._embedding_manager.extract_for_frames(frame_indices, frame_loader)
        )
        timing["dinov2_extraction"] = time.perf_counter() - t0
        logger.info(
            "Extracted embeddings for %d frames.", len(embeddings_cache)
        )

        # ----------------------------------------------------------------
        # Stage 3: Shot segmentation — pass pre-extracted embeddings so
        # detect_shots does not re-sample / re-extract a second time.
        # embeddings_array must be ordered identically to frame_indices.
        # ----------------------------------------------------------------
        t0 = time.perf_counter()
        embeddings_array = np.stack(
            [embeddings_cache[idx] for idx in frame_indices]
        )
        shots: list[Shot] = shot_segmenter.detect_shots(
            video_path,
            self._embedding_manager,
            frame_indices=frame_indices,
            embeddings_array=embeddings_array,
        )
        timing["shot_segmentation"] = time.perf_counter() - t0
        logger.info("Detected %d shots.", len(shots))

        # ----------------------------------------------------------------
        # Stage 4: Build ShotEmbeddings by looking up the cache
        # ----------------------------------------------------------------
        t0 = time.perf_counter()
        shot_embeddings: list[ShotEmbeddings] = self._build_shot_embeddings(
            shots, frame_indices, embeddings_cache
        )
        timing["shot_embedding_assembly"] = time.perf_counter() - t0

        # ----------------------------------------------------------------
        # Stage 5: Complexity scoring
        # ----------------------------------------------------------------
        t0 = time.perf_counter()
        scored_shots: list[ScoredShot] = complexity_scorer.score_all(shot_embeddings)
        timing["complexity_scoring"] = time.perf_counter() - t0

        # ----------------------------------------------------------------
        # Stage 6: Complexity threshold lookup
        # ----------------------------------------------------------------
        t0 = time.perf_counter()
        complexity_threshold: float = get_complexity_threshold(cfg)
        timing["complexity_threshold_lookup"] = time.perf_counter() - t0

        # ----------------------------------------------------------------
        # Stage 7: Budget planning
        # ----------------------------------------------------------------
        t0 = time.perf_counter()
        budgeted_shots: list[BudgetedShot] = self._budget_planner.plan(
            scored_shots, complexity_threshold
        )
        timing["budget_planning"] = time.perf_counter() - t0
        budget_stats = self._budget_planner.get_stats(budgeted_shots)

        # ----------------------------------------------------------------
        # Stage 8: Dual keyframe selection
        # ----------------------------------------------------------------
        t0 = time.perf_counter()
        selected_keyframes: list[SelectedKeyframes] = (
            dual_keyframe_selector.select_all(budgeted_shots)
        )
        timing["dual_keyframe_selection"] = time.perf_counter() - t0

        # ----------------------------------------------------------------
        # Stage 9: Hierarchical diversity planning
        # Filter out budget=0 shots (empty keyframes) before tree building
        # ----------------------------------------------------------------
        t0 = time.perf_counter()
        tree = self._diversity_planner.build_tree(selected_keyframes)
        final_keyframes: list[SelectedKeyframes] = (
            self._diversity_planner.get_final_keyframes(tree)
        )
        timing["hierarchical_diversity"] = time.perf_counter() - t0

        compression_stats = self._diversity_planner.get_compression_stats(
            selected_keyframes, final_keyframes
        )

        # ----------------------------------------------------------------
        # Stage 10 (optional): SigLIP2 for final keyframes
        # ----------------------------------------------------------------
        if cfg.embeddings.siglip2_enabled:
            t0 = time.perf_counter()
            self._extract_siglip2_for_final(final_keyframes, frame_loader)
            timing["siglip2_extraction"] = time.perf_counter() - t0

        # ----------------------------------------------------------------
        # Stage 10b (optional): Dense signal extraction at candidate-frame
        # density — emotion + identity, independent of keyframe grid
        # ----------------------------------------------------------------
        if cfg.dense_signals.enabled:
            t0 = time.perf_counter()
            self._extract_dense_signals_stage(frame_indices, frame_loader)
            timing["dense_signal_extraction"] = time.perf_counter() - t0

        # ----------------------------------------------------------------
        # Stage 10c: Estimate novelty missed between selected keyframes
        # ----------------------------------------------------------------
        t0 = time.perf_counter()
        novelty_stats = self._estimate_novelty_missed(
            frame_indices=frame_indices,
            embeddings_cache=embeddings_cache,
            final_keyframes=final_keyframes,
            sample_shots=20,
        )
        timing["novelty_estimation"] = time.perf_counter() - t0
        logger.info(
            "Novelty missed — max=%.4f  mean=%.4f  shots_sampled=%d  shots_above_threshold=%d",
            novelty_stats["max_novelty_missed"],
            novelty_stats["mean_novelty_missed"],
            novelty_stats["shots_sampled"],
            novelty_stats["shots_above_threshold"],
        )

        # ----------------------------------------------------------------
        # Assemble manifest
        # ----------------------------------------------------------------
        return self._build_manifest(
            video_path=video_path,
            shots=shots,
            frame_indices=frame_indices,
            final_keyframes=final_keyframes,
            budget_stats=budget_stats,
            compression_stats=compression_stats,
            novelty_stats=novelty_stats,
            timing=timing,
        )

    # ------------------------------------------------------------------
    # Helper: build ShotEmbeddings from cache
    # ------------------------------------------------------------------

    def _build_shot_embeddings(
        self,
        shots: list[Shot],
        all_frame_indices: list[int],
        embeddings_cache: dict[int, np.ndarray],
    ) -> list[ShotEmbeddings]:
        """Assign pre-extracted embeddings to each shot."""
        # Build a set for O(1) lookup
        frame_set = set(all_frame_indices)
        result: list[ShotEmbeddings] = []

        for shot in shots:
            # Collect all candidate frame indices that fall within this shot
            shot_indices = [
                idx
                for idx in all_frame_indices
                if shot.start_frame <= idx <= shot.end_frame
            ]

            if not shot_indices:
                # No candidate frames in this shot window — use nearest
                # available frame as a single representative
                nearest = min(
                    all_frame_indices,
                    key=lambda idx: abs(
                        idx - (shot.start_frame + shot.end_frame) // 2
                    ),
                    default=None,
                )
                shot_indices = [nearest] if nearest is not None else []

            if shot_indices:
                # Keep only indices that are actually present in the cache so
                # that frame_indices and embeddings rows stay aligned 1-to-1.
                # Filtering here (rather than only in the np.stack list-comp)
                # ensures we never call np.stack on an empty sequence, which
                # would raise ValueError, and prevents a length mismatch
                # between ShotEmbeddings.frame_indices and .embeddings.
                shot_indices = [idx for idx in shot_indices if idx in embeddings_cache]

            if shot_indices:
                embs = np.stack(
                    [embeddings_cache[idx] for idx in shot_indices],
                    axis=0,
                )
            else:
                embs = np.empty((0, 768), dtype=np.float32)

            result.append(
                ShotEmbeddings(
                    shot=shot,
                    frame_indices=shot_indices,
                    embeddings=embs,
                )
            )

        return result

    # ------------------------------------------------------------------
    # Helper: optional SigLIP2 pass
    # ------------------------------------------------------------------

    def _extract_siglip2_for_final(
        self,
        final_keyframes: list[SelectedKeyframes],
        frame_loader,
    ) -> None:
        """Attach SigLIP2 embeddings to each SelectedKeyframes in-place."""
        for kf in final_keyframes:
            if not kf.frame_indices:
                continue
            frames = [frame_loader(idx) for idx in kf.frame_indices]
            kf.siglip_embeddings = self._embedding_manager.extract_siglip2(frames)

    # ------------------------------------------------------------------
    # Helper: estimate novelty missed between selected keyframes
    # ------------------------------------------------------------------

    def _estimate_novelty_missed(
        self,
        frame_indices: list[int],
        embeddings_cache: dict[int, np.ndarray],
        final_keyframes: list[SelectedKeyframes],
        sample_shots: Optional[int] = 20,
    ) -> dict:
        """Estimate max visual novelty missed between selected keyframes.

        For each (sampled) shot: find the candidate frame with maximum cosine
        distance from its nearest selected keyframe.  The max over all shots
        is the "max novelty missed" — how different the most-missed frame was
        from what we kept.

        Parameters
        ----------
        frame_indices:
            All candidate frame indices (from Stage 1 sampling).
        embeddings_cache:
            DINOv2 embeddings for every candidate frame, keyed by frame index.
        final_keyframes:
            List of SelectedKeyframes produced after hierarchical diversity
            planning (Stage 9 output).
        sample_shots:
            If not None, randomly sample at most this many shots before
            computing stats (reproducible, seed=42).  None = use all shots.

        Returns
        -------
        dict with keys:
            max_novelty_missed  — worst-case cosine distance from any candidate
                                  to its nearest selected keyframe (float)
            mean_novelty_missed — average max-per-shot cosine distance (float)
            shots_sampled       — number of shots actually evaluated (int)
            shots_above_threshold — shots where max_novelty > 0.15 (int)
        """
        _EMPTY: dict = {
            "max_novelty_missed": 0.0,
            "mean_novelty_missed": 0.0,
            "shots_sampled": 0,
            "shots_above_threshold": 0,
        }

        if not embeddings_cache or not final_keyframes:
            return _EMPTY

        # Build mapping: keyframe frame index → L2-normalised embedding.
        # Every entry in final_keyframes has a list of selected frame indices
        # and a corresponding (N, 768) embeddings array.
        kf_embs: list[np.ndarray] = []
        for kf in final_keyframes:
            for i, idx in enumerate(kf.frame_indices):
                if i < len(kf.embeddings):
                    kf_embs.append(kf.embeddings[i])

        if not kf_embs:
            return _EMPTY

        # Stack into (K, 768) matrix once — cosine similarity is just dot
        # product because embeddings are already L2-normalised.
        kf_matrix = np.stack(kf_embs, axis=0)  # (K, 768)

        # Determine the shot regions to evaluate.  Each SelectedKeyframes
        # object carries the Shot it belongs to, which gives [start, end].
        shots_to_eval: list[SelectedKeyframes] = list(final_keyframes)

        if sample_shots is not None and len(shots_to_eval) > sample_shots:
            rng = random.Random(42)
            shots_to_eval = rng.sample(shots_to_eval, sample_shots)

        _THRESHOLD = 0.15

        max_per_shot: list[float] = []

        for kf in shots_to_eval:
            shot = kf.shot
            # Collect all candidate frames that fall within this shot's window.
            candidates = [
                idx for idx in frame_indices
                if shot.start_frame <= idx <= shot.end_frame
                and idx in embeddings_cache
            ]

            if not candidates:
                continue

            # Build (C, 768) matrix of candidate embeddings.
            cand_matrix = np.stack(
                [embeddings_cache[idx] for idx in candidates], axis=0
            )  # (C, 768)

            # Cosine similarities: (C, K) = cand_matrix @ kf_matrix.T
            # Both are L2-normalised so dot product == cosine similarity.
            sim_matrix = cand_matrix @ kf_matrix.T  # (C, K)

            # For each candidate, best match = max similarity across keyframes.
            # Cosine distance = 1 - similarity (clamped to [0, 1]).
            best_sim = sim_matrix.max(axis=1)          # (C,)
            cos_dist = np.clip(1.0 - best_sim, 0.0, 1.0)  # (C,)

            max_per_shot.append(float(cos_dist.max()))

        if not max_per_shot:
            return _EMPTY

        max_novelty = float(max(max_per_shot))
        mean_novelty = float(sum(max_per_shot) / len(max_per_shot))
        above = sum(1 for v in max_per_shot if v > _THRESHOLD)

        return {
            "max_novelty_missed": round(max_novelty, 6),
            "mean_novelty_missed": round(mean_novelty, 6),
            "shots_sampled": len(max_per_shot),
            "shots_above_threshold": above,
        }

    # ------------------------------------------------------------------
    # Helper: assemble final manifest dict
    # ------------------------------------------------------------------

    def _build_manifest(
        self,
        video_path: str,
        shots: list[Shot],
        frame_indices: list[int],
        final_keyframes: list[SelectedKeyframes],
        budget_stats: dict,
        compression_stats: dict,
        novelty_stats: dict,
        timing: dict[str, float],
    ) -> dict:
        """Collate all pipeline outputs into the manifest schema."""
        # Build (frame_index, role) pairs then sort together so roles stay
        # aligned with selected_indices even when shots interleave.
        index_role_pairs: list[tuple[int, str]] = []
        for kf in final_keyframes:
            roles = kf.frame_roles if kf.frame_roles else ["unknown"] * len(kf.frame_indices)
            for idx, role in zip(kf.frame_indices, roles):
                index_role_pairs.append((idx, role))
        index_role_pairs.sort(key=lambda p: p[0])

        selected_indices: list[int] = [p[0] for p in index_role_pairs]
        frame_roles: list[str] = [p[1] for p in index_role_pairs]

        # Frames-per-shot distribution
        frames_per_shot: list[int] = [len(kf.frame_indices) for kf in final_keyframes]

        # Shot boundaries
        shot_boundaries: list[dict] = [
            {
                "start_frame": s.start_frame,
                "end_frame": s.end_frame,
                "duration_s": round(s.duration_s, 4),
                "fps": s.fps,
            }
            for s in shots
        ]

        return {
            "video_path": video_path,
            "total_shots": len(shots),
            "total_candidate_frames": len(frame_indices),
            "total_selected_frames": len(selected_indices),
            "frames_per_shot_distribution": frames_per_shot,
            "budget_stats": budget_stats,
            "compression_stats": compression_stats,
            "novelty_stats": novelty_stats,
            "selected_frame_indices": selected_indices,
            "frame_roles": frame_roles,
            "shot_boundaries": shot_boundaries,
            "timing": {k: round(v, 4) for k, v in timing.items()},
        }

    # ------------------------------------------------------------------
    # Fallback: uniform sampling
    # ------------------------------------------------------------------

    def _fallback(
        self,
        video_path: str,
        timing: dict[str, float],
        error: str = "",
    ) -> dict:
        """Uniform-FPS fallback when the main pipeline raises."""
        logger.warning("Entering fallback uniform sampling for %s", video_path)
        cfg = self._cfg
        fallback_fps = cfg.fallback.uniform_fps
        max_frames = cfg.fallback.max_fallback_frames

        try:
            import cv2  # type: ignore[import]

            cap = cv2.VideoCapture(video_path)
            video_fps: float = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            step = max(1, int(video_fps / fallback_fps))
            indices = list(range(0, total_frames, step))[:max_frames]

        except Exception as exc2:  # pylint: disable=broad-except
            logger.error("Fallback sampling also failed: %s", exc2)
            indices = []

        return {
            "video_path": video_path,
            "total_shots": 1,
            "total_candidate_frames": len(indices),
            "total_selected_frames": len(indices),
            "frames_per_shot_distribution": [len(indices)],
            "budget_stats": {},
            "compression_stats": {
                "total_original_frames": len(indices),
                "total_final_frames": len(indices),
                "compression_ratio": 1.0,
                "nodes_expanded": 0,
                "nodes_summarized": 0,
            },
            "selected_frame_indices": indices,
            "frame_roles": ["fallback"] * len(indices),
            "shot_boundaries": [],
            "timing": {k: round(v, 4) for k, v in timing.items()},
            "fallback": True,
            "fallback_error": error,
        }

    # ------------------------------------------------------------------
    # Helper: dense signal extraction (emotion + identity)
    # ------------------------------------------------------------------

    def _extract_dense_signals_stage(
        self,
        frame_indices: list[int],
        frame_loader,
    ) -> None:
        """Run DenseSignalExtractor at candidate-frame density.

        Results are logged but not yet included in the manifest — they are
        stored as a dense time-series for downstream consumers (e.g. the VLM
        captioning layer) rather than stuffed into the keyframe-level output.
        """
        try:
            from .dense_signal_extractor import DenseSignalExtractor
        except ImportError:
            logger.warning("DenseSignalExtractor not importable — skipping dense signals.")
            return

        cfg = self._cfg.dense_signals
        extractor = DenseSignalExtractor(
            batch_size=cfg.batch_size,
            face_detector=cfg.face_detector,
        )
        signals = extractor.extract(frame_indices, frame_loader)
        face_frames = sum(1 for e in signals.emotions if e and e.get("dominant_emotion"))
        logger.info(
            "Dense signals extracted: %d/%d frames had detectable faces, "
            "%d unique speaker clusters.",
            face_frames,
            len(frame_indices),
            len({s for s in signals.identities if s is not None}),
        )

    # ------------------------------------------------------------------
    # Lazy-loading optional modules
    # ------------------------------------------------------------------

    def _load_shot_segmenter(self):
        """Import and instantiate ShotSegmenter."""
        try:
            from .shot_segmenter import ShotSegmenter  # type: ignore[import]

            return ShotSegmenter(self._cfg)
        except ImportError as exc:
            raise ImportError(
                "ShotSegmenter not found. "
                "Expected module: ashfs/shot_segmenter.py with class ShotSegmenter."
            ) from exc

    def _load_complexity_scorer(self):
        """Import and instantiate ComplexityScorer."""
        try:
            from .complexity_scorer import ComplexityScorer  # type: ignore[import]

            return ComplexityScorer(self._cfg)
        except ImportError as exc:
            raise ImportError(
                "ComplexityScorer not found. "
                "Expected module: ashfs/complexity_scorer.py with class ComplexityScorer."
            ) from exc

    def _load_dual_keyframe_selector(self):
        """Import and instantiate DualKeyframeSelector."""
        try:
            from .dual_keyframe_selector import DualKeyframeSelector  # type: ignore[import]

            return DualKeyframeSelector(self._cfg)
        except ImportError as exc:
            raise ImportError(
                "DualKeyframeSelector not found. "
                "Expected module: ashfs/dual_keyframe_selector.py with class DualKeyframeSelector."
            ) from exc
