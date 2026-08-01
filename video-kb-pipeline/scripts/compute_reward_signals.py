"""CLI: recompute `reward_signals` rollups from `evaluation_scores` (L7) +
`human_feedback` (L8) — CLAUDE.md "PIPELINE ADDENDUM 3" -> "LEVEL 9 —
REWARD & PUNISHMENT", item 9a.

Not a live trigger — a periodic script, safe to rerun with no arguments
(UPSERTs on `(scope_type, scope_key)`, per the spec). Reads only from
`evaluation_scores`/`human_feedback`/`kg_edges`/`videos`/`edit_plans`/
`scenes` — never writes to any of those tables, only to `reward_signals`
and (for 9c) `pipeline_alerts`.

Formula (deterministic, no LLM — same "deterministic where possible"
discipline as the rest of this pipeline):

  1. Each `evaluation_scores` row contributes one sample: the average of
     its non-null 0-10 dimensions (intent_match, narrative_coherence,
     pacing_consistency, technical_cleanliness), normalized to -1..1 via
     `(dim / 10) * 2 - 1`.

  2. Each `human_feedback` row contributes one sample:
     `{positive: +1, neutral: 0, negative: -1}[sentiment] * weight`, where
     `weight = rating / 5.0` when `rating` is present, else a documented
     default of `0.6` (a rating-less report is real signal but weighted
     down relative to an explicit 5/5 or 1/5 — 0.6 was chosen as "more
     than half-strength, less than a confident extreme," not derived from
     any data; there simply isn't enough real feedback yet to fit this
     empirically. Revisit once volume justifies it — same "don't build for
     the data you don't have" discipline as everything else in this repo).

  3. Samples are grouped into scope buckets:
       - scope_type='client'            — by `videos.client_id`
       - scope_type='canonical_relation'— every distinct non-null
         `kg_edges.canonical_relation` present in a contributing video's
         graph gets that video's samples added to its bucket. This is a
         coarse, VIDEO-LEVEL attribution (not scene/edge-level) — the
         current schema has no direct join from an `evaluation_scores` or
         `human_feedback` row to the specific edges/relations "used" in
         the delivered cut (that would require walking
         cut_list_items -> scenes -> frame_analyses -> kg_nodes -> kg_edges,
         which is not built here). Documented limitation, not a hidden one:
         a video that used relation X and scored badly is treated as weak
         evidence against X, even though X may not be what actually made
         the video score badly. A precise per-relation join is a
         reasonable v-next once there's enough data to make the extra
         precision matter.
       - scope_type='pacing_style'/'color_style' — `human_feedback` rows
         only (evaluation_scores has no category dimension to split on),
         bucketed under a single scope_key='global' each, since the
         current schema has no more granular pacing/color "style tag" to
         group by yet (see CLAUDE.md's Style Learning System discussion —
         this is the coarse first cut of that idea, not the final shape).

  4. Each bucket's `reward_score` is confidence-weighted-shrunk toward 0
     (neutral): `reward_score = sum(samples) / (len(samples) + PRIOR_WEIGHT)`
     — a standard Bayesian-shrinkage-toward-a-neutral-prior formula.
     `PRIOR_WEIGHT = 3` was chosen as "don't fully trust an average of 1-2
     samples" (explicitly called out in the task): with 1 sample of +1.0,
     shrunk score is 1/(1+3) = 0.25, not 1.0; with 10 samples all +1.0, it's
     10/13 = 0.77, converging toward the raw average as evidence
     accumulates. `sample_count` stored on the row is the RAW
     (pre-shrinkage) count, not adjusted — 9b/9c gate on this raw count
     with their OWN, higher, explicit floors (5 and 10 respectively, both
     given directly in the CLAUDE.md spec text) before treating a row as
     actionable. This script does not enforce those floors itself — it
     writes every non-empty bucket it computes, honestly, regardless of
     size; withholding a row below a floor would hide the very rows a
     human might want to look at to see the pipeline explains itself.

9c (punishment -> ontology alert): after the rollup, any
`scope_type='canonical_relation'` row with `reward_score < -0.3` AND
`sample_count >= 10` gets a new `pipeline_alerts` row
(`alert_type='l9_canonical_relation_negative_reward'`) — same non-fatal,
alert-only, human-reviews-then-versions-the-ontology pattern already used
by `grounding_runner.py`'s OTHER-bucket check (extended, not replaced).

Usage:
    python scripts/compute_reward_signals.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import defaultdict

sys.path.insert(0, ".")

from dotenv import load_dotenv

load_dotenv()

from knowledge_base.postgres.client import get_pool  # noqa: E402
from knowledge_base.postgres.queries import (  # noqa: E402
    get_all_evaluation_scores_with_video,
    get_all_human_feedback,
    get_distinct_canonical_relations_for_video,
    get_reward_signals,
    get_video_ids_with_client,
    insert_pipeline_alert,
    upsert_reward_signal,
)

# Bayesian-shrinkage prior weight — see module docstring step 4 for the
# worked example / rationale. Not spec'd numerically in CLAUDE.md ("leaves
# the exact floor to you"), chosen conservatively.
_PRIOR_WEIGHT = 3.0

# Default sample-weight for a `human_feedback` row with no `rating` given
# (see module docstring step 2).
_UNRATED_FEEDBACK_WEIGHT = 0.6

_SENTIMENT_BASE = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}

# 9c thresholds — both given literally in CLAUDE.md 9c's own text, not
# invented here.
_RELATION_ALERT_REWARD_THRESHOLD = -0.3
_RELATION_ALERT_SAMPLE_FLOOR = 10

# 9b's floor, also given literally in CLAUDE.md 9b's text — not enforced by
# this script (9b's call sites enforce it), reproduced here only so the CLI
# output can honestly report which client buckets *would* qualify.
_CLIENT_FEWSHOT_SAMPLE_FLOOR = 5


def _eval_row_contribution(row: dict) -> float | None:
    """Average of this `evaluation_scores` row's non-null 0-10 dimensions,
    normalized to -1..1. Returns None if every dimension is null (nothing
    to contribute — should not happen in practice, but never crash on it)."""
    dims = [
        row.get("intent_match"),
        row.get("narrative_coherence"),
        row.get("pacing_consistency"),
        row.get("technical_cleanliness"),
    ]
    present = [d for d in dims if d is not None]
    if not present:
        return None
    avg_0_10 = sum(present) / len(present)
    return (avg_0_10 / 10.0) * 2.0 - 1.0


def _feedback_row_contribution(row: dict) -> float:
    base = _SENTIMENT_BASE.get(row["sentiment"], 0.0)
    rating = row.get("rating")
    weight = (rating / 5.0) if rating is not None else _UNRATED_FEEDBACK_WEIGHT
    return base * weight


def _shrink(samples: list[float]) -> tuple[float, int]:
    """Confidence-weighted shrinkage toward neutral (0) — see module
    docstring step 4. Returns (reward_score, raw_sample_count)."""
    n = len(samples)
    if n == 0:
        return 0.0, 0
    reward_score = sum(samples) / (n + _PRIOR_WEIGHT)
    return reward_score, n


async def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute reward_signals rollups from evaluation_scores + "
            "human_feedback. Safe to rerun with no arguments."
        )
    )
    parser.add_argument(
        "--skip-alerts", action="store_true",
        help="Compute/write reward_signals but skip the 9c pipeline_alerts check.",
    )
    args = parser.parse_args()

    pool = await get_pool()

    eval_rows = await get_all_evaluation_scores_with_video(pool)
    feedback_rows = await get_all_human_feedback(pool)
    client_by_video = await get_video_ids_with_client(pool)

    # video_id -> list of raw -1..1 samples from BOTH evaluation_scores and
    # human_feedback for that video — used to build the 'client' and
    # 'canonical_relation' buckets below (both are video-scoped
    # attributions; 'pacing_style'/'color_style' are handled separately
    # since they come only from human_feedback.category, not per-video).
    contributions_by_video: dict[str, list[float]] = defaultdict(list)

    for row in eval_rows:
        contrib = _eval_row_contribution(row)
        if contrib is not None and row.get("video_id"):
            contributions_by_video[str(row["video_id"])].append(contrib)

    for row in feedback_rows:
        if row.get("video_id"):
            contributions_by_video[str(row["video_id"])].append(_feedback_row_contribution(row))

    # --- scope_type='client' -------------------------------------------
    client_buckets: dict[str, list[float]] = defaultdict(list)
    for video_id, samples in contributions_by_video.items():
        client_id = client_by_video.get(video_id)
        if client_id:
            client_buckets[client_id].extend(samples)

    # --- scope_type='canonical_relation' --------------------------------
    relation_buckets: dict[str, list[float]] = defaultdict(list)
    for video_id, samples in contributions_by_video.items():
        if not samples:
            continue
        relations = await get_distinct_canonical_relations_for_video(pool, video_id)
        for relation in relations:
            relation_buckets[relation].extend(samples)

    # --- scope_type='pacing_style' / 'color_style' ----------------------
    pacing_samples = [
        _feedback_row_contribution(r) for r in feedback_rows if r["category"] == "pacing"
    ]
    color_samples = [
        _feedback_row_contribution(r) for r in feedback_rows if r["category"] == "color"
    ]

    written: list[dict] = []

    async def _write_bucket(scope_type: str, scope_key: str, samples: list[float]) -> None:
        if not samples:
            return
        reward_score, sample_count = _shrink(samples)
        await upsert_reward_signal(pool, scope_type, scope_key, reward_score, sample_count)
        written.append({
            "scope_type": scope_type,
            "scope_key": scope_key,
            "reward_score": round(reward_score, 4),
            "sample_count": sample_count,
        })

    for client_id, samples in client_buckets.items():
        await _write_bucket("client", client_id, samples)
    for relation, samples in relation_buckets.items():
        await _write_bucket("canonical_relation", relation, samples)
    await _write_bucket("pacing_style", "global", pacing_samples)
    await _write_bucket("color_style", "global", color_samples)

    print(json.dumps({"reward_signals_written": written}, indent=2))

    # --- 9c: punishment -> ontology-review alert -------------------------
    alerts_fired: list[dict] = []
    if not args.skip_alerts:
        relation_signals = await get_reward_signals(pool, scope_type="canonical_relation")
        for signal in relation_signals:
            if (
                signal.reward_score < _RELATION_ALERT_REWARD_THRESHOLD
                and signal.sample_count >= _RELATION_ALERT_SAMPLE_FLOOR
            ):
                await insert_pipeline_alert(
                    pool,
                    None,  # not tied to a single video — this is a cross-video relation signal
                    9,
                    "l9_canonical_relation_negative_reward",
                    signal.reward_score,
                    _RELATION_ALERT_REWARD_THRESHOLD,
                )
                alerts_fired.append({
                    "canonical_relation": signal.scope_key,
                    "reward_score": signal.reward_score,
                    "sample_count": signal.sample_count,
                })

    print(json.dumps({
        "alerts_fired": alerts_fired,
        "thresholds": {
            "client_fewshot_sample_floor_9b": _CLIENT_FEWSHOT_SAMPLE_FLOOR,
            "relation_alert_sample_floor_9c": _RELATION_ALERT_SAMPLE_FLOOR,
            "relation_alert_reward_threshold_9c": _RELATION_ALERT_REWARD_THRESHOLD,
        },
    }, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
