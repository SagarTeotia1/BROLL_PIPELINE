"""Shared LLM client construction for L4-L6's reasoning/planning/agent calls.

Provider: OpenRouter (OpenAI-compatible `chat.completions` API, real two-tier
Qwen model split — see shared/config.py's L4/L5/L6 `*_MODEL` settings). Every
call site across L4-L6 (grounding_runner.py, story_architect_runner.py,
planner_runner.py, color_grading_runner.py, caption_overlay.py,
background_selector.py, emphasis_selector.py, qa_agent.py) already speaks
pure OpenAI-compatible shape — `.chat.completions.create(...)`, forced tool
calls returned as `response.choices[0].message.tool_calls[0].function.
arguments` (a JSON string, not a pre-parsed dict) — so only client
construction needed to change when the provider moved from Groq to
OpenRouter, not any of the actual call bodies.

Groq is kept as an optional fallback (`shared/config.py`'s `GROQ_API_KEY`
stays available) — see `get_llm_client(prefer_groq=True)` — in case
OpenRouter has an outage or a specific call site needs Groq's lower latency
for an interactive path (e.g. L5's `apply_revision` loop). Not used by
default anywhere today; every call site defaults to OpenRouter.
"""

from __future__ import annotations

from shared.config import settings

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def get_llm_client(prefer_groq: bool = False):
    """Return an OpenAI-compatible client for L4-L6 reasoning calls.

    Args:
        prefer_groq: When True, construct a Groq client instead of
            OpenRouter (requires `settings.GROQ_API_KEY`). Default False —
            every call site uses OpenRouter unless it opts into Groq
            explicitly for a specific latency-sensitive path.

    Raises:
        RuntimeError: if the required API key for the selected provider
            isn't configured.
    """
    if prefer_groq:
        if not settings.GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY is not configured — cannot construct a Groq "
                "client (see shared/config.py). Omit prefer_groq=True to "
                "use OpenRouter instead."
            )
        import groq  # local import — keeps this an optional dependency

        return groq.Groq(api_key=settings.GROQ_API_KEY)

    if not settings.OPENROUTER_API_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not configured — L4-L6 reasoning calls "
            "cannot run without it (see shared/config.py). Set it in .env "
            "locally and in the `video-kb-secrets` Modal secret for "
            "deployed runs."
        )
    from openai import OpenAI  # local import — keeps this an optional
    # dependency for callers that never make an LLM call (e.g. fact-dedup-
    # only code paths).

    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=settings.OPENROUTER_API_KEY)


# ---------------------------------------------------------------------------
# LEVEL 7 -- EVALUATION (CLAUDE.md "PIPELINE ADDENDUM 3" -> "LEVEL 7 --
# EVALUATION" -> 7c, "Durable cost/latency tracking"). Closes audit gap #9:
# previously only `logger.info(...)` of token counts, no durable table. See
# `knowledge_base/postgres/migrations/017_l7_evaluation.sql` for the
# `llm_call_log` DDL.
# ---------------------------------------------------------------------------

# Best-effort $/1M-token pricing for models actually configured in
# shared/config.py's L4-L6 `*_MODEL` settings — mirrors the per-model price
# comments already sitting next to those settings. NOT authoritative (an
# OpenRouter-routed request's real price can vary slightly by which backend
# it lands on) — this is a static estimate for COST TRACKING, not billing.
# Chosen per 7c's "cost_usd can be a best-effort estimate ... or leave NULL"
# option: small hardcoded table + NULL fallback for anything not listed,
# rather than guessing a number for a model with no known public price.
_MODEL_PRICING_USD_PER_1M: dict[str, tuple[float, float]] = {
    # model_id: (prompt $/1M tokens, completion $/1M tokens)
    "qwen/qwen3-30b-a3b-instruct-2507": (0.04815, 0.1931),
    "qwen/qwen3-235b-a22b-2507": (0.09, 0.55),
    # deepseek-v3.2 (L4_GROUNDING_MODEL/L4_STORY_ARCHITECT_MODEL default) —
    # approximate public list price, not independently verified against an
    # OpenRouter invoice. Flagged here, not silently treated as exact.
    "deepseek/deepseek-v3.2": (0.27, 1.10),
}


def estimate_cost_usd(
    model: str, prompt_tokens: int | None, completion_tokens: int | None
) -> float | None:
    """Best-effort USD cost estimate for one LLM call. Returns None (never a
    guessed number) when *model* isn't in `_MODEL_PRICING_USD_PER_1M` or
    either token count is unknown — see module docstring above."""
    pricing = _MODEL_PRICING_USD_PER_1M.get(model)
    if pricing is None or prompt_tokens is None or completion_tokens is None:
        return None
    prompt_rate, completion_rate = pricing
    return (prompt_tokens / 1_000_000.0) * prompt_rate + (completion_tokens / 1_000_000.0) * completion_rate


async def log_llm_call(
    pool,
    *,
    video_id: str | None,
    level: int,
    stage: str,
    model: str,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    latency_ms: int | None = None,
) -> None:
    """Best-effort durable log of one LLM call (7c). Called from every
    `_call_tool`/`_call_groq_tool` call site across L4-L6, right after each
    site already reads `response.usage` for its own `logger.info` line —
    this just also persists it to `llm_call_log`.

    Non-fatal by design (rule mirrored from every other L6 agent's
    "observability must never break the actual work" precedent): any
    failure (missing pool, DB error, import error) is logged and swallowed
    here — never raised into the caller. The LLM-consuming call this wraps
    around must succeed or fail on its own merits, not on whether this
    logging call happened to work.
    """
    import logging as _logging

    logger = _logging.getLogger(__name__)
    if pool is None:
        logger.debug("log_llm_call: no pool provided — skipping (stage=%s, model=%s)", stage, model)
        return
    try:
        from knowledge_base.postgres.queries import insert_llm_call_log

        cost_usd = estimate_cost_usd(model, prompt_tokens, completion_tokens)
        await insert_llm_call_log(
            pool,
            video_id=video_id,
            level=level,
            stage=stage,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
        )
    except Exception as exc:  # noqa: BLE001 — observability logging must never break the caller
        logger.warning(
            "log_llm_call: failed to persist llm_call_log row (video_id=%s level=%s "
            "stage=%s model=%s): %s",
            video_id, level, stage, model, exc,
        )
