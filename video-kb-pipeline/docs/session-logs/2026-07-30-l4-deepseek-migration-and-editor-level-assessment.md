# Session Log — 2026-07-30

## L4 model migration (Qwen/Groq → DeepSeek V3.2/OpenRouter) + first live L4 run + dashboard L4 tab + editing-quality assessment

---

## 1. Model comparison discussion

**Qwen3.7 Flash vs Qwen3-30B (OpenRouter):**
- Qwen3.7 Flash: vision-language, $0.03/$0.13 per M, 1M ctx, single provider (Alibaba Cloud Int.), 100% uptime, 9.22% tool-call error rate, 35 tps.
- Qwen3-30B: text-only MoE, multi-provider (redundancy), generally cheaper, lower tool-error rate typically.
- Verdict for this pipeline (L4/L5/L6 — text-only structured tool calling): Qwen3-30B-class fits better. No vision need in these stages. Single-provider risk + 9.22% tool-error rate too risky for structured-output stages that already had a documented truncation/malformed-JSON bug.

**Best model for L4 specifically (Groq → OpenRouter reality check):**
- User clarified pipeline actually runs on OpenRouter (400+ models), not Groq as CLAUDE.md's L4 section states (doc was stale).
- Recommended split: cheap/fast tier (Claude Haiku 4.5 / Gemini 2.5 Flash-Lite) for Grounding Agent (high-volume, closed-set), strong tier (Claude Sonnet 4.5 / DeepSeek V3.2 / Gemini 2.5 Pro) for Story Architect (low-volume, narrative synthesis).
- Cheaper alternative requested → landed on **DeepSeek V3.2** for both L4 tiers: GPT-5-class reasoning claim, strong tool-use, $0.2072/$0.3108 per M (26% off), 164K context.
- Context-size concern (164K vs Qwen's 1M) addressed: non-issue — L4's own batch-cap design (rule 23) keeps every call well under 164K regardless of video length (grounding batches capped at 25 clusters/sub-batch after a prior truncation-bug fix; story architect batches 3-5 scenes/call with pruned ~80-100 tok/frame; rolling summary capped at 3 sentences).

## 2. Code changes made

**`shared/config.py`:**
- `L4_GROUNDING_MODEL` and `L4_STORY_ARCHITECT_MODEL` switched from Qwen split (`qwen/qwen3-30b-a3b-instruct-2507` / `qwen/qwen3-235b-a22b-2507`) to `deepseek/deepseek-v3.2` for both. Old Qwen IDs kept commented for rollback. L5/L6 model settings left untouched (Qwen, out of scope for this change).

**`pipeline/level4/grounding_runner.py`:**
- Added `"reasoning": {"enabled": False}` to `extra_body` in `_call_tool` — grounding is closed-set classification, no benefit from a reasoning pass, adds latency/cost at high call volume.

**`pipeline/level4/story_architect_runner.py`:**
- Added `"reasoning": {"enabled": True}` to `extra_body` — narrative synthesis (scene merging, causal links) benefits from reasoning; low call volume (~50/video) makes the cost acceptable.

**`.env.example`:** Updated commented `L4_GROUNDING_MODEL`/`L4_STORY_ARCHITECT_MODEL` lines to `deepseek/deepseek-v3.2`.

**`.env`:** Added explicit `L4_GROUNDING_MODEL=deepseek/deepseek-v3.2` / `L4_STORY_ARCHITECT_MODEL=deepseek/deepseek-v3.2` overrides (matches new config.py defaults). Flagged that this file has live secrets in plaintext (DB, R2, Neo4j, HF, Modal, OpenRouter) — confirm gitignored.

All edits `py_compile`-clean.

## 3. First live L4 run (video_id=97199656-d176-46aa-88b4-026670be4576)

Run via `modal run modal_app.py::run_level4_modal --video-id ...` (bare function call, bypasses the `run_full_pipeline` orchestrator that writes `processing_jobs` rows).

**Observed during run:**
- `run_relation_canonicalization`: OTHER bucket 19.7% of distinct relations — non-fatal alert (B4), matches CLAUDE.md's own documented historical range (15.5%→19.7%→29.6%) from a prior Qwen truncation bug; not necessarily a DeepSeek regression, could be genuine ontology v1 coverage gap.
- `run_story_architect`: `_disambiguate_canonical_scene_ids()` fired twice — caught Qwen reusing the same scene label ("Netflix subscription prompt screen") for two distinct adjacent scenes (139.8s/143.2s/145.1s), correctly suffixed to avoid UPSERT overwrite. Confirmed working as designed (this was CLAUDE.md's documented "fourth finding" fix).
- App completed without exceptions/traceback.

**Verified via direct DB query (not just trusting "app completed"):**
- `storylines.status = 'final'`, version 1, non-null synopsis, 15 beats. This only happens if `finalize_level4`'s 3 completeness checks (speaker-turn terminality, scene-coverage-no-gaps, storyline non-empty) all passed.
- `speaker_turns` empty for this video — expected/non-fatal (L2 pyannote diarization produced nothing for this video; not an L4 bug).
- No `processing_jobs(level=4)` row exists — expected, since that row is only written by the `run_full_pipeline` orchestrator wrapper, not by `run_level4_modal` itself, and the bare function was called directly.

**Verdict: L4 = complete and passed, first live run, under new DeepSeek V3.2 config.**

## 4. Cost/token numbers (OpenRouter dashboard, user-provided)

- 1-month rollup (3 models: Qwen 30B, Qwen 235B, DeepSeek V3.2 combined): $0.05 spend, 43 requests, 271K tokens, $0.20 blended $/1M, 0.9% cache hit rate.
- DeepSeek-V3.2-filtered slice (this run, 2m25s / 104-frame clip): $0.02 spend, 8 requests, 47K tokens, $0.34 blended $/1M, 6.0% cache hit rate.
- Extrapolation to a 1-hour / 1759-frame clip (not purely linear — L4 batches by scene/turn/relation-cluster count, tracks duration more than raw frame count):
  - Scale factors: duration 145s→3600s = 24.8x; frames 104→1759 = 16.9x.
  - Estimated 1hr: ~$0.35–0.50 spend, ~135–200 requests, ~800K–1.17M tokens.
  - Still well under B2's $0.02/video-min × 60min = $1.20 budget either way.
  - Caveat: recommended an actual 1hr test run over trusting extrapolation, since batch-cap non-linearity (esp. relation-cluster sub-batching if OTHER-bucket stays elevated) makes the high end uncertain.

## 5. Dashboard change — added "🎭 L4 Reasoning" tab

`dashboard.py` had no L4-specific view (tabs covered L1-L3 + raw L2 diarization only; no storylines/scenes browse view, no relation-canonicalization view).

**Added:**
- `storylines(vid)`, `scenes(vid)`, `relation_canon_stats(vid)` cached DB fetchers.
- Status bar extended from 3 to 4 columns (added "L4 Reasoning").
- New tab "🎭 L4 Reasoning" (inserted before Search tab) with:
  - 4 metrics: storyline status (derived from `storylines.status`, not `processing_jobs` — explicitly noted why: bare Modal function calls skip that table), scene count, unresolved speaker-turn count, OTHER-bucket relation %.
  - Storylines section: per-version expander with synopsis, cast, beats table.
  - Scenes section: dataframe + plotly timeline (Gantt) as a visual coverage-gap check, plus per-scene expanders (summary/emotional_arc/causal_link_to_next).
  - Relation canonicalization section: bar chart by canonical relation, raw→canonical mapping table.
  - Speaker-turn resolution section: bar chart by `resolution_method`.

Verified: `py_compile` clean, `streamlit run` started without errors (server smoke-tested on port 8511, then killed), and the 3 new SQL queries independently confirmed to run clean against the live video_id (1 storyline row, 43 scenes, 71 relation-stat rows).

## 6. Editing-quality-level assessment (L1-L5 editor framework, user-pasted)

Framework pasted by user: Level 1 (social-media AI / CapCut-tier) → Level 2 (junior editor) → Level 3 (mid-level editor, target) → Level 4 (senior editor: emotional pacing, comedic timing, knowing when not to cut, cinematic rhythm, audience retention optimization, brand style, sponsor integration, creative transitions) → Level 5 (creative director).

**First-pass (measured/honest) assessment:**
- Design target: Level 3, with some pieces (narrative causality via `storylines`/`scenes`, sequence-aware color matching, `usability_score` take-picking) reaching toward Level 3-4 ambition.
- Real gaps vs Level 3 checklist: no multicam (documented no-op, no schema concept), no motion graphics (explicitly deferred), audio ducking is a real no-op.
- Biggest caveat: every L4-L6 component's own STATUS line reads "IMPLEMENTED — not yet run end-to-end." FCPXML never test-imported into a real NLE. Color conversion formulas explicitly "unverified judgment calls, not checked against a color-science reference."
- Verdict: architecture targets Level 3 correctly on paper; in practice, zero human has reviewed any output yet, so no level claim is earned until an editor reviews a delivered cut and correction time is measured against the "<20min/hour" bar.

**Second pass, after user asked for unvarnished truth (no pleasing):**
- Reframed as currently closer to Level 1-2 in practice, dressed as Level 3-4 in architecture.
- Reasoning: (1) zero human has ever watched an output — every quality claim is architectural intent, not measured outcome; (2) the doc's own dense postmortem history (6+ numbered "real-video findings" in L4 alone — scene_index drops, boundary snapping, dupe-label overwrites, JSON truncation, participant-format mismatch) were all found before a single real client video ran, meaning only structural bugs have been caught so far, not editorial-judgment failures; (3) the mechanics most correlated with "feels professionally edited" are the weakest — cut-snapping is breath-pause avoidance (silence-avoidance ≠ rhythm), color formulas are unverified, `usability_score` is a technical-fault proxy (audio clipping) not a "best take" proxy; (4) missing multicam/motion-graphics/ducking are core gaps, not edge cases, for typical interview/podcast footage; (5) the "90-95% of mid-level editor" target figure is aspirational framing, not a measured number.
- Recommended next real milestone: render one full L1→L6 video, put it in front of a human editor blind to how it was made, time their correction pass — that number is the only ground truth available; everything else is plan.

## 7. Gap analysis — what's structurally missing to reach Level 4

1. **No audience-outcome feedback loop** (biggest gap) — no retention/drop-off data source exists anywhere in the schema; `correction_events` captures human edits (style/accuracy) but not viewer behavior. Level 4's "audience retention optimization" is unbuildable without this ground truth.
2. **No rhythm-as-a-sequence reasoning** — L5 picks/orders clips but doesn't treat cut-duration pattern across the whole timeline as a first-class decision (tighten→release→tighten shaping).
3. **No "don't cut here" signal** — architecture assumes cutting is the default action; no `hold_value_score` or equivalent to argue for preserving an uninterrupted take (reaction shot, vulnerable pause).
4. **No comedy-specific signal** — `beat_type`/`tension_level` are generic; comedic timing needs laugh/reaction-audio detection + precise punchline-pause timing (a different discipline than VAD breath-snap).
5. **No sponsor-segment concept at all** — no schema entity, no emotional-context-gated placement logic.
6. **No creative-transition decision layer** — L6 mechanics are literal (cut/color-match/overlay/zoom); no agent choosing whip-pan/match-cut/smash-cut for narrative effect.
7. **`client_style_profiles` is shallow** — a few manually-set JSONB fields, not learned from exemplar-video ingestion.
8. Sequencing point: none of the above is worth building before Level 3 is verified by a real human review — Level 4 signals need *more* ground truth than Level 3, and currently there is zero ground truth of any kind (no video has been watched by anyone yet).

## 8. Evaluations / guardrails inventory (verified against actual code, not just doc claims)

Confirmed real (grepped, not assumed):
1. Pydantic schema validation on every LLM tool response — reject+log on mismatch, never write raw text to a typed column.
2. Closed-set constraints — pid/canonical_relation must come from the given list, never invented.
3. Confidence-gated escalation — confirmed in `grounding_runner.py` (~line 460+): `<0.5` confidence triggers one bounded re-batch retry to same model.
4. `pipeline_alerts` writes — confirmed real (`grounding_runner.py:572`), not just logged: OTHER-bucket rate and escalation-trigger-rate both actually persisted.
5. Finalizer completeness gate — speaker-turn terminality, scene-coverage gap check, storyline non-empty — real, watched it pass live this session.
6. L5 duration hard-constraint + plan validation (scene_id exists, timestamps in bounds) — code exists, not yet run live.
7. QA Agent (L6 tail) — deterministic checks (blackdetect/silencedetect/loudness/caption-drift/color-clip) + one LLM intent-match pass, blocks delivery on fail — code exists, not yet run live.

**Real gap:** no offline eval harness. `tests/fixtures/` golden videos "to be recorded," not done. No human-preference/pairwise scoring loop. No automated editorial-quality judge — everything above catches *structural* failure (bad JSON, coverage gaps, intent mismatch), nothing catches *editorial* quality (is the pacing actually good).

## 9. Latency / smart routing / fallback assessment

- Every L4-L6 LLM call site uses `extra_body={"provider": {"require_parameters": True}}` (filters weak backends) but **no explicit routing mode** (no `"sort": "throughput"`/`"quality"`, no Nitro/Exacto pinning) — same routing config used for high-volume cheap calls and low-volume quality calls alike.
- **No automatic provider fallback** — `get_llm_client(prefer_groq=True)` exists but is manual opt-in per call site, not wired as an automatic circuit breaker on OpenRouter outage/timeout.
- **No per-stage wall-clock timeout** distinct from the retry-count loop (`_LLM_CALL_ATTEMPTS`) — a single hanging call from a slow backend provider can stall the whole sequential L1→L6 chain.
- **No `StepTimer` usage in L4/L5/L6** (confirmed via grep — L1-L3 have it per doc, L4-L6 don't) — no visibility into where latency actually goes in the reasoning/planning/action stages.

## 10. Token optimization / compression techniques actually in use

Confirmed via grep:
- Frame pruning for Story Architect (`_prune_frame`) — raw Qwen JSON (~300+ tok/frame) cut to ~80-100 tok/frame.
- Rolling summary hard-capped at 3 sentences, truncated defensively if a call ignores the instruction.
- Relation-cluster sampling — max 8 representative raw-relation strings shown per cluster, not the full membership.
- Batch caps everywhere (rule 23) — 25 clusters/call (relation canon), 3-5 scenes/call (story architect), 30-40 candidates/call (L5 Pass A).
- **No prompt caching used anywhere** — confirmed no `cache_control` / cache-friendly static-prefix design in any L4-L6 call site. Matches the near-zero (0.9%-6%) cache-hit rate seen in the OpenRouter dashboard. Not a problem at current spend ($0.02-0.05/month) but would matter at real production volume, since repeated system prompts across grounding/story-architect calls are exactly what prompt caching is for.

## 11. Open ask at end of session

User asked to spawn multiple agents to build toward Level 4. Pushed back rather than executing blind: building Level 4 features (sponsor placement, comedic timing, retention optimization) now stacks more unverified complexity on an unverified Level 3 base, and Level 4's core blocker (no audience-retention data source exists at all) can't be solved by more agents — there's nothing to build that signal from yet. Proposed alternative scope instead: (1) build the eval/golden-set/human-review harness so quality becomes measurable, (2) fix the latency/routing/fallback gaps above (real, scoped, valuable regardless of target level), (3) run one real L1→L6 video end-to-end and get a human review clock. Awaiting user decision on which direction to actually spawn agents for.
