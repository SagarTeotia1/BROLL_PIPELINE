# Test fixtures

2-3 golden reference videos + hand-verified expected output (per-frame
`frame_analyses`, `speaker_turns`, `scenes`, `storylines` rows a human has
checked by hand) to be recorded here — per CLAUDE.md "PART B — Hardening
(resolved decisions)" / B3.

## golden_97199656.json / golden_97199656_ffprobe.json (added L7 7a)

CLAUDE.md "PIPELINE ADDENDUM 3" -> "LEVEL 7 — EVALUATION" -> 7a, "Golden-set
bootstrap": the FIRST real golden case, for
`video_id=97199656-d176-46aa-88b4-026670be4576` /
`edit_plan_id=b276be9d-c803-4cba-b86e-e3da624c479f` — a real L4-finalized
storyline (15 scenes), a real L5 edit plan (7 `SELECT_CLIP` ops, ~131.5s
achieved duration), and a real L6 render (`out.mp4` in the repo root,
confirmed structurally valid). Regenerated (safe to rerun — read-only
against the live DB) via `python scripts/_bootstrap_l7_fixture.py`:

- `golden_97199656.json` — `storylines`/`scenes` rows, the `edit_plans` row
  (incl. `operations`), `cut_list_items`, and the most recent `qa_reports`
  row, all as they exist in the live Postgres DB at snapshot time.
- `golden_97199656_ffprobe.json` — `ffprobe -show_format -show_streams`
  output for `out.mp4` (duration/codec/resolution/size) — the expected-
  output baseline for `tests/integration/test_e2e_golden.py`.

Consumed by `tests/integration/test_e2e_golden.py`, which re-validates live
DB state + a fresh local `ffprobe` pass against this snapshot within
tolerance (scene count ±2, achieved duration ±5%, no `qa_status='fail'`) —
not byte-identical, per this doc's own extensively-documented LLM-
non-determinism caveat.

---

Not yet populated for any OTHER video: no other golden reference videos are
committed this pass. This directory exists so `tests/integration/` has a
stable place to point at once more golden videos + hand-verified expected
output are recorded, and so CI/local runs of `pytest tests/` don't fail on
a missing path.
