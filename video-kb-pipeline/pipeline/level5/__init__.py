"""Level-5 Planning — turns a user's editing intent into a structured,
executable `EditPlan` (ordered `EditOperation[]`) grounded in L4's
`storylines(status='final')` + `scenes` output.

See CLAUDE.md "LEVEL 5 — PLANNING (Editing Director's Plan)" for the full
design: two-pass structure (Selection & Scoring, then Sequencing &
Pacing), the EditPlan/EditOperation schema, hard-vs-soft constraint
enforcement, and the read contract (L5 never reaches back into raw
frame_analyses/kg_edges/speaker_turns).

Entry point: `pipeline.level5.planner_runner.run_level5_planning`.
"""
from __future__ import annotations
