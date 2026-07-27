"""color_analyzer — a production-grade colour-grading analysis engine.

The package understands the colour *grading* of an image (not enhancement) and
extracts a rich, consistent feature vector describing its grading style.  It
runs transparently on GPU (CuPy) or CPU (NumPy) via a single backend
abstraction — see :mod:`color_analyzer.analyzer.utils`.
"""

from __future__ import annotations

from typing import Any

from .analyzer.utils import Backend, ImageContext

__all__ = [
    "ColorGradingEngine", "EngineResult", "Backend", "ImageContext",
    "DecisionEngine", "GradingPlanExecutor", "to_executor_decision", "schema",
]
__version__ = "2.0.0"


def __getattr__(name: str) -> Any:  # PEP 562 lazy import to avoid import cycles
    if name in ("ColorGradingEngine", "EngineResult"):
        from .analyzer.engine import ColorGradingEngine, EngineResult

        return {"ColorGradingEngine": ColorGradingEngine, "EngineResult": EngineResult}[name]
    if name in ("DecisionEngine", "to_executor_decision"):
        from .analyzer.decision_engine import DecisionEngine, to_executor_decision

        return {
            "DecisionEngine": DecisionEngine,
            "to_executor_decision": to_executor_decision,
        }[name]
    if name == "GradingPlanExecutor":
        from .analyzer.grading_plan import GradingPlanExecutor

        return GradingPlanExecutor
    if name == "schema":
        from .analyzer import schema

        return schema
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
