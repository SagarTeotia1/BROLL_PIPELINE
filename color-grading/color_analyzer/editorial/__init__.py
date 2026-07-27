"""Editor-friendly colour analysis.

Describes the **editable colour state** of a frame — the controls a colourist
actually touches in Resolve, Lumetri or Lightroom — and nothing else.  No
histograms, no entropy, no feature vectors, no Lab/XYZ/YCrCb statistics.

Built on OpenCV and NumPy alone, with CUDA offload through ``cv2.cuda`` when the
installed OpenCV provides it.

    from color_analyzer.editorial import EditorialAnalyzer

    analyzer = EditorialAnalyzer()
    state = analyzer.analyze_path("frame.jpg")

    state["white_balance"]["temperature"]   # 5200
    state["hsl"]["orange"]["saturation"]    # 14
    state["look"]["mood"]                   # "warm"
"""

from __future__ import annotations

from .analyzer import DEFAULT_MAX_SIDE, EditorialAnalyzer, analyze_image
from .frame import Frame
from .gpu import Backend
from .schema import SchemaError, validate

__all__ = [
    "EditorialAnalyzer",
    "analyze_image",
    "Frame",
    "Backend",
    "validate",
    "SchemaError",
    "DEFAULT_MAX_SIDE",
]
