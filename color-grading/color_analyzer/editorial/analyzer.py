"""The orchestrator: run every stage against one frame, emit one document.

Each stage is an independent function taking a :class:`~.frame.Frame` and
returning a plain dict, so stages can be run, tested and replaced individually.
:class:`EditorialAnalyzer` wires them together and validates the result against
:mod:`.schema` before returning it — a document that reaches a caller has
already been checked for forbidden content.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Iterable, Iterator, Optional

import numpy as np

from . import schema
from .color import analyze_color
from .frame import Frame
from .gpu import Backend, default_backend
from .hsl import analyze_hsl
from .look import analyze_look
from .palette import analyze_palette
from .skin import analyze_skin
from .split_toning import analyze_split_toning
from .tone import analyze_tone
from .wheels import analyze_wheels
from .white_balance import analyze_white_balance

#: Default analysis resolution cap. Colour state converges well below delivery
#: resolution; analysing a 4K frame at full size costs eight times the work and
#: does not move any reading in this document.
DEFAULT_MAX_SIDE = 1024

#: Swatches in the palette.
DEFAULT_PALETTE_SIZE = 6


class EditorialAnalyzer:
    """Produces the editable colour state of a frame.

    Parameters
    ----------
    backend:
        Shared execution backend; built automatically when omitted.
    max_side:
        Analysis resolution cap. ``None`` analyses at native resolution.
    palette_size:
        Number of dominant-colour swatches to extract.
    """

    def __init__(
        self,
        backend: Optional[Backend] = None,
        max_side: Optional[int] = DEFAULT_MAX_SIDE,
        palette_size: int = DEFAULT_PALETTE_SIZE,
    ) -> None:
        self.backend = backend or default_backend()
        self.max_side = max_side
        self.palette_size = palette_size

    # -- entry points -------------------------------------------------------
    def analyze_path(self, path: str) -> Dict[str, Any]:
        """Analyse an image file."""
        return self._run(Frame.from_path(path, self.backend, self.max_side))

    def analyze_bgr(self, bgr: np.ndarray) -> Dict[str, Any]:
        """Analyse an OpenCV BGR frame."""
        return self._run(Frame.from_bgr(bgr, self.backend, self.max_side))

    def analyze_rgb(self, rgb: np.ndarray) -> Dict[str, Any]:
        """Analyse an RGB array."""
        return self._run(Frame.from_rgb(rgb, self.backend, self.max_side))

    def analyze_frames(self, frames: Iterable[np.ndarray],
                       is_bgr: bool = True) -> Iterator[Dict[str, Any]]:
        """Analyse a stream of decoded video frames, reusing this analyzer.

        Defaults to BGR because that is what an OpenCV decoder hands you.
        """
        for frame in frames:
            yield self.analyze_bgr(frame) if is_bgr else self.analyze_rgb(frame)

    # -- pipeline -----------------------------------------------------------
    def _run(self, frame: Frame) -> Dict[str, Any]:
        started = time.perf_counter()

        tone = analyze_tone(frame)
        white_balance = analyze_white_balance(frame)
        color = analyze_color(frame)
        wheels = analyze_wheels(frame)
        split = analyze_split_toning(frame)
        palette = analyze_palette(frame, self.palette_size)
        skin = analyze_skin(frame)
        hsl = analyze_hsl(frame)

        # Derived from the readings above; touches no pixels.
        look = analyze_look(tone, white_balance, color, split)

        document: Dict[str, Any] = {
            "meta": {
                "source": frame.source,
                "width": frame.width,
                "height": frame.height,
                "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 2),
                **self.backend.describe(),
            },
            "look": look,
            "white_balance": white_balance,
            "tone": tone,
            "color": color,
            "wheels": wheels,
            "split_toning": split,
            "palette": palette,
            "skin_tone": skin,
            "hsl": hsl,
        }

        schema.validate(document)
        return document


def analyze_image(path: str, **kwargs: Any) -> Dict[str, Any]:
    """Convenience wrapper: analyse one image file."""
    return EditorialAnalyzer(**kwargs).analyze_path(path)
