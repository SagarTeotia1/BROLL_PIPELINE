"""Temporal smoothing of per-frame emotion predictions.

Raw HSEmotion output flickers: a face can read Happy / Neutral / Happy on three
consecutive sampled frames simply because of blink or motion blur. Writing that
straight into the timeline would produce hundreds of 130 ms events.

Three strategies are available (``config.emotion.smoothing``):

``majority``
    Sliding window of the last ``window`` labels. The state switches only when the
    candidate label holds at least ``min_agree`` votes *and* beats the current label.

``ema``
    Exponential moving average over the probability *vectors*; the argmax of the
    smoothed vector is the state. Reacts faster than majority voting and keeps a
    usable confidence value.

``hybrid`` (default)
    EMA for the confidence, majority voting for the switch decision, plus a fast path:
    a single prediction above ``switch_confidence`` that agrees with the EMA argmax may
    switch immediately. This keeps genuine sharp reactions (a jump scare) while still
    suppressing flicker.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

from configs.config import EmotionConfig
from emotion.hsemotion import EMOTION_LABELS


@dataclass
class SmoothedEmotion:
    """Stable emotion state after smoothing."""

    label: str
    confidence: float
    changed: bool
    raw_label: str
    raw_confidence: float
    votes: int = 0

    def as_dict(self) -> dict:
        return {
            "emotion": self.label,
            "confidence": round(self.confidence, 4),
            "changed": self.changed,
            "raw": self.raw_label,
        }


class EmotionSmoother:
    """Per-track temporal filter over emotion predictions.

    One instance per track ID. It is deliberately tiny (a deque plus a vector) so
    thousands of tracks cost nothing.
    """

    def __init__(self, config: EmotionConfig, labels: Sequence[str] = EMOTION_LABELS) -> None:
        self.cfg = config
        self.labels = list(labels)
        self.mode = config.smoothing.lower()
        self.window: Deque[str] = deque(maxlen=max(1, config.window))
        self.conf_window: Deque[float] = deque(maxlen=max(1, config.window))
        self._ema: Optional[np.ndarray] = None
        self.current: Optional[str] = None
        self.current_confidence: float = 0.0
        self.updates: int = 0

    # -- main entry point ---------------------------------------------------
    def update(self, label: str, confidence: float, probabilities: Optional[np.ndarray] = None) -> SmoothedEmotion:
        """Feed one raw prediction and get the (possibly unchanged) stable state."""
        self.updates += 1

        # Low-confidence predictions are recorded but never allowed to force a switch.
        trustworthy = confidence >= self.cfg.confidence_threshold
        if trustworthy:
            self.window.append(label)
            self.conf_window.append(confidence)

        if probabilities is not None:
            probabilities = np.asarray(probabilities, dtype=np.float32)
            if self._ema is None:
                self._ema = probabilities.copy()
            else:
                a = float(self.cfg.ema_alpha)
                self._ema = a * probabilities + (1.0 - a) * self._ema

        if self.mode == "ema":
            new_label, new_conf, votes = self._decide_ema(label, confidence)
        elif self.mode == "majority":
            new_label, new_conf, votes = self._decide_majority(label, confidence)
        else:
            new_label, new_conf, votes = self._decide_hybrid(label, confidence, trustworthy)

        changed = new_label is not None and new_label != self.current
        if new_label is not None:
            self.current = new_label
            self.current_confidence = new_conf

        return SmoothedEmotion(
            label=self.current or label,
            confidence=self.current_confidence if self.current else confidence,
            changed=changed,
            raw_label=label,
            raw_confidence=confidence,
            votes=votes,
        )

    # -- strategies ---------------------------------------------------------
    def _decide_ema(self, label: str, confidence: float) -> Tuple[Optional[str], float, int]:
        if self._ema is None:
            return (label if confidence >= self.cfg.confidence_threshold else self.current), confidence, 0
        idx = int(np.argmax(self._ema))
        smoothed_label = self.labels[idx] if idx < len(self.labels) else label
        smoothed_conf = float(self._ema[idx])
        if self.current is None:
            return smoothed_label, smoothed_conf, 0
        if smoothed_label != self.current and smoothed_conf < self.cfg.confidence_threshold:
            return self.current, self.current_confidence, 0
        return smoothed_label, smoothed_conf, 0

    def _decide_majority(self, label: str, confidence: float) -> Tuple[Optional[str], float, int]:
        if not self.window:
            return self.current, self.current_confidence, 0
        counts = Counter(self.window)
        candidate, votes = counts.most_common(1)[0]
        if self.current is None:
            # Bootstrap as soon as we have any agreement at all.
            if votes >= min(self.cfg.min_agree, len(self.window)):
                return candidate, self._mean_confidence(candidate), votes
            return None, confidence, votes
        if candidate == self.current:
            return self.current, self._mean_confidence(candidate), votes
        if votes >= self.cfg.min_agree and votes > counts.get(self.current, 0):
            return candidate, self._mean_confidence(candidate), votes
        return self.current, self.current_confidence, votes

    def _decide_hybrid(
        self, label: str, confidence: float, trustworthy: bool
    ) -> Tuple[Optional[str], float, int]:
        maj_label, maj_conf, votes = self._decide_majority(label, confidence)

        ema_label: Optional[str] = None
        ema_conf = 0.0
        if self._ema is not None:
            idx = int(np.argmax(self._ema))
            ema_label = self.labels[idx] if idx < len(self.labels) else None
            ema_conf = float(self._ema[idx])

        # Fast path: a very confident frame that the EMA already agrees with switches now.
        if (
            trustworthy
            and confidence >= self.cfg.switch_confidence
            and ema_label == label
            and label != self.current
        ):
            return label, max(confidence, ema_conf), votes

        if maj_label is None:
            return None, confidence, votes
        # Blend the majority decision's confidence with the EMA for a stable readout.
        if ema_label == maj_label:
            return maj_label, float(max(maj_conf, ema_conf)), votes
        return maj_label, maj_conf, votes

    def _mean_confidence(self, label: str) -> float:
        vals = [c for lbl, c in zip(self.window, self.conf_window) if lbl == label]
        if not vals:
            return self.current_confidence
        return float(sum(vals) / len(vals))

    # -- state --------------------------------------------------------------
    def reset(self) -> None:
        """Clear history (e.g. after a scene cut on this track)."""
        self.window.clear()
        self.conf_window.clear()
        self._ema = None
        self.current = None
        self.current_confidence = 0.0

    @property
    def is_stable(self) -> bool:
        """True once a label has been established."""
        return self.current is not None


class SmootherRegistry:
    """Keeps one :class:`EmotionSmoother` per track and prunes dead ones."""

    def __init__(self, config: EmotionConfig, labels: Sequence[str] = EMOTION_LABELS) -> None:
        self.cfg = config
        self.labels = list(labels)
        self._smoothers: Dict[int, EmotionSmoother] = {}

    def get(self, track_id: int) -> EmotionSmoother:
        s = self._smoothers.get(track_id)
        if s is None:
            s = EmotionSmoother(self.cfg, self.labels)
            self._smoothers[track_id] = s
        return s

    def drop(self, track_id: int) -> None:
        self._smoothers.pop(track_id, None)

    def prune(self, alive: Sequence[int]) -> List[int]:
        """Remove smoothers whose track is gone; returns the dropped IDs."""
        alive_set = set(alive)
        dead = [tid for tid in self._smoothers if tid not in alive_set]
        for tid in dead:
            del self._smoothers[tid]
        return dead

    def reset(self) -> None:
        self._smoothers.clear()

    def __len__(self) -> int:
        return len(self._smoothers)


__all__ = ["EmotionSmoother", "SmoothedEmotion", "SmootherRegistry"]
