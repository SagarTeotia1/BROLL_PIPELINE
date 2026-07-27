"""Emotion recognition (HSEmotion) and temporal smoothing."""

from emotion.hsemotion import (  # noqa: F401
    EMOTION_COLORS,
    EMOTION_LABELS,
    EmotionResult,
    HSEmotionClassifier,
)
from emotion.smoothing import EmotionSmoother, SmoothedEmotion, SmootherRegistry  # noqa: F401

__all__ = [
    "HSEmotionClassifier",
    "EmotionResult",
    "EMOTION_LABELS",
    "EMOTION_COLORS",
    "EmotionSmoother",
    "SmoothedEmotion",
    "SmootherRegistry",
]
