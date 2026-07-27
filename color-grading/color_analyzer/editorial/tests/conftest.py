"""Synthetic frames with known colour properties, shared by the tests."""

from __future__ import annotations

import numpy as np

SIZE = (180, 320)  # height, width — small enough to keep the suite quick


def _base(height: int = SIZE[0], width: int = SIZE[1]) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.float32)


def portrait_frame() -> np.ndarray:
    """Skin-toned subject on a cool background, over a luminance ramp.

    Exercises every stage at once: skin detection, a split between warm and cool
    zones, several hue bands, and a non-degenerate tonal distribution.
    """
    height, width = SIZE
    frame = _base()
    ramp = np.linspace(0.15, 0.85, width, dtype=np.float32)[None, :]

    # Cool background.
    frame[..., 0] = ramp * 0.55
    frame[..., 1] = ramp * 0.75
    frame[..., 2] = ramp * 0.95

    # Warm skin-toned subject occupying the centre third.
    subject = slice(width // 3, 2 * width // 3)
    frame[:, subject, 0] = 0.72
    frame[:, subject, 1] = 0.54
    frame[:, subject, 2] = 0.43

    return np.clip(frame, 0.0, 1.0)


def flat_grey(level: float = 0.5) -> np.ndarray:
    """A perfectly neutral, perfectly flat frame."""
    return np.full((*SIZE, 3), float(level), dtype=np.float32)


def _cast(red: float, blue: float, textured: bool) -> np.ndarray:
    """Mid-grey with a channel cast, optionally with ordinary tonal variation."""
    frame = flat_grey(0.5)
    if textured:
        # Real footage always has some tonal spread. A perfectly flat frame's
        # defining characteristic is its flatness, which drowns out its colour.
        height, width = SIZE
        ramp = np.linspace(-0.18, 0.18, width, dtype=np.float32)[None, :, None]
        frame = np.clip(frame + ramp, 0.05, 0.95)
    frame[..., 0] *= red
    frame[..., 2] *= blue
    return np.clip(frame, 0.0, 1.0)


def warm_frame(textured: bool = True) -> np.ndarray:
    """A strong tungsten-like cast."""
    return _cast(1.45, 0.62, textured)


def cool_frame(textured: bool = True) -> np.ndarray:
    """A strong daylight/shade cast."""
    return _cast(0.68, 1.40, textured)


def clustered_frame() -> np.ndarray:
    """Distinct colour regions with texture — how real footage clusters.

    The palette's stability guarantee is about content that *has* clusters. A
    smooth gradient has none: any partition of a continuum is arbitrary, so its
    cluster boundaries are inherently unstable and no seeding strategy fixes
    that. This fixture is the realistic case.
    """
    height, width = SIZE
    frame = np.zeros((height, width, 3), dtype=np.float32)
    regions = [
        (0.00, 0.30, (0.82, 0.24, 0.18)),   # red
        (0.30, 0.55, (0.18, 0.58, 0.28)),   # green
        (0.55, 0.75, (0.20, 0.30, 0.78)),   # blue
        (0.75, 1.00, (0.90, 0.88, 0.84)),   # near-white
    ]
    for start, end, colour in regions:
        lo, hi = int(start * width), int(end * width)
        frame[:, lo:hi] = colour

    # Mild texture, so the regions are not mathematically perfect flats.
    rng = np.random.default_rng(11)
    frame += rng.uniform(-0.02, 0.02, frame.shape).astype(np.float32)
    return np.clip(frame, 0.0, 1.0)


def split_toned_frame() -> np.ndarray:
    """Teal shadows and orange highlights over a luminance ramp."""
    height, width = SIZE
    ramp = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :].repeat(height, 0)
    frame = np.stack(
        [ramp * 0.95 + 0.05, 0.30 + ramp * 0.45, 0.55 - ramp * 0.50], axis=-1
    )
    return np.clip(frame, 0.0, 1.0).astype(np.float32)


def hue_wheel_frame() -> np.ndarray:
    """Seven vertical bars, one centred on each HSL band."""
    import cv2

    from color_analyzer.editorial.hsl import BAND_CENTRES

    height, width = SIZE
    hsv = np.zeros((height, width, 3), dtype=np.float32)
    centres = list(BAND_CENTRES.values())
    step = width // len(centres)
    for index, centre in enumerate(centres):
        lo = index * step
        hi = width if index == len(centres) - 1 else (index + 1) * step
        hsv[:, lo:hi, 0] = centre
        hsv[:, lo:hi, 1] = 0.75
        hsv[:, lo:hi, 2] = 0.65
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def dark_scene_frame() -> np.ndarray:
    """A dark, warm, textured scene — the worst case for noise stability.

    HSV saturation divides by the channel maximum, so its noise sensitivity
    grows as the frame gets darker. Low-key footage is therefore the hardest
    thing to report stably, and worth testing explicitly rather than only on
    comfortably-exposed fixtures.
    """
    height, width = SIZE
    rng = np.random.default_rng(5)

    # A falloff from a warm light source, so the tone distribution is continuous.
    # Flat blocks would give a spiky, bimodal histogram, and quantities fitted
    # across the tonal range — gamma above all — are not defined on one.
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    falloff = np.clip(1.0 - np.hypot(x - 0.75, (y - 0.5) * 0.6) * 1.6, 0.05, 1.0)

    frame = np.stack([
        0.10 + falloff * 0.52,
        0.07 + falloff * 0.34,
        0.05 + falloff * 0.20,
    ], axis=-1).astype(np.float32)

    frame += rng.uniform(-0.012, 0.012, frame.shape).astype(np.float32)
    return np.clip(frame, 0.0, 1.0)


def graded_scene_frame() -> np.ndarray:
    """A frame shaped like real footage: gradients, mixed hues, texture.

    The render tests need this rather than :func:`clustered_frame`. Flat blocks
    of saturated colour respond far more strongly to a grade than photographic
    content does — an operation calibrated on real frames overshoots badly on
    them — and several readings are simply undefined there: gamma has no curve
    to fit across four discrete levels, and a fully saturated block pins the
    split-tone reading at 100 where it cannot move.

    Deliberately **middling** in every reading, too: near-neutral white balance,
    mid contrast, moderate saturation, no tonal zone empty. A control whose
    measurement is already pinned at a rail cannot move further in that
    direction, so a fixture with extreme readings silently turns round-trip
    assertions into tests of the clamp.
    """
    height, width = SIZE
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    rng = np.random.default_rng(23)

    # A soft lighting falloff spanning most of the tonal range, near-neutral.
    # Reaches well into shadow, so the lift wheel and the shadow end of the
    # split tone have a populated zone to act on.
    falloff = np.clip(1.15 - np.hypot(x - 0.62, (y - 0.45) * 0.7) * 1.5, 0.02, 0.96)
    frame = np.stack([
        0.01 + falloff * 0.95,
        0.01 + falloff * 0.92,
        0.01 + falloff * 0.89,
    ], axis=-1).astype(np.float32)

    # Mildly tinted regions, so every hue band has something in it without the
    # frame becoming two-toned. A strong cool patch beside a strong warm one
    # splits the near-neutral population white balance is measured from, and
    # warming the frame then reads as cooling it: the warmed pixels saturate out
    # of the neutral set and leave the cool ones behind.
    frame[: int(height * 0.40), : int(width * 0.30)] *= (0.78, 0.94, 1.20)
    frame[int(height * 0.55):, int(width * 0.34): int(width * 0.52)] = (0.30, 0.44, 0.31)
    frame[int(height * 0.20): int(height * 0.60), int(width * 0.62):] *= (1.08, 0.99, 0.92)

    frame += rng.uniform(-0.015, 0.015, frame.shape).astype(np.float32)
    return np.clip(frame, 0.0, 1.0)


def add_noise(frame: np.ndarray, amount: float, seed: int) -> np.ndarray:
    """Add reproducible uniform noise, as a stand-in for sensor noise."""
    rng = np.random.default_rng(seed)
    noise = rng.uniform(-amount, amount, frame.shape).astype(np.float32)
    return np.clip(frame + noise, 0.0, 1.0)
