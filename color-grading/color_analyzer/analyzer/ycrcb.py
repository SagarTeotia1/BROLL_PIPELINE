"""YCrCb analysis.

YCrCb separates luma (Y) from two chroma-difference channels.  It is the classic
space for skin detection because human skin clusters tightly in the Cr-Cb plane
regardless of luminance.  We report the mean/variance of each channel plus the
fraction of pixels falling inside the canonical skin chroma box
(``133<=Cr<=173`` and ``77<=Cb<=127`` on the 8-bit scale, converted to unity).
"""

from __future__ import annotations

from dataclasses import dataclass

from .utils import FeatureResult, ImageContext

# Canonical 8-bit skin-tone chroma bounds (Cr, Cb), normalised to [0,1].
_CR_LO, _CR_HI = 133.0 / 255.0, 173.0 / 255.0
_CB_LO, _CB_HI = 77.0 / 255.0, 127.0 / 255.0


@dataclass
class YCrCbFeatures(FeatureResult):
    """YCrCb analysis results (all channels in ``[0,1]``)."""

    mean_y: float = 0.0
    mean_cr: float = 0.0
    mean_cb: float = 0.0
    variance_y: float = 0.0
    variance_cr: float = 0.0
    variance_cb: float = 0.0
    skin_pixel_fraction: float = 0.0
    skin_mean_cr: float = 0.0
    skin_mean_cb: float = 0.0


class YCrCbAnalyzer:
    """Computes :class:`YCrCbFeatures` from an :class:`ImageContext`."""

    def analyze(self, ctx: ImageContext) -> YCrCbFeatures:
        xp = ctx.xp
        ycrcb = ctx.ycrcb.reshape(-1, 3)
        y = ycrcb[:, 0]
        cr = ycrcb[:, 1]
        cb = ycrcb[:, 2]

        mean = ycrcb.mean(axis=0)
        var = ycrcb.var(axis=0)

        # Boolean mask of chroma-plausible skin pixels (vectorised, no loops).
        skin_mask = (cr >= _CR_LO) & (cr <= _CR_HI) & (cb >= _CB_LO) & (cb <= _CB_HI)
        skin_count = float(skin_mask.sum())
        frac = skin_count / float(skin_mask.size)
        if skin_count > 0:
            skin_cr = float(cr[skin_mask].mean())
            skin_cb = float(cb[skin_mask].mean())
        else:
            skin_cr = 0.0
            skin_cb = 0.0

        return YCrCbFeatures(
            mean_y=float(mean[0]),
            mean_cr=float(mean[1]),
            mean_cb=float(mean[2]),
            variance_y=float(var[0]),
            variance_cr=float(var[1]),
            variance_cb=float(var[2]),
            skin_pixel_fraction=frac,
            skin_mean_cr=skin_cr,
            skin_mean_cb=skin_cb,
        )
