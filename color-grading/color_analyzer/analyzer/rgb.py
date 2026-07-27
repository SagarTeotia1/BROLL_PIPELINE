"""RGB channel analysis.

Extracts per-channel first/second-order statistics, entropy, and the inter-
channel balance ratios that characterise a colour cast at the RGB level.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .utils import ChannelStats, FeatureResult, ImageContext, channel_stats, safe_divide


@dataclass
class RGBFeatures(FeatureResult):
    """Container of RGB analysis results.

    ``r_over_g``/``r_over_b``/``g_over_b`` are ratios of channel means; values
    ``> 1`` indicate the numerator channel dominates (e.g. ``r_over_b > 1`` is a
    warm/red-biased cast).  ``channel_balance`` is the std of the three channel
    means normalised by their mean — 0 for a perfectly neutral image.
    """

    red: ChannelStats = field(default_factory=ChannelStats)
    green: ChannelStats = field(default_factory=ChannelStats)
    blue: ChannelStats = field(default_factory=ChannelStats)
    r_over_g: float = 1.0
    r_over_b: float = 1.0
    g_over_b: float = 1.0
    channel_balance: float = 0.0


class RGBAnalyzer:
    """Computes :class:`RGBFeatures` from an :class:`ImageContext`."""

    def __init__(self, bins: int = 256) -> None:
        self.bins = bins

    def analyze(self, ctx: ImageContext) -> RGBFeatures:
        xp = ctx.xp
        rgb = ctx.rgb
        red = channel_stats(xp, rgb[..., 0], self.bins, (0.0, 1.0))
        green = channel_stats(xp, rgb[..., 1], self.bins, (0.0, 1.0))
        blue = channel_stats(xp, rgb[..., 2], self.bins, (0.0, 1.0))

        r_mean, g_mean, b_mean = red.mean, green.mean, blue.mean
        # channel_balance: coefficient of variation of the three channel means.
        means = xp.asarray([r_mean, g_mean, b_mean])
        overall = float(means.mean())
        balance = float(means.std() / (overall + 1e-8))

        return RGBFeatures(
            red=red,
            green=green,
            blue=blue,
            r_over_g=float(safe_divide(xp, r_mean, g_mean)),
            r_over_b=float(safe_divide(xp, r_mean, b_mean)),
            g_over_b=float(safe_divide(xp, g_mean, b_mean)),
            channel_balance=balance,
        )
