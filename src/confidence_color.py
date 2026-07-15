"""Confidence-based keypoint colouring with configurable three-tier thresholds.

Usage::

    from .confidence_color import ConfidenceColorMapper

    mapper = ConfidenceColorMapper(low_threshold=0.3, mid_threshold=0.6)
    colour = mapper.get_colour(0.75)   # -> (0, 255, 0)  green (high)
    tier  = mapper.get_tier(0.45)      # -> 'M'
"""

# Default BGR colours for three confidence tiers
DEFAULT_LOW_COLOUR  = (0, 0, 255)      # red
DEFAULT_MID_COLOUR  = (0, 200, 255)    # yellow-orange
DEFAULT_HIGH_COLOUR = (0, 255, 0)      # green


class ConfidenceColorMapper:
    """Map a confidence value [0, 1] to one of three BGR colours.

    Parameters:
        low_threshold: values **below** this are "low".
        mid_threshold: values **below** this (and >= low) are "mid".
                       Values >= mid_threshold are "high".
        low_colour: BGR tuple for low-confidence keypoints.
        mid_colour: BGR tuple for mid-confidence keypoints.
        high_colour: BGR tuple for high-confidence keypoints.
    """

    def __init__(self, *,
                 low_threshold: float = 0.3,
                 mid_threshold: float = 0.6,
                 low_colour  = DEFAULT_LOW_COLOUR,
                 mid_colour  = DEFAULT_MID_COLOUR,
                 high_colour = DEFAULT_HIGH_COLOUR):
        self.low_threshold = low_threshold
        self.mid_threshold = mid_threshold
        self.low_colour  = low_colour
        self.mid_colour  = mid_colour
        self.high_colour = high_colour

    # -- public -----------------------------------------------------------

    def get_colour(self, conf: float):
        """Return the BGR colour for *conf*."""
        if conf < self.low_threshold:
            return self.low_colour
        if conf < self.mid_threshold:
            return self.mid_colour
        return self.high_colour

    def get_tier(self, conf: float) -> str:
        """Return a single-character tier label: ``'L'``, ``'M'``, or ``'H'``."""
        if conf < self.low_threshold:
            return 'L'
        if conf < self.mid_threshold:
            return 'M'
        return 'H'

    # -- legend helpers ---------------------------------------------------

    @property
    def legend_entries(self):
        """Return ``[(label, colour), ...]`` for drawing a confidence legend."""
        return [
            (f'High  (>= {self.mid_threshold:.1f})', self.high_colour),
            (f'Mid   ({self.low_threshold:.1f}-{self.mid_threshold:.1f})', self.mid_colour),
            (f'Low   (< {self.low_threshold:.1f})', self.low_colour),
        ]
