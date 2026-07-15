"""Centralized configuration constants for the pose detection system."""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
DEFAULT_MODEL = os.path.join(MODEL_DIR, "yolo26x-pose.pt")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")

# ---------------------------------------------------------------------------
# Keypoint indices (COCO 17-keypoint convention)
#   5: left shoulder    6: right shoulder
#   7: left elbow       8: right elbow
#   9: left wrist      10: right wrist
#  11: left hip        12: right hip
# ---------------------------------------------------------------------------
SHOW_KEYPOINTS = list(range(5, 13))  # [5, 6, 7, 8, 9, 10, 11, 12]

SKELETON = [
    (5, 6),     # left shoulder - right shoulder
    (5, 7),     # left shoulder - left elbow
    (7, 9),     # left elbow - left wrist
    (6, 8),     # right shoulder - right elbow
    (8, 10),    # right elbow - right wrist
    (5, 11),    # left shoulder - left hip
    (6, 12),    # right shoulder - right hip
    (11, 12),   # left hip - right hip
]

# BGR colors for each keypoint
KP_COLORS = {
    5:  (255, 0, 0),      # left shoulder  — blue
    6:  (0, 0, 255),      # right shoulder — red
    7:  (255, 128, 0),    # left elbow     — light blue
    8:  (0, 128, 255),    # right elbow    — orange
    9:  (255, 255, 0),    # left wrist     — cyan
    10: (0, 255, 255),    # right wrist    — yellow
    11: (128, 0, 255),    # left hip       — purple
    12: (255, 0, 128),    # right hip      — pink
}

LINE_COLOR = (0, 255, 0)       # skeleton line — green
CONF_THRESHOLD = 0.5            # minimum keypoint confidence

# ---------------------------------------------------------------------------
# Arm-side lookup: (shoulder_id, wrist_id, elbow_id, label)
# ---------------------------------------------------------------------------
ARM_SIDES = [
    (5, 9, 7, "L"),   # left arm
    (6, 10, 8, "R"),  # right arm
]

# ---------------------------------------------------------------------------
# Detection thresholds
# ---------------------------------------------------------------------------
DEFAULT_ANGLE_THRESHOLD = 30        # angle threshold when no reference line (degrees)
DEFAULT_LOOSE_ANGLE_THRESHOLD = 55  # loose angle threshold with reference line
DEFAULT_LINE_ANGLE_THRESHOLD = 40   # max angle between arm and reference line
DEFAULT_HOLD_FRAMES = 15            # consecutive frames to confirm action
DEFAULT_MIN_ARM_LEN = 30            # minimum arm pixel length to consider
FRAME_DECAY = 2                     # hold counter decay per frame (tolerates brief dropout)

# ---------------------------------------------------------------------------
# Confidence colour tiers (for keypoint visualisation)
# ---------------------------------------------------------------------------
CONF_LOW_THRESHOLD = 0.3             # below this: red
CONF_MID_THRESHOLD = 0.6             # below this (>= low): yellow;  above: green
