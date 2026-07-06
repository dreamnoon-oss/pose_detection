"""Entry point: Baoshan video — angle-based pointing detection."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultralytics import YOLO
from src.config import MODEL_DIR, DATA_DIR, OUTPUT_DIR
from src.detector import ParallelDetector
from src.annotation import load_annotations
from src.player import VideoPlayer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
VIDEO_PATH = r"\\10.151.2.205\共享文件2\司机行为规范样本采样\短视频\宝山1.mp4"
MODEL_PATH = str(Path(MODEL_DIR) / "yolo26x-pose.pt")
ANNOTATIONS_FILE = str(Path(DATA_DIR) / "regions_baoshan.json")

# ---------------------------------------------------------------------------
# Detection rules (unique conditions; each runs independently)
# ---------------------------------------------------------------------------
DETECTION_RULES = [
    {"name": "rule_A", "type": "pointing_with_line",
     "target_region": "region_1", "ref_line": "line_1"},
    {"name": "rule_B", "type": "pointing",
     "target_region": "region_2"},
    {"name": "rule_C", "type": "pointing",
     "target_region": "region_3"},
    {"name": "rule_D", "type": "pointing_with_line",
     "target_region": "region_4", "ref_line": "line_1"},
]

# ---------------------------------------------------------------------------
# Action mapping: which rule occurrence maps to which action
# ---------------------------------------------------------------------------
ACTION_MAPPING = [
    {"action": "动作1", "rule": "rule_A", "occurrence": 1},  # region_1 + line_1
    {"action": "动作2", "rule": "rule_B", "occurrence": 1},  # region_2
    {"action": "动作3", "rule": "rule_A", "occurrence": 2},  # region_1 + line_1 (同规则，第2次出现)
    {"action": "动作4", "rule": "rule_C", "occurrence": 1},  # region_3
    {"action": "动作5", "rule": "rule_D", "occurrence": 1},  # region_4 + line_1
]

DETECTION_KWARGS = {
    "angle_threshold": 30,
    "line_angle_threshold": 40,
    "loose_angle_threshold": 55,
    "min_arm_len": 30,
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    regions, lines = load_annotations(ANNOTATIONS_FILE)

    detector = ParallelDetector(
        DETECTION_RULES, regions, lines,
        hold_frames=15, frame_decay=2, cooldown_frames=45,
        detection_kwargs=DETECTION_KWARGS,
    )

    model = YOLO(MODEL_PATH)
    player = VideoPlayer(
        model, VIDEO_PATH, detector, ACTION_MAPPING,
        annotations_file=ANNOTATIONS_FILE,
        output_dir=str(Path(OUTPUT_DIR)),
        output_name="pose_output_baoshan.mp4",
    )
    player.run()
