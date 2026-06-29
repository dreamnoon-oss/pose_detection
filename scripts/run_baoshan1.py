"""Entry point: Baoshan 1 video — angle-based pointing detection."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultralytics import YOLO
from src.config import MODEL_DIR, DATA_DIR, OUTPUT_DIR
from src.state_machine import ActionStateMachine
from src.annotation import load_annotations
from src.player import VideoPlayer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
VIDEO_PATH = r"\\10.151.2.205\共享文件2\司机行为规范样本采样\短视频\宝山1.mp4"
MODEL_PATH = str(Path(MODEL_DIR) / "yolo26x-pose.pt")
ANNOTATIONS_FILE = str(Path(DATA_DIR) / "regions_baoshan1.json")

# ---------------------------------------------------------------------------
# Action sequence (angle-based detection)
# ---------------------------------------------------------------------------
ACTION_SEQUENCE = [
    {"name": "动作1", "target_region": "region_1", "ref_line": "line_1", "type": "pointing_with_line"},
    {"name": "动作2", "target_region": "region_2", "type": "pointing"},
    {"name": "动作3", "target_region": "region_1", "ref_line": "line_1", "type": "pointing_with_line"},
    {"name": "动作4", "target_region": "region_3", "type": "pointing"},
    {"name": "动作5", "target_region": "region_4", "ref_line": "line_1", "type": "pointing_with_line"},
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

    sm = ActionStateMachine(
        ACTION_SEQUENCE, regions, lines,
        hold_frames=15, frame_decay=2,
        detection_kwargs=DETECTION_KWARGS,
    )

    model = YOLO(MODEL_PATH)
    player = VideoPlayer(
        model, VIDEO_PATH, sm,
        annotations_file=ANNOTATIONS_FILE,
        output_dir=str(Path(OUTPUT_DIR)),
        output_name="pose_output_baoshan1.mp4",
    )
    player.run()
