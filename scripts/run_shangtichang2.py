"""Entry point: Shangti Field 2 video — parallel-line + pass-region detection."""

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
VIDEO_PATH = r"\\10.151.2.205\共享文件2\司机行为规范样本采样\短视频\上体场2.mp4"
MODEL_PATH = str(Path(MODEL_DIR) / "yolo26x-pose.pt")
ANNOTATIONS_FILE = str(Path(DATA_DIR) / "regions_shangtichang2.json")

# ---------------------------------------------------------------------------
# Action sequence
# ---------------------------------------------------------------------------
ACTION_SEQUENCE = [
    {"name": "动作1", "ref_line": "line_1", "type": "parallel_line"},
    {"name": "动作2", "ref_line": "line_2", "type": "parallel_line", "allow_elbow": True},
    {"name": "动作3", "ref_line": "line_1", "type": "parallel_line"},
    {"name": "动作4", "target_region": "region_1", "type": "pass_region"},
]

DETECTION_KWARGS = {
    "angle_threshold": 40,   # max angle between arm and reference line
    "min_arm_len": 30,
    "hold_frames": 15,
    "frame_decay": 2,
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
        output_name="pose_output_shangtichang2.mp4",
    )
    player.run()
