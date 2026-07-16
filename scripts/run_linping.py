"""Entry point: Linping video — parallel-line + pass-region detection."""

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
VIDEO_PATH = r"\\10.151.2.205\共享文件2\短视频\临平\clipped_segments\临平3.mp4"  # TODO: fill in video path
MODEL_PATH = str(Path(MODEL_DIR) / "yolo26x-pose.pt")
ANNOTATIONS_FILE = str(Path(DATA_DIR) / "regions_linping.json")

# ---------------------------------------------------------------------------
# Detection rules (unique conditions; each runs independently)
# ---------------------------------------------------------------------------
DETECTION_RULES = [
    {"name": "rule_A", "type": "parallel_line", "ref_line": "line_1", "min_arm_torso_angle": 0, "dynamic_angle": True},
    {"name": "rule_B", "type": "parallel_line", "ref_line": "line_2", "allow_elbow": True, "dynamic_angle": True},
    {"name": "rule_C", "type": "pass_region", "target_region": "region_1"},
]

# ---------------------------------------------------------------------------
# Action mapping: which rule occurrence maps to which action
# ---------------------------------------------------------------------------
ACTION_MAPPING = [
    {"action": "Act1 Call", "rule": "rule_A", "occurrence": 1},
    {"action": "Act2 CloseDoor", "rule": "rule_B", "occurrence": 1},
    {"action": "Act3 CheckGap", "rule": "rule_A", "occurrence": 2},
    {"action": "Act4 CheckLight", "rule": "rule_C", "occurrence": 1},
]

DETECTION_KWARGS = {
    "angle_threshold": 40,
    "min_arm_len": 30,
    "min_arm_torso_angle": 45,
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    regions, lines = load_annotations(ANNOTATIONS_FILE)

    detector = ParallelDetector(
        DETECTION_RULES, regions, lines,
        hold_frames=30, frame_decay=2, cooldown_frames=90,
        detection_kwargs=DETECTION_KWARGS,
    )

    model = YOLO(MODEL_PATH)
    player = VideoPlayer(
        model, VIDEO_PATH, detector, ACTION_MAPPING,
        annotations_file=ANNOTATIONS_FILE,
        output_dir=str(Path(OUTPUT_DIR)),
        output_name="pose_output_linping.mp4",
        imgsz=640, frame_skip=0,
        conf_low_threshold=0.3, conf_mid_threshold=0.6,
        train_mad_threshold=20,
    )
    player.run()
