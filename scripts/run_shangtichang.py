"""Entry point: Shangtichang video — parallel-line + pass-region detection."""

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
VIDEO_PATH = r"\\10.151.2.205\共享文件2\司机行为规范样本采样\短视频\上体场2.mp4"
MODEL_PATH = str(Path(MODEL_DIR) / "yolo26x-pose.pt")
ANNOTATIONS_FILE = str(Path(DATA_DIR) / "regions_shangtichang.json")

# ---------------------------------------------------------------------------
# Detection rules (unique conditions; each runs independently)
# ---------------------------------------------------------------------------
DETECTION_RULES = [
    {"name": "rule_A", "type": "parallel_line", "ref_line": "line_1"},
    {"name": "rule_B", "type": "parallel_line", "ref_line": "line_2", "allow_elbow": True},
    {"name": "rule_C", "type": "pass_region", "target_region": "region_1"},
]

# ---------------------------------------------------------------------------
# Action mapping: which rule occurrence maps to which action
# ---------------------------------------------------------------------------
ACTION_MAPPING = [
    {"action": "动作1", "rule": "rule_A", "occurrence": 1},  # 手指呼唤
    {"action": "动作2", "rule": "rule_B", "occurrence": 1},  # 手动关门
    {"action": "动作3", "rule": "rule_A", "occurrence": 2},  # 确认夹缝 (同规则，第2次出现)
    {"action": "动作4", "rule": "rule_C", "occurrence": 1},  # 确认站台指示灯
]

DETECTION_KWARGS = {
    "angle_threshold": 40,
    "min_arm_len": 30,
    "min_arm_torso_angle": 45,  # 手臂 vs 躯干夹角需 >45°，防止未抬臂的误触发
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
        output_name="pose_output_shangtichang.mp4",
        imgsz=480, frame_skip=1,
    )
    player.run()
