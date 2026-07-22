"""Test entry point: Pudongdadao — lightweight mode for quick iteration.

Differences from production run_pudongdadao.py:
  - Uses yolo26m-pose (medium model, ~2x faster inference)
  - FP16 half-precision enabled
  - show_arm_bend enabled (elbow bend angle in metrics panel)
  - Runs with frame_skip=0 for accuracy but faster model
  - Separate output filename to avoid overwriting production results
"""

import sys
import time
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
VIDEO_PATH = r"\\10.151.2.205\共享文件2\短视频\浦东大道\clipped_segments\浦东大道1.mp4"
MODEL_PATH = str(Path(MODEL_DIR) / "yolo26x-pose.pt")
ANNOTATIONS_FILE = str(Path(DATA_DIR) / "regions_pudongdadao.json")

# ---------------------------------------------------------------------------
# Detection rules
# ---------------------------------------------------------------------------
DETECTION_RULES = [
    {"name": "rule_A", "type": "parallel_line", "ref_line": "line_1", "min_arm_torso_angle": 0, "dynamic_angle": True},
    {"name": "rule_B", "type": "parallel_line", "ref_line": "line_2", "allow_elbow": True, "dynamic_angle": True},
    {"name": "rule_C", "type": "pass_region", "target_region": "region_1"},
    {"name": "rule_D", "type": "parallel_line", "ref_line": "line_1", "anti_parallel": True, "dynamic_angle": True},
]

# ---------------------------------------------------------------------------
# Action mapping
# ---------------------------------------------------------------------------
ACTION_MAPPING = [
    {"action": "Act1 Call", "rule": "rule_A", "occurrence": 1},
    {"action": "Act2 CloseDoor", "rule": "rule_B", "occurrence": 1},
    {"action": "Act3 CheckGap", "rule": "rule_A", "occurrence": 2},
    {"action": "Act4 CheckLight", "rule": "rule_C", "occurrence": 1},
    {"action": "Act5 CheckSwitch", "rule": "rule_D", "occurrence": 1},
]

DETECTION_KWARGS = {
    "angle_threshold": 40,
    "min_arm_len": 30,
    "min_arm_torso_angle": 45,
    "dynamic_angle_coeff": 0.6,
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    t_start = time.time()
    print("=" * 60)
    print("  浦东大道站 — 测试模式")
    print(f"  模型: yolo26x-pose")
    print(f"  FP16: 开启")
    print(f"  关键点角度补偿: 开启 (dynamic_angle_coeff=0.6)")
    print(f"  反平行检测: rule_D (道岔确认, 140°~180°)")
    print(f"  输出: pose_output_pudongdadao_test.mp4")
    print("=" * 60)

    regions, lines = load_annotations(ANNOTATIONS_FILE)
    print(f"已加载: {len(regions)} 个区域, {len(lines)} 条参考线")

    detector = ParallelDetector(
        DETECTION_RULES, regions, lines,
        hold_frames=20, frame_decay=2, cooldown_frames=90,
        detection_kwargs=DETECTION_KWARGS,
    )

    model = YOLO(MODEL_PATH)
    model.to("cuda" if __import__("torch").cuda.is_available() else "cpu")
    print(f"推理设备: {'CUDA' if __import__('torch').cuda.is_available() else 'CPU'}")
    print()

    player = VideoPlayer(
        model, VIDEO_PATH, detector, ACTION_MAPPING,
        annotations_file=ANNOTATIONS_FILE,
        output_dir=str(Path(OUTPUT_DIR)),
        output_name="pose_output_pudongdadao_test.mp4",
        station_name="浦东大道(测试)", model_path=MODEL_PATH,
        imgsz=640, frame_skip=0, half=True,
        conf_low_threshold=0.3, conf_mid_threshold=0.6,
        train_mad_threshold=20,
        show_arm_bend=True,
    )
    player.run()

    elapsed = time.time() - t_start
    print(f"\n测试完成, 总耗时: {elapsed:.1f}s")
