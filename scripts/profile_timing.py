"""无头性能分析：测量每阶段耗时，输出计时表格。"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
from ultralytics import YOLO
from src.config import MODEL_DIR, DATA_DIR, OUTPUT_DIR
from src.detector import ParallelDetector
from src.annotation import load_annotations
from src import visualization as viz

# —— 配置（与 run_shangtichang.py 一致） ——
VIDEO_PATH = r"D:\科研\申通技术中心\端头门司机行为分析\上体场2.mp4"
MODEL_PATH = str(Path(MODEL_DIR) / "yolo26x-pose.pt")
ANNOTATIONS_FILE = str(Path(DATA_DIR) / "regions_shangtichang.json")

DETECTION_RULES = [
    {"name": "rule_A", "type": "parallel_line", "ref_line": "line_1"},
    {"name": "rule_B", "type": "parallel_line", "ref_line": "line_2", "allow_elbow": True},
    {"name": "rule_C", "type": "pass_region", "target_region": "region_1"},
]

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


def main():
    # —— 加载 ——
    regions, lines = load_annotations(ANNOTATIONS_FILE)
    detector = ParallelDetector(
        DETECTION_RULES, regions, lines,
        hold_frames=30, frame_decay=2, cooldown_frames=90,
        detection_kwargs=DETECTION_KWARGS,
    )
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"视频: {width}x{height}, {fps:.1f}fps, {total_frames}帧, {total_frames/fps:.1f}秒")

    # 跑 10 秒
    max_frames = int(fps * 10)
    print(f"分析前 {max_frames} 帧（约10秒）\n")

    # —— 计时累积 ——
    timings = {
        "read": 0.0,
        "infer": 0.0,
        "detect": 0.0,
        "metrics": 0.0,
        "render_pose": 0.0,
        "render_rays": 0.0,
        "render_anno": 0.0,
        "render_status": 0.0,
        "render_metrics_draw": 0.0,
    }
    frame_count = 0

    t_start = time.perf_counter()
    for _ in range(max_frames):
        # — read —
        t0 = time.perf_counter()
        ret, frame = cap.read()
        timings["read"] += time.perf_counter() - t0
        if not ret:
            break

        # — YOLO inference —
        t0 = time.perf_counter()
        results = model(frame, verbose=False, conf=0.5, imgsz=640)
        timings["infer"] += time.perf_counter() - t0

        kp = results[0].keypoints if (results and results[0].keypoints is not None) else None

        # — parallel detection —
        t0 = time.perf_counter()
        active, new_events = detector.update(kp)
        timings["detect"] += time.perf_counter() - t0

        # — metrics compute —
        t0 = time.perf_counter()
        metrics = viz.compute_action_metrics(
            kp, ACTION_MAPPING, detector.rules, detector.regions, detector.lines,
            detector.detection_kwargs)
        timings["metrics"] += time.perf_counter() - t0

        # — render: pose skeleton —
        t0 = time.perf_counter()
        annotated = viz.draw_pose(frame, results)
        timings["render_pose"] += time.perf_counter() - t0

        # — render: arm rays —
        t0 = time.perf_counter()
        viz.draw_arm_rays(annotated, kp, detector.regions)
        timings["render_rays"] += time.perf_counter() - t0

        # — render: annotations (regions + lines) —
        t0 = time.perf_counter()
        viz.draw_annotations(annotated, detector.regions, detector.lines, None)
        timings["render_anno"] += time.perf_counter() - t0

        # — render: status overlay —
        t0 = time.perf_counter()
        annotated, status_bottom = viz.draw_status_overlay(
            annotated, detector.rules, active, detector.events, ACTION_MAPPING)
        timings["render_status"] += time.perf_counter() - t0

        # — render: action metrics —
        t0 = time.perf_counter()
        viz.draw_action_metrics(annotated, metrics, y=status_bottom + 6)
        timings["render_metrics_draw"] += time.perf_counter() - t0

        frame_count += 1
        if frame_count % 50 == 0:
            print(f"  已处理 {frame_count}/{max_frames} 帧...")

    total_time = time.perf_counter() - t_start
    cap.release()

    # —— 输出表格 ——
    print(f"\n{'='*65}")
    print(f"  性能分析 — 共 {frame_count} 帧，总耗时 {total_time:.2f} 秒")
    print(f"{'='*65}")
    print(f"  {'阶段':<25s} {'总耗时':>8s} {'占比':>7s} {'平均/帧':>10s}")
    print(f"  {'-'*55}")

    total_render = sum(timings[k] for k in timings if k.startswith("render_"))
    total_measured = sum(timings.values())
    other = total_time - total_measured  # 未测量部分

    for name, t in timings.items():
        pct = t / total_time * 100
        avg = t / frame_count * 1000
        print(f"  {name:<25s} {t:>7.2f}s {pct:>6.1f}% {avg:>8.2f}ms")

    # 汇总行
    render_total = sum(timings[k] for k in timings if k.startswith("render_"))
    print(f"  {'-'*55}")
    print(f"  {'[渲染小计]':<25s} {render_total:>7.2f}s {render_total/total_time*100:>6.1f}% {render_total/frame_count*1000:>8.2f}ms")
    print(f"  {'[GPU推理]':<25s} {timings['infer']:>7.2f}s {timings['infer']/total_time*100:>6.1f}% {timings['infer']/frame_count*1000:>8.2f}ms")
    print(f"  {'[未测量/other]':<25s} {other:>7.2f}s {other/total_time*100:>6.1f}% {other/frame_count*1000:>8.2f}ms")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
