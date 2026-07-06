"""Streamlit dashboard for driver pose action detection."""

import sys
import os
import time
import tempfile
import json
from pathlib import Path

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

from src.config import (
    MODEL_DIR, DATA_DIR, OUTPUT_DIR,
    DEFAULT_ANGLE_THRESHOLD, DEFAULT_LINE_ANGLE_THRESHOLD,
    DEFAULT_LOOSE_ANGLE_THRESHOLD, DEFAULT_HOLD_FRAMES,
    DEFAULT_MIN_ARM_LEN,
)
from src.detector import ParallelDetector
from src.analyzer import SequenceAnalyzer
from src.annotation import load_annotations
from src.visualization import (
    draw_pose, draw_annotations, draw_frame_info,
    draw_arm_rays, draw_status_overlay, draw_analysis_result,
    put_text_cn,
)
from src.config import SHOW_KEYPOINTS, SKELETON

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="司机姿态检测",
    page_icon="🚇",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("🚇 司机姿态检测")
st.sidebar.divider()

scenario = st.sidebar.selectbox(
    "选择场景", ["上体场2", "宝山1"], key="scenario"
)

# Model
model_options = {
    "yolo26x-pose.pt": str(Path(MODEL_DIR) / "yolo26x-pose.pt"),
    "yolo26m-pose.pt": str(Path(MODEL_DIR) / "yolo26m-pose.pt"),
}
existing_models = {k: v for k, v in model_options.items() if os.path.exists(v)}
if not existing_models:
    st.sidebar.error("模型文件不存在，请将 .pt 模型放入 models/ 目录")
    st.stop()

model_name = st.sidebar.selectbox("模型", list(existing_models.keys()))
model_path = existing_models[model_name]

# Video source
st.sidebar.divider()
video_option = st.sidebar.radio("视频来源", ["默认视频", "上传视频"])

video_path = None
if video_option == "默认视频":
    default_videos = {
        "上体场2": r"\\10.151.2.205\共享文件2\司机行为规范样本采样\短视频\上体场3.mp4",
        "宝山1": r"\\10.151.2.205\共享文件2\司机行为规范样本采样\短视频\宝山1.mp4",
    }
    video_path = default_videos[scenario]
    st.sidebar.info(f"视频: {Path(video_path).name}")
else:
    uploaded = st.sidebar.file_uploader("上传视频", type=["mp4", "avi", "mov"])
    if uploaded:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp.write(uploaded.read())
        video_path = tmp.name
        st.sidebar.success("视频已上传")

# Source video preview
st.sidebar.divider()
with st.sidebar.expander("🎬 检测视频", expanded=False):
    if video_path and os.path.exists(video_path):
        st.video(video_path)
    else:
        st.caption("选择视频后在此预览")

# Result video
st.sidebar.divider()
result_path = os.path.join(OUTPUT_DIR, "streamlit", f"output_{scenario}.mp4")
with st.sidebar.expander("📼 检测结果", expanded=False):
    if os.path.exists(result_path):
        st.video(result_path)
    else:
        st.caption("检测完成后在此查看结果视频")

# Parameters
st.sidebar.divider()
st.sidebar.subheader("检测参数")

with st.sidebar.expander("阈值设置", expanded=True):
    angle_threshold = st.slider("平行角度阈值", 10, 90, 40, 5)
    hold_frames = st.slider("确认帧数", 5, 60, 15, 1)
    cooldown_frames = st.slider("冷却帧数", 10, 120, 45, 5)
    frame_decay = st.slider("帧衰减", 0, 10, 2, 1)
    min_arm_len = st.slider("最小手臂长度(px)", 10, 100, 30, 5)

with st.sidebar.expander("高级参数", expanded=False):
    model_conf = st.slider("关键点置信度", 0.1, 1.0, 0.5, 0.05)
    torso_angle = st.slider("躯干夹角下限", 0, 90, 45, 5,
                            help="手臂 vs 躯干夹角需大于此值，0=不检查")
    line_angle_threshold = st.slider("参考线夹角", 10, 90, 40, 5)
    loose_angle_threshold = st.slider("区域指向松阈值", 10, 90, 55, 5)
    extend_ray = st.checkbox("延长手臂射线(6×)", True)

start = st.sidebar.button("▶ 开始检测", use_container_width=True, type="primary")

# ---------------------------------------------------------------------------
# Scenario configs
# ---------------------------------------------------------------------------
SCENARIO_CONFIGS = {
    "上体场": {
        "annotations_file": str(Path(DATA_DIR) / "regions_shangtichang.json"),
        "rules": [
            {"name": "rule_A", "type": "parallel_line", "ref_line": "line_1"},
            {"name": "rule_B", "type": "parallel_line", "ref_line": "line_2", "allow_elbow": True},
            {"name": "rule_C", "type": "pass_region", "target_region": "region_1"},
        ],
        "action_mapping": [
            {"action": "动作1 — 手指呼唤", "rule": "rule_A", "occurrence": 1},
            {"action": "动作2 — 手动关门", "rule": "rule_B", "occurrence": 1},
            {"action": "动作3 — 确认夹缝", "rule": "rule_A", "occurrence": 2},
            {"action": "动作4 — 确认指示灯", "rule": "rule_C", "occurrence": 1},
        ],
    },
    "宝山": {
        "annotations_file": str(Path(DATA_DIR) / "regions_baoshan.json"),
        "rules": [
            {"name": "rule_A", "type": "pointing_with_line",
             "target_region": "region_1", "ref_line": "line_1"},
            {"name": "rule_B", "type": "pointing", "target_region": "region_2"},
            {"name": "rule_C", "type": "pointing", "target_region": "region_3"},
            {"name": "rule_D", "type": "pointing_with_line",
             "target_region": "region_4", "ref_line": "line_1"},
        ],
        "action_mapping": [
            {"action": "动作1 — 指向前方", "rule": "rule_A", "occurrence": 1},
            {"action": "动作2 — 确认区域2", "rule": "rule_B", "occurrence": 1},
            {"action": "动作3 — 再次指向前方", "rule": "rule_A", "occurrence": 2},
            {"action": "动作4 — 确认区域3", "rule": "rule_C", "occurrence": 1},
            {"action": "动作5 — 确认区域4", "rule": "rule_D", "occurrence": 1},
        ],
    },
}

# ---------------------------------------------------------------------------
# Processing engine
# ---------------------------------------------------------------------------

def process_video(video_path, detector, model, action_mapping,
                  output_path, progress_callback=None):
    """Run detection on each frame, collect events, and write annotated output.

    Returns:
        ``(output_path, events, analysis, total_frames)``
    """
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    detector.reset()
    frame_idx = 0
    _last_progress = -1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False, conf=0.5)
        kp = results[0].keypoints if results[0].keypoints is not None else None
        active, new_events = detector.update(kp)

        annotated = draw_pose(frame, results)
        draw_arm_rays(annotated, kp, detector.regions)
        draw_annotations(annotated, detector.regions, detector.lines)
        annotated = draw_status_overlay(
            annotated, detector.rules, active,
            detector.events, action_mapping)
        draw_frame_info(annotated, frame_idx + 1, total, fps)

        out.write(annotated)

        if progress_callback:
            pct = int((frame_idx + 1) / total * 100)
            if pct != _last_progress:
                progress_callback(pct, active, new_events)
                _last_progress = pct

        frame_idx += 1

    cap.release()
    out.release()

    analyzer = SequenceAnalyzer(detector.events, action_mapping, fps=fps)
    analysis = analyzer.analyze()

    return output_path, detector.events, analysis, total, fps


# ---------------------------------------------------------------------------
# Main display
# ---------------------------------------------------------------------------
st.title("司机标准动作检测系统")
st.caption(f"场景: {scenario}")

if not start:
    # Preview mode — show scenario info
    cfg = SCENARIO_CONFIGS[scenario]
    regions, lines = load_annotations(cfg["annotations_file"])

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("检测规则")
        rule_data = []
        for r in cfg["rules"]:
            type_label = {
                "parallel_line": "平行线",
                "pass_region": "穿区域",
                "pointing": "角度指向",
                "pointing_with_line": "平+指",
            }.get(r["type"], r["type"])
            target = r.get("ref_line") or r.get("target_region") or "-"
            rule_data.append({"规则名": r["name"], "类型": type_label,
                              "目标": target, "备注": str(r.get("allow_elbow", "")) if r.get("allow_elbow") else ""})
        st.dataframe(rule_data, use_container_width=True, hide_index=True)

    with col2:
        st.subheader("动作序列")
        for i, am in enumerate(cfg["action_mapping"], 1):
            st.markdown(f"**{i}.** {am['action']}  `{am['rule']}#{am['occurrence']}`")

        st.divider()
        st.subheader("标注信息")
        st.metric("区域数", len(regions))
        st.metric("参考线数", len(lines))

    st.info("在左侧面板调整参数后，点击「开始检测」运行分析")
    st.stop()


# ---------------------------------------------------------------------------
# Run detection
# ---------------------------------------------------------------------------
cfg = SCENARIO_CONFIGS[scenario]
regions, lines = load_annotations(cfg["annotations_file"])

detector = ParallelDetector(
    cfg["rules"], regions, lines,
    hold_frames=hold_frames,
    frame_decay=frame_decay,
    cooldown_frames=cooldown_frames,
    detection_kwargs={
        "angle_threshold": angle_threshold,
        "line_angle_threshold": line_angle_threshold,
        "loose_angle_threshold": loose_angle_threshold,
        "min_arm_len": min_arm_len,
        "min_arm_torso_angle": torso_angle,
        "extend_ray": extend_ray,
    },
)

with st.spinner("加载模型..."):
    model = YOLO(model_path)

output_dir = os.path.join(OUTPUT_DIR, "streamlit")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, f"output_{scenario}.mp4")

# --- Progress UI ---
progress_bar = st.progress(0, text="准备检测...")
status_col1, status_col2, status_col3, status_col4 = st.columns(4)
pct_metric = status_col1.empty()
event_metric = status_col2.empty()
rule_metric = status_col3.empty()
fps_metric = status_col4.empty()

_active_display = st.empty()
st.divider()

def on_progress(pct, active, new_events):
    progress_bar.progress(pct, text=f"处理中... {pct}%")
    pct_metric.metric("进度", f"{pct}%")
    event_metric.metric("事件数", len(detector.events))
    rule_metric.metric("活跃规则", len(active))
    if active:
        names = ", ".join(f"{k}({v['hold']}/{v['required']})" for k, v in list(active.items())[:3])
        _active_display.caption(f"活跃: {names}")

t0 = time.time()
output_video, events, analysis, total_frames, video_fps = process_video(
    video_path, detector, model, cfg["action_mapping"],
    output_path, progress_callback=on_progress,
)
elapsed = time.time() - t0

progress_bar.progress(100, text="完成!")
fps_metric.metric("处理速度", f"{total_frames / elapsed:.1f} fps")
_active_display.empty()

st.balloons()

# ---------------------------------------------------------------------------
# Results: tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📊 总览", "🎬 检测视频", "📋 事件列表", "📈 分析报告"])

# --- Tab 1: Overview ---
with tab1:
    st.subheader("检测总览")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("总帧数", total_frames)
    m2.metric("检测事件", len(events))
    m3.metric("耗时", f"{elapsed:.1f}s")
    m4.metric("处理速度", f"{total_frames / elapsed:.1f} fps")
    m5.metric("视频FPS", f"{video_fps:.1f}")

    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("动作完成情况")
        for a in analysis["actions"]:
            if a["found"]:
                side_label = "左臂" if a.get("side") == "L" else \
                             "右臂" if a.get("side") == "R" else "?"
                ts = a.get("timestamp", 0)
                st.success(
                    f"✅ **{a['action']}**  —  {side_label}  "
                    f"@ {ts:.1f}s (帧{a.get('frame', '?')})  "
                    f"{a.get('angle', 0):.0f}°"
                )
            else:
                st.error(f"❌ **{a['action']}**  —  未检测到")

    with col_b:
        st.subheader("合规性判定")
        if analysis["all_found"] and analysis["order_valid"]:
            st.success("✅ 全部动作完成，顺序正确")
        elif analysis["all_found"]:
            st.warning("⚠️ 全部动作完成，但顺序异常")
        else:
            missing = [a["action"] for a in analysis["actions"] if not a["found"]]
            st.error(f"❌ 缺失动作: {', '.join(missing)}")

        st.subheader("规则触发统计")
        rule_counts = {}
        for e in events:
            rule_counts[e["rule"]] = rule_counts.get(e["rule"], 0) + 1
        for rn, count in rule_counts.items():
            st.metric(f"规则 `{rn}`", count)

# --- Tab 2: Video ---
with tab2:
    st.subheader("检测结果视频")
    if os.path.exists(output_video):
        st.info(f"已保存至: `{output_video}`")
        with open(output_video, "rb") as f:
            video_bytes = f.read()
        st.video(video_bytes)

        st.download_button(
            "⬇ 下载视频", video_bytes,
            file_name=f"pose_output_{scenario}.mp4",
            mime="video/mp4",
        )
    else:
        st.error("视频生成失败")

# --- Tab 3: Events ---
with tab3:
    st.subheader("原始检测事件")
    if events:
        event_data = []
        for e in events:
            event_data.append({
                "帧": e["frame"],
                "规则": e["rule"],
                "手臂": e["side"],
                "角度": f"{e['angle']:.1f}°",
                "腕坐标": f"({e['wrist'][0]:.0f}, {e['wrist'][1]:.0f})" if e["wrist"] else "-",
            })
        st.dataframe(event_data, use_container_width=True, hide_index=True)
    else:
        st.info("无检测事件")

# --- Tab 4: Analysis ---
with tab4:
    st.subheader("分析报告")
    analyzer = SequenceAnalyzer(events, cfg["action_mapping"], fps=video_fps)
    report = analyzer.summary()
    st.code(report, language=None)

    st.divider()
    st.subheader("参数记录")
    params = {
        "场景": scenario,
        "模型": model_name,
        "角度阈值": f"{angle_threshold}°",
        "确认帧数": hold_frames,
        "冷却帧数": cooldown_frames,
        "帧衰减": frame_decay,
        "最小臂长": f"{min_arm_len}px",
        "躯干夹角下限": f"{torso_angle}°",
        "参考线角度阈值": f"{line_angle_threshold}°",
        "区域松阈值": f"{loose_angle_threshold}°",
        "延长射线": extend_ray,
    }
    st.json(params)
