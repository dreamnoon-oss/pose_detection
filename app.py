"""Streamlit dashboard — background detection + result playback."""

import sys
import os
import time
import tempfile
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
import cv2
from ultralytics import YOLO

from src.config import MODEL_DIR, DATA_DIR, OUTPUT_DIR
from src.detector import ParallelDetector
from src.analyzer import SequenceAnalyzer
from src.annotation import load_annotations
from src.visualization import (
    draw_pose, draw_annotations, draw_frame_info,
    draw_arm_rays, draw_status_overlay,
)

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
# Shared state
# ---------------------------------------------------------------------------
class DetectionState:
    def __init__(self):
        self.lock = threading.Lock()
        self.progress = 0
        self.frame = 0
        self.total = 0
        self.events = []
        self.active = {}
        self.done = False
        self.result = None
        self.error = None

    def update(self, **kw):
        with self.lock:
            for k, v in kw.items():
                setattr(self, k, v)

    def snapshot(self):
        with self.lock:
            return {k: getattr(self, k) for k in
                    ["progress", "frame", "total", "events",
                     "active", "done", "result", "error"]}
        # need safe copy for mutable fields
        with self.lock:
            return {
                "progress": self.progress,
                "frame": self.frame,
                "total": self.total,
                "events": list(self.events),
                "active": dict(self.active),
                "done": self.done,
                "result": self.result,
                "error": self.error,
            }


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("🚇 司机姿态检测")
st.sidebar.divider()

scenario = st.sidebar.selectbox("选择场景", ["上体场2", "宝山1"], key="scenario")

model_options = {
    "yolo26x-pose.pt": str(Path(MODEL_DIR) / "yolo26x-pose.pt"),
    "yolo26m-pose.pt": str(Path(MODEL_DIR) / "yolo26m-pose.pt"),
}
existing = {k: v for k, v in model_options.items() if os.path.exists(v)}
if not existing:
    st.sidebar.error("模型文件不存在，请将 .pt 模型放入 models/ 目录")
    st.stop()

model_name = st.sidebar.selectbox("模型", list(existing.keys()))
model_path = existing[model_name]

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

st.sidebar.divider()
st.sidebar.subheader("检测参数")

with st.sidebar.expander("阈值设置", expanded=True):
    angle_threshold = st.slider("平行角度阈值", 10, 90, 40, 5)
    hold_frames = st.slider("确认帧数", 5, 60, 15, 1)
    cooldown_frames = st.slider("冷却帧数", 10, 120, 45, 5)
    frame_decay = st.slider("帧衰减", 0, 10, 2, 1)
    min_arm_len = st.slider("最小手臂长度(px)", 10, 100, 30, 5)

with st.sidebar.expander("高级参数", expanded=False):
    torso_angle = st.slider("躯干夹角下限", 0, 90, 45, 5)
    line_angle_threshold = st.slider("参考线夹角", 10, 90, 40, 5)
    loose_angle_threshold = st.slider("区域指向松阈值", 10, 90, 55, 5)
    extend_ray = st.checkbox("延长手臂射线(6×)", True)

# ---------------------------------------------------------------------------
# Scenario configs
# ---------------------------------------------------------------------------
SCENARIO_CONFIGS = {
    "上体场2": {
        "annotations_file": str(Path(DATA_DIR) / "regions_shangtichang2.json"),
        "rules": [
            {"name": "rule_A", "type": "parallel_line", "ref_line": "line_1"},
            {"name": "rule_B", "type": "parallel_line", "ref_line": "line_2", "allow_elbow": True},
            {"name": "rule_C", "type": "pass_region", "target_region": "region_1"},
        ],
        "action_mapping": [
            {"action": "动作1 - 手指呼唤", "rule": "rule_A", "occurrence": 1},
            {"action": "动作2 - 手动关门", "rule": "rule_B", "occurrence": 1},
            {"action": "动作3 - 确认夹缝", "rule": "rule_A", "occurrence": 2},
            {"action": "动作4 - 确认指示灯", "rule": "rule_C", "occurrence": 1},
        ],
    },
    "宝山1": {
        "annotations_file": str(Path(DATA_DIR) / "regions_baoshan1.json"),
        "rules": [
            {"name": "rule_A", "type": "pointing_with_line",
             "target_region": "region_1", "ref_line": "line_1"},
            {"name": "rule_B", "type": "pointing", "target_region": "region_2"},
            {"name": "rule_C", "type": "pointing", "target_region": "region_3"},
            {"name": "rule_D", "type": "pointing_with_line",
             "target_region": "region_4", "ref_line": "line_1"},
        ],
        "action_mapping": [
            {"action": "动作1 - 指向前方", "rule": "rule_A", "occurrence": 1},
            {"action": "动作2 - 确认区域2", "rule": "rule_B", "occurrence": 1},
            {"action": "动作3 - 再次指向前方", "rule": "rule_A", "occurrence": 2},
            {"action": "动作4 - 确认区域3", "rule": "rule_C", "occurrence": 1},
            {"action": "动作5 - 确认区域4", "rule": "rule_D", "occurrence": 1},
        ],
    },
}

# ---------------------------------------------------------------------------
# Background detection
# ---------------------------------------------------------------------------
def _process_thread(video_path, detector, model, action_mapping,
                    output_path, state):
    try:
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        detector.reset()
        frame_idx = 0
        _last_pct = -1

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, verbose=False, conf=0.5)
            kp = results[0].keypoints if results[0].keypoints is not None else None
            active, _new = detector.update(kp)

            annotated = draw_pose(frame, results)
            draw_arm_rays(annotated, kp, detector.regions)
            draw_annotations(annotated, detector.regions, detector.lines)
            annotated = draw_status_overlay(
                annotated, detector.rules, active,
                detector.events, action_mapping)
            draw_frame_info(annotated, frame_idx + 1, total, fps)
            out.write(annotated)

            pct = int((frame_idx + 1) / total * 100)
            if pct != _last_pct:
                state.update(progress=pct, frame=frame_idx + 1, total=total,
                             events=detector.events, active=active)
                _last_pct = pct
            frame_idx += 1

        cap.release()
        out.release()

        analyzer = SequenceAnalyzer(detector.events, action_mapping, fps=fps)
        analysis = analyzer.analyze()
        state.update(progress=100, done=True,
                     result=(output_path, list(detector.events),
                             analysis, total, fps))
    except Exception as e:
        import traceback
        state.update(error=f"{e}\n{traceback.format_exc()}", done=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
st.title("司机标准动作检测系统")
st.caption(f"场景: {scenario}")

# --- Preview ---
if "running" not in st.session_state:
    st.session_state.running = False

if not st.session_state.running:
    cfg = SCENARIO_CONFIGS[scenario]
    regions, lines = load_annotations(cfg["annotations_file"])

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("检测规则")
        rule_data = []
        for r in cfg["rules"]:
            labels = {"parallel_line": "平行线", "pass_region": "穿区域",
                      "pointing": "角度指向", "pointing_with_line": "平+指"}
            target = r.get("ref_line") or r.get("target_region") or "-"
            rule_data.append({"规则名": r["name"], "类型": labels.get(r["type"], r["type"]),
                              "目标": target})
        st.dataframe(rule_data, use_container_width=True, hide_index=True)

    with col2:
        st.subheader("动作序列")
        for i, am in enumerate(cfg["action_mapping"], 1):
            st.markdown(f"**{i}.** {am['action']}  `{am['rule']}#{am['occurrence']}`")
        st.divider()
        st.metric("区域数", len(regions))
        st.metric("参考线数", len(lines))

    st.info("调整参数后，点击侧边栏「开始检测」")

    if st.sidebar.button("▶ 开始检测", use_container_width=True, type="primary"):
        # Build detector + thread
        cfg2 = SCENARIO_CONFIGS[scenario]
        regions2, lines2 = load_annotations(cfg2["annotations_file"])

        detector = ParallelDetector(
            cfg2["rules"], regions2, lines2,
            hold_frames=hold_frames, frame_decay=frame_decay,
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

        model = YOLO(model_path)
        output_path = os.path.join(OUTPUT_DIR, "streamlit",
                                   f"output_{scenario}.mp4")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        state = DetectionState()
        thread = threading.Thread(
            target=_process_thread,
            args=(video_path, detector, model, cfg2["action_mapping"],
                  output_path, state),
            daemon=True,
        )

        st.session_state.running = True
        st.session_state.state = state
        st.session_state.thread = thread
        st.session_state.cfg = cfg2
        thread.start()
        st.rerun()

    st.stop()

# --- Running / done ---
state = st.session_state.state
snap = state.snapshot()

if not snap["done"]:
    # Progress display
    st.subheader("🔍 检测进行中...")

    bar_col, num_col = st.columns([4, 1])
    with bar_col:
        st.progress(snap["progress"] / 100.0, text=f"{snap['progress']}%")
    with num_col:
        st.metric("帧", f"{snap['frame']}/{snap['total']}")

    m1, m2 = st.columns(2)
    m1.metric("已检测事件", len(snap["events"]))
    if snap["active"]:
        names = ", ".join(f"{k}({v['hold']}/{v['required']})"
                          for k, v in list(snap["active"].items())[:3])
        m2.metric("活跃规则", names)
    else:
        m2.caption("暂无活跃规则")

    time.sleep(0.4)
    st.rerun()

# --- Done ---
if snap["error"]:
    st.error(f"检测出错: {snap['error']}")
    if st.button("重试"):
        st.session_state.running = False
        st.rerun()
    st.stop()

# Results
st.balloons()
output_path, events, analysis, total_frames, video_fps = snap["result"]

# ---- video player ----
st.subheader("🎬 检测结果")
st.caption("可拖动进度条、调整播放速度")
if os.path.exists(output_path):
    with open(output_path, "rb") as f:
        st.video(f.read())
else:
    st.error("视频生成失败")

# ---- analysis ----
st.divider()
tab1, tab2 = st.tabs(["📊 动作分析", "📋 事件列表"])

with tab1:
    m1, m2, m3 = st.columns(3)
    m1.metric("总帧数", total_frames)
    m2.metric("检测事件", len(events))
    m3.metric("视频FPS", f"{video_fps:.1f}")

    col_a, col_b = st.columns(2)
    with col_a:
        for a in analysis["actions"]:
            if a["found"]:
                side = "左臂" if a.get("side") == "L" else \
                       "右臂" if a.get("side") == "R" else "?"
                ts = a.get("timestamp", 0)
                st.success(f"✅ **{a['action']}**  -  {side}  @ {ts:.1f}s")
            else:
                st.error(f"❌ **{a['action']}**  -  未检测到")

    with col_b:
        if analysis["all_found"] and analysis["order_valid"]:
            st.success("✅ 全部完成，顺序正确")
        elif analysis["all_found"]:
            st.warning("⚠️ 全部完成，但顺序异常")
        else:
            missing = [a["action"] for a in analysis["actions"] if not a["found"]]
            st.error(f"❌ 缺失: {', '.join(missing)}")

        st.subheader("规则触发统计")
        rule_counts = {}
        for e in events:
            rule_counts[e["rule"]] = rule_counts.get(e["rule"], 0) + 1
        for rn, c in rule_counts.items():
            st.metric(f"`{rn}`", c)

with tab2:
    if events:
        st.dataframe(
            [{"帧": e["frame"], "规则": e["rule"],
              "手臂": e["side"], "角度": f"{e['angle']:.1f}°"}
             for e in events],
            use_container_width=True, hide_index=True,
        )

    st.subheader("分析报告")
    analyzer = SequenceAnalyzer(
        events, st.session_state.cfg["action_mapping"], fps=video_fps)
    st.code(analyzer.summary(), language=None)

# Reset
if st.button("🔄 重新检测", use_container_width=True):
    st.session_state.running = False
    st.rerun()
