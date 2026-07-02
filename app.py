"""Streamlit dashboard — zero-flicker detection with JS progress polling."""

import sys
import os
import json
import time
import tempfile
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
import streamlit.components.v1 as components
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

# Progress file location (served by Streamlit at /app/static/)
STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
PROGRESS_FILE = STATIC_DIR / "progress.json"

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
    st.sidebar.error("模型文件不存在")
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
# Background detection thread
# ---------------------------------------------------------------------------
def _process_thread(video_path, detector, model, action_mapping,
                    output_path):
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
        _last_write = 0
        _last_pct = -1

        def _write_progress(pct, done):
            try:
                with open(PROGRESS_FILE, 'w') as f:
                    json.dump({
                        'progress': pct,
                        'frame': frame_idx + 1,
                        'total': total,
                        'events': len(detector.events),
                        'active': [f"{k}({v['hold']}/{v['required']})"
                                   for k, v in list(detector.events[:0])],
                        'active_rules': [
                            {'name': k, 'hold': v['hold'], 'required': v['required']}
                            for k, v in list(detector.events[:0])],
                        'done': done,
                    }, f)
            except Exception:
                pass

        # Write initial progress
        _write_progress(0, False)

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

            # Write progress to JSON every ~8 frames (reduces I/O)
            if frame_idx - _last_write >= 8 or pct != _last_pct or pct == 100:
                active_info = []
                for name, hit in active.items():
                    active_info.append({
                        'name': name, 'hold': hit['hold'],
                        'required': hit['required'],
                    })
                with open(PROGRESS_FILE, 'w') as f:
                    json.dump({
                        'progress': pct,
                        'frame': frame_idx + 1,
                        'total': total,
                        'events': len(detector.events),
                        'active': active_info,
                        'done': False,
                    }, f)
                _last_write = frame_idx

            if pct != _last_pct:
                _last_pct = pct

            frame_idx += 1

        cap.release()
        out.release()

        analyzer = SequenceAnalyzer(detector.events, action_mapping, fps=fps)
        analysis = analyzer.analyze()

        # Write "done" state — JS will see this and notify Streamlit
        with open(PROGRESS_FILE, 'w') as f:
            json.dump({
                'progress': 100,
                'frame': total,
                'total': total,
                'events': len(detector.events),
                'active': [],
                'done': True,
            }, f)

        # Store result for the next Streamlit rerun
        st.session_state._result = {
            'output_path': output_path,
            'events': list(detector.events),
            'analysis': analysis,
            'total': total,
            'fps': fps,
            'cfg': action_mapping,
        }

    except Exception as e:
        import traceback
        st.session_state._error = f"{e}\n{traceback.format_exc()}"
        with open(PROGRESS_FILE, 'w') as f:
            json.dump({'progress': 0, 'done': True, 'error': str(e)}, f)


# ---------------------------------------------------------------------------
# Progress bar HTML/JS component
# ---------------------------------------------------------------------------
PROGRESS_HTML = """
<div id="root" style="padding: 24px; font-family: -apple-system, sans-serif;
     background: #0e1117; color: #e0e0e0; border-radius: 10px; min-height: 200px;">
  <h3 style="margin: 0 0 20px;">🔍 检测进行中...</h3>

  <div style="background: #1e2130; border-radius: 8px; height: 24px; overflow: hidden;
              margin-bottom: 16px;">
    <div id="bar" style="width: 0%; height: 100%; background: linear-gradient(90deg,
         #ff4b4b, #ff8c00, #ffd700, #4caf50); border-radius: 8px;
         transition: width 0.3s ease;"></div>
  </div>

  <div style="display: flex; gap: 40px; font-size: 15px;">
    <div><span style="color: #888;">进度</span>
         <b id="pct" style="margin-left: 8px;">0%</b></div>
    <div><span style="color: #888;">帧</span>
         <b id="frame" style="margin-left: 8px;">0/0</b></div>
    <div><span style="color: #888;">事件</span>
         <b id="events" style="margin-left: 8px;">0</b></div>
  </div>

  <div id="active-section" style="margin-top: 14px; font-size: 13px; color: #aaa;"></div>
</div>

<script>
(function() {
    var lastDone = false;

    function poll() {
        fetch('/app/static/progress.json?t=' + Date.now())
            .then(function(r) { return r.json(); })
            .then(function(d) {
                document.getElementById('bar').style.width = d.progress + '%';
                document.getElementById('pct').textContent = d.progress + '%';
                document.getElementById('frame').textContent =
                    d.frame + '/' + d.total;
                document.getElementById('events').textContent = d.events;

                var act = document.getElementById('active-section');
                if (d.active && d.active.length > 0) {
                    var names = d.active.map(function(a) {
                        return a.name + '(' + a.hold + '/' + a.required + ')';
                    }).join(', ');
                    act.textContent = '进行中: ' + names;
                } else if (!d.done) {
                    act.textContent = '';
                }

                if (d.done && !lastDone) {
                    lastDone = true;
                    if (d.error) {
                        document.getElementById('root').innerHTML =
                            '<h3 style="color:#ff4b4b;">❌ 检测失败</h3>' +
                            '<pre>' + (d.error || '') + '</pre>';
                    } else {
                        document.getElementById('root').innerHTML =
                            '<h3 style="color:#4caf50;">✅ 检测完成!</h3>' +
                            '<p>正在加载结果...</p>';
                    }
                    // Notify Streamlit — retry until the API is ready
                    function sendDone() {
                        var s = window.Streamlit;
                        if (s && s.setComponentValue) {
                            s.setComponentValue({done: true});
                        } else {
                            setTimeout(sendDone, 200);
                        }
                    }
                    sendDone();
                }
            })
            .catch(function() {});
    }

    poll();
    setInterval(poll, 500);
})();
</script>
"""

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
st.title("司机标准动作检测系统")
st.caption(f"场景: {scenario}")

# Check for completion from HTML component
component_value = None

# --- State: not started ---
if "phase" not in st.session_state:
    st.session_state.phase = "preview"

if st.session_state.phase == "preview":
    cfg = SCENARIO_CONFIGS[scenario]
    regions, lines = load_annotations(cfg["annotations_file"])

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("检测规则")
        labels = {"parallel_line": "平行线", "pass_region": "穿区域",
                  "pointing": "角度指向", "pointing_with_line": "平+指"}
        rule_data = []
        for r in cfg["rules"]:
            target = r.get("ref_line") or r.get("target_region") or "-"
            rule_data.append({"规则名": r["name"],
                              "类型": labels.get(r["type"], r["type"]),
                              "目标": target})
        st.dataframe(rule_data, use_container_width=True, hide_index=True)

    with col2:
        st.subheader("动作序列")
        for i, am in enumerate(cfg["action_mapping"], 1):
            st.markdown(f"**{i}.** {am['action']}  `{am['rule']}#{am['occurrence']}`")
        st.divider()
        st.metric("区域数", len(regions))
        st.metric("参考线数", len(lines))

    if st.sidebar.button("▶ 开始检测", use_container_width=True, type="primary"):
        # Prepare
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

        thread = threading.Thread(
            target=_process_thread,
            args=(video_path, detector, model, cfg2["action_mapping"],
                  output_path),
            daemon=True,
        )

        st.session_state.phase = "running"
        st.session_state.thread = thread
        thread.start()
        st.rerun()

    st.stop()

# --- State: running ---
if st.session_state.phase == "running":
    # Show the JS progress component — self-updating, no Streamlit reruns needed
    component_value = components.html(PROGRESS_HTML, height=220)

    # component_value is a dict only after JS calls setComponentValue;
    # on first render it's None or a DeltaGenerator — must guard.
    done_signal = (
        isinstance(component_value, dict) and component_value.get("done")
    )
    if done_signal or st.session_state.get("_result") or st.session_state.get("_error"):
        st.session_state.phase = "results"
        st.rerun()

    # Safety fallback: check progress.json from Python side every 4s
    # (very infrequent, negligible visual impact)
    try:
        with open(PROGRESS_FILE, 'r') as f:
            if json.load(f).get('done'):
                if st.session_state.get("_result") or st.session_state.get("_error"):
                    st.session_state.phase = "results"
                    st.rerun()
    except Exception:
        pass

    time.sleep(4)
    st.rerun()

# --- State: results ---
if st.session_state.phase == "results":
    if st.session_state.get("_error"):
        st.error(f"检测出错: {st.session_state._error}")
        if st.button("重试"):
            for k in ["phase", "thread", "_result", "_error"]:
                st.session_state.pop(k, None)
            st.rerun()
        st.stop()

    result = st.session_state.get("_result")
    if not result:
        st.warning("等待结果...")
        time.sleep(1)
        st.rerun()

    st.balloons()

    output_path = result['output_path']
    events = result['events']
    analysis = result['analysis']
    total_frames = result['total']
    video_fps = result['fps']
    action_mapping = result['cfg']

    st.subheader("🎬 检测结果")
    st.caption("可拖动进度条、调整播放速度")
    if os.path.exists(output_path):
        with open(output_path, "rb") as f:
            st.video(f.read())
    else:
        st.error("视频生成失败")

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
                missing = [a["action"] for a in analysis["actions"]
                           if not a["found"]]
                st.error(f"❌ 缺失: {', '.join(missing)}")

            rule_counts = {}
            for e in events:
                rule_counts[e["rule"]] = rule_counts.get(e["rule"], 0) + 1
            if rule_counts:
                st.subheader("规则触发")
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
        analyzer = SequenceAnalyzer(events, action_mapping, fps=video_fps)
        st.code(analyzer.summary(), language=None)

    if st.button("🔄 重新检测", use_container_width=True):
        for k in ["phase", "thread", "_result", "_error"]:
            st.session_state.pop(k, None)
        if PROGRESS_FILE.exists():
            PROGRESS_FILE.unlink()
        st.rerun()
