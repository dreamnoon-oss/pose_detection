"""Generate detection reports as CSV (opens directly in Excel)."""

import csv
import datetime
import os
import io


STANDARD_ACTIONS = [
    "开门后手指呼唤",
    "手动关门",
    "关门后确认夹缝",
    "开车前确认站台指示灯",
    "开车前确认站台道岔",
]

KEYPOINT_LABELS = {
    5: "左肩", 6: "右肩", 7: "左肘", 8: "右肘",
    9: "左手腕", 10: "右手腕", 11: "左髋", 12: "右髋",
}


def generate_report(output_path, *,
                    station_name, script_name,
                    video_path, output_video_path,
                    model_path, imgsz, model_conf,
                    train_summary,
                    action_results,
                    action_mapping,
                    detection_kwargs,
                    rules):
    """Generate a CSV detection report (UTF-8 BOM for Excel compatibility).

    Args:
        output_path: Path to save the .csv file.
        station_name: Human-readable station name (e.g. 上体场).
        script_name: Script filename (e.g. run_shangtichang.py).
        video_path: Input video path.
        output_video_path: Annotated output video path.
        model_path: YOLO model file path.
        imgsz: Model input resolution.
        model_conf: Model confidence threshold.
        train_summary: Dict with keys arrive, depart, duration (all str or None).
        action_results: List of per-action result dicts from SequenceAnalyzer.
        action_mapping: The ACTION_MAPPING list from the run script.
        detection_kwargs: Dict of detection parameters.
        rules: List of detection rule dicts.
    """
    buf = io.StringIO()
    w = csv.writer(buf)

    # ── Title ──
    w.writerow(["司机动作检测报告"])
    w.writerow([])

    # ── 基本信息 ──
    w.writerow(["基本信息"])
    w.writerow(["站点名称", station_name])
    w.writerow(["执行脚本", script_name])
    w.writerow(["检测日期", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
    w.writerow(["视频文件", video_path])
    w.writerow(["输出视频", output_video_path])
    w.writerow([])

    # ── 模型参数 ──
    w.writerow(["模型参数"])
    w.writerow(["模型", os.path.basename(model_path)])
    w.writerow(["输入分辨率", f"{imgsz}×{imgsz}"])
    kp_text = "5-12（" + "/".join(KEYPOINT_LABELS.values()) + "）"
    w.writerow(["检测关键点", kp_text])
    w.writerow(["关键点置信度阈值", str(model_conf)])
    w.writerow([])

    # ── 列车进出站 ──
    w.writerow(["列车进出站"])
    if train_summary:
        w.writerow(["列车到站", train_summary.get("arrive", "—")])
        w.writerow(["列车离站", train_summary.get("depart", "—")])
        w.writerow(["停靠时长", train_summary.get("duration", "—")])
    else:
        w.writerow(["状态", "未启用列车检测"])
    w.writerow([])

    # ── 动作检测结果 ──
    w.writerow(["动作检测结果"])
    w.writerow(["序号", "动作名称", "检测结果", "时间",
                "conf", "hit_rate", "margin", "合格"])

    result_lookup = {}
    for r in action_results:
        key = (r['rule'], r['occurrence'])
        result_lookup[key] = r

    for idx in range(5):
        action_name = STANDARD_ACTIONS[idx]

        if idx < len(action_mapping):
            mapping = action_mapping[idx]
            result = result_lookup.get((mapping['rule'], mapping.get('occurrence', 1)))
            found = result and result.get('found')
        else:
            result = None
            found = None

        row = [str(idx + 1), action_name]

        if found is None and result is None:
            row += ["不需要", "—", "—", "—", "—", "—"]
        elif found:
            ts = f"{result['timestamp']:.1f}s" if result.get('timestamp') else "—"
            row += ["已检测", ts,
                    _fmt(result.get('conf')),
                    _fmt(result.get('hit_rate')),
                    _fmt_deg(result.get('margin')),
                    "合格"]
        else:
            row += ["未检测", "—", "—", "—", "—", "不合格"]

        w.writerow(row)

    w.writerow([])

    # ── 总体评估 ──
    w.writerow(["总体评估"])
    total_expected = len(action_mapping)
    total_found = sum(1 for r in action_results if r.get('found'))
    w.writerow(["期望动作数", str(total_expected)])
    w.writerow(["已检出", f"共{total_expected}个动作，检出{total_found}个"])

    order_ok = True
    prev_frame = 0
    for r in action_results:
        if r.get('found') and r.get('frame', 0) < prev_frame:
            order_ok = False
            break
        if r.get('found'):
            prev_frame = r['frame']
    w.writerow(["顺序合规", "通过" if order_ok else "异常"])
    w.writerow([])

    # ── 说明 ──
    w.writerow(["指标说明"])
    w.writerow(["conf", "检测持续期内肩/远端/肘三点关键点的平均置信度（0~1，越高越可信）"])
    w.writerow(["hit_rate", "命中帧数÷持续期总帧数（0~1，越高越稳定）"])
    w.writerow(["margin", "有效阈值−实际夹角（仅parallel_line规则，正值越大=角度越小=余量越充足）"])
    w.writerow(["", "非parallel_line规则（如pass_region）无conf/hit_rate/margin，以 — 表示"])

    # Write with UTF-8 BOM so Excel recognises Chinese characters
    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        f.write(buf.getvalue())


def _fmt(val):
    if val is None:
        return "—"
    return f"{val:.2f}"


def _fmt_deg(val):
    if val is None:
        return "—"
    return f"{val:.1f}°"
