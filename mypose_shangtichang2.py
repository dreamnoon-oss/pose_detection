from ultralytics import YOLO
import cv2
import os
import numpy as np
import time
import json
import math
from PIL import Image, ImageDraw, ImageFont

# 中文字体（Windows 用黑体）
_FONT_PATH = r"C:/Windows/Fonts/simhei.ttf"
_FONT_CACHE = {}  # size -> ImageFont


def _get_font(size):
    if size not in _FONT_CACHE:
        _FONT_CACHE[size] = ImageFont.truetype(_FONT_PATH, size)
    return _FONT_CACHE[size]


def put_text_cn(img, text, pos, font_size, color):
    """用 PIL 渲染中文（或任意文字）到 OpenCV BGR 图像上，返回图像"""
    b, g, r = int(color[0]), int(color[1]), int(color[2])
    font = _get_font(font_size)
    # 创建 RGBA 临时画布：与 img 同尺寸，全透明
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    canvas = Image.new("RGBA", (img.shape[1], img.shape[0]), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text(pos, text, font=font, fill=(r, g, b, 255))
    # 合成
    pil_img.paste(canvas, (0, 0), canvas)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def put_text_cn_with_bg(img, text, pos, font_size, color, bg_color, padding=4):
    """带背景色的中文文本渲染"""
    b, g, r = int(color[0]), int(color[1]), int(color[2])
    bg_b, bg_g, bg_r = int(bg_color[0]), int(bg_color[1]), int(bg_color[2])
    font = _get_font(font_size)
    bbox = font.getbbox(text)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x, y = pos
    # 背景
    cv2.rectangle(img, (x - padding, y - padding), (x + tw + padding, y + th + padding),
                  bg_color, -1)
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    canvas = Image.new("RGBA", (img.shape[1], img.shape[0]), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text(pos, text, font=font, fill=(r, g, b, 255))
    pil_img.paste(canvas, (0, 0), canvas)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# 加载姿态估计模型
model = YOLO(r"yolo26x-pose.pt")

# 视频路径（改成你自己的视频路径）
video_path = r"\\10.151.2.205\共享文件2\司机行为规范样本采样\短视频\上体场2.mp4"

# ========== 自定义绘制：只显示 5-12 号关键点 ==========
# 关键点定义：
# 5:左肩  6:右肩  7:左肘  8:右肘  9:左手腕  10:右手腕  11:左髋  12:右髋
SHOW_KEYPOINTS = list(range(5, 13))  # [5, 6, 7, 8, 9, 10, 11, 12]

# 骨架连接（只包含5-12号点之间的连线）
SKELETON = [
    (5, 6),    # 左肩 - 右肩
    (5, 7),    # 左肩 - 左肘
    (7, 9),    # 左肘 - 左手腕
    (6, 8),    # 右肩 - 右肘
    (8, 10),   # 右肘 - 右手腕
    (5, 11),   # 左肩 - 左髋
    (6, 12),   # 右肩 - 右髋
    (11, 12),  # 左髋 - 右髋
]

# 关键点颜色（BGR）
KP_COLORS = {
    5:  (255, 0, 0),     # 左肩 - 蓝色
    6:  (0, 0, 255),     # 右肩 - 红色
    7:  (255, 128, 0),   # 左肘 - 浅蓝
    8:  (0, 128, 255),   # 右肘 - 橙色
    9:  (255, 255, 0),   # 左手腕 - 青色
    10: (0, 255, 255),   # 右手腕 - 黄色
    11: (128, 0, 255),   # 左髋 - 紫色
    12: (255, 0, 128),   # 右髋 - 粉红
}
LINE_COLOR = (0, 255, 0)  # 骨架连线颜色 - 绿色
CONF_THRESHOLD = 0.5      # 关键点置信度阈值

# 画定的区域
regions_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "regions_shangtichang2.json")
saved_regions = []  # [{"name": str, "xywh": (x,y,w,h)}, ...]
saved_lines = []    # [{"name": str, "pts": [(x1,y1),(x2,y2)]}, ...]
if os.path.exists(regions_file):
    with open(regions_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        for r in data.get("regions", []):
            r["xywh"] = tuple(r["xywh"])
        saved_regions = data["regions"]
        for ln in data.get("lines", []):
            ln["pts"] = [tuple(p) for p in ln["pts"]]
        saved_lines = data.get("lines", [])


# ========== 动作状态机：依次指向不同区域 ==========

# 动作序列
# type="parallel_line": 手臂方向(肩→腕)与参考线方向大致相同即可
# type="pass_region": 肩→腕连线穿过目标区域即可
ACTION_SEQUENCE = [
    {"name": "动作1", "ref_line": "line_1", "type": "parallel_line"},
    {"name": "动作2", "ref_line": "line_2", "type": "parallel_line", "allow_elbow": True},
    {"name": "动作3", "ref_line": "line_1", "type": "parallel_line"},
    {"name": "动作4", "target_region": "region_1", "type": "pass_region"},
]

LINE_ANGLE_THRESHOLD = 40  # 手臂与参考线的最大夹角（度）
HOLD_FRAMES = 15           # 持续指向多少帧确认动作完成
MIN_ARM_LEN = 30           # 最小手臂像素长度

current_action_idx = 0     # 当前动作序号 (0-based)
hold_counter = 0           # 持续指向帧数
pointing_info = None       # 当前帧的指向状态


def _angle_between(v1, v2):
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 < 0.01 or n2 < 0.01:
        return 180.0
    return math.degrees(math.acos(max(-1.0, min(1.0, dot / (n1 * n2)))))


def _segments_intersect(p1, p2, p3, p4):
    """检查线段 p1-p2 与 p3-p4 是否相交"""
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True

    # 端点恰好落在另一线段上的情况
    if abs(d1) < 0.01:
        if min(p3[0], p4[0]) <= p1[0] <= max(p3[0], p4[0]) and \
           min(p3[1], p4[1]) <= p1[1] <= max(p3[1], p4[1]):
            return True
    if abs(d2) < 0.01:
        if min(p3[0], p4[0]) <= p2[0] <= max(p3[0], p4[0]) and \
           min(p3[1], p4[1]) <= p2[1] <= max(p3[1], p4[1]):
            return True

    return False


def check_arm_parallel_to_line(keypoints_obj, line_pts, allow_elbow=False):
    """检查手臂方向(肩→腕)与参考线方向是否大致相同
    allow_elbow=True 时，手腕检测不到则回退用肩→肘
    返回 (True/False, 手臂侧/None, 角度/None, 远端坐标, 肩坐标)
    """
    if keypoints_obj is None or line_pts is None:
        return False, None, None, None, None

    line_pt1, line_pt2 = line_pts
    line_dir = (line_pt2[0] - line_pt1[0], line_pt2[1] - line_pt1[1])

    for person_idx in range(len(keypoints_obj)):
        kps = keypoints_obj[person_idx]
        xy = kps.xy[0].cpu().numpy()
        conf = kps.conf[0].cpu().numpy()

        for shoulder_id, wrist_id, elbow_id, side in [(5, 9, 7, "L"), (6, 10, 8, "R")]:
            if conf[shoulder_id] > CONF_THRESHOLD:
                shoulder = (float(xy[shoulder_id][0]), float(xy[shoulder_id][1]))

                far_id = None
                if conf[wrist_id] > CONF_THRESHOLD:
                    far_id = wrist_id
                elif allow_elbow and conf[elbow_id] > CONF_THRESHOLD:
                    far_id = elbow_id

                if far_id is None:
                    continue

                far_pt = (float(xy[far_id][0]), float(xy[far_id][1]))
                arm_dir = (far_pt[0] - shoulder[0], far_pt[1] - shoulder[1])
                if math.hypot(*arm_dir) <= MIN_ARM_LEN:
                    continue

                angle = _angle_between(arm_dir, line_dir)
                if angle < LINE_ANGLE_THRESHOLD:
                    return True, side, angle, far_pt, shoulder

    return False, None, None, None, None


def check_arm_passes_region(keypoints_obj, region_xywh):
    """检查肩→腕连线是否穿过目标区域
    返回 (True/False, 手臂侧/None, 角度/None, 腕坐标, 肩坐标)
    """
    if keypoints_obj is None:
        return False, None, None, None, None

    rx, ry, rw, rh = region_xywh
    edges = [
        ((rx, ry), (rx + rw, ry)),
        ((rx, ry + rh), (rx + rw, ry + rh)),
        ((rx, ry), (rx, ry + rh)),
        ((rx + rw, ry), (rx + rw, ry + rh)),
    ]

    for person_idx in range(len(keypoints_obj)):
        kps = keypoints_obj[person_idx]
        xy = kps.xy[0].cpu().numpy()
        conf = kps.conf[0].cpu().numpy()

        for shoulder_id, wrist_id, side in [(5, 9, "L"), (6, 10, "R")]:
            if conf[shoulder_id] > CONF_THRESHOLD and conf[wrist_id] > CONF_THRESHOLD:
                shoulder = (float(xy[shoulder_id][0]), float(xy[shoulder_id][1]))
                wrist = (float(xy[wrist_id][0]), float(xy[wrist_id][1]))
                if math.hypot(wrist[0] - shoulder[0], wrist[1] - shoulder[1]) <= MIN_ARM_LEN:
                    continue

                # 肩或腕在区域内
                if rx <= shoulder[0] <= rx + rw and ry <= shoulder[1] <= ry + rh:
                    return True, side, 0.0, wrist, shoulder
                if rx <= wrist[0] <= rx + rw and ry <= wrist[1] <= ry + rh:
                    return True, side, 0.0, wrist, shoulder

                # 手臂线段与矩形四边相交
                for e1, e2 in edges:
                    if _segments_intersect(shoulder, wrist, e1, e2):
                        return True, side, 0.0, wrist, shoulder

    return False, None, None, None, None


def _get_region_by_name(name):
    for r in saved_regions:
        if r["name"] == name:
            return r["xywh"]
    return None


def _get_line_by_name(name):
    for ln in saved_lines:
        if ln["name"] == name:
            return ln["pts"]
    return None


def update_state_machine(results):
    """每帧调用：根据姿态检测结果推进状态机"""
    global current_action_idx, hold_counter, pointing_info

    if current_action_idx >= len(ACTION_SEQUENCE):
        pointing_info = None
        return

    target = ACTION_SEQUENCE[current_action_idx]
    kp = results[0].keypoints if results[0].keypoints is not None else None

    action_type = target.get("type", "parallel_line")

    if action_type == "parallel_line":
        ref_line_name = target.get("ref_line")
        if ref_line_name is None:
            pointing_info = None
            return
        line_pts = _get_line_by_name(ref_line_name)
        if line_pts is None:
            pointing_info = None
            return
        allow_elbow = target.get("allow_elbow", False)
        is_pointing, side, angle, wrist, shoulder = check_arm_parallel_to_line(kp, line_pts, allow_elbow)
        region_name = ref_line_name
    elif action_type == "pass_region":
        region_name = target.get("target_region")
        if region_name is None:
            pointing_info = None
            return
        region = _get_region_by_name(region_name)
        if region is None:
            pointing_info = None
            return
        is_pointing, side, angle, wrist, shoulder = check_arm_passes_region(kp, region)
    else:
        pointing_info = None
        return

    if is_pointing:
        hold_counter += 1
        pointing_info = {
            "action_name": target["name"],
            "region_name": region_name,
            "hold": hold_counter,
            "required": HOLD_FRAMES,
            "angle": angle,
            "side": side,
            "wrist": wrist,
            "shoulder": shoulder,
        }
        if hold_counter >= HOLD_FRAMES:
            print(f">>> {target['name']} 完成！")
            current_action_idx += 1
            hold_counter = 0
            if current_action_idx >= len(ACTION_SEQUENCE):
                print(f"===== 全部 {len(ACTION_SEQUENCE)} 个动作完成! =====")
    else:
        hold_counter = max(0, hold_counter - 2)  # 缓慢衰减，容忍短暂丢帧
        pointing_info = None


def draw_status_overlay(frame, info):
    """在画面左上角绘制精美的状态机面板（支持中文）"""
    h, w = frame.shape[:2]
    num_actions = len(ACTION_SEQUENCE)

    # 字体大小
    title_font = 18
    row_font = 15
    detail_font = 14

    # 面板尺寸计算
    panel_w = 320
    title_h = 34
    row_h = 28
    detail_h = 52 if info is not None else 0
    panel_h = title_h + num_actions * row_h + detail_h + 10
    panel_x, panel_y = 12, 12

    # 半透明背景面板
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (20, 20, 20), -1)
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (60, 60, 60), 1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    # 标题栏
    title_x = panel_x + 12
    title_y = panel_y + 8
    frame = put_text_cn(frame, "◉ 动作状态机", (title_x, title_y), title_font, (180, 180, 180))
    sep_y = title_y + 24
    cv2.line(frame, (panel_x + 12, sep_y), (panel_x + panel_w - 12, sep_y), (60, 60, 60), 1)

    all_done = current_action_idx >= num_actions

    # 逐行动作
    for i, act in enumerate(ACTION_SEQUENCE):
        row_y = panel_y + title_h + i * row_h + 4
        x0 = panel_x + 14

        if i < current_action_idx:
            icon, icon_color = "✓", (80, 220, 80)
            text_color = (130, 220, 130)
        elif i == current_action_idx and not all_done:
            icon, icon_color = "▶", (0, 210, 255)
            text_color = (255, 255, 255)
        else:
            icon, icon_color = "○", (100, 100, 100)
            text_color = (130, 130, 130)

        target_desc = act.get("target_region") or act.get("ref_line") or "?"
        line_text = f"{act['name']}  →  {target_desc}"

        # 用 cv2 画图标符号（ASCII 安全）
        cv2.putText(frame, icon, (x0, row_y + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, icon_color, 1, cv2.LINE_AA)
        frame = put_text_cn(frame, line_text, (x0 + 22, row_y), row_font, text_color)

    # 详情行（当前动作有指向信息时）
    if info is not None and not all_done:
        detail_y0 = panel_y + title_h + num_actions * row_h
        detail_x0 = panel_x + 14

        side_name = "左臂" if info["side"] == "L" else "右臂" if info["side"] == "R" else "?"
        detail_line = f"角度: {info['angle']:.0f}°  |  {side_name}  |  {info['hold']}/{info['required']} 帧"
        frame = put_text_cn(frame, detail_line, (detail_x0, detail_y0 + 4), detail_font, (200, 200, 200))

        # 进度条
        bar_x, bar_y = detail_x0, detail_y0 + 30
        bar_w, bar_h = panel_w - 28, 6
        progress = min(1.0, info["hold"] / info["required"])
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
        if progress > 0:
            bar_color = (0, 200, 255) if progress < 1.0 else (0, 255, 100)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * progress), bar_y + bar_h),
                          bar_color, -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), 1)

    # 全部完成
    if all_done:
        done_y = panel_y + title_h + num_actions * row_h + 8
        frame = put_text_cn(frame, "全部完成!", (panel_x + 14, done_y), title_font, (0, 255, 100))

    return frame


def draw_pose(frame, results):
    """自定义绘制：只在原始帧上画 5-12 号关键点和骨架"""
    annotated = frame.copy()

    if results[0].keypoints is None:
        return annotated

    keypoints = results[0].keypoints  # 所有人
    boxes = results[0].boxes

    for person_idx in range(len(keypoints)):
        kps = keypoints[person_idx]
        xy = kps.xy[0].cpu().numpy()   # (17, 2)
        conf = kps.conf[0].cpu().numpy()  # (17,)

        # 画骨架连线
        for (i, j) in SKELETON:
            if conf[i] > CONF_THRESHOLD and conf[j] > CONF_THRESHOLD:
                pt1 = (int(xy[i][0]), int(xy[i][1]))
                pt2 = (int(xy[j][0]), int(xy[j][1]))
                cv2.line(annotated, pt1, pt2, LINE_COLOR, 2, cv2.LINE_AA)

        # 画关键点
        for kp_id in SHOW_KEYPOINTS:
            if conf[kp_id] > CONF_THRESHOLD:
                x, y = int(xy[kp_id][0]), int(xy[kp_id][1])
                color = KP_COLORS.get(kp_id, (0, 255, 0))
                cv2.circle(annotated, (x, y), 6, color, -1, cv2.LINE_AA)
                cv2.circle(annotated, (x, y), 6, (255, 255, 255), 1, cv2.LINE_AA)

        # 画检测框
        if boxes is not None and person_idx < len(boxes):
            box = boxes[person_idx].xyxy[0].cpu().numpy().astype(int)
            cv2.rectangle(annotated, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            label = f"person {boxes[person_idx].conf[0].item():.2f}"
            cv2.putText(annotated, label, (box[0], box[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 画保存的区域
    for region in saved_regions:
        x, y, w, h = region["xywh"]
        name = region["name"]
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 255, 0), 2)
        cv2.putText(annotated, name, (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

    # 画参考线
    for ln in saved_lines:
        pt1, pt2 = ln["pts"]
        name = ln["name"]
        cv2.line(annotated, pt1, pt2, (0, 200, 255), 2)
        cv2.arrowedLine(annotated, pt1, pt2, (0, 200, 255), 2, tipLength=0.08)
        cv2.putText(annotated, name, (pt1[0] + 5, pt1[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)

    return annotated


# ========== 实时预览播放器：支持暂停/进度条拖拽 ==========

cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
delay = max(1, int(1000 / fps))

# 确保输出目录存在
save_dir = r"/ultralytics-8.4.75/runs/pose/predict"
os.makedirs(save_dir, exist_ok=True)

# 输出视频保存
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(os.path.join(save_dir, "pose_output.mp4"), fourcc, fps, (width, height))

window_name = "Pose Detection (5-12 keypoints)"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, min(width, 1280), min(height, 720))

# 状态变量
paused = False
last_frame = None
trackbar_pos = [0]
user_seeking = [False]

def on_trackbar(pos):
    user_seeking[0] = True
    trackbar_pos[0] = pos

cv2.createTrackbar("Progress", window_name, 0, max(0, total_frames - 1), on_trackbar)

print(f"视频信息: {width}x{height}, {fps:.1f}fps, 共{total_frames}帧, 时长{total_frames/fps:.1f}秒")
print(f"显示关键点: 5-12 (左肩/右肩/左肘/右肘/左手腕/右手腕/左髋/右髋)")
print("操作说明:")
print("  空格 = 暂停/继续")
print("  Q    = 退出")
print("  拖拽进度条 = 跳转到指定帧")
print("  ----- 暂停时可用 -----")
print("  R    = 鼠标框选区域（指示灯等）")
print("  L    = 鼠标点击两点画参考线（沿列车）")
print("  D    = 删除最后一个区域")
print("  S    = 保存区域+参考线到 regions_baoshan1.json")
print("  ----- 随时可用 -----")
print("  Z    = 重置动作状态机到第1步")

while True:
    # 检测用户是否拖拽了进度条
    if user_seeking[0]:
        user_seeking[0] = False
        cap.set(cv2.CAP_PROP_POS_FRAMES, trackbar_pos[0])
        ret, frame = cap.read()
        if ret:
            results = model(frame, verbose=False, conf=0.5)
            last_frame = draw_pose(frame, results)
            cv2.imshow(window_name, last_frame)
        continue

    if not paused:
        ret, frame = cap.read()
        if not ret:
            paused = True
            print("播放结束，已自动暂停")
            if last_frame is not None:
                cv2.putText(last_frame, "END - PAUSED", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow(window_name, last_frame)
            continue

        # 姿态检测 + 自定义绘制
        t0 = time.time()
        results = model(frame, verbose=False, conf=0.5)
        update_state_machine(results)
        annotated = draw_pose(frame, results)
        annotated = draw_status_overlay(annotated, pointing_info)
        infer_ms = (time.time() - t0) * 1000  # 推理耗时(ms)
        last_frame = annotated.copy()

        # 写入输出视频
        out.write(annotated)

        # 更新进度条
        cur = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        cv2.setTrackbarPos("Progress", window_name, cur)

        # 帧数信息放右上角
        info = f"Frame: {cur}/{total_frames}  ({cur/fps:.1f}s)"
        (tw, th), _ = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.putText(annotated, info, (width - tw - 10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
        cv2.imshow(window_name, annotated)

        # 扣除推理时间，保持正常播放速度
        wait_ms = max(1, delay - int(infer_ms))
        key = cv2.waitKey(wait_ms) & 0xFF
    else:
        pass  # 暂停画面的绘制统一在下面处理

    # 按键处理（非暂停状态已在上方处理）
    if paused:
        key = cv2.waitKey(30) & 0xFF
    else:
        key = key  # 保持上面已获取的 key

    if key == ord('q'):
        break
    elif key == ord(' '):
        paused = not paused
        print("暂停" if paused else "继续")
    elif key == ord('r') and paused and last_frame is not None:
        # 暂停时按 R：用鼠标框选区域（可反复按 R 框选多个区域）
        print(f"请用鼠标框选第 {len(saved_regions) + 1} 个区域，回车确认，Esc 取消...")
        roi = cv2.selectROI(window_name, last_frame, False)
        if roi[2] > 0 and roi[3] > 0:
            name = f"region_{len(saved_regions) + 1}"
            saved_regions.append({"name": name, "xywh": roi})
            print(f">>> {name}: x={roi[0]}, y={roi[1]}, w={roi[2]}, h={roi[3]}")
            print(f"    已选 {len(saved_regions)} 个区域，继续按 R 添加，按 S 保存")
        else:
            print("取消框选")
    elif key == ord('s') and paused:
        # 暂停时按 S：保存区域到文件
        save_data = {
            "video": video_path,
            "frame": int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
            "width": width,
            "height": height,
            "regions": [{"name": r["name"], "xywh": list(r["xywh"])} for r in saved_regions],
            "lines": [{"name": ln["name"], "pts": [list(p) for p in ln["pts"]]} for ln in saved_lines],
        }
        with open(regions_file, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        print(f"✅ 已保存 {len(saved_regions)} 个区域到: {regions_file}")
    elif key == ord('d') and paused and saved_regions:
        # 暂停时按 D：删除最后一个区域
        removed = saved_regions.pop()
        print(f"已删除 {removed['name']}，剩余 {len(saved_regions)} 个区域")
    elif key == ord('l') and paused and last_frame is not None:
        # 暂停时按 L：鼠标点击两个点画参考线
        print("请在画面上点击两个点画参考线（沿列车方向）...")
        line_pts = []

        def on_line_click(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                line_pts.append((x, y))
                tmp = last_frame.copy()
                for pt in line_pts:
                    cv2.circle(tmp, pt, 5, (0, 165, 255), -1)
                if len(line_pts) == 2:
                    cv2.line(tmp, line_pts[0], line_pts[1], (0, 200, 255), 2)
                    cv2.arrowedLine(tmp, line_pts[0], line_pts[1], (0, 200, 255), 2, tipLength=0.08)
                cv2.imshow(window_name, tmp)
                if len(line_pts) >= 2:
                    print(f"  点 {len(line_pts)}: ({x}, {y})")

        cv2.setMouseCallback(window_name, on_line_click)
        while len(line_pts) < 2:
            k2 = cv2.waitKey(30) & 0xFF
            if k2 == 27 or k2 == ord('q'):
                break
        cv2.setMouseCallback(window_name, lambda *args: None)
        if len(line_pts) == 2:
            name = f"line_{len(saved_lines) + 1}"
            saved_lines.append({"name": name, "pts": line_pts})
            print(f">>> {name}: {line_pts[0]} -> {line_pts[1]}")
            print(f"    当前共 {len(saved_lines)} 条参考线，按 S 保存")
        else:
            print("取消画线")
    elif key == ord('z'):
        # 按 Z：重置状态机
        current_action_idx = 0
        hold_counter = 0
        pointing_info = None
        print("状态机已重置到动作1")

    # 更新暂停画面（重新绘制区域+参考线）
    if paused and last_frame is not None:
        paused_frame = last_frame.copy()
        for region in saved_regions:
            x, y, w, h = region["xywh"]
            cv2.rectangle(paused_frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
            cv2.putText(paused_frame, region["name"], (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
        for ln in saved_lines:
            pt1, pt2 = ln["pts"]
            cv2.line(paused_frame, pt1, pt2, (0, 200, 255), 2)
            cv2.arrowedLine(paused_frame, pt1, pt2, (0, 200, 255), 2, tipLength=0.08)
            cv2.putText(paused_frame, ln["name"], (pt1[0] + 5, pt1[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
        cv2.putText(paused_frame, "PAUSED", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        cv2.imshow(window_name, paused_frame)

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"结果已保存到: {save_dir}\\pose_output.mp4")
