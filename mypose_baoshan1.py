from ultralytics import YOLO
import cv2
import os
import numpy as np
import time
import json
import math

# 加载姿态估计模型
model = YOLO(r"yolo26x-pose.pt")

# 视频路径（改成你自己的视频路径）
video_path = r"\\10.151.2.205\共享文件2\司机行为规范样本采样\短视频\宝山1.mp4"

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
regions_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "regions_baoshan1.json")
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

# 动作序列：每个动作需要指向哪个区域
# ref_line 不为空时，需要同时满足"手臂沿参考线"+"大致朝向区域"双重判断
ACTION_SEQUENCE = [
    {"name": "动作1", "target_region": "region_1", "ref_line": "line_1"},
    {"name": "动作2", "target_region": "region_2", "ref_line": None},
    {"name": "动作3", "target_region": "region_1", "ref_line": "line_1"},
    {"name": "动作4", "target_region": "region_3", "ref_line": None},
    {"name": "动作5", "target_region": "region_4", "ref_line": "line_1"},
]

ANGLE_THRESHOLD = 30       # 无参考线时的角度阈值（度）
LOOSE_ANGLE_THRESHOLD = 55 # 有参考线时的宽松角度阈值（度）
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


def _min_angle_to_rect(wrist, arm_dir, rect):
    """手臂方向与矩形区域(四角+中心)的最小夹角"""
    rx, ry, rw, rh = rect
    corners = [
        (rx, ry), (rx + rw, ry), (rx, ry + rh), (rx + rw, ry + rh),
        (rx + rw / 2, ry + rh / 2),
    ]
    min_angle = 180.0
    for cx, cy in corners:
        to_corner = (cx - wrist[0], cy - wrist[1])
        angle = _angle_between(arm_dir, to_corner)
        if angle < min_angle:
            min_angle = angle
    return min_angle


def check_pointing(keypoints_obj, region_xywh):
    """角度法：手臂方向与区域(四角+中心)的最小夹角 < 阈值即命中
    返回 (True/False, 手臂侧/None, 角度/None, 腕坐标, 肩坐标)
    """
    if keypoints_obj is None:
        return False, None, None, None, None

    for person_idx in range(len(keypoints_obj)):
        kps = keypoints_obj[person_idx]
        xy = kps.xy[0].cpu().numpy()
        conf = kps.conf[0].cpu().numpy()

        for shoulder_id, wrist_id, side in [(5, 9, "L"), (6, 10, "R")]:
            if conf[shoulder_id] > CONF_THRESHOLD and conf[wrist_id] > CONF_THRESHOLD:
                shoulder = (float(xy[shoulder_id][0]), float(xy[shoulder_id][1]))
                wrist = (float(xy[wrist_id][0]), float(xy[wrist_id][1]))
                arm_dir = (wrist[0] - shoulder[0], wrist[1] - shoulder[1])
                if math.hypot(*arm_dir) <= MIN_ARM_LEN:
                    continue

                min_angle = _min_angle_to_rect(wrist, arm_dir, region_xywh)
                if min_angle < ANGLE_THRESHOLD:
                    return True, side, min_angle, wrist, shoulder

    return False, None, None, None, None


def check_pointing_with_line(keypoints_obj, region_xywh, line_pts):
    """双重判断：手臂平行于参考线 + 手臂角度覆盖区域框
    返回同 check_pointing 格式
    """
    if keypoints_obj is None or line_pts is None:
        return False, None, None, None, None

    line_pt1, line_pt2 = line_pts
    line_dir = (line_pt2[0] - line_pt1[0], line_pt2[1] - line_pt1[1])

    for person_idx in range(len(keypoints_obj)):
        kps = keypoints_obj[person_idx]
        xy = kps.xy[0].cpu().numpy()
        conf = kps.conf[0].cpu().numpy()

        for shoulder_id, wrist_id, side in [(5, 9, "L"), (6, 10, "R")]:
            if conf[shoulder_id] > CONF_THRESHOLD and conf[wrist_id] > CONF_THRESHOLD:
                shoulder = (float(xy[shoulder_id][0]), float(xy[shoulder_id][1]))
                wrist = (float(xy[wrist_id][0]), float(xy[wrist_id][1]))
                arm_dir = (wrist[0] - shoulder[0], wrist[1] - shoulder[1])
                if math.hypot(*arm_dir) <= MIN_ARM_LEN:
                    continue

                # 条件1：手臂与参考线大致平行
                angle_with_line = _angle_between(arm_dir, line_dir)
                if angle_with_line > LINE_ANGLE_THRESHOLD:
                    continue

                # 条件2：手臂方向覆盖区域框内任意位置
                min_angle = _min_angle_to_rect(wrist, arm_dir, region_xywh)
                if min_angle < LOOSE_ANGLE_THRESHOLD:
                    return True, side, min_angle, wrist, shoulder

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
    region = _get_region_by_name(target["target_region"])
    if region is None:
        pointing_info = None
        return

    kp = results[0].keypoints if results[0].keypoints is not None else None

    # 有参考线就用双重判断，否则用纯角度判断
    ref_line_name = target.get("ref_line")
    if ref_line_name:
        line_pts = _get_line_by_name(ref_line_name)
        is_pointing, side, angle, wrist, shoulder = check_pointing_with_line(kp, region, line_pts)
    else:
        is_pointing, side, angle, wrist, shoulder = check_pointing(kp, region)

    if is_pointing:
        hold_counter += 1
        pointing_info = {
            "action_name": target["name"],
            "region_name": target["target_region"],
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
                print("===== 全部 5 个动作完成! =====")
    else:
        hold_counter = max(0, hold_counter - 2)  # 缓慢衰减，容忍短暂丢帧
        pointing_info = None


def draw_status_overlay(frame, info):
    """在画面左上角绘制状态机信息"""
    h = frame.shape[0]

    # 半透明背景
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (420, 115), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    if current_action_idx >= len(ACTION_SEQUENCE):
        cv2.putText(frame, "ALL ACTIONS COMPLETE!", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    elif info is not None:
        cv2.putText(frame, f"{info['action_name']} / 5", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)
        cv2.putText(frame, f"Target: {info['region_name']}  |  Angle: {info['angle']:.0f}deg  |  Arm: {info['side']}",
                    (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

        # 进度条
        bar_x, bar_y, bar_w, bar_h = 10, 65, 260, 14
        progress = min(1.0, info["hold"] / info["required"])
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * progress), bar_y + bar_h),
                      (0, 220, 0), -1)
        cv2.putText(frame, f"{info['hold']} / {info['required']}",
                    (bar_x + 8, bar_y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    else:
        act = ACTION_SEQUENCE[current_action_idx]
        cv2.putText(frame, f"WAITING: {act['name']} -> {act['target_region']}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # 底部：已完成动作
    if current_action_idx > 0:
        done = ", ".join([a["name"] for a in ACTION_SEQUENCE[:current_action_idx]])
        cv2.putText(frame, f"Completed: {done}", (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)


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
        draw_status_overlay(annotated, pointing_info)
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
