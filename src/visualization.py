"""Drawing and rendering utilities for pose detection visualization."""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import (
    SHOW_KEYPOINTS, SKELETON, KP_COLORS, LINE_COLOR, CONF_THRESHOLD,
)

# ---------------------------------------------------------------------------
# Chinese text rendering (PIL-based)
# ---------------------------------------------------------------------------

_FONT_PATH = r"C:/Windows/Fonts/simhei.ttf"
_FONT_CACHE = {}


def _get_font(size):
    if size not in _FONT_CACHE:
        _FONT_CACHE[size] = ImageFont.truetype(_FONT_PATH, size)
    return _FONT_CACHE[size]


def put_text_cn(img, text, pos, font_size, color):
    """Render Chinese (or any) text onto an OpenCV BGR image using PIL."""
    b, g, r = int(color[0]), int(color[1]), int(color[2])
    font = _get_font(font_size)
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    canvas = Image.new("RGBA", (img.shape[1], img.shape[0]), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text(pos, text, font=font, fill=(r, g, b, 255))
    pil_img.paste(canvas, (0, 0), canvas)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def put_text_cn_with_bg(img, text, pos, font_size, color, bg_color, padding=4):
    """Render Chinese text with a filled background rectangle."""
    font = _get_font(font_size)
    bbox = font.getbbox(text)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x, y = pos
    cv2.rectangle(img, (x - padding, y - padding),
                  (x + tw + padding, y + th + padding), bg_color, -1)
    return put_text_cn(img, text, pos, font_size, color)


# ---------------------------------------------------------------------------
# Pose skeleton drawing
# ---------------------------------------------------------------------------

def draw_pose(frame, results):
    """Draw keypoints 5-12 and skeleton connections on a copy of *frame*."""
    annotated = frame.copy()

    if results[0].keypoints is None:
        return annotated

    keypoints = results[0].keypoints
    boxes = results[0].boxes

    for person_idx in range(len(keypoints)):
        kps = keypoints[person_idx]
        xy = kps.xy[0].cpu().numpy()
        conf = kps.conf[0].cpu().numpy()

        # Skeleton lines
        for i, j in SKELETON:
            if conf[i] > CONF_THRESHOLD and conf[j] > CONF_THRESHOLD:
                pt1 = (int(xy[i][0]), int(xy[i][1]))
                pt2 = (int(xy[j][0]), int(xy[j][1]))
                cv2.line(annotated, pt1, pt2, LINE_COLOR, 2, cv2.LINE_AA)

        # Keypoint circles
        for kp_id in SHOW_KEYPOINTS:
            if conf[kp_id] > CONF_THRESHOLD:
                x, y = int(xy[kp_id][0]), int(xy[kp_id][1])
                color = KP_COLORS.get(kp_id, (0, 255, 0))
                cv2.circle(annotated, (x, y), 6, color, -1, cv2.LINE_AA)
                cv2.circle(annotated, (x, y), 6, (255, 255, 255), 1, cv2.LINE_AA)

        # Bounding box
        if boxes is not None and person_idx < len(boxes):
            box = boxes[person_idx].xyxy[0].cpu().numpy().astype(int)
            cv2.rectangle(annotated, (box[0], box[1]), (box[2], box[3]),
                          (0, 255, 0), 2)
            label = f"person {boxes[person_idx].conf[0].item():.2f}"
            cv2.putText(annotated, label, (box[0], box[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return annotated


# ---------------------------------------------------------------------------
# Annotation overlay (regions + reference lines)
# ---------------------------------------------------------------------------

def draw_annotations(frame, regions, lines):
    """Draw saved rectangular regions and reference lines on *frame* (in-place)."""
    for region in regions:
        x, y, w, h = region["xywh"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
        cv2.putText(frame, region["name"], (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

    for ln in lines:
        pt1, pt2 = ln["pts"]
        cv2.line(frame, pt1, pt2, (0, 200, 255), 2)
        cv2.arrowedLine(frame, pt1, pt2, (0, 200, 255), 2, tipLength=0.08)
        cv2.putText(frame, ln["name"], (pt1[0] + 5, pt1[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)


# ---------------------------------------------------------------------------
# Status panel overlay (top-left)
# ---------------------------------------------------------------------------

def draw_status_overlay(frame, action_sequence, current_action_idx, pointing_info):
    """Draw an elegant status panel (top-left) with Chinese text support.

    Args:
        frame: BGR image (modified in-place).
        action_sequence: list of action dicts.
        current_action_idx: index into *action_sequence*.
        pointing_info: dict from ActionStateMachine.pointing_info, or None.
    """
    h = frame.shape[0]
    num_actions = len(action_sequence)
    all_done = current_action_idx >= num_actions

    # Panel sizing
    title_font = 18
    row_font = 15
    detail_font = 14
    panel_w = 320
    title_h = 34
    row_h = 28
    detail_h = 52 if (pointing_info is not None and not all_done) else 0
    panel_h = title_h + num_actions * row_h + detail_h + 10
    panel_x, panel_y = 12, 12

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y),
                  (panel_x + panel_w, panel_y + panel_h), (20, 20, 20), -1)
    cv2.rectangle(overlay, (panel_x, panel_y),
                  (panel_x + panel_w, panel_y + panel_h), (60, 60, 60), 1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    # Title
    title_x = panel_x + 12
    title_y = panel_y + 8
    frame_before = frame
    frame = put_text_cn(frame, "◉ 动作状态机", (title_x, title_y),
                         title_font, (180, 180, 180))
    sep_y = title_y + 24
    cv2.line(frame, (panel_x + 12, sep_y),
             (panel_x + panel_w - 12, sep_y), (60, 60, 60), 1)

    # Action rows
    for i, act in enumerate(action_sequence):
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

        cv2.putText(frame, icon, (x0, row_y + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, icon_color, 1, cv2.LINE_AA)
        frame = put_text_cn(frame, line_text, (x0 + 22, row_y), row_font, text_color)

    # Detail row (when a hit is active)
    if pointing_info is not None and not all_done:
        detail_y0 = panel_y + title_h + num_actions * row_h
        detail_x0 = panel_x + 14

        side_name = "左臂" if pointing_info["side"] == "L" else \
                    "右臂" if pointing_info["side"] == "R" else "?"
        detail_line = (f"角度: {pointing_info['angle']:.0f}°  |  {side_name}"
                       f"  |  {pointing_info['hold']}/{pointing_info['required']} 帧")
        frame = put_text_cn(frame, detail_line, (detail_x0, detail_y0 + 4),
                            detail_font, (200, 200, 200))

        # Progress bar
        bar_x, bar_y = detail_x0, detail_y0 + 30
        bar_w, bar_h = panel_w - 28, 6
        progress = min(1.0, pointing_info["hold"] / pointing_info["required"])
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
        if progress > 0:
            bar_color = (0, 200, 255) if progress < 1.0 else (0, 255, 100)
            cv2.rectangle(frame, (bar_x, bar_y),
                          (bar_x + int(bar_w * progress), bar_y + bar_h),
                          bar_color, -1)
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), 1)

    # All-done message
    if all_done:
        done_y = panel_y + title_h + num_actions * row_h + 8
        frame = put_text_cn(frame, "全部完成!",
                            (panel_x + 14, done_y), title_font, (0, 255, 100))

    return frame


# ---------------------------------------------------------------------------
# Frame info overlay (top-right)
# ---------------------------------------------------------------------------

def draw_frame_info(frame, cur_frame, total_frames, fps):
    """Draw frame counter and timestamp at the top-right corner."""
    text = f"Frame: {cur_frame}/{total_frames}  ({cur_frame / fps:.1f}s)"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.putText(frame, text, (frame.shape[1] - tw - 10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)


def draw_pause_indicator(frame):
    """Overlay a red PAUSED label at the top-left."""
    cv2.putText(frame, "PAUSED", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
