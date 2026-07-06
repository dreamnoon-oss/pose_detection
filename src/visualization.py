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
# Detection dashboard overlay (top-left)
# ---------------------------------------------------------------------------

# Compact labels for rule types
_RULE_TYPE_LABEL = {
    "parallel_line": "平行",
    "pass_region": "穿越",
    "pointing": "指向",
    "pointing_with_line": "平+指",
}


def _map_actions_from_events(action_mapping, events):
    """Compute per-action found/missing status from events in chronological order.

    Returns a list of dicts compatible with the old status-overlay format.
    """
    rule_events = {}
    for e in events:
        rule_events.setdefault(e['rule'], []).append(e)

    results = []
    for mapping in action_mapping:
        rule_name = mapping['rule']
        occurrence = mapping.get('occurrence', 1)
        candidates = rule_events.get(rule_name, [])
        if len(candidates) >= occurrence:
            ev = candidates[occurrence - 1]
            results.append({
                'action': mapping['action'],
                'found': True,
                'frame': ev['frame'],
                'side': ev['side'],
                'angle': ev['angle'],
            })
        else:
            results.append({'action': mapping['action'], 'found': False})
    return results


def draw_status_overlay(frame, rules, active, events, action_mapping=None, *,
                         align_right=False):
    """Draw a parallel-detection dashboard.

    Shows every rule's hold progress, completed event count, and (optionally)
    action-mapping status with found/missing indicators.

    Args:
        frame: BGR image (modified in-place).
        rules: list of rule dicts (name, type, ref_line / target_region).
        active: dict of rule_name → hit-info for currently-accumulating rules.
        events: list of all completed event dicts so far.
        action_mapping: optional list of ``{action, rule, occurrence}``.
        align_right: if True, place panel at top-right instead of top-left.
    """
    action_mapping = action_mapping or []
    event_counts = {}
    for e in events:
        event_counts[e['rule']] = event_counts.get(e['rule'], 0) + 1

    action_status = _map_actions_from_events(action_mapping, events)

    # Layout constants
    title_font, rule_font, action_font, tiny_font = 18, 14, 13, 12
    panel_w = 360
    title_h = 34
    rule_row_h = 36
    action_row_h = 22
    sep_h = 4
    pad = 12

    n_rules = len(rules)
    n_actions = len(action_status)
    action_section_h = (sep_h + n_actions * action_row_h + 6) if n_actions else 0
    panel_h = title_h + n_rules * rule_row_h + action_section_h + 10
    panel_x = (frame.shape[1] - panel_w - 12) if align_right else 12
    panel_y = 12

    # Background
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y),
                  (panel_x + panel_w, panel_y + panel_h), (20, 20, 20), -1)
    cv2.rectangle(overlay, (panel_x, panel_y),
                  (panel_x + panel_w, panel_y + panel_h), (60, 60, 60), 1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    # Title
    x0 = panel_x + pad
    frame = put_text_cn(frame, "并行检测面板", (x0, panel_y + 8),
                         title_font, (180, 180, 180))
    sep_y = panel_y + title_h - 4
    cv2.line(frame, (x0, sep_y), (panel_x + panel_w - pad, sep_y),
             (60, 60, 60), 1)

    # Rule rows
    for i, rule in enumerate(rules):
        name = rule['name']
        row_y = panel_y + title_h + i * rule_row_h

        # Type icon + target description
        rtype = rule.get('type', 'parallel_line')
        icon = _RULE_TYPE_LABEL.get(rtype, '?')
        target_desc = rule.get('ref_line') or rule.get('target_region') or '?'
        header = f"{icon} {target_desc}"
        frame = put_text_cn(frame, header, (x0, row_y + 1), rule_font, (220, 220, 220))

        # Right side: angle + side + hold counter
        if name in active:
            hit = active[name]
            side_label = "L" if hit['side'] == 'L' else \
                         "R" if hit['side'] == 'R' else "?"
            info = f"{hit['angle']:.0f}deg {side_label}  {hit['hold']}/{hit['required']}"
            (tw, _), _ = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.putText(frame, info, (panel_x + panel_w - tw - pad, row_y + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

            # Mini progress bar
            bar_x, bar_y = x0, row_y + 17
            bar_w, bar_h = panel_w - pad * 2, 4
            progress = min(1.0, hit['hold'] / hit['required'])
            cv2.rectangle(frame, (bar_x, bar_y),
                          (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
            bar_color = (0, 200, 255) if progress < 1.0 else (0, 255, 100)
            cv2.rectangle(frame, (bar_x, bar_y),
                          (bar_x + int(bar_w * progress), bar_y + bar_h),
                          bar_color, -1)
        else:
            cv2.putText(frame, "--", (panel_x + panel_w - pad - 20, row_y + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

        # Event count
        count = event_counts.get(name, 0)
        frame = put_text_cn(frame, f"已触发: {count}次",
                            (x0 + 4, row_y + rule_row_h - 8),
                            tiny_font, (140, 140, 140))

    # Action mapping section
    if action_status:
        act_y0 = panel_y + title_h + n_rules * rule_row_h + sep_h
        cv2.line(frame, (x0, act_y0), (panel_x + panel_w - pad, act_y0),
                 (60, 60, 60), 1)

        for i, a in enumerate(action_status):
            row_y = act_y0 + 6 + i * action_row_h
            if a['found']:
                icon, color = "V", (80, 220, 80)
                side_label = "左" if a.get('side') == 'L' else \
                             "右" if a.get('side') == 'R' else ""
                text = f"{a['action']}  @帧{a['frame']}  {side_label}臂"
            else:
                icon, color = "O", (100, 100, 100)
                text = a['action']

            cv2.putText(frame, icon, (x0, row_y + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            frame = put_text_cn(frame, text, (x0 + 18, row_y + 1),
                                action_font, color)

    return frame, panel_y + panel_h


def draw_analysis_result(frame, analysis):
    """Overlay the final analysis result on the end frame (bottom-left)."""
    if analysis is None:
        return frame

    lines = []
    for a in analysis['actions']:
        if a['found']:
            lines.append(f"  [OK] {a['action']}  @ {a['timestamp']:.1f}s")
        else:
            lines.append(f"  [X] {a['action']}  未检测到")

    if analysis['all_found']:
        conclusion = "[OK] 全部完成，顺序正确" if analysis['order_valid'] else \
                     "[X] 全部完成，但顺序异常"
    else:
        missing = [a['action'] for a in analysis['actions'] if not a['found']]
        conclusion = f"[X] 缺失: {', '.join(missing)}"
    color = (0, 255, 100) if (analysis['all_found'] and analysis['order_valid']) else (0, 100, 255)

    font_size = 16
    line_h = 22
    box_h = len(lines) * line_h + 38
    box_w = 340
    box_x, box_y = 12, frame.shape[0] - box_h - 20

    overlay = frame.copy()
    cv2.rectangle(overlay, (box_x, box_y),
                  (box_x + box_w, box_y + box_h), (20, 20, 20), -1)
    cv2.rectangle(overlay, (box_x, box_y),
                  (box_x + box_w, box_y + box_h), (60, 60, 60), 1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

    frame = put_text_cn(frame, "分析结果", (box_x + 12, box_y + 8),
                         font_size, (200, 200, 200))
    for i, line in enumerate(lines):
        y = box_y + 30 + i * line_h
        frame = put_text_cn(frame, line, (box_x + 14, y), 13, (220, 220, 220))

    # Right-align the conclusion text
    font = _get_font(14)
    bbox = font.getbbox(conclusion)
    tw = bbox[2] - bbox[0]
    frame = put_text_cn(frame, conclusion,
                        (box_x + box_w - tw - 14, box_y + box_h - 10),
                        14, color)

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


# ---------------------------------------------------------------------------
# Debug: draw extended arm rays for pass_region rules
# ---------------------------------------------------------------------------

def draw_arm_rays(frame, keypoints_obj, regions):
    """Draw shoulder→elbow→wrist segments and shoulder→extended-wrist rays.

    Arm segments (shoulder→elbow→wrist) are drawn in thick cyan/magenta.
    Extended rays are green if they hit a region, red otherwise.

    Args:
        frame: BGR image (modified in-place).
        keypoints_obj: ultralytics ``Keypoints`` or None.
        regions: list of region dicts ``{name, xywh}``.
    """
    import math
    if keypoints_obj is None:
        return

    for person_idx in range(len(keypoints_obj)):
        kps = keypoints_obj[person_idx]
        xy = kps.xy[0].cpu().numpy()
        conf = kps.conf[0].cpu().numpy()

        for shoulder_id, wrist_id, elbow_id, side in [(5, 9, 7, "L"), (6, 10, 8, "R")]:
            side_color = (255, 255, 0) if side == "L" else (255, 0, 255)  # cyan L, magenta R

            # Draw shoulder→elbow segment (thick, always when both visible)
            if conf[shoulder_id] > 0.3 and conf[elbow_id] > 0.3:
                s_pt = (int(xy[shoulder_id][0]), int(xy[shoulder_id][1]))
                e_pt = (int(xy[elbow_id][0]), int(xy[elbow_id][1]))
                cv2.line(frame, s_pt, e_pt, side_color, 3, cv2.LINE_AA)
                cv2.circle(frame, e_pt, 5, side_color, -1, cv2.LINE_AA)

            # Draw elbow→wrist segment
            if conf[elbow_id] > 0.3 and conf[wrist_id] > 0.3:
                e_pt = (int(xy[elbow_id][0]), int(xy[elbow_id][1]))
                w_pt = (int(xy[wrist_id][0]), int(xy[wrist_id][1]))
                cv2.line(frame, e_pt, w_pt, side_color, 3, cv2.LINE_AA)
                cv2.circle(frame, w_pt, 5, side_color, -1, cv2.LINE_AA)

            # Extended ray (shoulder→wrist→extended, for region hit testing)
            if conf[shoulder_id] <= 0.5 or conf[wrist_id] <= 0.5:
                continue

            sx, sy = float(xy[shoulder_id][0]), float(xy[shoulder_id][1])
            wx, wy = float(xy[wrist_id][0]), float(xy[wrist_id][1])
            dx, dy = wx - sx, wy - sy
            arm_len = math.hypot(dx, dy)
            if arm_len <= 30:
                continue

            # Extended point
            ex = wx + (dx / arm_len) * arm_len * 6.0
            ey = wy + (dy / arm_len) * arm_len * 6.0

            # Check against each region
            hit = False
            for r in regions:
                rx, ry, rw, rh = r['xywh']
                edges = [
                    ((rx, ry), (rx + rw, ry)),
                    ((rx, ry + rh), (rx + rw, ry + rh)),
                    ((rx, ry), (rx, ry + rh)),
                    ((rx + rw, ry), (rx + rw, ry + rh)),
                ]
                if rx <= ex <= rx + rw and ry <= ey <= ry + rh:
                    hit = True
                    break
                if rx <= sx <= rx + rw and ry <= sy <= ry + rh:
                    hit = True
                    break
                for e1, e2 in edges:
                    if _segments_cross((sx, sy), (ex, ey), e1, e2):
                        hit = True
                        break
                if hit:
                    break

            color = (0, 255, 0) if hit else (0, 0, 255)
            pt1 = (int(sx), int(sy))
            pt2 = (int(ex), int(ey))
            cv2.line(frame, pt1, pt2, color, 2, cv2.LINE_AA)
            cv2.circle(frame, (int(wx), int(wy)), 4, color, -1)
            cv2.circle(frame, (int(ex), int(ey)), 4, color, -1)


def _segments_cross(p1, p2, p3, p4):
    """Check whether line segments p1-p2 and p3-p4 intersect."""
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False


# ---------------------------------------------------------------------------
# Real-time per-action metric display
# ---------------------------------------------------------------------------

def compute_action_metrics(keypoints_obj, action_mapping, rules, regions, lines,
                           detection_kwargs):
    """Compute the raw metric for each action from the current frame.

    Iterates over all detected persons and both arms, returning the best
    (closest) metric per action regardless of detection thresholds.

    Returns a list of dicts: ``{action, metric_type, value, side, segment}``
    """
    import math
    from .geometry import angle_between, min_angle_to_rect

    results = []
    if keypoints_obj is None:
        for m in action_mapping:
            results.append({'action': m['action'], 'metric_type': 'none', 'value': None})
        return results

    rule_lookup = {r['name']: r for r in rules}
    region_lookup = {r['name']: r['xywh'] for r in regions}
    line_lookup = {ln['name']: ln['pts'] for ln in lines}
    kw = detection_kwargs or {}

    for mapping in action_mapping:
        rule = rule_lookup.get(mapping['rule'])
        if rule is None:
            results.append({'action': mapping['action'], 'metric_type': 'none', 'value': None})
            continue

        rtype = rule['type']
        best = None  # (value, side, segment_label)

        for xy, conf in _iter_persons_metric(keypoints_obj):
            for shoulder_id, wrist_id, elbow_id, side in [(5, 9, 7, 'L'), (6, 10, 8, 'R')]:
                if conf[shoulder_id] <= CONF_THRESHOLD:
                    continue

                shoulder = (float(xy[shoulder_id][0]), float(xy[shoulder_id][1]))

                if rtype == 'parallel_line':
                    line_pts = line_lookup.get(rule.get('ref_line', ''))
                    if line_pts is None:
                        continue
                    line_dir = (line_pts[1][0] - line_pts[0][0],
                                line_pts[1][1] - line_pts[0][1])

                    allow_elbow = rule.get('allow_elbow', False)
                    far_pt = segment_label = None

                    if conf[wrist_id] > CONF_THRESHOLD:
                        far_pt = (float(xy[wrist_id][0]), float(xy[wrist_id][1]))
                        segment_label = '肩腕'
                    elif allow_elbow and conf[elbow_id] > CONF_THRESHOLD:
                        far_pt = (float(xy[elbow_id][0]), float(xy[elbow_id][1]))
                        segment_label = '肩肘'

                    if far_pt is None:
                        continue
                    arm_dir = (far_pt[0] - shoulder[0], far_pt[1] - shoulder[1])
                    if math.hypot(*arm_dir) <= kw.get('min_arm_len', 30):
                        continue
                    ang = angle_between(arm_dir, line_dir)
                    if best is None or ang < best[0]:
                        best = (ang, side, segment_label)

                elif rtype == 'pointing' or rtype == 'pointing_with_line':
                    region_xywh = region_lookup.get(rule.get('target_region', ''))
                    if region_xywh is None:
                        continue
                    if conf[wrist_id] <= CONF_THRESHOLD:
                        continue
                    wrist = (float(xy[wrist_id][0]), float(xy[wrist_id][1]))
                    arm_dir = (wrist[0] - shoulder[0], wrist[1] - shoulder[1])
                    if math.hypot(*arm_dir) <= kw.get('min_arm_len', 30):
                        continue

                    if rtype == 'pointing_with_line':
                        line_pts = line_lookup.get(rule.get('ref_line', ''))
                        if line_pts is not None:
                            line_dir = (line_pts[1][0] - line_pts[0][0],
                                        line_pts[1][1] - line_pts[0][1])
                            ang_line = angle_between(arm_dir, line_dir)
                        else:
                            ang_line = 180
                        ang_rect = min_angle_to_rect(wrist, arm_dir, region_xywh)
                        # Show the line angle as primary metric
                        if best is None or ang_line < best[0]:
                            best = (ang_line, side, '腕')
                    else:  # pointing
                        ang = min_angle_to_rect(wrist, arm_dir, region_xywh)
                        if best is None or ang < best[0]:
                            best = (ang, side, '腕')

                elif rtype == 'pass_region':
                    region_xywh = region_lookup.get(rule.get('target_region', ''))
                    if region_xywh is None:
                        continue
                    if conf[wrist_id] <= CONF_THRESHOLD:
                        continue
                    rx, ry, rw, rh = region_xywh
                    wrist = (float(xy[wrist_id][0]), float(xy[wrist_id][1]))
                    arm_vec = (wrist[0] - shoulder[0], wrist[1] - shoulder[1])
                    arm_len = math.hypot(*arm_vec)
                    if arm_len <= kw.get('min_arm_len', 30):
                        continue
                    if kw.get('extend_ray', True):
                        ex = wrist[0] + arm_vec[0] / arm_len * arm_len * 6.0
                        ey = wrist[1] + arm_vec[1] / arm_len * arm_len * 6.0
                        far = (ex, ey)
                    else:
                        far = wrist
                    hit = (rx <= shoulder[0] <= rx + rw and ry <= shoulder[1] <= ry + rh) or \
                          (rx <= far[0] <= rx + rw and ry <= far[1] <= ry + rh)
                    if not hit:
                        edges = [((rx, ry), (rx + rw, ry)),
                                 ((rx, ry + rh), (rx + rw, ry + rh)),
                                 ((rx, ry), (rx, ry + rh)),
                                 ((rx + rw, ry), (rx + rw, ry + rh))]
                        for e1, e2 in edges:
                            if _segments_cross(shoulder, far, e1, e2):
                                hit = True
                                break
                    # 0 = hit, 1 = miss (so lower is better)
                    val = 0.0 if hit else 1.0
                    if best is None or val < best[0]:
                        best = (val, side, '腕')

        results.append({
            'action': mapping['action'],
            'metric_type': rtype,
            'value': best[0] if best else None,
            'side': best[1] if best else None,
            'segment': best[2] if best else None,
        })

    return results


def draw_action_metrics(frame, metrics, *, x=None, y=None):
    """Draw per-action real-time metrics on the left side, below the main panel.

    Args:
        frame: BGR image (modified in-place).
        metrics: list from ``compute_action_metrics()``.
        x, y: top-left position. Defaults to below the main detection panel.
    """
    if not metrics:
        return frame

    font_size = 14
    row_h = 22
    pad = 12
    panel_w = 300

    n = len(metrics)
    panel_h = n * row_h + pad * 2
    if x is None:
        x = 12
    if y is None:
        y = 12 + 34 + 3 * 36 + (4 + 4 * 22 + 6) + 22  # below main overlay

    # Background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + panel_w, y + panel_h), (20, 20, 20), -1)
    cv2.rectangle(overlay, (x, y), (x + panel_w, y + panel_h), (60, 60, 60), 1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    for i, m in enumerate(metrics):
        row_y = y + pad + i * row_h
        action_label = m['action']
        val = m['value']

        if val is None:
            text = f"{action_label}: --"
            color = (100, 100, 100)
        elif m['metric_type'] == 'pass_region':
            if val == 0.0:
                text = f"{action_label}: 穿过"
                color = (80, 220, 80)
            else:
                text = f"{action_label}: 未穿过"
                color = (200, 120, 80)
        else:
            seg = m.get('segment', '')
            side = m.get('side', '')
            text = f"{action_label}: {val:.0f}deg ({side}臂{seg})" if side else f"{action_label}: {val:.0f}deg"
            # Color: green if close to 0 (parallel), yellow if medium, red if far
            if val <= 30:
                color = (80, 220, 80)
            elif val <= 55:
                color = (80, 200, 255)
            else:
                color = (200, 140, 80)

        frame = put_text_cn(frame, text, (x + pad, row_y),
                            font_size, color)

    return frame


def _iter_persons_metric(keypoints_obj):
    """Yield ``(xy, conf)`` numpy arrays for each detected person."""
    for person_idx in range(len(keypoints_obj)):
        kps = keypoints_obj[person_idx]
        xy = kps.xy[0].cpu().numpy()
        conf = kps.conf[0].cpu().numpy()
        yield xy, conf
