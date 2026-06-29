"""Load, save, and interactive annotation of regions and reference lines."""

import json
import cv2
import os


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def load_annotations(json_path):
    """Load regions and lines from a JSON annotation file.

    Returns:
        ``(regions, lines)`` where each region is ``{"name": str, "xywh": tuple}``
        and each line is ``{"name": str, "pts": [pt1, pt2]}`` with pts as tuples.
    """
    regions, lines = [], []
    if not os.path.exists(json_path):
        return regions, lines

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for r in data.get("regions", []):
        r["xywh"] = tuple(r["xywh"])
        regions.append(r)
    for ln in data.get("lines", []):
        ln["pts"] = [tuple(p) for p in ln["pts"]]
        lines.append(ln)
    return regions, lines


def save_annotations(json_path, regions, lines, video_path, frame_idx,
                     width, height):
    """Write regions and lines to a JSON file."""
    save_data = {
        "video": video_path,
        "frame": frame_idx,
        "width": width,
        "height": height,
        "regions": [{"name": r["name"], "xywh": list(r["xywh"])} for r in regions],
        "lines": [{"name": ln["name"], "pts": [list(p) for p in ln["pts"]]} for ln in lines],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Interactive annotation tools (called while paused)
# ---------------------------------------------------------------------------

def select_roi(window_name, frame, saved_regions):
    """OpenCV ROI selection. Returns updated *saved_regions* list."""
    roi = cv2.selectROI(window_name, frame, False)
    if roi[2] > 0 and roi[3] > 0:
        name = f"region_{len(saved_regions) + 1}"
        saved_regions.append({"name": name, "xywh": roi})
        print(f">>> {name}: x={roi[0]}, y={roi[1]}, w={roi[2]}, h={roi[3]}")
        print(f"    已选 {len(saved_regions)} 个区域，继续按 R 添加，按 S 保存")
    else:
        print("取消框选")
    return saved_regions


def draw_line_interactive(window_name, frame, saved_lines):
    """Two-click line drawing. Blocks until two points are set or Esc is pressed.

    Returns updated *saved_lines* list.
    """
    print("请在画面上点击两个点画参考线（沿列车方向）...")
    line_pts = []

    def on_line_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            line_pts.append((x, y))
            tmp = frame.copy()
            for pt in line_pts:
                cv2.circle(tmp, pt, 5, (0, 165, 255), -1)
            if len(line_pts) == 2:
                cv2.line(tmp, line_pts[0], line_pts[1], (0, 200, 255), 2)
                cv2.arrowedLine(tmp, line_pts[0], line_pts[1],
                                (0, 200, 255), 2, tipLength=0.08)
            cv2.imshow(window_name, tmp)
            if len(line_pts) >= 2:
                print(f"  点 {len(line_pts)}: ({x}, {y})")

    cv2.setMouseCallback(window_name, on_line_click)
    while len(line_pts) < 2:
        k = cv2.waitKey(30) & 0xFF
        if k == 27 or k == ord('q'):
            break
    cv2.setMouseCallback(window_name, lambda *args: None)

    if len(line_pts) == 2:
        name = f"line_{len(saved_lines) + 1}"
        saved_lines.append({"name": name, "pts": line_pts})
        print(f">>> {name}: {line_pts[0]} -> {line_pts[1]}")
        print(f"    当前共 {len(saved_lines)} 条参考线，按 S 保存")
    else:
        print("取消画线")
    return saved_lines


def remove_last_region(saved_regions):
    """Remove the most recently added region."""
    if saved_regions:
        removed = saved_regions.pop()
        print(f"已删除 {removed['name']}，剩余 {len(saved_regions)} 个区域")
    return saved_regions
