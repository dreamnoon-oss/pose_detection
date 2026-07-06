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


def load_background_info(json_path):
    """Load background image path and track ROI name from a JSON annotation file.

    Returns:
        ``(bg_image_path, track_roi_name)`` or ``(None, None)`` if not configured.
    """
    if not os.path.exists(json_path):
        return None, None

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    bg = data.get("background")
    if bg is None:
        return None, None

    data_dir = os.path.dirname(json_path)
    bg_path = os.path.join(data_dir, bg["image"])
    track_roi = data.get("track_roi")
    return bg_path, track_roi


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


def save_background(json_path, frame, frame_idx):
    """Save the current raw frame as a background reference image.

    The PNG is saved alongside the JSON file; a ``background`` key is added
    (or updated) in the JSON pointing to the image file and capture frame.
    If a ``track_roi`` field does not yet exist in the JSON, the first region
    is automatically assigned as the track monitoring ROI.

    Args:
        json_path: path to the annotations JSON file.
        frame: BGR numpy array (raw video frame, no overlays).
        frame_idx: frame number where the background was captured.
    """
    import os

    data_dir = os.path.dirname(json_path)
    bg_name = os.path.splitext(os.path.basename(json_path))[0] + "_background.png"
    bg_path = os.path.join(data_dir, bg_name)
    cv2.imwrite(bg_path, frame)
    print(f">>> 背景帧已保存: {bg_name} (帧 {frame_idx})")

    data = {}
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

    data["background"] = {
        "image": bg_name,
        "frame": frame_idx,
    }

    # Auto-set track_roi to first region if not already set
    if "track_roi" not in data and data.get("regions"):
        data["track_roi"] = data["regions"][0]["name"]
        print(f"    自动设置 track_roi = {data['track_roi']}")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"    已更新 {os.path.basename(json_path)} 中的 background 字段")
