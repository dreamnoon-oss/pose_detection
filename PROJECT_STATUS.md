# Project Status — 端头门司机行为分析

**Updated:** 2026-07-06

## Architecture

v2 parallel detection + post-hoc analysis.  
All rules run independently per frame → timestamped events → mapped to action sequence after video ends → compliance report.

## Current State

### Detection engine (`src/detection.py` + `src/detector.py`)
- 4 detection types: `parallel_line`, `pass_region`, `pointing`, `pointing_with_line`
- COCO 17-keypoint, indices 5-12 (shoulders/elbows/wrists/hips)
- Per-rule hold counter (15 frames confirm, -2 decay) + 45-frame cooldown
- Normal mode: `frame_skip=0, imgsz=640` (every-frame detection)

### Visualization (`src/visualization.py`)
- **All English, pure OpenCV rendering** — no PIL dependency, no Chinese text
- `draw_status_overlay` — detection panel (top-left): rules, hold progress, fired count, action status
- `draw_action_metrics` — per-action real-time angles (deg/S-W/S-E/W)
- `draw_arm_rays` — cyan/magenta arm segments, green/red extended rays
- `draw_annotations` — regions + reference lines overlay
- `draw_train_status` — train arrival/departure badge (top-right)
- `draw_analysis_result` — final result overlay (bottom-left)
- `draw_frame_info` — frame counter (top-right)

### Player (`src/player.py`)
- Interactive video player with progress bar, pause/seek/annotate
- Pause: only left panel (right panel removed)
- Keys: Space=pause, Q=quit, R=draw region, L=draw line, T=track ROI, B=save bg, S=save JSON, Z=reset

### Scenarios

| Scenario | Script | Actions | Types |
|----------|--------|---------|-------|
| Shangtichang | `scripts/run_shangtichang.py` | Act1 Call, Act2 CloseDoor, Act3 CheckGap, Act4 CheckLight | PAR + CROSS |
| Baoshan | `scripts/run_baoshan.py` | Act1 PointFwd, Act2 CheckR2, Act3 PointFwd, Act4 CheckR3, Act5 CheckR4 | P+L + POINT |

### Web UI
- `app.py` — Streamlit dashboard with parameter controls, video preview, results tabs

### Train Detection
- Background-frame differencing via `src/train_detector.py`
- MAD-based hysteresis threshold in track ROI

## Key Parameters

| Param | Value | Notes |
|-------|-------|-------|
| angle_threshold | 40° | arm vs ref_line (PAR) |
| min_arm_torso_angle | 45° | prevents false triggers |
| hold_frames | 15 | consecutive confirm count |
| frame_decay | 2/frame | tolerates brief dropout |
| cooldown | 45 frames | prevents event splitting |
| ray extend | 6× | pass_region extension |

## Recent Changes

- **PIL → cv2 rendering**: Replaced all `put_text_cn` (PIL-based Chinese rendering) with native `cv2.putText` + English labels. Eliminated ~300MB/frame memory churn, ~100ms+ saved per frame.
- **Pause UX**: Removed right-side redundant panel on pause.
- **Detection mode**: Set `frame_skip=0, imgsz=640` (full resolution, every frame).
- **English labels**: All action names, metrics, and analysis output in English.
