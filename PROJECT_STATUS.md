# Project Status — 端头门司机行为分析

**Updated:** 2026-07-15

## Architecture

v2 parallel detection + post-hoc analysis.  
All rules run independently per frame → timestamped events → mapped to action sequence after video ends → compliance report.

## Current State

### Detection engine (`src/detection.py` + `src/detector.py`)
- 4 detection types: `parallel_line`, `pass_region`, `pointing`, `pointing_with_line`
- COCO 17-keypoint, indices 5-12 (shoulders/elbows/wrists/hips)
- Per-rule hold counter (30 frames confirm, -2 decay) + 90-frame cooldown
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
- `draw_confidence_legend` — confidence tier colour legend (bottom-right)

### Confidence Colour (`src/confidence_color.py` — NEW)
- Three-tier keypoint confidence colouring: red (<0.3), yellow (0.3-0.6), green (>0.6)
- Configurable per-station via `conf_low_threshold` / `conf_mid_threshold` in VideoPlayer
- Applied to pose keypoints, elbow circles, and wrist circles
- Legend drawn at bottom-right corner

### Player (`src/player.py`)
- Interactive video player with progress bar, pause/seek/annotate
- Pause: only left panel (right panel removed)
- Keys: Space=pause, Q=quit, R=draw region, L=draw line, T=track ROI, B=save bg, S=save JSON, Z=reset
- Confidence mapper created in constructor, passed through all render calls

### Scenarios

| Scenario | Script | Actions | Types | Status |
|----------|--------|---------|-------|--------|
| Shangtichang | `scripts/run_shangtichang.py` | Act1 Call, Act2 CloseDoor, Act3 CheckGap, Act4 CheckLight | PAR + CROSS | Done |
| Baoshan | `scripts/run_baoshan.py` | Act1 PointFwd, Act2 CheckR2, Act3 PointFwd, Act4 CheckR3, Act5 CheckR4 | P+L + POINT | Done |
| Jingansi | `scripts/run_jingansi.py` | Act1 Call, Act2 CloseDoor, Act3 CheckGap, Act4 CheckLight | PAR + CROSS | Rules configured; pending bg save for train detection |
| Pudongdadao | `scripts/run_pudongdadao.py` | TBD | TBD | Placeholder |
| Linping | `scripts/run_linping.py` | TBD | TBD | Placeholder |
| Longhuazhong | `scripts/run_longhuazhong.py` | TBD | TBD | Placeholder |
| Tangqiao | `scripts/run_tangqiao.py` | TBD | TBD | Placeholder |

### Web UI
- `app.py` — Streamlit dashboard with parameter controls, video preview, results tabs

### Train Detection
- Background-frame differencing via `src/train_detector.py`
- Pure accumulation (no decay, no reset): MAD > 30 increments counter, MAD ≤ 30 does nothing
- Arrival: 20 cumulative frames above 30 → confirmed. Departure: 20 frames below 15 → confirmed
- Real-time MAD + hold counter displayed at top-right via `draw_train_status`
- Requires: `track` region + saved background image in annotations JSON

## Key Parameters

| Param | Value | Notes |
|-------|-------|-------|
| angle_threshold | 40° | arm vs ref_line (PAR) |
| min_arm_torso_angle | 45° | prevents false triggers |
| hold_frames | 30 | consecutive confirm count |
| frame_decay | 2/frame | tolerates brief dropout |
| cooldown | 90 frames | prevents event splitting |
| ray extend | 6× | pass_region extension |
| conf_low_threshold | 0.3 | red keypoints below this |
| conf_mid_threshold | 0.6 | yellow below this, green above |

## Recent Changes

- **Confidence colour system** (`src/confidence_color.py`): New reusable module for three-tier keypoint colouring. Configurable thresholds passed through VideoPlayer to all render functions. All 7 station scripts updated with defaults. Legend overlay added to bottom-right corner.
- **Jingansi station activated**: Detection rules and action mapping set to match Shangtichang (PAR + CROSS). Train detection pending background frame save.
- **New station scripts**: Added `run_pudongdadao.py`, `run_linping.py`, `run_jingansi.py`, `run_longhuazhong.py`, `run_tangqiao.py`.
- **Critical bugfix: false trackbar seek loop** — `cv2.setTrackbarPos` triggers the trackbar callback on every frame. Fix: skip seek when position delta ≤ 1.
- **Train detector rewritten**: Pure accumulation counter (no decay, no reset on MAD drop).
- **Detection thresholds doubled**: `hold_frames` 15→30, `cooldown_frames` 45→90.
- **PIL → cv2 rendering**: Eliminated ~300MB/frame memory churn.
- **English labels**: All action names, metrics, and analysis output in English.
