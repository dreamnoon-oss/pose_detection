# Project Status — 端头门司机行为分析

**Updated:** 2026-07-13

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
| Pudongdadao | `scripts/run_pudongdadao.py` | TBD | TBD |
| Linping | `scripts/run_linping.py` | TBD | TBD |
| Jingansi | `scripts/run_jingansi.py` | TBD | TBD |
| Longhuazhong | `scripts/run_longhuazhong.py` | TBD | TBD |
| Tangqiao | `scripts/run_tangqiao.py` | TBD | TBD |

### Web UI
- `app.py` — Streamlit dashboard with parameter controls, video preview, results tabs

### Train Detection
- Background-frame differencing via `src/train_detector.py`
- Pure accumulation (no decay, no reset): MAD > 30 increments counter, MAD ≤ 30 does nothing
- Arrival: 20 cumulative frames above 30 → confirmed. Departure: 20 frames below 15 → confirmed
- Real-time MAD + hold counter displayed at top-right via `draw_train_status`

## Key Parameters

| Param | Value | Notes |
|-------|-------|-------|
| angle_threshold | 40° | arm vs ref_line (PAR) |
| min_arm_torso_angle | 45° | prevents false triggers |
| hold_frames | 30 | consecutive confirm count (was 15) |
| frame_decay | 2/frame | tolerates brief dropout |
| cooldown | 90 frames | prevents event splitting (was 45) |
| ray extend | 6× | pass_region extension |

## Recent Changes

- **New station scripts**: Added `run_pudongdadao.py`, `run_linping.py`, `run_jingansi.py`, `run_longhuazhong.py`, `run_tangqiao.py` — detection rules and action mappings TBD.
- **Critical bugfix: false trackbar seek loop** — `cv2.setTrackbarPos` triggers the trackbar callback on every frame, causing `_handle_seek` to run every other iteration. This double-processed frames (extra cap.set + YOLO + reset), cutting effective GPU throughput in half and resetting the train detector hold counter every 2 frames. Fix: skip seek when position delta ≤ 1.
- **Train detector rewritten**: Pure accumulation counter (no decay, no reset on MAD drop). MAD > 30 increments arrival hold, MAD < 15 increments departure hold. Confirmed at 20 cumulative frames each. Real-time MAD + hold progress displayed in top-right overlay.
- **Track ROI UX**: T key now toggles select/delete. Fixed bug where existing track region from JSON wasn't recognized without a saved background.
- **Detection thresholds doubled**: `hold_frames` 15→30, `cooldown_frames` 45→90.
- **Performance profiling**: `scripts/profile_timing.py`. On 3060 with yolo26x-pose: GPU inference 53ms/frame (75%).
- **PIL → cv2 rendering**: Eliminated ~300MB/frame memory churn.
- **English labels**: All action names, metrics, and analysis output in English.
