# Project Status — 端头门司机行为分析

**Updated:** 2026-07-23

## Architecture

v2 parallel detection + post-hoc analysis.  
All rules run independently per frame → timestamped events → mapped to action sequence after video ends → compliance report.

## Current State

### Detection engine (`src/detection.py` + `src/detector.py`)
- 4 detection types: `parallel_line`, `pass_region`, `pointing`, `pointing_with_line`
- COCO 17-keypoint, indices 5-12 (shoulders/elbows/wrists/hips)
- Per-rule hold counter (30 frames confirm, -2 decay) + 90-frame cooldown
- Normal mode: `frame_skip=0, imgsz=640` (every-frame detection)
- All 7 stations configured and operational

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

### Confidence Colour (`src/confidence_color.py`)
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
| Jingansi | `scripts/run_jingansi.py` | Act1 Call, Act2 CloseDoor, Act3 CheckGap, Act4 CheckLight, Act5 CheckSwitch | PAR + CROSS | Done |
| Tangqiao | `scripts/run_tangqiao.py` | Act1 Call, Act2 CloseDoor, Act3 CheckGap, Act4 CheckLight | PAR + CROSS | Done |
| Pudongdadao | `scripts/run_pudongdadao.py` | Act1 Call, Act2 CloseDoor, Act3 CheckGap, Act4 CheckLight, Act5 CheckSwitch | PAR + CROSS | Done |
| Linping | `scripts/run_linping.py` | Act1 Call, Act2 CloseDoor, Act3 CheckGap, Act4 CheckLight | PAR + CROSS | Done |
| Longhuazhong | `scripts/run_longhuazhong.py` | Act1 Call, Act2 CloseDoor, Act3 CheckGap, Act4 CheckLight, Act5 CheckSwitch | PAR + CROSS | Done |

### Train Detection (all configured stations)
| Station | Background | Track ROI | Train MAD Threshold |
|---------|-----------|-----------|---------------------|
| Shangtichang | Yes | Yes | 20 |
| Baoshan | Yes | Yes | 20 |
| Jingansi | Yes | Yes | 20 |
| Tangqiao | Yes | Yes | 20 |
| Pudongdadao | Yes | Yes | 20 |
| Linping | Yes | Yes | 20 |
| Longhuazhong | Yes | Yes | 20 |

### Train Detection
- Background-frame differencing via `src/train_detector.py`
- Pure accumulation (no decay, no reset): MAD > 20 increments counter, MAD ≤ 20 does nothing
- Arrival: 20 cumulative frames above 20 → confirmed. Departure: 20 frames below 15 → confirmed
- Real-time MAD + hold counter displayed at top-right via `draw_train_status`
- Requires: `track` region + saved background image in annotations JSON

## Key Parameters

| Param | Value | Notes |
|-------|-------|-------|
| angle_threshold | 40° | arm vs ref_line (PAR) |
| min_arm_torso_angle | 45° | prevents false triggers (per-rule overridable) |
| hold_frames | 20 | consecutive confirm count |
| frame_decay | 2/frame | tolerates brief dropout |
| cooldown | 90 frames | prevents event splitting |
| ray extend | 6× | pass_region extension |
| conf_low_threshold | 0.3 | red keypoints below this |
| conf_mid_threshold | 0.6 | yellow below this, green above |
| train_mad_threshold | 20 | MAD above this → train arriving |
| dynamic_angle_coeff | 0.6 | elbow bend compensation for 2D foreshortening |

## Recent Changes

- **2026-07-23**:
  - **静安寺、龙华中新增道岔检测（Act5 CheckSwitch）**: 复用 `line_1` + `anti_parallel`，与浦东大道逻辑一致。两站动作数从 4→5。
  - **删除浦东大道测试脚本** (`scripts/run_pudongdadao_test.py`): 测试功能已合并至正式版，不再需要独立脚本。

- **2026-07-22**:
  - **同帧冲突仲裁** (`src/detector.py`): 同一帧多个角度类规则同时触发时，计算归一化分数（`angle / effective_threshold`），只保留最可信的一个。被淘汰的规则不进冷却，可立即重新累积。`pass_region` 豁免。
  - **`data/` 目录加入 `.gitignore`**: 标注数据不再上传至远程仓库（保密）。

- **2026-07-16**: 
  - **Confidence quality metrics**: conf (keypoint avg), hit_rate (hit/total frames), margin (effective threshold − actual angle) computed at event trigger and displayed in SequenceAnalyzer summary.
  - **CSV report generation** (`src/reporter.py`): Auto-generated after video ends to `output/report/report_xxx.csv`. Contains station info, model params, train arrival/departure, per-action results with quality metrics, overall evaluation.
  - **Output directory restructured**: `output/video/` for annotated videos, `output/report/` for CSV reports.
  - **Standard 5-action template**: 开门后手指呼唤 / 手动关门 / 关门后确认夹缝 / 开车前确认站台指示灯 / 开车前确认站台道岔. Missing actions marked "不需要".
  - All stations now pass `station_name` and `model_path` to VideoPlayer for report generation.
  - `TrainDetector` added `train_info` property for structured arrival/departure data.
  - Dynamic angle compensation: all stations use `dynamic_angle` on parallel_line rules. Effective threshold = 40° + arm_bend × 0.6, compensating for 2D foreshortening.
  - Pudongdadao, Linping, Longhuazhong stations activated with full rules and action mappings.
  - Removed deprecated Streamlit dashboard (`app.py`) and v1 serial state machine (`src/state_machine.py`).
  - Fixed `save_annotations` to preserve `background` and `track_roi` fields on save. Cleaned up stale cache and old package layout.
