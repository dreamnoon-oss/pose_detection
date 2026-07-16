"""Interactive video player with pose detection, annotation, and action tracking."""

import os
import time
import cv2

from . import visualization as viz
from .annotation import (select_roi, draw_line_interactive,
                         remove_last_region, remove_last_line,
                         save_annotations, save_background,
                         load_background_info)
from .analyzer import SequenceAnalyzer
from .confidence_color import ConfidenceColorMapper
from .config import CONF_LOW_THRESHOLD, CONF_MID_THRESHOLD
from .train_detector import TrainDetector


class VideoPlayer:
    """Interactive player that runs YOLO pose detection on a video stream.

    Parameters:
        model: Loaded ultralytics YOLO model.
        video_path: Path to the input video.
        detector: ``ParallelDetector`` instance.
        action_mapping: List of ``{action, rule, occurrence}`` dicts.
        annotations_file: Path to the regions/lines JSON file.
        output_dir: Directory for the annotated output video.
        output_name: Filename for the output video.
        model_conf: Confidence threshold passed to the YOLO model.
    """

    def __init__(self, model, video_path, detector, action_mapping=None, *,
                 annotations_file, output_dir, output_name="pose_output.mp4",
                 model_conf=0.5, imgsz=640, frame_skip=0, half=False,
                 conf_low_threshold=CONF_LOW_THRESHOLD,
                 conf_mid_threshold=CONF_MID_THRESHOLD,
                 train_mad_threshold=20,
                 show_arm_bend=False):
        self.model = model
        self.video_path = video_path
        self.detector = detector
        self.action_mapping = action_mapping or []
        self.annotations_file = annotations_file
        self.output_dir = output_dir
        self.output_name = output_name
        self.model_conf = model_conf
        self.imgsz = imgsz            # model input resolution
        self.frame_skip = frame_skip  # 0=every frame, 1=every 2nd, 2=every 3rd, etc.
        self.half = half              # FP16 inference
        self.train_mad_threshold = train_mad_threshold
        self.show_arm_bend = show_arm_bend

        self.conf_mapper = ConfidenceColorMapper(
            low_threshold=conf_low_threshold,
            mid_threshold=conf_mid_threshold)

        self.cap = None
        self.out = None
        self._paused = False
        self._last_frame = None
        self._trackbar_pos = 0
        self._user_seeking = False
        self._setting_trackbar = False
        self._analysis = None
        self._last_active = {}
        self._last_metrics = []
        self._last_kp = None
        self._last_raw_frame = None
        self._last_results = None

        # Train detector (optional — enabled when background + track_roi exist)
        self.train_detector = None
        _bg_path, _track_name = load_background_info(annotations_file)
        self._track_roi_name = _track_name  # may be None
        # Fallback: scan loaded regions for one named "track"
        if self._track_roi_name is None:
            for r in self.detector.regions:
                if r['name'] == 'track':
                    self._track_roi_name = 'track'
                    break
        if _bg_path and self._track_roi_name and os.path.exists(_bg_path):
            track_roi = self._lookup_roi(self._track_roi_name)
            if track_roi is not None:
                self.train_detector = TrainDetector(
                    _bg_path, track_roi, fps=1.0,
                    high_threshold=self.train_mad_threshold)
                self.detector.enabled = False  # wait for train arrival
                print(f"列车检测已启用  track_roi={_track_name}  "
                      f"background={os.path.basename(_bg_path)}  "
                      f"(动作检测等待列车到站)")
        self._last_train_state = None
        self._last_train_mad = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self):
        """Start the interactive detection loop. Blocks until the user presses Q."""
        self._open()
        self._print_help()

        try:
            while True:
                if not self._window_exists():
                    break

                was_seeking = self._user_seeking
                self._user_seeking = False
                if was_seeking:
                    self._handle_seek()
                    continue

                if not self._paused:
                    key = self._process_frame()
                else:
                    key = None

                if self._paused:
                    k = cv2.waitKey(30)
                    if k == -1:
                        # Window was closed via the X button
                        if not self._window_exists():
                            break
                    key = k & 0xFF

                if key is not None:
                    if self._handle_key(key):
                        break  # quit

                self._draw_paused_frame()

        except KeyboardInterrupt:
            print("\n用户中断")

        finally:
            self._close()

    # ------------------------------------------------------------------
    # Internal: I/O
    # ------------------------------------------------------------------

    def _open(self):
        self.cap = cv2.VideoCapture(self.video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.train_detector is not None:
            self.train_detector.fps = self.fps
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.delay = max(1, int(1000 / self.fps))

        os.makedirs(self.output_dir, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_path = os.path.join(self.output_dir, self.output_name)
        self.out = cv2.VideoWriter(out_path, fourcc, self.fps,
                                   (self.width, self.height))

        self.window_name = "Pose Detection (5-12 keypoints)"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, min(self.width, 1280), min(self.height, 720))
        cv2.createTrackbar("Progress", self.window_name, 0,
                           max(0, self.total_frames - 1), self._on_trackbar)

    def _close(self):
        # Print train summary on manual exit too
        if self.train_detector is not None:
            end_time = (self.cap.get(cv2.CAP_PROP_POS_FRAMES) / self.fps
                        if self.fps else 0)
            print("\n" + self.train_detector.summary(end_time))
        if self.cap is not None:
            self.cap.release()
        if self.out is not None:
            self.out.release()
        cv2.destroyAllWindows()
        print(f"结果已保存到: {os.path.join(self.output_dir, self.output_name)}")

    # ------------------------------------------------------------------
    # Internal: per-frame processing
    # ------------------------------------------------------------------

    def _process_frame(self):
        """Read one frame, run detection + state machine + rendering. Returns key code."""
        ret, frame = self.cap.read()
        if not ret:
            self._paused = True
            print("播放结束，已自动暂停")
            if self._last_frame is not None:
                viz.draw_pause_indicator(self._last_frame)
                cv2.putText(self._last_frame, "END", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                self._last_frame = self._run_analysis(self._last_frame)
                cv2.imshow(self.window_name, self._last_frame)
            return None

        t0 = time.time()

        # Pose detection
        cur_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        if (self.frame_skip <= 0 or cur_frame % (self.frame_skip + 1) == 0
                or self._last_results is None):
            results = self.model(frame, verbose=False, conf=self.model_conf,
                                 imgsz=self.imgsz, half=self.half)
            self._last_results = results
        else:
            results = self._last_results
        kp = results[0].keypoints if (results and results[0].keypoints is not None) else None

        kp = self._filter_best_person(kp, results)
        if kp is not None and results is not None:
            results[0].keypoints = kp
            if results[0].boxes is not None and len(results[0].boxes) > len(kp):
                results[0].boxes = results[0].boxes[[results[0].boxes.conf.argmax().item()]]

        # Parallel detection
        active, new_events = self.detector.update(kp)
        self._last_active = active
        self._last_kp = kp

        # Per-action real-time metrics
        metrics = viz.compute_action_metrics(
            kp, self.action_mapping, self.detector.rules,
            self.detector.regions, self.detector.lines,
            self.detector.detection_kwargs)
        self._last_metrics = metrics

        # Render pipeline
        self._last_raw_frame = frame.copy()
        annotated = viz.draw_pose(frame, results, self.conf_mapper)
        viz.draw_arm_rays(annotated, kp, self.detector.regions, self.conf_mapper)
        viz.draw_annotations(annotated, self.detector.regions, self.detector.lines,
                             self._track_roi_name)
        viz.draw_confidence_legend(annotated, self.conf_mapper)

        # Train detection (background only, result printed at end)
        if self.train_detector is not None:
            prev_state = self.train_detector.state
            train_state, train_mad = self.train_detector.update(frame)
            self._last_train_state = train_state
            self._last_train_mad = train_mad
            # Enable action detection when train arrives
            if prev_state != 'PRESENT' and train_state == 'PRESENT':
                self.detector.enable()
            elif prev_state == 'PRESENT' and train_state != 'PRESENT':
                self.detector.enabled = False
        td = self.train_detector
        viz.draw_train_status(annotated, self._last_train_state, self._last_train_mad,
                              hold_counter=(td.hold_counter if td else 0),
                              hold_target=(td.hold_target if td else 0))

        annotated, status_bottom = viz.draw_status_overlay(
            annotated, self.detector.rules, active,
            self.detector.events, self.action_mapping)
        annotated = viz.draw_action_metrics(annotated, metrics, y=status_bottom + 6,
                                              show_arm_bend=self.show_arm_bend)

        infer_ms = (time.time() - t0) * 1000
        self._last_frame = annotated.copy()

        # Write output
        self.out.write(annotated)

        # Progress tracking
        if self._window_exists():
            cur = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            self._setting_trackbar = True
            cv2.setTrackbarPos("Progress", self.window_name, cur)
            self._setting_trackbar = False
            viz.draw_frame_info(annotated, cur, self.total_frames, self.fps)
            cv2.imshow(self.window_name, annotated)

        wait_ms = max(1, self.delay - int(infer_ms))
        return cv2.waitKey(wait_ms) & 0xFF

    # ------------------------------------------------------------------
    # Internal: controls
    # ------------------------------------------------------------------

    def _handle_key(self, key):
        """Process a key press. Returns True if the player should quit."""
        if key == ord('q'):
            return True

        elif key == ord(' '):
            self._paused = not self._paused
            print("暂停" if self._paused else "继续")

        elif key == ord('z'):
            self.detector.reset()
            self._analysis = None
            print("检测器已重置，所有事件已清空")

        elif key == ord('r') and self._paused and self._last_frame is not None:
            print(f"请用鼠标框选第 {len(self.detector.regions) + 1} 个区域，回车确认，Esc 取消...")
            self.detector.regions = select_roi(self.window_name, self._last_frame,
                                                self.detector.regions)

        elif key == ord('l') and self._paused and self._last_frame is not None:
            self.detector.lines = draw_line_interactive(self.window_name, self._last_frame,
                                                         self.detector.lines)

        elif key == ord('s') and self._paused:
            cur = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            save_annotations(self.annotations_file, self.detector.regions,
                             self.detector.lines, self.video_path, cur,
                             self.width, self.height)
            print(f"✅ 已保存 {len(self.detector.regions)} 个区域 + "
                  f"{len(self.detector.lines)} 条参考线 -> {self.annotations_file}")

        elif key == ord('t') and self._paused and self._last_frame is not None:
            if self._track_roi_name is not None:
                # Track ROI already exists — delete it
                self.detector.regions = [r for r in self.detector.regions
                                         if r['name'] != 'track']
                self._track_roi_name = None
                self.train_detector = None
                print(">>> 轨道区域已删除")
            else:
                # No track ROI — select a new one
                print("请框选轨道区域（回车确认，Esc 取消）...")
                self.detector.regions = select_roi(self.window_name, self._last_frame,
                                                   self.detector.regions)
                if self.detector.regions:
                    last = self.detector.regions[-1]
                    if last['name'].startswith('region_'):
                        last['name'] = 'track'
                        self._track_roi_name = 'track'
                        print(f">>> 轨道区域已更新: {last['xywh']}")

        elif key == ord('b') and self._paused and self._last_raw_frame is not None:
            if self._track_roi_name is None:
                print("请先按 T 框选轨道区域")
            else:
                cur = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                save_background(self.annotations_file, self._last_raw_frame, cur,
                               self._track_roi_name)
                self._activate_train_detector()

        elif key == ord('d') and self._paused:
            self.detector.regions = remove_last_region(self.detector.regions)

        elif key == ord('k') and self._paused:
            self.detector.lines = remove_last_line(self.detector.lines)

        return False

    def _handle_seek(self):
        """Seek to the trackbar position."""
        self._user_seeking = False
        cur = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        if abs(self._trackbar_pos - cur) <= 1:
            return  # programmatic update, skip
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self._trackbar_pos)
        ret, frame = self.cap.read()
        if ret:
            self._last_raw_frame = frame.copy()
            if self.train_detector is not None:
                self.train_detector.reset(self._trackbar_pos)
                self.detector.enabled = False
            results = self.model(frame, verbose=False, conf=self.model_conf,
                                 imgsz=self.imgsz, half=self.half)
            kp = results[0].keypoints if (results and results[0].keypoints is not None) else None
            kp = self._filter_best_person(kp, results)
            if kp is not None and results is not None:
                results[0].keypoints = kp
                if results[0].boxes is not None and len(results[0].boxes) > len(kp):
                    results[0].boxes = results[0].boxes[[results[0].boxes.conf.argmax().item()]]
            active, _ = self.detector.update(kp)
            self._last_active = active
            self._last_results = results
            self._last_frame = viz.draw_pose(frame, results, self.conf_mapper)
            viz.draw_arm_rays(self._last_frame, kp, self.detector.regions, self.conf_mapper)
            viz.draw_annotations(self._last_frame, self.detector.regions, self.detector.lines,
                                 self._track_roi_name)
            viz.draw_confidence_legend(self._last_frame, self.conf_mapper)
            if self.train_detector is not None:
                train_state, train_mad = self.train_detector.update(frame)
                self._last_train_state = train_state
                self._last_train_mad = train_mad
            self._last_frame, status_bottom = viz.draw_status_overlay(
                self._last_frame, self.detector.rules,
                active, self.detector.events, self.action_mapping)
            metrics = viz.compute_action_metrics(
                kp, self.action_mapping, self.detector.rules,
                self.detector.regions, self.detector.lines,
                self.detector.detection_kwargs)
            self._last_metrics = metrics
            self._last_frame = viz.draw_action_metrics(
                self._last_frame, metrics, y=status_bottom + 6,
                show_arm_bend=self.show_arm_bend)
            cv2.imshow(self.window_name, self._last_frame)

    def _draw_paused_frame(self):
        """Re-render the paused frame with annotation overlays."""
        if self._paused and self._last_frame is not None:
            paused_frame = self._last_frame.copy()
            viz.draw_annotations(paused_frame, self.detector.regions, self.detector.lines,
                                 self._track_roi_name)
            viz.draw_pause_indicator(paused_frame)
            cur = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) if self.cap else 0
            viz.draw_frame_info(paused_frame, cur, self.total_frames, self.fps)
            td = self.train_detector
            viz.draw_train_status(paused_frame, self._last_train_state, self._last_train_mad,
                                  hold_counter=(td.hold_counter if td else 0),
                                  hold_target=(td.hold_target if td else 0))
            cv2.imshow(self.window_name, paused_frame)

    def _on_trackbar(self, pos):
        if self._setting_trackbar:
            return
        self._user_seeking = True
        self._trackbar_pos = pos

    def _window_exists(self):
        try:
            return cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) >= 1
        except cv2.error:
            return False

    # ------------------------------------------------------------------
    # Internal: help text
    # ------------------------------------------------------------------

    def _print_help(self):
        print(f"视频信息: {self.width}x{self.height}, {self.fps:.1f}fps, "
              f"共{self.total_frames}帧, 时长{self.total_frames / self.fps:.1f}秒")
        print(f"显示关键点: 5-12 (左肩/右肩/左肘/右肘/左手腕/右手腕/左髋/右髋)")
        print("操作说明:")
        print("  空格 = 暂停/继续")
        print("  Q    = 退出")
        print("  拖拽进度条 = 跳转到指定帧")
        print("  ----- 暂停时可用 -----")
        print("  R    = 鼠标框选区域（指示灯等）")
        print("  L    = 鼠标点击两点画参考线（沿列车）")
        print("  D    = 删除最后一个区域")
        print("  K    = 删除最后一条参考线")
        print("  S    = 保存区域+参考线到 JSON 文件")
        print("  B    = 保存当前帧为背景参考图（轨道空闲时）")
        print("  T    = 框选/删除轨道监控区域（用于列车检测）")
        print("  ----- 随时可用 -----")
        print("  Z    = 重置检测器，清空所有事件")

    # ------------------------------------------------------------------
    # Internal: analysis
    # ------------------------------------------------------------------

    @staticmethod
    def _filter_best_person(kp, results):
        """Keep only the highest-confidence bounding-box person (driver)."""
        if kp is None or results is None or results[0].boxes is None:
            return kp
        boxes = results[0].boxes
        if len(boxes) <= 1:
            return kp
        best_idx = boxes.conf.argmax().item()
        return kp[[best_idx]]

    def _lookup_roi(self, name):
        """Return xywh tuple for a region by name, or None."""
        for r in self.detector.regions:
            if r['name'] == name:
                return r['xywh']
        return None

    def _activate_train_detector(self):
        """Create or recreate TrainDetector from current annotations."""
        bg_path, track_name = load_background_info(self.annotations_file)
        if not bg_path or not track_name or not os.path.exists(bg_path):
            return
        self._track_roi_name = track_name
        roi = self._lookup_roi(track_name)
        if roi is not None:
            self.train_detector = TrainDetector(
                bg_path, roi, fps=getattr(self, 'fps', 1.0),
                high_threshold=self.train_mad_threshold)
            self._last_train_state = None
            self._last_train_mad = 0.0
            print(f"列车检测已激活  track_roi={track_name}")

    def _run_analysis(self, frame):
        """Run sequence analysis on the recorded events and overlay results."""
        analyzer = SequenceAnalyzer(
            self.detector.events, self.action_mapping, fps=self.fps)
        self._analysis = analyzer.analyze()
        print("\n" + analyzer.summary())
        if self.train_detector is not None:
            end_time = self.total_frames / self.fps if self.fps else 0
            print("\n" + self.train_detector.summary(end_time))
        return viz.draw_analysis_result(frame, self._analysis)
