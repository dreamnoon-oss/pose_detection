"""Interactive video player with pose detection, annotation, and action tracking."""

import os
import time
import cv2

from . import visualization as viz
from .annotation import select_roi, draw_line_interactive, remove_last_region, save_annotations
from .analyzer import SequenceAnalyzer


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
                 model_conf=0.5):
        self.model = model
        self.video_path = video_path
        self.detector = detector
        self.action_mapping = action_mapping or []
        self.annotations_file = annotations_file
        self.output_dir = output_dir
        self.output_name = output_name
        self.model_conf = model_conf

        self.cap = None
        self.out = None
        self._paused = False
        self._last_frame = None
        self._trackbar_pos = 0
        self._user_seeking = False
        self._analysis = None
        self._last_active = {}
        self._last_metrics = []
        self._last_kp = None

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

                if self._user_seeking:
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
        results = self.model(frame, verbose=False, conf=self.model_conf)
        kp = results[0].keypoints if results[0].keypoints is not None else None

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
        annotated = viz.draw_pose(frame, results)
        viz.draw_arm_rays(annotated, kp, self.detector.regions)
        viz.draw_annotations(annotated, self.detector.regions, self.detector.lines)
        annotated, status_bottom = viz.draw_status_overlay(
            annotated, self.detector.rules, active,
            self.detector.events, self.action_mapping)
        annotated = viz.draw_action_metrics(annotated, metrics, y=status_bottom + 6)

        infer_ms = (time.time() - t0) * 1000
        self._last_frame = annotated.copy()

        # Write output
        self.out.write(annotated)

        # Progress tracking
        if self._window_exists():
            cur = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            cv2.setTrackbarPos("Progress", self.window_name, cur)
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

        elif key == ord('d') and self._paused:
            self.detector.regions = remove_last_region(self.detector.regions)

        return False

    def _handle_seek(self):
        """Seek to the trackbar position."""
        self._user_seeking = False
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self._trackbar_pos)
        ret, frame = self.cap.read()
        if ret:
            results = self.model(frame, verbose=False, conf=self.model_conf)
            kp = results[0].keypoints if results[0].keypoints is not None else None
            active, _ = self.detector.update(kp)
            self._last_active = active
            self._last_frame = viz.draw_pose(frame, results)
            viz.draw_arm_rays(self._last_frame, kp, self.detector.regions)
            viz.draw_annotations(self._last_frame, self.detector.regions, self.detector.lines)
            self._last_frame, status_bottom = viz.draw_status_overlay(
                self._last_frame, self.detector.rules,
                active, self.detector.events, self.action_mapping)
            metrics = viz.compute_action_metrics(
                kp, self.action_mapping, self.detector.rules,
                self.detector.regions, self.detector.lines,
                self.detector.detection_kwargs)
            self._last_metrics = metrics
            self._last_frame = viz.draw_action_metrics(
                self._last_frame, metrics, y=status_bottom + 6)
            cv2.imshow(self.window_name, self._last_frame)

    def _draw_paused_frame(self):
        """Re-render the paused frame with annotation overlays."""
        if self._paused and self._last_frame is not None:
            paused_frame = self._last_frame.copy()
            viz.draw_annotations(paused_frame, self.detector.regions, self.detector.lines)
            viz.draw_pause_indicator(paused_frame)
            paused_frame, status_bottom = viz.draw_status_overlay(
                paused_frame, self.detector.rules,
                self._last_active, self.detector.events, self.action_mapping,
                align_right=True)
            paused_frame = viz.draw_action_metrics(
                paused_frame, self._last_metrics, x=12, y=status_bottom + 6)
            cur = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) if self.cap else 0
            viz.draw_frame_info(paused_frame, cur, self.total_frames, self.fps)
            cv2.imshow(self.window_name, paused_frame)

    def _on_trackbar(self, pos):
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
        print("  S    = 保存区域+参考线到 JSON 文件")
        print("  ----- 随时可用 -----")
        print("  Z    = 重置检测器，清空所有事件")

    # ------------------------------------------------------------------
    # Internal: analysis
    # ------------------------------------------------------------------

    def _run_analysis(self, frame):
        """Run sequence analysis on the recorded events and overlay results."""
        analyzer = SequenceAnalyzer(
            self.detector.events, self.action_mapping, fps=self.fps)
        self._analysis = analyzer.analyze()
        print("\n" + analyzer.summary())
        return viz.draw_analysis_result(frame, self._analysis)
