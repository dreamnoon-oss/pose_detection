"""Interactive video player with pose detection, annotation, and action tracking."""

import os
import time
import cv2

from . import visualization as viz
from .annotation import select_roi, draw_line_interactive, remove_last_region, save_annotations


class VideoPlayer:
    """Interactive player that runs YOLO pose detection on a video stream.

    Parameters:
        model: Loaded ultralytics YOLO model.
        video_path: Path to the input video.
        state_machine: ``ActionStateMachine`` instance.
        annotations_file: Path to the regions/lines JSON file.
        output_dir: Directory for the annotated output video.
        output_name: Filename for the output video.
        model_conf: Confidence threshold passed to the YOLO model.
    """

    def __init__(self, model, video_path, state_machine, *,
                 annotations_file, output_dir, output_name="pose_output.mp4",
                 model_conf=0.5):
        self.model = model
        self.video_path = video_path
        self.sm = state_machine
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self):
        """Start the interactive detection loop. Blocks until the user presses Q."""
        self._open()
        self._print_help()

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
                key = cv2.waitKey(30) & 0xFF

            if key is not None:
                if self._handle_key(key):
                    break  # quit

            self._draw_paused_frame()

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
                cv2.imshow(self.window_name, self._last_frame)
            return None

        t0 = time.time()

        # Pose detection
        results = self.model(frame, verbose=False, conf=self.model_conf)
        kp = results[0].keypoints if results[0].keypoints is not None else None

        # State machine
        pointing_info = self.sm.update(kp)

        # Render pipeline
        annotated = viz.draw_pose(frame, results)
        viz.draw_annotations(annotated, self.sm.regions, self.sm.lines)
        annotated = viz.draw_status_overlay(
            annotated, self.sm.action_sequence,
            self.sm.current_idx, pointing_info or self.sm.pointing_info)

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
            self.sm.reset()
            print("状态机已重置到动作1")

        elif key == ord('r') and self._paused and self._last_frame is not None:
            print(f"请用鼠标框选第 {len(self.sm.regions) + 1} 个区域，回车确认，Esc 取消...")
            self.sm.regions = select_roi(self.window_name, self._last_frame,
                                          self.sm.regions)

        elif key == ord('l') and self._paused and self._last_frame is not None:
            self.sm.lines = draw_line_interactive(self.window_name, self._last_frame,
                                                    self.sm.lines)

        elif key == ord('s') and self._paused:
            cur = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            save_annotations(self.annotations_file, self.sm.regions, self.sm.lines,
                             self.video_path, cur, self.width, self.height)
            print(f"✅ 已保存 {len(self.sm.regions)} 个区域 + {len(self.sm.lines)} 条参考线 -> {self.annotations_file}")

        elif key == ord('d') and self._paused:
            self.sm.regions = remove_last_region(self.sm.regions)

        return False

    def _handle_seek(self):
        """Seek to the trackbar position."""
        self._user_seeking = False
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self._trackbar_pos)
        ret, frame = self.cap.read()
        if ret:
            results = self.model(frame, verbose=False, conf=self.model_conf)
            self._last_frame = viz.draw_pose(frame, results)
            cv2.imshow(self.window_name, self._last_frame)

    def _draw_paused_frame(self):
        """Re-render the paused frame with annotation overlays."""
        if self._paused and self._last_frame is not None:
            paused_frame = self._last_frame.copy()
            viz.draw_annotations(paused_frame, self.sm.regions, self.sm.lines)
            viz.draw_pause_indicator(paused_frame)
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
        print("  Z    = 重置动作状态机到第1步")
