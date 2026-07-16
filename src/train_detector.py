"""Train arrival/departure detection via background frame differencing."""

import cv2
import numpy as np


class TrainDetector:
    """Detect train presence by comparing ROI against a saved background frame.

    Simple consecutive-frame counter: MAD above threshold increments arrival
    hold, below threshold resets to 0.  No decay — strictly consecutive.
    """

    def __init__(self, background_path, roi_xywh, *,
                 fps=1.0, high_threshold=20, low_threshold=15,
                 arrive_frames=20, depart_frames=20):
        self.background = cv2.imread(background_path)
        if self.background is None:
            raise FileNotFoundError(f"Cannot read background: {background_path}")
        self.roi = roi_xywh
        self.fps = fps
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.arrive_frames = arrive_frames
        self.depart_frames = depart_frames

        self.state = 'AWAY'
        self.frame_num = 0
        self.mad = 0.0
        self.hold_counter = 0
        self.hold_target = arrive_frames  # current target (may change on state switch)
        self.events = []

    def update(self, frame):
        """Process one frame. Returns ``(state, mad)``."""
        self.frame_num += 1

        x, y, w, h = self.roi
        fh, fw = frame.shape[:2]
        x, y = max(0, min(x, fw - 1)), max(0, min(y, fh - 1))
        w, h = max(1, min(w, fw - x)), max(1, min(h, fh - y))

        roi_cur = frame[y:y + h, x:x + w]
        roi_bg = self.background[y:y + h, x:x + w]

        self.mad = float(np.mean(np.abs(
            roi_cur.astype(float) - roi_bg.astype(float))))

        if self.state == 'AWAY':
            if self.mad > self.high_threshold:
                self.hold_counter += 1
                self.hold_target = self.arrive_frames
                if self.hold_counter >= self.arrive_frames:
                    self.state = 'PRESENT'
                    ts = self.frame_num / self.fps if self.fps else 0
                    self.events.append((self.frame_num, ts, 'arrived'))
                    print(f">>> 列车到站: {ts:.1f}s (frame {self.frame_num})")
                    self.hold_counter = 0
        else:
            if self.mad < self.low_threshold:
                self.hold_counter += 1
                self.hold_target = self.depart_frames
                if self.hold_counter >= self.depart_frames:
                    self.state = 'AWAY'
                    ts = self.frame_num / self.fps if self.fps else 0
                    self.events.append((self.frame_num, ts, 'departed'))
                    print(f">>> 列车离站: {ts:.1f}s (frame {self.frame_num})")
                    self.hold_counter = 0

        return self.state, self.mad

    def summary(self, current_time=None):
        """Return human-readable arrival/departure summary."""
        lines = ["=" * 50,
                 "  列车进出站检测",
                 "=" * 50]
        for _frame, ts, etype in self.events:
            label = "列车到站" if etype == 'arrived' else "列车离站"
            lines.append(f"  {label}: {ts:.1f}s")

        if not self.events:
            if self.state == 'PRESENT' and current_time is not None:
                lines.append(f"  列车在场 (退出时 @ {current_time:.1f}s)")
            else:
                lines.append("  未检测到列车进出站事件")
        elif self.events[-1][2] == 'arrived':
            if current_time is not None:
                lines.append(f"  ---> 停靠时段: {self.events[-1][1]:.1f}s ~ "
                             f"{current_time:.1f}s (退出时列车仍在站内)")
            else:
                lines.append("  ---> 列车仍在站内（未检测到离站）")
        else:
            t_arrive = next(ts for f, ts, e in self.events if e == 'arrived')
            t_depart = next(ts for f, ts, e in self.events if e == 'departed')
            lines.append(f"  ---> 停靠时段: {t_arrive:.1f}s ~ {t_depart:.1f}s")
        lines.append("=" * 50)
        return "\n".join(lines)

    def reset(self, frame_num=0):
        """Reset internal state (call after seeking)."""
        self.state = 'AWAY'
        self.frame_num = frame_num
        self.mad = 0.0
        self.hold_counter = 0
        self.hold_target = self.arrive_frames

    @property
    def status_label(self):
        labels = {
            'AWAY': '轨道空闲',
            'PRESENT': '列车在场',
        }
        return labels.get(self.state, self.state)

    @property
    def train_info(self):
        """Return structured arrival/departure info for reporting."""
        info = {"arrive": None, "depart": None, "duration": None}
        for _frame, ts, etype in self.events:
            if etype == 'arrived':
                info["arrive"] = f"{ts:.1f}s"
            elif etype == 'departed':
                info["depart"] = f"{ts:.1f}s"
        if info["arrive"] and info["depart"]:
            t_arrive = next(ts for f, ts, e in self.events if e == 'arrived')
            t_depart = next(ts for f, ts, e in self.events if e == 'departed')
            info["duration"] = f"{t_depart - t_arrive:.1f}s"
        return info
