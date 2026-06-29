"""Configurable action state machine for sequential driver action detection.

Each action in the sequence defines:
    - name: display label
    - type: detection strategy ("parallel_line" | "pass_region" | "pointing" | "pointing_with_line")
    - ref_line: reference line name (for parallel_line / pointing_with_line)
    - target_region: region name (for pass_region / pointing*)
    - allow_elbow: elbow fallback flag (parallel_line only)
"""

from . import detection as det


class ActionStateMachine:
    """Sequential action detector with hold-frame confirmation."""

    def __init__(self, action_sequence, regions=None, lines=None, *,
                 hold_frames=15, frame_decay=2, detection_kwargs=None):
        self.action_sequence = action_sequence
        self.regions = regions or []
        self.lines = lines or []
        self.hold_frames = hold_frames
        self.frame_decay = frame_decay
        self.detection_kwargs = detection_kwargs or {}

        self.current_idx = 0
        self.hold_counter = 0
        self.pointing_info = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self):
        """Reset the state machine to the first action."""
        self.current_idx = 0
        self.hold_counter = 0
        self.pointing_info = None

    @property
    def is_complete(self):
        return self.current_idx >= len(self.action_sequence)

    def update(self, keypoints_obj):
        """Process one frame of keypoints data.

        Args:
            keypoints_obj: ultralytics ``Keypoints`` or None.

        Returns:
            ``pointing_info`` dict or None.
        """
        if self.is_complete:
            self.pointing_info = None
            return None

        target = self.action_sequence[self.current_idx]
        action_type = target.get("type", "parallel_line")
        is_hit = False
        side = angle = wrist = shoulder = None
        region_name = "?"

        if action_type == "parallel_line":
            line = self._get_line(target.get("ref_line"))
            if line is not None:
                is_hit, side, angle, wrist, shoulder = det.check_arm_parallel_to_line(
                    keypoints_obj, line,
                    allow_elbow=target.get("allow_elbow", False),
                    **self.detection_kwargs,
                )
                region_name = target.get("ref_line", "?")

        elif action_type == "pass_region":
            region = self._get_region(target.get("target_region"))
            if region is not None:
                is_hit, side, angle, wrist, shoulder = det.check_arm_passes_region(
                    keypoints_obj, region,
                    **self.detection_kwargs,
                )
                region_name = target.get("target_region", "?")

        elif action_type == "pointing":
            region = self._get_region(target.get("target_region"))
            if region is not None:
                is_hit, side, angle, wrist, shoulder = det.check_pointing(
                    keypoints_obj, region,
                    **self.detection_kwargs,
                )
                region_name = target.get("target_region", "?")

        elif action_type == "pointing_with_line":
            region = self._get_region(target.get("target_region"))
            line = self._get_line(target.get("ref_line"))
            if region is not None and line is not None:
                is_hit, side, angle, wrist, shoulder = det.check_pointing_with_line(
                    keypoints_obj, region, line,
                    **self.detection_kwargs,
                )
                region_name = target.get("target_region", "?")

        # --- update counters ---
        if is_hit:
            self.hold_counter += 1
            self.pointing_info = {
                "action_name": target["name"],
                "region_name": region_name,
                "hold": self.hold_counter,
                "required": self.hold_frames,
                "angle": angle,
                "side": side,
                "wrist": wrist,
                "shoulder": shoulder,
            }
            if self.hold_counter >= self.hold_frames:
                self.current_idx += 1
                self.hold_counter = 0
                if self.is_complete:
                    self.pointing_info = None
                return self.pointing_info
        else:
            self.hold_counter = max(0, self.hold_counter - self.frame_decay)
            self.pointing_info = None

        return self.pointing_info

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_region(self, name):
        for r in self.regions:
            if r["name"] == name:
                return r["xywh"]
        return None

    def _get_line(self, name):
        for ln in self.lines:
            if ln["name"] == name:
                return ln["pts"]
        return None
