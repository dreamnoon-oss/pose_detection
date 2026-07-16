"""Parallel multi-rule detector for driver action recognition.

Each detection rule runs independently every frame with its own hold counter
and cooldown. When a rule's counter reaches ``hold_frames``, a timestamped
event is emitted and the rule enters cooldown so the same gesture isn't
split into multiple events.
"""

from . import detection as det


class ParallelDetector:
    """Run multiple detection rules independently, recording timestamped events.

    Set ``enabled = False`` to pause detection (e.g. before train arrival).
    When disabled, ``update()`` returns empty results and does not accumulate.
    """

    def __init__(self, rules, regions, lines, *,
                 hold_frames=15, frame_decay=2, cooldown_frames=45,
                 detection_kwargs=None):
        self.rules = rules
        self.regions = regions or []
        self.lines = lines or []
        self.hold_frames = hold_frames
        self.frame_decay = frame_decay
        self.cooldown_frames = cooldown_frames
        self.detection_kwargs = detection_kwargs or {}

        # Per-rule state
        self.hold_counters = {r['name']: 0 for r in rules}
        self.cooldown_counters = {r['name']: 0 for r in rules}
        self.frame_number = 0
        self.events = []
        self.enabled = True

        # Per-rule quality tracking (reset on event fire or hold→0)
        self._hit_counts = {r['name']: 0 for r in rules}
        self._first_hit_frames = {r['name']: 0 for r in rules}
        self._conf_sums = {r['name']: [0.0, 0.0, 0.0] for r in rules}  # [s, f, e]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self):
        """Clear all counters and events."""
        for name in self.hold_counters:
            self.hold_counters[name] = 0
            self.cooldown_counters[name] = 0
            self._hit_counts[name] = 0
            self._first_hit_frames[name] = 0
            self._conf_sums[name] = [0.0, 0.0, 0.0]
        self.frame_number = 0
        self.events.clear()

    def enable(self):
        """Enable detection and reset all hold/cooldown counters (events preserved)."""
        self.enabled = True
        for name in self.hold_counters:
            self.hold_counters[name] = 0
            self.cooldown_counters[name] = 0
            self._hit_counts[name] = 0
            self._first_hit_frames[name] = 0
            self._conf_sums[name] = [0.0, 0.0, 0.0]

    def update(self, keypoints_obj):
        """Process one frame against all rules.

        Does nothing when ``self.enabled`` is False.

        Returns:
            ``(active, new_events)`` where *active* is a dict of rule_name →
            hit-info for rules currently accumulating, and *new_events* is a
            list of events that just completed this frame.
        """
        self.frame_number += 1
        active = {}
        new_events = []

        if not self.enabled:
            return active, new_events

        for rule in self.rules:
            name = rule['name']

            # Cooldown phase — ignore hits until it expires
            if self.cooldown_counters[name] > 0:
                self.cooldown_counters[name] -= 1
                continue

            is_hit, side, angle, wrist, shoulder, eff_thresh, kp_confs = \
                self._detect(rule, keypoints_obj)

            if is_hit:
                if self.hold_counters[name] == 0:
                    self._first_hit_frames[name] = self.frame_number
                self.hold_counters[name] += 1
                self._hit_counts[name] += 1
                if kp_confs:
                    s_c, f_c, e_c = kp_confs
                    self._conf_sums[name][0] += s_c
                    self._conf_sums[name][1] += f_c
                    self._conf_sums[name][2] += e_c

                active[name] = {
                    'rule': name,
                    'side': side,
                    'angle': angle,
                    'wrist': wrist,
                    'shoulder': shoulder,
                    'hold': self.hold_counters[name],
                    'required': self.hold_frames,
                }

                if self.hold_counters[name] >= self.hold_frames:
                    total_frames = self.frame_number - self._first_hit_frames[name] + 1 \
                        if self._first_hit_frames[name] > 0 else self.hold_frames
                    hit_rate = self._hit_counts[name] / total_frames
                    n_hits = max(self._hit_counts[name], 1)
                    avg_conf = sum(self._conf_sums[name]) / (n_hits * 3)
                    margin = eff_thresh - angle if eff_thresh is not None else None

                    event = {
                        'rule': name,
                        'frame': self.frame_number,
                        'side': side,
                        'angle': angle,
                        'wrist': wrist,
                        'shoulder': shoulder,
                        'conf': round(avg_conf, 3),
                        'hit_rate': round(hit_rate, 3),
                        'margin': round(margin, 1) if margin is not None else None,
                    }
                    self.events.append(event)
                    new_events.append(event)
                    self.hold_counters[name] = 0
                    self.cooldown_counters[name] = self.cooldown_frames
                    self._hit_counts[name] = 0
                    self._first_hit_frames[name] = 0
                    self._conf_sums[name] = [0.0, 0.0, 0.0]
            else:
                self.hold_counters[name] = max(
                    0, self.hold_counters[name] - self.frame_decay)
                if self.hold_counters[name] == 0:
                    self._hit_counts[name] = 0
                    self._first_hit_frames[name] = 0
                    self._conf_sums[name] = [0.0, 0.0, 0.0]

        return active, new_events

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _detect(self, rule, keypoints_obj):
        """Route to the correct detection function based on rule type."""
        rtype = rule['type']
        kw = self.detection_kwargs  # shorthand

        if rtype == 'parallel_line':
            line = self._get_line(rule.get('ref_line'))
            if line is None:
                return False, None, None, None, None, None, None
            return det.check_arm_parallel_to_line(
                keypoints_obj, line,
                min_arm_len=kw.get('min_arm_len', 30),
                angle_threshold=kw.get('angle_threshold', 40),
                allow_elbow=rule.get('allow_elbow', False),
                min_arm_torso_angle=rule.get('min_arm_torso_angle',
                    kw.get('min_arm_torso_angle', 0.0)),
                dynamic_angle=rule.get('dynamic_angle', False),
            )

        elif rtype == 'pass_region':
            region = self._get_region(rule.get('target_region'))
            if region is None:
                return False, None, None, None, None, None, None
            is_hit, side, angle, wrist, shoulder = det.check_arm_passes_region(
                keypoints_obj, region,
                min_arm_len=kw.get('min_arm_len', 30),
                extend_ray=kw.get('extend_ray', True),
            )
            return is_hit, side, angle, wrist, shoulder, None, None

        elif rtype == 'pointing':
            region = self._get_region(rule.get('target_region'))
            if region is None:
                return False, None, None, None, None, None, None
            is_hit, side, angle, wrist, shoulder = det.check_pointing(
                keypoints_obj, region,
                min_arm_len=kw.get('min_arm_len', 30),
                angle_threshold=kw.get('angle_threshold', 30),
            )
            return is_hit, side, angle, wrist, shoulder, None, None

        elif rtype == 'pointing_with_line':
            region = self._get_region(rule.get('target_region'))
            line = self._get_line(rule.get('ref_line'))
            if region is None or line is None:
                return False, None, None, None, None, None, None
            is_hit, side, angle, wrist, shoulder = det.check_pointing_with_line(
                keypoints_obj, region, line,
                min_arm_len=kw.get('min_arm_len', 30),
                line_angle_threshold=kw.get('line_angle_threshold', 40),
                loose_angle_threshold=kw.get('loose_angle_threshold', 55),
            )
            return is_hit, side, angle, wrist, shoulder, None, None

        return False, None, None, None, None, None, None

    def _get_region(self, name):
        for r in self.regions:
            if r['name'] == name:
                return r['xywh']
        return None

    def _get_line(self, name):
        for ln in self.lines:
            if ln['name'] == name:
                return ln['pts']
        return None
