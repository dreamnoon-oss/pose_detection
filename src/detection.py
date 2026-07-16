"""Detection algorithms for driver action recognition.

Each function receives ``keypoints_obj`` (ultralytics Keypoints, multi-person) and
iterates over every detected person + arm side, returning on the first match.
"""

import math

from .config import CONF_THRESHOLD, ARM_SIDES
from .geometry import angle_between, min_angle_to_rect, segments_intersect


# ---------------------------------------------------------------------------
# Approach A: arm parallel to reference line (+ optional elbow fallback)
# ---------------------------------------------------------------------------

def check_arm_parallel_to_line(keypoints_obj, line_pts, *,
                               min_arm_len=30, angle_threshold=40,
                               allow_elbow=False,
                               min_arm_torso_angle=0.0,
                               dynamic_angle=False):
    """Check whether any person's arm is roughly parallel to a reference line.

    Args:
        keypoints_obj: ultralytics ``Keypoints`` (multi-person).
        line_pts: ``[(x1,y1), (x2,y2)]`` defining the reference line.
        min_arm_len: minimum arm pixel length to consider.
        angle_threshold: maximum angle (degrees) between arm and line to count
            as parallel. When *dynamic_angle* is True and elbow+wrist are valid,
            the effective threshold is ``angle_threshold + arm_bend``.
        allow_elbow: if True and wrist is below confidence, fall back to
            shoulder→elbow (no dynamic adjustment in fallback mode).
        min_arm_torso_angle: if > 0, the arm→torso angle (shoulder→wrist vs
            shoulder→hip, same side) must exceed this value. Default 0 = no check.
        dynamic_angle: if True, compensate for 2D foreshortening by adding the
            elbow-bend angle ``(shoulder→elbow vs elbow→wrist)`` to the threshold.

    Returns:
        ``(is_parallel, side, angle, far_point, shoulder)``
    """
    if keypoints_obj is None or line_pts is None:
        return False, None, None, None, None

    line_dir = (line_pts[1][0] - line_pts[0][0],
                line_pts[1][1] - line_pts[0][1])

    _shoulder_to_hip = {5: 11, 6: 12}  # same-side hip for each shoulder

    for xy, conf in _iter_persons(keypoints_obj):
        for shoulder_id, wrist_id, elbow_id, side in ARM_SIDES:
            if conf[shoulder_id] <= CONF_THRESHOLD:
                continue

            shoulder = (float(xy[shoulder_id][0]), float(xy[shoulder_id][1]))

            effective_threshold = angle_threshold
            far_id = None

            if conf[wrist_id] > CONF_THRESHOLD:
                far_id = wrist_id
                if dynamic_angle and conf[elbow_id] > CONF_THRESHOLD:
                    elbow = (float(xy[elbow_id][0]), float(xy[elbow_id][1]))
                    upper = (elbow[0] - shoulder[0], elbow[1] - shoulder[1])
                    lower = (float(xy[wrist_id][0]) - elbow[0],
                             float(xy[wrist_id][1]) - elbow[1])
                    arm_bend = angle_between(upper, lower)
                    effective_threshold = angle_threshold + arm_bend * 0.6
            elif allow_elbow and conf[elbow_id] > CONF_THRESHOLD:
                far_id = elbow_id

            if far_id is None:
                continue

            far_pt = (float(xy[far_id][0]), float(xy[far_id][1]))
            arm_dir = (far_pt[0] - shoulder[0], far_pt[1] - shoulder[1])
            if math.hypot(*arm_dir) <= min_arm_len:
                continue

            ang = angle_between(arm_dir, line_dir)
            if ang >= effective_threshold:
                continue

            # Optional: arm vs torso angle check (same-side shoulder→hip)
            if min_arm_torso_angle > 0:
                hip_id = _shoulder_to_hip.get(shoulder_id)
                if hip_id is not None and conf[hip_id] > CONF_THRESHOLD:
                    hip = (float(xy[hip_id][0]), float(xy[hip_id][1]))
                    torso_dir = (hip[0] - shoulder[0], hip[1] - shoulder[1])
                    torso_ang = angle_between(arm_dir, torso_dir)
                    if torso_ang <= min_arm_torso_angle:
                        continue

            return True, side, ang, far_pt, shoulder

    return False, None, None, None, None


# ---------------------------------------------------------------------------
# Approach B: arm segment passes through a rectangular region
# ---------------------------------------------------------------------------

def check_arm_passes_region(keypoints_obj, region_xywh, *, min_arm_len=30,
                            extend_ray=True):
    """Check whether any person's shoulder→wrist segment crosses a rectangle.

    If *extend_ray* is True (the default), the arm direction is extended 3x
    beyond the wrist so that pointing *toward* the region counts even when
    the hand hasn't reached it yet.

    Returns:
        ``(is_passing, side, angle, wrist, shoulder)``
    """
    if keypoints_obj is None:
        return False, None, None, None, None

    rx, ry, rw, rh = region_xywh
    edges = [
        ((rx, ry), (rx + rw, ry)),
        ((rx, ry + rh), (rx + rw, ry + rh)),
        ((rx, ry), (rx, ry + rh)),
        ((rx + rw, ry), (rx + rw, ry + rh)),
    ]

    for xy, conf in _iter_persons(keypoints_obj):
        for shoulder_id, wrist_id, _elbow_id, side in ARM_SIDES:
            if conf[shoulder_id] <= CONF_THRESHOLD or conf[wrist_id] <= CONF_THRESHOLD:
                continue

            shoulder = (float(xy[shoulder_id][0]), float(xy[shoulder_id][1]))
            wrist = (float(xy[wrist_id][0]), float(xy[wrist_id][1]))
            arm_vec = (wrist[0] - shoulder[0], wrist[1] - shoulder[1])
            arm_len = math.hypot(*arm_vec)
            if arm_len <= min_arm_len:
                continue

            # Build the test segment: shoulder → wrist (optionally extended)
            far_pt = wrist
            if extend_ray:
                # Extend 3× beyond the wrist in the arm direction
                dx = arm_vec[0] / arm_len
                dy = arm_vec[1] / arm_len
                extend_len = arm_len * 6.0
                far_pt = (wrist[0] + dx * extend_len,
                          wrist[1] + dy * extend_len)

            # Endpoint inside region
            if rx <= shoulder[0] <= rx + rw and ry <= shoulder[1] <= ry + rh:
                return True, side, 0.0, wrist, shoulder
            if rx <= far_pt[0] <= rx + rw and ry <= far_pt[1] <= ry + rh:
                return True, side, 0.0, wrist, shoulder

            # Segment (or extended ray) crosses any region edge
            for e1, e2 in edges:
                if segments_intersect(shoulder, far_pt, e1, e2):
                    return True, side, 0.0, wrist, shoulder

    return False, None, None, None, None


# ---------------------------------------------------------------------------
# Approach C: angle-based pointing toward a region (legacy / Baoshan variant)
# ---------------------------------------------------------------------------

def check_pointing(keypoints_obj, region_xywh, *,
                   min_arm_len=30, angle_threshold=30):
    """Check whether any person's arm points toward a region (angle-based).

    Returns:
        ``(is_pointing, side, angle, wrist, shoulder)``
    """
    if keypoints_obj is None:
        return False, None, None, None, None

    for xy, conf in _iter_persons(keypoints_obj):
        for shoulder_id, wrist_id, _elbow_id, side in ARM_SIDES:
            if conf[shoulder_id] <= CONF_THRESHOLD or conf[wrist_id] <= CONF_THRESHOLD:
                continue

            shoulder = (float(xy[shoulder_id][0]), float(xy[shoulder_id][1]))
            wrist = (float(xy[wrist_id][0]), float(xy[wrist_id][1]))
            arm_dir = (wrist[0] - shoulder[0], wrist[1] - shoulder[1])
            if math.hypot(*arm_dir) <= min_arm_len:
                continue

            ang = min_angle_to_rect(wrist, arm_dir, region_xywh)
            if ang < angle_threshold:
                return True, side, ang, wrist, shoulder

    return False, None, None, None, None


def check_pointing_with_line(keypoints_obj, region_xywh, line_pts, *,
                             min_arm_len=30, line_angle_threshold=40,
                             loose_angle_threshold=55):
    """Combined check: arm parallel to line AND roughly toward region.

    Returns:
        ``(is_pointing, side, angle, wrist, shoulder)``
    """
    if keypoints_obj is None or line_pts is None:
        return False, None, None, None, None

    line_dir = (line_pts[1][0] - line_pts[0][0],
                line_pts[1][1] - line_pts[0][1])

    for xy, conf in _iter_persons(keypoints_obj):
        for shoulder_id, wrist_id, _elbow_id, side in ARM_SIDES:
            if conf[shoulder_id] <= CONF_THRESHOLD or conf[wrist_id] <= CONF_THRESHOLD:
                continue

            shoulder = (float(xy[shoulder_id][0]), float(xy[shoulder_id][1]))
            wrist = (float(xy[wrist_id][0]), float(xy[wrist_id][1]))
            arm_dir = (wrist[0] - shoulder[0], wrist[1] - shoulder[1])
            if math.hypot(*arm_dir) <= min_arm_len:
                continue

            ang_to_line = angle_between(arm_dir, line_dir)
            if ang_to_line > line_angle_threshold:
                continue

            ang_to_rect = min_angle_to_rect(wrist, arm_dir, region_xywh)
            if ang_to_rect < loose_angle_threshold:
                return True, side, ang_to_rect, wrist, shoulder

    return False, None, None, None, None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _iter_persons(keypoints_obj):
    """Yield ``(xy, conf)`` numpy arrays for each detected person."""
    for person_idx in range(len(keypoints_obj)):
        kps = keypoints_obj[person_idx]
        xy = kps.xy[0].cpu().numpy()
        conf = kps.conf[0].cpu().numpy()
        yield xy, conf
