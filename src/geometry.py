"""Geometry utilities for pose-based action detection."""

import math


def angle_between(v1, v2):
    """Return the angle in degrees between two 2D vectors."""
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 < 0.01 or n2 < 0.01:
        return 180.0
    return math.degrees(math.acos(max(-1.0, min(1.0, dot / (n1 * n2)))))


def min_angle_to_rect(wrist, arm_dir, rect):
    """Return the minimum angle from arm direction to any corner/center of a rectangle."""
    rx, ry, rw, rh = rect
    corners = [
        (rx, ry),
        (rx + rw, ry),
        (rx, ry + rh),
        (rx + rw, ry + rh),
        (rx + rw / 2, ry + rh / 2),
    ]
    min_ang = 180.0
    for cx, cy in corners:
        to_corner = (cx - wrist[0], cy - wrist[1])
        ang = angle_between(arm_dir, to_corner)
        if ang < min_ang:
            min_ang = ang
    return min_ang


def segments_intersect(p1, p2, p3, p4):
    """Check whether line segments p1-p2 and p3-p4 intersect."""
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True

    # Edge case: endpoint lies exactly on the other segment
    if abs(d1) < 0.01:
        if min(p3[0], p4[0]) <= p1[0] <= max(p3[0], p4[0]) and \
           min(p3[1], p4[1]) <= p1[1] <= max(p3[1], p4[1]):
            return True
    if abs(d2) < 0.01:
        if min(p3[0], p4[0]) <= p2[0] <= max(p3[0], p4[0]) and \
           min(p3[1], p4[1]) <= p2[1] <= max(p3[1], p4[1]):
            return True
    return False
