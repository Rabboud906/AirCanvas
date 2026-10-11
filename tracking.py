import math
import numpy as np

def point(hand, index, w, h):
    return (max(0, min(w - 1, round(hand[index].x * w))),
            max(0, min(h - 1, round(hand[index].y * h))))

def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def angle(a, b, c):
    u = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    v = np.asarray(c, dtype=float) - np.asarray(b, dtype=float)
    length = np.linalg.norm(u) * np.linalg.norm(v)
    if length < 1e-6:
        return 0.0
    return math.degrees(math.acos(float(np.clip(np.dot(u, v) / length, -1, 1))))

def finger_up(hand, base, middle, tip, w, h):
    wrist = point(hand, 0, w, h)
    joint = point(hand, middle, w, h)
    end = point(hand, tip, w, h)
    return (angle(point(hand, base, w, h), joint, end) > 155
            and distance(end, wrist) > distance(joint, wrist) * 1.07)

def gesture(hand, w, h):
    index = finger_up(hand, 5, 6, 8, w, h)
    middle = finger_up(hand, 9, 10, 12, w, h)
    ring = finger_up(hand, 13, 14, 16, w, h)
    pinky = finger_up(hand, 17, 18, 20, w, h)
    if index and middle and ring and pinky:
        return "ERASE"
    if index and middle and not ring and not pinky:
        return "SELECT"
    if index and not middle and not ring and not pinky:
        return "DRAW"
    return "PAUSE"
