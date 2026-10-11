import cv2
import numpy as np
from tracking import distance
import time

def rainbow_color():
    hsv = np.uint8([[[int(time.monotonic() * 65) % 180, 255, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return tuple(int(v) for v in bgr)

def draw_operation(paint, mask, op):
    """Replay an operation on BOTH the color image and visibility mask."""
    coords = op["points"]
    if not coords:
        return
    kind = op["kind"]
    erasing = kind == "erase"
    thickness = op["width"]
    color = (0, 0, 0) if erasing else op["color"]
    value = 0 if erasing else 255

    def segment(a, b):
        cv2.line(paint, a, b, color, thickness, cv2.LINE_AA)
        cv2.line(mask, a, b, value, thickness, cv2.LINE_AA)

    if kind in ("pen", "erase", "rainbow"):
        if len(coords) == 1:
            cv2.circle(paint, coords[0], max(1, thickness // 2), color, -1, cv2.LINE_AA)
            cv2.circle(mask, coords[0], max(1, thickness // 2), value, -1, cv2.LINE_AA)
        else:
            for i, (a, b) in enumerate(zip(coords, coords[1:])):
                if kind == "rainbow":
                    shade = op["colors"][i]
                    cv2.line(paint, a, b, shade, thickness, cv2.LINE_AA)
                    cv2.line(mask, a, b, 255, thickness, cv2.LINE_AA)
                else:
                    segment(a, b)
    elif kind == "line":
        segment(coords[0], coords[-1])
    elif kind == "rect":
        cv2.rectangle(paint, coords[0], coords[-1], color, thickness, cv2.LINE_AA)
        cv2.rectangle(mask, coords[0], coords[-1], value, thickness, cv2.LINE_AA)
    elif kind == "circle":
        radius = round(distance(coords[0], coords[-1]))
        cv2.circle(paint, coords[0], radius, color, thickness, cv2.LINE_AA)
        cv2.circle(mask, coords[0], radius, value, thickness, cv2.LINE_AA)

def render_strokes(shape, operations):
    paint = np.zeros(shape, dtype=np.uint8)
    mask = np.zeros(shape[:2], dtype=np.uint8)
    for operation in operations:
        draw_operation(paint, mask, operation)
    return paint, mask

def blend(background, paint, mask):
    alpha = (mask.astype(np.float32) / 255)[:, :, None]
    return (background.astype(np.float32) * (1 - alpha)
            + paint.astype(np.float32) * alpha).astype(np.uint8)
