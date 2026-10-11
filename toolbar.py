import time
import cv2
from settings import PALETTE, BUTTON_HEIGHT, TOOLBAR_HEIGHT, HOVER_SECONDS, MIN_BRUSH, MAX_BRUSH


def make_buttons(width):
    items = ([{"name": name, "kind": "color", "value": col} for name, col in PALETTE]
             + [{"name": "RAINBOW", "kind": "rainbow", "value": None}]
             + [{"name": name, "kind": "tool", "value": tool}
                for name, tool in [("PEN", "pen"), ("LINE", "line"),
                                   ("RECT", "rect"), ("CIRCLE", "circle")]])
    for i, item in enumerate(items):
        item["rect"] = (round(i * width / len(items)), 0,
                        round((i + 1) * width / len(items)) - 1, BUTTON_HEIGHT)
    return items


def selected_button(item, color, tool, rainbow):
    kind = item["kind"]
    return ((kind == "color" and not rainbow and item["value"] == color)
            or (kind == "rainbow" and rainbow)
            or (kind == "tool" and item["value"] == tool))


def slider_bounds(width):
    margin = min(80, max(24, width // 12))
    return margin, width - margin, 108


def slider_hit(point, width):
    x1, x2, y = slider_bounds(width)
    return x1 <= point[0] <= x2 and 90 <= point[1] <= 130


def slider_to_size(x, width):
    x1, x2, _ = slider_bounds(width)
    ratio = max(0.0, min(1.0, (x - x1) / max(1, x2 - x1)))
    return round(MIN_BRUSH + ratio * (MAX_BRUSH - MIN_BRUSH))


def draw_toolbar(image, buttons, color, size, tool, rainbow, hovered, hover_at):
    width = image.shape[1]
    cv2.rectangle(image, (0, 0), (width, TOOLBAR_HEIGHT), (35, 35, 35), -1)
    for i, item in enumerate(buttons):
        x1, _, x2, _ = item["rect"]
        fill = item["value"] if item["kind"] == "color" else (72, 72, 72)
        if item["kind"] == "rainbow":
            fill = (170, 85, 220)
        cv2.rectangle(image, (x1 + 2, 6), (x2 - 2, BUTTON_HEIGHT - 8), fill, -1)
        border = (0, 255, 255) if selected_button(item, color, tool, rainbow) else (140, 140, 140)
        cv2.rectangle(image, (x1 + 2, 6), (x2 - 2, BUTTON_HEIGHT - 8), border, 2)
        scale = 0.32 if (x2 - x1) < 80 else 0.42
        tw = cv2.getTextSize(item["name"], cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
        fg = (0, 0, 0) if item["name"] in ("YELLOW", "WHITE", "GREEN") else (255, 255, 255)
        cv2.putText(image, item["name"], (max(x1 + 1, (x1 + x2 - tw) // 2), 43),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, fg, 1, cv2.LINE_AA)
        if hovered == i and hover_at is not None:
            progress = min(1.0, (time.monotonic() - hover_at) / HOVER_SECONDS)
            cv2.rectangle(image, (x1 + 4, BUTTON_HEIGHT - 6),
                          (x1 + 4 + int((x2 - x1 - 8) * progress), BUTTON_HEIGHT - 2),
                          (0, 255, 255), -1)

    # Number-line slider: selectable by holding two fingers and moving left or right.
    x1, x2, y = slider_bounds(width)
    cv2.line(image, (x1, y), (x2, y), (195, 195, 195), 3, cv2.LINE_AA)
    for n in [2, 10, 20, 30, 40]:
        x = round(x1 + (n - MIN_BRUSH) / (MAX_BRUSH - MIN_BRUSH) * (x2 - x1))
        cv2.line(image, (x, y - 7), (x, y + 7), (195, 195, 195), 2)
        cv2.putText(image, str(n), (x - 8, y + 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)
    marker = round(x1 + (size - MIN_BRUSH) / (MAX_BRUSH - MIN_BRUSH) * (x2 - x1))
    cv2.circle(image, (marker, y), 12, (0, 215, 255), -1, cv2.LINE_AA)
    cv2.circle(image, (marker, y), 12, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, f"{size}px", (max(4, min(width - 55, marker - 18)), 89),
                cv2.FONT_HERSHEY_SIMPLEX, 0.47, (255, 255, 255), 1, cv2.LINE_AA)