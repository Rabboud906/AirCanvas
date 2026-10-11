"""Air Canvas entry point. Run: python main.py"""
from collections import deque
from pathlib import Path
import time

import cv2
import mediapipe as mp
import numpy as np

from settings import (MODEL_PATH, TOOLBAR_HEIGHT, BUTTON_HEIGHT, SMOOTHING,
                      AVERAGE_FRAMES, STABLE_FRAMES, HOVER_SECONDS)
from tracking import point, distance, gesture
from drawing import rainbow_color, draw_operation, render_strokes, blend
from toolbar import make_buttons, draw_toolbar, slider_hit, slider_to_size

def main():
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not camera.isOpened():
        raise RuntimeError("Cannot access webcam")

    history = deque(maxlen=AVERAGE_FRAMES)
    smooth = None
    committed = []
    redo_stack = []
    active = None
    current_mode = "PAUSE"
    candidate = "PAUSE"
    stable_count = 0
    current_color = (255, 0, 0)
    current_width = 7
    current_tool = "pen"
    rainbow = False
    whiteboard = False
    hovered = None
    hover_at = None
    selection_locked = False
    started = time.monotonic()
    previous_timestamp = -1
    fps_prev = time.monotonic()
    fps = 0.0
    buttons = None
    paint = None
    mask = None

    try:
        with mp.tasks.vision.HandLandmarker.create_from_options(options) as detector:
            while True:
                ok, frame = camera.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                if paint is None:
                    paint, mask = render_strokes(frame.shape, committed)
                    buttons = make_buttons(w)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp = max(int((time.monotonic() - started) * 1000), previous_timestamp + 1)
                previous_timestamp = timestamp
                result = detector.detect_for_video(image, timestamp)
                cursor = None
                eraser_preview = None
                bbox = None
                raw_mode = "PAUSE"
                operation_changed = False

                if result.hand_landmarks:
                    hand = result.hand_landmarks[0]
                    hand_points = [point(hand, i, w, h) for i in range(21)]
                    bbox = (max(0, min(p[0] for p in hand_points) - 12),
                            max(TOOLBAR_HEIGHT, min(p[1] for p in hand_points) - 12),
                            min(w - 1, max(p[0] for p in hand_points) + 12),
                            min(h - 1, max(p[1] for p in hand_points) + 12))
                    raw = point(hand, 8, w, h)
                    history.append(raw)
                    avg = np.mean(np.asarray(history, dtype=float), axis=0)
                    smooth = avg if smooth is None else smooth * (1 - SMOOTHING) + avg * SMOOTHING
                    cursor = tuple(np.rint(smooth).astype(int))
                    raw_mode = gesture(hand, w, h)
                else:
                    history.clear()
                    smooth = None

                if raw_mode == candidate:
                    stable_count += 1
                else:
                    candidate = raw_mode
                    stable_count = 1

                if stable_count >= STABLE_FRAMES and candidate != current_mode:
                    if active is not None:
                        committed.append(active)
                        redo_stack.clear()
                        operation_changed = True
                        active = None
                    current_mode = candidate
                    hovered = None
                    hover_at = None
                    selection_locked = False

                # Pause drawing while a gesture transition is being confirmed.
                effective_mode = current_mode if raw_mode == current_mode else "TRANSITION"

                if effective_mode == "DRAW" and cursor is not None and cursor[1] >= TOOLBAR_HEIGHT:
                    if active is None:
                        active = {"kind": "rainbow" if rainbow and current_tool == "pen" else current_tool,
                                  "color": current_color, "width": current_width,
                                  "points": [cursor], "colors": []}
                    else:
                        active["points"].append(cursor)
                        if active["kind"] == "rainbow":
                            active["colors"].append(rainbow_color())
                elif active is not None and effective_mode not in ("DRAW", "ERASE", "TRANSITION"):
                    committed.append(active)
                    redo_stack.clear()
                    operation_changed = True
                    active = None

                if effective_mode == "ERASE" and result.hand_landmarks:
                    hand = result.hand_landmarks[0]
                    palm_ids = (0, 5, 9, 13, 17)
                    center = tuple(np.rint(np.mean([point(hand, j, w, h) for j in palm_ids], axis=0)).astype(int))
                    radius = max(22, int(distance(point(hand, 5, w, h), point(hand, 17, w, h)) * 0.85))
                    eraser_preview = (center, radius)
                    if center[1] >= TOOLBAR_HEIGHT:
                        if active is None or active["kind"] != "erase":
                            active = {"kind": "erase", "color": (0, 0, 0),
                                      "width": radius * 2, "points": [center]}
                        else:
                            active["points"].append(center)
                elif effective_mode == "SELECT" and cursor is not None:
                    # Brush width follows the fingertip on the number line.
                    if slider_hit(cursor, w):
                        current_width = slider_to_size(cursor[0], w)

                    index = next((i for i, item in enumerate(buttons)
                                  if item["rect"][0] <= cursor[0] <= item["rect"][2]
                                  and cursor[1] <= BUTTON_HEIGHT), None)
                    if index is None:
                        hovered, hover_at, selection_locked = None, None, False
                    elif index != hovered:
                        hovered, hover_at, selection_locked = index, time.monotonic(), False
                    elif not selection_locked and time.monotonic() - hover_at >= HOVER_SECONDS:
                        button = buttons[index]
                        if button["kind"] == "color":
                            current_color, rainbow = button["value"], False
                        elif button["kind"] == "rainbow":
                            rainbow = True
                        elif button["kind"] == "tool":
                            current_tool = button["value"]
                        selection_locked = True
                else:
                    hovered, hover_at, selection_locked = None, None, False

                # A new completed stroke or eraser stroke triggers canvas reconstruction.
                if operation_changed:
                    paint, mask = render_strokes(frame.shape, committed)

                # Preview active operation over the committed canvas.
                work_paint = paint.copy()
                work_mask = mask.copy()
                if active is not None:
                    draw_operation(work_paint, work_mask, active)

                background = np.full_like(frame, 255) if whiteboard else frame.copy()
                output = blend(background, work_paint, work_mask)
                draw_toolbar(output, buttons, current_color, current_width,
                             current_tool, rainbow, hovered, hover_at)

                if bbox is not None and bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                    cv2.rectangle(output, bbox[:2], bbox[2:], (60, 230, 60), 2)
                    cv2.putText(output, "HAND", (bbox[0], max(TOOLBAR_HEIGHT + 16, bbox[1] - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 230, 60), 2)
                if eraser_preview is not None:
                    cv2.circle(output, *eraser_preview, (0, 0, 255), 2)
                elif cursor is not None:
                    cv2.circle(output, cursor, 9, (0, 210, 255), 2)

                now = time.monotonic()
                dt = max(now - fps_prev, 0.0001)
                fps = 0.85 * fps + 0.15 * (1 / dt) if fps else 1 / dt
                fps_prev = now
                label = (f"{effective_mode} | {current_tool.upper()} | {current_width}px "
                         f"| FPS {fps:.0f}")
                cv2.putText(output, label, (14, TOOLBAR_HEIGHT + 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.63, (10, 10, 10) if whiteboard else (255, 255, 255), 2)
                cv2.putText(output, "1 finger: draw  |  2: tools  |  palm: erase",
                            (14, h - 43), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (15, 15, 15) if whiteboard else (255, 255, 255), 2)
                cv2.putText(output, "Z: undo  Y: redo  B: whiteboard  C: clear  S: save  Q: quit",
                            (14, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                            (15, 15, 15) if whiteboard else (255, 255, 255), 2)

                cv2.imshow("Air Canvas", output)
                key = cv2.waitKey(1) & 0xFF

                # Commit an in-progress stroke before undo/redo or saving.
                if key in (ord("z"), ord("y"), ord("s"), ord("b")) and active is not None:
                    committed.append(active)
                    redo_stack.clear()
                    active = None
                    paint, mask = render_strokes(frame.shape, committed)

                if key == ord("q"):
                    break
                elif key == ord("c"):
                    committed.clear()
                    redo_stack.clear()
                    active = None
                    paint, mask = render_strokes(frame.shape, committed)
                elif key == ord("z") and committed:
                    redo_stack.append(committed.pop())
                    paint, mask = render_strokes(frame.shape, committed)
                elif key == ord("y") and redo_stack:
                    committed.append(redo_stack.pop())
                    paint, mask = render_strokes(frame.shape, committed)
                elif key == ord("b"):
                    whiteboard = not whiteboard
                elif key == ord("s"):
                    Path("output").mkdir(exist_ok=True)
                    filename = f"output/air_canvas_{time.time_ns()}.png"
                    white = np.full_like(frame, 255)
                    saved = blend(white, paint, mask)
                    if cv2.imwrite(filename, saved):
                        print(f"Saved: {filename}")
                    else:
                        print(f"Could not save: {filename}")
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
