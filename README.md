# Air Canvas : Draw in the Air with Hand Gestures

Air Canvas is a Python computer-vision drawing application that lets you draw and edit artwork by moving your hand in front of a webcam. It uses a pretrained **MediaPipe Hand Landmarker** model to detect a hand and its 21 landmarks, and **OpenCV** to display the camera feed, toolbar, and drawings. No custom model training is required.

## Features

- **Gesture-controlled drawing:** Draw with one raised index finger, navigate/select with two fingers, and erase strokes by moving an open palm.
- **Customizable brush:** Choose blue, red, green, yellow, white, or a changing rainbow color. Adjust brush width from **2 to 40 pixels** using a number-line slider.
- **Shape tools:** Draw freehand strokes, straight lines, rectangles, and circles.
- **Smoother tracking:** A moving-average filter and exponential smoothing reduce fingertip jitter; gesture confirmation helps avoid accidental mode switches.
- **Editing:** Undo, redo, or clear the canvas.
- **Viewing and exporting:** Toggle between live camera and whiteboard backgrounds, then save drawings as PNG files.
- **Detection feedback:** A hand bounding box, gesture/mode label, brush width, and approximate FPS appear on screen. If no hand is detected, drawing pauses.

## Demo

Add a short demo video or GIF here before submitting. Show drawing, changing colors and brush width, making a shape, open-palm erasing, and saving a drawing. You can also include an example PNG in `output/`.

<!-- Example: ![Air Canvas demo](assets/demo.gif) -->

## Requirements

- Python 3.10+ with a version supported by the installed MediaPipe release
- A working webcam
- macOS, Windows, or Linux with a desktop environment capable of opening an OpenCV window
- Dependencies: `mediapipe`, `opencv-python`, `numpy`

## Installation

From the `AirCanvas` project folder:

**1. Create and activate a virtual environment.**

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

**2. Install dependencies.**

```bash
python -m pip install -r requirements.txt
```

**3. Download the pretrained hand-detection model.**

Create the model folder (if it does not exist), then download the MediaPipe Hand Landmarker model. From the project root on macOS / Linux:

```bash
mkdir -p models
curl -L "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" -o models/hand_landmarker.task
```

On Windows, create `models/` and download that URL into `models/hand_landmarker.task` using a browser.

Your project should look like:

```text
AirCanvas/
├── main.py
├── tracking.py
├── drawing.py
├── toolbar.py
├── settings.py
├── requirements.txt
├── models/
│   └── hand_landmarker.task
└── output/                 # Created when saving
```

**4. Start the application.**

```bash
python main.py
```

Allow webcam access if your operating system asks. Press **Q** in the Air Canvas window to quit.

## Controls

| Input | Action |
| --- | --- |
| ☝️ One raised index finger | Draw using the selected tool |
| ✌️ Raised index and middle fingers | Move/select without drawing |
| ✌️ Over a color or tool button | Hover briefly to select it |
| ✌️ Over the brush-size number line | Move horizontally to choose a width from 2 to 40 px |
| ✋ Open palm | Erase strokes near your palm |
| Other gestures / no hand | Pause drawing |
| **Z** | Undo |
| **Y** | Redo |
| **B** | Toggle webcam / whiteboard background |
| **C** | Clear all drawings |
| **S** | Save PNG into `output/` |
| **Q** | Quit |

To draw a shape, select **LINE**, **RECT**, or **CIRCLE** with two fingers, then use one finger to set its starting point and move to its endpoint. Switch out of drawing mode to commit the shape. Select **PEN** to return to freehand drawing.

## How it works

1. **Capture:** OpenCV reads and mirrors each webcam frame.
2. **Detect:** MediaPipe's pretrained model finds hand landmarks. The app uses the index fingertip (landmark 8) to control the cursor.
3. **Recognize gestures:** Joint positions and angles are used to classify DRAW, SELECT, ERASE, and PAUSE. The gesture must remain consistent for multiple frames before a mode change.
4. **Smooth motion:** Recent fingertip positions are averaged and smoothed to reduce jitter.
5. **Interact:** The app maps gestures and fingertip positions to drawing operations, toolbar selections, or palm erasing.
6. **Render:** OpenCV composites strokes over either the webcam or a white background. Operations are stored so they can be replayed for undo/redo and export.

The custom logic is in **gesture recognition, filtering, interaction modes, toolbar controls, stroke rendering, shapes, and editing**, not in training the hand-detection model.

## Project organization

| File | Responsibility |
| --- | --- |
| `main.py` | Webcam loop, application state, event handling, keyboard shortcuts |
| `tracking.py` | Hand landmark coordinates and gesture classification |
| `drawing.py` | Stroke/shape rendering, rainbow colors, erasing, compositing |
| `toolbar.py` | Color/tool buttons and interactive brush-size number line |
| `settings.py` | Configurable tracking, gesture, and UI parameters |
| `requirements.txt` | Required Python packages |

## Limitations and potential improvements

- Hand detection may be less reliable in poor lighting, with fast movements, or when fingers leave the camera frame.
- Recognition uses hand-landmark heuristics, so unusual orientations or partially hidden fingers may cause incorrect mode changes.
- Smoothing reduces jitter but also introduces some cursor delay.
- Only one hand is tracked at a time.
- The program is intended for a local desktop with an available webcam, not a browser or headless server.

Potential improvements include gesture calibration, mouse-free save/undo controls, pressure-like brush effects, and automated tests for gesture and slider calculations.

## Troubleshooting

**`Cannot access webcam`:** Verify camera permissions and close other applications using the webcam. Try changing `cv2.VideoCapture(0)` to another camera index if needed.

**Model file not found:** Make sure `models/hand_landmarker.task` exists and you launched the program from the project root.

**MediaPipe prints XNNPACK or feedback-manager messages:** These can be informational; investigate the actual Python traceback if the program stops.

## Acknowledgments

Built with [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker), [OpenCV](https://opencv.org/), and [NumPy](https://numpy.org/).
