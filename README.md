# realtime-detection-tracker

A real-time object detection and multi-object tracking pipeline built with YOLOv8, Supervision (ByteTrack + annotators), OpenCV, and Gradio.

The pipeline performs the following stages on each frame:
1. Detection with YOLOv8
2. Tracking/association with ByteTrack
3. Annotation with bounding boxes, labels, traces, and FPS overlay

## Pipeline Architecture

```text
Video Frame
   |
   v
YOLOv8 Detection
   |
   v
ByteTrack Association
   |
   v
Annotated Frame
```

## Project Structure

```text
realtime-detection-tracker/
├── src/
│   ├── detector.py
│   ├── tracker.py
│   ├── annotator.py
│   ├── pipeline.py
│   ├── video_runner.py
│   └── app.py
├── scripts/
│   ├── run_video.py
│   ├── run_webcam.py
│   └── download_sample.py
├── tests/
│   ├── test_detector.py
│   ├── test_tracker.py
│   └── test_pipeline.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Setup

1. Create and activate a Python 3.10+ environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download a demo video:

```bash
python scripts/download_sample.py
```

## Usage

### 1) CLI: Process a Video File

```bash
python scripts/run_video.py --input data/sample.mp4 --output outputs/annotated.mp4 --model n --conf 0.3
```

### 2) CLI: Live Webcam Detection

```bash
python scripts/run_webcam.py --model n --conf 0.3 --display-width 1280
```

### 3) Gradio Web UI

```bash
python -m src.app
```

Then open the local URL shown in the terminal, upload a video, configure settings, and process.

## Example Output

Place your output preview media here.

- GIF placeholder: `docs/demo.gif`
- Screenshot placeholder: `docs/screenshot.png`

## Performance Benchmarks (Placeholder)

| Model Size | Approx FPS (1080p, CPU/GPU dependent) |
|------------|----------------------------------------|
| nano (n)   | TBD                                    |
| small (s)  | TBD                                    |
| medium (m) | TBD                                    |
| large (l)  | TBD                                    |

## Supported Classes (COCO-80)

person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light,
fire hydrant, stop sign, parking meter, bench, bird, cat, dog, horse, sheep, cow,
elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee,
skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard,
tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple,
sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, couch,
potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone,
microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear,
hair drier, toothbrush

## Running Tests

```bash
pytest -q
```

## Notes

- Library modules use logging and avoid direct `print` output.
- CLI scripts print concise run summaries.
- By default, the detector loads `yolov8n.pt` for speed.
