"""CLI to run live webcam detection and tracking."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import DetectionTrackingPipeline
from src.video_runner import run_on_webcam


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for webcam mode."""
    parser = argparse.ArgumentParser(description="Run realtime detection tracker on webcam.")
    parser.add_argument("--model", choices=["n", "s", "m", "l", "x"], default="n", help="YOLOv8 model size.")
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold.")
    parser.add_argument("--classes", nargs="*", default=None, help="Optional class names to keep.")
    parser.add_argument("--display-width", type=int, default=1280, help="Display width for webcam preview.")
    return parser.parse_args()


def main() -> None:
    """Run webcam tracking from CLI."""
    args = parse_args()

    pipeline = DetectionTrackingPipeline(
        model_size=args.model,
        confidence_threshold=args.conf,
        classes=args.classes,
        theme="dark",
    )
    summary = run_on_webcam(pipeline=pipeline, display_width=args.display_width)

    print("Webcam session ended")
    print(f"Frames: {summary['total_frames']}")
    print(f"Average FPS: {summary['avg_fps']:.2f}")
    print(f"Unique tracked objects: {summary['total_unique_tracks']}")


if __name__ == "__main__":
    main()
