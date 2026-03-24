"""CLI to process a video file and save annotated output."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import DetectionTrackingPipeline
from src.video_runner import run_on_video


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for video processing."""
    parser = argparse.ArgumentParser(description="Run realtime detection tracker on a video file.")
    parser.add_argument("--input", required=True, help="Path to input video file.")
    parser.add_argument("--output", default="outputs/annotated.mp4", help="Output video path.")
    parser.add_argument("--model", choices=["n", "s", "m", "l", "x"], default="n", help="YOLOv8 model size.")
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold.")
    parser.add_argument("--classes", nargs="*", default=None, help="Optional class names to keep.")
    parser.add_argument("--show-preview", action="store_true", help="Show processing preview window.")
    return parser.parse_args()


def main() -> None:
    """Run video processing from CLI."""
    args = parse_args()

    pipeline = DetectionTrackingPipeline(
        model_size=args.model,
        confidence_threshold=args.conf,
        classes=args.classes,
        theme="dark",
    )
    summary = run_on_video(
        input_path=args.input,
        output_path=args.output,
        pipeline=pipeline,
        show_preview=args.show_preview,
    )

    print("Processing complete")
    print(f"Output: {summary['output_path']}")
    print(f"Frames: {summary['total_frames']}")
    print(f"Average FPS: {summary['avg_fps']:.2f}")
    print(f"Unique tracked objects: {summary['total_unique_tracks']}")


if __name__ == "__main__":
    main()
