"""Video/webcam execution helpers for the detection-tracking pipeline."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import cv2
from tqdm import tqdm

from src.pipeline import DetectionTrackingPipeline

LOGGER = logging.getLogger(__name__)


def run_on_video(
    input_path: str,
    output_path: str,
    pipeline: DetectionTrackingPipeline,
    show_preview: bool = False,
) -> dict[str, Any]:
    """Process a video file and save an annotated output video.

    Args:
        input_path: Path to source video.
        output_path: Path for annotated video.
        pipeline: Initialized pipeline.
        show_preview: Show a live preview window while processing.

    Returns:
        Summary stats including average FPS and total tracked objects.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open input video: {input_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output_file),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    started_at = time.perf_counter()
    progress_total = total_frames if total_frames > 0 else None

    with tqdm(total=progress_total, desc="Processing video", unit="frame") as pbar:
        while True:
            success, frame = cap.read()
            if not success:
                break

            annotated = pipeline.process_frame(frame)
            writer.write(annotated)

            if show_preview:
                cv2.imshow("Realtime Detection Tracker", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    LOGGER.info("Preview interrupted by user.")
                    break

            pbar.update(1)

    elapsed = max(time.perf_counter() - started_at, 1e-6)
    cap.release()
    writer.release()
    if show_preview:
        cv2.destroyAllWindows()

    stats = pipeline.get_stats()
    avg_fps = stats["total_frames_processed"] / elapsed if stats["total_frames_processed"] else 0.0
    summary = {
        "total_frames": stats["total_frames_processed"],
        "avg_fps": avg_fps,
        "current_fps": stats["current_fps"],
        "total_unique_tracks": stats["total_unique_tracks"],
        "active_tracks": stats["active_tracks"],
        "output_path": str(output_file),
    }
    LOGGER.info(
        "Video run completed. Avg FPS: %.2f | Total tracked objects: %d",
        summary["avg_fps"],
        summary["total_unique_tracks"],
    )
    return summary


def run_on_webcam(
    pipeline: DetectionTrackingPipeline,
    display_width: int = 1280,
) -> dict[str, Any]:
    """Run real-time detection/tracking from webcam until user presses `q`."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Unable to open webcam (index 0).")

    started_at = time.perf_counter()

    while True:
        success, frame = cap.read()
        if not success:
            LOGGER.warning("Webcam frame read failed. Exiting stream loop.")
            break

        if display_width > 0:
            height, width = frame.shape[:2]
            scale = display_width / float(width)
            display_height = int(height * scale)
            frame = cv2.resize(frame, (display_width, display_height), interpolation=cv2.INTER_LINEAR)

        annotated = pipeline.process_frame(frame)
        cv2.imshow("Realtime Detection Tracker (Webcam)", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    elapsed = max(time.perf_counter() - started_at, 1e-6)
    cap.release()
    cv2.destroyAllWindows()

    stats = pipeline.get_stats()
    avg_fps = stats["total_frames_processed"] / elapsed if stats["total_frames_processed"] else 0.0
    summary = {
        "total_frames": stats["total_frames_processed"],
        "avg_fps": avg_fps,
        "current_fps": stats["current_fps"],
        "total_unique_tracks": stats["total_unique_tracks"],
        "active_tracks": stats["active_tracks"],
    }
    LOGGER.info(
        "Webcam run completed. Avg FPS: %.2f | Total tracked objects: %d",
        summary["avg_fps"],
        summary["total_unique_tracks"],
    )
    return summary
