"""End-to-end detection, tracking, and annotation pipeline."""

from __future__ import annotations

import time
from collections import deque
from typing import Any

import numpy as np

from src.annotator import FrameAnnotator
from src.detector import ObjectDetector
from src.tracker import ObjectTracker


class DetectionTrackingPipeline:
    """Compose detector, tracker, and annotator into one frame processor."""

    def __init__(
        self,
        model_size: str = "n",
        confidence_threshold: float = 0.3,
        classes: list[str] | None = None,
        theme: str = "dark",
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
    ) -> None:
        """Initialize pipeline components and runtime state."""
        self.confidence_threshold = confidence_threshold
        self.classes = classes

        self.detector = ObjectDetector(model_size=model_size)
        self.tracker = ObjectTracker(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
        )
        self.annotator = FrameAnnotator(theme=theme)

        self._fps_window: deque[float] = deque(maxlen=30)
        self._total_frames_processed = 0
        self._last_tracked = None

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Run detect -> track -> annotate on a single frame."""
        start = time.perf_counter()

        detections = self.detector.detect(
            frame,
            confidence_threshold=self.confidence_threshold,
            classes=self.classes,
        )
        tracked = self.tracker.update(detections)

        elapsed = max(time.perf_counter() - start, 1e-6)
        self._fps_window.append(1.0 / elapsed)
        self._total_frames_processed += 1
        self._last_tracked = tracked

        class_names = self.detector.model.names
        return self.annotator.annotate(
            frame=frame,
            detections=tracked,
            fps=self.current_fps,
            class_names=class_names,
        )

    @property
    def current_fps(self) -> float:
        """Return rolling-average FPS over the last 30 processed frames."""
        if not self._fps_window:
            return 0.0
        return float(sum(self._fps_window) / len(self._fps_window))

    def get_stats(self) -> dict[str, Any]:
        """Return runtime summary statistics for the pipeline."""
        active_tracks = 0
        if self._last_tracked is not None and self._last_tracked.tracker_id is not None:
            active_tracks = int(np.sum(self._last_tracked.tracker_id >= 0))

        return {
            "current_fps": self.current_fps,
            "total_frames_processed": self._total_frames_processed,
            "active_tracks": active_tracks,
            "total_unique_tracks": self.tracker.get_track_count(),
        }
