"""Tests for end-to-end pipeline behavior."""

from __future__ import annotations

import numpy as np
import pytest

sv = pytest.importorskip("supervision")

from src import pipeline as pipeline_module


class _FakeDetector:
    def __init__(self, model_size: str = "n") -> None:
        self.model = type("Model", (), {"names": {0: "person"}})()

    def detect(self, frame: np.ndarray, confidence_threshold: float = 0.3, classes=None) -> sv.Detections:
        return sv.Detections(
            xyxy=np.array([[5, 5, 20, 20]], dtype=np.float32),
            confidence=np.array([0.9], dtype=np.float32),
            class_id=np.array([0], dtype=np.int64),
        )


class _FakeTracker:
    def __init__(self, *args, **kwargs) -> None:
        self._count = 0

    def update(self, detections: sv.Detections) -> sv.Detections:
        detections.tracker_id = np.array([1], dtype=np.int64)
        self._count = 1
        return detections

    def get_track_count(self) -> int:
        return self._count


class _FakeAnnotator:
    def __init__(self, theme: str = "dark") -> None:
        self.theme = theme

    def annotate(self, frame: np.ndarray, detections: sv.Detections, fps: float, class_names=None) -> np.ndarray:
        # Return an unchanged copy to validate frame shape consistency.
        return frame.copy()


def test_pipeline_process_frame_and_stats(monkeypatch) -> None:
    """Pipeline should return same-shaped frame and expected stats fields."""
    monkeypatch.setattr(pipeline_module, "ObjectDetector", _FakeDetector)
    monkeypatch.setattr(pipeline_module, "ObjectTracker", _FakeTracker)
    monkeypatch.setattr(pipeline_module, "FrameAnnotator", _FakeAnnotator)

    pipeline = pipeline_module.DetectionTrackingPipeline(model_size="n", confidence_threshold=0.3)
    frame = np.zeros((120, 160, 3), dtype=np.uint8)

    output = pipeline.process_frame(frame)
    stats = pipeline.get_stats()

    assert isinstance(output, np.ndarray)
    assert output.shape == frame.shape
    assert set(stats.keys()) == {
        "current_fps",
        "total_frames_processed",
        "active_tracks",
        "total_unique_tracks",
    }
    assert stats["total_frames_processed"] == 1
