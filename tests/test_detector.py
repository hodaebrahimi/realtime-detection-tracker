"""Tests for detector module."""

from __future__ import annotations

import numpy as np
import pytest

sv = pytest.importorskip("supervision")

from src import detector as detector_module


class _DummyYOLO:
    def __init__(self, _: str) -> None:
        self.names = {0: "person", 2: "car"}

    def predict(self, frame: np.ndarray, conf: float, verbose: bool = False):
        return [object()]


def test_detector_returns_supervision_detections(monkeypatch) -> None:
    """Detector should return sv.Detections and support class filtering."""

    def _fake_from_ultralytics(_: object) -> sv.Detections:
        return sv.Detections(
            xyxy=np.array([[0, 0, 10, 10], [20, 20, 40, 40]], dtype=np.float32),
            confidence=np.array([0.9, 0.85], dtype=np.float32),
            class_id=np.array([0, 2], dtype=np.int64),
        )

    monkeypatch.setattr(detector_module, "YOLO", _DummyYOLO)
    monkeypatch.setattr(sv.Detections, "from_ultralytics", staticmethod(_fake_from_ultralytics))

    detector = detector_module.ObjectDetector(model_size="n")
    frame = np.zeros((128, 128, 3), dtype=np.uint8)
    detections = detector.detect(frame, confidence_threshold=0.3, classes=["person"])

    assert isinstance(detections, sv.Detections)
    assert len(detections) == 1
    assert int(detections.class_id[0]) == 0


def test_detector_handles_empty_frame(monkeypatch) -> None:
    """Detector should return empty detections for empty input frames."""
    monkeypatch.setattr(detector_module, "YOLO", _DummyYOLO)
    detector = detector_module.ObjectDetector(model_size="n")

    empty = np.array([], dtype=np.uint8)
    detections = detector.detect(empty)

    assert isinstance(detections, sv.Detections)
    assert len(detections) == 0
