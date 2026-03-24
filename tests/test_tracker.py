"""Tests for tracker module."""

from __future__ import annotations

import numpy as np
import pytest

sv = pytest.importorskip("supervision")

from src import tracker as tracker_module


class _DummyByteTrack:
    def __init__(self, *args, **kwargs) -> None:
        self._next_id = 1

    def update_with_detections(self, detections: sv.Detections) -> sv.Detections:
        count = len(detections)
        if count == 0:
            detections.tracker_id = np.array([], dtype=np.int64)
            return detections

        # Keep IDs stable by assigning fixed IDs for the first two detections.
        ids = np.array([1 + idx for idx in range(count)], dtype=np.int64)
        detections.tracker_id = ids
        return detections


def test_tracker_assigns_ids_and_counts_unique(monkeypatch) -> None:
    """ObjectTracker should attach tracker IDs and track unique ID count."""
    monkeypatch.setattr(tracker_module.sv, "ByteTrack", _DummyByteTrack)

    tracker = tracker_module.ObjectTracker(track_activation_threshold=0.2, lost_track_buffer=10)

    det_1 = sv.Detections(
        xyxy=np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float32),
        confidence=np.array([0.8, 0.7], dtype=np.float32),
        class_id=np.array([0, 2], dtype=np.int64),
    )
    tracked_1 = tracker.update(det_1)

    det_2 = sv.Detections(
        xyxy=np.array([[1, 1, 11, 11], [21, 21, 31, 31]], dtype=np.float32),
        confidence=np.array([0.82, 0.72], dtype=np.float32),
        class_id=np.array([0, 2], dtype=np.int64),
    )
    tracked_2 = tracker.update(det_2)

    assert tracked_1.tracker_id is not None
    assert tracked_2.tracker_id is not None
    assert list(tracked_1.tracker_id) == [1, 2]
    assert list(tracked_2.tracker_id) == [1, 2]
    assert tracker.get_track_count() == 2
