"""ByteTrack wrapper built on supervision."""

from __future__ import annotations

import logging

import numpy as np
import supervision as sv

LOGGER = logging.getLogger(__name__)


class ObjectTracker:
    """Assign and maintain track IDs for frame detections."""

    def __init__(
        self,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
    ) -> None:
        """Initialize a ByteTrack tracker.

        Args:
            track_activation_threshold: Confidence threshold for track activation.
            lost_track_buffer: Number of frames to keep lost tracks alive.
        """
        self.track_activation_threshold = track_activation_threshold
        self.lost_track_buffer = lost_track_buffer
        self._unique_track_ids: set[int] = set()

        try:
            self._tracker = sv.ByteTrack(
                track_activation_threshold=track_activation_threshold,
                lost_track_buffer=lost_track_buffer,
            )
        except TypeError:
            LOGGER.warning(
                "Installed supervision version does not expose all ByteTrack tuning "
                "parameters. Falling back to default ByteTrack constructor."
            )
            self._tracker = sv.ByteTrack()

    def update(self, detections: sv.Detections) -> sv.Detections:
        """Update tracker state and return detections with tracker IDs."""
        tracked = self._tracker.update_with_detections(detections)

        if tracked.tracker_id is not None and len(tracked.tracker_id) > 0:
            valid_ids = tracked.tracker_id[tracked.tracker_id >= 0]
            if len(valid_ids) > 0:
                self._unique_track_ids.update(int(track_id) for track_id in np.unique(valid_ids))

        return tracked

    def get_track_count(self) -> int:
        """Return total number of unique tracks seen since initialization."""
        return len(self._unique_track_ids)
