"""YOLOv8 object detector wrapper."""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import supervision as sv
from ultralytics import YOLO

LOGGER = logging.getLogger(__name__)

MODEL_SIZE_TO_WEIGHTS = {
    "n": "yolov8n.pt",
    "s": "yolov8s.pt",
    "m": "yolov8m.pt",
    "l": "yolov8l.pt",
    "x": "yolov8x.pt",
}


class ObjectDetector:
    """Detect objects in video frames using YOLOv8."""

    def __init__(self, model_size: str = "n") -> None:
        """Initialize detector with a YOLOv8 model size.

        Args:
            model_size: One of `n`, `s`, `m`, `l`, `x`.
        """
        normalized_size = model_size.strip().lower()
        if normalized_size not in MODEL_SIZE_TO_WEIGHTS:
            raise ValueError(
                f"Unsupported model size '{model_size}'. "
                f"Expected one of {sorted(MODEL_SIZE_TO_WEIGHTS)}."
            )

        weights = MODEL_SIZE_TO_WEIGHTS[normalized_size]
        LOGGER.info("Loading YOLOv8 model: %s", weights)
        self.model = YOLO(weights)

    def detect(
        self,
        frame: np.ndarray,
        confidence_threshold: float = 0.3,
        classes: list[str] | None = None,
    ) -> sv.Detections:
        """Run inference and return filtered detections.

        Args:
            frame: Input frame in BGR format.
            confidence_threshold: Minimum confidence score.
            classes: Optional list of COCO class names to keep.

        Returns:
            Supervision detections filtered by confidence and class.
        """
        if frame is None or frame.size == 0:
            LOGGER.warning("Received an empty frame; returning no detections.")
            return sv.Detections.empty()

        results = self.model.predict(frame, conf=confidence_threshold, verbose=False)
        if not results:
            return sv.Detections.empty()

        detections = sv.Detections.from_ultralytics(results[0])
        if len(detections) == 0:
            return detections

        # Confidence filtering is already done by YOLO, but this keeps behavior explicit.
        if detections.confidence is not None:
            confidence_mask = detections.confidence >= confidence_threshold
            detections = detections[confidence_mask]

        if classes:
            class_name_to_id = self._get_class_name_mapping()
            requested_ids = self._resolve_class_names(classes, class_name_to_id)
            if requested_ids:
                class_id_mask = np.isin(detections.class_id, requested_ids)
                detections = detections[class_id_mask]
            else:
                LOGGER.warning(
                    "No valid class names in filter: %s. Returning no detections.",
                    classes,
                )
                return sv.Detections.empty()

        return detections

    def _get_class_name_mapping(self) -> dict[str, int]:
        """Return model class name to class id mapping."""
        model_names = self.model.names

        if isinstance(model_names, dict):
            return {str(name).lower(): int(class_id) for class_id, name in model_names.items()}

        if isinstance(model_names, list):
            return {str(name).lower(): idx for idx, name in enumerate(model_names)}

        return {}

    def _resolve_class_names(
        self,
        class_names: Iterable[str],
        class_name_to_id: dict[str, int],
    ) -> list[int]:
        """Resolve class names to class IDs, warning on unknown names."""
        resolved_ids: list[int] = []
        for class_name in class_names:
            normalized_name = class_name.strip().lower()
            if normalized_name in class_name_to_id:
                resolved_ids.append(class_name_to_id[normalized_name])
            else:
                LOGGER.warning("Unknown class name '%s'; skipping.", class_name)

        # Keep deterministic ordering and uniqueness.
        return sorted(set(resolved_ids))
