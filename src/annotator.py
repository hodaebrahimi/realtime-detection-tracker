"""Frame annotation utilities for detections and tracks."""

from __future__ import annotations

import cv2
import numpy as np
import supervision as sv


class FrameAnnotator:
    """Annotate frames with boxes, labels, traces, and FPS overlay."""

    def __init__(self, theme: str = "dark") -> None:
        """Initialize frame annotator.

        Args:
            theme: `light` or `dark`, controls text/background contrast.
        """
        normalized_theme = theme.strip().lower()
        if normalized_theme not in {"light", "dark"}:
            raise ValueError("theme must be either 'light' or 'dark'")

        self.theme = normalized_theme
        self._bbox_annotator = sv.BoundingBoxAnnotator(color_lookup=sv.ColorLookup.TRACK)
        self._label_annotator = sv.LabelAnnotator(
            color_lookup=sv.ColorLookup.TRACK,
            text_color=sv.Color.BLACK if self.theme == "light" else sv.Color.WHITE,
            text_scale=0.45,
            text_thickness=1,
        )
        self._trace_annotator = sv.TraceAnnotator(
            color_lookup=sv.ColorLookup.TRACK,
            trace_length=30,
            thickness=2,
        )

        if self.theme == "light":
            self._fps_text_color = (20, 20, 20)
            self._fps_bg_color = (245, 245, 245)
        else:
            self._fps_text_color = (245, 245, 245)
            self._fps_bg_color = (20, 20, 20)

    def annotate(
        self,
        frame: np.ndarray,
        detections: sv.Detections,
        fps: float,
        class_names: list[str] | dict[int, str] | None = None,
    ) -> np.ndarray:
        """Draw all overlays for a frame.

        Args:
            frame: Input frame.
            detections: Tracked detections.
            fps: Current estimated FPS.
            class_names: Optional class-name lookup.

        Returns:
            Annotated frame.
        """
        annotated = frame.copy()

        labels = self._build_labels(detections, class_names)
        annotated = self._trace_annotator.annotate(scene=annotated, detections=detections)
        annotated = self._bbox_annotator.annotate(scene=annotated, detections=detections)
        annotated = self._label_annotator.annotate(
            scene=annotated,
            detections=detections,
            labels=labels,
        )

        self._draw_fps_overlay(annotated, fps)
        return annotated

    def _build_labels(
        self,
        detections: sv.Detections,
        class_names: list[str] | dict[int, str] | None,
    ) -> list[str]:
        """Build labels of format `class #ID (conf%)`."""
        labels: list[str] = []
        confidence_values = detections.confidence if detections.confidence is not None else []

        for idx in range(len(detections)):
            class_id = int(detections.class_id[idx]) if detections.class_id is not None else -1
            tracker_id = (
                int(detections.tracker_id[idx])
                if detections.tracker_id is not None and detections.tracker_id[idx] is not None
                else -1
            )
            confidence = float(confidence_values[idx]) if idx < len(confidence_values) else 0.0
            class_label = self._resolve_class_name(class_id, class_names)
            labels.append(f"{class_label} #{tracker_id} ({confidence * 100:.0f}%)")

        return labels

    @staticmethod
    def _resolve_class_name(
        class_id: int,
        class_names: list[str] | dict[int, str] | None,
    ) -> str:
        """Resolve class id to readable class name."""
        if isinstance(class_names, dict):
            return class_names.get(class_id, str(class_id))

        if isinstance(class_names, list) and 0 <= class_id < len(class_names):
            return class_names[class_id]

        return str(class_id)

    def _draw_fps_overlay(self, frame: np.ndarray, fps: float) -> None:
        """Draw FPS text with a compact background in the top-left corner."""
        fps_text = f"FPS: {fps:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        x, y = 16, 32

        (text_width, text_height), baseline = cv2.getTextSize(fps_text, font, font_scale, thickness)
        cv2.rectangle(
            frame,
            (x - 8, y - text_height - 8),
            (x + text_width + 8, y + baseline + 8),
            self._fps_bg_color,
            -1,
        )
        cv2.putText(
            frame,
            fps_text,
            (x, y),
            font,
            font_scale,
            self._fps_text_color,
            thickness,
            cv2.LINE_AA,
        )
