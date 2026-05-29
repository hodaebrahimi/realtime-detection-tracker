"""Gradio web app for realtime-detection-tracker."""

from __future__ import annotations

import tempfile
from pathlib import Path

import gradio as gr

from src.pipeline import DetectionTrackingPipeline
from src.video_runner import run_on_video

COMMON_COCO_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "dog",
    "cat",
    "bird",
    "backpack",
]

MODEL_LABEL_TO_SIZE = {
    "nano": "n",
    "small": "s",
    "medium": "m",
}


def _build_stats_text(stats: dict[str, float | int | str]) -> str:
    """Format pipeline stats for display in UI."""
    return (
        f"Total Frames: {stats['total_frames']}\n"
        f"Average FPS: {stats['avg_fps']:.2f}\n"
        f"Unique Objects Tracked: {stats['total_unique_tracks']}\n"
        f"Active Tracks (last frame): {stats['active_tracks']}"
    )


def process_video(
    video_path: str | None,
    model_size_label: str,
    confidence_threshold: float,
    class_filter: list[str],
) -> tuple[str | None, str]:
    """Process an uploaded video and return output path + stats text."""
    if not video_path:
        return None, "Please upload a video file first."

    model_size = MODEL_LABEL_TO_SIZE[model_size_label]
    selected_classes = class_filter if class_filter else None

    pipeline = DetectionTrackingPipeline(
        model_size=model_size,
        confidence_threshold=confidence_threshold,
        classes=selected_classes,
        theme="dark",
    )

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_output:
        output_path = tmp_output.name

    stats = run_on_video(
        input_path=video_path,
        output_path=output_path,
        pipeline=pipeline,
        show_preview=False,
    )
    return output_path, _build_stats_text(stats)


def run_sample_video(
    model_size_label: str,
    confidence_threshold: float,
    class_filter: list[str],
) -> tuple[str | None, str]:
    """Run processing on downloaded sample video if available."""
    sample_path = Path("data/sample.mp4")
    if not sample_path.exists():
        return None, "Sample video not found. Run scripts/download_sample.py first."

    return process_video(
        video_path=str(sample_path),
        model_size_label=model_size_label,
        confidence_threshold=confidence_threshold,
        class_filter=class_filter,
    )


def create_app() -> gr.Blocks:
    """Create and return the Gradio Blocks interface."""
    with gr.Blocks(title="realtime-detection-tracker") as demo:
        gr.Markdown("# realtime-detection-tracker")
        gr.Markdown("Upload a video and run YOLOv8 + ByteTrack annotation in one click.")

        with gr.Row():
            with gr.Column():
                input_video = gr.Video(label="Input Video", sources=["upload"])
                model_size = gr.Dropdown(
                    label="Model Size",
                    choices=["nano", "small", "medium"],
                    value="nano",
                )
                confidence = gr.Slider(
                    label="Confidence Threshold",
                    minimum=0.1,
                    maximum=0.9,
                    value=0.3,
                    step=0.05,
                )
                class_filter = gr.CheckboxGroup(
                    label="Optional Class Filter",
                    choices=COMMON_COCO_CLASSES,
                    value=[],
                )
                process_button = gr.Button("Process Video", variant="primary")
                sample_button = gr.Button("Try Sample Video")

            with gr.Column():
                output_video = gr.Video(label="Annotated Output", interactive=False)
                stats_text = gr.Textbox(label="Stats Summary", lines=6)

        process_button.click(
            fn=process_video,
            inputs=[input_video, model_size, confidence, class_filter],
            outputs=[output_video, stats_text],
        )
        sample_button.click(
            fn=run_sample_video,
            inputs=[model_size, confidence, class_filter],
            outputs=[output_video, stats_text],
        )

    return demo


def main() -> None:
    """Run the Gradio app."""
    app = create_app()
    app.launch()


if __name__ == "__main__":
    main()
