"""
Unit and Integration Smoke Tests for YOLO11-Pose Video Pipeline
"""

import json
from pathlib import Path
import cv2
import numpy as np
import pytest

from src.core.pose_detector import YOLOPoseDetector, PersonDetection
from src.pipeline.video_annotator import VideoAnnotator
from src.pipeline.video_processor import VideoProcessor

MODEL_PATH = Path("models/pretrained/yolo11n-pose.pt")


def test_missing_video_raises_file_not_found():
    """VideoProcessor must raise FileNotFoundError for non-existent video files."""
    processor = VideoProcessor(model_path=MODEL_PATH)
    with pytest.raises(FileNotFoundError, match="Input video does not exist"):
        processor.process_video(
            input_video_path="data/samples/non_existent_video_12345.mp4",
            output_video_path="data/processed/dummy_out.mp4",
        )


def test_invalid_model_path_raises_error():
    """YOLOPoseDetector must raise FileNotFoundError if weights do not exist."""
    with pytest.raises(FileNotFoundError, match="YOLO pose model file not found"):
        YOLOPoseDetector(model_path="models/pretrained/non_existent_model.pt")


def test_detector_initialization_and_empty_frame():
    """Verify detector loads correctly and handles blank/empty frames without error."""
    assert MODEL_PATH.exists(), f"Weights missing at {MODEL_PATH}"
    detector = YOLOPoseDetector(model_path=MODEL_PATH)

    # 1. Blank black image
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = detector.detect(blank_frame)
    assert isinstance(detections, list)
    # A pure black frame should have 0 person detections
    assert len(detections) == 0

    # 2. None / empty array handling
    assert detector.detect(None) == []
    assert detector.detect(np.array([])) == []


def test_video_annotator_on_synthetic_detections():
    """Verify VideoAnnotator draws correctly on a frame."""
    annotator = VideoAnnotator()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Synthetic detection
    dummy_box = np.array([100, 100, 200, 300], dtype=np.float32)
    dummy_kpts = np.zeros((17, 3), dtype=np.float32)
    # Give some joints high confidence
    dummy_kpts[0] = [150, 120, 0.9]  # nose
    dummy_kpts[5] = [130, 150, 0.8]  # left shoulder
    dummy_kpts[6] = [170, 150, 0.8]  # right shoulder

    detection = PersonDetection(box=dummy_box, confidence=0.88, keypoints=dummy_kpts)
    annotated = annotator.annotate_frame(
        frame=frame,
        detections=[detection],
        frame_idx=0,
        total_frames=10,
        fps=30.0,
        device_name="Test Device",
    )

    assert annotated.shape == frame.shape
    # Frame should no longer be completely black
    assert np.any(annotated > 0)


def test_synthetic_video_pipeline_smoke_test(tmp_path):
    """
    End-to-end smoke test:
    Generates a 15-frame synthetic video, processes it through VideoProcessor,
    and asserts output video and summary report are valid.
    """
    synth_input = tmp_path / "synthetic_input.mp4"
    synth_output = tmp_path / "synthetic_output.mp4"
    synth_summary = tmp_path / "synthetic_summary.json"

    # Generate synthetic 15-frame video (640x480, 30fps)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(synth_input), fourcc, 30.0, (640, 480))
    for i in range(15):
        frame = np.full((480, 640, 3), fill_value=i * 10, dtype=np.uint8)
        # Draw a synthetic circle
        cv2.circle(frame, (100 + i * 5, 200), 40, (0, 255, 0), -1)
        out.write(frame)
    out.release()

    assert synth_input.exists() and synth_input.stat().st_size > 0

    # Process through VideoProcessor
    processor = VideoProcessor(model_path=MODEL_PATH)
    summary = processor.process_video(
        input_video_path=synth_input,
        output_video_path=synth_output,
        summary_path=synth_summary,
    )

    # Verify output video
    assert synth_output.exists()
    assert synth_output.stat().st_size > 0

    # Verify summary dictionary and JSON file
    assert summary["status"] == "COMPLETED"
    assert summary["processing_metrics"]["processed_frames"] == 15
    assert synth_summary.exists()

    with open(synth_summary, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["processing_metrics"]["processed_frames"] == 15
