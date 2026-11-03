"""
Video Processor Module
Modular video processing pipeline for person detection, pose extraction,
frame annotation, and structured metrics reporting.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union
import json
import logging
import os
import time
import cv2
import numpy as np

from src.core.pose_detector import YOLOPoseDetector
from src.core.posture_classifier import PostureClassifier, PostureSmoother, PostureType
from src.pipeline.video_annotator import VideoAnnotator

logger = logging.getLogger(__name__)


class VideoProcessor:
    """Processes video files through the YOLO11-Pose detection, posture classification, and annotation pipeline."""

    def __init__(
        self,
        detector: Optional[YOLOPoseDetector] = None,
        model_path: Union[str, Path] = "models/pretrained/yolo11n-pose.pt",
        device: Optional[str] = None,
        conf_threshold: float = 0.5,
        annotator: Optional[VideoAnnotator] = None,
        posture_classifier: Optional[PostureClassifier] = None,
        enable_smoothing: bool = True,
        smoothing_window: int = 5,
    ):
        """
        Initializes the VideoProcessor.

        Args:
            detector: Pre-initialized YOLOPoseDetector or None to create one.
            model_path: Path to YOLO weights if detector is None.
            device: Computing device ('cuda', 'cpu', or None).
            conf_threshold: Person detection confidence threshold.
            annotator: Custom VideoAnnotator or None for default.
            posture_classifier: Custom PostureClassifier or None for default.
            enable_smoothing: Whether to apply temporal smoothing on posture classifications.
            smoothing_window: Number of consecutive frames for sliding window smoothing.
        """
        if detector is not None:
            self.detector = detector
        else:
            self.detector = YOLOPoseDetector(
                model_path=model_path,
                device=device,
                conf_threshold=conf_threshold,
            )

        self.annotator = annotator or VideoAnnotator()
        self.posture_classifier = posture_classifier or PostureClassifier()
        self.enable_smoothing = enable_smoothing
        self.smoother = PostureSmoother(window_size=smoothing_window) if enable_smoothing else None


    def process_video(
        self,
        input_video_path: Union[str, Path],
        output_video_path: Union[str, Path],
        summary_path: Optional[Union[str, Path]] = None,
        max_frames: Optional[int] = None,
        progress_callback: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Processes an input video file, writes annotated output video, and generates summary metrics.

        Args:
            input_video_path: Path to input video file (.mp4, .avi, etc.).
            output_video_path: Destination path for annotated video.
            summary_path: Optional path to save JSON metrics report.
            max_frames: Optional cap on the number of frames to process.
            progress_callback: Optional callback fn(current_frame, total_frames, fps).

        Returns:
            Dictionary containing processing summary and statistics.
        """
        input_path = Path(input_video_path)
        output_path = Path(output_video_path)

        # 1. Input Validation
        if not input_path.exists():
            raise FileNotFoundError(f"Input video does not exist: {input_path.resolve()}")

        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise ValueError(f"OpenCV failed to open video file: {input_path.resolve()}")

        # 2. Extract Video Metadata
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0 or np.isnan(fps):
            fps = 30.0  # Fallback default FPS
        if width <= 0 or height <= 0:
            cap.release()
            raise ValueError(f"Invalid video dimensions ({width}x{height}) in {input_path}")

        frames_to_process = total_frames
        if max_frames is not None and max_frames > 0:
            frames_to_process = min(total_frames, max_frames) if total_frames > 0 else max_frames

        # 3. Setup Video Writer
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        if not writer.isOpened():
            # Try alternate codec fallback
            logger.warning("Failed with 'mp4v' codec. Attempting fallback to 'avc1'.")
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Failed to create video writer at: {output_path.resolve()}")

        # 4. Processing Loop State
        if self.smoother is not None:
            self.smoother.reset()

        processed_frames = 0
        frames_with_detections = 0
        total_detections_count = 0
        confidences = []
        posture_counts: Dict[str, int] = {
            PostureType.STANDING.value: 0,
            PostureType.SITTING.value: 0,
            PostureType.LYING.value: 0,
            PostureType.UNKNOWN.value: 0,
        }
        device_name = self.detector.get_device_info()

        start_time = time.perf_counter()
        rolling_fps = fps

        logger.info(
            f"Processing video '{input_path.name}': {width}x{height} @ {fps:.1f} FPS, "
            f"{frames_to_process} frames to analyze on {device_name}"
        )

        try:
            while True:
                if max_frames is not None and processed_frames >= max_frames:
                    break

                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                frame_start = time.perf_counter()

                # Detect people and pose keypoints
                detections = self.detector.detect(frame)

                # Classify posture for each detected person
                num_detected = len(detections)
                if num_detected > 0:
                    frames_with_detections += 1
                    total_detections_count += num_detected
                    for det in detections:
                        confidences.append(float(det.confidence))
                        # Classify posture using geometric feature analysis
                        p_res = self.posture_classifier.classify(det.keypoints, det.box)
                        if self.smoother is not None:
                            p_res = self.smoother.update(p_res)
                        det.posture = p_res.label
                        det.posture_confidence = p_res.confidence
                        posture_counts[det.posture] = posture_counts.get(det.posture, 0) + 1

                # Calculate instantaneous FPS
                frame_duration = time.perf_counter() - frame_start
                if frame_duration > 0:
                    inst_fps = 1.0 / frame_duration
                    rolling_fps = 0.9 * rolling_fps + 0.1 * inst_fps

                # Annotate frame
                annotated = self.annotator.annotate_frame(
                    frame=frame,
                    detections=detections,
                    frame_idx=processed_frames,
                    total_frames=frames_to_process,
                    fps=rolling_fps,
                    device_name=device_name,
                )

                # Write annotated frame
                writer.write(annotated)
                processed_frames += 1

                if progress_callback:
                    progress_callback(processed_frames, frames_to_process, rolling_fps)

        finally:
            cap.release()
            writer.release()

        elapsed_time = max(0.001, time.perf_counter() - start_time)
        avg_fps = processed_frames / elapsed_time

        # 5. Build Summary Report
        avg_conf = float(np.mean(confidences)) if confidences else 0.0
        min_conf = float(np.min(confidences)) if confidences else 0.0
        max_conf = float(np.max(confidences)) if confidences else 0.0

        summary: Dict[str, Any] = {
            "status": "COMPLETED",
            "input_video": str(input_path.resolve()),
            "output_video": str(output_path.resolve()),
            "video_metadata": {
                "width": width,
                "height": height,
                "fps": round(float(fps), 2),
                "total_video_frames": total_frames,
            },
            "processing_metrics": {
                "processed_frames": processed_frames,
                "frames_with_detections": frames_with_detections,
                "detection_frame_ratio": round(
                    frames_with_detections / max(1, processed_frames), 4
                ),
                "total_person_detections": total_detections_count,
                "avg_persons_per_frame": round(
                    total_detections_count / max(1, processed_frames), 3
                ),
                "confidence_stats": {
                    "min": round(min_conf, 4),
                    "max": round(max_conf, 4),
                    "mean": round(avg_conf, 4),
                },
                "posture_counts": posture_counts,
                "posture_distribution": {
                    k: round(v / max(1, total_detections_count), 4)
                    for k, v in posture_counts.items()
                },
                "elapsed_time_seconds": round(elapsed_time, 2),
                "average_processing_fps": round(avg_fps, 2),
                "hardware_device": device_name,
            },
        }


        # 6. Verify Output Video File
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError(f"Output video file is missing or empty at {output_path}")

        summary["output_video_size_bytes"] = output_path.stat().st_size

        # 7. Save Summary JSON if requested
        if summary_path is not None:
            sum_path = Path(summary_path)
            sum_path.parent.mkdir(parents=True, exist_ok=True)
            with open(sum_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            summary["summary_json_path"] = str(sum_path.resolve())
            logger.info(f"Summary metrics saved to {sum_path.resolve()}")

        logger.info(
            f"Video processing finished: {processed_frames} frames in {elapsed_time:.1f}s "
            f"({avg_fps:.1f} FPS) -> {output_path.name}"
        )
        return summary
