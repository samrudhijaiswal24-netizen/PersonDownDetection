"""
YOLO11 Pose Detector Module
Wraps Ultralytics YOLO11-Pose model for person detection and 17-keypoint extraction.
Supports CUDA acceleration and robust error handling.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union
import logging
import numpy as np
import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)


@dataclass
class PersonDetection:
    """Represents a single detected person with pose keypoints."""
    box: np.ndarray  # [x1, y1, x2, y2] in pixel coordinates
    confidence: float  # Bounding box confidence score [0.0 - 1.0]
    keypoints: np.ndarray  # (17, 3) array: [x, y, confidence] for each COCO keypoint
    posture: str = "UNKNOWN"
    posture_confidence: float = 0.0

    @property
    def x1(self) -> float:
        return float(self.box[0])

    @property
    def y1(self) -> float:
        return float(self.box[1])

    @property
    def x2(self) -> float:
        return float(self.box[2])

    @property
    def y2(self) -> float:
        return float(self.box[3])

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)


class YOLOPoseDetector:
    """YOLO11-Pose detector for person and keypoint extraction."""

    # Standard COCO 17 Pose Keypoint Names
    KEYPOINT_NAMES = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]

    def __init__(
        self,
        model_path: Union[str, Path] = "models/pretrained/yolo11n-pose.pt",
        device: Optional[str] = None,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
    ):
        """
        Initialize the YOLO11-Pose detector.

        Args:
            model_path: Path to YOLO11-Pose weights file (.pt).
            device: 'cuda', 'cuda:0', 'cpu', or None (auto-detect).
            conf_threshold: Minimum confidence score for person detection.
            iou_threshold: NMS IoU threshold.
        """
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # Device selection: prefer CUDA if available
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda:0"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.model = self._load_model()
        logger.info(
            f"YOLOPoseDetector initialized: model='{self.model_path.name}', "
            f"device='{self.device}', conf={self.conf_threshold}, iou={self.iou_threshold}"
        )

    def _load_model(self) -> YOLO:
        """Loads and verifies the YOLO model weights."""
        if not self.model_path.exists():
            # If specified path does not exist, check default location or raise
            raise FileNotFoundError(
                f"YOLO pose model file not found at: {self.model_path.resolve()}. "
                "Ensure pretrained weights exist before running inference."
            )

        try:
            model = YOLO(str(self.model_path))
            return model
        except Exception as e:
            raise RuntimeError(
                f"Failed to load YOLO model from '{self.model_path}': {str(e)}"
            ) from e

    def detect(self, frame: np.ndarray) -> List[PersonDetection]:
        """
        Runs pose estimation on a single BGR video frame.

        Args:
            frame: OpenCV BGR image (H, W, C) numpy array.

        Returns:
            List of PersonDetection objects (empty list if no persons detected).
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            logger.warning("Empty or invalid frame passed to YOLOPoseDetector.detect().")
            return []

        try:
            results = self.model.predict(
                source=frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False,
            )
        except Exception as e:
            logger.error(f"Inference error in YOLOPoseDetector: {e}")
            raise RuntimeError(f"YOLO inference failed: {e}") from e

        if not results or len(results) == 0:
            return []

        res = results[0]
        detections: List[PersonDetection] = []

        if res.boxes is None or len(res.boxes) == 0:
            return detections

        boxes_xyxy = res.boxes.xyxy.cpu().numpy()
        boxes_conf = res.boxes.conf.cpu().numpy()

        # Keypoints data shape: (N, 17, 3) -> [x, y, conf]
        if res.keypoints is not None and res.keypoints.data is not None:
            kpts_data = res.keypoints.data.cpu().numpy()
        else:
            kpts_data = np.zeros((len(boxes_xyxy), 17, 3), dtype=np.float32)

        for i in range(len(boxes_xyxy)):
            detection = PersonDetection(
                box=boxes_xyxy[i],
                confidence=float(boxes_conf[i]),
                keypoints=kpts_data[i],
            )
            detections.append(detection)

        return detections

    def get_device_info(self) -> str:
        """Returns human-readable device name."""
        if "cuda" in self.device and torch.cuda.is_available():
            dev_idx = 0
            if ":" in self.device:
                try:
                    dev_idx = int(self.device.split(":")[1])
                except ValueError:
                    dev_idx = 0
            return f"GPU: {torch.cuda.get_device_name(dev_idx)}"
        return "CPU"
