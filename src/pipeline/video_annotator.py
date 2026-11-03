"""
Video Annotator Module
Renders visual overlays onto video frames: bounding boxes, confidence tags,
COCO 17-keypoint pose skeletons, and telemetry HUD.
"""

from typing import List, Optional, Tuple
import cv2
import numpy as np

from src.core.pose_detector import PersonDetection


class VideoAnnotator:
    """Draws bounding boxes, pose skeletons, and telemetry on frames."""

    # COCO 17 Keypoint Skeleton Connections: (pt1_idx, pt2_idx, BGR_color)
    SKELETON_CONNECTIONS = [
        # Face / Head (Cyan: 255, 255, 0)
        (0, 1, (255, 255, 0)),
        (0, 2, (255, 255, 0)),
        (1, 3, (255, 255, 0)),
        (2, 4, (255, 255, 0)),
        # Torso (Gold/Yellow: 0, 215, 255)
        (5, 6, (0, 215, 255)),
        (5, 11, (0, 215, 255)),
        (6, 12, (0, 215, 255)),
        (11, 12, (0, 215, 255)),
        # Left Arm (Lime Green: 0, 255, 128)
        (5, 7, (0, 255, 128)),
        (7, 9, (0, 255, 128)),
        # Right Arm (Orange: 0, 140, 255)
        (6, 8, (0, 140, 255)),
        (8, 10, (0, 140, 255)),
        # Left Leg (Sky Blue: 255, 191, 0)
        (11, 13, (255, 191, 0)),
        (13, 15, (255, 191, 0)),
        # Right Leg (Pink/Magenta: 204, 0, 255)
        (12, 14, (204, 0, 255)),
        (14, 16, (204, 0, 255)),
    ]

    KEYPOINT_COLOR = (0, 255, 255)  # Yellow dots
    BOX_COLOR = (0, 200, 0)         # Green bounding box

    def __init__(
        self,
        keypoint_conf_threshold: float = 0.3,
        line_thickness: int = 2,
        point_radius: int = 4,
    ):
        self.kpt_conf_thresh = keypoint_conf_threshold
        self.line_thickness = line_thickness
        self.point_radius = point_radius

    def draw_skeleton(self, frame: np.ndarray, keypoints: np.ndarray) -> None:
        """
        Draws the 17-keypoint skeleton connections and joints.

        Args:
            frame: BGR image array.
            keypoints: (17, 3) array [x, y, conf].
        """
        if keypoints is None or len(keypoints) < 17:
            return

        # 1. Draw Bones / Limbs
        for pt1_idx, pt2_idx, color in self.SKELETON_CONNECTIONS:
            p1 = keypoints[pt1_idx]
            p2 = keypoints[pt2_idx]

            # Only draw if both keypoints have confidence above threshold
            if p1[2] >= self.kpt_conf_thresh and p2[2] >= self.kpt_conf_thresh:
                x1, y1 = int(round(p1[0])), int(round(p1[1]))
                x2, y2 = int(round(p2[0])), int(round(p2[1]))
                cv2.line(frame, (x1, y1), (x2, y2), color, self.line_thickness, cv2.LINE_AA)

        # 2. Draw Keypoint Joints
        for i in range(17):
            pt = keypoints[i]
            if pt[2] >= self.kpt_conf_thresh:
                x, y = int(round(pt[0])), int(round(pt[1]))
                cv2.circle(frame, (x, y), self.point_radius, self.KEYPOINT_COLOR, -1, cv2.LINE_AA)
                cv2.circle(frame, (x, y), self.point_radius + 1, (0, 0, 0), 1, cv2.LINE_AA)

    POSTURE_COLORS = {
        "STANDING": (0, 200, 0),      # Bright Green
        "SITTING": (255, 180, 0),     # Azure / Sky Blue
        "LYING": (30, 30, 230),       # High-visibility Red / Crimson
        "UNKNOWN": (140, 140, 140),   # Neutral Gray
    }

    def draw_bounding_box(
        self,
        frame: np.ndarray,
        detection: PersonDetection,
        person_idx: int,
    ) -> None:
        """Draws bounding box with posture classification label tag."""
        x1, y1 = int(round(detection.x1)), int(round(detection.y1))
        x2, y2 = int(round(detection.x2)), int(round(detection.y2))

        # Clamp coordinates to frame boundaries
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        posture = getattr(detection, "posture", "UNKNOWN") or "UNKNOWN"
        box_color = self.POSTURE_COLORS.get(posture, self.BOX_COLOR)
        thickness = self.line_thickness + (1 if posture == "LYING" else 0)

        # Bounding box rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness, cv2.LINE_AA)

        # Label banner: "Person #1: 92% | STANDING"
        label = f"Person #{person_idx + 1}: {detection.confidence * 100:.0f}% | {posture}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        (txt_w, txt_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)

        tag_y1 = max(0, y1 - txt_h - 6)
        tag_y2 = y1
        cv2.rectangle(frame, (x1, tag_y1), (x1 + txt_w + 6, tag_y2), box_color, -1)
        cv2.putText(
            frame,
            label,
            (x1 + 3, tag_y2 - baseline),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness,
            cv2.LINE_AA,
        )

    def draw_hud(
        self,
        frame: np.ndarray,
        frame_idx: int,
        total_frames: int,
        fps: float,
        num_persons: int,
        device_name: str,
        posture_breakdown: Optional[str] = None,
    ) -> None:
        """Draws telemetry HUD at the top-left corner."""
        hud_lines = [
            f"Frame: {frame_idx + 1}/{total_frames} | FPS: {fps:.1f}",
            f"Persons Detected: {num_persons}" + (f" ({posture_breakdown})" if posture_breakdown else ""),
            f"Hardware: {device_name}",
        ]

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        line_height = 22

        # Draw semi-transparent background box
        box_w = 340
        box_h = len(hud_lines) * line_height + 14
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (10 + box_w, 10 + box_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        # Draw border
        cv2.rectangle(frame, (10, 10), (10 + box_w, 10 + box_h), (80, 80, 80), 1)

        # Draw text lines
        for i, line in enumerate(hud_lines):
            y_pos = 28 + (i * line_height)
            # Distinct color for hardware line
            color = (0, 255, 128) if "GPU" in line else (230, 230, 230)
            cv2.putText(
                frame,
                line,
                (20, y_pos),
                font,
                font_scale,
                color,
                font_thickness,
                cv2.LINE_AA,
            )

    def annotate_frame(
        self,
        frame: np.ndarray,
        detections: List[PersonDetection],
        frame_idx: int = 0,
        total_frames: int = 0,
        fps: float = 0.0,
        device_name: str = "GPU",
    ) -> np.ndarray:
        """
        Annotates a frame with detections, skeletons, and HUD.

        Args:
            frame: Input BGR frame.
            detections: List of PersonDetection objects.
            frame_idx: Current frame index.
            total_frames: Total video frames.
            fps: Current processing FPS.
            device_name: Processing hardware description.

        Returns:
            Annotated BGR frame copy.
        """
        annotated = frame.copy()

        # Draw detections and skeletons
        for idx, det in enumerate(detections):
            self.draw_bounding_box(annotated, det, idx)
            self.draw_skeleton(annotated, det.keypoints)

        # Posture summary for HUD
        posture_breakdown = None
        if detections:
            posture_counts = {}
            for d in detections:
                p = getattr(d, "posture", "UNKNOWN") or "UNKNOWN"
                posture_counts[p] = posture_counts.get(p, 0) + 1
            posture_breakdown = ", ".join(f"{k}:{v}" for k, v in posture_counts.items())

        # Draw telemetry HUD
        self.draw_hud(
            annotated,
            frame_idx=frame_idx,
            total_frames=total_frames,
            fps=fps,
            num_persons=len(detections),
            device_name=device_name,
            posture_breakdown=posture_breakdown,
        )

        return annotated

