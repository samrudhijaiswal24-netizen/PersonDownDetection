"""
Unified Person Down Detection Pipeline Module.

Phase 7: End-to-End Person Down Detection Application.
Integrates:
- YOLO11-Pose detection (YOLOPoseDetector)
- Geometric posture classification with Phase 6 foreshortening & crouch safeguards (PostureClassifier)
- Posture temporal smoothing (PostureSmoother)
- Spatial zone polygon disambiguation (SpatialZoneManager)
- Temporal persistence & state tracking with Phase 5 hysteresis default (TemporalStateTracker)
- Visual annotation & telemetry HUD rendering (VideoAnnotator, ZoneVisualizer)

Preserves all Phase 1-6 model weights, configs, baseline reports, and evaluation artifacts intact.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import json
import logging
import os
import time
import cv2
import numpy as np

from src.core.pose_detector import YOLOPoseDetector, PersonDetection
from src.core.posture_classifier import PostureClassifier, PostureSmoother, PostureType
from src.core.spatial_zones import (
    ZoneManager,
    ZoneType,
    ZoneEventClassification,
    ZoneResult,
    CameraZoneConfig,
)
from src.core.temporal_tracker import (
    TemporalStateTracker,
    TemporalTrackerConfig,
    TemporalState,
    TemporalStateUpdate,
)
from src.pipeline.video_annotator import VideoAnnotator
from src.pipeline.zone_visualizer import (
    ZoneVisualizer,
    COLOR_SOFA_FILL,
    COLOR_SOFA_BORDER,
    COLOR_FLOOR_FILL,
    COLOR_FLOOR_BORDER,
)

logger = logging.getLogger(__name__)

SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
RESEARCH_PROTOTYPE_DISCLAIMER = (
    "Research prototype for person down and fall detection. "
    "Not a certified medical diagnostic device or hospital emergency alert system."
)


@dataclass
class DetectedCandidateEvent:
    """Represents a discrete candidate event detected by the pipeline."""
    event_id: int
    start_frame: int
    trigger_frame: int
    end_frame: int
    start_time_seconds: float
    trigger_time_seconds: float
    end_time_seconds: float
    duration_seconds: float
    duration_frames: int
    status: str  # "ACTIVE" or "RESOLVED"
    zone_name: Optional[str]
    peak_posture: str
    peak_confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "start_frame": self.start_frame,
            "trigger_frame": self.trigger_frame,
            "end_frame": self.end_frame,
            "start_time_seconds": round(self.start_time_seconds, 2),
            "trigger_time_seconds": round(self.trigger_time_seconds, 2),
            "end_time_seconds": round(self.end_time_seconds, 2),
            "duration_seconds": round(self.duration_seconds, 2),
            "duration_frames": self.duration_frames,
            "status": self.status,
            "zone_name": self.zone_name or "FLOOR",
            "peak_posture": self.peak_posture,
            "peak_confidence": round(self.peak_confidence, 4),
        }


class PersonDownPipeline:
    """
    Unified end-to-end processing pipeline integrating pose detection,
    posture classification, spatial zones, temporal tracking, and annotation.
    """

    def __init__(
        self,
        model_path: Union[str, Path] = "models/pretrained/yolo11n-pose.pt",
        zones_config_path: Union[str, Path] = "configs/zones_config.yaml",
        tracker_config_path: Union[str, Path] = "configs/temporal_tracker_config.yaml",
        device: Optional[str] = None,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        enable_smoothing: bool = True,
        smoothing_window: int = 5,
    ):
        """
        Initializes pipeline components.
        """
        self.model_path = Path(model_path)
        self.zones_config_path = Path(zones_config_path)
        self.tracker_config_path = Path(tracker_config_path)

        # 1. Initialize Detector
        if not self.model_path.exists():
            raise FileNotFoundError(f"YOLO11-Pose model weights not found at: {self.model_path.resolve()}")

        self.detector = YOLOPoseDetector(
            model_path=self.model_path,
            device=device,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
        )

        # 2. Initialize Posture Classifier & Smoother (with Phase 6 logic)
        self.posture_classifier = PostureClassifier()
        self.enable_smoothing = enable_smoothing
        self.smoother = PostureSmoother(window_size=smoothing_window) if enable_smoothing else None

        # 3. Initialize Spatial Zone Manager
        if not self.zones_config_path.exists():
            raise FileNotFoundError(f"Zones config not found at: {self.zones_config_path.resolve()}")
        self.zone_manager = ZoneManager(self.zones_config_path)

        # 4. Initialize Tracker Configuration
        if not self.tracker_config_path.exists():
            raise FileNotFoundError(f"Temporal tracker config not found at: {self.tracker_config_path.resolve()}")
        self.tracker_config = TemporalTrackerConfig.from_yaml(self.tracker_config_path)

        # 5. Visualizers
        self.annotator = VideoAnnotator()
        self.zone_visualizer = ZoneVisualizer(alpha=0.22)

        logger.info(
            f"PersonDownPipeline initialized successfully on {self.detector.get_device_info()} "
            f"(Zones: {self.zones_config_path.name}, Tracker: {self.tracker_config_path.name})"
        )

    @staticmethod
    def validate_video_file(file_path: Union[str, Path]) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
        """
        Validates whether a file exists, has a supported video extension, and can be opened by OpenCV.

        Returns:
            Tuple of (is_valid, error_message, video_metadata_dict)
        """
        p = Path(file_path)
        if not p.exists():
            return False, f"File not found: {p.resolve()}", None

        if p.is_dir():
            return False, f"Specified path is a directory, not a video file: {p.name}", None

        ext = p.suffix.lower()
        if ext not in SUPPORTED_VIDEO_EXTENSIONS:
            return (
                False,
                f"Unsupported video format '{ext}'. Supported formats: {', '.join(sorted(SUPPORTED_VIDEO_EXTENSIONS))}",
                None,
            )

        if p.stat().st_size == 0:
            return False, "Video file is empty (0 bytes).", None

        cap = cv2.VideoCapture(str(p.resolve()))
        if not cap.isOpened():
            return False, "Failed to decode video stream. The file may be corrupted or use an unsupported codec.", None

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if width <= 0 or height <= 0:
            return False, f"Invalid video dimensions: {width}x{height}", None

        nominal_fps = 30.0 if (fps <= 0 or np.isnan(fps)) else round(float(fps), 2)
        duration_sec = round(total_frames / nominal_fps, 2) if total_frames > 0 else 0.0

        metadata = {
            "filename": p.name,
            "filepath": str(p.resolve()),
            "width": width,
            "height": height,
            "fps": nominal_fps,
            "total_frames": total_frames,
            "duration_seconds": duration_sec,
            "size_bytes": p.stat().st_size,
        }
        return True, None, metadata

    def resolve_camera_config_id(self, input_path: Union[str, Path], user_camera_id: Optional[str] = None) -> str:
        """
        Resolves the appropriate camera zone configuration ID based on user selection or filename hint.
        """
        if user_camera_id and user_camera_id.lower() != "auto":
            if user_camera_id in self.zone_manager.camera_configs:
                return user_camera_id

        name = Path(input_path).stem.lower()
        if "cam2" in name:
            return "cam2"
        elif "chute09" in name:
            return "cam1_chute09"
        else:
            return "cam1"  # Default calibrated MCFD camera

    def draw_pipeline_hud(
        self,
        frame: np.ndarray,
        frame_idx: int,
        total_frames: int,
        fps: float,
        camera_id: str,
        temporal_state: TemporalState,
        posture_label: str,
        zone_name: Optional[str],
        has_person: bool,
        elapsed_down_sec: float,
        active_candidate_event: bool,
    ) -> np.ndarray:
        """
        Renders a high-contrast, professional telemetry HUD at the top and bottom of the frame.
        """
        h, w = frame.shape[:2]
        out = frame.copy()

        # Top Bar (Telemetry Header)
        top_h = 44
        top_sub = out[0:top_h, 0:w]
        top_bar = np.zeros_like(top_sub)
        cv2.addWeighted(top_sub, 0.25, top_bar, 0.75, 0, top_sub)
        out[0:top_h, 0:w] = top_sub

        # Left Info: Camera, Frame, Time
        time_sec = frame_idx / max(1.0, self.tracker_config.fps)
        m, s = divmod(int(time_sec), 60)
        time_str = f"{m:02d}:{s:02d}"
        info_str = f"CAM: {camera_id.upper()} | FRAME: {frame_idx:04d}/{total_frames} ({time_str}) | FPS: {fps:.1f}"
        cv2.putText(out, info_str, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 1, cv2.LINE_AA)

        # Right Badge: Current Temporal State
        if temporal_state == TemporalState.PERSON_DOWN_CANDIDATE:
            badge_text = f"ALERT: PERSON DOWN CANDIDATE ({elapsed_down_sec:.1f}s)"
            badge_color = (30, 30, 235)  # Vivid Red
            text_color = (255, 255, 255)
        elif temporal_state == TemporalState.POSSIBLE_PERSON_DOWN:
            badge_text = f"MONITORING FLOOR EVIDENCE ({elapsed_down_sec:.1f}s)"
            badge_color = (30, 180, 240)  # Amber / Orange
            text_color = (0, 0, 0)
        elif temporal_state == TemporalState.UNKNOWN:
            badge_text = "STATE: UNKNOWN (EVIDENCE GAP)"
            badge_color = (130, 130, 130)  # Gray
            text_color = (255, 255, 255)
        else:
            badge_text = "STATUS: NORMAL ACTIVITY"
            badge_color = (50, 190, 50)  # Emerald Green
            text_color = (255, 255, 255)

        t_size = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)[0]
        bx1 = w - t_size[0] - 22
        by1 = 6
        bx2 = w - 10
        by2 = top_h - 6
        cv2.rectangle(out, (bx1, by1), (bx2, by2), badge_color, -1)
        cv2.rectangle(out, (bx1, by1), (bx2, by2), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(out, badge_text, (bx1 + 6, by2 - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.52, text_color, 2, cv2.LINE_AA)

        # Bottom Telemetry Strip (Subject Status)
        bot_h = 32
        bot_sub = out[h - bot_h:h, 0:w]
        bot_bar = np.zeros_like(bot_sub)
        cv2.addWeighted(bot_sub, 0.35, bot_bar, 0.65, 0, bot_sub)
        out[h - bot_h:h, 0:w] = bot_sub

        if not has_person:
            subj_str = "DETECTION: NO PERSON IN FIELD OF VIEW (STATE MEMORY ACTIVE)"
            subj_col = (180, 180, 180)
        else:
            z_display = zone_name if zone_name else "OUTSIDE CONFIGURED ZONES"
            subj_str = f"SUBJECT: {posture_label} | ZONE: {z_display} | DOWN TIMER: {elapsed_down_sec:.1f}s"
            subj_col = (0, 255, 255) if posture_label == "LYING" else (220, 220, 220)

        cv2.putText(out, subj_str, (14, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.46, subj_col, 1, cv2.LINE_AA)

        # Active candidate event pulsating red border
        if active_candidate_event:
            cv2.rectangle(out, (2, 2), (w - 3, h - 3), (0, 0, 255), 4, cv2.LINE_AA)

        return out

    @staticmethod
    def _init_video_writer(
        output_path: Union[str, Path],
        fps: float,
        width: int,
        height: int,
    ) -> Optional[cv2.VideoWriter]:
        """Initializes a VideoWriter configured for HTML5 browser-compatible H.264 (avc1) first,
        falling back to standard H.264, and then mp4v if H.264 encoding is unavailable."""
        resolved_path = str(Path(output_path).resolve())
        candidates = []
        if os.name == "nt" and hasattr(cv2, "CAP_MSMF"):
            candidates.append((cv2.CAP_MSMF, "avc1"))
        candidates.extend([
            (None, "avc1"),
            (None, "H264"),
            (None, "mp4v"),
        ])

        for backend, tag in candidates:
            fourcc = cv2.VideoWriter_fourcc(*tag)
            try:
                if backend is not None:
                    writer = cv2.VideoWriter(resolved_path, backend, fourcc, fps, (width, height))
                else:
                    writer = cv2.VideoWriter(resolved_path, fourcc, fps, (width, height))
                if writer.isOpened():
                    logger.info(f"Initialized VideoWriter with codec '{tag}' (backend={backend}) at {resolved_path}")
                    return writer
            except Exception as e:
                logger.debug(f"VideoWriter attempt failed for {tag} ({backend}): {e}")

        logger.warning(f"Could not open VideoWriter with any codec for {resolved_path}")
        return None

    def process_video(
        self,
        input_video_path: Union[str, Path],
        output_video_path: Optional[Union[str, Path]] = None,
        camera_id: str = "auto",
        max_frames: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int, float, str, int, float], None]] = None,
        frame_callback: Optional[Callable[[int, np.ndarray, Dict[str, Any]], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> Dict[str, Any]:
        """
        Executes end-to-end video processing through the detection, zone, temporal, and annotation pipeline.

        Args:
            input_video_path: Path to source video file.
            output_video_path: Destination path for annotated MP4 video.
            camera_id: Camera zone config key ('cam1', 'cam2', or 'auto').
            max_frames: Optional cap on frames to process.
            progress_callback: Callback(current_frame, total_frames, fps, current_state, event_count, percent_complete).
            frame_callback: Callback(frame_idx, frame_bgr, telemetry_dict) for live display.
            cancel_check: Function returning True if execution should abort early.

        Returns:
            Structured processing report dictionary.
        """
        # 1. Input Validation
        is_valid, err_msg, meta = self.validate_video_file(input_video_path)
        if not is_valid:
            raise ValueError(f"Video input validation failed: {err_msg}")

        input_path = Path(input_video_path)
        cam_cfg_id = self.resolve_camera_config_id(input_path, camera_id)

        # 2. Output Path Setup
        if output_video_path is None:
            out_dir = Path("data/processed/phase7")
            out_dir.mkdir(parents=True, exist_ok=True)
            output_path = out_dir / f"annotated_{input_path.stem}.mp4"
        else:
            output_path = Path(output_video_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

        # 3. Initialize Fresh Temporal Tracker & Smoother for this video
        tracker = TemporalStateTracker(self.tracker_config)
        if self.smoother is not None:
            self.smoother.reset()

        cap = cv2.VideoCapture(str(input_path.resolve()))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        container_fps = cap.get(cv2.CAP_PROP_FPS)
        nominal_fps = self.tracker_config.fps
        # MCFD files have 120.0 in AVI header; real-world capture rate is 30.0 FPS
        video_fps = nominal_fps if (container_fps > 60.0 or container_fps <= 0 or np.isnan(container_fps)) else float(container_fps)
        frames_to_process = total_frames if (max_frames is None or max_frames <= 0) else min(total_frames, max_frames)

        # 4. Initialize Video Writer with Codec Fallbacks (H.264 / avc1 preferred for HTML5 browser playback)
        writer = self._init_video_writer(output_path, video_fps, width, height)
        if not writer or not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Failed to create video writer at: {output_path.resolve()}")

        # 5. Get Camera Zone Polygons for Overlay (scaled to match frame dimensions)
        cam_zones = self.zone_manager.get_scaled_zones(cam_cfg_id, (width, height))

        # Processing Metrics Accumulators
        processed_frames = 0
        frames_with_person = 0
        posture_counts: Dict[str, int] = {
            PostureType.STANDING.value: 0,
            PostureType.SITTING.value: 0,
            PostureType.LYING.value: 0,
            PostureType.UNKNOWN.value: 0,
        }
        zone_counts: Dict[str, int] = {
            ZoneType.FLOOR_ZONE.value: 0,
            ZoneType.BED_SOFA_ZONE.value: 0,
            ZoneType.OUTSIDE_ZONE.value: 0,
        }

        # Event History Tracking
        detected_events: List[DetectedCandidateEvent] = []
        active_event: Optional[DetectedCandidateEvent] = None
        current_state = TemporalState.NORMAL
        start_time = time.perf_counter()
        rolling_fps = video_fps

        logger.info(
            f"Starting Phase 7 processing: '{input_path.name}' ({frames_to_process} frames) "
            f"using camera config '{cam_cfg_id}' on {self.detector.get_device_info()}"
        )

        try:
            while processed_frames < frames_to_process:
                if cancel_check and cancel_check():
                    logger.warning("Processing cancelled by user request.")
                    break

                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                frame_start = time.perf_counter()

                # A. YOLO11-Pose Detection
                persons = self.detector.detect(frame)
                num_persons = len(persons)

                if num_persons == 0:
                    has_person = False
                    posture_label = "NO_DETECTION"
                    posture_conf = 0.0
                    zone_type = ZoneType.OUTSIDE_ZONE.value
                    zone_name = None
                    event_classification = "NO_DETECTION"
                    if self.smoother is not None:
                        self.smoother.reset()
                    primary_zone_res = None
                else:
                    has_person = True
                    frames_with_person += 1
                    # Primary subject selection
                    best_person = max(persons, key=lambda p: p.confidence)

                    # B. Posture Classification & Smoothing (Phase 6 enhancements)
                    p_res = self.posture_classifier.classify(best_person.keypoints, best_person.box)
                    if self.smoother is not None:
                        p_res = self.smoother.update(p_res)
                    posture_label = p_res.label
                    posture_conf = float(p_res.confidence)
                    best_person.posture = posture_label
                    best_person.posture_confidence = posture_conf

                    posture_counts[posture_label] = posture_counts.get(posture_label, 0) + 1

                    # C. Spatial Zone Classification (with resolution-aware scaling)
                    z_res = self.zone_manager.classify_zone(
                        camera_id=cam_cfg_id,
                        posture=posture_label,
                        keypoints=best_person.keypoints,
                        box=best_person.box,
                        frame_dimensions=(width, height),
                    )
                    zone_type = z_res.zone_type.value
                    zone_name = z_res.zone_name
                    event_classification = z_res.event_classification.value
                    zone_counts[zone_type] = zone_counts.get(zone_type, 0) + 1
                    primary_zone_res = z_res

                # D. Temporal State Tracker Update
                t_upd = tracker.update(
                    frame_idx=processed_frames,
                    event_classification=event_classification,
                    posture=posture_label,
                    has_person=has_person,
                    zone_type=zone_type,
                )
                current_state = t_upd.state

                # E. Discrete Event Management (Duplicate Prevention)
                is_candidate = (current_state == TemporalState.PERSON_DOWN_CANDIDATE)

                if t_upd.is_new_candidate_event:
                    # New distinct candidate event triggered
                    ev_id = len(detected_events) + 1
                    start_f = max(0, processed_frames - t_upd.accumulated_floor_frames + 1)
                    active_event = DetectedCandidateEvent(
                        event_id=ev_id,
                        start_frame=start_f,
                        trigger_frame=processed_frames,
                        end_frame=processed_frames,
                        start_time_seconds=start_f / max(1.0, video_fps),
                        trigger_time_seconds=processed_frames / max(1.0, video_fps),
                        end_time_seconds=processed_frames / max(1.0, video_fps),
                        duration_seconds=round(t_upd.elapsed_down_seconds, 2),
                        duration_frames=1,
                        status="ACTIVE",
                        zone_name=zone_name,
                        peak_posture=posture_label,
                        peak_confidence=posture_conf,
                    )
                    detected_events.append(active_event)
                    logger.warning(
                        f"ALERT TRIGGERED: Event #{ev_id} at frame {processed_frames} "
                        f"({active_event.trigger_time_seconds:.2f}s) in zone '{zone_name}'"
                    )

                elif active_event is not None and is_candidate:
                    # Continue existing active candidate event
                    active_event.end_frame = processed_frames
                    active_event.end_time_seconds = processed_frames / max(1.0, video_fps)
                    active_event.duration_frames = active_event.end_frame - active_event.start_frame + 1
                    active_event.duration_seconds = active_event.duration_frames / max(1.0, video_fps)
                    if posture_conf > active_event.peak_confidence:
                        active_event.peak_confidence = posture_conf

                elif active_event is not None and not is_candidate:
                    # Event resolved (recovered to upright, safe furniture, or reset)
                    active_event.status = "RESOLVED"
                    logger.info(
                        f"Event #{active_event.event_id} resolved at frame {processed_frames}. "
                        f"Total duration: {active_event.duration_seconds:.2f}s"
                    )
                    active_event = None

                # F. Visual Annotation
                annotated = frame.copy()

                # 1. Semi-transparent Zone Polygons
                if cam_zones:
                    annotated = self.zone_visualizer.draw_zones(annotated, cam_zones, draw_labels=True)

                # 2. Person Skeletons, BBoxes, Reference Points
                if has_person and num_persons > 0:
                    for p_idx, person in enumerate(persons):
                        self.annotator.draw_bounding_box(annotated, person, p_idx)
                        self.annotator.draw_skeleton(annotated, person.keypoints)

                    if primary_zone_res is not None:
                        annotated = self.zone_visualizer.draw_reference_point(annotated, primary_zone_res)

                # 3. Telemetry HUD Overlay
                annotated = self.draw_pipeline_hud(
                    frame=annotated,
                    frame_idx=processed_frames,
                    total_frames=frames_to_process,
                    fps=rolling_fps,
                    camera_id=cam_cfg_id,
                    temporal_state=current_state,
                    posture_label=posture_label,
                    zone_name=zone_name,
                    has_person=has_person,
                    elapsed_down_sec=t_upd.elapsed_down_seconds,
                    active_candidate_event=(active_event is not None and active_event.status == "ACTIVE"),
                )

                # 4. Write Frame
                writer.write(annotated)

                # Calculate instantaneous rolling FPS
                frame_dur = time.perf_counter() - frame_start
                if frame_dur > 0:
                    inst_fps = 1.0 / frame_dur
                    rolling_fps = 0.9 * rolling_fps + 0.1 * inst_fps

                processed_frames += 1

                # Telemetry dictionary for callbacks
                telemetry = {
                    "frame": processed_frames,
                    "total_frames": frames_to_process,
                    "fps": round(rolling_fps, 1),
                    "temporal_state": current_state.value,
                    "posture": posture_label,
                    "posture_confidence": round(posture_conf, 4),
                    "zone_type": zone_type,
                    "zone_name": zone_name,
                    "has_person": has_person,
                    "elapsed_down_seconds": round(t_upd.elapsed_down_seconds, 2),
                    "active_events_count": len(detected_events),
                    "is_active_candidate": (active_event is not None),
                }

                if progress_callback:
                    pct = round((processed_frames / max(1, frames_to_process)) * 100, 1)
                    progress_callback(
                        processed_frames,
                        frames_to_process,
                        rolling_fps,
                        current_state.value,
                        len(detected_events),
                        pct,
                    )

                if frame_callback:
                    frame_callback(processed_frames, annotated, telemetry)

        finally:
            cap.release()
            writer.release()

        # Wrap up any active candidate event at end of video
        if active_event is not None:
            active_event.status = "RESOLVED"
            active_event = None

        elapsed_total = max(0.001, time.perf_counter() - start_time)
        overall_avg_fps = round(processed_frames / elapsed_total, 2)

        # 6. Verify and Package Output Report
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError(f"Annotated output video is missing or corrupt at: {output_path.resolve()}")

        posture_dist = {
            k: round(v / max(1, frames_with_person), 4)
            for k, v in posture_counts.items()
        }
        zone_dist = {
            k: round(v / max(1, frames_with_person), 4)
            for k, v in zone_counts.items()
        }

        report = {
            "status": "COMPLETED",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "input_video": {
                "filename": input_path.name,
                "path": str(input_path.resolve()),
                "camera_config_id": cam_cfg_id,
            },
            "output_video": {
                "filename": output_path.name,
                "path": str(output_path.resolve()),
                "size_bytes": output_path.stat().st_size,
            },
            "video_metadata": {
                "width": width,
                "height": height,
                "fps": round(video_fps, 2),
                "total_frames": total_frames,
                "processed_frames": processed_frames,
                "duration_seconds": round(processed_frames / max(1.0, video_fps), 2),
            },
            "processing_performance": {
                "elapsed_time_seconds": round(elapsed_total, 2),
                "average_fps": overall_avg_fps,
                "hardware_device": self.detector.get_device_info(),
            },
            "detection_metrics": {
                "frames_with_person": frames_with_person,
                "person_presence_ratio": round(frames_with_person / max(1, processed_frames), 4),
                "posture_counts": posture_counts,
                "posture_distribution": posture_dist,
                "zone_counts": zone_counts,
                "zone_distribution": zone_dist,
            },
            "temporal_summary": {
                "final_state": current_state.value,
                "total_candidate_events_triggered": len(detected_events),
                "detected_events": [ev.to_dict() for ev in detected_events],
            },
            "disclaimer": RESEARCH_PROTOTYPE_DISCLAIMER,
        }

        logger.info(
            f"Pipeline finished: {processed_frames} frames processed in {elapsed_total:.1f}s "
            f"({overall_avg_fps} FPS). Total candidate events: {len(detected_events)}."
        )
        return report
