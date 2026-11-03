"""
Spatial Zone Module for Bed / Sofa vs Floor Disambiguation.

Phase 3.2: Spatial Zone Configuration and Bed/Floor Disambiguation.
Distinguishes furniture resting from floor collapse candidates using camera-specific
polygon boundaries and keypoint/bounding-box reference point derivations.
Preserves all Phase 1, Phase 2, and Phase 3.1 code, weights, and outputs.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
import cv2
import numpy as np
import yaml

logger = logging.getLogger(__name__)


class ZoneType(str, Enum):
    """Types of spatial zones configurable per camera."""
    BED_SOFA_ZONE = "BED_SOFA_ZONE"
    FLOOR_ZONE = "FLOOR_ZONE"
    OUTSIDE_ZONE = "OUTSIDE_ZONE"


class ZoneEventClassification(str, Enum):
    """
    Semantic event classifications combining posture and spatial location.
    Explicitly separates emergency candidate detection from normal activities and safe resting.
    """
    NORMAL_ACTIVITY = "NORMAL_ACTIVITY"              # Posture is NOT LYING (STANDING, SITTING, UNKNOWN)
    BED_SOFA_LYING = "BED_SOFA_LYING"                # Posture is LYING and inside BED/SOFA_ZONE (Safe resting)
    FLOOR_LYING_CANDIDATE = "FLOOR_LYING_CANDIDATE"  # Posture is LYING and inside FLOOR_ZONE (Collapse candidate)
    UNKNOWN_ZONE_LYING = "UNKNOWN_ZONE_LYING"        # Posture is LYING but outside configured zones or ambiguous


class ReferencePointSource(str, Enum):
    """Source used to derive the person reference point."""
    MID_HIP = "MID_HIP"                      # Primary: Pelvic core from COCO hips (keypoints 11, 12)
    TORSO_MID = "TORSO_MID"                  # Secondary: Midpoint between shoulder center and available hip
    BBOX_CENTER = "BBOX_CENTER"              # Tertiary: Bounding box centroid (x1+x2)/2, (y1+y2)/2
    BBOX_BOTTOM_MID = "BBOX_BOTTOM_MID"      # Ground contact point (x1+x2)/2, y2 for upright postures
    UNRELIABLE = "UNRELIABLE"                # Keypoints/box invalid, out of frame, or NaN


@dataclass
class PersonReferencePoint:
    """Represents a computed 2D reference point on the image plane."""
    x: float
    y: float
    source: ReferencePointSource
    confidence: float
    is_valid: bool = True


@dataclass
class PolygonZone:
    """A 2D polygonal region defined in pixel coordinates."""
    name: str
    zone_type: ZoneType
    polygon: np.ndarray  # Shape (N, 2), dtype int32
    description: str = ""

    def contains_point(self, point: Tuple[float, float]) -> bool:
        """
        Tests whether a 2D point (x, y) is inside or on the boundary of this polygon.
        Returns True if measure >= 0 (inside or on edge).
        """
        if len(self.polygon) < 3:
            return False
        res = cv2.pointPolygonTest(self.polygon, (float(point[0]), float(point[1])), False)
        return res >= 0


@dataclass
class CameraZoneConfig:
    """Camera-specific spatial zones configuration."""
    camera_id: str
    resolution: Tuple[int, int]
    status: str
    zones: List[PolygonZone] = field(default_factory=list)


@dataclass
class ZoneResult:
    """Outcome of spatial zone classification for a single person detection."""
    zone_type: ZoneType
    zone_name: Optional[str]
    event_classification: ZoneEventClassification
    reference_point: PersonReferencePoint
    camera_id: str
    is_validated_camera: bool
    details: Dict[str, Any] = field(default_factory=dict)


class ZoneManager:
    """
    Manages camera-specific zone configurations and performs point-in-polygon
    reference point evaluation and event disambiguation.
    """

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        self.camera_configs: Dict[str, CameraZoneConfig] = {}
        self.default_outside_behavior: str = "UNKNOWN_ZONE"
        self.default_overlap_behavior: str = "UNKNOWN_ZONE"
        self.min_keypoint_conf: float = 0.30

        if config_path is not None:
            self.load_config(config_path)

    def load_config(self, config_path: Union[str, Path]) -> None:
        """Loads and parses camera-specific polygon configurations from YAML."""
        p = Path(config_path)
        if not p.exists():
            raise FileNotFoundError(f"Zone config file does not exist: {p.resolve()}")

        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        policy = data.get("default_policy", {})
        self.default_outside_behavior = policy.get("outside_zone_behavior", "UNKNOWN_ZONE")
        self.default_overlap_behavior = policy.get("overlap_behavior", "UNKNOWN_ZONE")
        self.min_keypoint_conf = float(policy.get("reference_point_min_confidence", 0.30))

        self.camera_configs.clear()
        for cam_id, cam_data in data.get("cameras", {}).items():
            res = tuple(cam_data.get("resolution", [720, 480]))
            status = cam_data.get("status", "UNVALIDATED")
            zones_list = []

            for z in cam_data.get("zones", []):
                poly_pts = np.array(z.get("polygon", []), dtype=np.int32)
                z_type_str = z.get("type", "OUTSIDE_ZONE")
                try:
                    z_type = ZoneType(z_type_str)
                except ValueError:
                    z_type = ZoneType.OUTSIDE_ZONE

                zone_obj = PolygonZone(
                    name=z.get("name", "unnamed_zone"),
                    zone_type=z_type,
                    polygon=poly_pts,
                    description=z.get("description", ""),
                )
                zones_list.append(zone_obj)

            self.camera_configs[cam_id] = CameraZoneConfig(
                camera_id=cam_id,
                resolution=res,
                status=status,
                zones=zones_list,
            )

        logger.info(f"Loaded zone configurations for {len(self.camera_configs)} cameras from {p.name}")

    def derive_reference_point(
        self,
        keypoints: Optional[np.ndarray],
        box: Union[np.ndarray, Tuple[float, float, float, float], List[float]],
        posture: str,
        frame_shape: Optional[Tuple[int, int]] = None,
    ) -> PersonReferencePoint:
        """
        Derives person reference point based on body posture and keypoint reliability.
        COCO keypoint indices:
          5: left_shoulder, 6: right_shoulder
          11: left_hip, 12: right_hip
        """
        # Validate bounding box
        if box is None or len(box) < 4:
            return PersonReferencePoint(0.0, 0.0, ReferencePointSource.UNRELIABLE, 0.0, is_valid=False)

        x1, y1, x2, y2 = [float(v) for v in box[:4]]
        if np.isnan(x1) or np.isnan(y1) or np.isnan(x2) or np.isnan(y2) or x2 <= x1 or y2 <= y1:
            return PersonReferencePoint(0.0, 0.0, ReferencePointSource.UNRELIABLE, 0.0, is_valid=False)

        bbox_center_x = (x1 + x2) / 2.0
        bbox_center_y = (y1 + y2) / 2.0
        bbox_bottom_x = (x1 + x2) / 2.0
        bbox_bottom_y = y2

        # For upright postures (STANDING, SITTING), bottom-center represents ground surface contact
        if posture in ("STANDING", "SITTING"):
            return PersonReferencePoint(
                x=bbox_bottom_x,
                y=bbox_bottom_y,
                source=ReferencePointSource.BBOX_BOTTOM_MID,
                confidence=1.0,
                is_valid=True,
            )

        # For horizontal / ambiguous postures (LYING, UNKNOWN), derive pelvic core center
        if keypoints is not None and isinstance(keypoints, np.ndarray) and keypoints.shape[0] >= 17:
            l_hip = keypoints[11]
            r_hip = keypoints[12]
            l_conf = float(l_hip[2]) if len(l_hip) > 2 else 0.0
            r_conf = float(r_hip[2]) if len(r_hip) > 2 else 0.0

            # 1. Primary: Both hips detected reliably (mid-hip pelvis point)
            if l_conf >= self.min_keypoint_conf and r_conf >= self.min_keypoint_conf:
                mid_x = (float(l_hip[0]) + float(r_hip[0])) / 2.0
                mid_y = (float(l_hip[1]) + float(r_hip[1])) / 2.0
                avg_conf = (l_conf + r_conf) / 2.0
                return PersonReferencePoint(
                    x=mid_x,
                    y=mid_y,
                    source=ReferencePointSource.MID_HIP,
                    confidence=avg_conf,
                    is_valid=True,
                )

            # 2. Secondary: One hip reliable + shoulders reliable (torso midpoint)
            l_sho = keypoints[5]
            r_sho = keypoints[6]
            sho_conf = (float(l_sho[2]) + float(r_sho[2])) / 2.0 if len(l_sho) > 2 and len(r_sho) > 2 else 0.0

            if sho_conf >= self.min_keypoint_conf:
                sho_mid_x = (float(l_sho[0]) + float(r_sho[0])) / 2.0
                sho_mid_y = (float(l_sho[1]) + float(r_sho[1])) / 2.0

                if l_conf >= self.min_keypoint_conf:
                    torso_x = (sho_mid_x + float(l_hip[0])) / 2.0
                    torso_y = (sho_mid_y + float(l_hip[1])) / 2.0
                    return PersonReferencePoint(torso_x, torso_y, ReferencePointSource.TORSO_MID, (sho_conf + l_conf) / 2.0, True)
                elif r_conf >= self.min_keypoint_conf:
                    torso_x = (sho_mid_x + float(r_hip[0])) / 2.0
                    torso_y = (sho_mid_y + float(r_hip[1])) / 2.0
                    return PersonReferencePoint(torso_x, torso_y, ReferencePointSource.TORSO_MID, (sho_conf + r_conf) / 2.0, True)

        # 3. Tertiary Fallback: Bounding box centroid
        return PersonReferencePoint(
            x=bbox_center_x,
            y=bbox_center_y,
            source=ReferencePointSource.BBOX_CENTER,
            confidence=0.5,
            is_valid=True,
        )

    def get_scaled_zones(
        self,
        camera_id: str,
        frame_dimensions: Optional[Union[Tuple[int, int], Tuple[int, int, int], List[int]]] = None,
    ) -> List[PolygonZone]:
        """
        Returns camera zones with polygon vertices scaled to the target frame dimensions.
        Preserves unscaled polygons if dimensions match or are not specified.
        """
        if camera_id not in self.camera_configs:
            return []
        cam_config = self.camera_configs[camera_id]
        if frame_dimensions is None:
            return cam_config.zones

        config_w, config_h = cam_config.resolution
        try:
            if len(frame_dimensions) == 3:
                frame_h, frame_w = float(frame_dimensions[0]), float(frame_dimensions[1])
            elif len(frame_dimensions) == 2:
                frame_w, frame_h = float(frame_dimensions[0]), float(frame_dimensions[1])
            else:
                return cam_config.zones
        except (ValueError, TypeError, IndexError):
            return cam_config.zones

        if frame_w <= 0 or frame_h <= 0 or (frame_w == config_w and frame_h == config_h):
            return cam_config.zones

        scale_x = frame_w / float(config_w)
        scale_y = frame_h / float(config_h)

        scaled_zones = []
        for z in cam_config.zones:
            scaled_poly = np.array(
                [[int(round(pt[0] * scale_x)), int(round(pt[1] * scale_y))] for pt in z.polygon],
                dtype=np.int32,
            )
            scaled_zones.append(
                PolygonZone(
                    name=z.name,
                    zone_type=z.zone_type,
                    polygon=scaled_poly,
                    description=z.description,
                )
            )
        return scaled_zones

    def classify_zone(
        self,
        camera_id: str,
        posture: str,
        keypoints: Optional[np.ndarray],
        box: Union[np.ndarray, Tuple[float, float, float, float], List[float]],
        frame_dimensions: Optional[Union[Tuple[int, int], Tuple[int, int, int], List[int]]] = None,
    ) -> ZoneResult:
        """
        Classifies spatial location and event candidate for a detected person.
        Follows strict Phase 3.2 rules:
        - Non-LYING -> NORMAL_ACTIVITY.
        - LYING in BED_SOFA_ZONE -> BED_SOFA_LYING.
        - LYING in FLOOR_ZONE -> FLOOR_LYING_CANDIDATE.
        - LYING outside / invalid / ambiguous overlap -> UNKNOWN_ZONE_LYING.

        Supports resolution-aware scaling from actual video-frame dimensions into
        configured zone resolution (e.g. 1200x896 -> 720x480).
        """
        ref_pt = self.derive_reference_point(keypoints, box, posture)

        # Check camera configuration
        if camera_id not in self.camera_configs:
            logger.debug(f"Camera ID '{camera_id}' not found in zone configuration.")
            return ZoneResult(
                zone_type=ZoneType.OUTSIDE_ZONE,
                zone_name=None,
                event_classification=(
                    ZoneEventClassification.NORMAL_ACTIVITY
                    if posture != "LYING"
                    else ZoneEventClassification.UNKNOWN_ZONE_LYING
                ),
                reference_point=ref_pt,
                camera_id=camera_id,
                is_validated_camera=False,
                details={"reason": "camera_not_configured"},
            )

        cam_config = self.camera_configs[camera_id]
        is_validated = (cam_config.status == "VALIDATED")

        # Unvalidated camera (e.g. chute09/cam1 requiring manual review)
        if not is_validated:
            return ZoneResult(
                zone_type=ZoneType.OUTSIDE_ZONE,
                zone_name=None,
                event_classification=(
                    ZoneEventClassification.NORMAL_ACTIVITY
                    if posture != "LYING"
                    else ZoneEventClassification.UNKNOWN_ZONE_LYING
                ),
                reference_point=ref_pt,
                camera_id=camera_id,
                is_validated_camera=False,
                details={"status": cam_config.status, "reason": "camera_unvalidated"},
            )

        # If reference point could not be computed reliably
        if not ref_pt.is_valid:
            return ZoneResult(
                zone_type=ZoneType.OUTSIDE_ZONE,
                zone_name=None,
                event_classification=(
                    ZoneEventClassification.NORMAL_ACTIVITY
                    if posture != "LYING"
                    else ZoneEventClassification.UNKNOWN_ZONE_LYING
                ),
                reference_point=ref_pt,
                camera_id=camera_id,
                is_validated_camera=is_validated,
                details={"reason": "unreliable_reference_point"},
            )

        # ---------------------------------------------------------------------
        # Resolution-Aware Coordinate Mapping
        # Map reference point from actual video resolution into configured zone resolution
        # ---------------------------------------------------------------------
        config_width, config_height = cam_config.resolution
        frame_width: Optional[float] = None
        frame_height: Optional[float] = None

        if frame_dimensions is not None:
            try:
                if len(frame_dimensions) == 3:
                    # OpenCV frame.shape: (height, width, channels)
                    frame_height = float(frame_dimensions[0])
                    frame_width = float(frame_dimensions[1])
                elif len(frame_dimensions) == 2:
                    # Standard resolution tuple: (width, height)
                    frame_width = float(frame_dimensions[0])
                    frame_height = float(frame_dimensions[1])
                else:
                    logger.warning(
                        f"Unexpected frame_dimensions format: {frame_dimensions}. Expected (width, height) or (H, W, C)."
                    )
            except (ValueError, TypeError, IndexError) as err:
                logger.warning(f"Failed to parse frame_dimensions {frame_dimensions}: {err}")

        scale_x = 1.0
        scale_y = 1.0
        scaling_details: Dict[str, Any] = {
            "original_reference_point": (round(float(ref_pt.x), 2), round(float(ref_pt.y), 2)),
        }

        if frame_width is not None and frame_height is not None and frame_width > 0 and frame_height > 0:
            config_ar = float(config_width) / float(config_height)
            frame_ar = frame_width / frame_height

            # Safety Policy: If orientation is fundamentally incompatible (portrait vs landscape),
            # do not silently assume calibration is valid; treat as calibration ambiguity.
            is_portrait_vs_landscape = (frame_ar < 1.0 and config_ar >= 1.0) or (frame_ar >= 1.0 and config_ar < 1.0)
            if is_portrait_vs_landscape:
                logger.warning(
                    f"Incompatible video aspect ratio ({frame_ar:.2f}) for camera '{camera_id}' "
                    f"(calibrated for {config_ar:.2f}). Treating as ambiguous calibration."
                )
                return ZoneResult(
                    zone_type=ZoneType.OUTSIDE_ZONE,
                    zone_name=None,
                    event_classification=(
                        ZoneEventClassification.NORMAL_ACTIVITY
                        if posture != "LYING"
                        else ZoneEventClassification.UNKNOWN_ZONE_LYING
                    ),
                    reference_point=ref_pt,
                    camera_id=camera_id,
                    is_validated_camera=is_validated,
                    details={
                        "reason": "aspect_ratio_incompatible",
                        "aspect_ratio_incompatible": True,
                        "frame_aspect_ratio": round(frame_ar, 3),
                        "config_aspect_ratio": round(config_ar, 3),
                        "frame_dimensions": (int(frame_width), int(frame_height)),
                        "config_resolution": (config_width, config_height),
                    },
                )

            scale_x = float(config_width) / frame_width
            scale_y = float(config_height) / frame_height
            scaling_details["frame_dimensions"] = (int(frame_width), int(frame_height))
            scaling_details["config_resolution"] = (config_width, config_height)
            scaling_details["scale_factors"] = (round(scale_x, 4), round(scale_y, 4))
            ar_diff = abs(frame_ar - config_ar) / config_ar
            if ar_diff > 0.05:
                scaling_details["aspect_ratio_delta"] = round(ar_diff, 3)
        else:
            scaling_details["scale_factors"] = (1.0, 1.0)

        # Scaled point used strictly for polygon containment check
        x_scaled = ref_pt.x * scale_x
        y_scaled = ref_pt.y * scale_y
        scaling_details["scaled_reference_point"] = (round(float(x_scaled), 2), round(float(y_scaled), 2))

        pt_for_test = (float(x_scaled), float(y_scaled))
        matched_zones: List[PolygonZone] = []
        for zone in cam_config.zones:
            if zone.contains_point(pt_for_test):
                matched_zones.append(zone)

        # Case 1: Point is in overlapping polygons
        if len(matched_zones) > 1:
            types = set(z.zone_type for z in matched_zones)
            if len(types) > 1:
                # Genuinely ambiguous overlap -> treat as UNKNOWN_ZONE
                logger.debug(f"Ambiguous boundary overlap detected at {pt_for_test} between {[z.name for z in matched_zones]}")
                details = dict(scaling_details)
                details["overlapping_zones"] = [z.name for z in matched_zones]
                return ZoneResult(
                    zone_type=ZoneType.OUTSIDE_ZONE,
                    zone_name="AMBIGUOUS_OVERLAP",
                    event_classification=(
                        ZoneEventClassification.NORMAL_ACTIVITY
                        if posture != "LYING"
                        else ZoneEventClassification.UNKNOWN_ZONE_LYING
                    ),
                    reference_point=ref_pt,
                    camera_id=camera_id,
                    is_validated_camera=is_validated,
                    details=details,
                )

        # Case 2: Exactly one matching zone
        if len(matched_zones) >= 1:
            matched = matched_zones[0]
            if posture != "LYING":
                event_cls = ZoneEventClassification.NORMAL_ACTIVITY
            elif matched.zone_type == ZoneType.BED_SOFA_ZONE:
                event_cls = ZoneEventClassification.BED_SOFA_LYING
            elif matched.zone_type == ZoneType.FLOOR_ZONE:
                event_cls = ZoneEventClassification.FLOOR_LYING_CANDIDATE
            else:
                event_cls = ZoneEventClassification.UNKNOWN_ZONE_LYING

            details = dict(scaling_details)
            details["matched_zone"] = matched.name
            return ZoneResult(
                zone_type=matched.zone_type,
                zone_name=matched.name,
                event_classification=event_cls,
                reference_point=ref_pt,
                camera_id=camera_id,
                is_validated_camera=is_validated,
                details=details,
            )

        # Case 3: Outside all configured zones
        # Rule 8: Never silently default to FLOOR_ZONE!
        details = dict(scaling_details)
        details["reason"] = "outside_all_zones"
        return ZoneResult(
            zone_type=ZoneType.OUTSIDE_ZONE,
            zone_name=None,
            event_classification=(
                ZoneEventClassification.NORMAL_ACTIVITY
                if posture != "LYING"
                else ZoneEventClassification.UNKNOWN_ZONE_LYING
            ),
            reference_point=ref_pt,
            camera_id=camera_id,
            is_validated_camera=is_validated,
            details=details,
        )
