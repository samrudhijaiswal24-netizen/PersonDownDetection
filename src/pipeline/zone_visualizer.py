"""
Spatial Zone Visualization Module.

Phase 3.2: Spatial Zone Configuration and Bed/Floor Disambiguation.
Draws polygon zone overlays, reference point crosshairs, and disambiguation HUD banners
onto frames and video streams. Preserves original inputs and existing Phase 1/Phase 2 files.
"""

from typing import List, Optional, Tuple, Union
import cv2
import numpy as np

from src.core.spatial_zones import (
    ZoneType,
    ZoneEventClassification,
    ZoneResult,
    CameraZoneConfig,
    PolygonZone,
)


# BGR Color Palette
COLOR_SOFA_FILL = (230, 180, 50)       # Cyan/Teal (BGR)
COLOR_SOFA_BORDER = (255, 120, 0)     # Vivid Blue/Cyan border
COLOR_FLOOR_FILL = (40, 120, 240)     # Amber/Orange fill
COLOR_FLOOR_BORDER = (0, 80, 255)     # Deep Orange border

COLOR_HUD_SAFE = (220, 140, 30)       # Blue/Cyan
COLOR_HUD_NORMAL = (60, 180, 60)      # Green
COLOR_HUD_ALERT = (30, 30, 220)       # Red
COLOR_HUD_UNKNOWN = (30, 200, 220)    # Yellow


class ZoneVisualizer:
    """Visualizes polygon zones and disambiguated event telemetry on video frames."""

    def __init__(self, alpha: float = 0.22):
        self.alpha = alpha

    def draw_zones(
        self,
        frame: np.ndarray,
        zones: List[PolygonZone],
        draw_labels: bool = True,
    ) -> np.ndarray:
        """Draws semi-transparent polygon fills and dashed/solid borders on frame."""
        overlay = frame.copy()
        out_frame = frame.copy()

        for zone in zones:
            if len(zone.polygon) < 3:
                continue

            pts = zone.polygon.reshape((-1, 1, 2))
            fill_color = COLOR_SOFA_FILL if zone.zone_type == ZoneType.BED_SOFA_ZONE else COLOR_FLOOR_FILL
            border_color = COLOR_SOFA_BORDER if zone.zone_type == ZoneType.BED_SOFA_ZONE else COLOR_FLOOR_BORDER

            # Semi-transparent fill
            cv2.fillPoly(overlay, [pts], fill_color)

            # Solid boundary
            cv2.polylines(out_frame, [pts], isClosed=True, color=border_color, thickness=2, lineType=cv2.LINE_AA)

            if draw_labels:
                # Place label near top-left of polygon
                min_x = int(np.min(zone.polygon[:, 0]))
                min_y = int(np.min(zone.polygon[:, 1]))
                lbl = "BED/SOFA ZONE (SAFE)" if zone.zone_type == ZoneType.BED_SOFA_ZONE else "FLOOR ZONE (MONITORED)"
                cv2.putText(
                    out_frame,
                    f"[{zone.name.upper()}] {lbl}",
                    (min_x + 8, min_y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    out_frame,
                    f"[{zone.name.upper()}] {lbl}",
                    (min_x + 8, min_y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    border_color,
                    1,
                    cv2.LINE_AA,
                )

        # Blend fill overlay
        cv2.addWeighted(overlay, self.alpha, out_frame, 1.0 - self.alpha, 0, out_frame)
        return out_frame

    def draw_reference_point(
        self,
        frame: np.ndarray,
        zone_result: ZoneResult,
    ) -> np.ndarray:
        """Draws target crosshair and source tag for person reference point."""
        ref_pt = zone_result.reference_point
        if not ref_pt.is_valid:
            return frame

        cx, cy = int(round(ref_pt.x)), int(round(ref_pt.y))

        # Color based on event classification
        if zone_result.event_classification == ZoneEventClassification.FLOOR_LYING_CANDIDATE:
            color = (0, 0, 255)      # Red
        elif zone_result.event_classification == ZoneEventClassification.BED_SOFA_LYING:
            color = (255, 200, 0)    # Cyan
        elif zone_result.event_classification == ZoneEventClassification.UNKNOWN_ZONE_LYING:
            color = (0, 220, 255)    # Yellow
        else:
            color = (0, 255, 0)      # Green

        # Crosshair
        cv2.circle(frame, (cx, cy), 7, color, 2, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 2, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.line(frame, (cx - 11, cy), (cx + 11, cy), color, 1, cv2.LINE_AA)
        cv2.line(frame, (cx, cy - 11), (cx, cy + 11), color, 1, cv2.LINE_AA)

        # Tag
        src_tag = f"Ref: {ref_pt.source.value} ({cx},{cy})"
        cv2.putText(frame, src_tag, (cx + 12, cy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, src_tag, (cx + 12, cy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1, cv2.LINE_AA)
        return frame

    def draw_hud(
        self,
        frame: np.ndarray,
        zone_results: List[ZoneResult],
        frame_idx: int,
        fps: float,
        camera_id: str,
    ) -> np.ndarray:
        """Draws top telemetry banner displaying zone disambiguation status."""
        h, w = frame.shape[:2]
        hud_height = 42

        # Semi-transparent top bar
        sub = frame[0:hud_height, 0:w]
        bar = np.zeros_like(sub)
        cv2.addWeighted(sub, 0.25, bar, 0.75, 0, sub)
        frame[0:hud_height, 0:w] = sub

        # Header info
        info_text = f"CAM: {camera_id} | FRAME: {frame_idx:04d} | FPS: {fps:.1f}"
        cv2.putText(frame, info_text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1, cv2.LINE_AA)

        # Event status badge
        if not zone_results:
            status_text = "[NO PERSON DETECTED]"
            badge_color = (160, 160, 160)
        else:
            # Prioritize floor candidate if any person triggers it
            primary_res = min(
                zone_results,
                key=lambda r: (
                    0 if r.event_classification == ZoneEventClassification.FLOOR_LYING_CANDIDATE
                    else 1 if r.event_classification == ZoneEventClassification.UNKNOWN_ZONE_LYING
                    else 2 if r.event_classification == ZoneEventClassification.BED_SOFA_LYING
                    else 3
                ),
            )
            ev = primary_res.event_classification
            z_name = primary_res.zone_name or "OUTSIDE"

            if ev == ZoneEventClassification.FLOOR_LYING_CANDIDATE:
                status_text = f"[FLOOR COLLAPSE CANDIDATE] Zone: {z_name}"
                badge_color = COLOR_HUD_ALERT
            elif ev == ZoneEventClassification.BED_SOFA_LYING:
                status_text = f"[RESTING ON SOFA (SAFE)] Zone: {z_name}"
                badge_color = COLOR_HUD_SAFE
            elif ev == ZoneEventClassification.UNKNOWN_ZONE_LYING:
                status_text = f"[LYING IN UNCONFIGURED ZONE] Zone: {z_name}"
                badge_color = COLOR_HUD_UNKNOWN
            else:
                status_text = f"[NORMAL ACTIVITY] Zone: {z_name}"
                badge_color = COLOR_HUD_NORMAL

        text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0]
        tx = w - text_size[0] - 14
        cv2.putText(frame, status_text, (tx, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, status_text, (tx, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.55, badge_color, 2, cv2.LINE_AA)
        return frame

    def render_calibration_preview(
        self,
        base_image: np.ndarray,
        camera_config: CameraZoneConfig,
        sample_detections: Optional[List[ZoneResult]] = None,
        title: str = "Zone Calibration Preview",
    ) -> np.ndarray:
        """Produces a pristine calibration preview image showing polygons, buffer gap, and labels."""
        preview = base_image.copy()
        preview = self.draw_zones(preview, camera_config.zones, draw_labels=True)

        if sample_detections:
            for z_res in sample_detections:
                preview = self.draw_reference_point(preview, z_res)

        # Header banner
        h, w = preview.shape[:2]
        header_bar = preview[0:40, 0:w]
        dark = np.zeros_like(header_bar)
        cv2.addWeighted(header_bar, 0.2, dark, 0.8, 0, header_bar)
        preview[0:40, 0:w] = header_bar

        cv2.putText(
            preview,
            f"{title} | Camera: {camera_config.camera_id} [{camera_config.status}]",
            (14, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return preview
