"""
Unit Tests for Spatial Zone Configuration and Bed/Floor Disambiguation.

Phase 3.2: Spatial Zone Configuration and Bed/Floor Disambiguation.
Tests polygon boundaries, overlap handling, reference-point derivation & fallbacks,
invalid/missing camera configurations, each event classification outcome, and camera-specific isolation.
"""

from pathlib import Path
import numpy as np
import pytest

from src.core.spatial_zones import (
    ZoneType,
    ZoneEventClassification,
    ReferencePointSource,
    PersonReferencePoint,
    PolygonZone,
    CameraZoneConfig,
    ZoneManager,
)


@pytest.fixture
def sample_zone_manager(tmp_path) -> ZoneManager:
    """Creates a temporary ZoneManager with controlled test polygons."""
    config_file = tmp_path / "test_zones.yaml"
    config_content = """
default_policy:
  outside_zone_behavior: "UNKNOWN_ZONE"
  overlap_behavior: "UNKNOWN_ZONE"
  reference_point_min_confidence: 0.30

cameras:
  cam_test:
    resolution: [720, 480]
    status: "VALIDATED"
    zones:
      - name: "sofa_zone"
        type: "BED_SOFA_ZONE"
        polygon:
          - [370, 230]
          - [630, 230]
          - [630, 450]
          - [370, 450]
      - name: "floor_zone"
        type: "FLOOR_ZONE"
        polygon:
          - [70, 50]
          - [350, 50]
          - [350, 400]
          - [70, 400]

  cam_unvalidated:
    resolution: [720, 480]
    status: "REQUIRES_MANUAL_REVIEW"
    zones: []
"""
    config_file.write_text(config_content, encoding="utf-8")
    return ZoneManager(config_path=config_file)


class TestPolygonContainment:
    """Tests 2D polygon boundaries, points inside, points outside, and vertex edges."""

    def test_point_strictly_inside(self):
        poly = PolygonZone(
            name="test_box",
            zone_type=ZoneType.FLOOR_ZONE,
            polygon=np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.int32),
        )
        assert poly.contains_point((50.0, 50.0)) is True

    def test_point_strictly_outside(self):
        poly = PolygonZone(
            name="test_box",
            zone_type=ZoneType.FLOOR_ZONE,
            polygon=np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.int32),
        )
        assert poly.contains_point((150.0, 50.0)) is False
        assert poly.contains_point((-10.0, 50.0)) is False

    def test_point_on_boundary_or_vertex(self):
        poly = PolygonZone(
            name="test_box",
            zone_type=ZoneType.FLOOR_ZONE,
            polygon=np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.int32),
        )
        # On vertex
        assert poly.contains_point((0.0, 0.0)) is True
        # On edge
        assert poly.contains_point((50.0, 0.0)) is True
        assert poly.contains_point((100.0, 50.0)) is True


class TestOverlapAndAmbiguityHandling:
    """Verifies non-overlapping zones and safe handling of ambiguous overlap."""

    def test_ambiguous_overlap_returns_unknown_zone(self, tmp_path):
        # Create configuration where two zones intentionally overlap
        config_file = tmp_path / "overlap_zones.yaml"
        config_content = """
default_policy:
  overlap_behavior: "UNKNOWN_ZONE"
cameras:
  cam_overlap:
    resolution: [720, 480]
    status: "VALIDATED"
    zones:
      - name: "sofa_zone"
        type: "BED_SOFA_ZONE"
        polygon: [[100, 100], [300, 100], [300, 300], [100, 300]]
      - name: "floor_zone"
        type: "FLOOR_ZONE"
        polygon: [[200, 200], [400, 200], [400, 400], [200, 400]]
"""
        config_file.write_text(config_content, encoding="utf-8")
        zm = ZoneManager(config_path=config_file)

        # Point (250, 250) is inside BOTH sofa_zone and floor_zone!
        # Requirement 1: If ambiguous, mark UNKNOWN_ZONE rather than declaring safe or dangerous
        box = [230.0, 230.0, 270.0, 270.0]
        res = zm.classify_zone("cam_overlap", "LYING", keypoints=None, box=box)
        assert res.zone_type == ZoneType.OUTSIDE_ZONE
        assert res.zone_name == "AMBIGUOUS_OVERLAP"
        assert res.event_classification == ZoneEventClassification.UNKNOWN_ZONE_LYING

    def test_buffer_gap_between_zones_returns_unknown(self, sample_zone_manager):
        # In sample_zone_manager, floor ends at X=350, sofa starts at X=370.
        # X=360 is in the unassigned buffer gap.
        box = [350.0, 290.0, 370.0, 310.0]  # Center = (360, 300)
        res = sample_zone_manager.classify_zone("cam_test", "LYING", keypoints=None, box=box)
        assert res.zone_type == ZoneType.OUTSIDE_ZONE
        assert res.event_classification == ZoneEventClassification.UNKNOWN_ZONE_LYING


class TestReferencePointDerivation:
    """Validates keypoint array indices, confidence thresholds, and fallback hierarchy."""

    def test_primary_mid_hip_when_both_hips_reliable(self, sample_zone_manager):
        kps = np.zeros((17, 3), dtype=np.float32)
        kps[11] = [100.0, 200.0, 0.90]  # left_hip
        kps[12] = [120.0, 220.0, 0.80]  # right_hip
        box = [80.0, 150.0, 140.0, 250.0]

        ref = sample_zone_manager.derive_reference_point(kps, box, posture="LYING")
        assert ref.source == ReferencePointSource.MID_HIP
        assert ref.x == pytest.approx(110.0)
        assert ref.y == pytest.approx(210.0)
        assert ref.is_valid is True

    def test_secondary_torso_mid_when_one_hip_low_confidence(self, sample_zone_manager):
        kps = np.zeros((17, 3), dtype=np.float32)
        kps[5] = [90.0, 100.0, 0.85]   # left_shoulder
        kps[6] = [110.0, 100.0, 0.85]  # right_shoulder
        kps[11] = [100.0, 200.0, 0.85]  # left_hip (good)
        kps[12] = [120.0, 220.0, 0.10]  # right_hip (low confidence < 0.30)
        box = [80.0, 90.0, 130.0, 230.0]

        ref = sample_zone_manager.derive_reference_point(kps, box, posture="LYING")
        assert ref.source == ReferencePointSource.TORSO_MID
        # Shoulder mid = (100, 100), Left hip = (100, 200) -> Torso mid = (100, 150)
        assert ref.x == pytest.approx(100.0)
        assert ref.y == pytest.approx(150.0)

    def test_tertiary_bbox_center_when_all_keypoints_missing(self, sample_zone_manager):
        kps = np.zeros((17, 3), dtype=np.float32)  # All zero confidence
        box = [100.0, 100.0, 200.0, 300.0]

        ref = sample_zone_manager.derive_reference_point(kps, box, posture="LYING")
        assert ref.source == ReferencePointSource.BBOX_CENTER
        assert ref.x == pytest.approx(150.0)
        assert ref.y == pytest.approx(200.0)

    def test_upright_posture_derives_bbox_bottom_mid(self, sample_zone_manager):
        kps = np.zeros((17, 3), dtype=np.float32)
        box = [100.0, 100.0, 200.0, 300.0]

        ref_standing = sample_zone_manager.derive_reference_point(kps, box, posture="STANDING")
        assert ref_standing.source == ReferencePointSource.BBOX_BOTTOM_MID
        assert ref_standing.x == pytest.approx(150.0)
        assert ref_standing.y == pytest.approx(300.0)

        ref_sitting = sample_zone_manager.derive_reference_point(kps, box, posture="SITTING")
        assert ref_sitting.source == ReferencePointSource.BBOX_BOTTOM_MID

    def test_invalid_box_returns_unreliable(self, sample_zone_manager):
        ref = sample_zone_manager.derive_reference_point(None, [100.0, 100.0, 50.0, 50.0], posture="LYING")
        assert ref.is_valid is False
        assert ref.source == ReferencePointSource.UNRELIABLE


class TestClassificationOutcomes:
    """Verifies all required event classification rules from Section 3."""

    def test_non_lying_posture_always_normal_activity(self, sample_zone_manager):
        # Person standing or sitting inside the FLOOR_ZONE
        # Contact point is (200, 350) which is inside FLOOR_ZONE
        box = [150.0, 100.0, 250.0, 350.0]
        res_standing = sample_zone_manager.classify_zone("cam_test", "STANDING", None, box)
        assert res_standing.event_classification == ZoneEventClassification.NORMAL_ACTIVITY
        assert res_standing.zone_type == ZoneType.FLOOR_ZONE

        res_sitting = sample_zone_manager.classify_zone("cam_test", "SITTING", None, box)
        assert res_sitting.event_classification == ZoneEventClassification.NORMAL_ACTIVITY

        res_unknown = sample_zone_manager.classify_zone("cam_test", "UNKNOWN", None, box)
        assert res_unknown.event_classification == ZoneEventClassification.NORMAL_ACTIVITY

    def test_lying_in_bed_sofa_zone_classifies_bed_sofa_lying(self, sample_zone_manager):
        # Center in sofa zone: (500, 340)
        box = [400.0, 300.0, 600.0, 380.0]
        res = sample_zone_manager.classify_zone("cam_test", "LYING", None, box)
        assert res.zone_type == ZoneType.BED_SOFA_ZONE
        assert res.event_classification == ZoneEventClassification.BED_SOFA_LYING

    def test_lying_in_floor_zone_classifies_floor_lying_candidate(self, sample_zone_manager):
        # Center in floor zone: (200, 200)
        box = [150.0, 150.0, 250.0, 250.0]
        res = sample_zone_manager.classify_zone("cam_test", "LYING", None, box)
        assert res.zone_type == ZoneType.FLOOR_ZONE
        assert res.event_classification == ZoneEventClassification.FLOOR_LYING_CANDIDATE

    def test_lying_outside_configured_zones_returns_unknown_zone_lying(self, sample_zone_manager):
        # Center at (10, 10) which is outside both zones
        box = [0.0, 0.0, 20.0, 20.0]
        res = sample_zone_manager.classify_zone("cam_test", "LYING", None, box)
        assert res.zone_type == ZoneType.OUTSIDE_ZONE
        assert res.event_classification == ZoneEventClassification.UNKNOWN_ZONE_LYING
        # NEVER silently defaults to FLOOR_ZONE!
        assert res.event_classification != ZoneEventClassification.FLOOR_LYING_CANDIDATE

    def test_unvalidated_camera_returns_unknown_zone_lying(self, sample_zone_manager):
        # Camera is marked REQUIRES_MANUAL_REVIEW
        box = [150.0, 150.0, 250.0, 250.0]
        res = sample_zone_manager.classify_zone("cam_unvalidated", "LYING", None, box)
        assert res.is_validated_camera is False
        assert res.event_classification == ZoneEventClassification.UNKNOWN_ZONE_LYING

    def test_missing_camera_config_returns_unknown_zone_lying(self, sample_zone_manager):
        # Camera not in config
        box = [150.0, 150.0, 250.0, 250.0]
        res = sample_zone_manager.classify_zone("non_existent_cam", "LYING", None, box)
        assert res.is_validated_camera is False
        assert res.event_classification == ZoneEventClassification.UNKNOWN_ZONE_LYING


class TestRealConfigFile:
    """Verifies that configs/zones_config.yaml is valid and contains validated cameras."""

    def test_zones_config_yaml_loads(self):
        config_path = Path("configs/zones_config.yaml")
        assert config_path.exists()

        zm = ZoneManager(config_path)
        assert "cam1" in zm.camera_configs
        assert "cam2" in zm.camera_configs
        assert "cam1_chute09" in zm.camera_configs

        # cam1 has validated status and 2 non-overlapping zones
        cam1 = zm.camera_configs["cam1"]
        assert cam1.status == "VALIDATED"
        assert len(cam1.zones) == 2

        # cam2 has validated status and 2 non-overlapping zones
        cam2 = zm.camera_configs["cam2"]
        assert cam2.status == "VALIDATED"
        assert len(cam2.zones) == 2

        # cam1_chute09 is marked as requiring manual review
        cam09 = zm.camera_configs["cam1_chute09"]
        assert cam09.status == "REQUIRES_MANUAL_REVIEW"


class TestResolutionAwareCoordinateScaling:
    """
    Validates resolution-aware coordinate scaling in ZoneManager:
    - Unchanged behavior at calibrated 720x480 resolution
    - Proportional polygon scaling in get_scaled_zones()
    - Accurate reference-point mapping in classify_zone()
    - Format compatibility: (W, H) vs OpenCV (H, W, C)
    - Aspect-ratio incompatibility and invalid-dimension safeguards
    """

    def test_unchanged_behavior_at_native_resolution(self, sample_zone_manager):
        # 1. get_scaled_zones with matching (720, 480) returns original polygons
        scaled_zones = sample_zone_manager.get_scaled_zones("cam_test", (720, 480))
        orig_zones = sample_zone_manager.camera_configs["cam_test"].zones
        assert len(scaled_zones) == len(orig_zones)
        for s_z, o_z in zip(scaled_zones, orig_zones):
            np.testing.assert_array_equal(s_z.polygon, o_z.polygon)

        # 2. None frame_dimensions also returns original unscaled polygons
        none_zones = sample_zone_manager.get_scaled_zones("cam_test", None)
        for n_z, o_z in zip(none_zones, orig_zones):
            np.testing.assert_array_equal(n_z.polygon, o_z.polygon)

        # 3. classify_zone at native resolution produces scale_factors (1.0, 1.0)
        box = [150.0, 150.0, 250.0, 250.0]  # Center = (200, 200) inside floor_zone
        res = sample_zone_manager.classify_zone("cam_test", "LYING", None, box, frame_dimensions=(720, 480))
        assert res.zone_type == ZoneType.FLOOR_ZONE
        assert res.event_classification == ZoneEventClassification.FLOOR_LYING_CANDIDATE
        assert res.details.get("scale_factors") == (1.0, 1.0)
        assert res.details.get("scaled_reference_point") == (200.0, 200.0)

    def test_get_scaled_zones_scales_vertices_proportionally(self, sample_zone_manager):
        # cam_test configured at 720x480; scale to 1440x960 (2.0x scale)
        scaled_zones = sample_zone_manager.get_scaled_zones("cam_test", (1440, 960))
        assert len(scaled_zones) == 2

        sofa_scaled = next(z for z in scaled_zones if z.name == "sofa_zone")
        # Original sofa: [[370, 230], [630, 230], [630, 450], [370, 450]]
        expected_sofa = np.array([[740, 460], [1260, 460], [1260, 900], [740, 900]], dtype=np.int32)
        np.testing.assert_array_equal(sofa_scaled.polygon, expected_sofa)

        floor_scaled = next(z for z in scaled_zones if z.name == "floor_zone")
        # Original floor: [[70, 50], [350, 50], [350, 400], [70, 400]]
        expected_floor = np.array([[140, 100], [700, 100], [700, 800], [140, 800]], dtype=np.int32)
        np.testing.assert_array_equal(floor_scaled.polygon, expected_floor)

    def test_classify_zone_scales_coordinates_at_higher_resolution(self, sample_zone_manager):
        # Target frame: 1440x960 (2x of 720x480)
        # Floor center at (400, 200) in 1440x960 corresponds to (200, 100) in 720x480
        # In 720x480, (400, 200) is outside both floor (X<=350) and sofa (Y>=230)
        box_scaled = [350.0, 150.0, 450.0, 250.0]  # Center = (400, 200)

        # Without frame_dimensions (unscaled), point (400, 200) falls outside configured zones
        res_unscaled = sample_zone_manager.classify_zone("cam_test", "LYING", None, box_scaled)
        assert res_unscaled.zone_type == ZoneType.OUTSIDE_ZONE
        assert res_unscaled.event_classification == ZoneEventClassification.UNKNOWN_ZONE_LYING

        # With frame_dimensions=(1440, 960), point (400, 200) maps to (200, 100) -> FLOOR_ZONE
        res_scaled = sample_zone_manager.classify_zone(
            "cam_test", "LYING", None, box_scaled, frame_dimensions=(1440, 960)
        )
        assert res_scaled.zone_type == ZoneType.FLOOR_ZONE
        assert res_scaled.event_classification == ZoneEventClassification.FLOOR_LYING_CANDIDATE
        assert res_scaled.details["scale_factors"] == (0.5, 0.5)
        assert res_scaled.details["scaled_reference_point"] == (200.0, 100.0)

    def test_frame_dimensions_tuple_formats(self, sample_zone_manager):
        # Both (width, height) and OpenCV (height, width, channels) must yield identical scaling
        box_scaled = [350.0, 350.0, 450.0, 450.0]
        res_wh = sample_zone_manager.classify_zone(
            "cam_test", "LYING", None, box_scaled, frame_dimensions=(1440, 960)
        )
        res_hwc = sample_zone_manager.classify_zone(
            "cam_test", "LYING", None, box_scaled, frame_dimensions=(960, 1440, 3)
        )
        assert res_wh.zone_type == res_hwc.zone_type == ZoneType.FLOOR_ZONE
        assert res_wh.details["scaled_reference_point"] == res_hwc.details["scaled_reference_point"]

    def test_aspect_ratio_incompatibility_safeguard(self, sample_zone_manager):
        # Portrait frame (480, 720) on landscape camera config (720, 480)
        box = [100.0, 100.0, 200.0, 200.0]
        res = sample_zone_manager.classify_zone(
            "cam_test", "LYING", None, box, frame_dimensions=(480, 720)
        )
        assert res.zone_type == ZoneType.OUTSIDE_ZONE
        assert res.event_classification == ZoneEventClassification.UNKNOWN_ZONE_LYING
        assert res.details.get("reason") == "aspect_ratio_incompatible"
        assert res.details.get("aspect_ratio_incompatible") is True

        # Non-lying posture with incompatible aspect ratio should return NORMAL_ACTIVITY
        res_standing = sample_zone_manager.classify_zone(
            "cam_test", "STANDING", None, box, frame_dimensions=(480, 720)
        )
        assert res_standing.event_classification == ZoneEventClassification.NORMAL_ACTIVITY

    def test_invalid_frame_dimensions_safe_fallback(self, sample_zone_manager):
        # Zero or negative dimensions should not crash; fall back to unscaled zones
        zero_zones = sample_zone_manager.get_scaled_zones("cam_test", (0, 0))
        orig_zones = sample_zone_manager.camera_configs["cam_test"].zones
        for z_z, o_z in zip(zero_zones, orig_zones):
            np.testing.assert_array_equal(z_z.polygon, o_z.polygon)

        # Invalid format list
        invalid_format_zones = sample_zone_manager.get_scaled_zones("cam_test", [1, 2, 3, 4, 5])
        assert len(invalid_format_zones) == len(orig_zones)

        # classify_zone with invalid format falls back gracefully
        box = [150.0, 150.0, 250.0, 250.0]
        res_invalid = sample_zone_manager.classify_zone(
            "cam_test", "LYING", None, box, frame_dimensions=(0, 0)
        )
        assert res_invalid.zone_type == ZoneType.FLOOR_ZONE
        assert res_invalid.details["scale_factors"] == (1.0, 1.0)

    def test_unknown_camera_returns_empty_scaled_zones(self, sample_zone_manager):
        scaled = sample_zone_manager.get_scaled_zones("non_existent_camera", (1280, 720))
        assert scaled == []

