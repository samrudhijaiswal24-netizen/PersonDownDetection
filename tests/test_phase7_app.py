"""
Unit and Integration Test Suite for Phase 7 Person Down Detection Application.

Validates:
1. Video input validation (non-existent, unsupported formats, zero bytes).
2. Pipeline invocation and output report structure.
3. Candidate-event generation and duplicate suppression.
4. Bed/sofa resting vs floor-lying disambiguation.
5. Error handling for missing models and zero detections.
6. Preservation of Phase 5 recovery default and Phase 6 posture classification.
7. FastAPI REST API endpoints using TestClient.
"""

from pathlib import Path
import cv2
import numpy as np
import pytest
from starlette.testclient import TestClient

from src.api.app import app
from src.core.pose_detector import PersonDetection
from src.core.posture_classifier import PostureClassifier, PostureType
from src.core.spatial_zones import ZoneManager, ZoneType, ZoneEventClassification
from src.core.temporal_tracker import (
    TemporalStateTracker,
    TemporalTrackerConfig,
    TemporalState,
)
from src.pipeline.person_down_pipeline import (
    PersonDownPipeline,
    SUPPORTED_VIDEO_EXTENSIONS,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def test_client():
    return TestClient(app)


@pytest.fixture
def pipeline_instance():
    return PersonDownPipeline(
        model_path="models/pretrained/yolo11n-pose.pt",
        zones_config_path="configs/zones_config.yaml",
        tracker_config_path="configs/temporal_tracker_config.yaml",
    )


# ============================================================================
# 1. Video Input Validation Tests
# ============================================================================

class TestVideoInputValidation:
    """Tests for PersonDownPipeline.validate_video_file."""

    def test_nonexistent_video_file_fails(self):
        ok, err, meta = PersonDownPipeline.validate_video_file("data/samples/missing_file_xyz123.mp4")
        assert not ok
        assert "File not found" in err
        assert meta is None

    def test_directory_as_video_fails(self, tmp_path):
        dummy_dir = tmp_path / "not_a_video"
        dummy_dir.mkdir()
        ok, err, meta = PersonDownPipeline.validate_video_file(dummy_dir)
        assert not ok
        assert "directory" in err.lower()
        assert meta is None

    def test_unsupported_extension_fails(self, tmp_path):
        bad_file = tmp_path / "test.txt"
        bad_file.write_text("not a video")
        ok, err, meta = PersonDownPipeline.validate_video_file(bad_file)
        assert not ok
        assert "Unsupported video format" in err
        assert meta is None

    def test_empty_zero_byte_video_fails(self, tmp_path):
        empty_mp4 = tmp_path / "empty.mp4"
        empty_mp4.touch()
        ok, err, meta = PersonDownPipeline.validate_video_file(empty_mp4)
        assert not ok
        assert "empty" in err.lower()
        assert meta is None

    def test_valid_video_metadata_extraction(self):
        sample = PROJECT_ROOT / "data" / "samples" / "mcfd" / "mcfd_chute14_cam1.avi"
        assert sample.exists()
        ok, err, meta = PersonDownPipeline.validate_video_file(sample)
        assert ok
        assert err is None
        assert meta is not None
        assert meta["width"] == 720
        assert meta["height"] == 480
        assert meta["total_frames"] == 1499
        assert meta["fps"] > 0


# ============================================================================
# 2. Pipeline Invocation & Output Handling Tests
# ============================================================================

class TestPipelineInvocation:
    """Tests for PersonDownPipeline execution and reporting."""

    def test_pipeline_execution_on_sample_clip(self, pipeline_instance, tmp_path):
        sample = PROJECT_ROOT / "data" / "samples" / "mcfd" / "mcfd_chute14_cam1.avi"
        out_vid = tmp_path / "test_out.mp4"

        report = pipeline_instance.process_video(
            input_video_path=sample,
            output_video_path=out_vid,
            camera_id="cam1",
            max_frames=30,  # Quick test slice
        )

        assert report["status"] == "COMPLETED"
        assert out_vid.exists()
        assert out_vid.stat().st_size > 0

        # Verify report structure
        assert "video_metadata" in report
        assert "processing_performance" in report
        assert "detection_metrics" in report
        assert "temporal_summary" in report
        assert report["video_metadata"]["processed_frames"] == 30
        assert report["processing_performance"]["average_fps"] > 0
        assert "disclaimer" in report

    def test_invalid_model_path_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="YOLO11-Pose model weights not found"):
            PersonDownPipeline(model_path="models/non_existent_yolo.pt")


# ============================================================================
# 3. Candidate-Event Display & Duplicate Suppression Tests
# ============================================================================

class TestCandidateEventManagement:
    """Verifies that consecutive candidate frames produce exactly one discrete event."""

    def test_consecutive_candidate_frames_do_not_duplicate_events(self):
        cfg = TemporalTrackerConfig(fps=30.0, persistence_seconds=1.0)
        tracker = TemporalStateTracker(cfg)

        events_triggered = []
        active_event = None

        # 1. 30 frames of floor lying candidate
        for f in range(45):
            upd = tracker.update(
                frame_idx=f,
                event_classification="FLOOR_LYING_CANDIDATE",
                posture="LYING",
                has_person=True,
                zone_type="FLOOR_ZONE",
            )
            if upd.is_new_candidate_event:
                events_triggered.append(f)
                active_event = {"start": f, "status": "ACTIVE"}

        # Exactly 1 new candidate event was triggered at threshold frame
        assert len(events_triggered) == 1
        assert active_event is not None
        assert active_event["status"] == "ACTIVE"

        # 2. Recovery frame resets candidate
        upd = tracker.update(
            frame_idx=46,
            event_classification="NORMAL_ACTIVITY",
            posture="STANDING",
            has_person=True,
            zone_type="FLOOR_ZONE",
        )
        assert upd.state == TemporalState.NORMAL

    def test_distinct_falls_produce_distinct_events(self):
        cfg = TemporalTrackerConfig(fps=30.0, persistence_seconds=0.5)
        tracker = TemporalStateTracker(cfg)

        event_triggers = []
        # First fall: 20 frames
        for f in range(20):
            upd = tracker.update(f, "FLOOR_LYING_CANDIDATE", "LYING", True, "FLOOR_ZONE")
            if upd.is_new_candidate_event:
                event_triggers.append(f)

        # Standing recovery: 5 frames
        for f in range(20, 25):
            tracker.update(f, "NORMAL_ACTIVITY", "STANDING", True, "FLOOR_ZONE")

        # Second fall: 20 frames
        for f in range(25, 45):
            upd = tracker.update(f, "FLOOR_LYING_CANDIDATE", "LYING", True, "FLOOR_ZONE")
            if upd.is_new_candidate_event:
                event_triggers.append(f)

        # Must have exactly 2 distinct events
        assert len(event_triggers) == 2


# ============================================================================
# 4. Bed / Sofa vs Floor Disambiguation Tests
# ============================================================================

class TestBedSofaDisambiguation:
    """Verifies that resting on a bed or sofa never triggers a floor collapse alert."""

    def test_sofa_lying_never_triggers_candidate_event(self):
        cfg = TemporalTrackerConfig(fps=30.0, persistence_seconds=1.0)
        tracker = TemporalStateTracker(cfg)

        # 100 consecutive frames of lying on sofa
        for f in range(100):
            upd = tracker.update(
                frame_idx=f,
                event_classification="BED_SOFA_LYING",
                posture="LYING",
                has_person=True,
                zone_type="BED_SOFA_ZONE",
            )
            assert upd.state == TemporalState.NORMAL
            assert not upd.is_new_candidate_event
            assert upd.accumulated_floor_frames == 0

    def test_zone_manager_distinguishes_sofa_and_floor(self):
        zm = ZoneManager("configs/zones_config.yaml")

        # Keypoints placed in Cam1 sofa polygon: [450, 300]
        kpts_sofa = np.zeros((17, 3), dtype=np.float32)
        kpts_sofa[11] = [450, 300, 0.9]
        kpts_sofa[12] = [460, 300, 0.9]
        res_sofa = zm.classify_zone("cam1", "LYING", kpts_sofa, [400, 250, 500, 350])
        assert res_sofa.zone_type == ZoneType.BED_SOFA_ZONE
        assert res_sofa.event_classification == ZoneEventClassification.BED_SOFA_LYING

        # Keypoints placed in Cam1 floor polygon: [200, 250]
        kpts_floor = np.zeros((17, 3), dtype=np.float32)
        kpts_floor[11] = [200, 250, 0.9]
        kpts_floor[12] = [210, 250, 0.9]
        res_floor = zm.classify_zone("cam1", "LYING", kpts_floor, [150, 200, 250, 300])
        assert res_floor.zone_type == ZoneType.FLOOR_ZONE
        assert res_floor.event_classification == ZoneEventClassification.FLOOR_LYING_CANDIDATE


# ============================================================================
# 5. Configuration & Model Invariant Preservation Tests
# ============================================================================

class TestConfigurationInvariants:
    """Verifies that Phase 5 recovery defaults and Phase 6 detection logic are preserved."""

    def test_phase5_recovery_confirmation_frames_default_is_one(self):
        cfg = TemporalTrackerConfig.from_yaml("configs/temporal_tracker_config.yaml")
        assert cfg.upright_recovery_confirmation_frames == 1, (
            "Phase 5 invariant violated: upright_recovery_confirmation_frames must remain 1"
        )

    def test_phase6_foreshortened_lying_posture_preserved(self):
        classifier = PostureClassifier()
        # Horizontal box with 55 degree 2D torso
        box = [100.0, 100.0, 230.0, 200.0]  # W=130, H=100 (W/H = 1.30)
        kpts = np.zeros((17, 3), dtype=np.float32)
        kpts[5] = [130.0, 120.0, 0.9]   # left shoulder
        kpts[6] = [170.0, 120.0, 0.9]   # right shoulder
        kpts[11] = [156.0, 158.0, 0.9]  # left hip (dy=38, dx=26 -> ~55 deg)
        kpts[12] = [196.0, 158.0, 0.9]  # right hip

        res = classifier.classify(kpts, box)
        assert res.label == PostureType.LYING.value

    def test_phase6_crouching_bend_over_straight_legs_rejected(self):
        classifier = PostureClassifier()
        # Vertical box (standing/bending forward at waist with straight legs)
        box = [100.0, 100.0, 160.0, 280.0]  # W=60, H=180 (W/H = 0.33)
        kpts = np.zeros((17, 3), dtype=np.float32)
        # Shoulders offset forward from hips
        kpts[5] = [80.0, 155.0, 0.9]
        kpts[6] = [100.0, 155.0, 0.9]
        # Hips
        kpts[11] = [150.0, 180.0, 0.9]
        kpts[12] = [170.0, 180.0, 0.9]
        # Straight vertical legs
        kpts[13] = [150.0, 230.0, 0.9]  # knees
        kpts[14] = [170.0, 230.0, 0.9]
        kpts[15] = [150.0, 280.0, 0.9]  # ankles
        kpts[16] = [170.0, 280.0, 0.9]

        res = classifier.classify(kpts, box)
        assert res.label == PostureType.STANDING.value


# ============================================================================
# 6. FastAPI Endpoints Integration Tests
# ============================================================================

class TestFastApiEndpoints:
    """Tests for REST API endpoints using Starlette/FastAPI TestClient."""

    def test_get_dashboard_html(self, test_client):
        res = test_client.get("/")
        assert res.status_code == 200
        assert "Person Down Detection System" in res.text
        assert "html" in res.headers["content-type"].lower()

    def test_get_system_status(self, test_client):
        res = test_client.get("/api/status")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ONLINE"
        assert "model_name" in data
        assert "hardware_device" in data
        assert "tracker_config" in data
        assert data["tracker_config"]["upright_recovery_confirmation_frames"] == 1

    def test_get_samples_list(self, test_client):
        res = test_client.get("/api/samples")
        assert res.status_code == 200
        data = res.json()
        assert "samples" in data
        assert len(data["samples"]) > 0
        filenames = [s["filename"] for s in data["samples"]]
        assert "mcfd_chute14_cam1.avi" in filenames

    def test_process_invalid_sample_returns_error(self, test_client):
        res = test_client.post("/api/process", json={"sample_path": "invalid_sample_404.mp4"})
        assert res.status_code in (400, 404)

    def test_get_nonexistent_job_returns_404(self, test_client):
        res = test_client.get("/api/jobs/non_existent_job_id_999")
        assert res.status_code == 404

    def test_get_camera_zones(self, test_client):
        res = test_client.get("/api/zones")
        assert res.status_code == 200
        zones = res.json()
        assert "cam1" in zones
        assert "cam2" in zones
        assert len(zones["cam1"]["zones"]) >= 2
