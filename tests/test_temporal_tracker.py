"""
Unit and Integration Tests for Phase 4: Temporal State Tracking and Alert Persistence.

Tests:
- Brief false lying bursts (crouching/bending transients)
- Sustained floor lying meets persistence threshold
- Safe sofa/bed lying suppression
- Detector dropouts with bounded state memory (grace vs unknown vs reset)
- Immediate recovery on standing or sitting
- Unknown zone handling without floor counter advance
- Intermittent detection jitter within evidence ratio
- Configuration loading from YAML
"""

import pytest
from pathlib import Path
from src.core.temporal_tracker import (
    TemporalState,
    TemporalTrackerConfig,
    TemporalStateTracker,
    TemporalStateUpdate,
)

CONFIG_PATH = Path("configs/temporal_tracker_config.yaml")


class TestTemporalTrackerConfig:
    """Tests for configuration defaults, derived properties, and YAML loading."""

    def test_default_config_properties(self):
        cfg = TemporalTrackerConfig(
            fps=30.0,
            persistence_seconds=1.5,
            min_evidence_ratio=0.65,
            max_dropout_grace_seconds=0.5,
            max_disappearance_seconds=1.5,
        )
        assert cfg.fps == 30.0
        assert cfg.persistence_frames == 45
        assert cfg.grace_frames == 15
        assert cfg.max_disappearance_frames == 45

    def test_load_from_yaml(self):
        assert CONFIG_PATH.exists(), f"Missing config file at {CONFIG_PATH}"
        cfg = TemporalTrackerConfig.from_yaml(CONFIG_PATH)
        assert cfg.fps == 30.0
        assert cfg.persistence_seconds == 1.5
        assert cfg.persistence_frames == 45
        assert cfg.grace_frames == 15
        assert cfg.max_disappearance_frames == 45
        assert cfg.reset_on_upright_posture is True
        assert cfg.reset_on_safe_furniture is True


class TestTemporalStateTransitions:
    """Tests for temporal state tracking transitions and persistence rules."""

    @pytest.fixture
    def tracker(self):
        cfg = TemporalTrackerConfig(
            fps=30.0,
            persistence_seconds=1.5,     # 45 frames
            min_evidence_ratio=0.65,     # >= 65% floor evidence
            max_dropout_grace_seconds=0.5,  # 15 frames
            max_disappearance_seconds=1.5,  # 45 frames
            reset_on_upright_posture=True,
            reset_on_safe_furniture=True,
        )
        return TemporalStateTracker(config=cfg)

    def test_single_lying_prediction_does_not_trigger_candidate(self, tracker):
        """Requirement 1: A single LYING prediction must not immediately trigger a persistent alert."""
        update = tracker.update(
            frame_idx=10,
            event_classification="FLOOR_LYING_CANDIDATE",
            posture="LYING",
            has_person=True,
            zone_type="FLOOR_ZONE",
        )
        assert update.state == TemporalState.POSSIBLE_PERSON_DOWN
        assert update.is_new_candidate_event is False
        assert tracker.total_candidate_events_triggered == 0
        assert update.accumulated_floor_frames == 1

    def test_brief_crouching_burst_suppressed(self, tracker):
        """Requirement 7: Crouching bursts (<45 frames) must stay in POSSIBLE_PERSON_DOWN and revert to NORMAL."""
        # Simulate 9 frames of crouching (empirically observed max in chute24)
        for i in range(9):
            update = tracker.update(
                frame_idx=i,
                event_classification="FLOOR_LYING_CANDIDATE",
                posture="LYING",
                has_person=True,
                zone_type="FLOOR_ZONE",
            )
            assert update.state == TemporalState.POSSIBLE_PERSON_DOWN
            assert update.is_new_candidate_event is False

        assert tracker.total_candidate_events_triggered == 0

        # Person stands up on frame 10
        update = tracker.update(
            frame_idx=9,
            event_classification="NORMAL_ACTIVITY",
            posture="STANDING",
            has_person=True,
            zone_type="FLOOR_ZONE",
        )
        assert update.state == TemporalState.NORMAL
        assert update.accumulated_floor_frames == 0
        assert tracker.total_candidate_events_triggered == 0

    def test_sustained_floor_lying_promotion(self, tracker):
        """Requirement 2: Sustained floor evidence promotes to PERSON_DOWN_CANDIDATE at 45 frames (1.5s)."""
        # Feed 44 frames: all should be POSSIBLE_PERSON_DOWN
        for i in range(44):
            update = tracker.update(
                frame_idx=i,
                event_classification="FLOOR_LYING_CANDIDATE",
                posture="LYING",
                has_person=True,
                zone_type="FLOOR_ZONE",
            )
            assert update.state == TemporalState.POSSIBLE_PERSON_DOWN
            assert update.is_new_candidate_event is False

        # Frame 45 reaches persistence requirement (45 frames @ 30 FPS = 1.5s)
        update_45 = tracker.update(
            frame_idx=44,
            event_classification="FLOOR_LYING_CANDIDATE",
            posture="LYING",
            has_person=True,
            zone_type="FLOOR_ZONE",
        )
        assert update_45.state == TemporalState.PERSON_DOWN_CANDIDATE
        assert update_45.is_new_candidate_event is True
        assert tracker.total_candidate_events_triggered == 1
        assert update_45.elapsed_down_seconds == 1.5

        # Frame 46 continues candidate state without re-triggering new event
        update_46 = tracker.update(
            frame_idx=45,
            event_classification="FLOOR_LYING_CANDIDATE",
            posture="LYING",
            has_person=True,
            zone_type="FLOOR_ZONE",
        )
        assert update_46.state == TemporalState.PERSON_DOWN_CANDIDATE
        assert update_46.is_new_candidate_event is False
        assert tracker.total_candidate_events_triggered == 1

    def test_sofa_lying_never_triggers_floor_candidate(self, tracker):
        """Requirement 6: Bed/sofa lying must not become a floor-collapse candidate."""
        for i in range(60):
            update = tracker.update(
                frame_idx=i,
                event_classification="BED_SOFA_LYING",
                posture="LYING",
                has_person=True,
                zone_type="BED_SOFA_ZONE",
            )
            assert update.state == TemporalState.NORMAL
            assert update.is_new_candidate_event is False
            assert update.accumulated_floor_frames == 0

        assert tracker.total_candidate_events_triggered == 0

    def test_detector_dropout_bridged_by_grace_period(self, tracker):
        """Requirement 4: Brief detector dropouts held by bounded state memory."""
        # 45 frames of sustained floor lying
        for i in range(45):
            tracker.update(i, "FLOOR_LYING_CANDIDATE", "LYING", True, "FLOOR_ZONE")

        assert tracker.state == TemporalState.PERSON_DOWN_CANDIDATE

        # 10 frames of detector dropout (grace period is 15 frames)
        for i in range(45, 55):
            update = tracker.update(i, "NORMAL_ACTIVITY", "NO_DETECTION", False, None)
            assert update.state == TemporalState.PERSON_DOWN_CANDIDATE
            assert update.details.get("state_memory") == "holding_within_grace"

        # Frame 55: floor lying resumes
        update_resume = tracker.update(55, "FLOOR_LYING_CANDIDATE", "LYING", True, "FLOOR_ZONE")
        assert update_resume.state == TemporalState.PERSON_DOWN_CANDIDATE
        assert update_resume.is_new_candidate_event is False
        assert tracker.total_candidate_events_triggered == 1

    def test_prolonged_dropout_downgrades_to_unknown(self, tracker):
        """Requirement 4: Prolonged dropout (>15 frames, <=45 frames) downgrades to UNKNOWN, not asserted down."""
        for i in range(45):
            tracker.update(i, "FLOOR_LYING_CANDIDATE", "LYING", True, "FLOOR_ZONE")

        assert tracker.state == TemporalState.PERSON_DOWN_CANDIDATE

        # 15 frames held within grace
        for i in range(45, 60):
            update = tracker.update(i, "NORMAL_ACTIVITY", "NO_DETECTION", False, None)
            assert update.state == TemporalState.PERSON_DOWN_CANDIDATE

        # Frame 61 (16th dropout frame) exceeds grace -> UNKNOWN
        update_16 = tracker.update(60, "NORMAL_ACTIVITY", "NO_DETECTION", False, None)
        assert update_16.state == TemporalState.UNKNOWN
        assert update_16.details.get("state_memory") == "downgraded_to_unknown"

    def test_extended_absence_resets_to_normal(self, tracker):
        """Requirement 4: Absence > 45 frames resets tracker to NORMAL."""
        for i in range(45):
            tracker.update(i, "FLOOR_LYING_CANDIDATE", "LYING", True, "FLOOR_ZONE")

        # 46 dropout frames total
        for i in range(45, 91):
            tracker.update(i, "NORMAL_ACTIVITY", "NO_DETECTION", False, None)

        assert tracker.state == TemporalState.NORMAL
        assert tracker.consecutive_floor_frames == 0

    def test_standing_recovery_resets_candidate(self, tracker):
        """Requirement 5: Standing up immediately resets PERSON_DOWN_CANDIDATE to NORMAL."""
        for i in range(50):
            tracker.update(i, "FLOOR_LYING_CANDIDATE", "LYING", True, "FLOOR_ZONE")

        assert tracker.state == TemporalState.PERSON_DOWN_CANDIDATE

        # Reliable standing detected
        update = tracker.update(50, "NORMAL_ACTIVITY", "STANDING", True, "FLOOR_ZONE")
        assert update.state == TemporalState.NORMAL
        assert tracker.consecutive_floor_frames == 0
        assert len(tracker.evidence_window) == 0

    def test_sitting_recovery_resets_candidate(self, tracker):
        """Requirement 5: Sitting up immediately resets candidate to NORMAL."""
        for i in range(50):
            tracker.update(i, "FLOOR_LYING_CANDIDATE", "LYING", True, "FLOOR_ZONE")

        assert tracker.state == TemporalState.PERSON_DOWN_CANDIDATE

        update = tracker.update(50, "NORMAL_ACTIVITY", "SITTING", True, "FLOOR_ZONE")
        assert update.state == TemporalState.NORMAL
        assert tracker.consecutive_floor_frames == 0

    def test_unknown_zone_does_not_advance_floor_counter(self, tracker):
        """Requirement: Unknown zones must not advance down counter."""
        update = tracker.update(0, "UNKNOWN_ZONE_LYING", "LYING", True, "UNKNOWN_ZONE")
        assert update.state == TemporalState.UNKNOWN
        assert update.accumulated_floor_frames == 0

        # Feed another unknown zone frame
        update2 = tracker.update(1, "UNKNOWN_ZONE_LYING", "LYING", True, "UNKNOWN_ZONE")
        assert update2.state == TemporalState.UNKNOWN
        assert update2.accumulated_floor_frames == 0
        assert tracker.total_candidate_events_triggered == 0

    def test_jitter_within_evidence_ratio_promotes_candidate(self, tracker):
        """Window with >=65% floor evidence and interleaved dropouts promotes to candidate on detection."""
        # 20 floor frames, 8 dropout frames, then 17 floor frames (total 45 frames, 37 floor = 82.2% >= 65%)
        for i in range(20):
            tracker.update(i, "FLOOR_LYING_CANDIDATE", "LYING", True, "FLOOR_ZONE")

        for i in range(20, 28):
            tracker.update(i, "NORMAL_ACTIVITY", "NO_DETECTION", False, None)

        for i in range(28, 45):
            tracker.update(i, "FLOOR_LYING_CANDIDATE", "LYING", True, "FLOOR_ZONE")

        assert tracker.state == TemporalState.PERSON_DOWN_CANDIDATE
        assert tracker.total_candidate_events_triggered == 1

    def test_jitter_below_evidence_ratio_does_not_promote(self, tracker):
        """Window with <65% floor evidence does NOT promote to candidate."""
        # Alternating: 1 floor frame, 2 dropouts (15 floor frames out of 45 = 33.3% < 65%)
        for i in range(45):
            if i % 3 == 0:
                tracker.update(i, "FLOOR_LYING_CANDIDATE", "LYING", True, "FLOOR_ZONE")
            else:
                tracker.update(i, "NORMAL_ACTIVITY", "NO_DETECTION", False, None)

        assert tracker.state != TemporalState.PERSON_DOWN_CANDIDATE
        assert tracker.total_candidate_events_triggered == 0
