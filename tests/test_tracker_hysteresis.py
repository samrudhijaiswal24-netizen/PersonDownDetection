"""
Tests for Stage B: Tracker Hysteresis Investigation.

Requirements:
1. Isolated tests for transient upright misclassifications (1-frame and 2-frame glitches).
2. Genuine recovery latency measurement from first upright frame to NORMAL.
   Verify recovery occurs within 0.35 seconds (at verified 30.0 FPS, <= 10 frames).
3. Prolonged occlusion handling with hysteresis:
   - Dropout grace (<= 15 frames)
   - Disappearance downgrade (<= 45 frames)
   - Extended absence reset (> 45 frames)
4. ADL false-alarm rejection:
   - Crouching bursts never promoted
   - Sofa/bed resting never promoted
"""

import pytest
from src.core.temporal_tracker import (
    TemporalState,
    TemporalTrackerConfig,
    TemporalStateTracker,
)


class TestTransientGlitchRejection:
    """Tests evaluating tracker stability against transient upright posture misclassifications."""

    def test_baseline_resets_immediately_on_single_upright_frame(self):
        """Baseline N=1: A single transient STANDING frame immediately resets candidate to NORMAL."""
        cfg = TemporalTrackerConfig(
            fps=30.0,
            persistence_seconds=1.0,  # 30 frames
            upright_recovery_confirmation_frames=1,
        )
        tracker = TemporalStateTracker(config=cfg)

        # Build up to PERSON_DOWN_CANDIDATE
        for i in range(35):
            tracker.update(i, "FLOOR_LYING_CANDIDATE", "LYING", True, "FLOOR_ZONE")

        assert tracker.state == TemporalState.PERSON_DOWN_CANDIDATE
        assert tracker.total_candidate_events_triggered == 1

        # Frame 35: 1-frame transient STANDING glitch
        up_glitch = tracker.update(35, "NORMAL_ACTIVITY", "STANDING", True, "FLOOR_ZONE")
        assert up_glitch.state == TemporalState.NORMAL
        assert up_glitch.details.get("transition") == "recovered_or_safe_furniture"
        assert tracker.consecutive_floor_frames == 0

        # Frame 36: Floor lying resumes, but tracker was reset, so state is POSSIBLE_PERSON_DOWN
        up_resume = tracker.update(36, "FLOOR_LYING_CANDIDATE", "LYING", True, "FLOOR_ZONE")
        assert up_resume.state == TemporalState.POSSIBLE_PERSON_DOWN
        assert up_resume.is_new_candidate_event is False

    @pytest.mark.parametrize("n_confirm", [2, 3, 5, 10])
    def test_hysteresis_absorbs_single_transient_standing_glitch(self, n_confirm: int):
        """With N >= 2, a single transient upright frame is absorbed without resetting candidate state."""
        cfg = TemporalTrackerConfig(
            fps=30.0,
            persistence_seconds=1.0,  # 30 frames
            upright_recovery_confirmation_frames=n_confirm,
        )
        tracker = TemporalStateTracker(config=cfg)

        for i in range(35):
            tracker.update(i, "FLOOR_LYING_CANDIDATE", "LYING", True, "FLOOR_ZONE")

        assert tracker.state == TemporalState.PERSON_DOWN_CANDIDATE
        assert tracker.total_candidate_events_triggered == 1

        # Frame 35: 1-frame transient STANDING glitch
        up_glitch = tracker.update(35, "NORMAL_ACTIVITY", "STANDING", True, "FLOOR_ZONE")
        assert up_glitch.state == TemporalState.PERSON_DOWN_CANDIDATE
        assert up_glitch.details.get("transition") == "pending_upright_recovery"
        assert tracker.upright_recovery_frames == 1

        # Frame 36: Genuine floor lying resumes
        up_resume = tracker.update(36, "FLOOR_LYING_CANDIDATE", "LYING", True, "FLOOR_ZONE")
        assert up_resume.state == TemporalState.PERSON_DOWN_CANDIDATE
        assert up_resume.details.get("transition") == "maintaining_candidate"
        assert up_resume.is_new_candidate_event is False
        assert tracker.upright_recovery_frames == 0
        # Remains exactly 1 candidate event (no spurious split)
        assert tracker.total_candidate_events_triggered == 1

    @pytest.mark.parametrize("n_confirm", [3, 5, 10])
    def test_hysteresis_absorbs_two_frame_standing_glitch(self, n_confirm: int):
        """With N >= 3, a 2-frame upright glitch is absorbed without dropping candidate state."""
        cfg = TemporalTrackerConfig(
            fps=30.0,
            persistence_seconds=1.0,
            upright_recovery_confirmation_frames=n_confirm,
        )
        tracker = TemporalStateTracker(config=cfg)

        for i in range(35):
            tracker.update(i, "FLOOR_LYING_CANDIDATE", "LYING", True, "FLOOR_ZONE")

        assert tracker.state == TemporalState.PERSON_DOWN_CANDIDATE

        # 2 frames of spurious STANDING
        up1 = tracker.update(35, "NORMAL_ACTIVITY", "STANDING", True, "FLOOR_ZONE")
        assert up1.state == TemporalState.PERSON_DOWN_CANDIDATE
        assert tracker.upright_recovery_frames == 1

        up2 = tracker.update(36, "NORMAL_ACTIVITY", "STANDING", True, "FLOOR_ZONE")
        assert up2.state == TemporalState.PERSON_DOWN_CANDIDATE
        assert tracker.upright_recovery_frames == 2

        # Frame 37: Lying resumes
        up3 = tracker.update(37, "FLOOR_LYING_CANDIDATE", "LYING", True, "FLOOR_ZONE")
        assert up3.state == TemporalState.PERSON_DOWN_CANDIDATE
        assert tracker.upright_recovery_frames == 0
        assert tracker.total_candidate_events_triggered == 1

    def test_two_frame_glitch_triggers_reset_when_n_equals_two(self):
        """With N=2, a 2-frame upright sequence meets threshold and resets candidate."""
        cfg = TemporalTrackerConfig(
            fps=30.0,
            persistence_seconds=1.0,
            upright_recovery_confirmation_frames=2,
        )
        tracker = TemporalStateTracker(config=cfg)

        for i in range(35):
            tracker.update(i, "FLOOR_LYING_CANDIDATE", "LYING", True, "FLOOR_ZONE")

        # Frame 35: 1st standing frame -> pending
        up1 = tracker.update(35, "NORMAL_ACTIVITY", "STANDING", True, "FLOOR_ZONE")
        assert up1.state == TemporalState.PERSON_DOWN_CANDIDATE

        # Frame 36: 2nd standing frame -> reaches N=2 -> reset
        up2 = tracker.update(36, "NORMAL_ACTIVITY", "STANDING", True, "FLOOR_ZONE")
        assert up2.state == TemporalState.NORMAL
        assert up2.details.get("transition") == "recovered_or_safe_furniture"


class TestGenuineRecoveryLatency:
    """
    Tests measuring genuine recovery latency from first upright frame to NORMAL.
    Requirement: Verify recovery occurs within 0.35 seconds across evaluated thresholds.
    """

    @pytest.mark.parametrize("n_confirm,expected_latency_sec", [
        (1, 1 / 30.0),   # 0.0333s
        (2, 2 / 30.0),   # 0.0667s
        (3, 3 / 30.0),   # 0.1000s
        (5, 5 / 30.0),   # 0.1667s
        (10, 10 / 30.0), # 0.3333s
    ])
    def test_standing_recovery_latency_within_350ms(self, n_confirm: int, expected_latency_sec: float):
        """Verify recovery latency in frames and seconds meets <= 0.35s requirement."""
        fps = 30.0
        cfg = TemporalTrackerConfig(
            fps=fps,
            persistence_seconds=1.0,
            upright_recovery_confirmation_frames=n_confirm,
        )
        tracker = TemporalStateTracker(config=cfg)

        # Establish PERSON_DOWN_CANDIDATE
        for i in range(35):
            tracker.update(i, "FLOOR_LYING_CANDIDATE", "LYING", True, "FLOOR_ZONE")
        assert tracker.state == TemporalState.PERSON_DOWN_CANDIDATE

        # Genuine recovery starts at frame 35
        first_upright_frame = 35
        transition_to_normal_frame = None

        for f_idx in range(35, 60):
            up = tracker.update(f_idx, "NORMAL_ACTIVITY", "STANDING", True, "FLOOR_ZONE")
            if up.state == TemporalState.NORMAL and transition_to_normal_frame is None:
                transition_to_normal_frame = f_idx
                break

        assert transition_to_normal_frame is not None
        latency_frames = transition_to_normal_frame - first_upright_frame + 1
        latency_sec = latency_frames / fps

        assert latency_frames == n_confirm
        assert pytest.approx(latency_sec, rel=1e-4) == expected_latency_sec
        # Strict requirement: recovery occurs within 0.35 seconds
        assert latency_sec <= 0.35, f"Recovery latency {latency_sec}s exceeded 0.35s maximum"

    @pytest.mark.parametrize("n_confirm", [1, 2, 3, 5, 10])
    def test_sitting_recovery_latency_matches_standing(self, n_confirm: int):
        """Sitting posture recovery behaves identically to standing posture."""
        fps = 30.0
        cfg = TemporalTrackerConfig(
            fps=fps,
            persistence_seconds=1.0,
            upright_recovery_confirmation_frames=n_confirm,
        )
        tracker = TemporalStateTracker(config=cfg)

        for i in range(35):
            tracker.update(i, "FLOOR_LYING_CANDIDATE", "LYING", True, "FLOOR_ZONE")

        first_upright_frame = 35
        transition_to_normal_frame = None
        for f_idx in range(35, 60):
            up = tracker.update(f_idx, "NORMAL_ACTIVITY", "SITTING", True, "FLOOR_ZONE")
            if up.state == TemporalState.NORMAL and transition_to_normal_frame is None:
                transition_to_normal_frame = f_idx
                break

        latency_frames = transition_to_normal_frame - first_upright_frame + 1
        assert latency_frames == n_confirm
        assert (latency_frames / fps) <= 0.35

    @pytest.mark.parametrize("n_confirm", [1, 2, 3, 5, 10])
    def test_safe_furniture_recovery_latency(self, n_confirm: int):
        """Safe furniture resting (BED_SOFA_LYING) confirms recovery within N frames <= 0.35s."""
        fps = 30.0
        cfg = TemporalTrackerConfig(
            fps=fps,
            persistence_seconds=1.0,
            upright_recovery_confirmation_frames=n_confirm,
        )
        tracker = TemporalStateTracker(config=cfg)

        for i in range(35):
            tracker.update(i, "FLOOR_LYING_CANDIDATE", "LYING", True, "FLOOR_ZONE")

        first_safe_frame = 35
        transition_frame = None
        for f_idx in range(35, 60):
            up = tracker.update(f_idx, "BED_SOFA_LYING", "LYING", True, "SOFA_ZONE")
            if up.state == TemporalState.NORMAL and transition_frame is None:
                transition_frame = f_idx
                break

        latency_frames = transition_frame - first_safe_frame + 1
        assert latency_frames == n_confirm
        assert (latency_frames / fps) <= 0.35


class TestProlongedOcclusionWithHysteresis:
    """Tests evaluating interaction between recovery hysteresis and detector dropouts."""

    @pytest.mark.parametrize("n_confirm", [1, 2, 3, 5, 10])
    def test_dropout_grace_and_absence_invariance(self, n_confirm: int):
        """Dropout grace (15 frames), downgrade (45 frames), and reset (>45 frames) are invariant to N."""
        cfg = TemporalTrackerConfig(
            fps=30.0,
            persistence_seconds=1.0,
            max_dropout_grace_seconds=0.5,   # 15 frames
            max_disappearance_seconds=1.5,   # 45 frames
            upright_recovery_confirmation_frames=n_confirm,
        )
        tracker = TemporalStateTracker(config=cfg)

        for i in range(35):
            tracker.update(i, "FLOOR_LYING_CANDIDATE", "LYING", True, "FLOOR_ZONE")
        assert tracker.state == TemporalState.PERSON_DOWN_CANDIDATE

        # 15 frames of dropout: held in grace
        for i in range(35, 50):
            up = tracker.update(i, "NORMAL_ACTIVITY", "NO_DETECTION", False, None)
            assert up.state == TemporalState.PERSON_DOWN_CANDIDATE
            assert up.details.get("state_memory") == "holding_within_grace"

        # Frame 50 (16th dropout): downgrade to UNKNOWN
        up_16 = tracker.update(50, "NORMAL_ACTIVITY", "NO_DETECTION", False, None)
        assert up_16.state == TemporalState.UNKNOWN
        assert up_16.details.get("state_memory") == "downgraded_to_unknown"

        # Frames 51 to 79: remain UNKNOWN
        for i in range(51, 80):
            up = tracker.update(i, "NORMAL_ACTIVITY", "NO_DETECTION", False, None)
            assert up.state == TemporalState.UNKNOWN

        # Frame 81 (46th dropout): reset to NORMAL
        up_46 = tracker.update(80, "NORMAL_ACTIVITY", "NO_DETECTION", False, None)
        assert up_46.state == TemporalState.NORMAL
        assert up_46.details.get("state_memory") == "reset_after_extended_absence"

    @pytest.mark.parametrize("n_confirm", [2, 3, 5, 10])
    def test_dropout_resets_pending_upright_counter(self, n_confirm: int):
        """If a dropout occurs during pending recovery, the pending counter is cleared."""
        cfg = TemporalTrackerConfig(
            fps=30.0,
            persistence_seconds=1.0,
            upright_recovery_confirmation_frames=n_confirm,
        )
        tracker = TemporalStateTracker(config=cfg)

        for i in range(35):
            tracker.update(i, "FLOOR_LYING_CANDIDATE", "LYING", True, "FLOOR_ZONE")

        # 1 frame of STANDING (pending)
        up_stand = tracker.update(35, "NORMAL_ACTIVITY", "STANDING", True, "FLOOR_ZONE")
        assert up_stand.state == TemporalState.PERSON_DOWN_CANDIDATE
        assert tracker.upright_recovery_frames == 1

        # Dropout on next frame
        up_drop = tracker.update(36, "NORMAL_ACTIVITY", "NO_DETECTION", False, None)
        assert tracker.upright_recovery_frames == 0
        assert up_drop.state == TemporalState.PERSON_DOWN_CANDIDATE


class TestADLFalseAlarmRejectionWithHysteresis:
    """Tests evaluating ADL false alarm rejection under hysteresis."""

    @pytest.mark.parametrize("n_confirm", [1, 2, 3, 5, 10])
    def test_crouching_burst_never_promoted_to_candidate(self, n_confirm: int):
        """Crouching burst (10 frames) is rejected across all hysteresis configurations."""
        cfg = TemporalTrackerConfig(
            fps=30.0,
            persistence_seconds=1.0,  # 30 frames
            upright_recovery_confirmation_frames=n_confirm,
        )
        tracker = TemporalStateTracker(config=cfg)

        # 10 frames crouching (classified as FLOOR_LYING_CANDIDATE)
        for i in range(10):
            up = tracker.update(i, "FLOOR_LYING_CANDIDATE", "LYING", True, "FLOOR_ZONE")
            assert up.state == TemporalState.POSSIBLE_PERSON_DOWN
            assert up.is_new_candidate_event is False

        # Returns to standing
        for i in range(10, 25):
            up = tracker.update(i, "NORMAL_ACTIVITY", "STANDING", True, "FLOOR_ZONE")
            assert up.state != TemporalState.PERSON_DOWN_CANDIDATE
            assert up.is_new_candidate_event is False

        assert tracker.total_candidate_events_triggered == 0

    @pytest.mark.parametrize("n_confirm", [1, 2, 3, 5, 10])
    def test_sofa_lying_never_promoted_to_candidate(self, n_confirm: int):
        """Sustained sofa lying is rejected across all hysteresis configurations."""
        cfg = TemporalTrackerConfig(
            fps=30.0,
            persistence_seconds=1.0,
            upright_recovery_confirmation_frames=n_confirm,
        )
        tracker = TemporalStateTracker(config=cfg)

        for i in range(60):
            up = tracker.update(i, "BED_SOFA_LYING", "LYING", True, "SOFA_ZONE")
            assert up.state == TemporalState.NORMAL
            assert up.is_new_candidate_event is False

        assert tracker.total_candidate_events_triggered == 0


class TestPhase5EdgeCaseTransitions:
    """
    Phase 5 isolated tests verifying edge-case transitions:
    1. Candidate -> detector dropout grace -> upright recovery transition.
    2. Candidate -> transient BED_SOFA_LYING glitch -> floor lying resumption.
    """

    @pytest.mark.parametrize("n_confirm", [1, 2, 3, 5, 10])
    def test_candidate_dropout_grace_followed_by_upright_recovery(self, n_confirm: int):
        """Dropout within grace period bridges cleanly into recovery confirmation upon upright frame."""
        cfg = TemporalTrackerConfig(
            fps=30.0,
            persistence_seconds=1.0,
            max_dropout_grace_seconds=0.5,  # 15 frames
            upright_recovery_confirmation_frames=n_confirm,
        )
        tracker = TemporalStateTracker(config=cfg)

        # 35 frames: reach candidate state
        for i in range(35):
            tracker.update(i, "FLOOR_LYING_CANDIDATE", "LYING", True, "FLOOR_ZONE")
        assert tracker.state == TemporalState.PERSON_DOWN_CANDIDATE

        # 5 frames: brief occlusion/dropout (within 15-frame grace)
        for i in range(35, 40):
            up_drop = tracker.update(i, "NORMAL_ACTIVITY", "NO_DETECTION", False, None)
            assert up_drop.state == TemporalState.PERSON_DOWN_CANDIDATE
            assert up_drop.details.get("state_memory") == "holding_within_grace"

        # Frame 40: First upright frame post-dropout
        up_first_stand = tracker.update(40, "NORMAL_ACTIVITY", "STANDING", True, "FLOOR_ZONE")
        assert tracker.dropout_gap_frames == 0
        assert tracker.upright_recovery_frames == 1

        if n_confirm == 1:
            assert up_first_stand.state == TemporalState.NORMAL
            assert up_first_stand.details.get("transition") == "recovered_or_safe_furniture"
        else:
            assert up_first_stand.state == TemporalState.PERSON_DOWN_CANDIDATE
            assert up_first_stand.details.get("transition") == "pending_upright_recovery"

            # Subsequent standing frames up to N confirmation
            for f_idx in range(41, 40 + n_confirm):
                up_stand = tracker.update(f_idx, "NORMAL_ACTIVITY", "STANDING", True, "FLOOR_ZONE")
                if f_idx < 40 + n_confirm - 1:
                    assert up_stand.state == TemporalState.PERSON_DOWN_CANDIDATE
                else:
                    assert up_stand.state == TemporalState.NORMAL
                    assert up_stand.details.get("transition") == "recovered_or_safe_furniture"

    @pytest.mark.parametrize("n_confirm", [2, 3, 5, 10])
    def test_candidate_transient_bed_sofa_lying_absorbed_by_hysteresis(self, n_confirm: int):
        """Transient single-frame BED_SOFA_LYING is absorbed without terminating candidate state."""
        cfg = TemporalTrackerConfig(
            fps=30.0,
            persistence_seconds=1.0,
            upright_recovery_confirmation_frames=n_confirm,
        )
        tracker = TemporalStateTracker(config=cfg)

        for i in range(35):
            tracker.update(i, "FLOOR_LYING_CANDIDATE", "LYING", True, "FLOOR_ZONE")
        assert tracker.state == TemporalState.PERSON_DOWN_CANDIDATE
        assert tracker.total_candidate_events_triggered == 1

        # Frame 35: 1 frame of safe furniture misclassification
        up_sofa = tracker.update(35, "BED_SOFA_LYING", "LYING", True, "SOFA_ZONE")
        assert up_sofa.state == TemporalState.PERSON_DOWN_CANDIDATE
        assert up_sofa.details.get("transition") == "pending_upright_recovery"
        assert tracker.upright_recovery_frames == 1

        # Frame 36: Floor lying resumes
        up_resume = tracker.update(36, "FLOOR_LYING_CANDIDATE", "LYING", True, "FLOOR_ZONE")
        assert up_resume.state == TemporalState.PERSON_DOWN_CANDIDATE
        assert up_resume.details.get("transition") == "maintaining_candidate"
        assert up_resume.is_new_candidate_event is False
        assert tracker.upright_recovery_frames == 0
        assert tracker.total_candidate_events_triggered == 1

