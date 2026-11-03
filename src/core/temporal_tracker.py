"""
Temporal State Tracking and Alert Persistence Module for Person Down Detection.

Phase 4: Temporal State Tracking, Alert Persistence, and False-Alarm Suppression.
Enforces multi-frame evidence persistence, bounded state memory across detector dropouts,
and standing recovery resets. Grounded in MCFD Technical Report 1350 real-world 30 FPS timebase.
Preserves all Phase 1-3.2 files and outputs intact.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from collections import deque
import logging
import yaml

logger = logging.getLogger(__name__)


class TemporalState(str, Enum):
    """Temporal tracking states for person down monitoring."""
    NORMAL = "NORMAL"                                  # Upright, moving, sitting, or resting on safe furniture
    POSSIBLE_PERSON_DOWN = "POSSIBLE_PERSON_DOWN"      # Floor-lying detected, accumulating persistence time
    PERSON_DOWN_CANDIDATE = "PERSON_DOWN_CANDIDATE"    # Sustained floor-lying meets persistence threshold
    UNKNOWN = "UNKNOWN"                                # Prolonged dropout, ambiguous zone, or insufficient evidence


@dataclass
class TemporalTrackerConfig:
    """Configurable thresholds for temporal persistence and state memory."""
    fps: float = 30.0                              # Real-world capture rate (Technical Report 1350)
    persistence_seconds: float = 1.5               # Required duration in floor zone for candidate promotion
    min_evidence_ratio: float = 0.65               # Minimum fraction of floor-lying frames in persistence window
    max_dropout_grace_seconds: float = 0.5         # Bounded dropout gap bridged by state memory (e.g. 15 frames)
    max_disappearance_seconds: float = 1.5         # Disappearance duration after which state transitions to UNKNOWN
    reset_on_upright_posture: bool = True          # Instant reset to NORMAL when STANDING/SITTING is reliably detected
    reset_on_safe_furniture: bool = True           # Instant reset to NORMAL when entering BED_SOFA_ZONE
    upright_recovery_confirmation_frames: int = 1  # Consecutive upright frames required to confirm recovery (1 = baseline instant reset)

    @property
    def persistence_frames(self) -> int:
        return max(1, int(round(self.fps * self.persistence_seconds)))

    @property
    def grace_frames(self) -> int:
        return max(1, int(round(self.fps * self.max_dropout_grace_seconds)))

    @property
    def max_disappearance_frames(self) -> int:
        return max(1, int(round(self.fps * self.max_disappearance_seconds)))

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "TemporalTrackerConfig":
        """Loads configuration from YAML file."""
        p = Path(yaml_path)
        if not p.exists():
            raise FileNotFoundError(f"Temporal tracker config not found: {p.resolve()}")

        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        tb = data.get("timebase", {})
        pers = data.get("persistence", {})
        mem = data.get("state_memory", {})
        rec = data.get("recovery", {})

        return cls(
            fps=float(tb.get("nominal_fps", 30.0)),
            persistence_seconds=float(pers.get("persistence_seconds", 1.5)),
            min_evidence_ratio=float(pers.get("min_evidence_ratio", 0.65)),
            max_dropout_grace_seconds=float(mem.get("max_dropout_grace_seconds", 0.5)),
            max_disappearance_seconds=float(mem.get("max_disappearance_seconds", 1.5)),
            reset_on_upright_posture=bool(rec.get("reset_on_upright_posture", True)),
            reset_on_safe_furniture=bool(rec.get("reset_on_safe_furniture", True)),
            upright_recovery_confirmation_frames=int(rec.get("upright_recovery_confirmation_frames", 1)),
        )


@dataclass
class TemporalStateUpdate:
    """Outcome of state tracker update for a single frame."""
    frame_idx: int
    state: TemporalState
    previous_state: TemporalState
    elapsed_down_seconds: float
    accumulated_floor_frames: int
    dropout_gap_frames: int
    is_new_candidate_event: bool
    details: Dict[str, Any] = field(default_factory=dict)


class TemporalStateTracker:
    """
    Tracks person-down state over time using bounded state memory and configurable persistence.
    Prevents single-frame glitches, crouching transients, and sofa resting from triggering false alerts.
    """

    def __init__(self, config: Optional[TemporalTrackerConfig] = None):
        self.config = config or TemporalTrackerConfig()
        self.state = TemporalState.NORMAL
        self.consecutive_floor_frames = 0
        self.dropout_gap_frames = 0
        self.upright_recovery_frames = 0
        self.total_candidate_events_triggered = 0

        # Rolling window storing binary indicators: 1 if FLOOR_LYING_CANDIDATE, 0 otherwise
        self.evidence_window: deque = deque(maxlen=self.config.persistence_frames)

    def reset(self) -> None:
        """Resets tracker to initial NORMAL state."""
        self.state = TemporalState.NORMAL
        self.consecutive_floor_frames = 0
        self.dropout_gap_frames = 0
        self.upright_recovery_frames = 0
        self.evidence_window.clear()

    def update(
        self,
        frame_idx: int,
        event_classification: str,
        posture: str,
        has_person: bool,
        zone_type: Optional[str] = None,
    ) -> TemporalStateUpdate:
        """
        Updates state tracker with current frame detection and zone classification.

        Rules:
        1. Non-LYING upright posture (STANDING, SITTING) or BED_SOFA_LYING -> Confirms recovery after N frames (default 1).
        2. FLOOR_LYING_CANDIDATE -> Clears dropout counter, pushes evidence to rolling window.
           Promotes to PERSON_DOWN_CANDIDATE only when window evidence meets threshold.
        3. Detector dropouts (NO_DETECTION) -> Bridged within grace window; downgraded if prolonged.
        4. UNKNOWN_ZONE_LYING -> Does not advance floor down time.
        """
        prev_state = self.state
        is_new_candidate = False
        details: Dict[str, Any] = {}

        # Rule 1: Reliable recovery or safe furniture resting (with configurable confirmation hysteresis)
        is_upright = (self.config.reset_on_upright_posture and posture in ("STANDING", "SITTING"))
        is_safe_furniture = (self.config.reset_on_safe_furniture and event_classification == "BED_SOFA_LYING")

        if is_upright or is_safe_furniture:
            self.upright_recovery_frames += 1
            if self.upright_recovery_frames >= self.config.upright_recovery_confirmation_frames:
                self.state = TemporalState.NORMAL
                self.consecutive_floor_frames = 0
                self.dropout_gap_frames = 0
                self.evidence_window.clear()
                details["transition"] = "recovered_or_safe_furniture"
                details["upright_confirmation_frames"] = self.upright_recovery_frames
            else:
                self.dropout_gap_frames = 0
                self.evidence_window.append(0)
                details["transition"] = "pending_upright_recovery"
                details["upright_observed_frames"] = self.upright_recovery_frames

        # Rule 2: Active Floor Lying Evidence
        elif event_classification == "FLOOR_LYING_CANDIDATE":
            self.upright_recovery_frames = 0
            self.dropout_gap_frames = 0
            self.consecutive_floor_frames += 1
            self.evidence_window.append(1)

            # Check persistence condition
            window_len = len(self.evidence_window)
            evidence_count = sum(self.evidence_window)
            evidence_ratio = (evidence_count / window_len) if window_len > 0 else 0.0

            details["window_evidence_ratio"] = round(evidence_ratio, 2)
            details["consecutive_frames"] = self.consecutive_floor_frames

            if self.state == TemporalState.PERSON_DOWN_CANDIDATE:
                # Maintain active candidate status while floor-lying evidence continues
                details["transition"] = "maintaining_candidate"
            elif window_len >= self.config.persistence_frames and evidence_ratio >= self.config.min_evidence_ratio:
                is_new_candidate = True
                self.total_candidate_events_triggered += 1
                self.state = TemporalState.PERSON_DOWN_CANDIDATE
                details["transition"] = "promoted_to_candidate"
            else:
                self.state = TemporalState.POSSIBLE_PERSON_DOWN
                details["transition"] = "accumulating_persistence"

        # Rule 3: Detector Dropout Handling (Bounded State Memory)
        elif not has_person or posture == "NO_DETECTION":
            self.upright_recovery_frames = 0
            self.dropout_gap_frames += 1
            self.evidence_window.append(0)
            details["dropout_gap_frames"] = self.dropout_gap_frames

            if self.state in (TemporalState.POSSIBLE_PERSON_DOWN, TemporalState.PERSON_DOWN_CANDIDATE, TemporalState.UNKNOWN):
                if self.dropout_gap_frames <= self.config.grace_frames:
                    # Bounded state memory: hold state during brief dropout grace period
                    details["state_memory"] = "holding_within_grace"
                elif self.dropout_gap_frames <= self.config.max_disappearance_frames:
                    # Prolonged gap: downgrade to UNKNOWN (do NOT assert person remains down)
                    self.state = TemporalState.UNKNOWN
                    details["state_memory"] = "downgraded_to_unknown"
                else:
                    # Extended absence: reset to NORMAL (person has left field of view or scenario ended)
                    self.state = TemporalState.NORMAL
                    self.consecutive_floor_frames = 0
                    details["state_memory"] = "reset_after_extended_absence"
            else:
                self.state = TemporalState.NORMAL

        # Rule 4: Ambiguous / Unknown Zone Lying (Insufficient Evidence)
        elif event_classification == "UNKNOWN_ZONE_LYING" or zone_type == "UNKNOWN_ZONE":
            self.upright_recovery_frames = 0
            self.dropout_gap_frames = 0
            self.evidence_window.append(0)
            self.state = TemporalState.UNKNOWN
            details["transition"] = "unknown_zone_insufficient_evidence"

        # Rule 5: Outside Configured Zones or Unclassified Posture
        else:
            self.upright_recovery_frames = 0
            self.dropout_gap_frames = 0
            self.evidence_window.append(0)
            if self.state in (TemporalState.POSSIBLE_PERSON_DOWN, TemporalState.PERSON_DOWN_CANDIDATE):
                self.consecutive_floor_frames = max(0, self.consecutive_floor_frames - 1)
                if self.consecutive_floor_frames == 0:
                    self.state = TemporalState.NORMAL
            elif self.state == TemporalState.UNKNOWN:
                pass
            else:
                self.state = TemporalState.NORMAL
            details["transition"] = "unclassified_or_other"

        elapsed_sec = self.consecutive_floor_frames / self.config.fps

        return TemporalStateUpdate(
            frame_idx=frame_idx,
            state=self.state,
            previous_state=prev_state,
            elapsed_down_seconds=round(elapsed_sec, 2),
            accumulated_floor_frames=self.consecutive_floor_frames,
            dropout_gap_frames=self.dropout_gap_frames,
            is_new_candidate_event=is_new_candidate,
            details=details,
        )
