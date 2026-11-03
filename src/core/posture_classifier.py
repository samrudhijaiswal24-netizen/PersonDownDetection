"""
Posture Classifier Module
Geometric and rule-based classifier using 17 COCO pose keypoints and bounding box geometry.
Classifies detected persons into STANDING, SITTING, LYING, or UNKNOWN.
Supports keypoint reliability filtering, multi-factor scoring, and temporal smoothing.
"""

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union
import logging
import math
import numpy as np

logger = logging.getLogger(__name__)


class PostureType(str, Enum):
    """Enumeration of recognized human postures."""
    STANDING = "STANDING"
    SITTING = "SITTING"
    LYING = "LYING"
    UNKNOWN = "UNKNOWN"


@dataclass
class PostureResult:
    """Detailed result of posture classification."""
    posture: PostureType
    confidence: float
    features: Dict[str, float] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return self.posture.value


class PostureClassifier:
    """
    Classifies human posture from 17 COCO pose keypoints and bounding box geometry.
    Purely postural classification: does not infer spatial zones (bed vs floor).
    """

    # COCO Keypoint Index Constants
    NOSE = 0
    L_EYE, R_EYE = 1, 2
    L_EAR, R_EAR = 3, 4
    L_SHOULDER, R_SHOULDER = 5, 6
    L_ELBOW, R_ELBOW = 7, 8
    L_WRIST, R_WRIST = 9, 10
    L_HIP, R_HIP = 11, 12
    L_KNEE, R_KNEE = 13, 14
    L_ANKLE, R_ANKLE = 15, 16

    def __init__(
        self,
        keypoint_conf_threshold: float = 0.35,
        min_valid_keypoints: int = 4,
        lying_torso_angle_max: float = 38.0,
        standing_torso_angle_min: float = 50.0,
        standing_aspect_ratio_max: float = 0.70,
        lying_aspect_ratio_min: float = 1.10,
        knee_flexion_sitting_min: float = 65.0,
        knee_flexion_sitting_max: float = 135.0,
    ):
        """
        Initializes the PostureClassifier with configurable thresholds.

        Args:
            keypoint_conf_threshold: Minimum confidence for a keypoint to be considered reliable.
            min_valid_keypoints: Minimum reliable keypoints needed for geometry analysis.
            lying_torso_angle_max: Max torso angle with horizontal (deg) to qualify as lying.
            standing_torso_angle_min: Min torso angle with horizontal (deg) for standing.
            standing_aspect_ratio_max: Max bounding box aspect ratio (W/H) for standing.
            lying_aspect_ratio_min: Min bounding box aspect ratio (W/H) for lying.
            knee_flexion_sitting_min: Min knee angle (deg) indicative of sitting.
            knee_flexion_sitting_max: Max knee angle (deg) indicative of sitting.
        """
        self.kpt_conf_thresh = keypoint_conf_threshold
        self.min_valid_kpts = min_valid_keypoints
        self.lying_torso_angle_max = lying_torso_angle_max
        self.standing_torso_angle_min = standing_torso_angle_min
        self.standing_aspect_ratio_max = standing_aspect_ratio_max
        self.lying_aspect_ratio_min = lying_aspect_ratio_min
        self.knee_flexion_sitting_min = knee_flexion_sitting_min
        self.knee_flexion_sitting_max = knee_flexion_sitting_max

    def _get_midpoint(
        self,
        keypoints: np.ndarray,
        idx1: int,
        idx2: int,
        min_conf: Optional[float] = None,
    ) -> Optional[Tuple[float, float]]:
        """Computes midpoint between two keypoints if at least one is confident."""
        thresh = min_conf if min_conf is not None else self.kpt_conf_thresh
        c1 = keypoints[idx1, 2] >= thresh
        c2 = keypoints[idx2, 2] >= thresh

        if c1 and c2:
            return (
                float((keypoints[idx1, 0] + keypoints[idx2, 0]) / 2.0),
                float((keypoints[idx1, 1] + keypoints[idx2, 1]) / 2.0),
            )
        elif c1:
            return float(keypoints[idx1, 0]), float(keypoints[idx1, 1])
        elif c2:
            return float(keypoints[idx2, 0]), float(keypoints[idx2, 1])
        return None

    def _calculate_angle_with_horizontal(
        self,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
    ) -> float:
        """
        Calculates the acute angle in degrees between vector (p1 -> p2) and the horizontal plane.
        Returned angle is in [0.0, 90.0] degrees (0 = completely horizontal, 90 = vertical).
        """
        dx = abs(p2[0] - p1[0])
        dy = abs(p2[1] - p1[1])
        rad = math.atan2(dy, max(1e-5, dx))
        return math.degrees(rad)

    def _calculate_joint_angle(
        self,
        a: Tuple[float, float],
        b: Tuple[float, float],
        c: Tuple[float, float],
    ) -> float:
        """
        Calculates internal angle ABC in degrees at vertex B.
        Returns angle in [0.0, 180.0] degrees.
        """
        ba = (a[0] - b[0], a[1] - b[1])
        bc = (c[0] - b[0], c[1] - b[1])

        mag_ba = math.hypot(ba[0], ba[1])
        mag_bc = math.hypot(bc[0], bc[1])

        if mag_ba < 1e-4 or mag_bc < 1e-4:
            return 180.0

        dot = (ba[0] * bc[0] + ba[1] * bc[1]) / (mag_ba * mag_bc)
        dot = max(-1.0, min(1.0, dot))
        return math.degrees(math.acos(dot))

    def extract_features(
        self,
        keypoints: np.ndarray,
        box: np.ndarray,
    ) -> Dict[str, float]:
        """
        Extracts geometric and posture features from 17 COCO keypoints and bounding box.

        Args:
            keypoints: (17, 3) array [x, y, conf].
            box: [x1, y1, x2, y2] array.

        Returns:
            Dictionary of computed geometric features.
        """
        features: Dict[str, float] = {}

        # 1. Bounding box features
        x1, y1, x2, y2 = box[:4]
        box_w = max(1.0, float(x2 - x1))
        box_h = max(1.0, float(y2 - y1))
        aspect_ratio = box_w / box_h  # W/H
        features["aspect_ratio"] = aspect_ratio
        features["box_width"] = box_w
        features["box_height"] = box_h

        # 2. Keypoint counts and reliability
        valid_mask = keypoints[:, 2] >= self.kpt_conf_thresh
        valid_count = int(np.sum(valid_mask))
        features["valid_keypoints_count"] = float(valid_count)
        features["mean_valid_conf"] = (
            float(np.mean(keypoints[valid_mask, 2])) if valid_count > 0 else 0.0
        )

        # 3. Torso Angle with Horizontal (Primary with relaxed fallback for motion blur/occlusion)
        shoulder = self._get_midpoint(keypoints, self.L_SHOULDER, self.R_SHOULDER)
        hip = self._get_midpoint(keypoints, self.L_HIP, self.R_HIP)

        if not (shoulder and hip):
            # Relaxed fallback (conf >= 0.25) to bridge partial low-confidence keypoints
            shoulder = self._get_midpoint(keypoints, self.L_SHOULDER, self.R_SHOULDER, min_conf=0.25)
            hip = self._get_midpoint(keypoints, self.L_HIP, self.R_HIP, min_conf=0.25)

        if shoulder and hip:
            torso_angle = self._calculate_angle_with_horizontal(shoulder, hip)
            torso_dy = abs(hip[1] - shoulder[1])
            torso_dx = abs(hip[0] - shoulder[0])
            features["torso_angle"] = torso_angle
            features["torso_vertical_ratio"] = torso_dy / box_h
            features["torso_horizontal_ratio"] = torso_dx / box_w
        else:
            features["torso_angle"] = -1.0
            features["torso_vertical_ratio"] = 0.0
            features["torso_horizontal_ratio"] = 0.0

        # 4. Knee flexion angle (Hip - Knee - Ankle)
        knee_angles = []
        thigh_horiz_ratios = []

        # Left leg
        if (
            keypoints[self.L_HIP, 2] >= self.kpt_conf_thresh
            and keypoints[self.L_KNEE, 2] >= self.kpt_conf_thresh
            and keypoints[self.L_ANKLE, 2] >= self.kpt_conf_thresh
        ):
            ang_l = self._calculate_joint_angle(
                (keypoints[self.L_HIP, 0], keypoints[self.L_HIP, 1]),
                (keypoints[self.L_KNEE, 0], keypoints[self.L_KNEE, 1]),
                (keypoints[self.L_ANKLE, 0], keypoints[self.L_ANKLE, 1]),
            )
            knee_angles.append(ang_l)

        # Right leg
        if (
            keypoints[self.R_HIP, 2] >= self.kpt_conf_thresh
            and keypoints[self.R_KNEE, 2] >= self.kpt_conf_thresh
            and keypoints[self.R_ANKLE, 2] >= self.kpt_conf_thresh
        ):
            ang_r = self._calculate_joint_angle(
                (keypoints[self.R_HIP, 0], keypoints[self.R_HIP, 1]),
                (keypoints[self.R_KNEE, 0], keypoints[self.R_KNEE, 1]),
                (keypoints[self.R_ANKLE, 0], keypoints[self.R_ANKLE, 1]),
            )
            knee_angles.append(ang_r)

        # Thigh vertical span vs horizontal span (sitting indicator)
        knee = self._get_midpoint(keypoints, self.L_KNEE, self.R_KNEE)
        if hip and knee:
            thigh_dy = abs(knee[1] - hip[1])
            thigh_dx = abs(knee[0] - hip[0])
            features["thigh_vertical_ratio"] = thigh_dy / box_h
            features["thigh_horizontal_ratio"] = thigh_dx / box_w
        else:
            features["thigh_vertical_ratio"] = 0.0
            features["thigh_horizontal_ratio"] = 0.0

        if knee_angles:
            features["knee_angle"] = float(np.mean(knee_angles))
        else:
            features["knee_angle"] = -1.0

        # 5. Keypoint spatial span ratio
        if valid_count >= 2:
            valid_x = keypoints[valid_mask, 0]
            valid_y = keypoints[valid_mask, 1]
            span_x = max(1.0, float(np.ptp(valid_x)))
            span_y = max(1.0, float(np.ptp(valid_y)))
            features["keypoint_span_ratio"] = span_x / span_y
        else:
            features["keypoint_span_ratio"] = aspect_ratio

        return features

    def classify(
        self,
        keypoints: np.ndarray,
        box: np.ndarray,
    ) -> PostureResult:
        """
        Classifies posture from pose keypoints and bounding box.

        Args:
            keypoints: (17, 3) array [x, y, confidence].
            box: [x1, y1, x2, y2] bounding box coordinates.

        Returns:
            PostureResult with posture enum, confidence score, features, and explainability reasons.
        """
        if keypoints is None or len(keypoints) < 17:
            return PostureResult(
                posture=PostureType.UNKNOWN,
                confidence=0.0,
                reasons=["Keypoint array missing or invalid format"],
            )

        features = self.extract_features(keypoints, box)
        valid_kpts = int(features.get("valid_keypoints_count", 0))
        reasons = []

        # 1. Check Keypoint Reliability
        if valid_kpts < self.min_valid_kpts:
            return PostureResult(
                posture=PostureType.UNKNOWN,
                confidence=0.5,
                features=features,
                reasons=[f"Insufficient valid keypoints ({valid_kpts} < {self.min_valid_kpts})"],
            )

        aspect_ratio = features["aspect_ratio"]
        torso_angle = features["torso_angle"]
        knee_angle = features["knee_angle"]
        span_ratio = features["keypoint_span_ratio"]
        thigh_v_ratio = features.get("thigh_vertical_ratio", 0.0)
        torso_v_ratio = features.get("torso_vertical_ratio", 0.0)

        # Body orientation flags
        is_horizontally_elongated = (aspect_ratio >= self.lying_aspect_ratio_min)
        is_vertically_slender = (aspect_ratio <= self.standing_aspect_ratio_max)

        # Scores for candidate postures
        s_lying = 0.0
        s_standing = 0.0
        s_sitting = 0.0

        # -------------------------------------------------------------
        # Factor A: Torso Angle Analysis (Primary geometric indicator)
        # -------------------------------------------------------------
        if torso_angle >= 0:
            if torso_angle <= self.lying_torso_angle_max:
                # Flat/horizontal torso
                weight = 1.0 - (torso_angle / max(1.0, self.lying_torso_angle_max))
                s_lying += 0.45 + 0.15 * weight
                reasons.append(f"Torso horizontal ({torso_angle:.1f} deg <= {self.lying_torso_angle_max} deg)")
            elif torso_angle >= self.standing_torso_angle_min:
                # Vertical/upright torso
                if is_horizontally_elongated and torso_v_ratio < 0.28:
                    # Foreshortened lying: 2D torso appears inclined due to perspective, but body is horizontal
                    s_lying += 0.35
                    s_sitting += 0.15
                    reasons.append(f"Foreshortened torso on horizontal body (angle={torso_angle:.1f} deg, v_ratio={torso_v_ratio:.2f})")
                else:
                    weight = (torso_angle - self.standing_torso_angle_min) / (90.0 - self.standing_torso_angle_min + 1e-4)
                    s_standing += 0.35 + 0.15 * weight
                    s_sitting += 0.35 + 0.10 * weight
                    reasons.append(f"Torso upright ({torso_angle:.1f} deg >= {self.standing_torso_angle_min} deg)")
            else:
                # Intermediate/inclined torso (38 - 50 deg)
                if is_horizontally_elongated:
                    s_lying += 0.35
                    s_sitting += 0.15
                    reasons.append(f"Torso inclined on horizontal body ({torso_angle:.1f} deg)")
                else:
                    s_sitting += 0.25
                    s_lying += 0.20
                    reasons.append(f"Torso inclined ({torso_angle:.1f} deg)")
        else:
            # Torso keypoints missing; fall back to span and box
            reasons.append("Torso angle unavailable (missing shoulder/hip keypoints)")

        # -------------------------------------------------------------
        # Factor B: Aspect Ratio & Spatial Span
        # -------------------------------------------------------------
        if is_horizontally_elongated:
            s_lying += 0.40
            if span_ratio >= 1.0:
                s_lying += 0.15
            reasons.append(f"Elongated aspect ratio (W/H={aspect_ratio:.2f})")
        elif is_vertically_slender:
            s_standing += 0.35
            if span_ratio <= 0.65:
                s_standing += 0.15
            reasons.append(f"Vertical slender aspect ratio (W/H={aspect_ratio:.2f})")
        else:
            # Moderate aspect ratio (0.70 to 1.10)
            s_sitting += 0.30
            reasons.append(f"Moderate aspect ratio (W/H={aspect_ratio:.2f})")

        # -------------------------------------------------------------
        # Factor C: Leg Flexion & Thigh Orientation
        # -------------------------------------------------------------
        if knee_angle > 0:
            if self.knee_flexion_sitting_min <= knee_angle <= self.knee_flexion_sitting_max:
                if is_horizontally_elongated:
                    # Bent knees on horizontal body is consistent with curled/side lying, not upright sitting
                    s_lying += 0.10
                    reasons.append(f"Knee bent on horizontal body ({knee_angle:.1f} deg)")
                else:
                    s_sitting += 0.35
                    reasons.append(f"Knee bent ({knee_angle:.1f} deg in sitting range)")
            elif knee_angle > 145.0:
                s_standing += 0.30
                reasons.append(f"Legs straight ({knee_angle:.1f} deg > 145 deg)")
        elif thigh_v_ratio > 0 and thigh_v_ratio < 0.18 and torso_angle >= 45.0 and not is_horizontally_elongated:
            # Thigh is horizontal while torso is upright -> sitting
            s_sitting += 0.25
            reasons.append(f"Horizontal thigh orientation (v_ratio={thigh_v_ratio:.2f})")

        # -------------------------------------------------------------
        # Factor D: Normalization and Ambiguity Resolution
        # -------------------------------------------------------------
        scores = {
            PostureType.LYING: s_lying,
            PostureType.STANDING: s_standing,
            PostureType.SITTING: s_sitting,
        }

        # Sort candidate postures by score
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        top_posture, top_score = ranked[0]
        second_posture, second_score = ranked[1]

        features["score_lying"] = round(s_lying, 3)
        features["score_standing"] = round(s_standing, 3)
        features["score_sitting"] = round(s_sitting, 3)

        # Ambiguity checks
        if top_score < 0.45:
            # Weak evidence across all classes
            return PostureResult(
                posture=PostureType.UNKNOWN,
                confidence=0.40,
                features=features,
                reasons=["Ambiguous posture: all candidate geometric scores below threshold", *reasons],
            )

        # Check for near tie between top two classes
        if (top_score - second_score) < 0.08 and top_score < 0.70:
            return PostureResult(
                posture=PostureType.UNKNOWN,
                confidence=0.45,
                features=features,
                reasons=[
                    f"Ambiguous boundary between {top_posture.value} ({top_score:.2f}) "
                    f"and {second_posture.value} ({second_score:.2f})",
                    *reasons,
                ],
            )

        # Crouch / Bending-over protection:
        # If candidate is LYING, but person is standing on straight vertical legs (AR <= 0.85, knee > 145 deg, thigh_v >= 0.20),
        # they are bending forward while standing on their feet, NOT collapsed on the floor.
        if top_posture == PostureType.LYING and aspect_ratio <= 0.85 and knee_angle > 145.0 and thigh_v_ratio >= 0.20:
            top_posture = PostureType.STANDING
            top_score = max(top_score, s_standing + 0.30)
            reasons.append(f"Crouch/bend protection: standing on straight legs (AR={aspect_ratio:.2f}, knee={knee_angle:.1f} deg)")

        # Calculate normalized confidence [0.55 - 0.98]
        norm_conf = min(0.98, max(0.55, top_score / 1.15))

        return PostureResult(
            posture=top_posture,
            confidence=round(norm_conf, 3),
            features=features,
            reasons=reasons,
        )


class PostureSmoother:
    """
    Applies sliding window temporal smoothing to stabilize posture predictions
    and avoid single-frame flickering.
    """

    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.history: deque = deque(maxlen=window_size)

    def update(self, result: PostureResult) -> PostureResult:
        """
        Updates sliding history and returns temporally smoothed PostureResult.

        Args:
            result: Raw frame-level PostureResult.

        Returns:
            Smoothed PostureResult.
        """
        self.history.append(result)

        if len(self.history) == 1:
            return result

        # Compute weighted vote: weigh by classification confidence and recency
        votes: Dict[PostureType, float] = {
            PostureType.STANDING: 0.0,
            PostureType.SITTING: 0.0,
            PostureType.LYING: 0.0,
            PostureType.UNKNOWN: 0.0,
        }

        total_weight = 0.0
        for i, item in enumerate(self.history):
            recency = 1.0 + (i / len(self.history))  # Recent frames have higher weight
            w = item.confidence * recency
            votes[item.posture] += w
            total_weight += w

        best_posture = max(votes.items(), key=lambda kv: kv[1])[0]
        smoothed_conf = votes[best_posture] / max(1e-5, total_weight)

        return PostureResult(
            posture=best_posture,
            confidence=round(min(0.98, max(0.50, smoothed_conf)), 3),
            features=result.features,
            reasons=[f"Smoothed over window of {len(self.history)} frames", *result.reasons[:2]],
        )

    def reset(self) -> None:
        """Clears smoothing history."""
        self.history.clear()
