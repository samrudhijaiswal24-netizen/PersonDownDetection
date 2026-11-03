"""
Unit Tests for Posture Classification Module
Tests synthetic postures (STANDING, SITTING, LYING), low-confidence keypoints,
ambiguous cases, temporal smoothing, and video processor integration.
"""

import json
from pathlib import Path
import numpy as np
import pytest

from src.core.posture_classifier import (
    PostureClassifier,
    PostureResult,
    PostureSmoother,
    PostureType,
)
from src.pipeline.video_processor import VideoProcessor


def make_blank_keypoints() -> np.ndarray:
    """Returns a (17, 3) keypoint array initialized to zeros."""
    return np.zeros((17, 3), dtype=np.float32)


def test_synthetic_standing_posture():
    """Verify that vertical torso, straight legs, and slender box classify as STANDING."""
    classifier = PostureClassifier()
    kpts = make_blank_keypoints()

    # Head (y: 60 - 80)
    kpts[0] = [150, 60, 0.95]   # nose
    kpts[1] = [145, 55, 0.90]   # left eye
    kpts[2] = [155, 55, 0.90]   # right eye

    # Shoulders (y: 100)
    kpts[5] = [130, 100, 0.90]  # left shoulder
    kpts[6] = [170, 100, 0.90]  # right shoulder

    # Hips (y: 200) -> dy = 100, dx = 0 -> torso angle = 90 deg
    kpts[11] = [135, 200, 0.88] # left hip
    kpts[12] = [165, 200, 0.88] # right hip

    # Knees (y: 300)
    kpts[13] = [135, 300, 0.85] # left knee
    kpts[14] = [165, 300, 0.85] # right knee

    # Ankles (y: 390) -> straight legs (knee angle ~ 180 deg)
    kpts[15] = [135, 390, 0.82] # left ankle
    kpts[16] = [165, 390, 0.82] # right ankle

    box = np.array([110, 40, 190, 410], dtype=np.float32)  # w=80, h=370 -> W/H = 0.22

    res = classifier.classify(kpts, box)
    assert res.posture == PostureType.STANDING
    assert res.confidence >= 0.70
    assert res.features["torso_angle"] >= 80.0
    assert res.features["aspect_ratio"] <= 0.35


def test_synthetic_lying_posture():
    """Verify that horizontal torso, wide aspect ratio, and flat joints classify as LYING."""
    classifier = PostureClassifier()
    kpts = make_blank_keypoints()

    # Person lying horizontally across x: 80 to 420, y around 250
    kpts[0] = [90, 248, 0.85]   # nose
    kpts[5] = [140, 245, 0.88]  # left shoulder
    kpts[6] = [140, 255, 0.88]  # right shoulder

    # Hips (x: 250) -> dx = 110, dy = 3 -> torso angle ~ 1.5 deg
    kpts[11] = [250, 246, 0.85] # left hip
    kpts[12] = [250, 256, 0.85] # right hip

    # Knees (x: 330)
    kpts[13] = [330, 250, 0.80] # left knee
    kpts[14] = [330, 255, 0.80] # right knee

    # Ankles (x: 410)
    kpts[15] = [410, 252, 0.78] # left ankle
    kpts[16] = [410, 254, 0.78] # right ankle

    box = np.array([70, 220, 430, 280], dtype=np.float32)  # w=360, h=60 -> W/H = 6.0

    res = classifier.classify(kpts, box)
    assert res.posture == PostureType.LYING
    assert res.confidence >= 0.75
    assert res.features["torso_angle"] <= 15.0
    assert res.features["aspect_ratio"] >= 2.0


def test_synthetic_sitting_posture():
    """Verify that upright torso + 90-degree bent knees classify as SITTING."""
    classifier = PostureClassifier()
    kpts = make_blank_keypoints()

    # Torso is upright (y: 100 to 180)
    kpts[5] = [130, 100, 0.88]  # left shoulder
    kpts[6] = [170, 100, 0.88]  # right shoulder
    kpts[11] = [130, 180, 0.88] # left hip
    kpts[12] = [170, 180, 0.88] # right hip

    # Knees extended horizontally forward to x=230, y=185 (sitting on chair)
    kpts[13] = [220, 185, 0.85] # left knee
    kpts[14] = [240, 185, 0.85] # right knee

    # Ankles down vertically from knees to y=270, x=230 (knee angle ~ 90 deg)
    kpts[15] = [220, 270, 0.82] # left ankle
    kpts[16] = [240, 270, 0.82] # right ankle

    # Moderate aspect ratio
    box = np.array([110, 80, 260, 285], dtype=np.float32)  # w=150, h=205 -> W/H = 0.73

    res = classifier.classify(kpts, box)
    assert res.posture == PostureType.SITTING
    assert res.confidence >= 0.60
    assert 70.0 <= res.features["knee_angle"] <= 120.0


def test_low_confidence_keypoints_returns_unknown():
    """If valid keypoints are below threshold (< 4), classify as UNKNOWN."""
    classifier = PostureClassifier(min_valid_keypoints=4)
    kpts = make_blank_keypoints()

    # Only 2 keypoints have confidence >= 0.35
    kpts[0] = [100, 100, 0.9]
    kpts[1] = [110, 100, 0.8]
    # All other keypoints have confidence < 0.35
    box = np.array([50, 50, 150, 200], dtype=np.float32)

    res = classifier.classify(kpts, box)
    assert res.posture == PostureType.UNKNOWN
    assert any("Insufficient valid keypoints" in r for r in res.reasons)


def test_all_zero_keypoints_returns_unknown():
    """All zeros must cleanly return UNKNOWN without any math exception."""
    classifier = PostureClassifier()
    kpts = make_blank_keypoints()
    box = np.array([0, 0, 100, 100], dtype=np.float32)

    res = classifier.classify(kpts, box)
    assert res.posture == PostureType.UNKNOWN


def test_ambiguous_posture_returns_unknown():
    """Ambiguous geometry with conflicting indicators returns UNKNOWN."""
    classifier = PostureClassifier()
    kpts = make_blank_keypoints()

    # Only diagonal points with conflicting scores
    kpts[5] = [100, 100, 0.40]
    kpts[6] = [140, 120, 0.40]
    kpts[11] = [150, 160, 0.40]
    kpts[12] = [170, 180, 0.40]

    # Torso is at exactly 45 degrees, square box
    box = np.array([90, 90, 200, 200], dtype=np.float32)  # W/H = 1.0

    res = classifier.classify(kpts, box)
    # Ambiguous torso angle and box should result in UNKNOWN or low-confidence
    assert res.posture in (PostureType.UNKNOWN, PostureType.SITTING)
    if res.posture == PostureType.UNKNOWN:
        assert any("Ambiguous" in r for r in res.reasons)


def test_posture_smoother_filters_transient_glitch():
    """PostureSmoother should suppress a single-frame outlier glitch."""
    smoother = PostureSmoother(window_size=5)

    standing_res = PostureResult(posture=PostureType.STANDING, confidence=0.90)
    glitch_lying = PostureResult(posture=PostureType.LYING, confidence=0.60)

    # Sequence: STANDING, STANDING, GLITCH_LYING, STANDING
    smoother.update(standing_res)
    smoother.update(standing_res)
    smoothed_3 = smoother.update(glitch_lying)  # Glitch frame
    smoothed_4 = smoother.update(standing_res)

    # The 3rd frame is smoothed by the prior standing frames
    assert smoothed_3.posture == PostureType.STANDING
    assert smoothed_4.posture == PostureType.STANDING

    # Reset smoother
    smoother.reset()
    assert len(smoother.history) == 0


def test_video_processor_reports_posture_counts(tmp_path):
    """Verify that VideoProcessor includes posture_counts and distribution in output."""
    import cv2
    synth_input = tmp_path / "posture_test_in.mp4"
    synth_output = tmp_path / "posture_test_out.mp4"
    synth_summary = tmp_path / "posture_test_sum.json"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(synth_input), fourcc, 30.0, (640, 480))
    for i in range(10):
        frame = np.full((480, 640, 3), 40, dtype=np.uint8)
        out.write(frame)
    out.release()

    processor = VideoProcessor(model_path="models/pretrained/yolo11n-pose.pt")
    summary = processor.process_video(
        input_video_path=synth_input,
        output_video_path=synth_output,
        summary_path=synth_summary,
    )

    metrics = summary["processing_metrics"]
    assert "posture_counts" in metrics
    assert "posture_distribution" in metrics
    for p in ["STANDING", "SITTING", "LYING", "UNKNOWN"]:
        assert p in metrics["posture_counts"]
        assert p in metrics["posture_distribution"]


# =========================================================================
# Phase 6: Foreshortened Lying, Crouch Protection & Robust Keypoint Tests
# =========================================================================

def test_foreshortened_lying_posture():
    """
    Verify that an elongated body (AR >= 1.10) with perspective-foreshortened torso
    angle (e.g. 55 deg) is correctly classified as LYING rather than SITTING or UNKNOWN.
    """
    classifier = PostureClassifier()
    kpts = make_blank_keypoints()

    # Head and shoulders at y ~ 200
    kpts[0] = [250, 188, 0.90]  # nose
    kpts[5] = [260, 195, 0.90]  # left shoulder
    kpts[6] = [242, 205, 0.95]  # right shoulder

    # Hips at y ~ 238 (dy=38, dx=26 -> angle ~ 55.6 deg)
    kpts[11] = [282, 234, 0.98] # left hip
    kpts[12] = [272, 241, 0.99] # right hip

    # Legs extended horizontally
    kpts[13] = [319, 253, 0.95] # left knee
    kpts[14] = [307, 259, 0.95] # right knee
    kpts[15] = [334, 296, 0.90] # left ankle
    kpts[16] = [319, 305, 0.90] # right ankle

    # Horizontally elongated bounding box (w=190, h=165 -> W/H = 1.15)
    box = np.array([172, 165, 362, 330], dtype=np.float32)

    res = classifier.classify(kpts, box)
    assert res.posture == PostureType.LYING
    assert res.confidence >= 0.70
    assert any("Foreshortened torso on horizontal body" in r for r in res.reasons)


def test_lying_with_bent_knees_on_horizontal_body():
    """
    Verify that bent knees (e.g. 125 deg) on a horizontally elongated body (AR >= 1.10)
    remain classified as LYING (curled/side lying) rather than SITTING.
    """
    classifier = PostureClassifier()
    kpts = make_blank_keypoints()

    # Shoulders (horizontal)
    kpts[5] = [150, 220, 0.90]
    kpts[6] = [150, 240, 0.90]

    # Hips (horizontal line with shoulders, torso angle ~ 15 deg)
    kpts[11] = [230, 240, 0.90]
    kpts[12] = [230, 260, 0.90]

    # Knees bent upwards (sitting knee range: 110 deg)
    kpts[13] = [280, 210, 0.85]
    kpts[14] = [280, 230, 0.85]

    # Ankles
    kpts[15] = [330, 240, 0.85]
    kpts[16] = [330, 260, 0.85]

    # Wide box (w=220, h=80 -> W/H = 2.75)
    box = np.array([130, 195, 350, 275], dtype=np.float32)

    res = classifier.classify(kpts, box)
    assert res.posture == PostureType.LYING
    assert res.confidence >= 0.75
    assert any("Knee bent on horizontal body" in r for r in res.reasons)


def test_crouching_bend_over_straight_legs_rejected():
    """
    Verify that a person standing on straight vertical legs (knee > 145 deg)
    bending forward at the waist (AR <= 0.85, horizontal torso angle ~ 23 deg) is protected from false LYING.
    """
    classifier = PostureClassifier()
    kpts = make_blank_keypoints()

    # Person bent over: shoulders forward at (110, 185), hips back at (180, 160)
    # dx = 70, dy = 25 -> torso angle = atan2(25, 70) ~ 19.6 deg (horizontal torso)
    kpts[0] = [95, 195, 0.90]   # nose forward
    kpts[5] = [110, 180, 0.90]  # left shoulder
    kpts[6] = [110, 190, 0.90]  # right shoulder

    # Hips back and elevated
    kpts[11] = [180, 155, 0.90] # left hip
    kpts[12] = [180, 165, 0.90] # right hip

    # Knees vertically down from hips (y=215)
    kpts[13] = [180, 215, 0.90]
    kpts[14] = [180, 215, 0.90]

    # Ankles vertically down on ground (y=275) -> straight legs (knee angle ~ 180 deg)
    kpts[15] = [180, 275, 0.90]
    kpts[16] = [180, 275, 0.90]

    # Box: w = 115, h = 145 -> W/H = 0.79 (AR <= 0.85)
    box = np.array([85, 145, 200, 290], dtype=np.float32)

    res = classifier.classify(kpts, box)
    assert res.posture == PostureType.STANDING
    assert any("Crouch/bend protection" in r for r in res.reasons)


def test_relaxed_keypoint_confidence_fallback():
    """
    Verify that when shoulder and hip keypoints have partial low confidence (e.g. 0.28 < 0.35),
    the relaxed fallback successfully computes torso geometry rather than returning -1.0.
    """
    classifier = PostureClassifier(keypoint_conf_threshold=0.35)
    kpts = make_blank_keypoints()

    # Confidences at 0.28 (below 0.35, but above 0.25 fallback)
    kpts[5] = [100, 100, 0.28]  # left shoulder
    kpts[11] = [100, 180, 0.28] # left hip
    kpts[13] = [100, 240, 0.40] # knee
    kpts[15] = [100, 300, 0.40] # ankle

    box = np.array([80, 80, 130, 320], dtype=np.float32)

    features = classifier.extract_features(kpts, box)
    assert features["torso_angle"] >= 80.0
    assert features["torso_vertical_ratio"] > 0.0

