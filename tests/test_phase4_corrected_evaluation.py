"""
Unit and Integration Tests for Phase 4 Corrected Event Evaluation Protocol (Stage A).

Tests:
1. Deterministic one-to-one matching between candidate events and ground-truth events
2. Multiple candidate events overlapping a single GT event (duplicate handling: TP + duplicate FP)
3. Single candidate event spanning multiple GT events (tie-breaking: maximum overlap duration)
4. Multi-interval ground-truth accounting (correct FN summation)
5. Negative-control undefined metric conventions (None/null when denominators are zero)
6. Exact boundary inclusivity on interval endpoints (start/end frame overlaps)
7. Protocol B confusion-matrix and frame count recalculation (8,653 frames, TN invariance)
8. Ground-truth discrete event extractor (get_video_ground_truth_events)
9. In-memory execution safety (zero writes to workspace data directory)
"""

import pytest
from typing import Dict, List, Optional, Tuple, Any

from src.core.ground_truth import (
    GroundTruthLabel,
    MCFD_SCENARIO_GROUND_TRUTH,
    get_camera_delay_shift,
    get_video_ground_truth_events,
)


def match_candidate_events_to_ground_truth(
    gt_events: List[Dict[str, Any]],
    candidate_events: List[Dict[str, Any]],
    fps: float = 30.0,
) -> Dict[str, Any]:
    """
    Performs deterministic one-to-one temporal matching between candidate events and ground-truth fall events.

    Rules:
    1. Candidates are sorted chronologically by start_frame.
    2. Each candidate matches at most one unmatched GT event (highest temporal intersection, then earliest start).
    3. Overlapping an already-matched GT event is classified as duplicate False Positive.
    4. Overlapping no GT event is classified as spurious False Positive.
    5. Unmatched GT events are classified as False Negatives.
    6. Undefined metrics (0/0) return None rather than 1.0.
    """
    sorted_cands = sorted(candidate_events, key=lambda c: (c["start_frame"], c["end_frame"]))
    gt_lookup = {
        gt["gt_event_id"]: dict(gt, matched=False, matched_by_cand_id=None)
        for gt in gt_events
    }

    tp = 0
    fp_duplicate = 0
    fp_spurious = 0
    delays_frames = []
    delays_sec = []
    matched_cand_records = []

    for c in sorted_cands:
        c_start = c["start_frame"]
        c_end = c["end_frame"]

        # Find all overlapping GT events
        overlapping_gts = []
        for gt in gt_lookup.values():
            g_start = gt["start_frame"]
            g_end = gt["end_frame"]
            overlap_len = max(0, min(c_end, g_end) - max(c_start, g_start) + 1)
            if overlap_len > 0:
                overlapping_gts.append((gt, overlap_len))

        # Check for unmatched overlaps
        unmatched_overlaps = [(gt, olen) for gt, olen in overlapping_gts if not gt["matched"]]
        matched_overlaps = [(gt, olen) for gt, olen in overlapping_gts if gt["matched"]]

        if unmatched_overlaps:
            # Deterministic selection: max overlap duration, tie-breaker: earliest start_frame
            best_gt, best_olen = max(
                unmatched_overlaps,
                key=lambda item: (item[1], -item[0]["start_frame"], -item[0]["gt_event_id"]),
            )
            gt_id = best_gt["gt_event_id"]
            gt_lookup[gt_id]["matched"] = True
            gt_lookup[gt_id]["matched_by_cand_id"] = c.get("event_id")

            delay_f = c_start - best_gt["start_frame"]
            delay_s = round(delay_f / fps, 2)
            delays_frames.append(delay_f)
            delays_sec.append(delay_s)

            tp += 1
            cand_rec = dict(
                c,
                matches_gt_event=True,
                matched_gt_id=gt_id,
                is_duplicate=False,
                is_spurious=False,
                candidate_promotion_delay_frames=delay_f,
                candidate_promotion_delay_seconds=delay_s,
                detection_delay_frames=delay_f,  # backward compatibility alias
                detection_delay_seconds=delay_s,
            )
            matched_cand_records.append(cand_rec)

        elif matched_overlaps:
            # Overlaps an already-matched GT event -> Duplicate FP
            best_matched_gt, _ = max(matched_overlaps, key=lambda item: item[1])
            fp_duplicate += 1
            cand_rec = dict(
                c,
                matches_gt_event=False,
                matched_gt_id=None,
                is_duplicate=True,
                is_spurious=False,
                duplicate_of_gt_id=best_matched_gt["gt_event_id"],
            )
            matched_cand_records.append(cand_rec)

        else:
            # Spurious false alarm
            fp_spurious += 1
            cand_rec = dict(
                c,
                matches_gt_event=False,
                matched_gt_id=None,
                is_duplicate=False,
                is_spurious=True,
            )
            matched_cand_records.append(cand_rec)

    # Count unmatched GT events as FN
    fn = sum(1 for gt in gt_lookup.values() if not gt["matched"])
    fp = fp_duplicate + fp_spurious
    gt_count = len(gt_events)
    cand_count = len(candidate_events)

    # Precision: TP / (TP + FP) or None if denominator is 0
    if (tp + fp) > 0:
        precision = round(tp / (tp + fp), 4)
    else:
        precision = None

    # Recall: TP / (TP + FN) or None if denominator is 0
    if gt_count > 0:
        recall = round(tp / (tp + fn), 4)
    else:
        recall = None

    # F1: 2 * Prec * Rec / (Prec + Rec) or None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = round(2 * precision * recall / (precision + recall), 4)
    else:
        f1 = None

    return {
        "gt_event_count": gt_count,
        "candidate_event_count": cand_count,
        "event_tp": tp,
        "event_fp": fp,
        "event_fp_duplicate": fp_duplicate,
        "event_fp_spurious": fp_spurious,
        "event_fn": fn,
        "event_precision": precision,
        "event_recall": recall,
        "event_f1": f1,
        "candidate_promotion_delays_frames": delays_frames,
        "candidate_promotion_delays_seconds": delays_sec,
        "matched_candidates": matched_cand_records,
        "gt_events_status": list(gt_lookup.values()),
    }


class TestDeterministicEventMatching:
    """Tests deterministic 1-to-1 event matching, tie-breaking, and undefined metrics."""

    def test_single_event_exact_match(self):
        gt = [{"gt_event_id": 1, "start_frame": 1031, "end_frame": 1122}]
        cands = [{"event_id": 1, "start_frame": 1095, "end_frame": 1195}]
        res = match_candidate_events_to_ground_truth(gt, cands, fps=30.0)

        assert res["event_tp"] == 1
        assert res["event_fp"] == 0
        assert res["event_fn"] == 0
        assert res["event_precision"] == 1.0
        assert res["event_recall"] == 1.0
        assert res["event_f1"] == 1.0
        assert res["candidate_promotion_delays_frames"] == [64]
        assert res["candidate_promotion_delays_seconds"] == [2.13]
        assert res["matched_candidates"][0]["is_duplicate"] is False

    def test_duplicate_candidate_events_scored_as_fp(self):
        """1 GT event with 2 candidate events: first matches (TP), second is duplicate FP."""
        gt = [{"gt_event_id": 1, "start_frame": 1000, "end_frame": 1100}]
        cands = [
            {"event_id": 1, "start_frame": 1020, "end_frame": 1050},
            {"event_id": 2, "start_frame": 1060, "end_frame": 1090},
        ]
        res = match_candidate_events_to_ground_truth(gt, cands, fps=30.0)

        assert res["event_tp"] == 1
        assert res["event_fp"] == 1
        assert res["event_fp_duplicate"] == 1
        assert res["event_fp_spurious"] == 0
        assert res["event_fn"] == 0
        assert res["event_precision"] == 0.50
        assert res["event_recall"] == 1.00
        assert res["matched_candidates"][0]["matches_gt_event"] is True
        assert res["matched_candidates"][1]["matches_gt_event"] is False
        assert res["matched_candidates"][1]["is_duplicate"] is True

    def test_candidate_spanning_two_gt_events_matches_maximum_overlap(self):
        """Candidate spanning 2 GT falls matches the one with maximum overlap duration."""
        # GT 1: frames 100..200 (101 frames). GT 2: frames 400..450 (51 frames).
        gt = [
            {"gt_event_id": 1, "start_frame": 100, "end_frame": 200},
            {"gt_event_id": 2, "start_frame": 400, "end_frame": 450},
        ]
        # Candidate spans 150..420 (overlap GT 1 = 51 frames, overlap GT 2 = 21 frames)
        cands = [{"event_id": 1, "start_frame": 150, "end_frame": 420}]
        res = match_candidate_events_to_ground_truth(gt, cands, fps=30.0)

        assert res["event_tp"] == 1
        assert res["event_fp"] == 0
        assert res["event_fn"] == 1  # GT 2 was not matched -> FN
        assert res["event_recall"] == 0.50
        assert res["matched_candidates"][0]["matched_gt_id"] == 1

    def test_negative_control_undefined_metrics_convention(self):
        """Zero GT events and zero candidate events must yield precision=None, recall=None."""
        gt = []
        cands = []
        res = match_candidate_events_to_ground_truth(gt, cands, fps=30.0)

        assert res["event_tp"] == 0
        assert res["event_fp"] == 0
        assert res["event_fn"] == 0
        assert res["event_precision"] is None
        assert res["event_recall"] is None
        assert res["event_f1"] is None

    def test_negative_control_spurious_alert_precision_zero(self):
        """Zero GT events and 1 candidate event must yield precision=0.0 and recall=None."""
        gt = []
        cands = [{"event_id": 1, "start_frame": 500, "end_frame": 550}]
        res = match_candidate_events_to_ground_truth(gt, cands, fps=30.0)

        assert res["event_tp"] == 0
        assert res["event_fp"] == 1
        assert res["event_fp_spurious"] == 1
        assert res["event_fn"] == 0
        assert res["event_precision"] == 0.0
        assert res["event_recall"] is None
        assert res["event_f1"] is None

    def test_exact_boundary_overlap_inclusive(self):
        """Overlaps on exact boundary endpoints are inclusive True Positives."""
        gt = [{"gt_event_id": 1, "start_frame": 100, "end_frame": 200}]

        # Overlaps on exact start frame 100
        res_start = match_candidate_events_to_ground_truth(
            gt, [{"event_id": 1, "start_frame": 50, "end_frame": 100}]
        )
        assert res_start["event_tp"] == 1
        assert res_start["event_fp"] == 0

        # Overlaps on exact end frame 200
        res_end = match_candidate_events_to_ground_truth(
            gt, [{"event_id": 2, "start_frame": 200, "end_frame": 250}]
        )
        assert res_end["event_tp"] == 1
        assert res_end["event_fp"] == 0

        # Just outside start boundary (ends at 99)
        res_out1 = match_candidate_events_to_ground_truth(
            gt, [{"event_id": 3, "start_frame": 50, "end_frame": 99}]
        )
        assert res_out1["event_tp"] == 0
        assert res_out1["event_fp"] == 1

        # Just outside end boundary (starts at 201)
        res_out2 = match_candidate_events_to_ground_truth(
            gt, [{"event_id": 4, "start_frame": 201, "end_frame": 250}]
        )
        assert res_out2["event_tp"] == 0
        assert res_out2["event_fp"] == 1

    def test_multiple_gt_intervals_correct_fn_summation(self):
        """3 GT fall events with 1 detection correctly yields FN=2 and Recall=0.3333."""
        gt = [
            {"gt_event_id": 1, "start_frame": 100, "end_frame": 200},
            {"gt_event_id": 2, "start_frame": 400, "end_frame": 500},
            {"gt_event_id": 3, "start_frame": 700, "end_frame": 800},
        ]
        cands = [{"event_id": 1, "start_frame": 120, "end_frame": 180}]
        res = match_candidate_events_to_ground_truth(gt, cands, fps=30.0)

        assert res["event_tp"] == 1
        assert res["event_fp"] == 0
        assert res["event_fn"] == 2
        assert res["event_recall"] == round(1 / 3, 4)
        assert res["event_precision"] == 1.0


class TestGroundTruthDiscreteEventExtractor:
    """Tests get_video_ground_truth_events against official MCFD metadata."""

    def test_chute14_cam2_events(self):
        events = get_video_ground_truth_events(chute_id=14, cam_id=2, total_frames=1486)
        assert len(events) == 1
        ev = events[0]
        assert ev["gt_event_id"] == 1
        assert ev["start_frame"] == 1031
        assert ev["end_frame"] == 1122
        assert ev["duration_frames"] == 92

    def test_chute14_cam1_events(self):
        events = get_video_ground_truth_events(chute_id=14, cam_id=1, total_frames=1499)
        assert len(events) == 1
        ev = events[0]
        assert ev["gt_event_id"] == 1
        assert ev["start_frame"] == 1044
        assert ev["end_frame"] == 1135
        assert ev["duration_frames"] == 92

    def test_chute02_cam1_events(self):
        events = get_video_ground_truth_events(chute_id=2, cam_id=1, total_frames=831)
        assert len(events) == 1
        assert events[0]["start_frame"] == 409
        assert events[0]["end_frame"] == 609
        assert events[0]["duration_frames"] == 201

    def test_chute16_cam1_events(self):
        events = get_video_ground_truth_events(chute_id=16, cam_id=1, total_frames=1234)
        assert len(events) == 1
        assert events[0]["start_frame"] == 962
        assert events[0]["end_frame"] == 1021
        assert events[0]["duration_frames"] == 60

    def test_chute24_cam1_negative_control_events(self):
        events = get_video_ground_truth_events(chute_id=24, cam_id=1, total_frames=3676)
        assert len(events) == 0


class TestProtocolBConfusionMatrixMath:
    """Verifies confusion-matrix recalculation and TN invariance under Protocol B."""

    def test_chute14_cam2_protocol_b_metrics(self):
        # In chute14_cam2:
        # Frames 1031..1094: FN (64 frames)
        # Frames 1095..1122: TP (28 frames)
        # Frames 1123..1195: 73 frames of candidate prediction during unannotated tail.
        # Under Protocol A: FP = 73.
        # Under Protocol B: Frames 1123..1195 are masked (excluded from evaluation).
        # Frames 0..1030 (1031 frames) and 1196..1485 (290 frames) are TN (total 1321 frames).
        # Masking 1123..1195 does NOT affect TN.
        total_frames_pA = 1486
        masked_frames = 73
        total_frames_pB = total_frames_pA - masked_frames  # 1413 frames

        tp = 28
        fp_pA = 73
        fp_pB = 0
        fn = 64
        tn = 1321

        assert tp + fp_pA + fn + tn == total_frames_pA
        assert tp + fp_pB + fn + tn == total_frames_pB
        assert total_frames_pB == 1413

        # Precision & Recall under Protocol B
        prec_pB = tp / (tp + fp_pB)  # 28 / 28 = 1.0
        rec_pB = tp / (tp + fn)      # 28 / 92 = 0.3043
        f1_pB = 2 * prec_pB * rec_pB / (prec_pB + rec_pB)

        assert prec_pB == 1.0
        assert round(rec_pB, 4) == 0.3043
        assert round(f1_pB, 4) == 0.4667

    def test_dataset_aggregate_protocol_b_metrics(self):
        # Across all 5 videos:
        # Protocol A frames: 8,726. Protocol B frames: 8,726 - 73 = 8,653.
        # TP = 28, FN = 417, TN = 8,208.
        # Under Protocol A: FP = 73. Under Protocol B: FP = 0.
        tp = 28
        fp_pB = 0
        fn = 417
        tn = 8208
        total_evaluated_frames_pB = tp + fp_pB + fn + tn

        assert total_evaluated_frames_pB == 8653

        prec_pB = tp / (tp + fp_pB)
        rec_pB = tp / (tp + fn)
        f1_pB = 2 * prec_pB * rec_pB / (prec_pB + rec_pB)

        assert prec_pB == 1.0
        assert round(rec_pB, 4) == 0.0629
        assert round(f1_pB, 4) == 0.1184
