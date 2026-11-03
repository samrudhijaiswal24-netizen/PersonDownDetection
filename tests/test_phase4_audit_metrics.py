"""
Unit tests for Phase 4.1 Independent Evaluation Audit and Event-Level Metrics.

Verifies:
- Event-level matching rules (temporal overlap, delay computation, false event rejection)
- Frame-level confusion matrix evaluation under strict benchmark vs masked protocols
- Precision, recall, and F1 calculations under edge cases (e.g. zero division safety)
- Physical capture timebase calculations (30 FPS vs 120 FPS header)
"""

import pytest
from typing import List, Dict, Tuple, Optional


def match_events(
    gt_intervals: List[Tuple[int, int]],
    candidate_events: List[Tuple[int, int]],
    fps: float = 30.0,
) -> Dict[str, any]:
    """
    Matches candidate events to ground-truth fall intervals.
    A candidate event matches a GT interval if there is temporal overlap.
    """
    gt_matched = [False] * len(gt_intervals)
    cand_matched = [False] * len(candidate_events)
    delays = []

    for c_idx, (c_start, c_end) in enumerate(candidate_events):
        for g_idx, (g_start, g_end) in enumerate(gt_intervals):
            # Check overlap: max(c_start, g_start) <= min(c_end, g_end)
            if max(c_start, g_start) <= min(c_end, g_end):
                cand_matched[c_idx] = True
                gt_matched[g_idx] = True
                delay_frames = c_start - g_start
                delays.append(round(delay_frames / fps, 2))
                break

    tp = sum(cand_matched)
    fp = len(candidate_events) - tp
    fn = len(gt_intervals) - sum(gt_matched)

    prec = (tp / (tp + fp)) if (tp + fp) > 0 else (1.0 if fp == 0 else 0.0)
    rec = (tp / (tp + fn)) if (tp + fn) > 0 else (1.0 if len(gt_intervals) == 0 else 0.0)
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "delays": delays,
    }


def compute_frame_matrix(
    y_true: List[str],
    y_pred: List[str],
    pos_label: str = "GROUND_LYING",
    pred_pos_label: str = "PERSON_DOWN_CANDIDATE",
    ignore_indices: Optional[List[int]] = None,
) -> Dict[str, any]:
    """Computes binary frame-level confusion matrix with optional masked indices."""
    tp, fp, fn, tn = 0, 0, 0, 0
    ignored = set(ignore_indices or [])

    for i, (t, p) in enumerate(zip(y_true, y_pred)):
        if i in ignored:
            continue
        is_gt_pos = (t == pos_label)
        is_pred_pos = (p == pred_pos_label)

        if is_gt_pos and is_pred_pos:
            tp += 1
        elif not is_gt_pos and is_pred_pos:
            fp += 1
        elif is_gt_pos and not is_pred_pos:
            fn += 1
        else:
            tn += 1

    prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
    }


class TestEventMatchingLogic:
    """Tests the event matching algorithm, temporal overlap, and delay calculation."""

    def test_single_matching_event(self):
        # GT interval: frames 1031 to 1122. Predicted event: frames 1095 to 1195.
        gt_intervals = [(1031, 1122)]
        candidate_events = [(1095, 1195)]
        res = match_events(gt_intervals, candidate_events, fps=30.0)

        assert res["tp"] == 1
        assert res["fp"] == 0
        assert res["fn"] == 0
        assert res["precision"] == 1.0
        assert res["recall"] == 1.0
        assert res["f1"] == 1.0
        # Delay: 1095 - 1031 = 64 frames = 64 / 30.0 = 2.13s
        assert res["delays"] == [2.13]

    def test_spurious_false_alarm_event(self):
        # Negative control video: 0 GT fall intervals, 1 spurious candidate event
        gt_intervals = []
        candidate_events = [(500, 560)]
        res = match_events(gt_intervals, candidate_events, fps=30.0)

        assert res["tp"] == 0
        assert res["fp"] == 1
        assert res["fn"] == 0
        assert res["precision"] == 0.0
        assert res["recall"] == 1.0
        assert res["f1"] == 0.0

    def test_negative_control_perfect_pass(self):
        # Negative control video: 0 GT intervals, 0 candidate events
        gt_intervals = []
        candidate_events = []
        res = match_events(gt_intervals, candidate_events, fps=30.0)

        assert res["tp"] == 0
        assert res["fp"] == 0
        assert res["fn"] == 0
        assert res["precision"] == 1.0
        assert res["recall"] == 1.0
        assert res["f1"] == 1.0

    def test_missed_event(self):
        # GT fall interval exists, but 0 candidate events triggered
        gt_intervals = [(409, 609)]
        candidate_events = []
        res = match_events(gt_intervals, candidate_events, fps=30.0)

        assert res["tp"] == 0
        assert res["fp"] == 0
        assert res["fn"] == 1
        assert res["precision"] == 1.0
        assert res["recall"] == 0.0
        assert res["f1"] == 0.0

    def test_dataset_aggregate_event_matching(self):
        # 5 separate videos: 4 fall scenarios + 1 negative control.
        # Video 1 (chute14_cam1): GT=(1044, 1135), Pred=[] -> FN
        # Video 2 (chute14_cam2): GT=(1031, 1122), Pred=[(1095, 1195)] -> TP (delay 2.13s)
        # Video 3 (chute02_cam1): GT=(409, 609),   Pred=[] -> FN
        # Video 4 (chute16_cam1): GT=(962, 1021),  Pred=[] -> FN
        # Video 5 (chute24_cam1): GT=None,         Pred=[] -> TN
        videos = [
            ([(1044, 1135)], []),
            ([(1031, 1122)], [(1095, 1195)]),
            ([(409, 609)], []),
            ([(962, 1021)], []),
            ([], []),
        ]
        tot_tp, tot_fp, tot_fn = 0, 0, 0
        delays = []
        for gt_list, cand_list in videos:
            res = match_events(gt_list, cand_list, fps=30.0)
            tot_tp += res["tp"]
            tot_fp += res["fp"]
            tot_fn += res["fn"]
            delays.extend(res["delays"])

        prec = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) > 0 else 0.0
        rec = tot_tp / (tot_tp + tot_fn) if (tot_tp + tot_fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        assert tot_tp == 1
        assert tot_fp == 0
        assert tot_fn == 3
        assert prec == 1.0
        assert rec == 0.25
        assert round(f1, 4) == 0.4000
        assert delays == [2.13]


class TestFrameLevelEvaluationProtocols:
    """Verifies frame-level metric calculations across strict benchmark vs masked protocols."""

    def test_chute14_cam2_strict_benchmark_protocol(self):
        # 1486 frames: 92 GT ground frames (1031..1122).
        # Predicted candidate: 101 frames (1095..1195).
        # Frames 1095..1122 (28 frames) are TP.
        # Frames 1123..1195 (73 frames) are FP because GT is BACKGROUND.
        # Frames 1031..1094 (64 frames) are FN.
        y_true = ["BACKGROUND"] * 1031 + ["GROUND_LYING"] * 92 + ["BACKGROUND"] * (1486 - 1123)
        y_pred = ["NORMAL"] * 1095 + ["PERSON_DOWN_CANDIDATE"] * 101 + ["NORMAL"] * (1486 - 1196)

        res = compute_frame_matrix(y_true, y_pred)
        assert res["tp"] == 28
        assert res["fp"] == 73
        assert res["fn"] == 64
        assert res["tn"] == 1321
        assert res["precision"] == round(28 / 101, 4)  # 0.2772
        assert res["recall"] == round(28 / 92, 4)     # 0.3043
        assert res["f1"] == 0.2902

    def test_chute14_cam2_masked_post_event_protocol(self):
        # Protocol B: Mask unannotated post-event interval (1123..1195)
        y_true = ["BACKGROUND"] * 1031 + ["GROUND_LYING"] * 92 + ["BACKGROUND"] * (1486 - 1123)
        y_pred = ["NORMAL"] * 1095 + ["PERSON_DOWN_CANDIDATE"] * 101 + ["NORMAL"] * (1486 - 1196)
        ignore = list(range(1123, 1196))  # 73 post-event frames

        res = compute_frame_matrix(y_true, y_pred, ignore_indices=ignore)
        assert res["tp"] == 28
        assert res["fp"] == 0
        assert res["fn"] == 64
        assert res["precision"] == 1.0
        assert res["recall"] == round(28 / 92, 4)  # 0.3043
        assert res["f1"] == round(2 * 1.0 * (28/92) / (1.0 + 28/92), 4)  # 0.4667


class TestPhysicalTimebaseMath:
    """Verifies that physical video timebase calculations reflect 30 FPS vs 120 FPS header."""

    def test_frame_to_time_conversion(self):
        real_fps = 30.0
        header_fps = 120.0
        frames = 45

        real_duration = frames / real_fps
        header_duration = frames / header_fps

        assert real_duration == 1.5
        assert header_duration == 0.375

    def test_delay_math(self):
        impact_frame = 1031
        candidate_frame = 1095
        delta_f = candidate_frame - impact_frame  # 64 frames

        real_delay = delta_f / 30.0
        assert round(real_delay, 2) == 2.13
