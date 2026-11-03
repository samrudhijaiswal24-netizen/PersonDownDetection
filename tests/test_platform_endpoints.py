"""
Integration Tests for Person Down Detection Multi-Page Platform Endpoints.

Tests all API endpoints for:
- Dashboard statistics (/api/dashboard/stats)
- Dataset Explorer (/api/dataset, /api/dataset/rescan)
- Event logging & CSV export (/api/events, /api/events/export/csv)
- Model evaluations & reports (/api/evaluations, /api/evaluations/download/{file})
- Spatial zones & calibration previews (/api/zones, /api/zones/preview/{cam})
- System diagnostics & health (/api/system/diagnostics)
- Multi-page HTML route resolution
- Path-traversal security safeguards
"""

import pytest
from fastapi.testclient import TestClient
from src.api.app import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_dashboard_stats_endpoint(client):
    res = client.get("/api/dashboard/stats")
    assert res.status_code == 200
    data = res.json()
    assert data["total_dataset_videos"] >= 8
    assert data["system_status"] == "ONLINE"
    assert data["benchmark_recall"] == 0.50
    assert data["benchmark_precision"] == 1.00
    assert data["benchmark_f1"] == 0.6667
    assert data["mean_alert_delay_seconds"] == 2.55
    assert isinstance(data["recent_jobs"], list)
    assert isinstance(data["recent_events"], list)


def test_dataset_discovery_and_filtering(client):
    res = client.get("/api/dataset")
    assert res.status_code == 200
    data = res.json()
    assert data["total_count"] >= 8
    assert "samples" in data["folders"]
    assert "AVI" in data["formats"] or "MP4" in data["formats"]

    # Test folder filter
    res_samples = client.get("/api/dataset?folder=samples")
    assert res_samples.status_code == 200
    assert res_samples.json()["filtered_count"] >= 1

    # Test search filter
    res_search = client.get("/api/dataset?search=chute14")
    assert res_search.status_code == 200
    assert res_search.json()["filtered_count"] >= 2
    for vid in res_search.json()["videos"]:
        assert "chute14" in vid["filename"].lower() or "chute14" in vid["scenario_description"].lower()


def test_dataset_rescan(client):
    res = client.post("/api/dataset/rescan")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "SUCCESS"
    assert data["discovered_count"] >= 8


def test_events_endpoint(client):
    res = client.get("/api/events")
    assert res.status_code == 200
    data = res.json()
    assert data["total_events"] >= 2
    for ev in data["events"]:
        assert ev["classification_type"] == "RESEARCH_CANDIDATE_EVENT"
        assert ev["status"] in ("BENCHMARK_CONFIRMED", "CANDIDATE_CONFIRMED")
        assert ev["peak_posture"] in ("LYING", "UNKNOWN")


def test_events_csv_export(client):
    res = client.get("/api/events/export/csv")
    assert res.status_code == 200
    assert "text/csv" in res.headers["content-type"]
    assert "attachment; filename=" in res.headers["content-disposition"]
    lines = res.text.strip().split("\r\n") if "\r\n" in res.text else res.text.strip().split("\n")
    assert len(lines) >= 3  # Header + at least 2 events
    assert lines[0].startswith("Event ID,Job ID,Source Video")


def test_evaluations_summary(client):
    res = client.get("/api/evaluations")
    assert res.status_code == 200
    data = res.json()
    assert data["report_metadata"]["phase"] == 6
    assert data["event_level_metrics"]["event_precision"] == 1.0
    assert data["event_level_metrics"]["event_recall"] == 0.5
    assert len(data["per_video_results"]) == 5
    assert len(data["baseline_hashes"]) >= 9
    assert len(data["available_reports"]) >= 4


def test_evaluations_download(client):
    res = client.get("/api/evaluations/download/phase6_detection_improvement_report.json")
    assert res.status_code == 200
    assert "application/json" in res.headers["content-type"]

    # Attempt path traversal rejection
    res_bad = client.get("/api/evaluations/download/../configs/zones_config.yaml")
    assert res_bad.status_code in (404, 400)


def test_zones_and_previews(client):
    res = client.get("/api/zones")
    assert res.status_code == 200
    data = res.json()
    assert "cam1" in data
    assert "cam2" in data
    assert "cam1_chute09" in data
    assert data["cam1"]["status"] == "VALIDATED"
    assert data["cam1_chute09"]["status"] == "REQUIRES_MANUAL_REVIEW"

    # Calibration previews
    res_p1 = client.get("/api/zones/preview/cam1")
    assert res_p1.status_code == 200
    assert res_p1.headers["content-type"] == "image/jpeg"

    res_p2 = client.get("/api/zones/preview/cam2")
    assert res_p2.status_code == 200
    assert res_p2.headers["content-type"] == "image/jpeg"


def test_system_diagnostics(client):
    res = client.get("/api/system/diagnostics")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ONLINE"
    assert data["scaling_engine_verified"] is True
    assert "data_uploads" in data["directories_status"]
    assert data["directories_status"]["data_uploads"]["exists"] is True


def test_process_path_traversal_protection(client):
    # Test path traversal injection attempt
    res = client.post("/api/process", json={"sample_path": "../../etc/passwd"})
    assert res.status_code == 400

    # Test valid path
    res_valid = client.post("/api/process", json={
        "sample_path": "data/samples/floor_fall.mp4",
        "camera_id": "cam1",
        "max_frames": 10,
    })
    assert res_valid.status_code == 200
    assert res_valid.json()["status"] == "QUEUED"


def test_page_routes(client):
    for path in ["/", "/monitoring", "/dataset", "/alerts", "/evaluation", "/zones", "/diagnostics"]:
        res = client.get(path)
        assert res.status_code == 200
        assert "<!DOCTYPE html>" in res.text


def test_video_streaming_and_playback(client):
    # Test streaming existing annotated video with inline disposition
    res = client.get("/api/videos/annotated_floor_fall.mp4")
    assert res.status_code in [200, 206]
    assert res.headers["content-type"] == "video/mp4"
    assert "attachment" not in res.headers.get("content-disposition", "")

    # Test non-existent video returns 404
    res_404 = client.get("/api/videos/nonexistent_clip_xyz.mp4")
    assert res_404.status_code == 404


def test_video_check_annotated_endpoint(client):
    # Test checking a playable MP4 sample that also has annotated video
    res = client.get("/api/videos/check_annotated/data/samples/floor_fall.mp4")
    assert res.status_code == 200
    data = res.json()
    assert data["source_playable"] is True
    assert data["source_format"] == "MP4"
    assert data["has_annotated"] is True
    assert "annotated_floor_fall.mp4" in data["annotated_filename"]

    # Test checking an AVI benchmark file
    res_avi = client.get("/api/videos/check_annotated/data/samples/mcfd/mcfd_chute14_cam1.avi")
    assert res_avi.status_code == 200
    data_avi = res_avi.json()
    assert data_avi["source_playable"] is False
    assert data_avi["source_format"] == "AVI"
    assert data_avi["has_annotated"] is True
    assert "chute14_cam1" in data_avi["annotated_filename"]


def test_video_source_preview_and_security(client):
    # Test streaming source MP4 video
    res = client.get("/api/videos/source/data/samples/floor_fall.mp4")
    assert res.status_code == 200
    assert res.headers["content-type"] == "video/mp4"

    # Test path traversal protection on source endpoint
    res_sec = client.get("/api/videos/source/../../windows/system32/cmd.exe")
    assert res_sec.status_code in [400, 403, 404]


def test_video_download_endpoint(client):
    # Test downloading video with attachment disposition
    res = client.get("/api/videos/download/annotated_floor_fall.mp4")
    assert res.status_code == 200
    assert res.headers["content-type"] == "video/mp4"
    assert "attachment" in res.headers.get("content-disposition", "")


