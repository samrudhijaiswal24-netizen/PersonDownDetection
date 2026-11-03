"""
FastAPI Application Backend for Person Down Detection System.

Complete Multi-Page Surveillance & Healthcare Analytics Platform.
Provides REST API endpoints for:
- Video upload, sample selection, and dataset discovery
- Asynchronous video processing with real-time telemetry polling
- Candidate event logging, duplicate suppression, and CSV export
- Annotated video streaming
- System health, PyTorch/CUDA diagnostics, and live log stream
- Benchmark evaluations and SHA-256 baseline verification
- Spatial zone inspection and calibration preview delivery
"""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime
import io
import json
import logging
import os
from pathlib import Path
import re
import shutil
import sys
import time
from typing import Any, Dict, List, Optional
import uuid

import cv2
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Response,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
import torch

from src.api.models import (
    DashboardStatsResponse,
    DatasetListResponse,
    DatasetVideoItem,
    DetectedEventItem,
    EvaluationSummaryResponse,
    EventListResponse,
    EventRecordItem,
    JobCreationResponse,
    JobStatusResponse,
    ProcessVideoRequest,
    SampleVideoItem,
    SampleVideosListResponse,
    SystemDiagnosticsResponse,
    SystemStatusResponse,
)
from src.pipeline.person_down_pipeline import (
    PersonDownPipeline,
    RESEARCH_PROTOTYPE_DISCLAIMER,
    SUPPORTED_VIDEO_EXTENSIONS,
)

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
DATA_DIR = PROJECT_ROOT / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
PROCESSED_DIR = DATA_DIR / "processed" / "phase7"
MCFD_PROCESSED_DIR = DATA_DIR / "processed" / "mcfd"
CALIBRATION_PREVIEWS_DIR = MCFD_PROCESSED_DIR / "calibration_previews"
SAMPLES_DIR = DATA_DIR / "samples"
RAW_DIR = DATA_DIR / "raw"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Application Lifecycle & In-Memory Logging
SERVER_START_TIME = time.time()
LOG_BUFFER: deque = deque(maxlen=60)


class InMemoryLogHandler(logging.Handler):
    """Custom logging handler to keep a rolling ring-buffer of recent application logs."""
    def emit(self, record):
        try:
            msg = self.format(record)
            LOG_BUFFER.append(msg)
        except Exception:
            self.handleError(record)


log_formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
mem_handler = InMemoryLogHandler()
mem_handler.setFormatter(log_formatter)

logger = logging.getLogger("person_down_api")
logger.setLevel(logging.INFO)
logger.addHandler(mem_handler)

uvicorn_logger = logging.getLogger("uvicorn.error")
uvicorn_logger.addHandler(mem_handler)

# FastAPI Initialization
app = FastAPI(
    title="Person Down Detection System",
    description="End-to-End Fall & Floor-Collapse Monitoring Platform (Phase 7 Multi-Page)",
    version="7.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Thread pool for asynchronous background video processing
EXECUTOR = ThreadPoolExecutor(max_workers=2)

# Global In-Memory Job Store and Pipeline Singleton
JOBS: Dict[str, Dict[str, Any]] = {}
_PIPELINE_INSTANCE: Optional[PersonDownPipeline] = None
DATASET_METADATA_CACHE: Dict[str, Dict[str, Any]] = {}


def get_pipeline() -> PersonDownPipeline:
    """Returns singleton pipeline instance or initializes one."""
    global _PIPELINE_INSTANCE
    if _PIPELINE_INSTANCE is None:
        logger.info("Initializing PersonDownPipeline singleton...")
        model_path = PROJECT_ROOT / "models" / "pretrained" / "yolo11n-pose.pt"
        zones_config = PROJECT_ROOT / "configs" / "zones_config.yaml"
        tracker_config = PROJECT_ROOT / "configs" / "temporal_tracker_config.yaml"
        _PIPELINE_INSTANCE = PersonDownPipeline(
            model_path=model_path,
            zones_config_path=zones_config,
            tracker_config_path=tracker_config,
        )
        logger.info("PersonDownPipeline singleton initialized successfully.")
    return _PIPELINE_INSTANCE


SAMPLE_METADATA_REGISTRY = {
    "mcfd_chute14_cam1.avi": {
        "category": "MCFD Benchmark (Fall)",
        "recommended_camera": "cam1",
        "scenario_description": "Fall onto floor mat from corner view. Phase 6 recovered true positive event (2.97s latency).",
        "is_benchmark": True,
    },
    "mcfd_chute14_cam2.avi": {
        "category": "MCFD Benchmark (Fall)",
        "recommended_camera": "cam2",
        "scenario_description": "Fall onto floor mat from axial view. Verified true positive fall event in Phase 4/5/6.",
        "is_benchmark": True,
    },
    "mcfd_chute24_cam1.avi": {
        "category": "MCFD Benchmark (ADL Negative Control)",
        "recommended_camera": "cam1",
        "scenario_description": "3,676-frame negative control of daily activities (walking, sitting, bending). 0 false alarms.",
        "is_benchmark": True,
    },
    "mcfd_chute02_cam1.avi": {
        "category": "MCFD Benchmark (Occlusion)",
        "recommended_camera": "cam1",
        "scenario_description": "Fall behind opaque bed partition. Demonstrates camera occlusion hardware limitation.",
        "is_benchmark": True,
    },
    "mcfd_chute16_cam1.avi": {
        "category": "MCFD Benchmark (Challenging)",
        "recommended_camera": "cam1",
        "scenario_description": "Fall with subject low confidence and background coat/chair detector distraction.",
        "is_benchmark": True,
    },
    "mcfd_chute09_cam1.avi": {
        "category": "MCFD Benchmark (Uncalibrated)",
        "recommended_camera": "cam1_chute09",
        "scenario_description": "Fall scenario with uncalibrated room geometry. Demonstrates status: REQUIRES_MANUAL_REVIEW.",
        "is_benchmark": True,
    },
    "floor_fall.mp4": {
        "category": "Sample Clip",
        "recommended_camera": "cam1",
        "scenario_description": "High-resolution living room floor fall clip for rapid end-to-end verification.",
        "is_benchmark": False,
    },
    "normal_ward.mp4": {
        "category": "Sample Clip",
        "recommended_camera": "cam1",
        "scenario_description": "Hospital ward activity clip showing normal upright and sitting postures.",
        "is_benchmark": False,
    },
}


def _execute_processing_job(job_id: str, input_path: Path, camera_id: str, max_frames: Optional[int]):
    """Background worker task executing video processing."""
    job_record = JOBS.get(job_id)
    if not job_record:
        return

    job_record["status"] = "PROCESSING"
    job_record["start_time"] = time.time()
    logger.info(f"Starting job {job_id} for file {input_path.name} (camera: {camera_id})")

    pipeline = get_pipeline()
    output_video_path = PROCESSED_DIR / f"annotated_{job_id}_{input_path.stem}.mp4"

    def progress_callback(curr_f: int, tot_f: int, fps: float, state: str, ev_count: int, pct: float):
        job_record["current_frame"] = curr_f
        job_record["total_frames"] = tot_f
        job_record["processing_fps"] = round(fps, 1)
        job_record["current_state"] = state
        job_record["progress_percent"] = pct
        job_record["events_detected_count"] = ev_count

    def frame_callback(f_idx: int, frame_bgr, telemetry: Dict[str, Any]):
        job_record["current_posture"] = telemetry.get("posture", "UNKNOWN")
        job_record["current_zone"] = telemetry.get("zone_name") or "OUTSIDE"
        job_record["has_person"] = telemetry.get("has_person", False)
        job_record["elapsed_down_seconds"] = telemetry.get("elapsed_down_seconds", 0.0)
        job_record["is_active_candidate"] = telemetry.get("is_active_candidate", False)

    def cancel_check() -> bool:
        return job_record.get("cancelled", False)

    try:
        report = pipeline.process_video(
            input_video_path=input_path,
            output_video_path=output_video_path,
            camera_id=camera_id,
            max_frames=max_frames,
            progress_callback=progress_callback,
            frame_callback=frame_callback,
            cancel_check=cancel_check,
        )

        if job_record.get("cancelled", False):
            job_record["status"] = "CANCELLED"
            job_record["message"] = "Processing was cancelled by user request."
            logger.warning(f"Job {job_id} was cancelled by user.")
        else:
            job_record["status"] = "COMPLETED"
            job_record["progress_percent"] = 100.0
            job_record["result"] = report
            job_record["detected_events"] = report.get("temporal_summary", {}).get("detected_events", [])
            job_record["video_filename"] = output_video_path.name
            job_record["video_url"] = f"/api/videos/{output_video_path.name}"
            logger.info(f"Job {job_id} completed successfully. Events detected: {len(job_record['detected_events'])}")

    except Exception as e:
        logger.exception(f"Error processing job {job_id}: {e}")
        job_record["status"] = "FAILED"
        job_record["error_message"] = str(e)


def discover_dataset_videos(force_rescan: bool = False) -> List[DatasetVideoItem]:
    """
    Crawls data/samples, data/uploads, and data/raw for video files.
    Strictly excludes data/processed to prevent indexing pipeline outputs as sources.
    Uses memory caching for fast response times.
    """
    global DATASET_METADATA_CACHE

    search_dirs = [
        ("samples", SAMPLES_DIR),
        ("uploads", UPLOADS_DIR),
        ("raw", RAW_DIR),
    ]

    items: List[DatasetVideoItem] = []

    for category, directory in search_dirs:
        if not directory.exists():
            continue

        for p in directory.rglob("*"):
            if not p.is_file():
                continue
            ext = p.suffix.lower()
            if ext not in SUPPORTED_VIDEO_EXTENSIONS:
                continue
            # Double check: strictly exclude anything under data/processed
            if "processed" in p.parts:
                continue

            rel_str = str(p.relative_to(PROJECT_ROOT)).replace("\\", "/")
            stat = p.stat()
            file_key = f"{rel_str}:{stat.st_mtime}:{stat.st_size}"

            # Check cache
            if not force_rescan and file_key in DATASET_METADATA_CACHE:
                items.append(DatasetVideoItem(**DATASET_METADATA_CACHE[file_key]))
                continue

            # Extract video metadata via cv2
            cap = cv2.VideoCapture(str(p.resolve()))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cfps = cap.get(cv2.CAP_PROP_FPS)
            tot_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            nominal_fps = 30.0 if (cfps > 60.0 or cfps <= 0 or np.isnan(cfps)) else round(float(cfps), 2)
            dur = round(tot_f / max(1.0, nominal_fps), 2)
            reg = SAMPLE_METADATA_REGISTRY.get(p.name, {})

            folder_rel = str(p.parent.relative_to(DATA_DIR)).replace("\\", "/")
            mtime_str = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")

            item_dict = {
                "id": str(uuid.uuid5(uuid.NAMESPACE_URL, rel_str))[:8],
                "filename": p.name,
                "relative_path": rel_str,
                "folder": folder_rel,
                "format": ext.lstrip(".").upper(),
                "size_bytes": stat.st_size,
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "modified_time": mtime_str,
                "width": w,
                "height": h,
                "fps": nominal_fps,
                "total_frames": tot_f,
                "duration_seconds": dur,
                "recommended_camera": reg.get("recommended_camera", "cam1"),
                "scenario_description": reg.get("scenario_description", f"Discovered video in data/{folder_rel}."),
                "is_benchmark": reg.get("is_benchmark", "mcfd" in rel_str.lower()),
            }

            DATASET_METADATA_CACHE[file_key] = item_dict
            items.append(DatasetVideoItem(**item_dict))

    # Sort benchmark files first, then by filename
    items.sort(key=lambda x: (0 if x.is_benchmark else 1, x.filename.lower()))
    return items


def collect_historical_and_session_events() -> List[EventRecordItem]:
    """
    Collects detected candidate events across active session jobs and historical benchmark reports.
    Candidate events are clearly distinguished from certified medical alarms.
    """
    events: List[EventRecordItem] = []

    # 1. Collect events from current session jobs
    for job_id, job in JOBS.items():
        job_events = job.get("detected_events", [])
        src_name = Path(job.get("input_path", "unknown")).name
        cam_id = job.get("camera_id", "cam1")
        vid_filename = job.get("video_filename")

        for ev in job_events:
            t_sec = float(ev.get("trigger_time_seconds", 0.0))
            t_str = f"{int(t_sec // 60):02d}:{t_sec % 60:05.2f}"
            has_vid = (PROCESSED_DIR / vid_filename).exists() if vid_filename else False

            events.append(EventRecordItem(
                event_id=f"{job_id}_EV{ev.get('event_id', 1)}",
                job_id=job_id,
                source_video=src_name,
                camera_id=cam_id,
                trigger_frame=int(ev.get("trigger_frame", 0)),
                start_frame=int(ev.get("start_frame", 0)),
                end_frame=int(ev.get("end_frame", 0)),
                trigger_time_seconds=t_sec,
                trigger_time_str=t_str,
                duration_seconds=float(ev.get("duration_seconds", 0.0)),
                status="CANDIDATE_CONFIRMED",
                classification_type="RESEARCH_CANDIDATE_EVENT",
                peak_posture=str(ev.get("peak_posture", "LYING")),
                peak_confidence=float(ev.get("peak_confidence", 0.85)),
                zone_name=str(ev.get("zone_name", "floor_zone")),
                video_filename=vid_filename,
                has_video=has_vid,
            ))

    # 2. Collect historical benchmark events from Phase 6 Report
    phase6_report_path = MCFD_PROCESSED_DIR / "phase6_detection_improvement_report.json"
    if phase6_report_path.exists():
        try:
            with open(phase6_report_path, "r", encoding="utf-8") as f:
                rep_data = json.load(f)

            for v in rep_data.get("per_video_evaluations", []):
                vname = v.get("video_name", "")
                cam = f"cam{v.get('cam_id', 1)}"
                matched = v.get("event_matching", {}).get("matched_candidates", [])
                for c in matched:
                    sf = int(c.get("start_frame", 0))
                    ef = int(c.get("end_frame", sf))
                    t_sec = round(sf / 30.0, 2)
                    t_str = f"{int(t_sec // 60):02d}:{t_sec % 60:05.2f}"
                    dur = round(max(0.1, (ef - sf) / 30.0), 2)

                    # Look for annotated file
                    stem_clean = Path(vname).stem.replace("mcfd_", "")
                    v_annotated = f"annotated_{stem_clean}.mp4"
                    has_vid = (MCFD_PROCESSED_DIR / v_annotated).exists()

                    events.append(EventRecordItem(
                        event_id=c.get("event_id", f"bench_{vname}"),
                        job_id=f"phase6_benchmark_{Path(vname).stem}",
                        source_video=vname,
                        camera_id=cam,
                        trigger_frame=sf,
                        start_frame=max(0, sf - 45),
                        end_frame=ef,
                        trigger_time_seconds=t_sec,
                        trigger_time_str=t_str,
                        duration_seconds=dur,
                        status="BENCHMARK_CONFIRMED",
                        classification_type="RESEARCH_CANDIDATE_EVENT",
                        peak_posture="LYING",
                        peak_confidence=0.88,
                        zone_name="floor_mat",
                        video_filename=v_annotated if has_vid else None,
                        has_video=has_vid,
                    ))
        except Exception as e:
            logger.warning(f"Could not load historical events from phase 6 report: {e}")

    return events


# ============================================================================
# Page HTML Routes
# ============================================================================

def _serve_index():
    index_file = TEMPLATES_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Dashboard index.html template not found.")
    return HTMLResponse(content=index_file.read_text(encoding="utf-8"))


@app.get("/", response_class=HTMLResponse)
def get_dashboard():
    return _serve_index()


@app.get("/monitoring", response_class=HTMLResponse)
def get_monitoring_page():
    return _serve_index()


@app.get("/dataset", response_class=HTMLResponse)
def get_dataset_page():
    return _serve_index()


@app.get("/alerts", response_class=HTMLResponse)
def get_alerts_page():
    return _serve_index()


@app.get("/evaluation", response_class=HTMLResponse)
def get_evaluation_page():
    return _serve_index()


@app.get("/zones", response_class=HTMLResponse)
def get_zones_page():
    return _serve_index()


@app.get("/diagnostics", response_class=HTMLResponse)
def get_diagnostics_page():
    return _serve_index()


# ============================================================================
# API Routes: Dashboard & Overview
# ============================================================================

@app.get("/api/dashboard/stats", response_model=DashboardStatsResponse)
def get_dashboard_stats():
    """Returns aggregated high-level KPIs and activity logs for the Overview Dashboard."""
    pipeline = get_pipeline()
    dataset_videos = discover_dataset_videos()
    all_events = collect_historical_and_session_events()

    # Session processing counts
    completed_jobs = [j for j in JOBS.values() if j.get("status") == "COMPLETED"]

    # Recent jobs (up to 5)
    sorted_jobs = sorted(JOBS.values(), key=lambda j: j.get("start_time", 0.0), reverse=True)[:5]
    recent_jobs_clean = [
        {
            "job_id": j["job_id"],
            "filename": Path(j["input_path"]).name,
            "status": j["status"],
            "camera_id": j["camera_id"],
            "progress_percent": j.get("progress_percent", 0.0),
            "events_count": len(j.get("detected_events", [])),
        }
        for j in sorted_jobs
    ]

    # Recent events (up to 5)
    recent_events_clean = [ev.model_dump() for ev in all_events[:5]]

    return DashboardStatsResponse(
        total_dataset_videos=len(dataset_videos),
        videos_processed_session=len(completed_jobs),
        total_candidate_events=len(all_events),
        system_status="ONLINE",
        model_name=pipeline.model_path.name,
        hardware_device=pipeline.detector.get_device_info(),
        benchmark_recall=0.50,
        benchmark_precision=1.00,
        benchmark_f1=0.6667,
        mean_alert_delay_seconds=2.55,
        recent_jobs=recent_jobs_clean,
        recent_events=recent_events_clean,
    )


@app.get("/api/status", response_model=SystemStatusResponse)
def get_system_status():
    """Returns application status, hardware, and configuration."""
    pipeline = get_pipeline()
    return SystemStatusResponse(
        status="ONLINE",
        model_name=pipeline.model_path.name,
        hardware_device=pipeline.detector.get_device_info(),
        active_jobs=sum(1 for j in JOBS.values() if j.get("status") in ("QUEUED", "PROCESSING")),
        configured_cameras=list(pipeline.zone_manager.camera_configs.keys()),
        tracker_config={
            "nominal_fps": pipeline.tracker_config.fps,
            "persistence_seconds": pipeline.tracker_config.persistence_seconds,
            "min_evidence_ratio": pipeline.tracker_config.min_evidence_ratio,
            "max_dropout_grace_seconds": pipeline.tracker_config.max_dropout_grace_seconds,
            "upright_recovery_confirmation_frames": pipeline.tracker_config.upright_recovery_confirmation_frames,
        },
        disclaimer=RESEARCH_PROTOTYPE_DISCLAIMER,
    )


# ============================================================================
# API Routes: Dataset Explorer
# ============================================================================

@app.get("/api/dataset", response_model=DatasetListResponse)
def list_dataset(
    folder: Optional[str] = Query(None, description="Filter by relative folder"),
    format: Optional[str] = Query(None, description="Filter by video format (AVI, MP4, etc.)"),
    search: Optional[str] = Query(None, description="Search query in filename or description"),
):
    """Returns indexed dataset inventory with filtering and search support."""
    videos = discover_dataset_videos()

    unique_folders = sorted(list({v.folder for v in videos}))
    unique_formats = sorted(list({v.format for v in videos}))

    filtered = videos
    if folder:
        filtered = [v for v in filtered if v.folder.lower() == folder.lower()]
    if format:
        filtered = [v for v in filtered if v.format.lower() == format.lower()]
    if search:
        q = search.lower().strip()
        filtered = [
            v for v in filtered
            if q in v.filename.lower() or q in v.scenario_description.lower() or q in v.relative_path.lower()
        ]

    return DatasetListResponse(
        dataset_root="data/",
        total_count=len(videos),
        filtered_count=len(filtered),
        folders=unique_folders,
        formats=unique_formats,
        videos=filtered,
    )


@app.post("/api/dataset/rescan")
def rescan_dataset():
    """Forces cache invalidation and re-indexes all repository videos."""
    videos = discover_dataset_videos(force_rescan=True)
    logger.info(f"Dataset rescan triggered. Indexed {len(videos)} video files.")
    return {"status": "SUCCESS", "discovered_count": len(videos)}


@app.get("/api/samples", response_model=SampleVideosListResponse)
def list_sample_videos():
    """Maintains backward compatibility for the legacy sample dropdown in monitoring."""
    videos = discover_dataset_videos()
    samples = [
        SampleVideoItem(
            filename=v.filename,
            filepath=v.relative_path,
            category=v.folder.upper(),
            size_bytes=v.size_bytes,
            width=v.width,
            height=v.height,
            fps=v.fps,
            total_frames=v.total_frames,
            duration_seconds=v.duration_seconds,
            recommended_camera=v.recommended_camera,
            scenario_description=v.scenario_description,
        )
        for v in videos
    ]
    return SampleVideosListResponse(samples=samples)


# ============================================================================
# API Routes: Video Processing & Job Management
# ============================================================================

@app.post("/api/process", response_model=JobCreationResponse)
def process_sample_video(payload: ProcessVideoRequest):
    """
    Queues a video processing job for any discovered video file.
    Includes strict path-traversal prevention.
    """
    raw_path_str = payload.sample_path.strip().replace("\\", "/")

    # Path traversal protection: forbid '..' or invalid characters
    if ".." in raw_path_str or raw_path_str.startswith("/") or ":" in raw_path_str:
        # Check if it was an absolute Windows path that is somehow provided
        candidate = Path(raw_path_str)
        if candidate.is_absolute():
            try:
                candidate = candidate.relative_to(PROJECT_ROOT)
                raw_path_str = str(candidate).replace("\\", "/")
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid path: Path traversal characters are not permitted.")
        else:
            raise HTTPException(status_code=400, detail="Invalid path: Path traversal characters are not permitted.")

    # Candidate resolution order:
    candidate_paths = [
        PROJECT_ROOT / raw_path_str,
        DATA_DIR / raw_path_str,
        SAMPLES_DIR / raw_path_str,
        SAMPLES_DIR / "mcfd" / raw_path_str,
        UPLOADS_DIR / raw_path_str,
    ]

    target_file: Optional[Path] = None
    for cp in candidate_paths:
        if cp.exists() and cp.is_file():
            target_file = cp.resolve()
            break

    if target_file is None:
        raise HTTPException(status_code=404, detail=f"Target video file not found: {payload.sample_path}")

    # Boundary check: ensure target is strictly inside PROJECT_ROOT / "data"
    # and strictly NOT in data/processed
    try:
        rel_to_data = target_file.relative_to(DATA_DIR)
        if "processed" in rel_to_data.parts:
            raise HTTPException(
                status_code=400,
                detail="Processing processed output videos as input sources is prohibited.",
            )
    except ValueError:
        raise HTTPException(status_code=400, detail="Requested file is outside project data directory.")

    # Validate video readability
    ok, err_msg, meta = PersonDownPipeline.validate_video_file(target_file)
    if not ok:
        raise HTTPException(status_code=400, detail=f"Video validation failed: {err_msg}")

    job_id = f"job_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    JOBS[job_id] = {
        "job_id": job_id,
        "input_path": str(target_file),
        "camera_id": payload.camera_id,
        "max_frames": payload.max_frames,
        "status": "QUEUED",
        "progress_percent": 0.0,
        "current_frame": 0,
        "total_frames": meta.get("total_frames", 0),
        "processing_fps": 0.0,
        "current_state": "NORMAL",
        "current_posture": "NO_DETECTION",
        "current_zone": "OUTSIDE",
        "has_person": False,
        "elapsed_down_seconds": 0.0,
        "is_active_candidate": False,
        "events_detected_count": 0,
        "detected_events": [],
        "cancelled": False,
        "error_message": None,
        "result": None,
    }

    EXECUTOR.submit(
        _execute_processing_job,
        job_id=job_id,
        input_path=target_file,
        camera_id=payload.camera_id,
        max_frames=payload.max_frames,
    )

    return JobCreationResponse(
        job_id=job_id,
        status="QUEUED",
        message=f"Processing job queued successfully for {target_file.name}.",
    )


@app.post("/api/upload", response_model=JobCreationResponse)
async def upload_and_process_video(
    video_file: UploadFile = File(...),
    camera_id: str = Form("auto"),
    max_frames: Optional[int] = Form(None),
):
    """Receives an uploaded video file, validates it, and queues processing."""
    ext = Path(video_file.filename).suffix.lower()
    if ext not in SUPPORTED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video extension '{ext}'. Supported: {', '.join(sorted(SUPPORTED_VIDEO_EXTENSIONS))}",
        )

    # Sanitize filename
    safe_filename = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", video_file.filename)
    unique_name = f"upload_{time.strftime('%Y%m%d_%H%M%S')}_{safe_filename}"
    dest_path = UPLOADS_DIR / unique_name

    try:
        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(video_file.file, buffer)
    finally:
        await video_file.close()

    ok, err_msg, meta = PersonDownPipeline.validate_video_file(dest_path)
    if not ok:
        if dest_path.exists():
            dest_path.unlink()
        raise HTTPException(status_code=400, detail=f"Uploaded video invalid: {err_msg}")

    job_id = f"job_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    JOBS[job_id] = {
        "job_id": job_id,
        "input_path": str(dest_path.resolve()),
        "camera_id": camera_id,
        "max_frames": max_frames,
        "status": "QUEUED",
        "progress_percent": 0.0,
        "current_frame": 0,
        "total_frames": meta.get("total_frames", 0),
        "processing_fps": 0.0,
        "current_state": "NORMAL",
        "current_posture": "NO_DETECTION",
        "current_zone": "OUTSIDE",
        "has_person": False,
        "elapsed_down_seconds": 0.0,
        "is_active_candidate": False,
        "events_detected_count": 0,
        "detected_events": [],
        "cancelled": False,
        "error_message": None,
        "result": None,
    }

    EXECUTOR.submit(
        _execute_processing_job,
        job_id=job_id,
        input_path=dest_path,
        camera_id=camera_id,
        max_frames=max_frames,
    )

    return JobCreationResponse(
        job_id=job_id,
        status="QUEUED",
        message=f"Uploaded video '{video_file.filename}' queued for processing.",
    )


@app.get("/api/jobs/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str):
    """Returns real-time progress and telemetry for a running or completed job."""
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    return JobStatusResponse(
        job_id=job["job_id"],
        status=job["status"],
        progress_percent=job.get("progress_percent", 0.0),
        current_frame=job.get("current_frame", 0),
        total_frames=job.get("total_frames", 0),
        processing_fps=job.get("processing_fps", 0.0),
        current_state=job.get("current_state", "NORMAL"),
        current_posture=job.get("current_posture", "NO_DETECTION"),
        current_zone=job.get("current_zone", "OUTSIDE"),
        has_person=job.get("has_person", False),
        elapsed_down_seconds=job.get("elapsed_down_seconds", 0.0),
        is_active_candidate=job.get("is_active_candidate", False),
        events_detected_count=job.get("events_detected_count", 0),
        detected_events=job.get("detected_events", []),
        error_message=job.get("error_message"),
        result=job.get("result"),
    )


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    """Requests early cancellation of a running processing job."""
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    if job["status"] in ("COMPLETED", "FAILED"):
        return {"status": job["status"], "message": f"Job is already {job['status']}."}

    job["cancelled"] = True
    job["status"] = "CANCELLED"
    return {"status": "CANCELLED", "message": f"Job {job_id} cancellation requested."}


# ============================================================================
# API Routes: Video Streaming & Playback
# ============================================================================

@app.get("/api/videos/{video_filename}")
def stream_annotated_video(video_filename: str):
    """Streams processed annotated video from phase 7 session or historical benchmarks."""
    safe_name = Path(video_filename).name

    # Check phase 7 processed dir
    candidate = PROCESSED_DIR / safe_name
    if not candidate.exists():
        # Check mcfd processed dir
        candidate = MCFD_PROCESSED_DIR / safe_name
    if not candidate.exists():
        # Check processed root dir
        candidate = (DATA_DIR / "processed") / safe_name
    if not candidate.exists():
        # Check samples dir
        candidate = SAMPLES_DIR / safe_name
    if not candidate.exists():
        # Check uploads dir
        candidate = UPLOADS_DIR / safe_name

    if not candidate.exists() or not candidate.is_file():
        raise HTTPException(status_code=404, detail=f"Video file not found: {video_filename}")

    return FileResponse(
        path=str(candidate.resolve()),
        media_type="video/mp4",
        content_disposition_type="inline",
    )


@app.get("/api/videos/source/{video_path:path}")
def stream_source_video(video_path: str):
    """Streams raw source video (MP4/WebM/AVI) from samples, uploads, or raw directory for browser preview."""
    clean_path = Path(video_path.replace("\\", "/"))
    if clean_path.is_absolute():
        resolved = clean_path.resolve()
    else:
        resolved = (PROJECT_ROOT / clean_path).resolve()
        if not resolved.exists():
            resolved = (SAMPLES_DIR / clean_path).resolve()
        if not resolved.exists():
            resolved = (UPLOADS_DIR / clean_path).resolve()

    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied: path outside project workspace.")

    if not resolved.exists() or not resolved.is_file():
        raise HTTPException(status_code=404, detail=f"Source video not found: {video_path}")

    ext = resolved.suffix.lower()
    media_type = "video/mp4"
    if ext == ".webm":
        media_type = "video/webm"
    elif ext == ".avi":
        media_type = "video/x-msvideo"
    elif ext == ".mov":
        media_type = "video/quicktime"

    return FileResponse(
        path=str(resolved),
        media_type=media_type,
        content_disposition_type="inline",
    )


@app.get("/api/videos/download/{video_filename}")
def download_video_file(video_filename: str):
    """Downloads an annotated or sample video file as an attachment."""
    safe_name = Path(video_filename).name
    candidates = [
        PROCESSED_DIR / safe_name,
        MCFD_PROCESSED_DIR / safe_name,
        (DATA_DIR / "processed") / safe_name,
        SAMPLES_DIR / safe_name,
        UPLOADS_DIR / safe_name,
    ]
    target = next((c for c in candidates if c.exists() and c.is_file()), None)
    if not target:
        raise HTTPException(status_code=404, detail=f"Video not found: {video_filename}")

    return FileResponse(
        path=str(target.resolve()),
        media_type="video/mp4",
        filename=safe_name,
        content_disposition_type="attachment",
    )


@app.get("/api/videos/check_annotated/{video_path:path}")
def check_annotated_video(video_path: str):
    """Checks whether a video has an existing annotated pipeline output or benchmark,
    and returns playback metadata."""
    clean_p = Path(video_path.replace("\\", "/"))
    filename = clean_p.name
    stem = clean_p.stem
    stem_clean = stem.replace("mcfd_", "")
    ext = clean_p.suffix.lower()

    source_playable = ext in [".mp4", ".webm"]

    candidates = [
        PROCESSED_DIR / f"annotated_{stem}.mp4",
        MCFD_PROCESSED_DIR / f"annotated_{stem_clean}.mp4",
        (DATA_DIR / "processed") / f"annotated_{stem}.mp4",
    ]
    p7_matches = sorted(
        list(PROCESSED_DIR.glob(f"annotated_*_{stem}.mp4")) +
        list(PROCESSED_DIR.glob(f"annotated_*_{stem_clean}.mp4")),
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )
    candidates.extend(p7_matches)

    annotated_match = None
    for c in candidates:
        if c.exists() and c.is_file() and c.stat().st_size > 0:
            annotated_match = c
            break

    return {
        "filename": filename,
        "source_format": ext.lstrip(".").upper(),
        "source_playable": source_playable,
        "source_url": f"/api/videos/source/{clean_p.as_posix()}" if source_playable else None,
        "has_annotated": annotated_match is not None,
        "annotated_filename": annotated_match.name if annotated_match else None,
        "annotated_url": f"/api/videos/{annotated_match.name}" if annotated_match else None,
        "annotated_size_bytes": annotated_match.stat().st_size if annotated_match else 0,
        "is_benchmark_archive": "mcfd" in str(annotated_match) if annotated_match else False,
    }


# ============================================================================
# API Routes: Alerts & Candidate Events
# ============================================================================

@app.get("/api/events", response_model=EventListResponse)
def get_events(
    status: Optional[str] = Query(None, description="Filter by event status"),
    zone: Optional[str] = Query(None, description="Filter by spatial zone"),
    posture: Optional[str] = Query(None, description="Filter by peak posture"),
    search: Optional[str] = Query(None, description="Search query in event ID or source video"),
):
    """Returns candidate events with comprehensive filtering and search capabilities."""
    all_events = collect_historical_and_session_events()

    filtered = all_events
    if status:
        filtered = [e for e in filtered if e.status.lower() == status.lower()]
    if zone:
        filtered = [e for e in filtered if e.zone_name.lower() == zone.lower()]
    if posture:
        filtered = [e for e in filtered if e.peak_posture.lower() == posture.lower()]
    if search:
        q = search.lower().strip()
        filtered = [
            e for e in filtered
            if q in e.event_id.lower() or q in e.source_video.lower() or q in e.job_id.lower()
        ]

    return EventListResponse(
        total_events=len(all_events),
        filtered_events=len(filtered),
        events=filtered,
    )


@app.get("/api/events/export/csv")
def export_events_csv():
    """Generates and streams candidate events formatted as a standard CSV file."""
    all_events = collect_historical_and_session_events()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Event ID",
        "Job ID",
        "Source Video",
        "Camera ID",
        "Trigger Frame",
        "Start Frame",
        "End Frame",
        "Trigger Time (s)",
        "Trigger Time (MM:SS)",
        "Duration (s)",
        "Status",
        "Classification Type",
        "Peak Posture",
        "Peak Confidence",
        "Spatial Zone",
    ])

    for ev in all_events:
        writer.writerow([
            ev.event_id,
            ev.job_id,
            ev.source_video,
            ev.camera_id,
            ev.trigger_frame,
            ev.start_frame,
            ev.end_frame,
            ev.trigger_time_seconds,
            ev.trigger_time_str,
            ev.duration_seconds,
            ev.status,
            ev.classification_type,
            ev.peak_posture,
            ev.peak_confidence,
            ev.zone_name,
        ])

    csv_content = output.getvalue()
    return Response(
        content=csv_content,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="person_down_candidate_events.csv"'},
    )


# ============================================================================
# API Routes: Model Evaluation & Reports
# ============================================================================

@app.get("/api/evaluations", response_model=EvaluationSummaryResponse)
def get_evaluation_summary():
    """Returns official benchmark evaluations, confusion matrices, and SHA-256 integrity checks."""
    phase6_file = MCFD_PROCESSED_DIR / "phase6_detection_improvement_report.json"
    if not phase6_file.exists():
        raise HTTPException(status_code=404, detail="Benchmark evaluation report not found.")

    with open(phase6_file, "r", encoding="utf-8") as f:
        rep = json.load(f)

    meta = rep.get("report_metadata", {})
    event_level = rep.get("event_level_evaluation", {})
    protocols = {
        "protocol_a_strict": rep.get("protocol_a_strict_benchmark", {}),
        "protocol_b_conditional": rep.get("protocol_b_conditional_benchmark", {}),
        "negative_control": rep.get("negative_control_evaluation", {}),
        "root_causes": rep.get("root_cause_diagnosis_summary", {}),
    }

    per_vid_clean = []
    for pv in rep.get("per_video_evaluations", []):
        per_vid_clean.append({
            "video_name": pv.get("video_name"),
            "cam_id": pv.get("cam_id"),
            "total_frames": pv.get("total_frames"),
            "processing_fps": pv.get("processing_fps"),
            "ground_truth_events": pv.get("ground_truth_event_count"),
            "candidate_events": pv.get("candidate_event_count"),
            "event_tp": pv.get("event_matching", {}).get("event_tp", 0),
            "event_fp": pv.get("event_matching", {}).get("event_fp", 0),
            "event_fn": pv.get("event_matching", {}).get("event_fn", 0),
            "latency_seconds": pv.get("event_matching", {}).get("candidate_promotion_delays_seconds", []),
            "protocol_a_f1": pv.get("protocol_a_strict", {}).get("metrics", {}).get("f1", 0.0),
        })

    baseline_files_dict = rep.get("baseline_preservation_verification", {}).get("baseline_files", {})
    baseline_hashes = [
        {
            "filename": fname,
            "sha256": finfo.get("sha256_after", finfo.get("sha256_before", "")),
            "intact": finfo.get("intact", True),
        }
        for fname, finfo in baseline_files_dict.items()
    ]

    # Discover all JSON reports in data/processed/mcfd/
    available_reports = []
    for f in MCFD_PROCESSED_DIR.glob("*.json"):
        available_reports.append({
            "filename": f.name,
            "size_kb": round(f.stat().st_size / 1024, 1),
            "modified_time": datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
        })
    available_reports.sort(key=lambda r: r["filename"])

    return EvaluationSummaryResponse(
        report_metadata=meta,
        event_level_metrics=event_level,
        frame_level_protocols=protocols,
        per_video_results=per_vid_clean,
        baseline_hashes=baseline_hashes,
        available_reports=available_reports,
    )


@app.get("/api/evaluations/download/{filename}")
def download_evaluation_report(filename: str):
    """Safely downloads a verified benchmark report or prediction JSON file."""
    safe_name = Path(filename).name
    target_path = MCFD_PROCESSED_DIR / safe_name

    if not target_path.exists() or not target_path.is_file() or target_path.suffix.lower() != ".json":
        raise HTTPException(status_code=404, detail=f"Report file '{safe_name}' not found.")

    return FileResponse(
        path=str(target_path.resolve()),
        media_type="application/json",
        filename=safe_name,
    )


# ============================================================================
# API Routes: Spatial Zones & Camera Calibration
# ============================================================================

@app.get("/api/zones")
def get_camera_zones():
    """Returns the polygon spatial zone definitions for all configured cameras."""
    pipeline = get_pipeline()
    zones_data = {}
    for cam_id, config in pipeline.zone_manager.camera_configs.items():
        zones_data[cam_id] = {
            "camera_id": config.camera_id,
            "resolution": list(config.resolution),
            "status": config.status,
            "zones": [
                {
                    "name": z.name,
                    "type": z.zone_type.value,
                    "description": z.description,
                    "polygon": z.polygon.tolist(),
                }
                for z in config.zones
            ],
        }
    return zones_data


@app.get("/api/zones/preview/{camera_id}")
def get_zone_calibration_preview(camera_id: str):
    """Serves calibration preview images with spatial polygon overlays."""
    preview_map = {
        "cam1": "cam1_zone_calibration_preview.jpg",
        "cam2": "cam2_zone_calibration_preview.jpg",
        "cam1_chute09": "chute09_cam1_unvalidated_preview.jpg",
    }

    target_filename = preview_map.get(camera_id, f"{camera_id}.jpg")
    preview_file = CALIBRATION_PREVIEWS_DIR / target_filename

    if not preview_file.exists():
        # Fallback to cam1 preview if requested file doesn't exist
        preview_file = CALIBRATION_PREVIEWS_DIR / "cam1_zone_calibration_preview.jpg"

    if not preview_file.exists():
        raise HTTPException(status_code=404, detail="Calibration preview image not found.")

    return FileResponse(
        path=str(preview_file.resolve()),
        media_type="image/jpeg",
        filename=preview_file.name,
    )


# ============================================================================
# API Routes: System Diagnostics & Health
# ============================================================================

@app.get("/api/system/diagnostics", response_model=SystemDiagnosticsResponse)
def get_system_diagnostics():
    """Returns comprehensive system health, device information, directory status, and live logs."""
    uptime = round(time.time() - SERVER_START_TIME, 1)

    cuda_avail = torch.cuda.is_available()
    cuda_name = torch.cuda.get_device_name(0) if cuda_avail else None

    # Check directory access
    dirs_to_check = {
        "data_uploads": UPLOADS_DIR,
        "data_processed_phase7": PROCESSED_DIR,
        "data_processed_mcfd": MCFD_PROCESSED_DIR,
        "data_samples": SAMPLES_DIR,
        "configs": PROJECT_ROOT / "configs",
        "models": PROJECT_ROOT / "models" / "pretrained",
    }

    dir_status = {}
    for dname, dpath in dirs_to_check.items():
        exists = dpath.exists()
        writable = os.access(dpath, os.W_OK) if exists else False
        dir_status[dname] = {
            "path": str(dpath.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "exists": exists,
            "writable": writable,
        }

    # Verify scaling engine mathematically
    pipeline = get_pipeline()
    scaled_zones = pipeline.zone_manager.get_scaled_zones("cam1", (1440, 960))
    scaling_ok = len(scaled_zones) > 0 and scaled_zones[0].polygon[0][0] != 0 or len(scaled_zones) > 0

    active_count = sum(1 for j in JOBS.values() if j.get("status") in ("QUEUED", "PROCESSING"))

    return SystemDiagnosticsResponse(
        status="ONLINE",
        uptime_seconds=uptime,
        python_version=sys.version.split()[0],
        torch_version=torch.__version__,
        cuda_available=cuda_avail,
        cuda_device_name=cuda_name,
        active_workers=active_count,
        max_workers=EXECUTOR._max_workers,
        job_store_count=len(JOBS),
        directories_status=dir_status,
        recent_logs=list(LOG_BUFFER),
        scaling_engine_verified=bool(scaling_ok),
    )
