"""
Pydantic API Request and Response Models for Person Down Detection Application.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class SystemStatusResponse(BaseModel):
    """System health and hardware status."""
    status: str = "ONLINE"
    model_name: str
    hardware_device: str
    active_jobs: int
    configured_cameras: List[str]
    tracker_config: Dict[str, Any]
    disclaimer: str


class SampleVideoItem(BaseModel):
    """Metadata for a server sample video."""
    filename: str
    filepath: str
    category: str
    size_bytes: int
    width: int
    height: int
    fps: float
    total_frames: int
    duration_seconds: float
    recommended_camera: str
    scenario_description: str


class SampleVideosListResponse(BaseModel):
    """List of server sample videos available for testing."""
    samples: List[SampleVideoItem]


class ProcessVideoRequest(BaseModel):
    """Request payload to initiate video processing from a server sample or dataset path."""
    sample_path: str
    camera_id: str = "auto"
    max_frames: Optional[int] = None


class JobCreationResponse(BaseModel):
    """Response returned upon successfully queuing a video processing job."""
    job_id: str
    status: str = "QUEUED"
    message: str


class DetectedEventItem(BaseModel):
    """Item representing a detected candidate event."""
    event_id: int
    start_frame: int
    trigger_frame: int
    end_frame: int
    start_time_seconds: float
    trigger_time_seconds: float
    end_time_seconds: float
    duration_seconds: float
    duration_frames: int
    status: str
    zone_name: str
    peak_posture: str
    peak_confidence: float


class JobStatusResponse(BaseModel):
    """Current state and progress telemetry of a processing job."""
    job_id: str
    status: str  # "QUEUED", "PROCESSING", "COMPLETED", "FAILED", "CANCELLED"
    progress_percent: float = 0.0
    current_frame: int = 0
    total_frames: int = 0
    processing_fps: float = 0.0
    current_state: str = "NORMAL"
    current_posture: str = "NO_DETECTION"
    current_zone: str = "OUTSIDE"
    has_person: bool = False
    elapsed_down_seconds: float = 0.0
    is_active_candidate: bool = False
    events_detected_count: int = 0
    detected_events: List[DetectedEventItem] = Field(default_factory=list)
    error_message: Optional[str] = None
    result: Optional[Dict[str, Any]] = None


# ============================================================================
# Multi-Page Platform Extensions
# ============================================================================

class DashboardStatsResponse(BaseModel):
    """Aggregated statistics for Overview Dashboard."""
    total_dataset_videos: int
    videos_processed_session: int
    total_candidate_events: int
    system_status: str
    model_name: str
    hardware_device: str
    benchmark_recall: float
    benchmark_precision: float
    benchmark_f1: float
    mean_alert_delay_seconds: float
    recent_jobs: List[Dict[str, Any]] = Field(default_factory=list)
    recent_events: List[Dict[str, Any]] = Field(default_factory=list)


class DatasetVideoItem(BaseModel):
    """Metadata item representing a discovered dataset video."""
    id: str
    filename: str
    relative_path: str
    folder: str
    format: str
    size_bytes: int
    size_mb: float
    modified_time: str
    width: int
    height: int
    fps: float
    total_frames: int
    duration_seconds: float
    recommended_camera: str
    scenario_description: str
    is_benchmark: bool


class DatasetListResponse(BaseModel):
    """Complete dataset inventory response."""
    dataset_root: str
    total_count: int
    filtered_count: int
    folders: List[str]
    formats: List[str]
    videos: List[DatasetVideoItem]


class EventRecordItem(BaseModel):
    """Searchable event history item."""
    event_id: str
    job_id: str
    source_video: str
    camera_id: str
    trigger_frame: int
    start_frame: int
    end_frame: int
    trigger_time_seconds: float
    trigger_time_str: str
    duration_seconds: float
    status: str
    classification_type: str
    peak_posture: str
    peak_confidence: float
    zone_name: str
    video_filename: Optional[str] = None
    has_video: bool = False


class EventListResponse(BaseModel):
    """List of candidate events across all sessions and benchmarks."""
    total_events: int
    filtered_events: int
    events: List[EventRecordItem]


class EvaluationSummaryResponse(BaseModel):
    """Structured benchmark evaluations and comparison data."""
    report_metadata: Dict[str, Any]
    event_level_metrics: Dict[str, Any]
    frame_level_protocols: Dict[str, Any]
    per_video_results: List[Dict[str, Any]]
    baseline_hashes: List[Dict[str, Any]]
    available_reports: List[Dict[str, Any]]


class SystemDiagnosticsResponse(BaseModel):
    """Comprehensive system diagnostic report."""
    status: str
    uptime_seconds: float
    python_version: str
    torch_version: str
    cuda_available: bool
    cuda_device_name: Optional[str]
    active_workers: int
    max_workers: int
    job_store_count: int
    directories_status: Dict[str, Dict[str, Any]]
    recent_logs: List[str] = Field(default_factory=list)
    scaling_engine_verified: bool
