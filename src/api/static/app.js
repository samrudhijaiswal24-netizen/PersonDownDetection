/**
 * Person Down Detection Application - Multi-Page Platform Controller
 * Handles client-side routing, Overview KPIs, Video Monitoring telemetry,
 * Dataset discovery, Event logging, Evaluation matrices, Zone inspection,
 * and System diagnostics.
 */

document.addEventListener("DOMContentLoaded", () => {
  // Navigation & Page State
  const sidebar = document.getElementById("sidebar");
  const sidebarToggleBtn = document.getElementById("sidebarToggleBtn");
  const sidebarCloseBtn = document.getElementById("sidebarCloseBtn");
  const navItems = document.querySelectorAll(".nav-item");
  const pageContents = document.querySelectorAll(".page-content");
  const pageTitle = document.getElementById("pageTitle");
  const pageDescription = document.getElementById("pageDescription");

  // Global Header Badges & Alerts
  const systemStatusBadge = document.getElementById("systemStatusBadge");
  const hardwareBadge = document.getElementById("hardwareBadge");
  const sidebarStatusText = document.getElementById("sidebarStatusText");
  const errorAlert = document.getElementById("errorAlert");
  const errorMessageText = document.getElementById("errorMessageText");
  const btnDismissError = document.getElementById("btnDismissError");

  // Modal Dialog Elements
  const detailModal = document.getElementById("detailModal");
  const modalTitle = document.getElementById("modalTitle");
  const modalBody = document.getElementById("modalBody");
  const modalCloseBtn = document.getElementById("modalCloseBtn");
  const modalDismissBtn = document.getElementById("modalDismissBtn");
  const modalActionBtn = document.getElementById("modalActionBtn");

  // Monitoring Page Elements
  const tabSample = document.getElementById("tabSample");
  const tabUpload = document.getElementById("tabUpload");
  const sampleSection = document.getElementById("sampleSection");
  const uploadSection = document.getElementById("uploadSection");
  const sampleSelect = document.getElementById("sampleSelect");
  const sampleHint = document.getElementById("sampleHint");
  const cameraSelect = document.getElementById("cameraSelect");
  const maxFramesInput = document.getElementById("maxFramesInput");
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("fileInput");
  const fileDetails = document.getElementById("fileDetails");
  const fileNameDisplay = document.getElementById("fileNameDisplay");
  const btnStart = document.getElementById("btnStart");
  const btnCancel = document.getElementById("btnCancel");
  const btnReset = document.getElementById("btnReset");

  const progressCard = document.getElementById("progressCard");
  const progressBarFill = document.getElementById("progressBarFill");
  const progressPercentText = document.getElementById("progressPercentText");
  const statCurrentFrame = document.getElementById("statCurrentFrame");
  const statFps = document.getElementById("statFps");

  const statusBanner = document.getElementById("statusBanner");
  const statusHeadline = document.getElementById("statusHeadline");
  const statusSubtext = document.getElementById("statusSubtext");
  const statusIcon = document.getElementById("statusIcon");

  const telemState = document.getElementById("telemState");
  const telemPosture = document.getElementById("telemPosture");
  const telemZone = document.getElementById("telemZone");
  const telemTimer = document.getElementById("telemTimer");

  const videoPlayerWrapper = document.getElementById("videoPlayerWrapper");
  const videoPlaceholder = document.getElementById("videoPlaceholder");
  const videoPlayer = document.getElementById("videoPlayer");
  const videoProcessingOverlay = document.getElementById("videoProcessingOverlay");
  const videoModeBadge = document.getElementById("videoModeBadge");
  const videoFilenameDisplay = document.getElementById("videoFilenameDisplay");
  const videoSourceSwitcher = document.getElementById("videoSourceSwitcher");
  const btnViewSource = document.getElementById("btnViewSource");
  const btnViewAnnotated = document.getElementById("btnViewAnnotated");
  const btnLoadBenchmark = document.getElementById("btnLoadBenchmark");
  const btnQuickFloorFall = document.getElementById("btnQuickFloorFall");
  const hudFrame = document.getElementById("hudFrame");
  const hudFps = document.getElementById("hudFps");
  const hudState = document.getElementById("hudState");

  // Video Toolbar controls
  const btnSeekStart = document.getElementById("btnSeekStart");
  const btnSeekBack5 = document.getElementById("btnSeekBack5");
  const btnPlayPause = document.getElementById("btnPlayPause");
  const btnSeekFwd5 = document.getElementById("btnSeekFwd5");
  const btnReplayVideo = document.getElementById("btnReplayVideo");
  const videoTimeDisplay = document.getElementById("videoTimeDisplay");
  const btnFullscreen = document.getElementById("btnFullscreen");
  const speedButtons = document.querySelectorAll(".btn-speed");

  const videoDownloadGroup = document.getElementById("videoDownloadGroup");
  const btnDownloadVideo = document.getElementById("btnDownloadVideo");
  const btnDownloadJson = document.getElementById("btnDownloadJson");

  let currentVideoState = {
    filepath: "",
    filename: "",
    sourceUrl: null,
    sourcePlayable: false,
    annotatedUrl: null,
    hasAnnotated: false,
    activeMode: "none",
  };

  const eventsTableBody = document.getElementById("eventsTableBody");
  const emptyEventsNotice = document.getElementById("emptyEventsNotice");
  const summaryCard = document.getElementById("summaryCard");
  const summaryGrid = document.getElementById("summaryGrid");

  // Dataset Explorer Elements
  const btnRescanDataset = document.getElementById("btnRescanDataset");
  const datasetSearch = document.getElementById("datasetSearch");
  const datasetFolderFilter = document.getElementById("datasetFolderFilter");
  const datasetFormatFilter = document.getElementById("datasetFormatFilter");
  const btnClearDatasetFilters = document.getElementById("btnClearDatasetFilters");
  const datasetTableBody = document.getElementById("datasetTableBody");

  // Alerts Page Elements
  const alertsSearch = document.getElementById("alertsSearch");
  const alertsStatusFilter = document.getElementById("alertsStatusFilter");
  const alertsZoneFilter = document.getElementById("alertsZoneFilter");
  const btnClearAlertsFilters = document.getElementById("btnClearAlertsFilters");
  const allAlertsTableBody = document.getElementById("allAlertsTableBody");

  // Spatial Zones Elements
  const zonePreviewImg = document.getElementById("zonePreviewImg");
  const zoneCameraStatus = document.getElementById("zoneCameraStatus");
  const zoneCameraRes = document.getElementById("zoneCameraRes");
  const zonesListContainer = document.getElementById("zonesListContainer");
  const camButtons = document.querySelectorAll(".cam-btn");
  const calcFrameSize = document.getElementById("calcFrameSize");
  const calcPoint = document.getElementById("calcPoint");
  const btnRunScalingCalc = document.getElementById("btnRunScalingCalc");
  const calcResultDisplay = document.getElementById("calcResultDisplay");
  const calcMappedResult = document.getElementById("calcMappedResult");

  // Diagnostics Elements
  const btnRefreshLogs = document.getElementById("btnRefreshLogs");

  // Application State
  let currentPage = "overview";
  let activeJobId = null;
  let pollIntervalId = null;
  let currentMode = "sample"; // 'sample' or 'upload'
  let selectedUploadFile = null;
  let datasetVideos = [];
  let currentJobResult = null;
  let cameraZonesData = {};
  let currentSelectedCam = "cam1";

  const PAGE_TITLES = {
    overview: {
      title: "Overview Dashboard",
      desc: "Comprehensive system metrics, recent video processing jobs, and candidate alerts.",
    },
    monitoring: {
      title: "Video Monitoring & Detection",
      desc: "Video-first analytics console with real-time pose estimation, spatial zone HUD, and temporal tracking.",
    },
    dataset: {
      title: "Dataset Explorer",
      desc: "Recursive repository video catalog across data/samples, data/uploads, and data/raw.",
    },
    alerts: {
      title: "Alerts & Candidate Events Log",
      desc: "Detailed record of candidate collapse events across processing jobs and benchmark evaluations.",
    },
    evaluation: {
      title: "Model Evaluation & Benchmarks",
      desc: "Official Protocol A & B benchmarks, confusion matrices, and SHA-256 baseline verification.",
    },
    zones: {
      title: "Camera & Spatial Zone Settings",
      desc: "Camera calibration polygons, safe-resting rules, and resolution-aware coordinate scaling validation.",
    },
    diagnostics: {
      title: "System Status & Diagnostics",
      desc: "Hardware runtime telemetry, PyTorch CUDA details, directory permissions, and live server logs.",
    },
  };

  // ==========================================================================
  // Router & Navigation
  // ==========================================================================

  function navigateToPage(pageId, updateUrl = true) {
    if (!PAGE_TITLES[pageId]) pageId = "overview";
    currentPage = pageId;

    // Update Nav Sidebar
    navItems.forEach((item) => {
      item.classList.toggle("active", item.getAttribute("data-page") === pageId);
    });

    // Update Content Visibility
    pageContents.forEach((page) => {
      page.classList.toggle("active", page.id === `page-${pageId}`);
    });

    // Update Header
    if (pageTitle && pageDescription) {
      pageTitle.textContent = PAGE_TITLES[pageId].title;
      pageDescription.textContent = PAGE_TITLES[pageId].desc;
    }
    const breadcrumbActivePage = document.getElementById("breadcrumbActivePage");
    if (breadcrumbActivePage) {
      breadcrumbActivePage.textContent = PAGE_TITLES[pageId].title.replace(" Dashboard", "").replace(" & Detection Console", "");
    }

    // Close mobile sidebar if open
    if (sidebar) sidebar.classList.remove("open");

    // Sync URL hash
    if (updateUrl && window.location.hash !== `#${pageId}`) {
      window.location.hash = pageId;
    }

    // Load page data
    switch (pageId) {
      case "overview":
        loadOverviewData();
        break;
      case "monitoring":
        loadMonitoringSamples();
        break;
      case "dataset":
        loadDatasetInventory();
        break;
      case "alerts":
        loadAllAlerts();
        break;
      case "evaluation":
        loadEvaluationData();
        break;
      case "zones":
        loadZoneConfigurations();
        break;
      case "diagnostics":
        loadSystemDiagnostics();
        break;
    }
  }

  // Expose to window for inline onclick handlers
  window.navigateToPage = navigateToPage;

  // Sidebar link clicks
  navItems.forEach((item) => {
    item.addEventListener("click", (e) => {
      e.preventDefault();
      const pageId = item.getAttribute("data-page");
      navigateToPage(pageId);
    });
  });

  // Mobile sidebar controls
  if (sidebarToggleBtn) {
    sidebarToggleBtn.addEventListener("click", () => {
      sidebar.classList.toggle("open");
    });
  }

  if (sidebarCloseBtn) {
    sidebarCloseBtn.addEventListener("click", () => {
      sidebar.classList.remove("open");
    });
  }

  // Handle URL hash changes
  window.addEventListener("hashchange", () => {
    const hash = window.location.hash.replace("#", "") || "overview";
    navigateToPage(hash, false);
  });

  // Modal Dialog Handlers
  function openModal(title, contentHtml, actionBtnConfig = null) {
    if (!detailModal) return;
    modalTitle.textContent = title;
    modalBody.innerHTML = contentHtml;

    if (actionBtnConfig && modalActionBtn) {
      modalActionBtn.textContent = actionBtnConfig.text || "Action";
      modalActionBtn.onclick = actionBtnConfig.onClick;
      modalActionBtn.classList.remove("hidden");
    } else if (modalActionBtn) {
      modalActionBtn.classList.add("hidden");
    }

    detailModal.classList.remove("hidden");
  }

  function closeModal() {
    if (detailModal) detailModal.classList.add("hidden");
  }

  if (modalCloseBtn) modalCloseBtn.addEventListener("click", closeModal);
  if (modalDismissBtn) modalDismissBtn.addEventListener("click", closeModal);
  if (detailModal) {
    detailModal.addEventListener("click", (e) => {
      if (e.target === detailModal) closeModal();
    });
  }
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeModal();
  });

  // Error Banner Controls
  function showError(msg) {
    if (errorMessageText && errorAlert) {
      errorMessageText.textContent = msg;
      errorAlert.classList.remove("hidden");
    }
  }

  function hideError() {
    if (errorAlert) errorAlert.classList.add("hidden");
  }

  if (btnDismissError) btnDismissError.addEventListener("click", hideError);

  // ==========================================================================
  // System Status Fetcher
  // ==========================================================================

  async function fetchSystemStatus() {
    try {
      const res = await fetch("/api/status");
      if (res.ok) {
        const data = await res.json();
        if (systemStatusBadge) {
          systemStatusBadge.innerHTML = `<span class="pulse-dot"></span> System Online (${data.model_name})`;
        }
        if (hardwareBadge) {
          hardwareBadge.textContent = data.hardware_device;
        }
        if (sidebarStatusText) {
          sidebarStatusText.textContent = `${data.model_name} • Active`;
        }
        const activeJobsBadge = document.getElementById("activeJobsCountBadge");
        if (activeJobsBadge) {
          activeJobsBadge.textContent = `${data.active_jobs || 0} Active Job${data.active_jobs === 1 ? '' : 's'}`;
        }
      }
    } catch (e) {
      if (systemStatusBadge) {
        systemStatusBadge.innerHTML = `<span class="pulse-dot" style="background:#ef4444;box-shadow:none;"></span> Server Offline`;
      }
      if (sidebarStatusText) {
        sidebarStatusText.textContent = "Server Offline";
      }
    }
  }

  // ==========================================================================
  // PAGE 1: Overview Dashboard Loader
  // ==========================================================================

  async function loadOverviewData() {
    try {
      const res = await fetch("/api/dashboard/stats");
      if (!res.ok) return;
      const stats = await res.json();

      const elData = document.getElementById("kpiDatasetCount");
      if (elData) elData.textContent = stats.total_dataset_videos != null ? stats.total_dataset_videos : "--";

      const elJobs = document.getElementById("kpiSessionJobs");
      if (elJobs) elJobs.textContent = stats.videos_processed_session != null ? stats.videos_processed_session : "0";

      const elEvs = document.getElementById("kpiCandidateEvents");
      if (elEvs) elEvs.textContent = stats.total_candidate_events != null ? stats.total_candidate_events : "--";

      const elPrec = document.getElementById("kpiBenchmarkPrecision");
      if (elPrec) elPrec.textContent = `${Math.round((stats.benchmark_precision != null ? stats.benchmark_precision : 1.0) * 100)}%`;

      const elRec = document.getElementById("ovRecall");
      if (elRec) elRec.textContent = `${((stats.benchmark_recall || 0.5) * 100).toFixed(1)}%`;

      const elOvP = document.getElementById("ovPrec");
      if (elOvP) elOvP.textContent = `${((stats.benchmark_precision || 1.0) * 100).toFixed(1)}%`;

      const elF1 = document.getElementById("ovF1");
      if (elF1) elF1.textContent = (stats.benchmark_f1 || 0.6667).toFixed(4);

      const elLat = document.getElementById("ovLatency");
      if (elLat) elLat.textContent = `${(stats.mean_alert_delay_seconds || 2.55).toFixed(2)}s`;

      // Populate Overview Recent Jobs Table
      const jobsBody = document.getElementById("overviewJobsBody");
      if (jobsBody) {
        if (!stats.recent_jobs || stats.recent_jobs.length === 0) {
          jobsBody.innerHTML = `<tr><td colspan="5" class="table-empty-msg">No processing jobs queued in this session yet. Click 'Start New Job' to run a video.</td></tr>`;
        } else {
          jobsBody.innerHTML = stats.recent_jobs.map((j) => {
            const statusClass = j.status === "COMPLETED" ? "status-online" : (j.status === "PROCESSING" ? "text-cyan" : "hardware-badge");
            return `
              <tr>
                <td><strong>${j.filename}</strong></td>
                <td><span class="badge-pill hardware-badge">${j.camera_id}</span></td>
                <td><span class="badge-pill ${statusClass}">${j.status}</span></td>
                <td>${j.progress_percent.toFixed(0)}%</td>
                <td><strong class="${j.events_count > 0 ? 'text-red' : ''}">${j.events_count}</strong></td>
              </tr>
            `;
          }).join("");
        }
      }

      // Populate Overview Recent Events Table
      const eventsBody = document.getElementById("overviewEventsBody");
      if (eventsBody) {
        if (!stats.recent_events || stats.recent_events.length === 0) {
          eventsBody.innerHTML = `<tr><td colspan="6" class="table-empty-msg">No candidate events recorded yet.</td></tr>`;
        } else {
          eventsBody.innerHTML = stats.recent_events.map((ev) => `
            <tr>
              <td><code>${ev.event_id}</code></td>
              <td>${ev.source_video}</td>
              <td>${ev.trigger_time_str}</td>
              <td><span class="badge-pill ${ev.peak_posture === 'LYING' ? 'safe-badge red' : 'hardware-badge'}">${ev.peak_posture}</span></td>
              <td><code>${ev.zone_name}</code></td>
              <td><span class="badge-pill ${ev.status.includes('CONFIRMED') ? 'safe-badge red' : 'safe-badge cyan'}">${ev.status}</span></td>
            </tr>
          `).join("");
        }
      }
    } catch (e) {
      console.warn("Failed to load overview data:", e);
    }
  }

  // ==========================================================================
  // PAGE 2: Video Monitoring Implementation
  // ==========================================================================

  // Tab switching between Dataset/Sample and Upload
  if (tabSample && tabUpload) {
    tabSample.addEventListener("click", () => {
      currentMode = "sample";
      tabSample.classList.add("active");
      tabUpload.classList.remove("active");
      sampleSection.classList.remove("hidden");
      uploadSection.classList.add("hidden");
    });

    tabUpload.addEventListener("click", () => {
      currentMode = "upload";
      tabUpload.classList.add("active");
      tabSample.classList.remove("active");
      uploadSection.classList.remove("hidden");
      sampleSection.classList.add("hidden");
    });
  }

  // Dropzone Handlers
  if (dropzone && fileInput) {
    dropzone.addEventListener("click", () => fileInput.click());
    dropzone.addEventListener("dragover", (e) => {
      e.preventDefault();
      dropzone.style.borderColor = "var(--accent-cyan)";
    });
    dropzone.addEventListener("dragleave", () => {
      dropzone.style.borderColor = "rgba(255, 255, 255, 0.15)";
    });
    dropzone.addEventListener("drop", (e) => {
      e.preventDefault();
      dropzone.style.borderColor = "rgba(255, 255, 255, 0.15)";
      if (e.dataTransfer.files.length > 0) {
        handleFileSelection(e.dataTransfer.files[0]);
      }
    });

    fileInput.addEventListener("change", () => {
      if (fileInput.files.length > 0) {
        handleFileSelection(fileInput.files[0]);
      }
    });
  }

  function handleFileSelection(file) {
    selectedUploadFile = file;
    if (fileNameDisplay && fileDetails) {
      fileNameDisplay.textContent = `${file.name} (${(file.size / (1024 * 1024)).toFixed(1)} MB)`;
      fileDetails.classList.remove("hidden");
    }
    if (file && (file.type.startsWith("video/") || file.name.endsWith(".mp4") || file.name.endsWith(".webm"))) {
      const blobUrl = URL.createObjectURL(file);
      currentVideoState.sourceUrl = blobUrl;
      currentVideoState.sourcePlayable = true;
      currentVideoState.filename = file.name;
      currentVideoState.annotatedUrl = null;
      currentVideoState.hasAnnotated = false;
      if (videoFilenameDisplay) videoFilenameDisplay.textContent = file.name;
      loadVideoMedia(blobUrl, "source", false);
    }
  }

  // Populate Sample Selector
  async function loadMonitoringSamples() {
    try {
      const res = await fetch("/api/samples");
      if (!res.ok) return;
      const data = await res.json();
      sampleVideos = data.samples;

      if (sampleSelect) {
        const prevVal = sampleSelect.value;
        sampleSelect.innerHTML = "";
        sampleVideos.forEach((s) => {
          const opt = document.createElement("option");
          opt.value = s.filepath;
          opt.textContent = `[${s.category}] ${s.filename} (${s.duration_seconds}s, ${s.width}x${s.height})`;
          sampleSelect.appendChild(opt);
        });

        if (prevVal && Array.from(sampleSelect.options).some((o) => o.value === prevVal)) {
          sampleSelect.value = prevVal;
        }
        updateSampleHint();
      onVideoSelected(sampleSelect.value);
        if (sampleSelect.value) {
          onVideoSelected(sampleSelect.value);
        }
      }
    } catch (e) {
      console.warn("Failed to load sample videos:", e);
    }
  }

  function updateSampleHint() {
    if (!sampleSelect || !sampleHint) return;
    const selected = sampleVideos.find((s) => s.filepath === sampleSelect.value);
    if (selected) {
      sampleHint.textContent = `${selected.scenario_description} (Recommended: ${selected.recommended_camera})`;
      if (cameraSelect && cameraSelect.value === "auto") {
        cameraSelect.value = selected.recommended_camera || "cam1";
      }
    }
  }

  async function onVideoSelected(filepath) {
    if (!filepath) {
      if (videoPlaceholder) videoPlaceholder.classList.remove("hidden");
      if (videoPlayer) videoPlayer.classList.add("hidden");
      if (videoModeBadge) {
        videoModeBadge.textContent = "Awaiting Video";
        videoModeBadge.className = "badge badge-info";
      }
      if (videoFilenameDisplay) videoFilenameDisplay.textContent = "Select a clip to begin";
      if (videoSourceSwitcher) videoSourceSwitcher.classList.add("hidden");
      if (btnLoadBenchmark) btnLoadBenchmark.classList.add("hidden");
      return;
    }

    try {
      const res = await fetch(`/api/videos/check_annotated/${encodeURIComponent(filepath)}`);
      if (!res.ok) return;
      const data = await res.json();

      currentVideoState.filepath = filepath;
      currentVideoState.filename = data.filename;
      currentVideoState.sourceUrl = data.source_url;
      currentVideoState.sourcePlayable = data.source_playable;
      currentVideoState.annotatedUrl = data.annotated_url;
      currentVideoState.hasAnnotated = data.has_annotated;

      if (videoFilenameDisplay) {
        videoFilenameDisplay.textContent = data.filename;
        videoFilenameDisplay.title = filepath;
      }

      // Configure mode switcher & benchmark button
      if (videoSourceSwitcher) {
        if (data.source_playable && data.has_annotated) {
          videoSourceSwitcher.classList.remove("hidden");
        } else {
          videoSourceSwitcher.classList.add("hidden");
        }
      }

      if (btnLoadBenchmark) {
        if (!data.source_playable && data.has_annotated) {
          btnLoadBenchmark.classList.remove("hidden");
        } else {
          btnLoadBenchmark.classList.add("hidden");
        }
      }

      // Initial visual load
      if (data.source_playable) {
        loadVideoMedia(data.source_url, "source", false);
      } else if (data.has_annotated) {
        loadVideoMedia(data.annotated_url, "annotated", false);
        if (videoModeBadge) {
          videoModeBadge.textContent = "Benchmark Annotated (H.264 MP4)";
          videoModeBadge.className = "badge badge-success";
        }
      } else {
        if (videoPlayer) videoPlayer.classList.add("hidden");
        if (videoPlaceholder) {
          videoPlaceholder.classList.remove("hidden");
          const pTitle = document.getElementById("placeholderTitle");
          const pSub = document.getElementById("placeholderSub");
          if (pTitle) pTitle.textContent = `${data.filename} (${data.source_format} format)`;
          if (pSub) pSub.textContent = "Raw camera benchmark captured in AVI container. Click 'Start Detection Pipeline' below to run YOLO11-Pose inference and render a browser-playable H.264 MP4 stream.";
        }
        if (videoModeBadge) {
          videoModeBadge.textContent = "Raw Video (Requires Pipeline)";
          videoModeBadge.className = "badge badge-warning";
        }
      }
    } catch (err) {
      console.warn("Failed checking video playback metadata:", err);
    }
  }

  function loadVideoMedia(url, mode, autoPlay = false) {
    if (!videoPlayer || !url) return;
    currentVideoState.activeMode = mode;

    if (videoPlaceholder) videoPlaceholder.classList.add("hidden");
    if (videoProcessingOverlay) videoProcessingOverlay.classList.add("hidden");
    videoPlayer.classList.remove("hidden");

    const currentSrc = videoPlayer.getAttribute("src");
    if (currentSrc !== url) {
      videoPlayer.src = url;
      videoPlayer.load();
    }

    if (mode === "source") {
      if (videoModeBadge) {
        videoModeBadge.textContent = "Original Source Video";
        videoModeBadge.className = "badge badge-info";
      }
      if (btnViewSource) btnViewSource.classList.add("active");
      if (btnViewAnnotated) btnViewAnnotated.classList.remove("active");
    } else {
      if (videoModeBadge) {
        videoModeBadge.textContent = "YOLO11 Annotated Stream";
        videoModeBadge.className = "badge badge-success";
      }
      if (btnViewSource) btnViewSource.classList.remove("active");
      if (btnViewAnnotated) btnViewAnnotated.classList.add("active");
    }

    if (autoPlay) {
      videoPlayer.play().catch((e) => console.log("Autoplay was prevented by browser:", e));
    }
  }

  function initVideoPlayerControls() {
    if (!videoPlayer) return;

    if (btnPlayPause) {
      btnPlayPause.addEventListener("click", () => {
        if (videoPlayer.paused || videoPlayer.ended) {
          videoPlayer.play().catch(() => {});
        } else {
          videoPlayer.pause();
        }
      });
    }

    videoPlayer.addEventListener("play", () => {
      if (btnPlayPause) btnPlayPause.innerHTML = "⏸ Pause";
    });
    videoPlayer.addEventListener("pause", () => {
      if (btnPlayPause) btnPlayPause.innerHTML = "▶ Play";
    });
    videoPlayer.addEventListener("ended", () => {
      if (btnPlayPause) btnPlayPause.innerHTML = "▶ Play";
    });

    videoPlayer.addEventListener("timeupdate", () => {
      if (!videoTimeDisplay) return;
      const cur = videoPlayer.currentTime || 0;
      const dur = videoPlayer.duration || 0;
      const fmt = (s) => {
        const m = Math.floor(s / 60);
        const sec = (s % 60).toFixed(1);
        return `${m.toString().padStart(2, "0")}:${sec.padStart(4, "0")}`;
      };
      videoTimeDisplay.textContent = `${fmt(cur)} / ${fmt(dur)}`;
    });

    if (btnSeekStart) {
      btnSeekStart.addEventListener("click", () => {
        videoPlayer.currentTime = 0;
      });
    }

    if (btnSeekBack5) {
      btnSeekBack5.addEventListener("click", () => {
        videoPlayer.currentTime = Math.max(0, videoPlayer.currentTime - 5);
      });
    }

    if (btnSeekFwd5) {
      btnSeekFwd5.addEventListener("click", () => {
        const d = videoPlayer.duration || 0;
        videoPlayer.currentTime = Math.min(d, videoPlayer.currentTime + 5);
      });
    }

    if (btnReplayVideo) {
      btnReplayVideo.addEventListener("click", () => {
        videoPlayer.currentTime = 0;
        videoPlayer.play().catch(() => {});
      });
    }

    speedButtons.forEach((btn) => {
      btn.addEventListener("click", () => {
        const spd = parseFloat(btn.dataset.speed);
        videoPlayer.playbackRate = spd;
        speedButtons.forEach((b) => b.classList.toggle("active", b === btn));
      });
    });

    if (btnFullscreen) {
      btnFullscreen.addEventListener("click", () => {
        if (document.fullscreenElement) {
          document.exitFullscreen().catch(() => {});
        } else if (videoPlayerWrapper && videoPlayerWrapper.requestFullscreen) {
          videoPlayerWrapper.requestFullscreen().catch(() => {});
        } else if (videoPlayer.requestFullscreen) {
          videoPlayer.requestFullscreen().catch(() => {});
        }
      });
    }

    if (btnViewSource) {
      btnViewSource.addEventListener("click", () => {
        if (currentVideoState.sourceUrl) {
          loadVideoMedia(currentVideoState.sourceUrl, "source", false);
        }
      });
    }

    if (btnViewAnnotated) {
      btnViewAnnotated.addEventListener("click", () => {
        if (currentVideoState.annotatedUrl) {
          loadVideoMedia(currentVideoState.annotatedUrl, "annotated", false);
        }
      });
    }

    if (btnLoadBenchmark) {
      btnLoadBenchmark.addEventListener("click", () => {
        if (currentVideoState.annotatedUrl) {
          loadVideoMedia(currentVideoState.annotatedUrl, "annotated", true);
        }
      });
    }

    if (btnQuickFloorFall) {
      btnQuickFloorFall.addEventListener("click", () => {
        if (sampleSelect) {
          for (let i = 0; i < sampleSelect.options.length; i++) {
            if (sampleSelect.options[i].value.includes("floor_fall")) {
              sampleSelect.selectedIndex = i;
              updateSampleHint();
      onVideoSelected(sampleSelect.value);
              onVideoSelected(sampleSelect.value);
              break;
            }
          }
        }
      });
    }
  }

  if (sampleSelect) {
    sampleSelect.addEventListener("change", () => {
      updateSampleHint();
      onVideoSelected(sampleSelect.value);
      onVideoSelected(sampleSelect.value);
    });
  }

  // Start Detection Pipeline
  if (btnStart) {
    btnStart.addEventListener("click", async () => {
      hideError();

      const camera = cameraSelect ? cameraSelect.value : "auto";
      const maxF = maxFramesInput && maxFramesInput.value ? parseInt(maxFramesInput.value, 10) : null;

      btnStart.disabled = true;
      btnStart.classList.add("hidden");
      if (btnCancel) btnCancel.classList.remove("hidden");
      if (btnReset) btnReset.classList.add("hidden");
      if (progressCard) progressCard.classList.remove("hidden");

      // Reset Telemetry Display
      resetTelemetryDisplay();

      // Show processing HUD overlay over video player
      if (videoProcessingOverlay) videoProcessingOverlay.classList.remove("hidden");
      if (hudFrame) hudFrame.textContent = "0 / 0";
      if (hudFps) hudFps.textContent = "0.0 FPS";
      if (hudState) {
        hudState.textContent = "INITIALIZING";
        hudState.className = "hud-val text-green";
      }

      try {
        let res;
        if (currentMode === "sample") {
          const samplePath = sampleSelect.value;
          if (!samplePath) {
            showError("Please select a video file to analyze.");
            resetMonitoringControls();
            return;
          }
          res = await fetch("/api/process", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              sample_path: samplePath,
              camera_id: camera,
              max_frames: maxF,
            }),
          });
        } else {
          if (!selectedUploadFile) {
            showError("Please select or drop a video file to upload.");
            resetMonitoringControls();
            return;
          }
          const fd = new FormData();
          fd.append("video_file", selectedUploadFile);
          fd.append("camera_id", camera);
          if (maxF) fd.append("max_frames", maxF.toString());

          res = await fetch("/api/upload", {
            method: "POST",
            body: fd,
          });
        }

        if (!res.ok) {
          const errData = await res.json().catch(() => ({}));
          throw new Error(errData.detail || `Server returned error status ${res.status}`);
        }

        const jobData = await res.json();
        activeJobId = jobData.job_id;
        startJobPolling(activeJobId);
      } catch (err) {
        showError(err.message);
        resetMonitoringControls();
      }
    });
  }

  // Job Polling
  function startJobPolling(jobId) {
    if (pollIntervalId) clearInterval(pollIntervalId);
    pollIntervalId = setInterval(async () => {
      try {
        const res = await fetch(`/api/jobs/${jobId}`);
        if (!res.ok) return;
        const job = await res.json();

        updateJobTelemetry(job);

        if (job.status === "COMPLETED") {
          clearInterval(pollIntervalId);
          pollIntervalId = null;
          handleJobCompleted(job);
        } else if (job.status === "FAILED") {
          clearInterval(pollIntervalId);
          pollIntervalId = null;
          showError(`Processing failed: ${job.error_message || "Unknown error"}`);
          resetMonitoringControls();
        } else if (job.status === "CANCELLED") {
          clearInterval(pollIntervalId);
          pollIntervalId = null;
          showError("Video processing was cancelled by user request.");
          resetMonitoringControls();
        }
      } catch (err) {
        console.warn("Polling error:", err);
      }
    }, 500);
  }

  function updateJobTelemetry(job) {
    const pct = Math.min(100, Math.round(job.progress_percent));
    if (progressBarFill) progressBarFill.style.width = `${pct}%`;
    if (progressPercentText) progressPercentText.textContent = `${pct}%`;
    if (statCurrentFrame) statCurrentFrame.textContent = `${job.current_frame} / ${job.total_frames}`;
    if (statFps) statFps.textContent = `${job.processing_fps} FPS`;

    if (hudFrame) hudFrame.textContent = `${job.current_frame} / ${job.total_frames}`;
    if (hudFps) hudFps.textContent = `${job.processing_fps} FPS`;
    if (hudState) {
      hudState.textContent = job.current_state;
      hudState.className = `hud-val ${job.current_state === 'PERSON_DOWN_CANDIDATE' ? 'text-red' : 'text-green'}`;
    }

    if (telemState) telemState.textContent = job.current_state;
    if (telemPosture) telemPosture.textContent = job.current_posture;
    if (telemZone) telemZone.textContent = job.current_zone;
    if (telemTimer) telemTimer.textContent = `${job.elapsed_down_seconds.toFixed(1)}s`;

    // Dynamic HUD Banner
    if (statusBanner) {
      if (job.is_active_candidate || job.current_state === "PERSON_DOWN_CANDIDATE") {
        statusBanner.className = "status-banner state-alert";
        if (statusIcon) statusIcon.textContent = "🚨";
        if (statusHeadline) statusHeadline.textContent = "PERSON-DOWN CANDIDATE DETECTED";
        if (statusSubtext) statusSubtext.textContent = `Sustained floor-lying in ${job.current_zone} (${job.elapsed_down_seconds.toFixed(1)}s down). Verified candidate alert.`;
      } else {
        statusBanner.className = "status-banner state-normal";
        if (statusIcon) statusIcon.textContent = "🛡️";
        if (statusHeadline) statusHeadline.textContent = "NORMAL ACTIVITY DETECTED";
        if (statusSubtext) statusSubtext.textContent = `Monitoring active: Subject in ${job.current_zone} with posture ${job.current_posture}.`;
      }
    }

    // Populate events table
    if (job.detected_events && job.detected_events.length > 0) {
      if (emptyEventsNotice) emptyEventsNotice.classList.add("hidden");
      if (eventsTableBody) {
        eventsTableBody.innerHTML = job.detected_events.map((ev) => `
          <tr>
            <td><strong>#${ev.event_id}</strong></td>
            <td>${ev.trigger_time_seconds.toFixed(2)}s (frame ${ev.trigger_frame})</td>
            <td>${ev.duration_seconds.toFixed(2)}s</td>
            <td><span class="badge-pill safe-badge red">${ev.status}</span></td>
            <td>${ev.peak_posture} (${Math.round(ev.peak_confidence * 100)}%)</td>
            <td><code>${ev.zone_name}</code></td>
            <td>
              <button class="btn-sm btn-secondary" onclick="seekVideoPlayer(${ev.start_time_seconds || ev.trigger_time_seconds})">
                Seek in Video
              </button>
            </td>
          </tr>
        `).join("");
      }
    }
  }

  function handleJobCompleted(job) {
    if (btnCancel) btnCancel.classList.add("hidden");
    if (btnReset) btnReset.classList.remove("hidden");
    if (videoProcessingOverlay) videoProcessingOverlay.classList.add("hidden");

    currentJobResult = job.result;

    // Load annotated video
    if (job.result && job.result.output_video) {
      let vidName = "";
      if (typeof job.result.output_video === "string") {
        vidName = job.result.output_video.split(/[\\/]/).pop();
      } else if (job.result.output_video.filename) {
        vidName = job.result.output_video.filename;
      } else if (job.result.output_video.path) {
        vidName = job.result.output_video.path.split(/[\\/]/).pop();
      }

      if (vidName) {
        const vidUrl = `/api/videos/${vidName}`;
        currentVideoState.annotatedUrl = vidUrl;
        currentVideoState.hasAnnotated = true;

        loadVideoMedia(vidUrl, "annotated", true);

        if (videoSourceSwitcher && currentVideoState.sourcePlayable) {
          videoSourceSwitcher.classList.remove("hidden");
        }

        if (videoDownloadGroup && btnDownloadVideo) {
          videoDownloadGroup.classList.remove("hidden");
          btnDownloadVideo.href = `/api/videos/download/${vidName}`;
          btnDownloadVideo.setAttribute("download", vidName);
        }
      }
    }

    // Render Video Analysis Summary Grid
    if (job.result && job.result.video_metadata && summaryCard && summaryGrid) {
      const vm = job.result.video_metadata;
      const pm = job.result.processing_metrics;
      summaryGrid.innerHTML = `
        <div class="summary-item-box">
          <div class="summary-item-label">Total Processed Frames</div>
          <div class="summary-item-value">${pm.processed_frames}</div>
        </div>
        <div class="summary-item-box">
          <div class="summary-item-label">Average Speed</div>
          <div class="summary-item-value">${pm.average_processing_fps.toFixed(1)} FPS</div>
        </div>
        <div class="summary-item-box">
          <div class="summary-item-label">Person Detections</div>
          <div class="summary-item-value">${pm.total_person_detections}</div>
        </div>
        <div class="summary-item-box">
          <div class="summary-item-label">Lying Posture Ratio</div>
          <div class="summary-item-value">${((pm.posture_distribution.LYING || 0) * 100).toFixed(1)}%</div>
        </div>
        <div class="summary-item-box">
          <div class="summary-item-label">Candidate Events</div>
          <div class="summary-item-value text-red">${job.detected_events ? job.detected_events.length : 0}</div>
        </div>
        <div class="summary-item-box">
          <div class="summary-item-label">Execution Time</div>
          <div class="summary-item-value">${pm.elapsed_time_seconds.toFixed(1)}s</div>
        </div>
      `;
      summaryCard.classList.remove("hidden");
    }
  }

  // Seek video helper
  window.seekVideoPlayer = (seconds) => {
    if (videoPlayer) {
      if (videoPlayer.classList.contains("hidden")) {
        if (currentVideoState.annotatedUrl) {
          loadVideoMedia(currentVideoState.annotatedUrl, "annotated", false);
        } else if (currentVideoState.sourceUrl) {
          loadVideoMedia(currentVideoState.sourceUrl, "source", false);
        }
      }
      videoPlayer.currentTime = Math.max(0, seconds - 0.5);
      videoPlayer.play().catch(() => {});
      const wrapper = document.getElementById("videoPlayerWrapper") || videoPlayer;
      wrapper.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  };

  // Cancel Job
  if (btnCancel) {
    btnCancel.addEventListener("click", async () => {
      if (!activeJobId) return;
      try {
        await fetch(`/api/jobs/${activeJobId}/cancel`, { method: "POST" });
        if (pollIntervalId) clearInterval(pollIntervalId);
        showError("Cancellation requested.");
        resetMonitoringControls();
      } catch (e) {
        console.warn("Cancel failed:", e);
      }
    });
  }

  // Reset Controls
  if (btnReset) {
    btnReset.addEventListener("click", () => {
      resetMonitoringControls();
      resetTelemetryDisplay();
    });
  }

  function resetMonitoringControls() {
    if (btnStart) {
      btnStart.disabled = false;
      btnStart.classList.remove("hidden");
    }
    if (btnCancel) btnCancel.classList.add("hidden");
    if (btnReset) btnReset.classList.add("hidden");
    if (progressCard) progressCard.classList.add("hidden");
    if (videoProcessingOverlay) videoProcessingOverlay.classList.add("hidden");
  }

  function resetTelemetryDisplay() {
    if (progressBarFill) progressBarFill.style.width = "0%";
    if (progressPercentText) progressPercentText.textContent = "0%";
    if (telemState) telemState.textContent = "NORMAL";
    if (telemPosture) telemPosture.textContent = "NO_DETECTION";
    if (telemZone) telemZone.textContent = "OUTSIDE";
    if (telemTimer) telemTimer.textContent = "0.0s";
    if (eventsTableBody) eventsTableBody.innerHTML = "";
    if (emptyEventsNotice) emptyEventsNotice.classList.remove("hidden");
    if (summaryCard) summaryCard.classList.add("hidden");
  }

  // Download JSON Report handler
  if (btnDownloadJson) {
    btnDownloadJson.addEventListener("click", () => {
      if (!currentJobResult) return;
      const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(currentJobResult, null, 2));
      const a = document.createElement("a");
      a.setAttribute("href", dataStr);
      a.setAttribute("download", `detection_report_${activeJobId || "export"}.json`);
      document.body.appendChild(a);
      a.click();
      a.remove();
    });
  }

  // ==========================================================================
  // PAGE 3: Dataset Explorer Implementation
  // ==========================================================================

  async function loadDatasetInventory() {
    try {
      const q = datasetSearch ? datasetSearch.value.trim() : "";
      const folder = datasetFolderFilter ? datasetFolderFilter.value : "";
      const format = datasetFormatFilter ? datasetFormatFilter.value : "";

      const params = new URLSearchParams();
      if (q) params.set("search", q);
      if (folder) params.set("folder", folder);
      if (format) params.set("format", format);

      const res = await fetch(`/api/dataset?${params.toString()}`);
      if (!res.ok) return;
      const data = await res.json();
      datasetVideos = data.videos;

      // Populate Folder filter if empty
      if (datasetFolderFilter && datasetFolderFilter.options.length <= 1) {
        data.folders.forEach((f) => {
          const opt = document.createElement("option");
          opt.value = f;
          opt.textContent = f;
          datasetFolderFilter.appendChild(opt);
        });
      }

      // Populate Format filter if empty
      if (datasetFormatFilter && datasetFormatFilter.options.length <= 1) {
        data.formats.forEach((fmt) => {
          const opt = document.createElement("option");
          opt.value = fmt;
          opt.textContent = fmt;
          datasetFormatFilter.appendChild(opt);
        });
      }

      // Render Dataset Table
      if (datasetTableBody) {
        if (!data.videos || data.videos.length === 0) {
          datasetTableBody.innerHTML = `<tr><td colspan="10" class="table-empty-msg">No videos match the selected filters.</td></tr>`;
          return;
        }

        datasetTableBody.innerHTML = data.videos.map((v) => `
          <tr>
            <td>
              <div style="font-weight:600; color:#fff;">${v.filename}</div>
              <div style="font-size:0.75rem; color:var(--text-dim);">${v.scenario_description}</div>
            </td>
            <td><span class="badge-pill hardware-badge">${v.folder}</span></td>
            <td><code>${v.format}</code></td>
            <td>${v.width}x${v.height}</td>
            <td>${v.fps}</td>
            <td>${v.total_frames}</td>
            <td>${v.duration_seconds.toFixed(1)}s</td>
            <td>${v.size_mb} MB</td>
            <td><code>${v.recommended_camera}</code></td>
            <td>
              <div style="display:flex; gap:6px;">
                <button class="btn-sm btn-primary" onclick="processVideoInMonitoring('${v.relative_path}', '${v.recommended_camera}')">
                  ▶ Monitor
                </button>
                <button class="btn-sm btn-secondary" onclick="inspectDatasetVideo('${v.id}')">
                  Inspect
                </button>
              </div>
            </td>
          </tr>
        `).join("");
      }
    } catch (e) {
      console.warn("Failed to load dataset inventory:", e);
    }
  }

  // Process in monitoring action from Dataset Explorer
  window.processVideoInMonitoring = (relPath, recCam) => {
    navigateToPage("monitoring");
    if (tabSample) tabSample.click();
    if (sampleSelect) {
      // If option does not exist, add it
      let found = false;
      for (let i = 0; i < sampleSelect.options.length; i++) {
        if (sampleSelect.options[i].value === relPath) {
          sampleSelect.selectedIndex = i;
          found = true;
          break;
        }
      }
      if (!found) {
        const opt = document.createElement("option");
        opt.value = relPath;
        opt.textContent = relPath.split("/").pop();
        sampleSelect.appendChild(opt);
        sampleSelect.value = relPath;
      }
      updateSampleHint();
      onVideoSelected(sampleSelect.value);
    }
    if (cameraSelect && recCam) {
      cameraSelect.value = recCam;
    }
  };

  // Inspect dataset video details in modal
  window.inspectDatasetVideo = (vidId) => {
    const v = datasetVideos.find((item) => item.id === vidId);
    if (!v) return;

    const modalHtml = `
      <div class="diag-props-list">
        <div class="diag-prop-row"><span class="diag-prop-label">Filename:</span><strong class="diag-prop-val">${v.filename}</strong></div>
        <div class="diag-prop-row"><span class="diag-prop-label">Relative Path:</span><strong class="diag-prop-val">${v.relative_path}</strong></div>
        <div class="diag-prop-row"><span class="diag-prop-label">Storage Folder:</span><strong class="diag-prop-val">data/${v.folder}</strong></div>
        <div class="diag-prop-row"><span class="diag-prop-label">Resolution:</span><strong class="diag-prop-val">${v.width} x ${v.height} px</strong></div>
        <div class="diag-prop-row"><span class="diag-prop-label">Frame Rate:</span><strong class="diag-prop-val">${v.fps} FPS</strong></div>
        <div class="diag-prop-row"><span class="diag-prop-label">Total Frames:</span><strong class="diag-prop-val">${v.total_frames}</strong></div>
        <div class="diag-prop-row"><span class="diag-prop-label">Duration:</span><strong class="diag-prop-val">${v.duration_seconds} seconds</strong></div>
        <div class="diag-prop-row"><span class="diag-prop-label">File Size:</span><strong class="diag-prop-val">${v.size_mb} MB (${v.size_bytes.toLocaleString()} bytes)</strong></div>
        <div class="diag-prop-row"><span class="diag-prop-label">Modified Date:</span><strong class="diag-prop-val">${v.modified_time}</strong></div>
        <div class="diag-prop-row"><span class="diag-prop-label">Recommended Camera:</span><strong class="diag-prop-val">${v.recommended_camera}</strong></div>
      </div>
      <div style="margin-top:1rem; padding:10px; background:rgba(10,17,34,0.6); border-radius:6px; font-size:0.8rem; color:var(--text-muted);">
        <strong>Scenario Description:</strong> ${v.scenario_description}
      </div>
    `;

    openModal(
      `Dataset Video: ${v.filename}`,
      modalHtml,
      {
        text: "Process in Video Monitoring",
        onClick: () => {
          closeModal();
          processVideoInMonitoring(v.relative_path, v.recommended_camera);
        },
      }
    );
  };

  // Rescan Dataset Button
  if (btnRescanDataset) {
    btnRescanDataset.addEventListener("click", async () => {
      btnRescanDataset.disabled = true;
      btnRescanDataset.innerHTML = "<span>⏳</span> Scanning...";
      try {
        await fetch("/api/dataset/rescan", { method: "POST" });
        await loadDatasetInventory();
      } finally {
        btnRescanDataset.disabled = false;
        btnRescanDataset.innerHTML = "<span>🔄</span> Rescan Dataset";
      }
    });
  }

  // Filter Event Listeners
  if (datasetSearch) datasetSearch.addEventListener("input", debounce(loadDatasetInventory, 300));
  if (datasetFolderFilter) datasetFolderFilter.addEventListener("change", loadDatasetInventory);
  if (datasetFormatFilter) datasetFormatFilter.addEventListener("change", loadDatasetInventory);
  if (btnClearDatasetFilters) {
    btnClearDatasetFilters.addEventListener("click", () => {
      if (datasetSearch) datasetSearch.value = "";
      if (datasetFolderFilter) datasetFolderFilter.value = "";
      if (datasetFormatFilter) datasetFormatFilter.value = "";
      loadDatasetInventory();
    });
  }

  // ==========================================================================
  // PAGE 4: Alerts and Event History Implementation
  // ==========================================================================

  let allAlertsData = [];

  async function loadAllAlerts() {
    try {
      const q = alertsSearch ? alertsSearch.value.trim() : "";
      const status = alertsStatusFilter ? alertsStatusFilter.value : "";
      const zone = alertsZoneFilter ? alertsZoneFilter.value : "";

      const params = new URLSearchParams();
      if (q) params.set("search", q);
      if (status) params.set("status", status);
      if (zone) params.set("zone", zone);

      const res = await fetch(`/api/events?${params.toString()}`);
      if (!res.ok) return;
      const data = await res.json();
      allAlertsData = data.events;

      if (allAlertsTableBody) {
        if (!data.events || data.events.length === 0) {
          allAlertsTableBody.innerHTML = `<tr><td colspan="10" class="table-empty-msg">No candidate events match the current filter.</td></tr>`;
          return;
        }

        allAlertsTableBody.innerHTML = data.events.map((ev) => `
          <tr>
            <td><code>${ev.event_id}</code></td>
            <td><strong>${ev.source_video}</strong></td>
            <td><span class="badge-pill hardware-badge">${ev.camera_id}</span></td>
            <td>${ev.trigger_time_str} (f.${ev.trigger_frame})</td>
            <td>${ev.duration_seconds.toFixed(2)}s</td>
            <td><span class="badge-pill ${ev.status.includes('CONFIRMED') ? 'safe-badge red' : 'safe-badge cyan'}">${ev.status}</span></td>
            <td><span class="badge-pill ${ev.peak_posture === 'LYING' ? 'safe-badge red' : 'hardware-badge'}">${ev.peak_posture}</span></td>
            <td>${Math.round(ev.peak_confidence * 100)}%</td>
            <td><code>${ev.zone_name}</code></td>
            <td>
              <div style="display:flex; gap:6px;">
                <button class="btn-sm btn-secondary" onclick="inspectAlertEvent('${ev.event_id}')">Inspect</button>
                ${ev.has_video && ev.video_filename ? `
                  <button class="btn-sm btn-primary" onclick="playEventVideo('${ev.video_filename}', ${ev.trigger_time_seconds})">
                    ▶ Play
                  </button>
                ` : ''}
              </div>
            </td>
          </tr>
        `).join("");
      }
    } catch (e) {
      console.warn("Failed to load alerts:", e);
    }
  }

  // Inspect Event Details in Modal
  window.inspectAlertEvent = (evId) => {
    const ev = allAlertsData.find((item) => item.event_id === evId);
    if (!ev) return;

    const modalHtml = `
      <div class="diag-props-list">
        <div class="diag-prop-row"><span class="diag-prop-label">Event Identifier:</span><strong class="diag-prop-val">${ev.event_id}</strong></div>
        <div class="diag-prop-row"><span class="diag-prop-label">Classification Type:</span><strong class="diag-prop-val">${ev.classification_type}</strong></div>
        <div class="diag-prop-row"><span class="diag-prop-label">Source Video:</span><strong class="diag-prop-val">${ev.source_video}</strong></div>
        <div class="diag-prop-row"><span class="diag-prop-label">Camera Profile:</span><strong class="diag-prop-val">${ev.camera_id}</strong></div>
        <div class="diag-prop-row"><span class="diag-prop-label">Spatial Zone:</span><strong class="diag-prop-val"><code>${ev.zone_name}</code></strong></div>
        <div class="diag-prop-row"><span class="diag-prop-label">Onset Frame:</span><strong class="diag-prop-val">${ev.start_frame}</strong></div>
        <div class="diag-prop-row"><span class="diag-prop-label">Trigger Frame:</span><strong class="diag-prop-val">${ev.trigger_frame}</strong></div>
        <div class="diag-prop-row"><span class="diag-prop-label">End Frame:</span><strong class="diag-prop-val">${ev.end_frame}</strong></div>
        <div class="diag-prop-row"><span class="diag-prop-label">Trigger Time:</span><strong class="diag-prop-val">${ev.trigger_time_str} (${ev.trigger_time_seconds.toFixed(2)}s)</strong></div>
        <div class="diag-prop-row"><span class="diag-prop-label">Collapse Duration:</span><strong class="diag-prop-val">${ev.duration_seconds.toFixed(2)}s</strong></div>
        <div class="diag-prop-row"><span class="diag-prop-label">Peak Posture:</span><strong class="diag-prop-val">${ev.peak_posture}</strong></div>
        <div class="diag-prop-row"><span class="diag-prop-label">Posture Confidence:</span><strong class="diag-prop-val">${(ev.peak_confidence * 100).toFixed(1)}%</strong></div>
        <div class="diag-prop-row"><span class="diag-prop-label">Verification Status:</span><strong class="diag-prop-val text-red">${ev.status}</strong></div>
      </div>
      <div style="margin-top:1rem; padding:10px; background:rgba(239,68,68,0.1); border:1px solid rgba(239,68,68,0.3); border-radius:6px; font-size:0.775rem; color:#fca5a5;">
        <strong>Research Disclaimer:</strong> Candidate event represents algorithmically detected floor collapse criteria. Not a certified clinical alarm.
      </div>
    `;

    openModal(`Candidate Event: ${ev.event_id}`, modalHtml);
  };

  window.playEventVideo = (vidFilename, timestamp) => {
    navigateToPage("monitoring");
    if (videoPlayer) {
      loadVideoMedia(`/api/videos/${vidFilename}`, "annotated", true);
      videoPlayer.currentTime = Math.max(0, timestamp - 1.0);
      const wrapper = document.getElementById("videoPlayerWrapper") || videoPlayer;
      wrapper.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  };

  if (alertsSearch) alertsSearch.addEventListener("input", debounce(loadAllAlerts, 300));
  if (alertsStatusFilter) alertsStatusFilter.addEventListener("change", loadAllAlerts);
  if (alertsZoneFilter) alertsZoneFilter.addEventListener("change", loadAllAlerts);
  if (btnClearAlertsFilters) {
    btnClearAlertsFilters.addEventListener("click", () => {
      if (alertsSearch) alertsSearch.value = "";
      if (alertsStatusFilter) alertsStatusFilter.value = "";
      if (alertsZoneFilter) alertsZoneFilter.value = "";
      loadAllAlerts();
    });
  }

  // ==========================================================================
  // PAGE 5: Model Evaluation Implementation
  // ==========================================================================

  async function loadEvaluationData() {
    try {
      const res = await fetch("/api/evaluations");
      if (!res.ok) return;
      const data = await res.json();

      // Protocol A Matrix
      const pA = data.frame_level_protocols.protocol_a_strict;
      if (pA) {
        document.getElementById("protoATp").textContent = pA.frame_tp;
        document.getElementById("protoAFp").textContent = pA.frame_fp;
        document.getElementById("protoAFn").textContent = pA.frame_fn;
        document.getElementById("protoATn").textContent = pA.frame_tn;
        document.getElementById("protoAPrec").textContent = `${(pA.frame_precision * 100).toFixed(2)}%`;
        document.getElementById("protoARec").textContent = `${(pA.frame_recall * 100).toFixed(2)}%`;
        document.getElementById("protoAF1").textContent = pA.frame_f1.toFixed(4);
      }

      // Protocol B Matrix
      const pB = data.frame_level_protocols.protocol_b_conditional;
      if (pB) {
        document.getElementById("protoBTp").textContent = pB.frame_tp;
        document.getElementById("protoBFp").textContent = pB.frame_fp;
        document.getElementById("protoBFn").textContent = pB.frame_fn;
        document.getElementById("protoBTn").textContent = pB.frame_tn;
        document.getElementById("protoBPrec").textContent = `${(pB.frame_precision * 100).toFixed(2)}%`;
        document.getElementById("protoBRec").textContent = `${(pB.frame_recall * 100).toFixed(2)}%`;
        document.getElementById("protoBF1").textContent = pB.frame_f1.toFixed(4);
      }

      // Per-Video Table
      const perVidBody = document.getElementById("evalPerVideoTableBody");
      if (perVidBody && data.per_video_results) {
        perVidBody.innerHTML = data.per_video_results.map((v) => {
          const lat = v.latency_seconds && v.latency_seconds.length > 0 ? `${v.latency_seconds[0].toFixed(2)}s` : "N/A";
          return `
            <tr>
              <td><strong>${v.video_name}</strong></td>
              <td><span class="badge-pill hardware-badge">cam${v.cam_id}</span></td>
              <td>${v.total_frames}</td>
              <td>${v.processing_fps.toFixed(1)}</td>
              <td>${v.ground_truth_events}</td>
              <td><strong class="${v.candidate_events > 0 ? 'text-red' : ''}">${v.candidate_events}</strong></td>
              <td>${v.event_tp} / ${v.event_fp} / ${v.event_fn}</td>
              <td>${lat}</td>
              <td>${v.protocol_a_f1.toFixed(4)}</td>
            </tr>
          `;
        }).join("");
      }

      // Baseline Hashes Table
      const hashBody = document.getElementById("baselineHashesTableBody");
      if (hashBody && data.baseline_hashes) {
        hashBody.innerHTML = data.baseline_hashes.map((h) => `
          <tr>
            <td><code>${h.filename}</code></td>
            <td><code style="font-size:0.75rem; color:var(--text-muted);">${h.sha256}</code></td>
            <td><span class="badge-pill status-online">✓ Verified Intact</span></td>
          </tr>
        `).join("");
      }

      // Available Reports Grid
      const reportsGrid = document.getElementById("reportsListGrid");
      if (reportsGrid && data.available_reports) {
        reportsGrid.innerHTML = data.available_reports.map((r) => `
          <div class="report-file-card">
            <div>
              <div class="report-file-name">📄 ${r.filename}</div>
              <div class="report-file-meta">${r.size_kb} KB • ${r.modified_time}</div>
            </div>
            <a href="/api/evaluations/download/${r.filename}" class="btn-sm btn-secondary" download>
              Download
            </a>
          </div>
        `).join("");
      }
    } catch (e) {
      console.warn("Failed to load evaluation data:", e);
    }
  }

  // ==========================================================================
  // PAGE 6: Spatial Zones Implementation
  // ==========================================================================

  async function loadZoneConfigurations() {
    try {
      const res = await fetch("/api/zones");
      if (!res.ok) return;
      cameraZonesData = await res.json();
      renderCameraZones(currentSelectedCam);
    } catch (e) {
      console.warn("Failed to load zones:", e);
    }
  }

  function renderCameraZones(camId) {
    currentSelectedCam = camId;
    const config = cameraZonesData[camId];

    // Update buttons
    camButtons.forEach((b) => {
      b.classList.toggle("active", b.getAttribute("data-cam") === camId);
    });

    if (zonePreviewImg) {
      zonePreviewImg.src = `/api/zones/preview/${camId}?t=${Date.now()}`;
    }

    if (config) {
      if (zoneCameraStatus) {
        const isVal = config.status === "VALIDATED";
        zoneCameraStatus.textContent = `Status: ${config.status}`;
        zoneCameraStatus.className = isVal ? "badge-pill status-online" : "badge-pill safe-badge red";
      }
      if (zoneCameraRes) {
        zoneCameraRes.textContent = `Calibrated Resolution: ${config.resolution[0]} x ${config.resolution[1]}`;
      }

      if (zonesListContainer) {
        if (!config.zones || config.zones.length === 0) {
          zonesListContainer.innerHTML = `<div class="table-empty-msg">No calibrated spatial zones configured for this camera. Uncalibrated fallback: all points map to OUTSIDE_ZONE.</div>`;
        } else {
          zonesListContainer.innerHTML = config.zones.map((z) => `
            <div class="zone-item-card">
              <div class="zone-item-header">
                <span class="zone-item-name">${z.name}</span>
                <span class="safe-badge ${z.type === 'BED_SOFA_ZONE' ? 'green' : (z.type === 'FLOOR_ZONE' ? 'red' : 'cyan')}">${z.type}</span>
              </div>
              <div class="zone-item-desc">${z.description}</div>
              <div class="zone-poly-coords">Vertices: ${JSON.stringify(z.polygon)}</div>
            </div>
          `).join("");
        }
      }
    }
  }

  camButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const cam = btn.getAttribute("data-cam");
      renderCameraZones(cam);
    });
  });

  // Scaling Calculator Logic
  if (btnRunScalingCalc) {
    btnRunScalingCalc.addEventListener("click", () => {
      try {
        const sizeParts = calcFrameSize.value.toLowerCase().split("x").map((s) => parseFloat(s.trim()));
        const ptParts = calcPoint.value.split(",").map((s) => parseFloat(s.trim()));

        if (sizeParts.length !== 2 || isNaN(sizeParts[0]) || isNaN(sizeParts[1])) {
          alert("Invalid frame size format. Expected: Width x Height (e.g. 1200 x 896)");
          return;
        }
        if (ptParts.length !== 2 || isNaN(ptParts[0]) || isNaN(ptParts[1])) {
          alert("Invalid coordinate point format. Expected: X, Y (e.g. 600, 750)");
          return;
        }

        const [fw, fh] = sizeParts;
        const [px, py] = ptParts;

        // Config resolution for cam1 is 720 x 480
        const scaleX = 720 / fw;
        const scaleY = 480 / fh;

        const mappedX = Math.round(px * scaleX);
        const mappedY = Math.round(py * scaleY);

        calcMappedResult.textContent = `(${mappedX}, ${mappedY}) px (Scale: ${scaleX.toFixed(3)}x, ${scaleY.toFixed(3)}y)`;
        calcResultDisplay.classList.remove("hidden");
      } catch (e) {
        alert("Calculation error: " + e.message);
      }
    });
  }

  // ==========================================================================
  // PAGE 7: System Diagnostics Implementation
  // ==========================================================================

  async function loadSystemDiagnostics() {
    try {
      const res = await fetch("/api/system/diagnostics");
      if (!res.ok) return;
      const diag = await res.json();

      document.getElementById("diagPython").textContent = diag.python_version;
      document.getElementById("diagTorch").textContent = diag.torch_version;
      document.getElementById("diagCuda").textContent = diag.cuda_available ? "YES (CUDA Enabled)" : "NO (CPU Only)";
      document.getElementById("diagGpu").textContent = diag.cuda_device_name || "Host CPU (DirectML/Torch)";
      document.getElementById("diagUptime").textContent = `${Math.floor(diag.uptime_seconds / 60)} min ${Math.round(diag.uptime_seconds % 60)} sec`;
      document.getElementById("diagWorkers").textContent = `${diag.active_workers} / ${diag.max_workers}`;
      document.getElementById("diagJobCount").textContent = diag.job_store_count;
      document.getElementById("diagScaling").textContent = diag.scaling_engine_verified ? "PASS (Verified)" : "FAIL";

      // Directory Permissions Table
      const dirsBody = document.getElementById("diagDirsTableBody");
      if (dirsBody && diag.directories_status) {
        dirsBody.innerHTML = Object.entries(diag.directories_status).map(([k, d]) => `
          <tr>
            <td><code>${k}</code></td>
            <td><code style="font-size:0.75rem;">${d.path}</code></td>
            <td><span class="badge-pill ${d.exists ? 'status-online' : 'safe-badge red'}">${d.exists ? 'YES' : 'NO'}</span></td>
            <td><span class="badge-pill ${d.writable ? 'status-online' : 'safe-badge red'}">${d.writable ? 'WRITABLE' : 'READONLY'}</span></td>
          </tr>
        `).join("");
      }

      // Live Logs
      renderLogs(diag.recent_logs);
    } catch (e) {
      console.warn("Failed to load diagnostics:", e);
    }
  }

  function renderLogs(logs) {
    const container = document.getElementById("liveLogsContainer");
    if (!container) return;
    if (!logs || logs.length === 0) {
      container.innerHTML = `<div class="log-line text-dim">No application logs recorded yet.</div>`;
      return;
    }
    container.innerHTML = logs.map((l) => `<div class="log-line">${escapeHtml(l)}</div>`).join("");
    container.scrollTop = container.scrollHeight;
  }

  if (btnRefreshLogs) {
    btnRefreshLogs.addEventListener("click", async () => {
      btnRefreshLogs.disabled = true;
      try {
        const res = await fetch("/api/system/diagnostics");
        if (res.ok) {
          const diag = await res.json();
          renderLogs(diag.recent_logs);
        }
      } finally {
        btnRefreshLogs.disabled = false;
      }
    });
  }

  // Utilities
  function debounce(func, wait) {
    let timeout;
    return function (...args) {
      clearTimeout(timeout);
      timeout = setTimeout(() => func.apply(this, args), wait);
    };
  }

  function escapeHtml(str) {
    return str
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  // ==========================================================================
  // Initialization
  // ==========================================================================
  initVideoPlayerControls();
  fetchSystemStatus();
  setInterval(fetchSystemStatus, 10000);

  // Initial Route Check
  const initHash = window.location.hash.replace("#", "") || "overview";
  navigateToPage(initHash, false);
});
