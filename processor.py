"""
processor.py – Production-Grade RTSP Monitoring & AI Pipeline
=============================================================
Key improvements over v1:
  - MediaPipe REMOVED: ArcFace SCRFD provides detection + landmarks natively
  - RTSP reader throttled to TARGET_INGEST_FPS (default 8) — no more 30fps spin
  - Monitoring loop runs at configurable PROCESSING_FPS (default 5) — not 100fps
  - ProcessPoolExecutor for inference — true CPU parallelism, bypasses Python GIL
  - Pre-allocated frame buffer — eliminates 6MB-per-frame copy allocations
  - Counter-based consensus — 10x faster than nested list scanning
  - Per-camera structured logging with rate limiting (log every N events)
  - Graceful reconnect with exponential backoff (not fixed 5s sleep)
  - Thread-safe design reviewed for all shared state
"""

"""
processor.py – Multi-Camera Vision Pipeline [v3 — Multi-Office Ready]
=====================================================================
Orchestrates high-resolution RTSP ingestion and real-time face recognition.

Key Features:
  - Office Grouping: Independent logic for DEV and KINFRA sites.
  - Stability Tuning: Custom FFmpeg profiles for 5MP H.264 high-res streams.
  - Near-Only Detection: Dynamic face-size filtering to ignore background personnel.
  - Consensus Logic: Multi-frame agreement to eliminate transient false positives.
"""

import asyncio
import datetime
import logging
import os
import random
import threading
import time
from collections import Counter, deque
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

import config
import database
import engine

log = logging.getLogger("processor")

# ──────────────────────────────────────────────────────────────────────────────
#  Tuning constants  (NOT user-facing config — these are engineering knobs)
# ──────────────────────────────────────────────────────────────────────────────

# RTSP reader: capture at this rate regardless of camera native FPS
TARGET_INGEST_FPS   = config.TARGET_INGEST_FPS

# Monitoring pipeline: run inference at this rate
PROCESSING_FPS      = config.PROCESSING_FPS
_PROCESS_INTERVAL   = 1.0 / PROCESSING_FPS

# Backoff for RTSP reconnection (seconds): starts at 2s, caps at 30s
_RECONNECT_MIN_DELAY  = 2.0
_RECONNECT_MAX_DELAY  = 30.0

# Dashboard MJPEG stream FPS
STREAM_FPS          = config.STREAM_FPS
_STREAM_INTERVAL    = 1.0 / STREAM_FPS

# Pre-allocated buffer resolution — Increased to support 5MP (2560x1920) natively
_BUF_H, _BUF_W = 2000, 3000


# ══════════════════════════════════════════════════════════════════════════════
#  1. VideoProcessor — Throttled RTSP Ingest Thread
# ══════════════════════════════════════════════════════════════════════════════

class VideoProcessor:
    """
    Runs a single background thread to read from an RTSP/local source.

    Design choices:
      - Single pre-allocated buffer (numpy zeros array) eliminates repeated
        malloc/free cycles that were causing RAM fragmentation.
      - FPS-throttled: only writes to the buffer at TARGET_INGEST_FPS,
        discarding intermediate frames (cap.grab() is cheap, cap.retrieve() is not).
      - Exponential backoff on reconnect instead of a fixed sleep.
      - _frame_ready Event allows consumers to block-wait instead of poll.
    """

    def __init__(self, name: str, url: str):
        self.name   = name
        self.url    = url

        # Pre-allocated shared buffer — write in-place, no per-frame malloc
        self._buf   = np.zeros((_BUF_H, _BUF_W, 3), dtype=np.uint8)
        self._lock  = threading.Lock()
        self._ready = threading.Event()   # set once a valid frame is stored
        self._actual_h = _BUF_H
        self._actual_w = _BUF_W

        self.cap: Optional[cv2.VideoCapture] = None
        self.running    = False
        self._thread: Optional[threading.Thread] = None
        self._reconnect_delay = _RECONNECT_MIN_DELAY

        # Diagnostics
        self.frames_captured  = 0
        self.frames_dropped   = 0
        self.last_frame_time  = 0.0

    # ── Public API ──────────────────────────────────────────────────────────

    def start(self):
        if self.running:
            return
        self.running = True
        self._thread = threading.Thread(
            target=self._capture_loop,
            name=f"rtsp-{self.name}",
            daemon=True,
        )
        self._thread.start()
        log.info("[RTSP:%s] Capture thread started (target=%d FPS).", self.name, TARGET_INGEST_FPS)

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=5)
        self._release_cap()
        log.info("[RTSP:%s] Capture thread stopped.", self.name)

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """
        Returns a copy of the latest frame, or None if no frame yet.
        Callers that only need dimensions can use frame.shape without this call.
        """
        if not self._ready.is_set():
            return None
        with self._lock:
            h, w = self._actual_h, self._actual_w
            return self._buf[:h, :w].copy()

    def get_frame_no_copy(self) -> Optional[np.ndarray]:
        """
        Returns a READ-ONLY view into the shared buffer.
        Caller MUST NOT modify the returned array and MUST hold no reference
        after releasing the internal lock context. Used only for MJPEG streaming.
        Safe because the streaming path only calls cv2.imencode (read-only).
        """
        if not self._ready.is_set():
            return None
        with self._lock:
            h, w = self._actual_h, self._actual_w
            return self._buf[:h, :w]

    # ── Internal ────────────────────────────────────────────────────────────

    def _release_cap(self):
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

    def _open_source(self) -> bool:
        """Open the camera source with optimized RTSP settings."""
        src = self.url
        if isinstance(src, str) and src.isdigit():
            src = int(src)

        if isinstance(src, str) and src.lower().startswith("rtsp"):
            # Base optimized options
            transport     = "tcp"
            fflags        = "nobuffer"
            buffer_size   = "1024000"
            reorder_queue = "128"
            max_delay     = "500000"
            
            # Special tuning for high-resolution 5MP cameras (Kinfra Site)
            is_kinfra_5mp = any(ip in str(src) for ip in [".221", ".225"]) or "Kinfra" in self.name
            
            if is_kinfra_5mp:
                transport     = "tcp"
                fflags        = "genpts" 
                buffer_size   = "67108864" # 64MB
                reorder_queue = "512"
                max_delay     = "1000000"
                log.info("[RTSP:%s] 5MP Stable-Lag-Fix: TCP + 64MB Buffer", self.name)
                
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                    f"rtsp_transport;{transport}|"
                    f"fflags;{fflags}|"
                    f"reorder_queue_size;{reorder_queue}|"
                    f"max_delay;{max_delay}|"
                    f"buffer_size;{buffer_size}|"
                    "probesize;30000000|"
                    "analyzeduration;30000000"
                )
            else:
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                    f"rtsp_transport;{transport}|"
                    f"fflags;{fflags}|"
                    f"reorder_queue_size;{reorder_queue}|"
                    f"max_delay;{max_delay}|"
                    f"buffer_size;{buffer_size}"
                )

            os.environ["OPENCV_FFMPEG_DEBUG"]             = "0"
            os.environ["OPENCV_LOG_LEVEL"]                = "QUIET"
            cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        else:
            cap = cv2.VideoCapture(src)

        if not cap.isOpened():
            cap.release()
            return False

        self.cap = cap
        # CRITICAL: Keep buffer small but not zero to avoid decoder starvation on 5MP
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
        self._reconnect_delay = _RECONNECT_MIN_DELAY  # reset backoff on success
        log.info("[RTSP:%s] Connected successfully.", self.name)
        return True

    def _capture_loop(self):
        """
        The main RTSP reader loop. Runs in a dedicated daemon thread.
        Uses a small buffer and fresh reads to ensure zero latency.
        """
        _frame_interval = 1.0 / TARGET_INGEST_FPS
        last_retrieve_time = 0.0

        while self.running:
            # ── Open / reconnect if needed ──────────────────────────────────
            if self.cap is None or not self.cap.isOpened():
                log.info("[RTSP:%s] Connecting... (retry delay=%.1fs)", self.name, self._reconnect_delay)
                if not self._open_source():
                    log.warning("[RTSP:%s] Failed to open source. Retrying in %.1fs", self.name, self._reconnect_delay)
                    time.sleep(self._reconnect_delay)
                    self._reconnect_delay = min(self._reconnect_delay * 1.5, _RECONNECT_MAX_DELAY)
                continue
            
            cap = self.cap
            if cap is None: continue

            # ── Read LATEST frame ──
            # With CAP_PROP_BUFFERSIZE=1, read() returns the most recent frame.
            ret, frame = cap.read()
            if not ret or frame is None:
                log.warning("[RTSP:%s] Stream read failed — reconnecting.", self.name)
                self._release_cap()
                time.sleep(1.0)
                continue

            now = time.monotonic()

            # ── Ingest to shared buffer at target FPS ─────────────
            if now - last_retrieve_time >= _frame_interval:
                # Write into pre-allocated buffer (in-place, no new allocation)
                h, w = frame.shape[:2]
                with self._lock:
                    if h != self._actual_h or w != self._actual_w:
                        # Resolution change/init — check if larger than max buffer
                        if h > _BUF_H or w > _BUF_W:
                            scale = min(_BUF_H / h, _BUF_W / w)
                            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
                            h, w = frame.shape[:2]
                    self._actual_h = h
                    self._actual_w = w
                    np.copyto(self._buf[:h, :w], frame)

                if not self._ready.is_set():
                    self._ready.set()

                self.frames_captured += 1
                self.last_frame_time = now
                last_retrieve_time = now
            else:
                self.frames_dropped += 1


# ══════════════════════════════════════════════════════════════════════════════
#  2. MonitoringLoop — Vision Pipeline Orchestrator
# ══════════════════════════════════════════════════════════════════════════════

class MonitoringLoop:
    """
    Orchestrates the full vision pipeline for a single camera:
      Frame → (blur check) → ArcFace detection + embedding
            → FAISS search → consensus → RF card → door unlock → DB log

    Changes from v1:
      - MediaPipe REMOVED. ArcFace SCRFD provides bboxes and 5 facial keypoints.
        We use the eye-pair distance from those keypoints as a face-size gate
        (same function as the old MediaPipe face_w/face_h check).
      - Processes at PROCESSING_FPS (5/sec) not at 100/sec.
      - Employee metadata served from database's in-memory cache — zero DB calls
        in the hot path after first encounter.
      - Counter-based consensus — O(n) vs O(n²) for the old list scan.
      - Log spam reduced: only logs per RECOGNITION event, not per frame.
    """

    def __init__(self, vproc: VideoProcessor):
        self.processor  = vproc
        self.name       = vproc.name
        self.running    = False

        # Recognition state
        self.cooldown_dict: Dict[int, float] = {}    # emp_id → next_trigger_time
        self.frame_count  = 0

        # Consensus window: each position holds a set of emp_ids seen that cycle
        self.id_history: deque = deque(maxlen=config.CONSENSUS_WINDOW)

        # Live UI stats (read by gen_frames_async)
        self.last_num_faces    = 0
        self.last_known_names: List[str] = []
        self.last_unknown_count = 0
        self.current_emp_ids:  List[int] = []
        self.last_unknown_log_time = 0.0
        self.last_faces_bboxes = []

        # Motion Detection State
        self.last_gray = None
        self.ai_active_until = 0.0

    # ── Decision Task ───────────────────────────────────────────────────────

    async def _handle_access_batch(self, batch: List[Dict[str, Any]]):
        """Processes a batch of people detected in the same cycle with professional, long-form announcements."""
        try:
            # 1. Run all RF checks in parallel
            rf_tasks = [engine.check_rf_card(p["rf_card"], camera_name=self.name, department=p.get("department", "")) for p in batch]
            rf_results = await asyncio.gather(*rf_tasks)
            
            granted = []
            denied  = []
            
            for i, res in enumerate(rf_results):
                rf_ok, checkin_status, exit_type = res
                p = batch[i].copy()
                p["rf_ok"] = rf_ok
                p["checkin_status"] = checkin_status
                p["exit_type"] = exit_type
                if rf_ok:
                    granted.append(p)
                else:
                    denied.append(p)
            
            # Time-based context
            hour = datetime.datetime.now().hour
            if 5 <= hour < 12:    period = "Good Morning"
            elif 12 <= hour < 17: period = "Good Afternoon"
            else:                 period = "Good Evening"
            
            # 2. Handle Granted (Longer, More Welcoming Announcements)
            is_exit_camera = any(k.lower() in self.name.lower() for k in config.EXIT_CAM_KEYWORDS)
            speaker_id = config.get_cam_setting(self.name, "SPEAKER_DEVICE_IDS")

            if granted:
                names_str = self._format_names([p["name"] for p in granted])
                is_group = len(granted) > 1
                
                if is_exit_camera:
                    # Choose primary exit type from batch (prefer the most descriptive one)
                    primary_exit = granted[0].get("exit_type", "EXIT")
                    
                    if primary_exit == "Tea-Break":
                        msg = f"Enjoy your tea break, , , {names_str}. See you back in a moment."
                    elif primary_exit == "Lunch":
                        msg = f"Enjoy your lunch, , , {names_str}. We hope you have a nice meal."
                    elif primary_exit == "RESTROOM":
                        msg = f"No problem, , , {names_str}. We will see you shortly."
                    else:
                        # Variations for Standard Exit (Safe journey / Thanks)
                        exit_v = [
                            f"Goodbye, , , {names_str}. Have a safe journey home and see you again soon.",
                            f"Thank you for your visit, , , {names_str}. We hope you have a peaceful and relaxing rest of your day.",
                            f"Take care, , , {names_str}. It was a pleasure having you here today. Goodbye."
                        ]
                        msg = random.choice(exit_v)
                else:
                    # Variations for Entrance (Warm welcome / Wishing success)
                    primary_exit = granted[0].get("exit_type", "EXIT")
                    all_str = " all" if is_group else ""
                    
                    if primary_exit == "Tea-Break":
                        msg = f"Welcome back from your tea break, , , {names_str}. Hope you are refreshed."
                    elif primary_exit == "Lunch":
                        msg = f"Welcome back from lunch, , , {names_str}. We hope you enjoyed your meal."
                    elif primary_exit == "RESTROOM":
                        msg = f"Welcome back, , , {names_str}. Glad to see you back."
                    else:
                        ent_v = [
                            f"{period}, , , {names_str}. A very warm welcome to you{all_str}. Wishing you a productive and wonderful day.",
                            f"Good to see you{all_str}, , , {names_str}. We hope you{all_str} have a successful and pleasant day today.",
                            f"Hello, , , {names_str}. Welcome back to the facility. Have a great day ahead."
                        ]
                        msg = random.choice(ent_v)
                
                # 2.2 Execute individual door/logging actions immediately (HIGHEST PRIORITY)
                for p in granted:
                    log.info("[Monitor:%s] ✅ ACCESS GRANTED: '%s'", self.name, p["name"])
                    asyncio.create_task(self._finalize_access(p))

                # 2.3 Follow with grouped announcement (SECOND PRIORITY)
                # ANN-01: Grouped greeting with small pause (,,, or ...) before names
                asyncio.create_task(engine.announce(msg, device_id=speaker_id))

            # 3. Handle Denied (Clear instructions)
            if denied:
                names_str = self._format_names([p["name"] for p in denied])
                
                # 3.1 Log rejection details immediately (HIGHEST PRIORITY)
                for p in denied:
                    log.warning("[Monitor:%s] ❌ ACCESS DENIED (RF invalid): '%s'", self.name, p["name"])
                    asyncio.create_task(database.log_access(
                        employee_id=p["emp_id"],
                        distance=0.0,
                        door_ok=False,
                        door_name=self.name
                    ))

                # 3.2 Follow with grouped announcement (Only for EXIT cameras)
                if is_exit_camera:
                    denied_v = [
                        f"Attention, , , {names_str}. A checkout is required before exiting. Please follow the checkout procedure.",
                        f"Pardon me, , , {names_str}. Exit access cannot be granted as checkout is required. Thank you for your cooperation."
                    ]
                    msg_denied = random.choice(denied_v)
                    speaker_id = config.get_cam_setting(self.name, "SPEAKER_DEVICE_IDS")
                    asyncio.create_task(engine.announce(msg_denied, device_id=speaker_id))

        except Exception as e:
            log.error("[Monitor:%s] Batch access task failed: %s", self.name, e)

    async def _finalize_access(self, p: Dict[str, Any]):
        """Unlocks door and logs for a single person with integrated IN/OUT reporting."""
        # 1. Trigger Door (WS or HTTP)
        door_ok = await engine.unlock_door(p["name"], employee_code=p["emp_code"], camera_name=self.name)
        
        # Determine if this is an Entrance or Exit context
        is_exit_camera = any(k.lower() in self.name.lower() for k in config.EXIT_CAM_KEYWORDS)
        is_exit_event  = p.get("exit_type") == "EXIT" or is_exit_camera
        
        # 2. Secondary Actions (Only if door opened successfully)
        if door_ok:
            # 2.1 GrapesOnline Attendance Logging (Only if NOT in API door mode)
            if config.DOOR_UNLOCK_MODE != "API":
                if is_exit_event:
                    asyncio.create_task(engine.log_exit(p["emp_code"]))
                else:
                    if p.get("checkin_status") in ("RdytoChkIn", "OUT"):
                        asyncio.create_task(engine.log_entry(p["emp_code"]))
            
            # 2.2 PC Automation (Wake-on-LAN / Lock / Shutdown)
            if config.PC_CONTROL_ENABLED and p.get("pc_control"):
                if is_exit_event:
                    if p.get("pc_ip"):
                        now_hour = datetime.datetime.now().hour
                        is_office_hour = config.PC_OFFICE_HOURS_START <= now_hour < config.PC_OFFICE_HOURS_END
                        exit_type = str(p.get("exit_type", "")).upper()
                        if exit_type == "EXIT" and not is_office_hour:
                            asyncio.create_task(engine.trigger_pc_stop(p["pc_ip"]))
                        else:
                            asyncio.create_task(engine.trigger_pc_lock(p["pc_ip"]))
                else:
                    if p.get("pc_mac"):
                        asyncio.create_task(engine.trigger_pc_start(p["pc_mac"]))

            # 2.3 SELF-IMPROVING IDENTITY: If confidence is high, update the employee's model
            if config.AUTO_UPDATE_ENABLED and p.get("score", 0) >= config.AUTO_UPDATE_THRESHOLD:
                if p.get("emp_id") and p.get("embedding") is not None:
                    # Trigger optimization in the background
                    asyncio.create_task(engine.auto_optimize_identity(
                        emp_id=p["emp_id"],
                        name=p["name"],
                        new_embedding=p["embedding"]
                    ))
        else:
            log.error("[Monitor:%s] ❌ Secondary actions skipped for '%s' (Door unlock failed).", self.name, p["name"])
        
        # 3. System Activity Logging (Local SQL audit)
        await database.log_access(
            employee_id=p["emp_id"],
            distance=0.0,
            door_ok=door_ok,
            door_name=self.name
        )

    @staticmethod
    def _format_names(names: List[str]) -> str:
        """Helper to format ['A', 'B', 'C'] into 'A, B, and C'."""
        if not names: return ""
        if len(names) == 1: return names[0]
        if len(names) == 2: return f"{names[0]} and {names[1]}"
        return ", ".join(names[:-1]) + ", and " + names[-1]

    # ── Lifecycle ───────────────────────────────────────────────────────────

    async def start(self):
        if not config.MONITOR_ENABLED:
            log.info("[Monitor:%s] Monitoring disabled in config.", self.name)
            return

        log.info("[Monitor:%s] Vision pipeline starting.", self.name)
        self.running = True
        self.processor.start()

        while self.running:
            t0 = asyncio.get_event_loop().time()
            try:
                await self._process_cycle()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.error("[Monitor:%s] Unhandled error: %s", self.name, exc, exc_info=True)

            # Precise FPS governor — sleep exactly enough to hit PROCESSING_FPS
            elapsed = asyncio.get_event_loop().time() - t0
            sleep_t = max(0.0, _PROCESS_INTERVAL - elapsed)
            await asyncio.sleep(sleep_t)

    def stop(self):
        self.running = False
        self.processor.stop()

    # ── Helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _is_blurry(frame: np.ndarray) -> bool:
        """Laplacian variance blur filter. Run on the incoming frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var() < config.BLUR_THRESHOLD

    def _face_too_small(self, face) -> bool:
        """
        Use ArcFace bounding box to check face size — replaces MediaPipe face_w check.
        face.bbox = [x1, y1, x2, y2] in pixel coords of the frame.
        Handles group entries (5+ people) by allowing smaller faces further away.
        """
        x1, y1, x2, y2 = face.bbox
        face_w = (x2 - x1)
        face_h = (y2 - y1)
        
        # Look up per-camera or per-group face size threshold
        min_size = int(config.get_cam_setting(self.name, "FACE_MIN_SIZE", config.FACE_MIN_SIZE))
        return face_w < min_size or face_h < min_size

    def _is_outside_roi(self, face_bbox, frame_shape) -> bool:
        """Checks if the face center point is within the allowed ROI (defined in % of frame)."""
        roi_raw = config.get_cam_setting(self.name, "ROI")
        if not roi_raw:
            return False
        
        try:
            h, w = frame_shape[:2]
            # ROI format: y1,x1,y2,x2 (all in 0-100 percent)
            ry1, rx1, ry2, rx2 = map(float, [x.strip() for x in roi_raw.split(",")])
            
            # Face center point
            fx1, fy1, fx2, fy2 = face_bbox
            cx = (fx1 + fx2) / 2 / w * 100
            cy = (fy1 + fy2) / 2 / h * 100
            
            # Return True if outside
            return not (rx1 <= cx <= rx2 and ry1 <= cy <= ry2)
        except Exception as e:
            log.warning("[Monitor:%s] ROI parse error: %s", self.name, e)
            return False

    def _get_face_crop(self, frame: np.ndarray, bbox: list, padding: int = 40) -> Optional[bytes]:
        """Extracts a padded face crop from the frame and encodes as JPEG."""
        try:
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = map(int, bbox)
            # Add padding
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0: return None
            
            # Resize if too large to save DB space
            if crop.shape[1] > 200:
                scale = 200 / crop.shape[1]
                crop = cv2.resize(crop, (200, int(crop.shape[0] * scale)))
                
            ret, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 70])
            return buf.tobytes() if ret else None
        except Exception:
            return None

    # ── Main pipeline cycle ─────────────────────────────────────────────────

    async def _process_cycle(self):
        """One recognition cycle. Called at PROCESSING_FPS rate."""
        frame = self.processor.get_latest_frame()
        if frame is None:
            return

        current_time = time.time()

        # ── Motion Detection Gate ───────────────────────────────────────────
        if config.MOTION_DETECTION_ENABLED:
            # Resize aggressively for extremely cheap motion detection
            small = cv2.resize(frame, (160, 120))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            
            if self.last_gray is None:
                self.last_gray = gray
                return  # Skip first frame to prime the baseline
                
            frame_diff = cv2.absdiff(self.last_gray, gray)
            self.last_gray = gray
            
            # Count pixels that changed significantly
            _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
            changed_pixels = cv2.countNonZero(thresh)
            diff_percent = (changed_pixels / 19200.0) * 100.0

            if diff_percent > config.MOTION_THRESHOLD:
                self.ai_active_until = current_time + config.MOTION_SLEEP_TIME

            if current_time > self.ai_active_until:
                # System is asleep, clear AI UI overlays and history
                self.last_num_faces = 0
                self.last_known_names = []
                self.last_unknown_count = 0
                self.last_faces_bboxes = []
                self.id_history.append(set())
                self.current_emp_ids = []
                # Keep yielding from generator but do not run ArcFace
                return
        # ────────────────────────────────────────────────────────────────────

        # Blur gate — skip motion-blurred frames (cheap Laplacian check)
        if self._is_blurry(frame):
            # Clear UI labels during motion blur to prevent ghosting
            self.last_faces_bboxes = []
            return

        # Run ArcFace detection + embedding (offloaded to thread pool, non-blocking)
        # Using full frame (instead of 'small') for better detection of distant faces in groupings
        face_results = await engine.extract_faces_full(frame)

        now = time.time()

        if not face_results:
            # No faces in frame — clear all stats to purge UI labels
            self.last_num_faces     = 0
            self.last_known_names   = []
            self.last_unknown_count = 0
            self.last_faces_bboxes  = []      # CRITICAL: Clear the labels
            self.id_history.append(set())
            self.current_emp_ids    = []
            return

        # Filter undersized faces (person too far from camera) or faces outside ROI
        face_results = [
            f for f in face_results 
            if not self._face_too_small(f['face']) and not self._is_outside_roi(f['face'].bbox, frame.shape)
        ]
        if not face_results:
            # All detected faces were filtered out — clear UI labels
            self.last_num_faces     = 0
            self.last_known_names   = []
            self.last_unknown_count = 0
            self.last_faces_bboxes  = []      # CRITICAL: Clear the labels
            self.id_history.append(set())
            return

        # FAISS batch search — all faces in one vectorised call
        embeddings    = np.array([f['embedding'] for f in face_results], dtype=np.float32)
        search_results = engine.search_index_multi(embeddings)

        # ── Build frame-level stats & Log Audit Snapshots ───────────────────
        known_names      = []
        unknown_count    = 0
        current_frame_ids = set()
        bboxes_info      = []

        threshold = config.FAISS_COSINE_THRESHOLD
        margin    = 0.08  # mark as ambiguous if within this range of threshold

        for i, (emp_id, score) in enumerate(search_results):
            bbox = face_results[i]['bbox']
            is_ambiguous = (threshold - margin) <= score <= (threshold + margin)
            
            if emp_id is not None:
                current_frame_ids.add(emp_id)
                emp_data = database._employee_cache.get(emp_id)
                if emp_data is None:
                    emp_data = await database.get_employee_by_id(emp_id)
                name = emp_data["name"] if emp_data else f"ID_{emp_id}"
                known_names.append(name)
                bboxes_info.append({'bbox': bbox, 'name': name, 'color': (0, 255, 0)})
                
                # Log audit snapshot for recognized faces (fire and forget)
                # We log it every cycle where confirmed, or slightly rate-limited
                # Here we just log it if it's potentially a door trigger cycle (confirmed ids check later)
            else:
                unknown_count += 1
                bboxes_info.append({'bbox': bbox, 'name': 'Unknown', 'color': (0, 0, 255)})
                
                # Log unknown face periodically to audit false positives
                if (now - self.last_unknown_log_time) > 30.0:
                    crop = self._get_face_crop(frame, bbox)
                    if crop:
                        asyncio.create_task(database.log_audit_snapshot(
                            employee_id=None, name="Unknown", camera=self.name,
                            score=score, granted=False, is_ambiguous=False, image_bytes=crop
                        ))

        # Update UI stats
        self.last_num_faces     = len(face_results)
        self.last_known_names   = known_names
        self.last_unknown_count = unknown_count
        self.last_faces_bboxes  = bboxes_info

        # ── Consensus: require N of last M frames to agree ───────────────────
        self.id_history.append(current_frame_ids)

        # Flatten history into a flat list of all seen IDs
        all_recent = [eid for frame_set in self.id_history for eid in frame_set]

        # Counter is O(n) — replaces the old O(n²) nested list.count() loop
        id_counts     = Counter(all_recent)
        confirmed_ids = [
            eid for eid, cnt in id_counts.items()
            if cnt >= config.CONSENSUS_THRESHOLD
        ]
        self.current_emp_ids = confirmed_ids

        if confirmed_ids:
            log.debug(
                "[Monitor:%s] Consensus confirmed: %s (history=%d frames)",
                self.name, confirmed_ids, len(self.id_history)
            )

        # ── Logging ──────────────────────────────────────────────────────────
        # Log known people immediately when they appear in the scene
        if known_names:
            log.info(
                "[Monitor:%s] Scene: %d face(s) | Known: %s",
                self.name,
                len(face_results),
                ", ".join(known_names)
            )

        # Log unknown people only once every 30 seconds to prevent log spam
        if unknown_count > 0:
            if (now - self.last_unknown_log_time) > 30.0:
                log.info(
                    "[Monitor:%s] Scene: %d unknown face(s) detected.",
                    self.name,
                    unknown_count
                )
                self.last_unknown_log_time = now

        # ── Decision loop: trigger door for current confirmed batch ───────────
        batch = []
        for emp_id in confirmed_ids:
            # Cooldown gate — prevent repeated triggers for same person
            if now < self.cooldown_dict.get(emp_id, 0.0):
                continue

            emp_data = database._employee_cache.get(emp_id)
            if emp_data is None:
                emp_data = await database.get_employee_by_id(emp_id)
            if emp_data is None:
                continue

            name     = emp_data["name"]
            emp_code = emp_data.get("employee_code", "")
            rf_card  = emp_data.get("rf_card", "")

            # Set cooldown IMMEDIATELY to prevent spawning multiple concurrent tasks
            self.cooldown_dict[emp_id] = now + config.MONITOR_COOLDOWN
            
            # Find the best result for this employee in the current frame to use for potential auto-update
            best_idx = -1
            max_score = -1.0
            for j, (eid, s) in enumerate(search_results):
                if eid == emp_id and s > max_score:
                    max_score = s
                    best_idx = j

            batch.append({
                "emp_id": emp_id,
                "name": name,
                "emp_code": emp_code,
                "rf_card": rf_card,
                "pc_mac": emp_data.get("pc_mac"),
                "pc_ip": emp_data.get("pc_ip"),
                "pc_control": emp_data.get("pc_control"),
                "score": max_score,
                "embedding": face_results[best_idx]['embedding'] if best_idx != -1 else None,
                "bbox": face_results[best_idx]['bbox'] if best_idx != -1 else None,
                "department": emp_data.get("department", "")
            })

        if batch:
            # Capture snapshots for the confirmed batch before launching task
            for p in batch:
                if p["bbox"]:
                    crop = self._get_face_crop(frame, p["bbox"])
                    if crop:
                        # Find if it was ambiguous
                        score = p["score"]
                        is_ambiguous = (config.FAISS_COSINE_THRESHOLD - 0.08) <= score <= (config.FAISS_COSINE_THRESHOLD + 0.08)
                        asyncio.create_task(database.log_audit_snapshot(
                            employee_id=p["emp_id"], name=p["name"], camera=self.name,
                            score=p["score"], granted=True, is_ambiguous=is_ambiguous, image_bytes=crop
                        ))

            # Spawn batch task — vision pipeline continues immediately
            asyncio.create_task(
                self._handle_access_batch(batch),
                name=f"access-batch-{now}"
            )


# ══════════════════════════════════════════════════════════════════════════════
#  3. Global registry & lifecycle
# ══════════════════════════════════════════════════════════════════════════════

_monitors: List[MonitoringLoop] = []


async def start_background_monitoring():
    global _monitors
    if _monitors:
        return  # already started

    for cam_name, cam_url in config.RTSP_CAMERAS.items():
        is_enabled = config.ENABLED_CAMERAS.get(cam_name, config.MONITOR_ENABLED)
        if not is_enabled:
            log.info("[Processor] Camera '%s' DISABLED — skipping.", cam_name)
            continue

        vproc = VideoProcessor(cam_name, cam_url)
        ml    = MonitoringLoop(vproc)
        _monitors.append(ml)
        asyncio.create_task(ml.start(), name=f"monitor-{cam_name}")

    log.info("[Processor] Initialised %d active camera(s).", len(_monitors))


def stop_background_monitoring():
    global _monitors
    for ml in _monitors:
        ml.stop()
    _monitors = []
    log.info("[Processor] All camera monitors stopped.")
