"""
processor.py – Persistent RTSP monitoring, frame buffering, and AI pipeline orchestration with Blink Liveness.
Supports multiple cameras.
"""

import asyncio
import logging
import os
import threading
import time
from typing import Optional, Dict, List

import cv2
import numpy as np
import mediapipe as mp

import config
import database
import engine

log = logging.getLogger("processor")

# ══════════════════════════════════════════════════════════════════════════════
#  1. VideoProcessor: Manages the RTSP ingestion thread.
# ══════════════════════════════════════════════════════════════════════════════

class VideoProcessor:
    """
    Dedicated class to handle cv2.VideoCapture in a background thread.
    Maintains a single-frame buffer to avoid processing stale data.
    """
    def __init__(self, name: str, url: str):
        self.name = name
        self.url = url
        self.cap: Optional[cv2.VideoCapture] = None
        self.latest_frame: Optional[np.ndarray] = None
        self.lock = threading.Lock()
        self.running = False
        self.thread: Optional[threading.Thread] = None

    def start(self):
        """Invoke the background thread."""
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        log.info("[Processor:%s] RTSP capture thread started.", self.name)

    def stop(self):
        """Stop the capture thread and release resources."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        if self.cap:
            self.cap.release()
        log.info("[Processor:%s] RTSP capture thread stopped.", self.name)

    def _capture_loop(self):
        """Continuously reads frames into the shared 'latest_frame' buffer."""
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                camera_source = self.url
                if isinstance(self.url, str) and self.url.isdigit():
                    camera_source = int(self.url)
                
                log.info("[Processor:%s] Opening camera source: %s", self.name, camera_source)
                
                if isinstance(camera_source, str) and camera_source.startswith("rtsp"):
                    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
                    self.cap = cv2.VideoCapture(camera_source, cv2.CAP_FFMPEG)
                else:
                    self.cap = cv2.VideoCapture(camera_source)

                if self.cap.isOpened():
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    log.info("[Processor:%s] Camera source opened successfully.", self.name)
                else:
                    log.warning("[Processor:%s] FAILED to open. Retrying in %d seconds...", self.name, config.RTSP_RECONNECT_DELAY)
                    self.cap = None
                    time.sleep(config.RTSP_RECONNECT_DELAY)
                    continue

            ret, frame = self.cap.read()
            if not ret:
                log.warning("[Processor:%s] RTSP read failed. Reconnecting...", self.name)
                if self.cap:
                    self.cap.release()
                self.cap = None
                time.sleep(1)
                continue

            with self.lock:
                self.latest_frame = frame

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Thread-safe retrieval of the latest captured frame."""
        with self.lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
            return None


# ══════════════════════════════════════════════════════════════════════════════
#  2. MonitoringLoop: Orchestrates the vision pipeline.
# ══════════════════════════════════════════════════════════════════════════════

class MonitoringLoop:
    """
    Consumes frames from VideoProcessor and performs:
    Detection -> Embedding -> FAISS Search -> Liveness (Blink) -> Door Trigger
    """
    def __init__(self, processor: VideoProcessor):
        self.processor = processor
        self.name = processor.name
        self.running = False
        self.cooldown_dict: Dict[int, float] = {}
        self.frame_count = 0
        
        # Liveness tracking
        self.last_blink_time = 0.0
        self.current_emp_id: Optional[int] = None
        
        # MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )
        self.id_history: List[int] = [] 
        self._current_ear: Optional[float] = None

    def _calculate_ear(self, landmarks, eye_indices):
        """Calculate Eye Aspect Ratio (EAR)."""
        p1 = np.array([landmarks[eye_indices[0]].x, landmarks[eye_indices[0]].y])
        p4 = np.array([landmarks[eye_indices[3]].x, landmarks[eye_indices[3]].y])
        p2 = np.array([landmarks[eye_indices[1]].x, landmarks[eye_indices[1]].y])
        p6 = np.array([landmarks[eye_indices[5]].x, landmarks[eye_indices[5]].y])
        p3 = np.array([landmarks[eye_indices[2]].x, landmarks[eye_indices[2]].y])
        p5 = np.array([landmarks[eye_indices[4]].x, landmarks[eye_indices[4]].y])
        
        ear = (np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (2.0 * np.linalg.norm(p1 - p4))
        return ear

    async def start(self):
        if not config.MONITOR_ENABLED:
            return

        log.info("[Monitor:%s] Starting live vision pipeline...", self.name)
        self.running = True
        self.processor.start()

        while self.running:
            try:
                await self._process_cycle()
                await asyncio.sleep(0.02)
            except Exception as exc:
                log.error("[Monitor:%s] Error: %s", self.name, exc, exc_info=True)
                await asyncio.sleep(1)

    def stop(self):
        self.running = False
        self.processor.stop()

    async def _process_cycle(self):
        frame = self.processor.get_latest_frame()
        if frame is None:
            return

        now = time.time()

        # 1. Liveness
        if config.LIVENESS_ENABLED:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self.mp_face_mesh.process(rgb)
            if res.multi_face_landmarks:
                landmarks = res.multi_face_landmarks[0].landmark
                LEFT_EYE = [362, 385, 387, 263, 373, 380]
                RIGHT_EYE = [33, 160, 158, 133, 153, 144]
                ear = (self._calculate_ear(landmarks, LEFT_EYE) + self._calculate_ear(landmarks, RIGHT_EYE)) / 2.0
                self._current_ear = ear
                
                if ear < config.BLINK_EAR_THRESHOLD:
                    self.last_blink_time = now

        # 2. Identity
        self.frame_count += 1
        if self.frame_count % config.MONITOR_N_FRAMES == 0:
            emb = await engine.extract_embedding(frame)
            if emb is not None:
                emp_id, dist = engine.search_index(emb)
                
                if emp_id is not None:
                    self.id_history.append(emp_id)
                    if len(self.id_history) > 3:
                        self.id_history.pop(0)
                    
                    counts = {}
                    for x in self.id_history:
                        counts[x] = counts.get(x, 0) + 1
                    
                    best_id = max(counts, key=counts.get) if counts else None
                    if best_id is not None and counts.get(best_id, 0) >= 2:
                        self.current_emp_id = best_id
                    else:
                        self.current_emp_id = None
                else:
                    if self.frame_count % 30 == 0:
                        log.info("[Monitor:%s] Unknown face detected (Distance: %.3f)", self.name, dist)
                    self.id_history = []
                    self.current_emp_id = None
            else:
                self.current_emp_id = None

        # 3. Decision Logic
        if self.current_emp_id is not None:
            emp_id: int = self.current_emp_id
            
            if now < self.cooldown_dict.get(emp_id, 0.0):
                return

            is_alive = not config.LIVENESS_ENABLED or (now - self.last_blink_time < 5.0)

            if is_alive:
                employee = await database.get_employee_by_id(emp_id)
                name = employee["name"] if employee else f"ID_{emp_id}"
                
                log.info("[Monitor:%s] ACCESS GRANTED: '%s' (Blink Verified)", self.name, name)
                emp_code = employee.get("employee_code", "") if employee else ""
                
                # Pass camera name to engine for door-specific logic
                door_ok = await engine.unlock_door(name, employee_code=emp_code, camera_name=self.name)
                await database.log_access(employee_id=emp_id, distance=0.0, door_ok=door_ok, door_name=self.name)
                
                self.cooldown_dict[emp_id] = now + config.MONITOR_COOLDOWN
                self.current_emp_id = None 
            else:
                if self.frame_count % 30 == 0:
                    employee = await database.get_employee_by_id(emp_id)
                    name = employee["name"] if employee else f"ID_{emp_id}"
                    msg = f"Current EAR: {self._current_ear:.3f}" if self._current_ear else "Liveness tracing..."
                    log.info("[Monitor:%s] Identity confirmed: '%s', waiting for blink...", self.name, name)

# Global trackers
_monitors: List[MonitoringLoop] = []

async def start_background_monitoring():
    global _monitors
    if not _monitors:
        for cam_name, cam_url in config.RTSP_CAMERAS.items():
            proc = VideoProcessor(cam_name, cam_url)
            ml = MonitoringLoop(proc)
            _monitors.append(ml)
            asyncio.create_task(ml.start())
        log.info("[Processor] Initialised %d camera(s).", len(_monitors))

def stop_background_monitoring():
    global _monitors
    for ml in _monitors:
        ml.stop()
    _monitors = []
