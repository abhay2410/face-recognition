"""
config.py – Centralised Configuration  [v2.1 — Tuned for 100 employees]
========================================================================
Tuning rationale for ~100 employee scale:
  - FAISS HNSW: M=48, efSearch=128 → near-perfect recall for 300 vectors
  - Multi-embedding: 3 diverse embeddings per person → covers angles + lighting
  - Cosine threshold: 0.50 → ArcFace sweet spot for high accuracy at this scale
  - Onboard frames: 20 → richer mean + better diversity selection
  - Consensus: 5/7 frames → faster trigger while still preventing false positives
  - Blur threshold: 80 → stricter quality gate for clean embeddings
"""

"""
config.py – Global Configuration Management [v3 — Multi-Office Architecture]
========================================================================
This module handles all environment-based configuration for the Face Access 
System. It supports a hierarchical configuration model allowing multiple 
offices (e.g., DEV, KINFRA) to share global defaults while overriding 
specific hardware, security, and integration settings.

Design Patterns:
  - Hierarchical Resolution: Camera-specific > Group-specific > Global Default.
  - Environment Injection: All settings are loaded from a central .env file.
  - Mapping: RTSP URLs, Speakers, and Door APIs are indexed by logical camera names.
"""

import logging
import os
import sys
from dotenv import load_dotenv

# ── Base directory ────────────────────────────────────────────────────────────

if getattr(sys, "frozen", False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

env_path = os.path.join(BASE_DIR, ".env")
load_dotenv(env_path)

# ── MS SQL Server ─────────────────────────────────────────────────────────────

MSSQL_SERVER     = os.getenv("MSSQL_SERVER",   "192.168.0.251,1433")
MSSQL_USER       = os.getenv("MSSQL_USER",     "sa")
MSSQL_PASSWORD   = os.getenv("MSSQL_PASSWORD", "sa@123")
MSSQL_DB         = os.getenv("MSSQL_DB",       "face_attendance")
MSSQL_DRIVER     = os.getenv("MSSQL_DRIVER",   "ODBC Driver 18 for SQL Server")
MSSQL_TRUST_CERT = os.getenv("MSSQL_TRUST_CERT", "yes")

# ── Face Recognition ──────────────────────────────────────────────────────────

EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "512"))

# Detection resolution:
#   MONITOR = live camera — full 640×640 for reliable group recognition
#   ENROL   = enrollment & /access API — full 640×640 for maximum quality
#   Unified engine now shares a single (640, 640) model to save VRAM.
ARC_FACE_DET_SIZE_MONITOR = (640, 640)
ARC_FACE_DET_SIZE_ENROL   = (640, 640)

# ── FAISS / Matching ──────────────────────────────────────────────────────────
#
# For ArcFace w600k_r50 (512-D cosine):
#   < 0.20  = definitely different people
#   0.40    = weak match
#   0.50    = solid match (good for 100 people with multi-embedding)
#   0.60    = strict match (use if getting false positives)
#   > 0.70  = very strict (may cause false rejections)
#
# With 3 diverse embeddings per person the system can tolerate a slightly
# higher threshold because angular variance is already captured in the index.

FAISS_COSINE_THRESHOLD = float(os.getenv("FAISS_COSINE_THRESHOLD", "0.50"))

# ── HNSW Index Parameters ─────────────────────────────────────────────────────
#
# For ~300 vectors (100 people × 3 embeddings):
#   M=48           → rich graph connectivity → ~99.9% recall
#   EF_SEARCH=128  → examines 128 candidates per query → near-exact
#   EF_CONSTRUCT=400 → superb index quality at build time
#
# Search speed at this scale (300 vectors): < 0.1ms — effectively free.
# Recall at these settings: indistinguishable from brute-force.

HNSW_M           = int(os.getenv("HNSW_M",           "48"))
HNSW_EF_SEARCH   = int(os.getenv("HNSW_EF_SEARCH",   "128"))
HNSW_EF_CONSTRUCT= int(os.getenv("HNSW_EF_CONSTRUCT","400"))

# ── Multi-Embedding Per Person ─────────────────────────────────────────────────
#
# Store MULTI_EMB_COUNT diverse anchor embeddings per person in FAISS.
# "Diverse" = greedy farthest-point selection from enrollment frames.
# This covers: different angles, expressions, lighting conditions.
#
# Effect: 100 people × 3 embeddings = 300 FAISS vectors total.
# Benefit: Dramatically reduces false rejections (person recognized from side,
#          glasses on, different lighting, etc.)
# No schema breaking change — stored in a separate DB column.

MULTI_EMB_COUNT = int(os.getenv("MULTI_EMB_COUNT", "3"))

# ── Enrollment ────────────────────────────────────────────────────────────────

# Capture 20 frames during enrollment (up from 8).
# More frames = better diversity selection + more robust mean embedding.
ONBOARD_FRAMES = int(os.getenv("ONBOARD_FRAMES", "20"))

# ── Audio Announcements ───────────────────────────────────────────────────────
# AUDIO_MODE: "API" (Network Speaker via HTTP POST) 
AUDIO_MODE           = os.getenv("AUDIO_MODE", "API").upper()

# ── External Door API ─────────────────────────────────────────────────────────
# DOOR_UNLOCK_MODE: "AUTO" (Detects URL), "HTTP", or "WEBSOCKET"
DOOR_UNLOCK_MODE     = os.getenv("DOOR_UNLOCK_MODE", "AUTO").upper()

EXTERNAL_API_ENABLED = os.getenv("EXTERNAL_API_ENABLED", "true").lower() == "true"
EXTERNAL_API_TIMEOUT = int(os.getenv("EXTERNAL_API_TIMEOUT", "10"))
RF_CHECK_API_URL     = os.getenv("RF_CHECK_API_URL", "") # Uses HTTP GET
SPEAKER_API_URL      = os.getenv("SPEAKER_API_URL", "")  # Uses HTTP POST
SPEAKER_DEVICE_ID    = os.getenv("SPEAKER_DEVICE_ID", "G-5889-6B1B-5AE0")
SPEAKER_DEVICE_IDS: dict = {}

# Parse per-camera speaker IDs (CamName:SpeakerID,...)
_SPEAKER_RAW = os.getenv("SPEAKER_DEVICE_IDS", "")
if _SPEAKER_RAW:
    for _entry in _SPEAKER_RAW.split(","):
        if ":" in _entry:
            _name, _sid = _entry.split(":", 1)
            SPEAKER_DEVICE_IDS[_name.strip()] = _sid.strip()

LOG_ENTRY_API_URL    = os.getenv("LOG_ENTRY_API_URL", "") # Uses HTTP POST
LOG_EXIT_API_URL     = os.getenv("LOG_EXIT_API_URL", "")  # Uses HTTP POST
DEVICE_MAC_ADDRESS   = os.getenv("DEVICE_MAC_ADDRESS", "")

# Remote Door Control
BRANCH_ID            = os.getenv("BRANCH_ID", "2")
REMOTE_DOOR_API_URL  = os.getenv("REMOTE_DOOR_API_URL", "")
 
# --- Remote PC Control (OFF BY DEFAULT) ---
PC_CONTROL_ENABLED   = os.getenv("PC_CONTROL_ENABLED", "false").lower() == "true"
PC_OFFICE_HOURS_START = int(os.getenv("PC_OFFICE_HOURS_START", "9"))
PC_OFFICE_HOURS_END   = int(os.getenv("PC_OFFICE_HOURS_END",   "18"))

_DEFAULT_DOOR_URL = os.getenv("EXTERNAL_API_URL", "")
EXTERNAL_API_URLS: dict = {}

if _DEFAULT_DOOR_URL:
    EXTERNAL_API_URLS = {
        "Exit":     _DEFAULT_DOOR_URL,
        "Entrance": _DEFAULT_DOOR_URL,
        "Default":  _DEFAULT_DOOR_URL,
    }

_DOOR_RAW = os.getenv("EXTERNAL_API_URLS", "")
if _DOOR_RAW:
    for _entry in _DOOR_RAW.split(","):
        if ":" in _entry:
            _name, _url = _entry.split(":", 1)
            EXTERNAL_API_URLS[_name.strip()] = _url.strip() # Uses HTTP POST or WebSocket

# ── Live Monitoring (RTSP) ────────────────────────────────────────────────────

_RTSP_RAW = os.getenv("RTSP_URLS", "")
if _RTSP_RAW:
    RTSP_CAMERAS: dict = {}
    for _entry in _RTSP_RAW.split(","):
        if ":" in _entry:
            _name, _url = _entry.split(":", 1)
            RTSP_CAMERAS[_name.strip()] = _url.strip()
else:
    _SINGLE_URL  = os.getenv("RTSP_URL", "rtsp://test:admin123@192.168.1.213:554/stream")
    RTSP_CAMERAS = {"Exit": _SINGLE_URL}

# ── Office / Group Management ──────────────────────────────────────────────────
OFFICE_GROUPS: dict = {}
_GROUPS_RAW = os.getenv("OFFICE_GROUPS", "")
if _GROUPS_RAW:
    for _entry in _GROUPS_RAW.split(","):
        if ":" in _entry:
            _gname, _cams = _entry.split(":", 1)
            _gname = _gname.strip().upper()
            OFFICE_GROUPS[_gname] = [c.strip() for c in _cams.split("|")]

def get_cam_group(cam_name: str) -> str:
    """Find which office group a camera belongs to."""
    for gname, cams in OFFICE_GROUPS.items():
        if cam_name in cams:
            return gname
    return "GLOBAL"

def get_cam_setting(cam_name: str, key: str, default=None):
    """
    Hierarchical lookup for camera settings:
    1. Dictionary-specific (e.g. if key is 'SPEAKER_DEVICE_IDS', check cam_name in that dict)
    2. Camera-specific ENV override (e.g. DEV_EXIT_BRANCH_ID)
    3. Group-specific ENV override (e.g. DEV_BRANCH_ID)
    4. Global default ENV
    """
    gname = get_cam_group(cam_name)
    
    # Check if the key refers to one of our pre-parsed dicts (SPEAKER_DEVICE_IDS, etc.)
    # We allow group-level fallback within these dicts
    dict_map = globals().get(key)
    if isinstance(dict_map, dict):
        # 1a. Try Camera specific
        if cam_name in dict_map:
            return dict_map[cam_name]
        # 1b. Try Group specific (e.g. look for "DEV" in SPEAKER_DEVICE_IDS)
        if gname in dict_map:
            return dict_map[gname]

    # 2. Check Group-specific ENV (e.g. KINFRA_BRANCH_ID)
    g_key = f"{gname}_{key.upper()}"
    val = os.getenv(g_key)
    if val is not None:
        return val

    # 3. Fallback to global ENV or hardcoded default
    return os.getenv(key, default)
    
# ── Performance Tuning ────────────────────────────────────────────────────────
# Capture at this rate regardless of camera native FPS
TARGET_INGEST_FPS = int(os.getenv("TARGET_INGEST_FPS", "10"))

# Run AI inference at this rate (Reduce to 5 for slower CPUs)
PROCESSING_FPS    = int(os.getenv("PROCESSING_FPS", "10"))

# Dashboard MJPEG stream FPS
STREAM_FPS        = int(os.getenv("STREAM_FPS", "10"))

RTSP_RECONNECT_DELAY = int(os.getenv("RTSP_RECONNECT_DELAY", "5"))
MONITOR_COOLDOWN     = int(os.getenv("MONITOR_COOLDOWN",     "10"))
MONITOR_ENABLED      = os.getenv("MONITOR_ENABLED",  "true").lower() == "true"

ENABLED_CAMERAS: dict = {}
_ENABLED_RAW    = os.getenv("CAMERAS_ENABLED", "")
if _ENABLED_RAW:
    for _entry in _ENABLED_RAW.split(","):
        if ":" in _entry:
            _name, _state = _entry.split(":", 1)
            ENABLED_CAMERAS[_name.strip()] = _state.strip().lower() == "true"

EXIT_CAM_KEYWORDS = [k.strip() for k in os.getenv("EXIT_CAM_KEYWORDS", "Exit,OUT").split(",")]

# ── ML / Accuracy Tuning ──────────────────────────────────────────────────────
#
# For 100-employee production scale:
#   FACE_MIN_SIZE=80   → accept face from further away (more realistic door distance)
#   BLUR_THRESHOLD=80  → stricter blur gate — only process sharp frames
#   CONSENSUS_WINDOW=7 → last 7 processed frames
#   CONSENSUS_THRESHOLD=5 → must appear in 5 of 7 = 71% (was 80% but now 7 frames)
#     This gives FASTER trigger (extra 2 frames buffer) while keeping high confidence.

FACE_MIN_SIZE       = int(os.getenv("FACE_MIN_SIZE",       "80"))
BLUR_THRESHOLD      = float(os.getenv("BLUR_THRESHOLD",    "80.0"))
CONSENSUS_WINDOW    = int(os.getenv("CONSENSUS_WINDOW",    "7"))
CONSENSUS_THRESHOLD = int(os.getenv("CONSENSUS_THRESHOLD", "5"))

# ── Self-Improving Identity ───────────────────────────────────────────────────
#
# If enabled, the system will automatically update an employee's face model
# if it detects them with extremely high confidence (> AUTO_UPDATE_THRESHOLD).
# This helps the system adapt to beard growth, new glasses, or aging.
AUTO_UPDATE_ENABLED   = os.getenv("AUTO_UPDATE_ENABLED", "true").lower() == "true"
AUTO_UPDATE_THRESHOLD = float(os.getenv("AUTO_UPDATE_THRESHOLD", "0.65"))

# ── Motion Detection (AI Sleep Mode) ──────────────────────────────────────────

MOTION_DETECTION_ENABLED = os.getenv("MOTION_DETECTION_ENABLED", "false").lower() == "true"
MOTION_THRESHOLD         = float(os.getenv("MOTION_THRESHOLD", "5.0"))
MOTION_SLEEP_TIME        = float(os.getenv("MOTION_SLEEP_TIME", "5.0"))


# ── FastAPI / Uvicorn ─────────────────────────────────────────────────────────

AUTH_USERNAME = os.getenv("AUTH_USERNAME", "admin")
AUTH_PASSWORD = os.getenv("AUTH_PASSWORD", "admin@123")
API_HOST      = os.getenv("API_HOST",      "0.0.0.0")
API_PORT      = int(os.getenv("API_PORT",  "8000"))

TRAY_ICON_ENABLED         = os.getenv("TRAY_ICON_ENABLED",         "false").lower() == "true"
STREAMING_DEFAULT_ENABLED = os.getenv("STREAMING_DEFAULT_ENABLED", "true").lower()  == "true"
FACE_LABELING_ENABLED     = os.getenv("FACE_LABELING_ENABLED",     "true").lower()  == "true"

# ── Logging ───────────────────────────────────────────────────────────────────

LOG_MAX_BYTES    = int(os.getenv("LOG_MAX_BYTES",    str(10 * 1024 * 1024)))
LOG_BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT", "2"))


# ── Runtime update (Settings UI) ─────────────────────────────────────────────

def update_env(updates: dict) -> bool:
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    else:
        lines = []

    for key, value in updates.items():
        new_line = f"{key}={value}\n"
        found    = False
        for i, line in enumerate(lines):
            if line.strip().startswith(f"{key}="):
                lines[i] = new_line
                found = True
                break
        if not found:
            lines.append(new_line)

        os.environ[key] = str(value)
        globals()[key]  = value

    with open(env_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    return True
