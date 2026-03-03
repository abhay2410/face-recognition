"""
config.py – Configuration loader for the FastAPI Door Access System.
Reads values from the .env file.
"""

import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# ─────────────────────────────────────────────
#  MS SQL Server
# ─────────────────────────────────────────────
MSSQL_SERVER     = os.getenv("MSSQL_SERVER", "202.88.254.148,65056")
MSSQL_USER       = os.getenv("MSSQL_USER", "sa")
MSSQL_PASSWORD   = os.getenv("MSSQL_PASSWORD", "sa@123")
MSSQL_DB         = os.getenv("MSSQL_DB", "face_attendance")
MSSQL_DRIVER     = os.getenv("MSSQL_DRIVER", "ODBC Driver 18 for SQL Server")
MSSQL_TRUST_CERT = os.getenv("MSSQL_TRUST_CERT", "yes")

# ─────────────────────────────────────────────
#  Face Recognition / FAISS
# ─────────────────────────────────────────────
EMBEDDING_DIM       = int(os.getenv("EMBEDDING_DIM", "512"))
FAISS_L2_THRESHOLD  = float(os.getenv("FAISS_L2_THRESHOLD", "0.60"))  # Lower = Stricter (Better security), Higher = Easier (Faster recognition)
ONBOARD_FRAMES      = int(os.getenv("ONBOARD_FRAMES", "5"))

# ─────────────────────────────────────────────
#  Live Monitoring (RTSP)
# ─────────────────────────────────────────────
_RTSP_RAW = os.getenv("RTSP_URLS", "")
if _RTSP_RAW:
    # Format: Name1:URL1,Name2:URL2
    RTSP_CAMERAS = {}
    for entry in _RTSP_RAW.split(","):
        if ":" in entry:
            name, url = entry.split(":", 1)
            RTSP_CAMERAS[name.strip()] = url.strip()
else:
    # Single fallback
    _SINGLE_URL = os.getenv("RTSP_URL", "rtsp://test:admin123@192.168.1.213:554/stream")
    RTSP_CAMERAS = {"Exit": _SINGLE_URL}

# ─────────────────────────────────────────────
#  External Door API
# ─────────────────────────────────────────────
EXTERNAL_API_ENABLED = os.getenv("EXTERNAL_API_ENABLED", "true").lower() == "true"
EXTERNAL_API_TIMEOUT = int(os.getenv("EXTERNAL_API_TIMEOUT", "10"))

_DOOR_RAW = os.getenv("EXTERNAL_API_URLS", "")
if _DOOR_RAW:
    EXTERNAL_API_URLS = {}
    for entry in _DOOR_RAW.split(","):
        if ":" in entry:
            name, url = entry.split(":", 1)
            EXTERNAL_API_URLS[name.strip()] = url.strip()
else:
    _SINGLE_DOOR = os.getenv("EXTERNAL_API_URL", "")
    EXTERNAL_API_URLS = {cam_name: _SINGLE_DOOR for cam_name in RTSP_CAMERAS.keys()}

# ─────────────────────────────────────────────
#  MediaPipe detector settings
# ─────────────────────────────────────────────
MP_MIN_DETECTION_CONFIDENCE = float(os.getenv("MP_MIN_DETECTION_CONFIDENCE", "0.6"))
MP_MODEL_SELECTION          = int(os.getenv("MP_MODEL_SELECTION", "1"))

RTSP_RECONNECT_DELAY = int(os.getenv("RTSP_RECONNECT_DELAY", "5"))
MONITOR_N_FRAMES     = int(os.getenv("MONITOR_N_FRAMES", "2"))        # Lower = More frequent checks (High CPU usage, better recognition)
MONITOR_COOLDOWN     = int(os.getenv("MONITOR_COOLDOWN", "10"))        # 5 second cooldown per person
MONITOR_ENABLED      = os.getenv("MONITOR_ENABLED", "true").lower() == "true"
LIVENESS_ENABLED     = os.getenv("LIVENESS_ENABLED", "true").lower() == "true"
BLINK_EAR_THRESHOLD  = float(os.getenv("BLINK_EAR_THRESHOLD", "0.500"))  # Higher = More sensitive (Easier to trigger), Lower = Stricter

# ─────────────────────────────────────────────
#  FastAPI / Uvicorn
# ─────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
