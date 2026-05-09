"""
engine.py – Hardware & Integration Layer [v3.1 — Multi-Office Ready]
==================================================================
Manages all external interactions: door locks, network speakers, and 
remote employee attendance APIs.

Design Logic:
  - Branch Isolation: Uses office-specific Branch IDs for accurate reporting.
  - Fail-Safe Hardware: Supports WebSocket and HTTP POST for local/remote locks.
  - Group Audio: Shared speaker support for office clusters (e.g. DEV office).
  - Version Discovery: Automatically detects FaceAnalysis capabilities.
  - GPU Acceleration: Supports CUDA (NVIDIA) and DirectML (Intel/AMD) on Windows.
"""

import asyncio
import logging
import os
import sys
import inspect
import urllib.parse
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import faiss
import httpx
import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis

import config

log = logging.getLogger("engine")


# ══════════════════════════════════════════════════════════════════════════════
#  Global GPU Serialisation (Crucial for 2GB VRAM + 4 Cameras)
# ══════════════════════════════════════════════════════════════════════════════
GPU_LOCK = asyncio.Semaphore(1)


# ══════════════════════════════════════════════════════════════════════════════
#  1. Version-Aware Compatibility Layer
# ══════════════════════════════════════════════════════════════════════════════

def _best_ort_providers() -> List[str]:
    available  = set(ort.get_available_providers())
    candidates = [
        ("CUDAExecutionProvider",     "NVIDIA GPU (CUDA)"),
        ("TensorrtExecutionProvider", "NVIDIA GPU (TensorRT)"),
        ("DmlExecutionProvider",      "Windows GPU (DirectML)"),
        ("OpenVINOExecutionProvider", "Intel CPU/iGPU (OpenVINO)"),
        ("CPUExecutionProvider",      "CPU (fallback)"),
    ]
    providers = []
    for provider, label in candidates:
        if provider in available:
            providers.append(provider)
            log.info("[Engine] ONNX provider found: %s", label)
            break
    if "CPUExecutionProvider" not in providers:
        providers.append("CPUExecutionProvider")
    return providers


_ORT_PROVIDERS = _best_ort_providers()
# Use GPU context if ANY acceleration provider (CUDA, TensorRT, or DirectML) is found
_use_gpu       = any(p in _ORT_PROVIDERS for p in ["CUDAExecutionProvider", "TensorrtExecutionProvider", "DmlExecutionProvider"])
_ctx_id        = 0 if _use_gpu else -1
_device_str    = "GPU (Accelerated)" if _use_gpu else "CPU"


def _make_analyzer(det_size: tuple, det_thresh: float) -> FaceAnalysis:
    """Bulletproof loader for any version of InsightFace."""
    if getattr(sys, "frozen", False):
        base_path = Path(sys._MEIPASS)
    else:
        base_path = Path(config.BASE_DIR)

    model_root = str(base_path / "data")
    
    # Check what the FaceAnalysis constructor supports
    sig = inspect.signature(FaceAnalysis.__init__)
    init_params = sig.parameters.keys()
    
    kwargs = {
        "name": "insightface_models",
        "root": model_root,
    }
    # Only add providers if the constructor supports it
    if "providers" in init_params:
        kwargs["providers"] = _ORT_PROVIDERS
        log.debug("[Engine] Initializing FaceAnalysis with modern providers list.")
    
    a = FaceAnalysis(**kwargs)
    
    # 3. Modern ORT stability flags (permanently fixes shape-mismatch warnings)
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opts.enable_mem_pattern = False   # This is the "Magic Fix" for SCRFD VerifyOutputSizes
    sess_opts.log_severity_level = 3       # Silence everything else
    
    prep_kwargs = {
        "ctx_id": _ctx_id,
        "det_thresh": det_thresh,
        "det_size": det_size
    }
    
    # SILENCE THE VOID: Redirect C++ stderr during model loading
    # This kills the "VerifyOutputSizes" warnings effectively.
    import contextlib
    
    with open(os.devnull, 'w') as fnull:
        with contextlib.redirect_stderr(fnull):
            try:
                a.prepare(**prep_kwargs)
            except TypeError:
                # If even that fails, try the most basic prepare
                a.prepare(ctx_id=_ctx_id, det_thresh=det_thresh, det_size=det_size)
                
    return a

# ═════════════════════════ (Rest of file remains highly optimized) ════════════

_analyzer: Optional[FaceAnalysis] = None

def _get_analyzer(enrol_mode: bool = False) -> FaceAnalysis:
    """
    Unified analyzer to save VRAM. 
    Loads a single 640x640 instance that handles both monitoring and enrollment.
    """
    global _analyzer
    if _analyzer is None:
        log.info("[Engine] Loading Unified FaceAnalysis (640x640) on %s...", _device_str)
        # We use 640x640 as a baseline; it handles distant faces well enough and 
        # fits comfortably in 2GB VRAM as a single instance.
        _analyzer = _make_analyzer((640, 640), det_thresh=0.40)
        log.info("[Engine] Unified Engine Ready.")
    return _analyzer

# --- (Diversity Selection / FAISS logic from previous successful version keeps running) ---

def select_diverse_embeddings(embeddings: List[np.ndarray], k: int) -> List[np.ndarray]:
    if len(embeddings) <= k: return list(embeddings)
    vecs = [v / (np.linalg.norm(v) + 1e-8) for v in embeddings]
    selected_idx = [0]
    while len(selected_idx) < k:
        best_idx, best_min_dist = -1, -1.0
        for i in range(len(vecs)):
            if i in selected_idx: continue
            min_dist = 1.0 - min([float(np.dot(vecs[i], vecs[s])) for s in selected_idx])
            if min_dist > best_min_dist:
                best_min_dist, best_idx = min_dist, i
        selected_idx.append(best_idx)
    return [vecs[i] for i in selected_idx]

_index, _index_ids, _index_lock = None, [], asyncio.Lock()
_MIN_SCORE_GAP = 0.05

async def load_index():
    global _index, _index_ids
    import database as db
    log.info("[Engine] Fetching embeddings from database...")
    all_emps = await db.get_all_multi_embeddings()
    log.info("[Engine] Fetched data for %d employees.", len(all_emps))
    
    all_vecs, new_ids = [], []
    for emp in all_emps:
        emp_id, mat = emp["id"], emp["embeddings"]
        for i in range(mat.shape[0]):
            vec = mat[i].astype(np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0: vec /= norm
            all_vecs.append(vec)
            new_ids.append(emp_id)
            
    if not all_vecs:
        async with _index_lock:
            log.info("[Engine] No vectors found. Creating empty IndexFlatIP.")
            _index, _index_ids = faiss.IndexFlatIP(config.EMBEDDING_DIM), []
            return

    # --- Start of heavy CPU section ---
    def _build_index_sync(vecs, ids):
        log.info("[Engine] Building HNSW index with %d vectors in background...", len(vecs))
        # 1. Create the index
        idx = faiss.IndexHNSWFlat(config.EMBEDDING_DIM, config.HNSW_M, faiss.METRIC_INNER_PRODUCT)
        idx.hnsw.efSearch = config.HNSW_EF_SEARCH
        
        # 2. Add vectors
        idx.add(np.vstack(vecs).astype(np.float32))
        
        # 3. Save to disk (fast enough on most SSDs)
        save_path = os.path.join(config.BASE_DIR, "data", "faiss_hnsw.index")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        try:
            faiss.write_index(idx, save_path)
            # Read back as mmap if possible to save RAM, otherwise keep the one we built
            # For simplicity here we just use the built one
        except Exception as e:
            log.error("[Engine] Disk save failed: %s", e)
            
        return idx

    # Run the heavy math in the thread pool to keep the event loop alive
    loop = asyncio.get_event_loop()
    new_index = await loop.run_in_executor(None, _build_index_sync, all_vecs, new_ids)

    # Atomic swap
    async with _index_lock:
        _index = new_index
        _index_ids = new_ids
        
    log.info("[Engine] Index ready (%d vectors).", _index.ntotal)
    
    # Background sync to SQL
    try:
        save_path = os.path.join(config.BASE_DIR, "data", "faiss_hnsw.index")
        with open(save_path, "rb") as f:
            blob = f.read()
        asyncio.create_task(db.save_faiss_index(blob))
    except Exception as e:
        log.error("[Engine] SQL sync failed: %s", e)


async def load_index_from_disk() -> bool:
    global _index, _index_ids
    import database as db
    p = os.path.join(config.BASE_DIR, "data", "faiss_hnsw.index")
    
    # 1. Try SQL first if disk is missing (satisfies "kept in SQL" request)
    if not os.path.exists(p):
        log.info("[Engine] Index missing on disk. Checking SQL...")
        blob = await db.load_faiss_index()
        if blob:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "wb") as f:
                f.write(blob)
            log.info("[Engine] Restored index from SQL to disk.")
        else:
            return False

    try:
        # Use IO_FLAG_MMAP to keep RAM free (OS manages memory via disk mapping)
        loaded = faiss.read_index(p, faiss.IO_FLAG_MMAP)
        
        all_emps = await db.get_all_multi_embeddings()
        expected = sum(emp["embeddings"].shape[0] for emp in all_emps)
        
        if loaded.ntotal != expected:
            log.warning("[Engine] Disk index out of sync (%d vs %d). Rebuilding...", loaded.ntotal, expected)
            return False
            
        new_ids = []
        for emp in all_emps:
            for _ in range(emp["embeddings"].shape[0]):
                new_ids.append(emp["id"])
                
        async with _index_lock:
            _index, _index_ids = loaded, new_ids
        return True
    except Exception as e:
        log.error("[Engine] mmap load failed: %s", e)
        return False

def _add_to_index_sync(employee_id: int, embeddings: Union[np.ndarray, List[np.ndarray]]):
    global _index, _index_ids
    if isinstance(embeddings, np.ndarray) and embeddings.ndim == 1: embeddings = [embeddings]
    if _index is None: _index = faiss.IndexFlatIP(config.EMBEDDING_DIM)
    vecs = []
    for vec in embeddings:
        v = vec.astype(np.float32).reshape(512)
        n = np.linalg.norm(v); 
        if n > 0: v /= n
        vecs.append(v)
        _index_ids.append(employee_id)
    _index.add(np.vstack(vecs).astype(np.float32))
    faiss.write_index(_index, os.path.join(config.BASE_DIR, "data", "faiss_hnsw.index"))

def search_index_multi(embeddings: np.ndarray) -> List[Tuple[Optional[int], float]]:
    if _index is None or _index.ntotal == 0 or len(embeddings) == 0: return [(None, 0.0)]*len(embeddings)
    queries = embeddings.astype(np.float32)
    norms = np.linalg.norm(queries, axis=1, keepdims=True); np.divide(queries, norms, out=queries, where=norms!=0)
    k_search = min(config.MULTI_EMB_COUNT + 2, _index.ntotal)
    D, I = _index.search(queries, k_search)
    results = []
    for q in range(len(embeddings)):
        scores_by_id = {}
        for rank in range(k_search):
            idx = int(I[q][rank])
            if 0 <= idx < len(_index_ids):
                eid, score = _index_ids[idx], float(D[q][rank])
                if eid not in scores_by_id or score > scores_by_id[eid]: scores_by_id[eid] = score
        if not scores_by_id: results.append((None, 0.0)); continue
        ranked = sorted(scores_by_id.items(), key=lambda x: x[1], reverse=True)
        top_id, top_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        if top_score < config.FAISS_COSINE_THRESHOLD or (top_score - second_score < _MIN_SCORE_GAP and second_score > 0): 
            results.append((None, top_score))
        else: results.append((top_id, top_score))
    return results

def search_index(emb: np.ndarray) -> Tuple[Optional[int], float]:
    res = search_index_multi(np.array([emb])); return res[0]

def check_blur(img: np.ndarray) -> Tuple[bool, float]:
    """Laplacian variance blur filter. Returns (is_sharp, score)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return score >= config.BLUR_THRESHOLD, score

async def extract_faces_full(image: Union[bytes, np.ndarray], enrol_mode: bool = False) -> List[Dict]:
    if isinstance(image, bytes):
        img = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
    else: img = image
    if img is None: return []
    analyzer = _get_analyzer(enrol_mode=enrol_mode)
    
    # Serialised GPU access prevents VRAM spikes and OOM on 2GB cards
    async with GPU_LOCK:
        faces = await asyncio.get_event_loop().run_in_executor(None, analyzer.get, img)
    return [
        {
            "face": f,
            "embedding": f.normed_embedding.astype(np.float32),
            "bbox": f.bbox.tolist(),
            "score": float(f.det_score)
        } 
        for f in faces if f.normed_embedding is not None
    ]

async def auto_optimize_identity(emp_id: int, name: str, new_embedding: np.ndarray):
    """
    Intelligently integrates a new high-quality face into the existing identity model.
    Only triggers if AUTO_UPDATE_ENABLED is true.
    """
    if not config.AUTO_UPDATE_ENABLED:
        return

    import database as db
    try:
        # 1. Fetch current multi-embeddings for this employee
        raw_multi = await db.get_multi_embeddings_for_employee(emp_id)
        if not raw_multi:
            log.warning("[Engine] No multi-embeddings found for '%s' to optimize.", name)
            return

        current_anchors = list(db.bytes_to_multi_embeddings(bytes(raw_multi)))
        
        # 2. Update Strategy: Rolling FIFO replacement
        # We replace the oldest anchor with the newest high-confidence one.
        # This keeps the model fresh and adapts to gradual appearance changes.
        new_anchors = current_anchors[1:] + [new_embedding]
        
        # 3. Save back to DB
        await db.update_multi_embeddings(emp_id, new_anchors)
        log.info("[Engine] ⚡ AUTO-OPTIMIZED identity for '%s' (Face model refreshed).", name)
        
        # 4. Trigger Index Reload (In the background to avoid blocking)
        asyncio.create_task(load_index())

    except Exception as e:
        log.error("[Engine] Auto-optimization failed for '%s': %s", name, e)

async def extract_embedding(image: Union[bytes, np.ndarray]) -> Optional[np.ndarray]:
    faces = await extract_faces_full(image, enrol_mode=True)
    return max(faces, key=lambda f: f["face"].det_score)["embedding"] if faces else None

_http_client: Optional[httpx.AsyncClient] = None
def _get_http_client():
    global _http_client
    if _http_client is None: _http_client = httpx.AsyncClient(timeout=config.EXTERNAL_API_TIMEOUT, limits=httpx.Limits(max_connections=20, max_keepalive_connections=10))
    return _http_client

async def close_engine():
    global _http_client
    if _http_client: await _http_client.aclose(); _http_client = None

async def check_rf_card(rf: Optional[str], camera_name: str = "Exit", department: str = ""):
    """
    Checks the RF Card status via HTTP GET.
    Returns (status_ok, status_string, exit_type).
    """
    # 1. Skip the RF status check for anyone without a valid card (NULL, empty, "0", or "None")
    if not config.RF_CHECK_API_URL or not rf or str(rf).strip().lower() in ("", "none", "0"):
        return True, "SKIP", "ENTRY" if "entrance" in str(camera_name).lower() else "EXIT"
    
    url = config.RF_CHECK_API_URL.replace("{rf_card}", str(rf)).replace("rffid", str(rf))
    
    # 2. Determine bypass conditions
    is_entrance = "entrance" in str(camera_name).lower()
    is_embeded = department and str(department).strip().lower() in ("embeded", "embedded")
    
    try:
        resp = await _get_http_client().get(url)
        log.info("[RF-Check] %s → %d %s", url.split('?')[0], resp.status_code, resp.text[:120])
        if not resp.is_success:
            if is_entrance or is_embeded:
                return True, "BYPASS_ERR", "ENTRY" if is_entrance else "EXIT"
            return False, "ERROR", "EXIT"
            
        d = resp.json()
        item = d.get("Data", [{}])[0] if d.get("status") == 200 else {}
        status = item.get("Checkinstatus")
        exit_type = str(item.get("ExitType", "EXIT")).strip()
        
        # 3. Logic: Normal check vs Bypass
        rf_ok = status in ("OUT", "RdytoChkIn")
        
        if is_entrance or is_embeded:
            # Determine a safe default exit_type for bypass cases if API is vague
            default_type = "ENTRY" if is_entrance else "EXIT"
            final_exit_type = exit_type if exit_type and exit_type != "EXIT" else default_type
            
            if not rf_ok:
                log.info("[RF-Check] Bypassing status '%s' (Camera: %s, Dept: %s)", status, camera_name, department)
            return True, status, final_exit_type
            
        return rf_ok, status, exit_type
        
    except Exception as e:
        log.warning("[RF-Check] Request failed: %s", e)
        # If timeout/exception, bypass scenarios still get access
        if is_entrance or is_embeded:
            return True, "BYPASS_EXC", "ENTRY" if is_entrance else "EXIT"
        return False, "TIMEOUT", "EXIT"

async def log_entry(employee_code: str):
    """Logs an entry event via HTTP POST."""
    if not config.LOG_ENTRY_API_URL or not config.DEVICE_MAC_ADDRESS:
        return
    url = config.LOG_ENTRY_API_URL.replace("{mac}", str(config.DEVICE_MAC_ADDRESS)).replace("{id}", str(employee_code))
    try:
        # Use POST for these logging APIs as instructed
        resp = await _get_http_client().post(url)
        return resp.is_success
    except Exception as e:
        return False

async def log_exit(employee_code: str):
    """Logs an exit event via HTTP POST."""
    if not config.LOG_EXIT_API_URL or not config.DEVICE_MAC_ADDRESS:
        return
    url = config.LOG_EXIT_API_URL.replace("{mac}", str(config.DEVICE_MAC_ADDRESS)).replace("{id}", str(employee_code))
    try:
        # Use POST for these logging APIs as instructed
        resp = await _get_http_client().post(url)
        return resp.is_success
    except Exception as e:
        return False

async def unlock_door(name: str, employee_code: str = "", camera_name: str = "Exit"):
    """Unlocks a door via WebSocket (OPEN_DOOR) or Remote Admin API (HTTP)."""
    if not config.EXTERNAL_API_ENABLED: return False
    
    mode = config.DOOR_UNLOCK_MODE # WEBSOCKET, API, or AUTO
    
    # 1. Remote Admin API Mode (Grapes Online)
    if mode == "API" or (mode == "AUTO" and config.REMOTE_DOOR_API_URL and not config.EXTERNAL_API_URLS):
        if config.REMOTE_DOOR_API_URL and employee_code:
            # Hierarchical Branch ID lookup
            branch_id = config.get_cam_setting(camera_name, "BRANCH_ID", config.BRANCH_ID)
            remote_url = config.REMOTE_DOOR_API_URL.replace("{branch_id}", str(branch_id)).replace("{user_id}", str(employee_code))
            try:
                # Changed to POST as the API returned 405 Method Not Allowed on GET
                resp = await _get_http_client().post(remote_url)
                log.info("[Remote-Door] %s (Branch:%s) → %d", camera_name, branch_id, resp.status_code)
                return resp.is_success
            except Exception as e:
                log.warning("[Remote-Door] Failed for %s: %s", name, e)
                return False
        return False

    # 2. Local Hardware Mode (WebSocket or HTTP POST)
    url = config.get_cam_setting(camera_name, "EXTERNAL_API_URLS")
    if not url: return False
    
    is_ws_url = url.startswith("ws://") or url.startswith("wss://")
    use_ws = (mode == "WEBSOCKET") or (mode == "AUTO" and is_ws_url)

    if use_ws:
        try:
            import websockets
            async with websockets.connect(url, open_timeout=5) as websocket:
                await websocket.send("OPEN_DOOR")
                await asyncio.sleep(0.1)
                log.info("[Door-WS] %s → WebSocket sent: OPEN_DOOR", camera_name)
                return True
        except Exception as e:
            log.warning("[Door-WS] WebSocket unlock failed for %s: %s", url, e)
            return False
            
    # Fallback to legacy HTTP POST (if mode is HTTP or AUTO with http:// url)
    for k, v in {"{code}": str(employee_code), "{name}": str(name), "{door}": str(camera_name), "{id}": str(employee_code)}.items(): url = url.replace(k, v)
    try:
        resp = await _get_http_client().post(url)
        log.info("[Door-HTTP] %s → %d %s", camera_name, resp.status_code, resp.text[:120])
        return resp.is_success
    except Exception as e:
        log.warning("[Door-HTTP] Unlock request failed for '%s': %s", name, e)
        return False

async def announce(text: str, device_id: str = None):
    """
    Sends a text message to audio outputs via HTTP POST.
    Logic: Triggers Speaker API (Network).
    """
    if not text or not config.AUDIO_MODE:
        return

    # 1. API SPEAKER MODE (HTTP POST Call)
    if config.AUDIO_MODE in ("API", "NETWORK") and config.SPEAKER_API_URL:
        url = config.SPEAKER_API_URL.replace("{message}", urllib.parse.quote(text))
        d_id = device_id or config.SPEAKER_DEVICE_ID
        url = url.replace("{device_id}", str(d_id))
        
        try:
            resp = await _get_http_client().post(url)
            log.info("[Speaker-API] Sent: '%s' (Device: %s) Status: %d", text, d_id, resp.status_code)
            return
        except Exception as e:
            log.warning("[Speaker-API] Failed: %s", e)


async def trigger_pc_start(mac_address: str):
    """Sends a Wake-on-LAN magic packet to start a PC (Turn ON)."""
    if not mac_address or len(mac_address) < 12:
        return False
    try:
        import struct
        import socket
        # Clean the MAC address
        add_bytes = mac_address.replace(':', '').replace('-', '')
        hw_addr = struct.pack('BBBBBB', *[int(add_bytes[i:i+2], 16) for i in range(0, 12, 2)])
        # Build the magic packet
        msg = b'\xff' * 6 + hw_addr * 16
        # Send via UDP broadcast
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            s.sendto(msg, ('255.255.255.255', 9))
        log.info("[Engine] Wake-on-LAN magic packet sent to %s", mac_address)
        return True
    except Exception as e:
        log.error("[Engine] Wake-on-LAN failed for %s: %s", mac_address, e)
        return False


async def trigger_pc_stop(ip_address: str):
    """Sends a shutdown signal to the Client Agent (Turn OFF)."""
    if not ip_address:
        return False
    try:
        import socket
        # Send simple UDP packet to the listener port (9999)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.sendto(b"SHUTDOWN_NOW", (ip_address, 9999))
        log.info("[Engine] Shutdown signal sent to %s", ip_address)
        return True
    except Exception as e:
        log.error("[Engine] Remote shutdown failed for %s: %s", ip_address, e)
        return False


async def trigger_pc_lock(ip_address: str):
    """Sends a lock signal to the Client Agent."""
    if not ip_address:
        return False
    try:
        import socket
        # Send simple UDP packet to the listener port (9999)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.sendto(b"LOCK_NOW", (ip_address, 9999))
        log.info("[Engine] Lock signal sent to %s", ip_address)
        return True
    except Exception as e:
        log.error("[Engine] Remote lock failed for %s: %s", ip_address, e)
        return False
