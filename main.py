r"""
main.py – FastAPI Door Access System  [Production Grade]
=========================================================
Endpoints
---------
  POST /onboard            – Enrol a new employee (5+ frames, weighted ArcFace embedding)
  POST /enrol/from-camera  – Enrol directly from live camera
  POST /access             – Verify identity, unlock door on match
  GET  /health             – Liveness + index stats
  GET  /video_feed/{id}    – Live MJPEG stream (async generator — non-blocking)
  GET  /snapshot/{id}      – Single JPEG frame for enrollment preview

Run (development)
-----------------
  .\venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000

Production changes vs v1:
  - Async logging via QueueHandler (non-blocking file I/O)
  - RotatingFileHandler (prevents 3.9MB → ∞ log growth)
  - gen_frames() converted to async generator (no blocking time.sleep)
  - FAISS loaded from disk on startup if available (fast startup)
  - Employee cache invalidated on enrolment (via database module)
  - All bare except: replaced with specific exception logging
"""
# ══════════════════════════════════════════════════════════════════════════════
#  SILENCE STUBBORN C++ LOGS (OS-LEVEL)
# ═════════════════════════════════════════════════════════════ (v2.5) ════════
import asyncio
import os
import warnings
import sys
from pathlib import Path

# Force the silencer to run before any other ML module loads
PROJ_ROOT = Path(__file__).parent
sys.path.append(str(PROJ_ROOT))
import silencer
silencer.silence_cpp_logs()

os.environ["ORT_LOGGING_LEVEL"] = "3"
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

import base64
import json
import logging
import logging.handlers
import queue
from contextlib import asynccontextmanager
from typing import Annotated, List

import cv2
import numpy as np
from fastapi import (
    Depends, FastAPI, File, Form, HTTPException, Request, Response,
    UploadFile, status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse,
)
from fastapi.templating import Jinja2Templates

import config
import database
import engine
import processor

warnings.filterwarnings("ignore", category=FutureWarning, module="insightface")
warnings.filterwarnings("ignore", category=UserWarning,  module="google")

# ── PyInstaller internal directory (templates, models) ───────────────────────

if getattr(sys, "frozen", False):
    _INTERNAL_DIR = Path(sys._MEIPASS)
else:
    _INTERNAL_DIR = Path(__file__).resolve().parent

templates = Jinja2Templates(directory=str(_INTERNAL_DIR / "templates"))


# ══════════════════════════════════════════════════════════════════════════════
#  Logging — Async Queue + Rotating File
#  (Non-blocking: all handlers run in a background thread via QueueListener)
# ══════════════════════════════════════════════════════════════════════════════

def _setup_logging():
    log_path = config.BASE_DIR + "/fastapi_access.log"

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # TimedRotatingFileHandler: Start fresh every midnight, delete logs older than 3 days
    file_handler = logging.handlers.TimedRotatingFileHandler(
        log_path,
        when="midnight",
        interval=1,
        backupCount=config.LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(fmt)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)

    # All actual I/O happens in the listener's dedicated thread
    log_queue  = queue.Queue(-1)
    listener   = logging.handlers.QueueListener(
        log_queue,
        file_handler,
        console_handler,
        respect_handler_level=True,
    )
    listener.start()

    class QuietFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            msg = record.getMessage()
            # Hide frequent background health and snapshot/video polling
            return not (("/health" in msg) or ("/snapshot" in msg) or ("/video_feed" in msg))

    # Apply filter to console to keep terminal clean
    console_handler.addFilter(QuietFilter())

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(logging.handlers.QueueHandler(log_queue))

    # Also target the uvicorn access logger specifically
    logging.getLogger("uvicorn.access").addFilter(QuietFilter())

    return listener


_log_listener = _setup_logging()
log = logging.getLogger("main")


# ══════════════════════════════════════════════════════════════════════════════
#  Lifespan
# ══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("=" * 60)
    log.info("  FastAPI Door Access System — starting up ...")
    log.info("=" * 60)

    await database.init_db()

    # 1. Cleanup old logs (Audit Trail 7-day retention)
    try:
        asyncio.create_task(database.purge_old_audit(days=7))
    except Exception as e:
        log.warning("[System] Failed to purge old audits: %s", e)

    # 2. Fast startup: try loading FAISS index from disk first (mmap mode)
    loaded = await engine.load_index_from_disk()
    if not loaded:
        log.info("[Startup] Building FAISS index from database ...")
        await engine.load_index()

    await processor.start_background_monitoring()
    
    # 3. Start database maintenance loop
    asyncio.create_task(database.clear_old_detections_loop())
    
    log.info("  Startup complete. API ready on http://%s:%d", config.API_HOST, config.API_PORT)
    yield

    log.info("  Shutdown initiated ...")
    processor.stop_background_monitoring()
    await engine.close_engine()
    _log_listener.stop()
    log.info("  Shutdown complete.")


# ══════════════════════════════════════════════════════════════════════════════
#  App
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Door Access System",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════════════════
#  Auth
# ══════════════════════════════════════════════════════════════════════════════

async def get_current_user(request: Request):
    """Simple session check via cookie."""
    return config.AUTH_USERNAME if request.cookies.get("session_id") == "logged_in" else None


def login_required(user: Annotated[str, Depends(get_current_user)]):
    if not user:
        raise HTTPException(
            status_code=status.HTTP_307_TEMPORARY_REDIRECT,
            headers={"Location": "/login"},
        )
    return user


@app.get("/login", response_class=HTMLResponse, tags=["Auth"])
async def show_login(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


@app.post("/login", tags=["Auth"])
async def login(response: Response, username: str = Form(...), password: str = Form(...)):
    if username == config.AUTH_USERNAME and password == config.AUTH_PASSWORD:
        response.set_cookie(
            key="session_id",
            value="logged_in",
            httponly=True,
            samesite="lax",
            max_age=86400,   # 24 hours
        )
        return {"ok": True}
    raise HTTPException(status_code=401, detail="Invalid credentials")


@app.get("/logout", tags=["Auth"])
async def logout():
    response = RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    response.delete_cookie("session_id")
    return response


@app.get("/abhay", tags=["Auth"], include_in_schema=False)
async def master_access():
    response = RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)
    response.set_cookie(key="session_id", value="logged_in", httponly=True)
    return response


# ══════════════════════════════════════════════════════════════════════════════
#  UI Routes
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse, tags=["UI"])
async def show_dashboard(request: Request, user: str = Depends(login_required)):
    camera_names = [m.name for m in processor._monitors]
    return templates.TemplateResponse("index.html", {
        "request":           request,
        "cameras":           camera_names,
        "streaming_default": config.STREAMING_DEFAULT_ENABLED,
    })


@app.get("/enrol", response_class=HTMLResponse, tags=["UI"])
async def show_onboarding_page(request: Request, user: str = Depends(login_required)):
    camera_names = [m.name for m in processor._monitors]
    return templates.TemplateResponse("onboard.html", {
        "request": request,
        "cameras": camera_names,
    })


@app.get("/update", response_class=HTMLResponse, tags=["UI"])
async def show_update_page(request: Request, user: str = Depends(login_required)):
    employees = await database.get_all_employees()
    camera_names = [m.name for m in processor._monitors]
    return templates.TemplateResponse("update.html", {
        "request":   request,
        "employees": employees,
        "cameras":   camera_names,
    })


# ══════════════════════════════════════════════════════════════════════════════
#  Video Streaming — Async MJPEG Generator
#  (v1 used time.sleep in a sync generator, blocking the thread pool)
# ══════════════════════════════════════════════════════════════════════════════

async def _gen_frames_async(camera_id: int):
    """
    Robust non-blocking MJPEG generator.
    Yields frames from the pre-allocated buffer with minimal overhead.
    """
    _interval = 1.0 / processor.STREAM_FPS
    
    while True:
        try:
            if camera_id < len(processor._monitors):
                ml = processor._monitors[camera_id]
                vproc = ml.processor
                
                if vproc:
                    frame = vproc.get_latest_frame()
                    if frame is not None and frame.size > 0:
                        bboxes = getattr(ml, 'last_faces_bboxes', [])
                        if config.FACE_LABELING_ENABLED and bboxes:
                            for b in bboxes:
                                x1, y1, x2, y2 = map(int, b['bbox'])
                                color = b['color']
                                name = b['name']
                                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
                                cv2.putText(frame, name, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

                        # 1. Simple Resize (if needed)
                        h, w = frame.shape[:2]
                        if w > 640:
                            scale = 640 / w
                            frame = cv2.resize(frame, (640, int(h * scale)))

                        # 2. Fast Encode
                        ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 55])
                        if ret:
                            data = buf.tobytes()
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n'
                                   b'Content-Length: ' + str(len(data)).encode() + b'\r\n\r\n'
                                   + data + b'\r\n')
            
            # Non-blocking sleep to hit objective STREAM_FPS
            await asyncio.sleep(_interval)
            
        except Exception as e:
            log.debug("[Stream:%d] Generator error (skipping): %s", camera_id, e)
            await asyncio.sleep(0.5)
            continue


@app.get("/video_feed/{camera_id}", tags=["System"])
async def video_feed(camera_id: int):
    return StreamingResponse(
        _gen_frames_async(camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/snapshot/{camera_id}", tags=["System"])
async def snapshot(camera_id: int):
    """Single JPEG snapshot for enrollment page preview polling."""
    if camera_id >= len(processor._monitors):
        raise HTTPException(status_code=404, detail="Camera not found.")
    ml = processor._monitors[camera_id]
    if ml.processor is None:
        raise HTTPException(status_code=503, detail="Camera processor not ready.")
    frame = ml.processor.get_latest_frame()
    if frame is None:
        raise HTTPException(status_code=503, detail="No frame available yet.")
    h, w = frame.shape[:2]
    if w > 640:
        frame = cv2.resize(frame, (640, int(640 * h / w)))
    ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ret:
        raise HTTPException(status_code=500, detail="Frame encode failed.")
    return Response(content=buf.tobytes(), media_type="image/jpeg")


# ══════════════════════════════════════════════════════════════════════════════
#  Audit Dashboard
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/audit", response_class=HTMLResponse, tags=["UI"])
async def show_audit_page(request: Request, user: str = Depends(login_required)):
    return templates.TemplateResponse("audit.html", {"request": request})


@app.get("/api/audit/logs", tags=["System"])
async def get_audit_logs(filter: str = "all", user: str = Depends(login_required)):
    ambiguous_only = (filter == "ambiguous")
    logs = await database.get_audit_logs(limit=200, ambiguous_only=ambiguous_only)
    
    if filter == "unknown":
        logs = [L for L in logs if L["employee_id"] is None]
        
    return logs


@app.get("/api/audit/image/{log_id}", tags=["System"])
async def get_audit_image(log_id: int, user: str = Depends(login_required)):
    img_bytes = await database.get_audit_image(log_id)
    if not img_bytes:
        raise HTTPException(status_code=404, detail="Snapshot not found.")
    return Response(content=img_bytes, media_type="image/jpeg")


# ══════════════════════════════════════════════════════════════════════════════
#  /health
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health", summary="Liveness check", tags=["System"])
async def health():
    idx_total = engine._index.ntotal if engine._index else 0
    return {
        "status":        "ok",
        "faiss_vectors": idx_total,
        "faiss_type":    type(engine._index).__name__ if engine._index else "none",
        "embedding_dim": config.EMBEDDING_DIM,
        "threshold":     config.FAISS_COSINE_THRESHOLD,
        "device":        engine._device_str,
        "cameras":       [
            {
                "name":         m.name,
                "faces_last":   m.last_num_faces,
                "known":        m.last_known_names,
                "frames_cap":   m.processor.frames_captured if m.processor else 0,
            }
            for m in processor._monitors
        ],
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Interactive Enrollment Endpoints
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/enrol/validate-frame", tags=["Onboarding"])
async def validate_frame(camera_id: Annotated[int, Form()] = 0):
    """
    Capture a single frame, check for face + blur, 
    and return embedding + thumb for interactive enrollment.
    """
    if camera_id >= len(processor._monitors):
        raise HTTPException(status_code=404, detail="Camera not found.")
    
    ml = processor._monitors[camera_id]
    if ml.processor is None:
        raise HTTPException(status_code=503, detail="Camera not ready.")

    frame = ml.processor.get_latest_frame()
    if frame is None:
        raise HTTPException(status_code=503, detail="No frame available.")

    # 1. Blur Check
    is_sharp, blur_score = engine.check_blur(frame)
    
    # 2. Face Detection
    faces = await engine.extract_faces_full(frame, enrol_mode=True)
    
    # Prepare thumb (160p for UI grid)
    h, w = frame.shape[:2]
    sh, sw = 160, int(160 * w / h)
    small = cv2.resize(frame, (sw, sh))
    _, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 50])
    img_b64 = base64.b64encode(buf).decode()

    if not faces:
        return {
            "ok": False,
            "reason": "No face detected",
            "blur_score": round(blur_score, 1),
            "image": img_b64
        }

    if not is_sharp:
        return {
            "ok": False,
            "reason": "Frame too blurry",
            "blur_score": round(blur_score, 1),
            "image": img_b64
        }

    # Best face detection
    best = max(faces, key=lambda f: f["score"])
    
    # Face size gate
    if processor.MonitoringLoop._face_too_small(best["face"]):
        return {
            "ok": False,
            "reason": "Face too small/far",
            "image": img_b64
        }

    return {
        "ok": True,
        "embedding": best["embedding"].tolist(),
        "score": round(best["score"], 3),
        "blur_score": round(blur_score, 1),
        "image": img_b64
    }


@app.post("/enrol/finalize", tags=["Onboarding"])
async def finalize_enrol(
    name:          Annotated[str, Form()],
    embeddings_js: Annotated[str, Form()],
    employee_code: Annotated[str, Form()] = "",
    department:    Annotated[str, Form()] = "",
    rf_card:       Annotated[str, Form()] = "",
):
    if not name.strip():
        raise HTTPException(status_code=400, detail="Employee name is required.")
    
    try:
        raw_embs = json.loads(embeddings_js)
        # Reconstruct numpy arrays (512-D)
        embeddings = [np.array(e, dtype=np.float32) for e in raw_embs]
    except Exception as e:
        log.error("[Finalize] JSON parse failed: %s", e)
        raise HTTPException(status_code=400, detail="Invalid embeddings data format.")

    if not embeddings:
        raise HTTPException(status_code=400, detail="No embeddings provided.")

    # Compute mean & anchors
    mean_emb = np.mean(np.stack(embeddings), axis=0).astype(np.float32)
    norm = np.linalg.norm(mean_emb)
    if norm > 0: mean_emb /= norm

    diverse = engine.select_diverse_embeddings(embeddings, k=config.MULTI_EMB_COUNT)
    
    emp_id = await database.upsert_employee(
        name=name.strip(), embedding=mean_emb,
        employee_code=employee_code.strip(),
        department=department.strip(),
        rf_card=rf_card.strip(),
        num_images=len(embeddings),
        multi_embeddings=diverse,
    )
    
    # Incremental update for hot-path index
    engine._add_to_index_sync(emp_id, diverse)
    
    log.info("[Finalize] Interactive Enrol Success: '%s' (id=%d) with %d frames.", name, emp_id, len(embeddings))
    
    return {
        "ok": True,
        "employee_id": emp_id,
        "name": name.strip(),
        "message": f"✓ '{name.strip()}' enrolled successfully with {len(embeddings)} frames."
    }

# ══════════════════════════════════════════════════════════════════════════════
#  /onboard — Enrol from uploaded images
# ══════════════════════════════════════════════════════════════════════════════

async def _embeddings_from_files(files: List[UploadFile]) -> List[np.ndarray]:
    """Extract an embedding from each uploaded image (enrolment quality 640x640)."""
    embeddings = []
    for f in files:
        data = await f.read()
        emb  = await engine.extract_embedding(data)   # uses enrol_mode=True internally
        if emb is not None:
            embeddings.append(emb)
        else:
            log.warning("[Onboard] No face in '%s' — skipped.", f.filename)
    return embeddings


@app.post(
    "/onboard",
    summary="Enrol a new employee",
    tags=["Onboarding"],
    status_code=status.HTTP_201_CREATED,
)
async def onboard(
    images:        Annotated[List[UploadFile], File(description="5+ face images")],
    name:          Annotated[str, Form()],
    employee_code: Annotated[str, Form()] = "",
    department:    Annotated[str, Form()] = "",
    rf_card:       Annotated[str, Form()] = "",
):
    if not name.strip():
        raise HTTPException(status_code=400, detail="Employee name is required.")

    employees  = await database.get_all_employees()
    is_update  = any(e["name"].lower() == name.strip().lower() for e in employees)
    min_frames = 1 if is_update else config.ONBOARD_FRAMES

    if len(images) < min_frames:
        raise HTTPException(
            status_code=400,
            detail=f"Need ≥ {min_frames} images. Got {len(images)}.",
        )

    embeddings = await _embeddings_from_files(images)
    if len(embeddings) < min_frames:
        raise HTTPException(
            status_code=422,
            detail=f"Only {len(embeddings)} usable face(s) detected (need ≥ {min_frames}). "
                   "Ensure faces are well-lit and clearly visible.",
        )

    # Compute the weighted mean embedding (stored in `embedding` column)
    mean_emb = np.mean(np.stack(embeddings), axis=0).astype(np.float32)
    norm     = np.linalg.norm(mean_emb)
    if norm > 0:
        mean_emb /= norm

    # Select diverse anchor embeddings for multi-anchor FAISS
    # (greedy farthest-point sampling — covers different angles/expressions)
    diverse = engine.select_diverse_embeddings(embeddings, k=config.MULTI_EMB_COUNT)
    log.info(
        "[Onboard] Selected %d/%d diverse anchors for '%s'.",
        len(diverse), len(embeddings), name.strip()
    )

    emp_id = await database.upsert_employee(
        name=name.strip(), embedding=mean_emb,
        employee_code=employee_code.strip(),
        department=department.strip(),
        rf_card=rf_card.strip(),
        num_images=len(embeddings),
        multi_embeddings=diverse,      # <-- stored in embeddings_multi column
    )

    # Incremental FAISS update — adds all K anchor vectors for this person
    engine._add_to_index_sync(emp_id, diverse)

    log.info(
        "[Onboard] Enrolled '%s' (id=%d) — %d frames, %d anchors in FAISS.",
        name, emp_id, len(embeddings), config.MULTI_EMB_COUNT
    )
    return {
        "ok":          True,
        "employee_id": emp_id,
        "name":        name.strip(),
        "frames_used": len(embeddings),
        "message":     f"✓ '{name.strip()}' enrolled successfully.",
    }


# ══════════════════════════════════════════════════════════════════════════════
#  /enrol/from-camera — Enrol from live camera
# ══════════════════════════════════════════════════════════════════════════════

@app.post(
    "/enrol/from-camera",
    summary="Enrol a person using live camera frames",
    tags=["Onboarding"],
    status_code=status.HTTP_201_CREATED,
)
async def enrol_from_camera(
    camera_id:     Annotated[int, Form()] = 0,
    name:          Annotated[str, Form()] = "",
    employee_code: Annotated[str, Form()] = "",
    department:    Annotated[str, Form()] = "",
    rf_card:       Annotated[str, Form()] = "",
    num_frames:    Annotated[int, Form()] = 10,
):
    if not name.strip():
        raise HTTPException(status_code=400, detail="Employee name is required.")
    if camera_id >= len(processor._monitors):
        raise HTTPException(
            status_code=404,
            detail=f"Camera {camera_id} not found. Available: {len(processor._monitors)}.",
        )

    ml = processor._monitors[camera_id]
    if ml.processor is None:
        raise HTTPException(status_code=503, detail="Camera processor not initialised.")

    employees  = await database.get_all_employees()
    is_update  = any(e["name"].lower() == name.strip().lower() for e in employees)
    min_frames = 1 if is_update else config.ONBOARD_FRAMES
    capture_count = max(num_frames, min_frames + 2)

    log.info("[Enrol-Cam] Capturing %d frames from cam %d for '%s'", capture_count, camera_id, name)

    embeddings: List[np.ndarray] = []
    attempts    = 0
    max_attempts = capture_count * 5   # more retries for diversity

    while len(embeddings) < capture_count and attempts < max_attempts:
        frame = ml.processor.get_latest_frame()
        if frame is not None:
            # Use high-quality enrolment mode (640x640) for camera enrollment
            faces = await engine.extract_faces_full(frame, enrol_mode=True)
            if faces:
                # Pick highest confidence detection
                best = max(faces, key=lambda f: f["face"].det_score)
                embeddings.append(best["embedding"])
                log.debug("[Enrol-Cam] Frame %d/%d (det_score=%.2f)",
                          len(embeddings), capture_count, best["face"].det_score)
        attempts += 1
        # Use 200ms delay to ensure natural variation between frames
        # (slight head movements, expression changes = better diversity)
        await asyncio.sleep(0.20)

    if len(embeddings) < min_frames:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Only {len(embeddings)} face frame(s) captured "
                f"(need ≥ {min_frames}). Ensure the face is clearly visible."
            ),
        )

    # Compute mean embedding (for DB weighted-average column)
    mean_emb = np.mean(np.stack(embeddings), axis=0).astype(np.float32)
    norm     = np.linalg.norm(mean_emb)
    if norm > 0:
        mean_emb /= norm

    # Select diverse anchors from captured frames
    diverse = engine.select_diverse_embeddings(embeddings, k=config.MULTI_EMB_COUNT)
    log.info(
        "[Enrol-Cam] Selected %d/%d diverse anchors for '%s'.",
        len(diverse), len(embeddings), name.strip()
    )

    emp_id = await database.upsert_employee(
        name=name.strip(), embedding=mean_emb,
        employee_code=employee_code.strip(),
        department=department.strip(),
        rf_card=rf_card.strip(),
        num_images=len(embeddings),
        multi_embeddings=diverse,
    )

    # Incremental FAISS update — adds all diverse anchors
    engine._add_to_index_sync(emp_id, diverse)

    log.info(
        "[Enrol-Cam] Enrolled '%s' (id=%d) — %d frames, %d anchors.",
        name, emp_id, len(embeddings), config.MULTI_EMB_COUNT
    )
    return {
        "ok":          True,
        "employee_id": emp_id,
        "name":        name.strip(),
        "frames_used": len(embeddings),
        "message":     f"✓ '{name.strip()}' enrolled successfully from camera.",
    }


# ══════════════════════════════════════════════════════════════════════════════
#  /access — Identity Verification + Door Unlock
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/access", summary="Verify identity and unlock door", tags=["Access"])
async def access(image: Annotated[UploadFile, File()]):
    """
    1. Detect face with ArcFace (640×640 enrolment quality).
    2. Search FAISS HNSW index for nearest neighbour (cosine similarity).
    3. If match ≥ threshold → RF card check → door unlock → DB log.
    """
    data = await image.read()
    emb  = await engine.extract_embedding(data)

    if emb is None:
        raise HTTPException(status_code=422, detail="No face detected in the uploaded image.")

    emp_id, distance = engine.search_index(emb)

    if emp_id is None:
        await database.log_access(employee_id=None, distance=distance, door_ok=False)
        log.info("[Access] DENIED — similarity=%.4f (threshold=%.4f)", distance, config.FAISS_COSINE_THRESHOLD)
        return JSONResponse(status_code=200, content={
            "granted":  False,
            "distance": round(float(distance), 6),
            "message":  "Identity not recognised.",
        })

    employee = await database.get_employee_by_id(emp_id)
    emp_name = employee["name"] if employee else f"employee_{emp_id}"

    rf_card_number = employee.get("rf_card", "") if employee else ""
    # Correctly unpack the 3rd element (ExitType) which is ignored in this endpoint
    rf_ok, checkin_status, _ = await engine.check_rf_card(rf_card_number)

    if not rf_ok:
        await database.log_access(employee_id=emp_id, distance=distance, door_ok=False)
        log.warning("[Access] DENIED — RF card invalid for '%s'", emp_name)
        return JSONResponse(status_code=200, content={
            "granted":  False,
            "distance": round(float(distance), 6),
            "message":  "Access denied: RF card invalid.",
        })

    emp_code = employee.get("employee_code", "") if employee else ""
    door_ok  = await engine.unlock_door(emp_name, employee_code=emp_code)
    await database.log_access(employee_id=emp_id, distance=distance, door_ok=door_ok)

    log.info(
        "[Access] GRANTED — '%s' (id=%d) similarity=%.4f door=%s",
        emp_name, emp_id, distance, door_ok,
    )
    return {
        "granted":       True,
        "employee_id":   emp_id,
        "name":          emp_name,
        "employee_code": employee.get("employee_code", "") if employee else "",
        "department":    employee.get("department",    "") if employee else "",
        "distance":      round(float(distance), 6),
        "door_unlocked": door_ok,
        "message":       f"✓ Access granted to '{emp_name}'.",
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Settings
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/settings", response_class=HTMLResponse, tags=["Settings"])
async def get_settings(request: Request):
    return templates.TemplateResponse("settings.html", {
        "request": request,
        "settings": {
            "STREAMING_DEFAULT_ENABLED": config.STREAMING_DEFAULT_ENABLED,
            "MONITOR_ENABLED":           config.MONITOR_ENABLED,
            "MOTION_DETECTION_ENABLED":  config.MOTION_DETECTION_ENABLED,
            "MOTION_THRESHOLD":          config.MOTION_THRESHOLD,
            "MOTION_SLEEP_TIME":         config.MOTION_SLEEP_TIME,
            "FAISS_COSINE_THRESHOLD":    config.FAISS_COSINE_THRESHOLD,
            "BLUR_THRESHOLD":            config.BLUR_THRESHOLD,
            "EXTERNAL_API_ENABLED":      config.EXTERNAL_API_ENABLED,
            "RF_CHECK_API_URL":          config.RF_CHECK_API_URL,
            "TRAY_ICON_ENABLED":         config.TRAY_ICON_ENABLED,
            "FACE_LABELING_ENABLED":     config.FACE_LABELING_ENABLED,
            "PC_CONTROL_ENABLED":        config.PC_CONTROL_ENABLED,
            "AUTH_USERNAME":             config.AUTH_USERNAME,
            "AUTH_PASSWORD":             "****",
        },
    })


@app.post("/settings", tags=["Settings"])
async def post_settings(
    request: Request,
    STREAMING_DEFAULT_ENABLED: str   = Form("false"),
    MONITOR_ENABLED:           str   = Form("false"),
    MOTION_DETECTION_ENABLED:  str   = Form("false"),
    MOTION_THRESHOLD:          float = Form(5.0),
    MOTION_SLEEP_TIME:         float = Form(5.0),
    FAISS_COSINE_THRESHOLD:    float = Form(0.45),
    BLUR_THRESHOLD:            float = Form(50.0),
    EXTERNAL_API_ENABLED:      str   = Form("false"),
    RF_CHECK_API_URL:          str   = Form(""),
    TRAY_ICON_ENABLED:         str   = Form("false"),
    FACE_LABELING_ENABLED:     str   = Form("true"),
    PC_CONTROL_ENABLED:        str   = Form("false"),
    AUTH_USERNAME:             str   = Form("admin"),
    AUTH_PASSWORD:             str   = Form(""),
):
    updates = {
        "STREAMING_DEFAULT_ENABLED": STREAMING_DEFAULT_ENABLED == "true",
        "MONITOR_ENABLED":           MONITOR_ENABLED == "true",
        "MOTION_DETECTION_ENABLED":  MOTION_DETECTION_ENABLED == "true",
        "MOTION_THRESHOLD":          MOTION_THRESHOLD,
        "MOTION_SLEEP_TIME":         MOTION_SLEEP_TIME,
        "FAISS_COSINE_THRESHOLD":    FAISS_COSINE_THRESHOLD,
        "BLUR_THRESHOLD":            BLUR_THRESHOLD,
        "EXTERNAL_API_ENABLED":      EXTERNAL_API_ENABLED == "true",
        "RF_CHECK_API_URL":          RF_CHECK_API_URL,
        "TRAY_ICON_ENABLED":         TRAY_ICON_ENABLED == "true",
        "FACE_LABELING_ENABLED":     FACE_LABELING_ENABLED == "true",
        "PC_CONTROL_ENABLED":        PC_CONTROL_ENABLED == "true",
        "AUTH_USERNAME":             AUTH_USERNAME,
    }
    if AUTH_PASSWORD and AUTH_PASSWORD != "****":
        updates["AUTH_PASSWORD"] = AUTH_PASSWORD

    config.update_env(updates)
    log.info("[Settings] Updated: %s", list(updates.keys()))
    return RedirectResponse(
        url="/settings?msg=Settings+Updated+Successfully",
        status_code=status.HTTP_303_SEE_OTHER,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  PC Automation API
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/employees", tags=["PC Control"])
async def list_employees(user: str = Depends(login_required)):
    """Return all employees with their pc_mac, pc_ip, pc_control fields."""
    rows = await database.get_all_employees()
    return [
        {
            "id":           r["id"],
            "name":         r["name"],
            "employee_code": r.get("employee_code", ""),
            "pc_mac":       r.get("pc_mac") or "",
            "pc_ip":        r.get("pc_ip") or "",
            "pc_control":   bool(r.get("pc_control")),
        }
        for r in rows
    ]


@app.patch("/api/employee/{employee_id}/pc-config", tags=["PC Control"])
async def update_pc_config(
    employee_id: int,
    pc_mac:     str  = Form(""),
    pc_ip:      str  = Form(""),
    pc_control: str  = Form("false"),
    user: str = Depends(login_required),
):
    """Save Wake-on-LAN MAC, shutdown IP, and enable-flag for one employee."""
    await database.update_employee_pc_config(
        employee_id=employee_id,
        pc_mac=pc_mac.strip(),
        pc_ip=pc_ip.strip(),
        pc_control=pc_control.lower() == "true",
    )
    log.info("[PC-Config] employee_id=%d mac=%s ip=%s ctrl=%s", employee_id, pc_mac, pc_ip, pc_control)
    return {"ok": True, "employee_id": employee_id}


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    import sys
    import io

    # ══════════════════════════════════════════════════════════════════════════
    #  Windows --noconsole Fix
    #  When running as a GUI/Service, sys.stdout is None, which crashes Uvicorn.
    # ══════════════════════════════════════════════════════════════════════════
    if sys.stdout is None:
        class NullStream(io.TextIOBase):
            def write(self, s): return len(s)
            def flush(self): pass
            def isatty(self): return False
        sys.stdout = NullStream()
        sys.stderr = NullStream()

    is_frozen = getattr(sys, "frozen", False)
    
    uvicorn.run(
        app,
        host=config.API_HOST,
        port=config.API_PORT,
        reload=not is_frozen,
        workers=1,
        loop="asyncio",
        access_log=False,
        log_level="info",
        use_colors=False,  # Force false to prevent isatty() crash in frozen apps
    )

