r"""
main.py – FastAPI Door Access System
=====================================
Endpoints
---------
  POST /onboard   – Enrol an employee (5+ frames, mean FaceNet embedding)
  POST /access    – Verify identity against FAISS index, unlock door on match
  GET  /health    – Liveness + index stats

Run
---
  .\venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

"""

import logging
import sys
from contextlib import asynccontextmanager
from typing import List, Annotated
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
import cv2
import numpy as np

import config
import database
import engine
import processor

# Templates setup
BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# ──────────────────────────────────────────────────────────────────────────────
#  Logging
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("fastapi_access.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("main")


# ──────────────────────────────────────────────────────────────────────────────
#  Lifespan
# ──────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("=" * 60)
    log.info("  FastAPI Door Access System  -  starting up ...")
    log.info("=" * 60)

    await database.init_db()
    await engine.load_index()
    await processor.start_background_monitoring()

    log.info("  Startup complete. API ready on http://%s:%d", config.API_HOST, config.API_PORT)
    yield

    log.info("  Shutdown complete.")
    processor.stop_background_monitoring()


# ──────────────────────────────────────────────────────────────────────────────
#  App
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Door Access System",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════════════════
#  UI Routes
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/enrol", response_class=HTMLResponse, tags=["UI"])
async def show_onboarding_page(request: Request):
    """
    Renders the beautiful onboarding UI.
    """
    return templates.TemplateResponse("onboard.html", {"request": request})


@app.get("/update", response_class=HTMLResponse, tags=["UI"])
async def show_update_page(request: Request):
    """
    Renders the UI for improving recognition accuracy of existing people.
    """
    employees = await database.get_all_employees()
    return templates.TemplateResponse("update.html", {"request": request, "employees": employees})


# ──────────────────────────────────────────────────────────────────────────────
#  Helper
# ──────────────────────────────────────────────────────────────────────────────

async def _embeddings_from_files(files: List[UploadFile]) -> List[np.ndarray]:
    """
    Decode each uploaded image file, extract its FaceNet embedding.
    Raises HTTP 422 if fewer than ONBOARD_FRAMES valid embeddings are produced.
    """
    embeddings: List[np.ndarray] = []
    for f in files:
        data = await f.read()
        emb  = await engine.extract_embedding(data)
        if emb is not None:
            embeddings.append(emb)
        else:
            log.warning("No face detected in uploaded file '%s' – skipped.", f.filename)

    return embeddings


# ──────────────────────────────────────────────────────────────────────────────
#  Live Feed Streaming
# ──────────────────────────────────────────────────────────────────────────────

from fastapi.responses import StreamingResponse

def gen_frames():
    """Video streaming generator function."""
    while True:
        if processor._monitor and processor._monitor.processor:
            frame = processor._monitor.processor.get_latest_frame()
            if frame is not None:
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        # Cap frame rate for the web view
        import time
        time.sleep(0.04) 

@app.get("/video_feed", tags=["System"])
async def video_feed():
    """
    Route to stream the live RTSP feed.
    Usage: <img src="http://localhost:8000/video_feed">
    """
    return StreamingResponse(gen_frames(), media_type='multipart/x-mixed-replace; boundary=frame')


# ══════════════════════════════════════════════════════════════════════════════
#  /health
# ══════════════════════════════════════════════════════════════════════════════

@app.get(
    "/health",
    summary="Liveness check",
    tags=["System"],
)
async def health():
    """
    Returns HTTP 200 with FAISS index stats.
    Use this endpoint for load-balancer health probes.
    """
    idx_total = engine._index.ntotal if engine._index else 0
    return {
        "status":         "ok",
        "faiss_vectors":  idx_total,
        "embedding_dim":  config.EMBEDDING_DIM,
        "threshold":      config.FAISS_L2_THRESHOLD,
        "device":         str(engine._device),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  /onboard
# ══════════════════════════════════════════════════════════════════════════════

@app.post(
    "/onboard",
    summary="Enrol a new employee",
    tags=["Onboarding"],
    status_code=status.HTTP_201_CREATED,
)
async def onboard(
    images:        Annotated[list[UploadFile], File(description="5 or more face images")],
    name:          Annotated[str, Form(description="Employee full name")],
    employee_code: Annotated[str, Form(description="HR system employee code (optional)")] = "",
    department:    Annotated[str, Form(description="Department name (optional)")] = "",
):
    """
    **Onboard a new employee.**

    Upload ≥ 5 frames of the employee's face.  The system will:
    1. Run MediaPipe face detection on each frame.
    2. Extract FaceNet 512-D embeddings.
    3. Compute the **mean embedding** across all valid frames.
    4. Persist the mean embedding as `VARBINARY(MAX)` in MSSQL.
    5. Add the vector to the live FAISS index (no restart needed).
    """
    if not name.strip():
        raise HTTPException(status_code=400, detail="Employee name is required.")

    # ── Handle Update Logic ──────────────────────────────────────────────────
    # Check if employee already exists to allow fewer photos for accuracy updates
    employees = await database.get_all_employees()
    is_update = any(e["name"].lower() == name.strip().lower() for e in employees)
    
    # Requirement: 5+ for new enrol, but 1+ is okay for improving existing
    min_frames = 1 if is_update else config.ONBOARD_FRAMES

    if len(images) < min_frames:
        raise HTTPException(
            status_code=400,
            detail=f"Please upload at least {min_frames} images. Got {len(images)}.",
        )

    embeddings = await _embeddings_from_files(images)

    if len(embeddings) < min_frames:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Only {len(embeddings)} frame(s) contained a detectable face "
                f"(need >= {min_frames}).  "
                "Ensure the images are well-lit and the face is clearly visible."
            ),
        )

    # ── Mean embedding ────────────────────────────────────────────────────────
    mean_emb = np.mean(np.stack(embeddings), axis=0).astype(np.float32)
    norm     = np.linalg.norm(mean_emb)
    if norm > 0:
        mean_emb /= norm   # re-normalise mean

    # ── Persist to MSSQL (Additive merge if name exists) ─────────────────────
    emp_id = await database.upsert_employee(
        name          = name.strip(),
        embedding      = mean_emb,
        employee_code  = employee_code.strip(),
        department     = department.strip(),
        num_images     = len(embeddings),
    )

    # ── Update live FAISS index (Incremental update is faster) ───────────
    engine.add_to_index(employee_id=emp_id, embedding=mean_emb)

    log.info(
        "Updated profile for '%s' (id=%d) with %d new frames.",
        name, emp_id, len(embeddings),
    )

    return {
        "ok":            True,
        "employee_id":   emp_id,
        "name":          name.strip(),
        "frames_used":   len(embeddings),
        "frames_total":  len(images),
        "message":       f"✓ '{name}' enrolled successfully.",
    }


# ══════════════════════════════════════════════════════════════════════════════
#  /access
# ══════════════════════════════════════════════════════════════════════════════

@app.post(
    "/access",
    summary="Verify identity and unlock door",
    tags=["Access"],
)
async def access(
    image: Annotated[UploadFile, File(description="Single face image for recognition")],
):
    """
    **Door access verification.**

    1. Detect face with MediaPipe → embed with FaceNet.
    2. Search FAISS `IndexFlatL2` for nearest neighbour.
    3. If distance ≤ threshold → identity confirmed → trigger door unlock API.
    4. Log the access event (employee_id, distance, door_api_ok) to MSSQL.

    Returns `granted: true` and employee details on a successful match.
    Returns `granted: false` with the FAISS distance on failure.
    """
    data = await image.read()

    # ── Embed ─────────────────────────────────────────────────────────────────
    emb = await engine.extract_embedding(data)
    if emb is None:
        raise HTTPException(
            status_code=422,
            detail="No face detected in the uploaded image.",
        )

    # ── FAISS search ─────────────────────────────────────────────────────────
    emp_id, distance = engine.search_index(emb)

    if emp_id is None:
        # Unknown face
        await database.log_access(employee_id=None, distance=distance, door_ok=False)
        log.info("Access DENIED – distance=%.4f (threshold=%.4f).", distance, config.FAISS_L2_THRESHOLD)
        return JSONResponse(
            status_code=200,
            content={
                "granted":  False,
                "distance": round(distance, 6),
                "message":  "Identity not recognised.",
            },
        )

    # ── Fetch employee info for the response ──────────────────────────────────
    employees = await database.get_all_employees()
    employee  = next((e for e in employees if e["id"] == emp_id), None)
    emp_name  = employee["name"] if employee else f"employee_{emp_id}"

    # ── Unlock door ───────────────────────────────────────────────────────────
    emp_code = employee.get("employee_code", "") if employee else ""
    door_ok = await engine.unlock_door(emp_name, employee_code=emp_code)

    # ── Log to MSSQL ──────────────────────────────────────────────────────────
    await database.log_access(employee_id=emp_id, distance=distance, door_ok=door_ok)

    log.info(
        "Access GRANTED – employee='%s' (id=%d)  distance=%.4f  door_ok=%s",
        emp_name, emp_id, distance, door_ok,
    )

    return {
        "granted":       True,
        "employee_id":   emp_id,
        "name":          emp_name,
        "employee_code": employee.get("employee_code", "") if employee else "",
        "department":    employee.get("department", "") if employee else "",
        "distance":      round(distance, 6),
        "door_unlocked": door_ok,
        "message":       f"✓ Access granted to '{emp_name}'.",
    }


# ──────────────────────────────────────────────────────────────────────────────
#  Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=config.API_HOST, port=config.API_PORT, reload=True)
