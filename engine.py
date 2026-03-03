"""
engine.py – FaceNet (MTCNN detect + InceptionResnetV1 embed) + FAISS core.

Architecture Notes
------------------
MediaPipe 0.10+ dropped the legacy `mp.solutions` API; it now requires a
separate .task model file for face detection.  Instead we use MTCNN from
facenet-pytorch, which is the canonical companion to InceptionResnetV1,
already installed, and requires no external downloads.

Public API
----------
load_index()                        – Build FAISS index from DB on startup
add_to_index(employee_id, embedding) – Append one vector to the live index
search_index(embedding)             – Return (employee_id, distance) or (None, inf)
extract_embedding(image_bytes)      – bytes  → 512-D float32 np.ndarray | None
unlock_door()                       – Async door-unlock call (configurable)
"""

import asyncio
import io
import logging
from functools import partial
import cv2
import httpx
import numpy as np
import faiss
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

from typing import Optional, Tuple, Union, List

import config

log = logging.getLogger("engine")

# ══════════════════════════════════════════════════════════════════════════════
#  1.  Device
# ══════════════════════════════════════════════════════════════════════════════

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ══════════════════════════════════════════════════════════════════════════════
#  2.  MTCNN face detector  (lazy singleton)
#      Used for real-time face detection + crop to 160×160
# ══════════════════════════════════════════════════════════════════════════════

_mtcnn: Optional[MTCNN] = None


def _get_mtcnn() -> MTCNN:
    global _mtcnn
    if _mtcnn is None:
        log.info("[Engine] Loading MTCNN face detector ...")
        _mtcnn = MTCNN(
            image_size=160,        # output crop size expected by InceptionResnetV1
            margin=20,             # margin around detected face (pixels)
            min_face_size=40,      # ignore faces smaller than this
            thresholds=[0.6, 0.7, 0.7],  # P-Net, R-Net, O-Net confidence thresholds
            factor=0.709,          # scale factor for image pyramid
            post_process=True,     # apply fixed_image_standardization (divide by 128, sub 1)
            device=_device,
            keep_all=False,        # return only the largest face
        )
        log.info("[Engine] MTCNN ready on %s.", _device)
    return _mtcnn


# ══════════════════════════════════════════════════════════════════════════════
#  3.  FaceNet model  (lazy singleton)
# ══════════════════════════════════════════════════════════════════════════════

_facenet: Optional[InceptionResnetV1] = None


def _get_facenet() -> InceptionResnetV1:
    global _facenet
    if _facenet is None:
        log.info("[Engine] Loading FaceNet InceptionResnetV1 (pretrained=vggface2) ...")
        _facenet = (
            InceptionResnetV1(pretrained="vggface2")
            .eval()
            .to(_device)
        )
        log.info("[Engine] FaceNet ready on %s.", _device)
    return _facenet


# ══════════════════════════════════════════════════════════════════════════════
#  4.  FAISS index  (IndexFlatL2)
# ══════════════════════════════════════════════════════════════════════════════

_index:     Optional[faiss.IndexFlatL2] = None
_index_ids: list[int]                   = []   # position i  →  employee_id


def _make_empty_index() -> faiss.IndexFlatL2:
    return faiss.IndexFlatL2(config.EMBEDDING_DIM)


async def load_index():
    """Rebuild the FAISS index from all embeddings currently stored in MSSQL."""
    global _index, _index_ids

    import database as db  # late import avoids circular at module load

    employees = await db.get_all_employees()
    _index     = _make_empty_index()
    _index_ids = []

    for emp in employees:
        raw = emp.get("embedding")
        if raw is None:
            continue
        vec = db.bytes_to_embedding(raw)
        if vec.shape[0] != config.EMBEDDING_DIM:
            continue
        _index.add(vec.reshape(1, -1).astype(np.float32))
        _index_ids.append(emp["id"])

    log.info("[Engine] FAISS index loaded with %d vector(s).", _index.ntotal)


def add_to_index(employee_id: int, embedding: np.ndarray):
    """Add a single embedding to the live FAISS index (no DB write here)."""
    global _index, _index_ids
    if _index is None:
        _index = _make_empty_index()
    _index.add(embedding.reshape(1, -1).astype(np.float32))
    _index_ids.append(employee_id)
    log.info("[Engine] FAISS index now has %d vector(s).", _index.ntotal)


def search_index(embedding: np.ndarray) -> Tuple[Optional[int], float]:
    """
    Search the FAISS index for the nearest neighbour.

    Returns
    -------
    (employee_id, l2_distance)  if a match is within FAISS_L2_THRESHOLD
    (None,        distance)     otherwise
    """
    if _index is None or _index.ntotal == 0:
        return None, float("inf")

    query = embedding.reshape(1, -1).astype(np.float32)
    D, I  = _index.search(query, 1)           # D=distances, I=indices
    dist  = float(D[0][0])
    idx   = int(I[0][0])

    if idx < 0 or idx >= len(_index_ids):
        return None, dist

    if dist <= config.FAISS_L2_THRESHOLD:
        return _index_ids[idx], dist
    return None, dist


# ══════════════════════════════════════════════════════════════════════════════
#  5.  Embedding extraction  (MTCNN detect → InceptionResnetV1 embed)
# ══════════════════════════════════════════════════════════════════════════════

def _compute_embedding_sync(image_input: Union[bytes, np.ndarray]) -> Optional[np.ndarray]:
    """
    Synchronous inference path (called from thread-pool).

    Pipeline
    --------
    1. Input decoding (if bytes) -> PIL RGB image
    2. MTCNN: detect + crop + preprocess to 160×160 tensor
    3. InceptionResnetV1: produce 512-D L2-normalised embedding
    """
    try:
        if isinstance(image_input, bytes):
            pil_img = Image.open(io.BytesIO(image_input)).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            # Convert BGR to RGB (assuming OpenCV input)
            if image_input.shape[2] == 3:
                pil_img = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
            else:
                pil_img = Image.fromarray(image_input)
        else:
            log.warning("[Engine] Unsupported input type: %s", type(image_input))
            return None
    except Exception as exc:
        log.warning("[Engine] Cannot decode image: %s", exc)
        return None

    mtcnn   = _get_mtcnn()
    facenet = _get_facenet()

    # MTCNN returns a (160,160,3) pre-processed float tensor,
    # or None if no face is detected.
    face_tensor = mtcnn(pil_img)  # shape: (3, 160, 160) or None

    if face_tensor is None:
        log.debug("[Engine] No face detected by MTCNN.")
        return None

    # Add batch dimension and move to device
    face_tensor = face_tensor.unsqueeze(0).to(_device)   # (1, 3, 160, 160)

    with torch.no_grad():
        emb = facenet(face_tensor)   # (1, 512)

    vec = emb[0].cpu().numpy().astype(np.float32)

    # L2-normalise (facenet-pytorch already does this internally,
    # but we re-normalise to be safe after any float32 casts)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm

    return vec


async def extract_embedding(image_input: Union[bytes, np.ndarray]) -> Optional[np.ndarray]:
    """Async wrapper: offload heavy inference to thread-pool executor."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, partial(_compute_embedding_sync, image_input)
    )


# ══════════════════════════════════════════════════════════════════════════════
#  6. Secure Door Unlock
# ══════════════════════════════════════════════════════════════════════════════

async def unlock_door(employee_name: str, employee_code: str = "", camera_name: str = "Exit") -> bool:
    """
    Trigger the secure door-unlock API for a specific camera/door.
    """
    if not config.EXTERNAL_API_ENABLED:
        log.info("[Door:%s] Unlock skipped (EXTERNAL_API_ENABLED=False).", camera_name)
        return False

    # Get URL for this specific door, fallback to 'Exit' or first available
    url = config.EXTERNAL_API_URLS.get(camera_name)
    if not url:
        # Fallback to any available URL
        url = next(iter(config.EXTERNAL_API_URLS.values()), "")

    if not url:
        log.warning("[Door:%s] Unlock failed: No URL configured.", camera_name)
        return False

    # Dynamic replacement
    if "{code}" in url:
        url = url.replace("{code}", str(employee_code))
    if "{name}" in url:
        url = url.replace("{name}", str(employee_name))
    if "{door}" in url:
        url = url.replace("{door}", str(camera_name))
    if "{id}" in url:
        url = url.replace("{id}", str(employee_code))

    log.info("[Door:%s] Unlocking for '%s' (Code: %s) -> %s", camera_name, employee_name, employee_code, url)

    try:
        async with httpx.AsyncClient(timeout=config.EXTERNAL_API_TIMEOUT) as client:
            # Using POST as verified earlier
            resp = await client.post(url, json={
                "employee": employee_name, 
                "code": employee_code,
                "door": camera_name,
                "action": "unlock"
            })
        if resp.is_success:
            log.info("[Door:%s] Unlock OK (HTTP %d).", camera_name, resp.status_code)
            return True
        else:
            log.warning("[Door:%s] Unlock HTTP %d: %s", camera_name, resp.status_code, resp.text[:100])
    except Exception as exc:
        log.warning("[Door:%s] Unlock failed: %s", camera_name, exc)

    return False
