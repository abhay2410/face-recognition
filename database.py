"""
database.py – Production MSSQL helpers  [v2 — Production Grade]
=================================================================
Design:
  - All blocking ODBC calls run in asyncio's default ThreadPoolExecutor so the
    FastAPI event loop is never blocked.
  - Thread-local connections: each worker thread reuses its own pyodbc connection
    instead of opening a new TCP socket on every call (was the main DB overhead).
  - In-memory employee cache with TTL (default 5 min). Eliminates DB round-trips
    in the hot recognition loop after first encounter.
  - clear_employee_cache() must be called after any enrolment or update.

Schema:
  employees  – identity + weighted mean ArcFace embedding (VARBINARY MAX)
  access_log – timestamped door-access events per camera
"""

import asyncio
import logging
import threading
import time
from datetime import datetime
from functools import partial, wraps
from typing import Dict, Optional

import numpy as np
import pyodbc

import config

log = logging.getLogger("database")

# ──────────────────────────────────────────────────────────────────────────────
#  Employee cache  (in-memory, TTL-based)
# ──────────────────────────────────────────────────────────────────────────────

_CACHE_TTL = 300.0   # seconds before a cached entry expires

# Structure: emp_id → (data_dict, timestamp)
_employee_cache: Dict[int, dict]  = {}     # raw cache (filled by DB and upsert)
_employee_cache_ts: Dict[int, float] = {}  # insertion timestamps


def _cache_get(emp_id: int) -> Optional[dict]:
    """Return cached employee dict if present and not expired, else None."""
    if emp_id not in _employee_cache:
        return None
    if time.monotonic() - _employee_cache_ts.get(emp_id, 0.0) > _CACHE_TTL:
        # Expired — evict
        _employee_cache.pop(emp_id, None)
        _employee_cache_ts.pop(emp_id, None)
        return None
    return _employee_cache[emp_id]


def _cache_set(emp_id: int, data: dict):
    _employee_cache[emp_id]    = data
    _employee_cache_ts[emp_id] = time.monotonic()


def clear_employee_cache(emp_id: Optional[int] = None):
    """
    Invalidate the employee cache after an enrolment or update.
    Pass emp_id to invalidate a single entry, or None to clear everything.
    """
    if emp_id is not None:
        _employee_cache.pop(emp_id, None)
        _employee_cache_ts.pop(emp_id, None)
    else:
        _employee_cache.clear()
        _employee_cache_ts.clear()

# ──────────────────────────────────────────────────────────────────────────────
#  Connection
# ──────────────────────────────────────────────────────────────────────────────

def _conn_str() -> str:
    return (
        f"DRIVER={{{config.MSSQL_DRIVER}}};"
        f"SERVER={config.MSSQL_SERVER};"
        f"DATABASE={config.MSSQL_DB};"
        f"UID={config.MSSQL_USER};"
        f"PWD={config.MSSQL_PASSWORD};"
        f"TrustServerCertificate={config.MSSQL_TRUST_CERT};"
        "Encrypt=no;"  # Driver 18 compatibility fix for many environments
    )


# ──────────────────────────────────────────────────────────────────────────────
#  Thread-local connection pool
#  Each asyncio thread-pool worker reuses one persistent ODBC connection
#  instead of opening a new TCP socket per call.
# ──────────────────────────────────────────────────────────────────────────────

_tl = threading.local()   # thread-local storage


def _get_conn() -> pyodbc.Connection:
    """
    Returns the thread-local pyodbc connection, creating it if necessary.
    Re-creates the connection if it has been closed or lost.
    """
    conn = getattr(_tl, "conn", None)
    if conn is None:
        conn = pyodbc.connect(_conn_str(), autocommit=True)
        _tl.conn = conn
        return conn
    # Ping the connection — cheap health check
    try:
        conn.cursor().execute("SELECT 1")
        return conn
    except pyodbc.Error:
        log.warning("[DB] Thread-local connection lost — reconnecting.")
        try:
            conn.close()
        except Exception:
            pass
        conn = pyodbc.connect(_conn_str(), autocommit=True)
        _tl.conn = conn
        return conn


def db_retry(max_attempts: int = 3, delay: float = 2.0):
    """Decorator to retry async DB functions manually if they fail due to pyodbc errors."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exc = Exception("Empty retry")
            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except (pyodbc.Error, Exception) as exc:
                    last_exc = exc
                    log.warning("[DB] %s attempt %d/%d failed: %s", func.__name__, attempt, max_attempts, exc)
                    if attempt < max_attempts:
                        await asyncio.sleep(delay)
            raise last_exc
        return wrapper
    return decorator


async def get_connection() -> pyodbc.Connection:
    """Return the thread-local pyodbc connection (non-blocking via executor)."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _get_conn)


# ──────────────────────────────────────────────────────────────────────────────
#  Schema bootstrap
# ──────────────────────────────────────────────────────────────────────────────


def _init_db_sync():
    """
    Create the employees and access_log tables if they do not exist.
    Each DDL statement is executed individually to avoid MSSQL parser issues
    with semicolon-split batches containing IF/BEGIN/END blocks.
    """
    conn = _get_conn()
    cur  = conn.cursor()
    # Note: with thread-local connections we do NOT call conn.close() —
    # the connection is kept alive for the lifetime of this thread.

    # --- employees table -------------------------------------------------------
    try:
        cur.execute("""
            IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='employees' AND xtype='U')
            BEGIN
                CREATE TABLE employees (
                    id             INT IDENTITY(1,1) PRIMARY KEY,
                    name           NVARCHAR(255)    NOT NULL UNIQUE,
                    employee_code  NVARCHAR(100),
                    department     NVARCHAR(100),
                    embedding      VARBINARY(MAX),
                    rf_card        NVARCHAR(100),
                    img_count      INT DEFAULT 0,
                    enrolled_at    DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            END
            ELSE
            BEGIN
                -- Migration: add img_count if not present
                IF NOT EXISTS (SELECT * FROM sys.columns WHERE object_id = OBJECT_ID('employees') AND name = 'img_count')
                BEGIN
                    ALTER TABLE employees ADD img_count INT DEFAULT 0
                END
                -- Migration: add rf_card if not present
                IF NOT EXISTS (SELECT * FROM sys.columns WHERE object_id = OBJECT_ID('employees') AND name = 'rf_card')
                BEGIN
                    ALTER TABLE employees ADD rf_card NVARCHAR(100)
                END
                -- Migration v2.1: add embeddings_multi for multi-anchor accuracy boost
                IF NOT EXISTS (SELECT * FROM sys.columns WHERE object_id = OBJECT_ID('employees') AND name = 'embeddings_multi')
                BEGIN
                    ALTER TABLE employees ADD embeddings_multi VARBINARY(MAX)
                END
                -- Migration: add pc_mac for Remote Start
                IF NOT EXISTS (SELECT * FROM sys.columns WHERE object_id = OBJECT_ID('employees') AND name = 'pc_mac')
                BEGIN
                    ALTER TABLE employees ADD pc_mac NVARCHAR(17)
                END
                -- Migration: add pc_ip for Remote Stop
                IF NOT EXISTS (SELECT * FROM sys.columns WHERE object_id = OBJECT_ID('employees') AND name = 'pc_ip')
                BEGIN
                    ALTER TABLE employees ADD pc_ip NVARCHAR(50)
                END
                -- Migration: add pc_control toggle for Remote Start/Stop
                IF NOT EXISTS (SELECT * FROM sys.columns WHERE object_id = OBJECT_ID('employees') AND name = 'pc_control')
                BEGIN
                    ALTER TABLE employees ADD pc_control BIT DEFAULT 0
                END
            END
        """)
        conn.commit()
    except Exception as exc:
        log.warning("[DB] employees table migration: %s", exc)

    # --- access_log table ------------------------------------------------------
    try:
        cur.execute("""
            IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='access_log' AND xtype='U')
            BEGIN
                CREATE TABLE access_log (
                    id           INT IDENTITY(1,1) PRIMARY KEY,
                    employee_id  INT,
                    matched_at   DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    distance     FLOAT,
                    door_api_ok  BIT DEFAULT 0,
                    door_name    NVARCHAR(50) DEFAULT 'Main',
                    CONSTRAINT FK_AccessLog_Employee
                        FOREIGN KEY (employee_id) REFERENCES employees(id)
                )
            END
            ELSE
            BEGIN
                -- Migration: add door_name if not present
                IF NOT EXISTS (SELECT * FROM sys.columns WHERE object_id = OBJECT_ID('access_log') AND name = 'door_name')
                BEGIN
                    ALTER TABLE access_log ADD door_name NVARCHAR(50) DEFAULT 'Main'
                END
            END
        """)
        conn.commit()
    except Exception as exc:
        log.warning("[DB] access_log table migration: %s", exc)

    # --- recognition_audit table -----------------------------------------------
    try:
        cur.execute("""
            IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='recognition_audit' AND xtype='U')
            BEGIN
                CREATE TABLE recognition_audit (
                    id             INT IDENTITY(1,1) PRIMARY KEY,
                    employee_id    INT NULL,
                    employee_name  NVARCHAR(255),
                    camera_name    NVARCHAR(50),
                    cosine_score   FLOAT,
                    door_granted   BIT DEFAULT 0,
                    is_ambiguous   BIT DEFAULT 0,
                    detected_at    DATETIME DEFAULT CURRENT_TIMESTAMP,
                    face_image     VARBINARY(MAX)
                )
            END
        """)
        conn.commit()
    except Exception as exc:
        log.warning("[DB] recognition_audit table bootstrap: %s", exc)

    # --- system_config table (for FAISS blob) ----------------------------------
    try:
        cur.execute("""
            IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='system_config' AND xtype='U')
            BEGIN
                CREATE TABLE system_config (
                    config_key    NVARCHAR(100) PRIMARY KEY,
                    config_value  VARBINARY(MAX),
                    updated_at    DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            END
        """)
        conn.commit()
    except Exception as exc:
        log.warning("[DB] system_config table bootstrap: %s", exc)

    # Thread-local conn — do NOT close here
    log.info("[DB] Schema initialised.")


@db_retry(max_attempts=5, delay=3.0)
async def init_db():
    """Attempt to initialise the DB schema with retries for network resilience."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _init_db_sync)


# ──────────────────────────────────────────────────────────────────────────────
#  Embedding serialisation helpers
# ──────────────────────────────────────────────────────────────────────────────

def embedding_to_bytes(vec: np.ndarray) -> bytes:
    """Pack a single float32 (512-D) numpy array → raw bytes for VARBINARY."""
    return vec.astype(np.float32).tobytes()


def bytes_to_embedding(raw: bytes) -> np.ndarray:
    """Unpack VARBINARY bytes → float32 numpy array (512-D)."""
    arr = np.frombuffer(raw, dtype=np.float32)
    return arr.copy()  # writable copy


def multi_embeddings_to_bytes(vecs: list) -> bytes:
    """
    Pack a list of float32 (512-D) arrays into a flat byte blob.
    Layout: [N×512 float32] stored row-major.
    N is recoverable as len(blob) // (512 * 4).
    """
    mat = np.vstack([v.astype(np.float32) for v in vecs])   # (N, 512)
    return mat.tobytes()


def bytes_to_multi_embeddings(raw: bytes) -> np.ndarray:
    """
    Unpack a blob → (N, 512) float32 array.
    N = total bytes / (512 * 4 bytes per float32).
    """
    flat = np.frombuffer(raw, dtype=np.float32).copy()
    n    = flat.shape[0] // 512
    return flat.reshape(n, 512)


# ──────────────────────────────────────────────────────────────────────────────
#  Employee CRUD
# ──────────────────────────────────────────────────────────────────────────────

def _upsert_employee_sync(
    name: str,
    embedding: np.ndarray,
    employee_code: str = "",
    department: str = "",
    rf_card: str = "",
    num_images: int = 1,
    multi_embeddings: Optional[list] = None,
):
    """
    Save or update an employee.

    Embedding strategy:
      - `embedding`        : weighted running mean of all enrollment frames
                            (used as the primary/fallback match vector)
      - `multi_embeddings` : list of K most diverse anchor embeddings from this
                            enrollment session — stored as a flat (K×512)
                            byte blob in `embeddings_multi` column.
                            FAISS loads all K anchors per person, dramatically
                            improving recall across angles and expressions.
    """
    raw  = embedding_to_bytes(embedding)
    conn = _get_conn()
    cur  = conn.cursor()

    cur.execute("SELECT id, embedding, img_count FROM employees WHERE name=?", (name,))
    row = cur.fetchone()

    if row:
        emp_id    = row[0]
        old_raw   = row[1]
        old_count = row[2] or 0

        if old_raw:
            old_emb      = bytes_to_embedding(old_raw)
            combined_sum = (old_emb * old_count) + (embedding * num_images)
            new_count    = old_count + num_images
            new_emb      = combined_sum / new_count
            norm = np.linalg.norm(new_emb)
            if norm > 0:
                new_emb /= norm
            raw = embedding_to_bytes(new_emb)
            log.info("[DB] Merged %d frame(s) for '%s' (total=%d).", num_images, name, new_count)
        else:
            new_count = num_images

        if multi_embeddings:
            multi_raw = multi_embeddings_to_bytes(multi_embeddings)
            cur.execute(
                "UPDATE employees SET embedding=?, embeddings_multi=?, img_count=? WHERE id=?",
                (raw, multi_raw, new_count, emp_id),
            )
        else:
            cur.execute(
                "UPDATE employees SET embedding=?, img_count=? WHERE id=?",
                (raw, new_count, emp_id),
            )
    else:
        if multi_embeddings:
            multi_raw = multi_embeddings_to_bytes(multi_embeddings)
            cur.execute(
                "INSERT INTO employees "
                "(name, employee_code, department, rf_card, embedding, embeddings_multi, img_count) "
                "VALUES (?,?,?,?,?,?,?)",
                (name, employee_code, department, rf_card, raw, multi_raw, num_images),
            )
        else:
            cur.execute(
                "INSERT INTO employees "
                "(name, employee_code, department, rf_card, embedding, img_count) "
                "VALUES (?,?,?,?,?,?)",
                (name, employee_code, department, rf_card, raw, num_images),
            )
        cur.execute("SELECT @@IDENTITY")
        emp_id = int(cur.fetchone()[0])

    conn.commit()

    emp_record = {
        "id":            emp_id,
        "name":          name,
        "employee_code": employee_code,
        "department":    department,
        "rf_card":       rf_card,
    }
    _cache_set(emp_id, emp_record)
    return emp_id


@db_retry(max_attempts=3, delay=2.0)
async def upsert_employee(
    name: str,
    embedding: np.ndarray,
    employee_code: str = "",
    department: str = "",
    rf_card: str = "",
    num_images: int = 1,
    multi_embeddings: Optional[list] = None,
) -> int:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, partial(
            _upsert_employee_sync,
            name, embedding, employee_code, department,
            rf_card, num_images, multi_embeddings,
        )
    )


def _get_all_employees_sync() -> list:
    conn = _get_conn()
    cur  = conn.cursor()
    cur.execute(
        "SELECT id, name, employee_code, department, rf_card, pc_mac, pc_ip, pc_control, embedding "
        "FROM employees ORDER BY name"
    )
    cols = [c[0] for c in cur.description]
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    # Warm the metadata cache while we have the data
    for row in rows:
        emp_id = row["id"]
        if _cache_get(emp_id) is None:
            _cache_set(emp_id, {
                "id":            row["id"],
                "name":          row["name"],
                "employee_code": row["employee_code"],
                "department":    row["department"],
                "rf_card":       row["rf_card"],
                "pc_mac":        row["pc_mac"],
                "pc_ip":         row["pc_ip"],
                "pc_control":    row["pc_control"],
            })
    return rows


@db_retry(max_attempts=3, delay=1.0)
async def get_all_employees() -> list:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _get_all_employees_sync)


def _get_all_multi_embeddings_sync() -> list:
    """
    Fetches id + all embeddings (multi + fallback) for every employee.
    Used by engine.load_index() to build the multi-anchor FAISS index.

    Returns list of dicts:
      { 'id': int, 'embeddings': np.ndarray (N, 512) }
    where N = config.MULTI_EMB_COUNT if multi stored, else N = 1 (mean only).
    """
    log.info("[DB] Fetching all embeddings from SQL...")
    conn = _get_conn()
    cur  = conn.cursor()
    try:
        cur.execute(
            "SELECT id, embedding, embeddings_multi FROM employees ORDER BY id"
        )
        results = []
        rows = cur.fetchall()
        log.info("[DB] SQL returned %d rows.", len(rows))
        for row in rows:
            emp_id, raw_mean, raw_multi = row[0], row[1], row[2]

            if raw_multi:
                # Multi-anchor embeddings available — use all of them
                mat = bytes_to_multi_embeddings(bytes(raw_multi))
                results.append({"id": emp_id, "embeddings": mat})
            elif raw_mean:
                # Fallback: single mean embedding (old enrolment)
                vec = bytes_to_embedding(bytes(raw_mean)).reshape(1, 512)
                results.append({"id": emp_id, "embeddings": vec})
            # Else: no embedding at all — skip this employee

        return results
    finally:
        cur.close()
        # conn.commit() is not needed for SELECT, but 
        # let's ensure the cursor is closed.



@db_retry(max_attempts=3, delay=1.0)
async def get_all_multi_embeddings() -> list:
    """Async wrapper for _get_all_multi_embeddings_sync."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _get_all_multi_embeddings_sync)


def _get_employee_by_id_sync(employee_id: int) -> Optional[dict]:
    # Fast path: TTL-aware cache hit
    cached = _cache_get(employee_id)
    if cached is not None:
        return cached

    conn = _get_conn()
    cur  = conn.cursor()
    cur.execute(
        "SELECT id, name, employee_code, department, rf_card, pc_mac, pc_ip, pc_control FROM employees WHERE id=?",
        (employee_id,)
    )
    row = cur.fetchone()
    if not row:
        return None
    cols = [c[0] for c in cur.description]
    res  = dict(zip(cols, row))
    _cache_set(employee_id, res)   # Populate cache for next hit
    return res


@db_retry(max_attempts=3, delay=1.0)
async def get_employee_by_id(employee_id: int) -> Optional[dict]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, partial(_get_employee_by_id_sync, employee_id))


def _delete_employee_sync(employee_id: int):
    conn = _get_conn()
    cur  = conn.cursor()
    cur.execute("DELETE FROM access_log WHERE employee_id=?", (employee_id,))
    cur.execute("DELETE FROM employees  WHERE id=?",          (employee_id,))
    conn.commit()   # Thread-local conn — do NOT close
    clear_employee_cache(employee_id)  # evict from cache immediately


@db_retry(max_attempts=3, delay=1.0)
async def delete_employee(employee_id: int):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, partial(_delete_employee_sync, employee_id))


# ──────────────────────────────────────────────────────────────────────────────
#  Access log
# ──────────────────────────────────────────────────────────────────────────────

def _log_access_sync(employee_id: Optional[int], distance: float, door_ok: bool, door_name: str = "Main"):
    conn = _get_conn()
    cur  = conn.cursor()
    cur.execute(
        "INSERT INTO access_log (employee_id, distance, door_api_ok, door_name) VALUES (?,?,?,?)",
        (employee_id, distance, 1 if door_ok else 0, door_name),
    )
    conn.commit()   # Thread-local conn — do NOT close


@db_retry(max_attempts=3, delay=1.0)
async def log_access(employee_id: Optional[int], distance: float, door_ok: bool, door_name: str = "Main"):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None, partial(_log_access_sync, employee_id, distance, door_ok, door_name)
    )


# ──────────────────────────────────────────────────────────────────────────────
#  Recognition Audit & FAISS Storage
# ──────────────────────────────────────────────────────────────────────────────

def _log_audit_snapshot_sync(
    employee_id: Optional[int],
    name: str,
    camera: str,
    score: float,
    granted: bool,
    is_ambiguous: bool,
    image_bytes: bytes
):
    conn = _get_conn()
    cur  = conn.cursor()
    cur.execute("""
        INSERT INTO recognition_audit 
        (employee_id, employee_name, camera_name, cosine_score, door_granted, is_ambiguous, face_image)
        VALUES (?,?,?,?,?,?,?)
    """, (employee_id, name, camera, score, 1 if granted else 0, 1 if is_ambiguous else 0, image_bytes))
    conn.commit()


@db_retry(max_attempts=3, delay=1.0)
async def log_audit_snapshot(
    employee_id: Optional[int],
    name: str,
    camera: str,
    score: float,
    granted: bool,
    is_ambiguous: bool,
    image_bytes: bytes
):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None, partial(
            _log_audit_snapshot_sync,
            employee_id, name, camera, score, granted, is_ambiguous, image_bytes
        )
    )


def _get_audit_logs_sync(limit: int = 100, ambiguous_only: bool = False) -> list:
    conn = _get_conn()
    cur  = conn.cursor()
    query = "SELECT TOP (?) id, employee_id, employee_name, camera_name, cosine_score, door_granted, is_ambiguous, detected_at FROM recognition_audit"
    if ambiguous_only:
        query += " WHERE is_ambiguous = 1"
    query += " ORDER BY detected_at DESC"
    
    cur.execute(query, (limit,))
    cols = [c[0] for c in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


@db_retry(max_attempts=3, delay=1.0)
async def get_audit_logs(limit: int = 100, ambiguous_only: bool = False) -> list:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, partial(_get_audit_logs_sync, limit, ambiguous_only))


def _get_audit_image_sync(log_id: int) -> Optional[bytes]:
    conn = _get_conn()
    cur  = conn.cursor()
    cur.execute("SELECT face_image FROM recognition_audit WHERE id=?", (log_id,))
    row = cur.fetchone()
    return bytes(row[0]) if row and row[0] else None


@db_retry(max_attempts=3, delay=1.0)
async def get_audit_image(log_id: int) -> Optional[bytes]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, partial(_get_audit_image_sync, log_id))


def _purge_old_audit_sync(days: int):
    conn = _get_conn()
    cur  = conn.cursor()
    # Direct SQL filter for Time
    cur.execute("DELETE FROM recognition_audit WHERE detected_at < DATEADD(day, ?, GETDATE())", (-days,))
    conn.commit()
    log.info("[DB] Purged audit records older than %d days.", days)


@db_retry(max_attempts=3, delay=1.0)
async def purge_old_audit(days: int = 7):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, partial(_purge_old_audit_sync, days))


def _save_faiss_index_sync(blob: bytes):
    conn = _get_conn()
    cur  = conn.cursor()
    cur.execute("""
        IF EXISTS (SELECT 1 FROM system_config WHERE config_key = 'faiss_index')
            UPDATE system_config SET config_value = ?, updated_at = GETDATE() WHERE config_key = 'faiss_index'
        ELSE
            INSERT INTO system_config (config_key, config_value) VALUES ('faiss_index', ?)
    """, (blob, blob))
    conn.commit()
    log.info("[DB] FAISS index blob saved to SQL system_config.")


@db_retry(max_attempts=3, delay=2.0)
async def save_faiss_index(blob: bytes):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, partial(_save_faiss_index_sync, blob))


def _load_faiss_index_sync() -> Optional[bytes]:
    conn = _get_conn()
    cur  = conn.cursor()
    cur.execute("SELECT config_value FROM system_config WHERE config_key = 'faiss_index'")
    row = cur.fetchone()
    return bytes(row[0]) if row and row[0] else None


@db_retry(max_attempts=3, delay=1.0)
async def load_faiss_index() -> Optional[bytes]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _load_faiss_index_sync)


# ──────────────────────────────────────────────────────────────────────────────
#  Background Maintenance
# ──────────────────────────────────────────────────────────────────────────────

async def clear_old_detections_loop(days: int = 2):
    """
    Background worker that runs every 6 hours.
    Keeps the access_log table lean by purging records older than 'days'.
    (Crucial for 2GB VRAM machines where SQL server shares memory).
    """
    await asyncio.sleep(60)  # Initial delay to let startup finish
    while True:
        try:
            log.info("[DB] Starting periodic maintenance...")
            await purge_old_audit(days=7)  # Clear the audit snapshots
            
            # Also clear the raw access_log
            conn = await get_connection()
            def _purge_log():
                c = conn.cursor()
                c.execute("DELETE FROM access_log WHERE matched_at < DATEADD(day, ?, GETDATE())", (-days,))
                conn.commit()
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, _purge_log)
            log.info("[DB] Maintenance complete. Access logs older than %d days purged.", days)
            
        except Exception as e:
            log.error("[DB] Maintenance loop error: %s", e)
        
        await asyncio.sleep(6 * 3600)  # Run every 6 hours
