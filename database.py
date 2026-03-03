"""
db.py – Async-friendly MSSQL helpers using pyodbc (thread-pool bridged via asyncio).

Schema managed here:
  employees  – identity + mean FaceNet embedding stored as VARBINARY(MAX)
  access_log – timestamped door-access events
"""

import asyncio
import logging
import struct
from datetime import datetime
from functools import partial, wraps
from typing import Optional, Dict

import numpy as np
import pyodbc

import config

log = logging.getLogger("database")

# In-memory cache for employee metadata (ID -> Dict)
_employee_cache: Dict[int, dict] = {}

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


def _get_conn() -> pyodbc.Connection:
    return pyodbc.connect(_conn_str())


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
                    print(f"[DB] {func.__name__} attempt {attempt}/{max_attempts} failed: {exc}")
                    if attempt < max_attempts:
                        await asyncio.sleep(delay)
            raise last_exc
        return wrapper
    return decorator


async def get_connection() -> pyodbc.Connection:
    """Return a new pyodbc connection (run in thread-pool so we don't block event loop)."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _get_conn)


# ──────────────────────────────────────────────────────────────────────────────
#  Schema bootstrap
# ──────────────────────────────────────────────────────────────────────────────

_SCHEMA_SQL = """
-- Employees table: stores identity + 512-D FaceNet mean embedding
IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='employees' AND xtype='U')
BEGIN
    CREATE TABLE employees (
        id             INT IDENTITY(1,1) PRIMARY KEY,
        name           NVARCHAR(255)    NOT NULL UNIQUE,
        employee_code  NVARCHAR(100),
        department     NVARCHAR(100),
        embedding      VARBINARY(MAX),   -- 512×4 bytes (float32 little-endian)
        enrolled_at    DATETIME DEFAULT CURRENT_TIMESTAMP
    );
END;

-- Access log table
IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='access_log' AND xtype='U')
BEGIN
    CREATE TABLE access_log (
        id          INT IDENTITY(1,1) PRIMARY KEY,
        employee_id INT,
        matched_at  DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        distance    FLOAT,            -- FAISS L2 distance
        door_api_ok BIT DEFAULT 0,
        CONSTRAINT FK_AccessLog_Employee FOREIGN KEY (employee_id)
            REFERENCES employees(id)
    );
END;
"""


def _init_db_sync():
    """
    Create the employees and access_log tables if they do not exist.
    Each DDL statement is executed individually to avoid MSSQL parser issues
    with semicolon-split batches containing IF/BEGIN/END blocks.
    """
    conn = _get_conn()
    cur  = conn.cursor()

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
            END
        """)
        conn.commit()
    except Exception as exc:
        print(f"[DB] employees table: {exc}")

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
        print(f"[DB] access_log table: {exc}")

    conn.close()
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
    """Pack float32 numpy array → raw bytes for VARBINARY(MAX)."""
    return vec.astype(np.float32).tobytes()


def bytes_to_embedding(raw: bytes) -> np.ndarray:
    """Unpack VARBINARY bytes → float32 numpy array."""
    arr = np.frombuffer(raw, dtype=np.float32)
    return arr.copy()  # writable copy


# ──────────────────────────────────────────────────────────────────────────────
#  Employee CRUD
# ──────────────────────────────────────────────────────────────────────────────

def _upsert_employee_sync(
    name: str,
    embedding: np.ndarray,
    employee_code: str = "",
    department: str = "",
    num_images: int = 1,
) -> int:
    """
    Saves or updates an employee. If updating, it performs a weighted 
    average of the new embedding with the existing one to improve accuracy.
    """
    raw  = embedding_to_bytes(embedding)
    conn = _get_conn()
    cur  = conn.cursor()

    cur.execute("SELECT id, embedding, img_count FROM employees WHERE name=?", (name,))
    row = cur.fetchone()

    if row:
        emp_id = row[0]
        old_raw = row[1]
        old_count = row[2] or 0
        
        if old_raw:
            old_emb = bytes_to_embedding(old_raw)
            # Weighted average to merge the new batch of photos
            combined_sum = (old_emb * old_count) + (embedding * num_images)
            new_count = old_count + num_images
            new_emb = combined_sum / new_count
            
            # Re-normalize for FAISS L2 consistency
            norm = np.linalg.norm(new_emb)
            if norm > 0:
                new_emb /= norm
                
            raw = embedding_to_bytes(new_emb)
            print(f"[DB] Merged {num_images} new photos for '{name}'. Total photos: {new_count}")
        else:
            new_count = num_images

        cur.execute(
            "UPDATE employees SET embedding=?, employee_code=?, department=?, img_count=? WHERE id=?",
            (raw, employee_code, department, new_count, emp_id),
        )
    else:
        cur.execute(
            "INSERT INTO employees (name, employee_code, department, embedding, img_count) "
            "VALUES (?,?,?,?,?)",
            (name, employee_code, department, raw, num_images),
        )
        cur.execute("SELECT @@IDENTITY")
        emp_id = int(cur.fetchone()[0])

    # Update cache
    _employee_cache[emp_id] = {
        "id": emp_id,
        "name": name,
        "employee_code": employee_code,
        "department": department
    }
    
    conn.commit()
    conn.close()
    return emp_id


@db_retry(max_attempts=3, delay=2.0)
async def upsert_employee(
    name: str,
    embedding: np.ndarray,
    employee_code: str = "",
    department: str = "",
    num_images: int = 1,
) -> int:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, partial(_upsert_employee_sync, name, embedding, employee_code, department, num_images)
    )


def _get_all_employees_sync() -> list[dict]:
    conn = _get_conn()
    cur  = conn.cursor()
    cur.execute(
        "SELECT id, name, employee_code, department, embedding FROM employees ORDER BY name"
    )
    cols = [c[0] for c in cur.description]
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    conn.close()
    return rows


@db_retry(max_attempts=3, delay=1.0)
async def get_all_employees() -> list[dict]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _get_all_employees_sync)


def _get_employee_by_id_sync(employee_id: int) -> Optional[dict]:
    # Check cache first
    if employee_id in _employee_cache:
        return _employee_cache[employee_id]

    conn = _get_conn()
    cur  = conn.cursor()
    cur.execute(
        "SELECT id, name, employee_code, department FROM employees WHERE id=?", (employee_id,)
    )
    row = cur.fetchone()
    if not row:
        conn.close()
        return None
    cols = [c[0] for c in cur.description]
    res = dict(zip(cols, row))
    conn.close()

    # Populate cache
    _employee_cache[employee_id] = res
    return res


@db_retry(max_attempts=3, delay=1.0)
async def get_employee_by_id(employee_id: int) -> Optional[dict]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, partial(_get_employee_by_id_sync, employee_id))


def _delete_employee_sync(employee_id: int):
    conn = _get_conn()
    cur  = conn.cursor()
    cur.execute("DELETE FROM access_log  WHERE employee_id=?", (employee_id,))
    cur.execute("DELETE FROM employees   WHERE id=?",          (employee_id,))
    conn.commit()
    conn.close()


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
    conn.commit()
    conn.close()


@db_retry(max_attempts=3, delay=1.0)
async def log_access(employee_id: Optional[int], distance: float, door_ok: bool, door_name: str = "Main"):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None, partial(_log_access_sync, employee_id, distance, door_ok, door_name)
    )
