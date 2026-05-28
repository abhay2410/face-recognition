# Agents Overview

## 📚 Introduction
This document provides a **high‑level description of all logical agents** that power the **FastAPI Door Access System** located in this repository. An *agent* here is any component that encapsulates a distinct responsibility and often runs asynchronously (e.g., background monitors, FAISS index manager, database helper, etc.). Understanding these agents helps developers extend the system, troubleshoot issues, or replace parts with custom implementations.

---

## 🏗️ System Architecture
Below is a **Mermaid flowchart** that illustrates the major agents and their interactions.

```mermaid
flowchart TD
    UI[UI (HTML/JS) 🌐] -->|HTTP Requests| API[FastAPI App 🚀]
    API --> Auth[Auth Agent (cookie check)]
    API -->|Endpoints| Enrol[Enrollment Agent] --> Engine[Engine Agent (FAISS, embeddings)]
    API -->|Endpoints| Access[Access Verification Agent] --> Engine
    API -->|Endpoints| Video[Video Streaming Agent] --> Processor[Camera Processor Agent]
    API -->|Endpoints| Audit[Audit Dashboard Agent] --> DB[Database Agent]
    API -->|Endpoints| Logs[Log Viewer Agent] --> DB
    Processor --> Engine
    Engine --> DB
    DB --> Persistence[SQLite / Filesystem]
    style UI fill:#FFF3E0,stroke:#F57C00,stroke-width:2px
    style API fill:#E3F2FD,stroke:#1976D2,stroke-width:2px
    style Processor fill:#F1F8E9,stroke:#43A047,stroke-width:2px
    style Engine fill:#EDE7F6,stroke:#7E57C2,stroke-width:2px
    style DB fill:#FFEBEE,stroke:#E53935,stroke-width:2px
```

---

## 🧩 Core Agents
| Agent | Location | Responsibility | Key Functions |
|-------|----------|----------------|----------------|
| **FastAPI App** | `main.py` | HTTP entry point, request routing, CORS, lifespan handling | `lifespan`, route decorators (`@app.get`, `@app.post`) |
| **Auth Agent** | `main.py` | Session validation via cookies | `get_current_user`, `login_required` |
| **UI Agent** | `templates/*` | Serves HTML pages for dashboard, enrolment, logs, etc. | `show_dashboard`, `show_onboarding_page`, `show_update_page` |
| **Video Streaming Agent** | `main.py` | Provides async MJPEG streams and snapshots | `_gen_frames_async`, `video_feed`, `snapshot` |
| **Enrollment Agent** | `main.py` | Handles enrolment via uploaded images or live camera | `/enrol/validate-frame`, `/enrol/finalize`, `/onboard`, `/enrol/from-camera` |
| **Access Verification Agent** | `main.py` (not fully shown) | Receives facial embedding, compares against FAISS index, unlocks door | (Endpoints omitted for brevity) |
| **Processor Agent** | `processor.py` | Captures frames from each RTSP camera, runs face detection, maintains latest frame buffer | `MonitoringLoop`, `processor.get_latest_frame` |
| **Engine Agent** | `engine.py` | Embedding extraction, FAISS index management, blur detection, face extraction utilities | `extract_embedding`, `select_diverse_embeddings`, `check_blur`, `load_index`, `load_index_from_disk` |
| **Database Agent** | `database.py` | SQLite‑based persistence for employees, audit logs, detection records | `init_db`, `upsert_employee`, `get_all_employees`, `get_audit_logs`, `clear_old_audit` |
| **Logging Agent** | `main.py` (setup) | Async non‑blocking logging with rotating files and console filtering | `_setup_logging`, `QueueHandler`, `QueueListener` |
| **Health Agent** | `main.py` | Liveness endpoint exposing runtime metrics | `/health` |
| **Audit Dashboard Agent** | `main.py` | Serves audit UI and JSON log retrieval | `/audit`, `/api/audit/logs`, `/api/audit/image/{log_id}` |
| **Log Viewer Agent** | `main.py` | Serves raw logs, download, clear actions | `/logs`, `/api/logs`, `/api/logs/download`, `/api/logs/clear` |

---

## 🔄 Background & Async Agents
1. **Processor Background Monitor** – started in `lifespan` via `processor.start_background_monitoring()`. Continuously reads RTSP streams, runs face detection, updates shared state.
2. **Database Maintenance Loop** – launched in `lifespan` (`database.clear_old_detections_loop()`) to prune stale detection entries.
3. **Audit Log Purge Task** – scheduled on startup (`database.purge_old_audit(days=7)`).
4. **Log Listener Thread** – created by the Logging Agent to off‑load I/O from the event loop.

---

## 📁 File Map (quick navigation)
- **`main.py`** – FastAPI entry, routing, lifespan, UI, health, logging.
- **`engine.py`** – Embedding / FAISS utilities.
- **`processor.py`** – Camera capture & detection loop.
- **`database.py`** – SQLite wrapper for employees & audit data.
- **`config.py`** – Global configuration constants.
- **`templates/`** – Jinja2 HTML templates for UI.
- **`static/`** (if present) – CSS/JS assets for a premium UI (currently minimal).

---

## 🎨 Design Recommendations (Premium UI)
While the backend agents are functional, the front‑end currently uses default Bootstrap styling. To achieve the **“wow” factor** required by the project guidelines, consider:
- Adding a dark‑mode theme with CSS variables.
- Using Google‑Fonts like **Inter** for modern typography.
- Applying subtle glass‑morphism cards for dashboard panels.
- Adding micro‑animations on button hover via **animejs‑animation** skill.
- Generating a professional favicon with the `favicon` skill.

These enhancements can be implemented without touching backend agents.

---

## 🛠️ Extending Agents
If you need to add a new capability (e.g., a *notification agent* for door unlock events), follow these steps:
1. Create a new module (e.g., `notifier.py`).
2. Register the agent in `main.py` either as a background task (`asyncio.create_task`) or as a FastAPI dependency.
3. Add appropriate endpoint(s) or hook(s) to invoke the agent.
4. Document the new agent in this `agents.md` file under a new table row.

---

## 📖 References
- **FastAPI Lifespan** – https://fastapi.tiangolo.com/advanced/events/
- **FAISS Indexing** – https://github.com/facebookresearch/faiss
- **Async Logging** – https://docs.python.org/3/library/logging.handlers.html#queuehandler
- **RTSP Camera Access** – OpenCV `cv2.VideoCapture` with `cv2.CAP_FFMPEG`.

---

*End of agents documentation.*
