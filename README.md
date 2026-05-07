# FaceID – FastAPI Door Access (InsightFace + FAISS)

Enterprise-grade face recognition system with **ArcFace (w600k)**, **FAISS IVF** indexing, and real-time **IP Camera** monitoring.

---

## 📁 Core Architecture

-   **Deep Learning (ArcFace)**: Powered by `insightface` for robust 512-D embedding extraction.
-   **Vector Search (FAISS)**: Uses `IndexHNSWFlat` for sub-millisecond search across thousands of identities.
-   **Vision Pipeline**: Async RTSP stream processing with low-latency decoding.
-   **Cloud/Local Database**: MS SQL Server storage for high-integrity audit logs and identity data.

---

## 📷 Enrollment Options

The system supports two primary ways to register users:

1.  **Direct Capture (New)**: Enroll someone directly from a live IP camera feed. The system captures multiple high-quality frames and computes a stable mean embedding automatically.
2.  **Batch Upload**: Upload multiple existing photos (Min 5 recommended) to generate a robust profile.

---

## ⚡ Performance Optimization

This version is optimized for maximum efficiency:
-   **Frame Resizing**: All AI inference happens on downscaled frames to reduce CPU/GPU load.
-   **Weighted Moving Average**: When you update or re-enroll a user, the system merges new biometric data with old data to gradually improve accuracy.
-   **Consensus Logic**: Requires multiple consecutive detections before triggering a door, preventing "glitch" triggers.

---

## 🚀 Quick Start

1.  **Setup Environment**:
    ```powershell
    .\venv\Scripts\activate
    pip install -r requirements.txt
    ```

2.  **Configure `.env`**:
    Edit your `RTSP_URLS` and `MSSQL_SERVER` credentials.

3.  **Run Application**:
    ```powershell
    python main.py
    ```
    Live Dashboard: `http://localhost:8000`

4.  **Clean Stop (Release Port 8000)**:
    If the port is stuck, run this in PowerShell:
    ```powershell
    Stop-Process -Id (Get-NetTCPConnection -LocalPort 8000).OwningProcess -Force
    ```

5.  **Build Executable**:
    ```powershell
    python make_exe.py
    ```

---

## 🔧 Troubleshooting

| Issue | Resolution |
| :--- | :--- |
| **Only Camera 0 visible** | Ensure `RTSP_URLS` in `.env` is formatted as `Name:URL,Name:URL`. |
| **Access Denied (False Negative)** | Use the "Update Photos" feature to add more frames of the person in different lighting. |
| **High CPU Usage** | You can disable the Background AI Monitoring in the Settings panel in the dashboard. |
| **Database Connection Error** | Verify you have 'ODBC Driver 18 for SQL Server' installed. |

---

## 📦 Requirements

-   Python 3.10+
-   MS SQL Server
-   onnxruntime (GPU recommended)
-   insightface, faiss-cpu
