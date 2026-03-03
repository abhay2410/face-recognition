# FaceID – Face Recognition Attendance System

A complete attendance management system using your **CP-E41A IP camera** and Python face recognition.

---

## 📁 Project Structure

```
face/
├── app.py              ← Flask web server (main entry point)
├── config.py           ← ⚙️  Camera URL & settings (edit this first!)
├── camera.py           ← Thread-safe camera stream reader
├── face_engine.py      ← Face encoding & recognition logic
├── recogniser.py       ← Background recognition worker thread
├── database.py         ← SQLite attendance database helpers
├── setup.py            ← One-time setup / dependency check
├── requirements.txt    ← Python packages
├── start.bat           ← Windows quick-launch script
├── data/
│   ├── attendance.db   ← Auto-created SQLite database
│   ├── known_faces/    ← Enrolled face images (per-person folders)
│   ├── exports/        ← CSV exports
│   └── encodings.pkl   ← Cached face encodings (auto-generated)
├── static/
│   ├── style.css       ← Premium dark-theme UI
│   └── snapshots/      ← Recognition snapshots saved here
└── templates/
    ├── base.html        ← Sidebar layout
    ├── index.html       ← Dashboard + live feed
    ├── attendance.html  ← Records & export
    └── enrol.html       ← Face registration
```

---

## ⚡ Quick Start

### Step 1 – Configure your camera

Open **`config.py`** and set your CP-E41A's IP address:

```python
# RTSP stream (recommended)
CAMERA_URL = "rtsp://admin:admin@192.168.1.64:554/stream"

# OR HTTP stream
CAMERA_URL = "http://admin:admin@192.168.1.64:8080/video"

# OR local USB webcam
CAMERA_URL = 0
```

> **Finding your camera IP:** Check your router's DHCP client list, or use the camera's mobile app.  
> **Default credentials:** Usually `admin / admin` or `admin / 12345`.  
> **CP-E41A RTSP port:** 554 (standard). Stream path may vary — try `/stream`, `/ch0`, or `/live`.

### Step 2 – Install & Run

**Option A – Double-click** `start.bat` *(installs everything automatically)*

**Option B – Manual:**
```powershell
cd "C:\Users\Abhay\Desktop\face"
pip install -r requirements.txt
python setup.py
python app.py
```

### Step 3 – Open the app

👉 **http://localhost:5000**

---

## 🎯 How to Use

### 1. Enrol Faces
- Go to **Enrol Face** tab
- Enter the student's Name, Roll No, Department
- Either **capture from webcam** or **upload a photo**
- Click **Enrol** — the system extracts and saves the face encoding

> 💡 Enrol **3–5 photos** per person (different angles/lighting) for best accuracy

### 2. Start Attendance
- Go to **Dashboard**
- Click **Start Camera** — the CP-E41A stream will appear
- The system automatically:
  - Detects faces in real time
  - Matches them to enrolled students
  - Marks attendance in the database (with a snapshot)
  - Shows live status on the dashboard

### 3. View & Export Records
- Go to **Attendance** tab
- Filter by date using the date picker
- Export to **CSV** with one click

---

## ⚙️ Configuration Options (`config.py`)

| Setting | Default | Description |
|---|---|---|
| `CAMERA_URL` | RTSP URL | Camera stream source |
| `FACE_TOLERANCE` | `0.50` | Match strictness (lower = stricter) |
| `COOLDOWN_MINUTES` | `30` | Re-mark interval per person |
| `RECOGNITION_INTERVAL_S` | `2` | Seconds between recognition passes |
| `FLASK_PORT` | `5000` | Web server port |

---

## 🔧 Troubleshooting

| Problem | Fix |
|---|---|
| Camera won't connect | Try changing `CAMERA_URL` from RTSP → HTTP. Check IP & credentials. |
| "No face detected" on enrol | Ensure good lighting; face must be clearly visible and frontal |
| False matches | Lower `FACE_TOLERANCE` to `0.40` or `0.45` |
| Missed recognitions | Raise `FACE_TOLERANCE` slightly (up to `0.55`) |
| Slow FPS | Reduce camera resolution in its app settings |

---

## 📦 Dependencies

- **Flask** – Web framework
- **face_recognition** – dlib-based face recognition
- **opencv-python** – Camera capture & image processing  
- **numpy** – Numerical arrays
- **Pillow** – Image saving
- **pandas** – Data handling
