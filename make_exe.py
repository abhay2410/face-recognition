# File: c:/Users/Abhay/Desktop/face/make_exe.py
# -------------------------------------------------
import os
import subprocess
import sys
import shutil
import datetime

def _force_remove_dir(path):
    """Force-remove a directory on Windows, killing locked files via rd /s /q."""
    if not os.path.exists(path):
        return
    print(f"Removing existing build directory: {path}")
    # Use Windows rd command which is more forceful than shutil on locked files
    result = subprocess.run(
        ["cmd", "/c", "rd", "/s", "/q", path],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or os.path.exists(path):
        # Fallback: try shutil
        try:
            shutil.rmtree(path, ignore_errors=True)
        except Exception:
            pass

def _kill_process_by_port(port=8000):
    """Kill whatever process is listening on the given port (e.g. 8000) to prevent directory lock."""
    try:
        # Run netstat to find PID listening on the port
        result = subprocess.run(
            ["netstat", "-ano"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if f"0.0.0.0:{port}" in line or f"127.0.0.1:{port}" in line or f"[::]:{port}" in line:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        pid = parts[-1]
                        if pid.isdigit() and int(pid) != os.getpid():
                            print(f"Killing process on port {port} (PID: {pid})...")
                            subprocess.run(["taskkill", "/f", "/pid", pid], capture_output=True)
    except Exception as e:
        print(f"Warning: could not kill process on port {port}: {e}")

def _kill_running_exe(name="FaceAccessSystem.exe"):
    """Kill any running instance of the built exe and tunnel so their files are not locked."""
    # Kill the main exe
    result = subprocess.run(
        ["taskkill", "/f", "/im", name],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        print(f"Killed running process: {name}")
    
    # Kill cloudflared tunnel client (which might hold locks on dist)
    result_cf = subprocess.run(
        ["taskkill", "/f", "/im", "cloudflared.exe"],
        capture_output=True,
        text=True,
    )
    if result_cf.returncode == 0:
        print("Killed running cloudflared process(es)")
        
    # Also kill any dev server on port 8000
    _kill_process_by_port(8000)

def build():
    print("=== Building Face Recognition EXE ==================================")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    venv_python = os.path.join(base_dir, "venv", "Scripts", "python.exe")

    # -----------------------------------------------------------------
    # 1. Use virtual-env if present (ensures deterministic build)
    # -----------------------------------------------------------------
    if os.path.exists(venv_python) and sys.executable != venv_python:
        print(f"Detected virtual environment at {venv_python}. Re-launching build...")
        subprocess.check_call([venv_python, __file__])
        return

    # -----------------------------------------------------------------
    # 2. Clean previous build artefacts (handle locked files)
    # -----------------------------------------------------------------
    _kill_running_exe()
    _force_remove_dir(os.path.join(base_dir, "dist"))
    _force_remove_dir(os.path.join(base_dir, "build"))

    # -----------------------------------------------------------------
    # 3. Ensure PyInstaller is available
    # -----------------------------------------------------------------
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print("PyInstaller not found – installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])

    # -----------------------------------------------------------------
    # 4. Assemble PyInstaller command
    # -----------------------------------------------------------------
    templates_dir = os.path.join(base_dir, "templates")
    models_dir    = os.path.join(base_dir, "data", "insightface_models")

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm",
        "--onedir",
        "--contents-directory", "_internal",
        "--noconsole",
        "--name", "FaceAccessSystem",
        "--icon", os.path.join(base_dir, "logo.ico"),
        # --- bundled data -------------------------------------------------
        f"--add-data={templates_dir}{os.pathsep}templates",
        f"--add-data={models_dir}{os.pathsep}data/insightface_models",
        f"--add-data={os.path.join(base_dir, 'logo.ico')}{os.pathsep}.",
        f"--add-data={os.path.join(base_dir, 'logo.png')}{os.pathsep}.",
        # --- collect required packages ------------------------------------
        "--collect-all", "fastapi",
        "--collect-all", "uvicorn",
        "--collect-all", "pydantic",
        "--collect-all", "insightface",
        "--collect-all", "faiss",
        # --- exclude heavy, unused libs -----------------------------------
        "--exclude-module", "tensorflow",
        "--exclude-module", "torch",
        "--exclude-module", "torchvision",
        "--exclude-module", "facenet-pytorch",
        "--exclude-module", "mkl",
        "--exclude-module", "PIL._tkinter_finder",
        # --- hidden imports (Uvicorn sub-modules) -----------------------
        "--hidden-import=uvicorn.logging",
        "--hidden-import=uvicorn.loops",
        "--hidden-import=uvicorn.loops.auto",
        "--hidden-import=uvicorn.protocols",
        "--hidden-import=uvicorn.protocols.http",
        "--hidden-import=uvicorn.protocols.http.auto",
        "--hidden-import=onnxruntime",
        "--hidden-import=engine",
        "--hidden-import=database",
        "--hidden-import=processor",
        "--hidden-import=config",
        # --- entry point -------------------------------------------------
        os.path.join(base_dir, "main.py"),
    ]

    print("Running PyInstaller (production mode)...")
    try:
        subprocess.check_call(cmd)
        exe_path = os.path.join(base_dir, "dist", "FaceAccessSystem", "FaceAccessSystem.exe")
        print("\n" + "=" * 50)
        print("[SUCCESS] PRODUCTION BUILD SUCCESSFUL!")
        print(f"Executable: {exe_path}")
        print("=" * 50)
        print("\nDeployment instructions:")
        print("1. Copy the entire 'dist/FaceAccessSystem' folder to the target machine.")
        print("2. Copy your .env file into that folder (next to the .exe).")
        print("3. Run FaceAccessSystem.exe")
        print("4. Access the API at http://localhost:8000")
    except subprocess.CalledProcessError as e:
        # -----------------------------------------------------------------
        # 5. Write any fatal error to a persistent log for later review
        # -----------------------------------------------------------------
        err_log = os.path.join(base_dir, "build_error.log")
        with open(err_log, "w", encoding="utf-8") as f:
            f.write(f"Build failed at {datetime.datetime.now(datetime.timezone.utc).isoformat()}\n")
            f.write(str(e))
        print(f"[ERROR] Build failed - see {err_log}")
        raise

if __name__ == "__main__":
    build()
