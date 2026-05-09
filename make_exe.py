import os
import subprocess
import sys

def build():
    print("Starting build process for Face Recognition EXE...")

    # 1. Detect and use venv if present
    base_dir = os.path.dirname(os.path.abspath(__file__))
    venv_python = os.path.join(base_dir, "venv", "Scripts", "python.exe")

    if os.path.exists(venv_python) and sys.executable != venv_python:
        print(f"Detected virtual environment at {venv_python}. Rerunning build with venv...")
        subprocess.check_call([venv_python, __file__])
        return

    # 2. Ensure PyInstaller is installed
    try:
        import PyInstaller
    except ImportError:
        print("PyInstaller not found. Installing in current environment...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])

    # 3. Define paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    main_script = os.path.join(base_dir, "main.py")   # <-- directly use main.py

    templates_dir = os.path.join(base_dir, "templates")
    models_dir    = os.path.join(base_dir, "data", "insightface_models")

    # 4. Build PyInstaller command
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm",
        "--onedir",
        "--contents-directory", "_internal",
        "--noconsole",          # Hide console window for background production use
        "--name", "FaceAccessSystem",
        "--icon", os.path.join(base_dir, "logo.ico"),
        # Bundle templates & models
        f"--add-data={templates_dir}{os.pathsep}templates",
        f"--add-data={models_dir}{os.pathsep}data/insightface_models",
        f"--add-data={os.path.join(base_dir, 'logo.ico')}{os.pathsep}.",
        f"--add-data={os.path.join(base_dir, 'logo.png')}{os.pathsep}.",
        # Collect all dependencies
        "--collect-all", "fastapi",
        "--collect-all", "uvicorn",
        "--collect-all", "pydantic",
        "--collect-all", "insightface",
        "--collect-all", "faiss",
        # Exclude massive unused libraries (keeps total size manageable)
        "--exclude-module", "tensorflow",
        "--exclude-module", "torch",
        "--exclude-module", "torchvision",
        "--exclude-module", "facenet-pytorch",
        "--exclude-module", "mkl",
        "--exclude-module", "PIL._tkinter_finder",
        # Hidden imports
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
        # Entry point
        main_script
    ]

    print(f"Running PyInstaller (Production Mode)...")

    try:
        subprocess.check_call(cmd)
        print("\n" + "="*50)
        print("PRODUCTION BUILD SUCCESSFUL!")
        print(f"Executable: {os.path.join(base_dir, 'dist', 'FaceAccessSystem', 'FaceAccessSystem.exe')}")
        print("="*50)
        print("\nDeployment Instructions:")
        print("1. Copy the entire 'dist/FaceAccessSystem' FOLDER to the client system.")
        print("2. Copy your '.env' file into that same folder (next to the .exe).")
        print("3. Run the .exe to start the system.")
        print("4. Access via http://localhost:8000")
    except subprocess.CalledProcessError as e:
        print(f"Build failed: {e}")

if __name__ == "__main__":
    build()
