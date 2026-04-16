import ctypes
import os
import sys

# Windows constants for File Descriptors (Low-level)
STD_OUTPUT_HANDLE = -11
STD_ERROR_HANDLE = -12

def silence_cpp_logs():
    """ 
    Redirects low-level C++ stderr (File Descriptor 2) to null.
    This is the ONLY way to stop the 'VerifyOutputSizes' warnings 
    coming from the ONNX DLL on Windows.
    """
    try:
        # 1. Open os.devnull (the black hole)
        devnull = os.open(os.devnull, os.O_WRONLY)
        # 2. Duplicate the current stderr (so we can restore it later if needed)
        # 3. Force Duplicate devnull into FD 2 (standard error)
        os.dup2(devnull, 2)
        # 4. Success — C++ DLLs will now speak into the void
    except Exception:
        # Fallback if OS-level access is denied (rare)
        pass
