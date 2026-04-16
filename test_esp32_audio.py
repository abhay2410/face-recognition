import sys
import os
import time

# Ensure we can import our modules
sys.path.append(os.getcwd())

import audio_engine
import config

def test_esp32():
    print("--- ESP32 Network Audio Test ---")
    print(f"Target IP: {config.ESP32_AUDIO_IP}")
    print(f"Mode: {config.AUDIO_MODE}")
    
    message = "Employee 1,   ID   detected.   Welcome."
    print(f"Sending audio: '{message}'")
    
    # This will generate TTS and stream UDP packets if mode is ESP32
    audio_engine.audio.speak(message)
    
    # Pacing: Give time for the whole message to stream
    print("Streaming in progress...")
    time.sleep(8)
    print("Test finished.")

if __name__ == "__main__":
    test_esp32()
