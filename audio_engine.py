import subprocess
import threading
import queue
import logging
import os
import wave
import time
import winsound
import config
import audio_udp_streamer
import audio_dsp  # EQ Engine

# Setup dedicated logger
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("AudioEngine")

class AudioEngine:
    """
    Handles audio announcements via Local TTS and/or ESP32 UDP Sink.
    Uses Native Windows Speech Synthesis (System.Speech) for 100% stability.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(AudioEngine, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        
        self.task_queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self._run_worker, name="AudioWorker", daemon=True)
        self.worker_thread.start()
        
        # Initialize UDP streamer if enabled (Supports 'ESP32' or 'UDP' labels)
        if config.AUDIO_MODE in ("ESP32", "UDP", "BOTH"):
            audio_udp_streamer.init_streamer(config.ESP32_AUDIO_IP, config.ESP32_AUDIO_PORT)
            
        self._initialized = True
        log.info(f"Audio Engine initialized using Native Windows Speech. Mode: {config.AUDIO_MODE}")

    def _generate_wav_native(self, text, output_path):
        """Uses PowerShell to generate a high-quality WAV file via native SAPI5."""
        # Map config to SAPI5 values
        # Rate: 160 (default) -> 0. Each 10 pts is roughly 1 speed unit.
        sapi_rate = max(-10, min(10, (config.TTS_VOICE_RATE - 160) // 10))
        sapi_vol = max(0, min(100, int(config.TTS_VOICE_VOLUME * 100)))
        
        ps_script = (
            f"Add-Type -AssemblyName System.Speech; "
            f"$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$s.SelectVoice('{config.TTS_VOICE_NAME}'); "
            f"$s.Rate = {sapi_rate}; "
            f"$s.Volume = {sapi_vol}; "
            f"$s.SetOutputToWaveFile('{output_path}'); "
            f"$s.Speak('{text}'); "
            f"$s.Dispose();"
        )
        try:
            subprocess.run(["powershell", "-Command", ps_script], 
                           check=True, capture_output=True, text=True)
            return True
        except Exception as e:
            log.error(f"Native TTS Generation failed: {e}")
            return False

    def _run_worker(self):
        """Worker thread that processes the queue and handles TTS/Streaming."""
        while True:
            try:
                text = self.task_queue.get(timeout=1.0)
                if text is None: break
                    
                temp_file = os.path.join(os.getcwd(), f"voice_{threading.get_ident()}.wav")
                
                try:
                    # 1. GENERATE NATIVE WAV
                    log.debug(f"Generating Native Speech: '{text}'")
                    if not self._generate_wav_native(text, temp_file):
                        continue
                    
                    if not os.path.exists(temp_file) or os.path.getsize(temp_file) < 500:
                        log.error(f"TTS Failed: Output file is invalid.")
                        continue

                    # 2. LOCAL PLAYBACK
                    if config.AUDIO_MODE in ("LOCAL", "BOTH"):
                        try:
                            winsound.PlaySound(temp_file, winsound.SND_FILENAME | winsound.SND_ASYNC)
                        except Exception as e:
                            log.error(f"Local Playback failed: {e}")

                    # 3. NETWORK STREAMING (UDP / ESP32)
                    if config.AUDIO_MODE in ("ESP32", "UDP", "BOTH") and config.ESP32_AUDIO_IP:
                        with wave.open(temp_file, 'rb') as w:
                            rate = w.getframerate()
                            pcm_raw = w.readframes(w.getnframes())
                        
                        # Apply EQ (Bass/Treble)
                        if config.VOICE_BASS != 0 or config.VOICE_TREBLE != 0:
                            log.debug(f"Applying EQ: Bass={config.VOICE_BASS}dB, Treble={config.VOICE_TREBLE}dB")
                            pcm_raw = audio_dsp.apply_eq(pcm_raw, rate, config.VOICE_BASS, config.VOICE_TREBLE)

                        if audio_udp_streamer.streamer:
                            audio_udp_streamer.streamer.stream_pcm(pcm_raw, sample_rate=rate)

                except Exception as e:
                    log.error(f"Audio Cycle error: {e}")
                finally:
                    # Keep file long enough for winsound and then clean up
                    time.sleep(1.0) 
                    if os.path.exists(temp_file):
                        try: os.remove(temp_file)
                        except: pass

            except queue.Empty:
                continue

    def speak(self, text: str):
        """Adds text to the audio queue for playback."""
        if not text:
            return
        self.task_queue.put(text)

# Singleton instance
audio = AudioEngine()

async def announce_local(text: str):
    """Convenience async wrapper."""
    audio.speak(text)
