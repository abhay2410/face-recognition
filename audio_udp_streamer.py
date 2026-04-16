import socket
import time
import logging

log = logging.getLogger("AudioStreamer")

class UDPStreamer:
    def __init__(self, ip, port, chunk_size=400):
        self.ip = ip
        self.port = port
        self.chunk_size = chunk_size
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Increase buffer for higher throughput if needed
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65536)

    def stream_pcm(self, pcm_data: bytes, sample_rate: int = 22050):
        """
        Streams PCM data over UDP with dynamic pacing.
        Calculates delay exactly to match the audio duration of the chunk.
        """
        if not pcm_data:
            return

        total_chunks = (len(pcm_data) + self.chunk_size - 1) // self.chunk_size
        
        # Calculate exact duration of one chunk in seconds
        # duration = bytes / (rate * channels * width)
        # channels=1, width=2 (16-bit mono)
        chunk_duration = self.chunk_size / (sample_rate * 1 * 2)
        
        # Pacing: Use 95% of duration to stay slightly ahead and prevent underflow
        pacing_delay = chunk_duration * 0.95
        
        # log.debug(f"[UDP] Starting stream: {len(pcm_data)} bytes, {total_chunks} chunks. Pacing: {pacing_delay*1000:.2f}ms")

        sent_count = 0
        try:
            for i in range(0, len(pcm_data), self.chunk_size):
                chunk = pcm_data[i:i + self.chunk_size]
                self.sock.sendto(chunk, (self.ip, self.port))
                sent_count += 1
                
                # Dynamic Delay
                time.sleep(pacing_delay)

            log.info(f"[UDP] Stream complete: {sent_count}/{total_chunks} chunks sent.")
        except Exception as e:
            log.error(f"[UDP] Stream error: {e}")

# Singleton instance management
streamer = None

def init_streamer(ip, port):
    global streamer
    if ip:
        streamer = UDPStreamer(ip, port)
