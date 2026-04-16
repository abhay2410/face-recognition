import math
import struct
import array

class BiquadFilter:
    """
    A simple Biquad filter implementation for PCM audio.
    Supports Low-shelf (Bass) and High-shelf (Treble) filters.
    """
    def __init__(self, filter_type, sample_rate, freq, q, gain_db):
        self.x1 = self.x2 = self.y1 = self.y2 = 0
        
        A = 10**(gain_db / 40)
        omega = 2 * math.pi * freq / sample_rate
        sn = math.sin(omega)
        cs = math.cos(omega)
        alpha = sn / (2 * q)
        beta = math.sqrt(A) / q

        if filter_type == "lowshelf":
            self.b0 = A * ((A + 1) - (A - 1) * cs + beta * sn)
            self.b1 = 2 * A * ((A - 1) - (A + 1) * cs)
            self.b2 = A * ((A + 1) - (A - 1) * cs - beta * sn)
            self.a0 = (A + 1) + (A - 1) * cs + beta * sn
            self.a1 = -2 * ((A - 1) + (A + 1) * cs)
            self.a2 = (A + 1) + (A - 1) * cs - beta * sn
        elif filter_type == "highshelf":
            self.b0 = A * ((A + 1) + (A - 1) * cs + beta * sn)
            self.b1 = -2 * A * ((A - 1) + (A + 1) * cs)
            self.b2 = A * ((A + 1) + (A - 1) * cs - beta * sn)
            self.a0 = (A + 1) - (A - 1) * cs + beta * sn
            self.a1 = 2 * ((A - 1) - (A + 1) * cs)
            self.a2 = (A + 1) - (A - 1) * cs - beta * sn

        # Normalize
        self.b0 /= self.a0
        self.b1 /= self.a0
        self.b2 /= self.a0
        self.a1 /= self.a0
        self.a2 /= self.a0

    def process(self, sample):
        out = self.b0 * sample + self.b1 * self.x1 + self.b2 * self.x2 - self.a1 * self.y1 - self.a2 * self.y2
        self.x2, self.x1 = self.x1, sample
        self.y2, self.y1 = self.y1, out
        return out

def apply_eq(pcm_bytes, sample_rate, bass_db, treble_db):
    """
    Applies Bass and Treble EQ to 16-bit Mono PCM data.
    """
    if bass_db == 0 and treble_db == 0:
        return pcm_bytes

    # Convert bytes to signed 16-bit integers
    samples = array.array('h', pcm_bytes)
    
    # Initialize filters
    # Bass: Low shelf at 200Hz
    # Treble: High shelf at 3000Hz
    filters = []
    if bass_db != 0:
        filters.append(BiquadFilter("lowshelf", sample_rate, 200, 0.707, bass_db))
    if treble_db != 0:
        filters.append(BiquadFilter("highshelf", sample_rate, 3000, 0.707, treble_db))

    if not filters:
        return pcm_bytes

    processed = array.array('h', [0] * len(samples))
    for i in range(len(samples)):
        s = float(samples[i])
        for f in filters:
            s = f.process(s)
        
        # Hard clipping
        if s > 32767: s = 32767
        if s < -32768: s = -32768
        processed[i] = int(s)

    return processed.tobytes()
