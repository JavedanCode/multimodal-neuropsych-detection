"""Decode PCM recordings and resample to the shared audio rate."""

import io
import wave
import numpy as np
from scipy.signal import resample_poly

RATE = 16000


WINDOW = 8 * RATE


def decode_audio(raw):
    with wave.open(io.BytesIO(raw)) as wav:
        channels = wav.getnchannels()
        rate = wav.getframerate()
        n = wav.getnframes()
        if (
            channels not in [1, 2]
            or rate not in [16000, 48000]
            or wav.getsampwidth() != 2
            or n / rate > 3600
        ):
            raise ValueError("Unsupported/unbounded WAV")
        pcm = wav.readframes(n)
    if len(pcm) != n * channels * 2:
        raise ValueError("Truncated WAV")
    values = (
        np.frombuffer(pcm, dtype="<i2")
        .reshape(-1, channels)
        .astype(np.float32)
        .mean(axis=1)
        / 32768.0
    )
    if rate != RATE:
        values = resample_poly(
            values, RATE // np.gcd(RATE, rate), rate // np.gcd(RATE, rate)
        ).astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite waveform")
    return values, {
        "channels": channels,
        "source_rate": rate,
        "source_frames": n,
        "duration_seconds": n / rate,
    }
