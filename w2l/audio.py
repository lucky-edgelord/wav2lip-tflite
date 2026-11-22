"""Mel spectrogram matching Wav2Lip's audio.py / hparams.py exactly.

Wav2Lip was trained on these features, so any deviation (window, fmin/fmax,
normalisation) degrades lip sync even if the model converts perfectly.
"""

import subprocess
import tempfile
import os

import numpy as np
import librosa
from scipy import signal

SAMPLE_RATE = 16000
N_FFT = 800
HOP = 200
WIN = 800
N_MELS = 80
FMIN, FMAX = 55, 7600
PREEMPHASIS = 0.97
REF_LEVEL_DB = 20
MIN_LEVEL_DB = -100
MAX_ABS = 4.0
MEL_STEP = 16

_mel_basis = None


def load_wav(path: str) -> np.ndarray:
    """Load any audio/video file as mono 16 kHz float32. Uses ffmpeg for non-wav inputs."""
    if path.lower().endswith(".wav"):
        return librosa.load(path, sr=SAMPLE_RATE)[0]
    with tempfile.TemporaryDirectory() as tmp:
        wav = os.path.join(tmp, "audio.wav")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", path,
                        "-ac", "1", "-ar", str(SAMPLE_RATE), wav], check=True)
        return librosa.load(wav, sr=SAMPLE_RATE)[0]


def melspectrogram(wav: np.ndarray) -> np.ndarray:
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS,
                                         fmin=FMIN, fmax=FMAX)
    emph = signal.lfilter([1, -PREEMPHASIS], [1], wav)
    spec = np.abs(librosa.stft(y=emph, n_fft=N_FFT, hop_length=HOP, win_length=WIN))
    mel = np.dot(_mel_basis, spec)
    min_level = np.exp(MIN_LEVEL_DB / 20 * np.log(10))
    db = 20 * np.log10(np.maximum(min_level, mel)) - REF_LEVEL_DB
    norm = (2 * MAX_ABS) * ((db - MIN_LEVEL_DB) / -MIN_LEVEL_DB) - MAX_ABS
    return np.clip(norm, -MAX_ABS, MAX_ABS).astype(np.float32)  # (80, T)


def mel_chunks(mel: np.ndarray, fps: float) -> list[np.ndarray]:
    """One (80, 16) window per video frame, same indexing as Wav2Lip's inference.py."""
    chunks = []
    mult = 80.0 / fps
    i = 0
    while True:
        start = int(i * mult)
        if start + MEL_STEP > mel.shape[1]:
            chunks.append(mel[:, -MEL_STEP:])
            break
        chunks.append(mel[:, start:start + MEL_STEP])
        i += 1
    return chunks
