from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

TARGET_SAMPLE_RATE = 48_000


def inspect_audio(path: Path) -> dict:
    data, sr = sf.read(path, always_2d=True, dtype="float32")
    mono = data.mean(axis=1)
    duration = len(mono) / sr if sr else 0
    peak = float(np.max(np.abs(mono))) if mono.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(mono)))) if mono.size else 0.0
    edge = max(1, int(sr * 0.02))
    long_silence = _longest_silence(mono, sr)
    return {
        "path": str(path.resolve()), "sample_rate": sr,
        "channels": data.shape[1], "frames": len(mono), "duration": duration,
        "peak": peak, "rms": rms, "clipping": bool(np.any(np.abs(mono) >= 0.999)),
        "has_nan": bool(not np.isfinite(mono).all()),
        "edge_silence_seconds": {
            "start": float(np.argmax(np.abs(mono) > 1e-4) / sr) if np.any(np.abs(mono) > 1e-4) else duration,
            "end": float(np.argmax(np.abs(mono[::-1]) > 1e-4) / sr) if np.any(np.abs(mono) > 1e-4) else duration,
        },
        "longest_silence_seconds": long_silence,
    }


def _longest_silence(audio: np.ndarray, sr: int, threshold: float = 1e-3) -> float:
    silent = np.abs(audio) < threshold
    if not silent.size:
        return 0.0
    padded = np.r_[False, silent, False]
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return float(np.max(changes[1::2] - changes[::2]) / sr) if len(changes) else 0.0


def write_wav(path: Path, audio, sample_rate: int = TARGET_SAMPLE_RATE) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(audio, dtype=np.float32).squeeze()
    if arr.ndim != 1 or not arr.size or not np.isfinite(arr).all():
        raise ValueError("Generated audio is empty, non-mono, or contains NaN/Inf")
    if sample_rate != TARGET_SAMPLE_RATE:
        raise ValueError(f"VoxCPM2 returned {sample_rate} Hz, expected 48000 Hz")
    sf.write(path, arr, sample_rate, subtype="PCM_24")
    if path.stat().st_size == 0:
        raise RuntimeError("Generated WAV is empty")
    return inspect_audio(path)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

