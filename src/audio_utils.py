from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from .errors import AudioValidationError

TARGET_SAMPLE_RATE = 48_000


def inspect_audio(path: Path) -> dict:
    data, sr = sf.read(path, always_2d=True, dtype="float32")
    mono = data.mean(axis=1)
    duration = len(mono) / sr if sr else 0
    peak = float(np.max(np.abs(mono))) if mono.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(mono)))) if mono.size else 0.0
    long_silence = _longest_silence(mono, sr)
    return {
        "path": str(path.resolve()),
        "sample_rate": sr,
        "channels": data.shape[1],
        "frames": len(mono),
        "duration": duration,
        "peak": float(np.max(np.abs(data))) if data.size else peak,
        "rms": rms,
        "clipping": bool(np.any(np.abs(data) >= 0.999)),
        "has_nan": bool(np.isnan(data).any()),
        "has_inf": bool(np.isinf(data).any()),
        "subtype": sf.info(path).subtype,
        "dc_offset": float(np.mean(mono)) if mono.size else 0.0,
        "silence_ratio": float(np.mean(np.abs(mono) < 1e-3)) if mono.size else 1.0,
        "max_sample_jump": float(np.max(np.abs(np.diff(mono)))) if mono.size > 1 else 0.0,
        "edge_silence_seconds": {
            "start": float(np.argmax(np.abs(mono) > 1e-4) / sr)
            if np.any(np.abs(mono) > 1e-4)
            else duration,
            "end": float(np.argmax(np.abs(mono[::-1]) > 1e-4) / sr)
            if np.any(np.abs(mono) > 1e-4)
            else duration,
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
        raise AudioValidationError("Generated audio is empty, non-mono, or contains NaN/Inf")
    if sample_rate != TARGET_SAMPLE_RATE:
        raise AudioValidationError(f"VoxCPM2 returned {sample_rate} Hz, expected 48000 Hz")
    # Encoding would irreversibly clip an out-of-range waveform. Fail before conversion.
    if np.max(np.abs(arr)) >= 1.0:
        raise AudioValidationError(
            "Generated waveform exceeds PCM full scale; refusing silent clipping"
        )
    with path.open("xb") as stream:
        sf.write(stream, arr, sample_rate, subtype="PCM_24", format="WAV")
    if path.stat().st_size == 0:
        raise RuntimeError("Generated WAV is empty")
    return inspect_audio(path)


def write_json(path: Path, payload: dict, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if exclusive:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        return
    # Manifest updates are atomic; candidate/master files use exclusive creation.
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=".json-", delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
