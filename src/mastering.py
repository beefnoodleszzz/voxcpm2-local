"""Optional mild mastering of human-approved audio, always to a new delivery file."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path
from uuid import uuid4

import numpy as np
import soundfile as sf

from .audio_utils import write_json, write_wav
from .provenance import file_sha256
from .qc import inspect_qc


def measure_loudness(path: Path) -> dict:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-i",
        str(path),
        "-af",
        "loudnorm=I=-23:TP=-1:LRA=7:print_format=json",
        "-f",
        "null",
        "-",
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=120)
    start, end = result.stderr.rfind("{"), result.stderr.rfind("}")
    if start < 0 or end < start:
        raise ValueError("FFmpeg did not return loudness measurements")
    measured = json.loads(result.stderr[start : end + 1])
    return {
        "integrated_lufs": float(measured["input_i"]),
        "true_peak_dbfs": float(measured["input_tp"]),
    }


def finalize_audio(
    approved: Path,
    root: Path,
    *,
    profile: str = "raw_master",
    trim: bool = False,
    peak_dbfs: float | None = None,
    target_lufs: float | None = None,
) -> dict:
    approved = approved.resolve()
    base = (root / "outputs/approved").resolve()
    if not approved.is_relative_to(base) or not approved.is_file():
        raise ValueError("Delivery accepts only an approved WAV")
    profiles = json.loads(
        (Path(__file__).resolve().parents[1] / "configs/mastering.json").read_text()
    )
    if profile not in profiles or profile == "note":
        raise ValueError("Unknown mastering profile")
    review = json.loads(approved.with_suffix(".json").read_text())
    if (
        review.get("decision") != "approve"
        or not review.get("listened")
        or review.get("audio_sha256") != file_sha256(approved)
    ):
        raise ValueError("Approved audio integrity/review check failed")
    if peak_dbfs is not None and (not math.isfinite(peak_dbfs) or not -20 <= peak_dbfs <= -0.1):
        raise ValueError("Peak target must be -20 to -0.1 dBFS")
    if target_lufs is not None and (not math.isfinite(target_lufs) or not -36 <= target_lufs <= -9):
        raise ValueError("Loudness target must be -36 to -9 LUFS")
    source_qc = inspect_qc(approved)
    if source_qc["qc"]["status"] == "FAIL":
        raise ValueError("Approved file no longer passes format/integrity QC")
    folder = root / "outputs/delivery" / approved.relative_to(base).parent / uuid4().hex
    folder.mkdir(parents=True, exist_ok=False)
    output = folder / approved.name
    data, sr = sf.read(approved, dtype="float32")
    gain_db, trimmed = 0.0, [0, 0]
    measured = None
    if trim:
        active = np.flatnonzero(np.abs(data) > 1e-4)
        if not active.size:
            raise ValueError("Cannot trim silence-only audio")
        start = max(0, int(active[0]) - int(sr * 0.08))
        end = min(len(data), int(active[-1]) + 1 + int(sr * 0.12))
        trimmed = [start, len(data) - end]
        data = data[start:end]
    if target_lufs is not None:
        measured = measure_loudness(approved)
        if not all(math.isfinite(v) for v in measured.values()):
            raise ValueError("Loudness is not measurable for this audio")
        gain_db = target_lufs - measured["integrated_lufs"]
        # Linear gain only. Respect true peak, do not silently add compression/limiting.
        gain_db = min(
            gain_db, (peak_dbfs if peak_dbfs is not None else -1) - measured["true_peak_dbfs"]
        )
    elif peak_dbfs is not None:
        gain_db = peak_dbfs - 20 * math.log10(float(np.max(np.abs(data))))
    if gain_db > 12:
        raise ValueError("More than 12 dB boost requested; review source loudness first")
    if trim or peak_dbfs is not None or target_lufs is not None:
        write_wav(output, data * 10 ** (gain_db / 20), sr)
    else:
        with approved.open("rb") as source, output.open("xb") as destination:
            shutil.copyfileobj(source, destination)
    inspection = inspect_qc(output)
    result = {
        "approved_output": str(approved),
        "output": str(output.resolve()),
        "approved_sha256": review["audio_sha256"],
        "audio_sha256": file_sha256(output),
        "review_id": review["review_id"],
        "source_review": review,
        "profile": profile,
        "processing": {
            "trim_frames": trimmed,
            "linear_gain_db": gain_db,
            "target_lufs": target_lufs,
            "peak_dbfs": peak_dbfs,
            "limiter": False,
            "eq": False,
            "compression": False,
        },
        "input_loudness": measured,
        **inspection,
    }
    if target_lufs is not None:
        result["output_loudness"] = measure_loudness(output)
        result["target_achieved"] = (
            abs(result["output_loudness"]["integrated_lufs"] - target_lufs) < 1
        )
    write_json(output.with_suffix(".json"), result, exclusive=True)
    if inspection["qc"]["status"] == "FAIL":
        raise ValueError(f"Delivery QC failed; retained for diagnosis: {output}")
    return result
