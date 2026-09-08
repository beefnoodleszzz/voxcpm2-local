"""Objective audio gates. Heuristics are warnings, never perceptual measurements."""

from __future__ import annotations

from pathlib import Path

from .audio_utils import inspect_audio

VERSION = "1"


def audio_qc(stats: dict, text: str = "", *, reference: bool = False) -> dict:
    failures, warnings = [], []
    for condition, reason in (
        (stats["sample_rate"] != 48000, "sample_rate"),
        (stats["channels"] != 1, "non_mono"),
        (stats.get("has_nan", False), "nan"),
        (stats.get("has_inf", False), "inf"),
        (stats.get("clipping", False), "clipping"),
        (not stats.get("frames", 0), "empty"),
        (stats.get("peak", 0) <= 1e-5, "silence_only"),
    ):
        if condition:
            failures.append(reason)
    if not reference and stats.get("subtype") != "PCM_24":
        failures.append("not_pcm24")
    for condition, reason in (
        (stats.get("rms", 0) < 0.003, "low_energy"),
        (stats.get("peak", 0) > 0.98, "near_full_scale"),
        (abs(stats.get("dc_offset", 0)) > 0.02, "dc_offset"),
        (stats.get("silence_ratio", 0) > 0.65, "excessive_silence_ratio"),
        (stats.get("longest_silence_seconds", 0) > 2, "long_silence"),
        (stats.get("edge_silence_seconds", {}).get("start", 0) > 1, "start_silence"),
        (stats.get("edge_silence_seconds", {}).get("end", 0) > 1, "end_silence"),
        (stats.get("max_sample_jump", 0) > 0.5, "waveform_discontinuity"),
        (reference and stats.get("duration", 0) < 5, "reference_too_short"),
        (reference and stats.get("duration", 0) > 15, "reference_longer_than_preferred"),
        (
            bool(text) and stats.get("duration", 0) < max(0.2, len(text) * 0.035),
            "suspiciously_short",
        ),
        (bool(text) and stats.get("duration", 0) > max(15, len(text) * 1.5), "suspiciously_long"),
    ):
        if condition:
            warnings.append(reason)
    return {
        "status": "FAIL" if failures else ("WARN" if warnings else "PASS"),
        "failures": failures,
        "warnings": warnings,
        "version": VERSION,
        "unmeasured": ["noise", "music", "reverb", "speaker_identity", "emotion", "repeated_audio"],
        "note": "Acoustic thresholds are conservative engineering heuristics, not calibrated perceptual scores",
    }


def inspect_qc(path: Path, text: str = "", *, reference: bool = False) -> dict:
    stats = inspect_audio(path)
    return {"stats": stats, "qc": audio_qc(stats, text, reference=reference)}
