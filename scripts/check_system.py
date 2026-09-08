#!/usr/bin/env python3
"""Read-only readiness check for the local BF16 VoxCPM2 production stack."""

import json
import platform
import sys
from importlib.metadata import PackageNotFoundError, version

import _bootstrap  # noqa: F401
from src.audio_utils import inspect_audio
from src.voice_library import VoiceLibrary
from src.provenance import model_identity, model_path


def package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def main():
    errors, warnings = [], []
    model = model_path()
    try:
        identity = model_identity(model)
    except (ValueError, OSError, KeyError) as exc:
        errors.append(f"model lock/integrity: {exc}")
        identity = {}
    required = ["config.json", "model.safetensors.index.json", "tokenizer.json"]
    if not model.is_dir():
        errors.append(f"BF16 model directory missing: {model}")
    for name in required:
        if not (model / name).is_file():
            errors.append(f"model file missing: {model / name}")
    if platform.machine() != "arm64":
        warnings.append(f"expected Apple Silicon arm64, got {platform.machine()}")
    if sys.version_info[:2] != (3, 11):
        errors.append(f"expected Python 3.11, got {platform.python_version()}")
    if package_version("mlx-audio") != "0.5.1":
        errors.append(f"expected mlx-audio 0.5.1, got {package_version('mlx-audio')}")

    library = VoiceLibrary()
    short_refs = 0
    for voice in library.list():
        for style in voice.get("styles", {}):
            try:
                entry = library.resolve_style(voice["id"], style)
                stats = inspect_audio(entry["audio"])
                if stats["sample_rate"] != 48000 or stats["channels"] != 1:
                    errors.append(f"{voice['id']}/{style}: reference must be mono 48kHz")
                if stats["duration"] < 5.0:
                    short_refs += 1
            except Exception as exc:
                errors.append(f"{voice['id']}/{style}: {exc}")
    if short_refs:
        warnings.append(f"{short_refs} voice references are shorter than the recommended 5 seconds")

    report = {
        "ok": not errors,
        "backend": "mlx",
        "machine": platform.machine(),
        "python": platform.python_version(),
        "mlx_audio": package_version("mlx-audio"),
        "model": str(model),
        "model_format": "BF16",
        "model_identity": identity,
        "voices": len(library.list()),
        "errors": errors,
        "warnings": warnings,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
