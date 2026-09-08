#!/usr/bin/env python3
import argparse
import json
import shutil
from pathlib import Path
from datetime import datetime, timezone
import _bootstrap  # noqa: F401

from scipy.signal import resample_poly
import soundfile as sf

from src.audio_utils import inspect_audio, write_wav, write_json
from src.provenance import file_sha256, read_model_lock, runtime_versions
from src.qc import audio_qc
from src.voice_library import ROOT


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--id", required=True)
    p.add_argument("--style", required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--transcript", required=True)
    p.add_argument("--name")
    p.add_argument("--description", default="")
    p.add_argument(
        "--provenance",
        default="user_supplied",
        choices=["user_supplied", "voice_design", "controllable"],
    )
    p.add_argument(
        "--commercial-use",
        action="store_true",
        default=None,
        help="Record only when the voice rights actually allow commercial use",
    )
    args = p.parse_args()
    if any(
        value in {".", ".."} or not value or Path(value).name != value
        for value in (args.id, args.style)
    ):
        raise SystemExit("Unsafe ID/style")
    if not args.reference.is_file() or not args.transcript.strip():
        raise SystemExit("Reference and exact transcript are required")
    stats = inspect_audio(args.reference)
    if stats["has_nan"] or stats["has_inf"] or stats["clipping"] or stats["peak"] <= 1e-5:
        raise SystemExit(
            "Reference is nonfinite, clipped or silence-only; select a clean reference"
        )
    warnings = []
    if not 5 <= stats["duration"] <= 15:
        warnings.append("recommended duration is 5–15 seconds")
    if stats["sample_rate"] < 16000:
        warnings.append("sample rate is below 16kHz")
    if stats["channels"] != 1:
        warnings.append("input is not mono; normalized copy will be mono")
    if stats["clipping"]:
        warnings.append("input contains clipping")
    if stats["longest_silence_seconds"] > 2:
        warnings.append("input contains >2s silence")
    target = ROOT / "voices" / args.id / args.style
    if not target.resolve().is_relative_to((ROOT / "voices").resolve()):
        raise SystemExit("Unsafe reference destination")
    if any(target.glob("*")):
        raise SystemExit(f"Refusing to alter existing canonical style: {target}")
    target.mkdir(parents=True, exist_ok=True)
    suffix = args.reference.suffix.lower() or ".wav"
    original = target / f"original{suffix}"
    if original.exists():
        raise SystemExit(f"Refusing to overwrite original: {original}")
    with args.reference.open("rb") as source, original.open("xb") as destination:
        shutil.copyfileobj(source, destination)
    data, sr = sf.read(args.reference, always_2d=True, dtype="float32")
    mono = data.mean(axis=1)
    if sr != 48000:
        mono = resample_poly(mono, 48000, sr)
        sr = 48000
    reference = target / "reference.wav"
    if reference.exists():
        raise SystemExit(f"Refusing to overwrite reference: {reference}")
    normalized_stats = write_wav(reference, mono, sr)
    with (target / "transcript.txt").open("x", encoding="utf-8") as stream:
        stream.write(args.transcript.strip() + "\n")
    voice_path = target.parent / "voice.json"
    voice = (
        json.loads(voice_path.read_text(encoding="utf-8"))
        if voice_path.exists()
        else {
            "id": args.id,
            "name": args.name or args.id,
            "language": "zh",
            "description": args.description,
            "default_style": args.style,
            "styles": {},
        }
    )
    voice["styles"][args.style] = {
        "reference": f"{args.style}/reference.wav",
        "transcript": f"{args.style}/transcript.txt",
        "original": f"{args.style}/{original.name}",
        "quality_warnings": warnings,
        "reference_sha256": file_sha256(reference),
        "original_sha256": file_sha256(original),
        "reference_stats": {**normalized_stats, "path": f"{args.style}/reference.wav"},
        "reference_qc": audio_qc(normalized_stats, reference=True),
        "transcript_text": args.transcript.strip(),
        "transcript_verified": "user_supplied_not_asr_verified",
        "provenance": args.provenance,
        "commercial_use": args.commercial_use,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    voice.setdefault("provenance", args.provenance)
    voice.setdefault(
        "source", "generated" if args.provenance != "user_supplied" else "user_supplied"
    )
    voice.setdefault("commercial_use", args.commercial_use)
    voice.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    voice.setdefault("tags", [])
    voice.setdefault("runtime_at_registration", runtime_versions())
    voice.setdefault("model_lock_at_registration", read_model_lock())
    write_json(voice_path, voice)
    print(
        json.dumps(
            {"voice": str(voice_path), "input_analysis": stats, "warnings": warnings},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
