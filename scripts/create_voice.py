#!/usr/bin/env python3
import argparse, json, shutil
from pathlib import Path
import _bootstrap  # noqa: F401

from scipy.signal import resample_poly
import soundfile as sf

from src.audio_utils import inspect_audio
from src.voice_library import ROOT

def main():
    p = argparse.ArgumentParser(); p.add_argument("--id", required=True); p.add_argument("--style", required=True)
    p.add_argument("--reference", type=Path, required=True); p.add_argument("--transcript", required=True)
    p.add_argument("--name"); p.add_argument("--description", default=""); args = p.parse_args()
    if Path(args.id).name != args.id or Path(args.style).name != args.style: raise SystemExit("Unsafe ID/style")
    if not args.reference.is_file() or not args.transcript.strip(): raise SystemExit("Reference and exact transcript are required")
    stats = inspect_audio(args.reference)
    warnings = []
    if not 5 <= stats["duration"] <= 15: warnings.append("recommended duration is 5–15 seconds")
    if stats["sample_rate"] < 16000: warnings.append("sample rate is below 16kHz")
    if stats["channels"] != 1: warnings.append("input is not mono; normalized copy will be mono")
    if stats["clipping"]: warnings.append("input contains clipping")
    if stats["longest_silence_seconds"] > 2: warnings.append("input contains >2s silence")
    target = ROOT / "voices" / args.id / args.style; target.mkdir(parents=True, exist_ok=True)
    suffix = args.reference.suffix.lower() or ".wav"; original = target / f"original{suffix}"
    if original.exists(): raise SystemExit(f"Refusing to overwrite original: {original}")
    shutil.copy2(args.reference, original)
    data, sr = sf.read(args.reference, always_2d=True, dtype="float32"); mono = data.mean(axis=1)
    if sr != 48000: mono = resample_poly(mono, 48000, sr); sr = 48000
    reference = target / "reference.wav"
    if reference.exists(): raise SystemExit(f"Refusing to overwrite reference: {reference}")
    sf.write(reference, mono, sr, subtype="PCM_24")
    (target / "transcript.txt").write_text(args.transcript.strip() + "\n", encoding="utf-8")
    voice_path = target.parent / "voice.json"
    voice = json.loads(voice_path.read_text(encoding="utf-8")) if voice_path.exists() else {
        "id": args.id, "name": args.name or args.id, "language": "zh",
        "description": args.description, "default_style": args.style, "styles": {}}
    voice["styles"][args.style] = {"reference": f"{args.style}/reference.wav", "transcript": f"{args.style}/transcript.txt",
                                    "original": f"{args.style}/{original.name}", "quality_warnings": warnings}
    voice_path.write_text(json.dumps(voice, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"voice": str(voice_path), "input_analysis": stats, "warnings": warnings}, ensure_ascii=False, indent=2))
if __name__ == "__main__": main()
