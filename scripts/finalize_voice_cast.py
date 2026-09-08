#!/usr/bin/env python3
"""Promote human-selected audition takes into the canonical voice library."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from src.voice_library import ROOT

def main():
    p = argparse.ArgumentParser(); p.add_argument("--selections", type=Path, default=ROOT / "configs/voice_cast_selections.json")
    args = p.parse_args(); choices = json.loads(args.selections.read_text(encoding="utf-8"))["selections"]
    cast = {v["id"]: v for v in json.loads((ROOT / "configs/voice_cast.json").read_text(encoding="utf-8"))["voices"]}
    selected = [(voice_id, n) for voice_id, n in choices.items() if n is not None]
    if not selected: raise SystemExit("No selections yet. Set candidate values to 1, 2, or 3.")
    for voice_id, candidate in selected:
        if voice_id not in cast or candidate not in (1, 2, 3): raise SystemExit(f"Invalid selection: {voice_id}={candidate}")
        voice = cast[voice_id]; wav = ROOT / "outputs/raw/voice_cast_v1" / voice_id / f"candidate_{candidate:02d}.wav"
        cmd = [sys.executable, str(ROOT / "scripts/create_voice.py"), "--id", voice_id, "--style", "neutral",
               "--reference", str(wav), "--transcript", voice["text"], "--name", voice["name"],
               "--description", voice["instruct"]]
        subprocess.run(cmd, cwd=ROOT, check=True)
    print(f"FINALIZED {len(selected)} canonical voices")

if __name__ == "__main__": main()
