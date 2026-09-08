#!/usr/bin/env python3
"""Promote selected emotion takes into each character's style library."""

import argparse
import json
import subprocess
import sys
from pathlib import Path
import _bootstrap  # noqa: F401
from src.voice_library import ROOT


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--selections", type=Path, default=ROOT / "configs/emotion_cast_selections.json")
    args = p.parse_args()
    choices = json.loads(args.selections.read_text(encoding="utf-8"))["choices"]
    cfg = json.loads((ROOT / "configs/emotion_cast.json").read_text(encoding="utf-8"))
    styles = cfg["styles"]
    selected = [(key, n) for key, n in choices.items() if n is not None]
    if not selected:
        raise SystemExit("No emotion selections yet. Set values to 1, 2, or 3.")
    for key, candidate in selected:
        try:
            character, style = key.split("/", 1)
        except ValueError:
            raise SystemExit(f"Invalid key: {key}")
        if character not in cfg["characters"] or style not in styles or candidate not in (1, 2, 3):
            raise SystemExit(f"Invalid selection: {key}={candidate}")
        wav = (
            ROOT
            / "outputs/raw/emotion_cast_v1"
            / character
            / style
            / f"candidate_{candidate:02d}.wav"
        )
        voice = json.loads((ROOT / "voices" / character / "voice.json").read_text(encoding="utf-8"))
        cmd = [
            sys.executable,
            str(ROOT / "scripts/create_voice.py"),
            "--id",
            character,
            "--style",
            style,
            "--reference",
            str(wav),
            "--transcript",
            styles[style]["text"],
            "--name",
            voice["name"],
            "--description",
            voice.get("description", ""),
        ]
        subprocess.run(cmd, cwd=ROOT, check=True)
    print(f"FINALIZED {len(selected)} emotion styles")


if __name__ == "__main__":
    main()
