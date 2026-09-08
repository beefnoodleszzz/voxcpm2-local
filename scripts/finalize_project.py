"""Deliver explicit approved WAVs, or all approved WAVs within a project."""

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from src.mastering import finalize_audio
from src.voice_library import ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="approved WAV path or project ID")
    parser.add_argument(
        "--profile",
        default="raw_master",
        choices=["raw_master", "drama_dialogue", "narration", "short_video"],
    )
    parser.add_argument("--trim", action="store_true")
    parser.add_argument("--peak-dbfs", type=float)
    parser.add_argument("--target-lufs", type=float)
    args = parser.parse_args()
    supplied = Path(args.input)
    if supplied.is_file():
        paths = [supplied]
    else:
        if supplied.name != args.input or args.input in {".", ".."}:
            parser.error("Provide an approved WAV or a safe project ID")
        paths = sorted((ROOT / "outputs/approved" / args.input).rglob("*.wav"))
    if not paths:
        parser.error("No approved WAVs found; review candidates first")
    results = [
        finalize_audio(
            path,
            ROOT,
            profile=args.profile,
            trim=args.trim,
            peak_dbfs=args.peak_dbfs,
            target_lufs=args.target_lufs,
        )
        for path in paths
    ]
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
