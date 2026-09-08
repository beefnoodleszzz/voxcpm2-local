#!/usr/bin/env python3
"""Resolve human emotion labels into safe, natural-dialogue batch input."""

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from src.voice_library import VoiceLibrary
from src.pronunciation import prepare_text
from src.schemas import BatchRequest

from src.emotions import ROUTES, INSTRUCTIONS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument(
        "--exact-performance",
        action="store_true",
        help="use a matching registered style with Ultimate instead of natural controllable mode",
    )
    args = parser.parse_args()
    source = json.loads(Path(args.input).read_text(encoding="utf-8"))
    library = VoiceLibrary()
    prepared = {k: v for k, v in source.items() if k != "lines"}
    prepared.setdefault("candidates", 3)
    prepared.setdefault("profile", "production")
    prepared.setdefault("inference_timesteps", 30)
    prepared.setdefault("cfg_value", 2.0)
    prepared.setdefault("warmup_patches", 0)
    prepared.setdefault("max_tokens", 2000)
    lines = []
    for raw in source.get("lines", []):
        line = dict(raw)
        character_id = line.get("character_id")
        if not character_id:
            raise ValueError(f"{line.get('id', '<unknown>')}: character_id is required")
        voice = library.get(character_id)
        emotion = line.pop("emotion", None)
        requested_style = line.get("style") or ROUTES.get(emotion, "neutral")
        available = voice.get("styles", {})
        if args.exact_performance and requested_style in available:
            line.update(style=requested_style, mode="ultimate")
            line.pop("instruct", None)
        else:
            line.update(style="neutral", mode="controllable")
            line.setdefault("instruct", INSTRUCTIONS.get(requested_style, INSTRUCTIONS["neutral"]))
        library.resolve_style(character_id, line["style"])
        preparation = prepare_text(
            line["text"],
            source.get("normalization_profile", "dialogue"),
            source.get("project_lexicon"),
            source.get("episode_lexicon"),
        )
        if emotion is not None and emotion not in ROUTES:
            preparation["normalization"]["warnings"].append(
                f"Unknown emotion {emotion!r}; neutral fallback"
            )
        line["preparation"] = preparation
        line["text"] = preparation["text_spoken"]
        lines.append(line)
    prepared["lines"] = lines
    prepared = BatchRequest.model_validate(prepared).model_dump()
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite: {destination}")
    from src.audio_utils import write_json

    write_json(destination, prepared, exclusive=True)
    print(destination)


if __name__ == "__main__":
    main()
