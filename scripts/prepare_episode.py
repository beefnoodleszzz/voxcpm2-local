#!/usr/bin/env python3
"""Resolve human emotion labels into safe, natural-dialogue batch input."""
import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from src.voice_library import VoiceLibrary

ROUTES = {
    "平静": "neutral", "中性": "neutral", "正常": "neutral", "叙述": "neutral", "neutral": "neutral",
    "温柔": "gentle", "安慰": "gentle", "宠溺": "gentle", "暖": "gentle", "gentle": "gentle",
    "生气": "angry", "愤怒": "angry", "压怒": "angry", "质问": "angry", "angry": "angry",
    "悲伤": "sad", "难过": "sad", "失落": "sad", "哽咽": "sad", "sad": "sad",
    "惊恐": "panic", "慌张": "panic", "害怕": "panic", "紧张": "panic", "panic": "panic",
    "耳语": "whisper", "低声": "whisper", "悄悄": "whisper", "气声": "whisper", "whisper": "whisper",
}
INSTRUCTIONS = {
    "neutral": "自然生活化对白，语气克制，停连随语义变化，不要播音腔，不要机械匀速",
    "gentle": "轻声安慰，语速稍慢，呼吸自然，句尾放松，不要刻意煽情",
    "angry": "压住怒气，咬字稍有力度，节奏有变化，不要持续喊叫",
    "sad": "克制失落，气息稍弱，允许自然停顿，不要哭腔过满",
    "panic": "紧张急促但吐字清楚，呼吸略乱，不要尖叫",
    "whisper": "近距离低声说话，气声自然，保持清晰，不要全句虚声",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--exact-performance", action="store_true",
                        help="use a matching registered style with Ultimate instead of natural controllable mode")
    args = parser.parse_args()
    source = json.loads(Path(args.input).read_text(encoding="utf-8"))
    library = VoiceLibrary()
    prepared = {k: v for k, v in source.items() if k != "lines"}
    prepared.setdefault("candidates", 3)
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
        lines.append(line)
    prepared["lines"] = lines
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite: {destination}")
    destination.write_text(json.dumps(prepared, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
