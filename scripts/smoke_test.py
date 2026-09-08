#!/usr/bin/env python3
import argparse
from pathlib import Path
import _bootstrap  # noqa: F401
from src.engine import VoxCPMEngine
from src.voice_library import ROOT


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=ROOT / "outputs/smoke/basic.wav")
    p.add_argument("--steps", type=int, default=30)
    args = p.parse_args()
    meta = VoxCPMEngine().design_voice(
        text="雨已经停了，可我还是觉得，今晚有些事情不会结束。",
        instruct="年轻中国男性，中低音，克制自然的影视对白，不是播音腔",
        output=args.output,
        seed=42,
        inference_timesteps=args.steps,
        cfg_value=2.0,
        warmup_patches=0,
        max_tokens=2000,
    )
    print(meta)


if __name__ == "__main__":
    main()
