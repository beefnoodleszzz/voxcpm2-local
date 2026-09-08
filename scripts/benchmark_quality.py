#!/usr/bin/env python3
import _bootstrap  # noqa: F401
from src.engine import VoxCPMEngine
from src.voice_library import ROOT

def main():
    engine = VoxCPMEngine(); folder = ROOT / "outputs/smoke/benchmark"
    for steps in (10, 20, 30):
        meta = engine.design_voice(text="雨已经停了，可我还是觉得，今晚有些事情不会结束。",
            instruct="年轻中国男性，中低音，克制自然的影视对白，不是播音腔",
            output=folder / f"steps_{steps}.wav", seed=42, inference_timesteps=steps,
            cfg_value=2.0, warmup_patches=0, max_tokens=2000)
        print(meta)
if __name__ == "__main__": main()
