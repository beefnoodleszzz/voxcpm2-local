#!/usr/bin/env python3
import argparse
from datetime import datetime
import _bootstrap  # noqa: F401
from src.engine import VoxCPMEngine
from src.voice_library import ROOT

def main():
    p = argparse.ArgumentParser(); p.add_argument("--character"); p.add_argument("--style", default="neutral")
    p.add_argument("--mode", choices=["design", "controllable", "ultimate"], default="controllable",
                   help="controllable is the natural-dialogue default; use ultimate for exact performance continuation")
    p.add_argument("--text", required=True)
    p.add_argument("--instruct", default="自然生活化对白，语气克制，停连随语义变化，不要播音腔，不要机械匀速")
    p.add_argument("--candidates", type=int, default=3)
    p.add_argument("--seed", type=int, default=42); p.add_argument("--steps", type=int, default=30); args = p.parse_args()
    folder = ROOT / "outputs/raw/cli" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    common = dict(text=args.text, candidates=args.candidates, seed=args.seed, inference_timesteps=args.steps,
                  cfg_value=2.0, warmup_patches=0, max_tokens=2000)
    engine = VoxCPMEngine()
    if args.mode == "design": metas = engine.generate_candidates("design_voice", folder, instruct=args.instruct, **common)
    elif args.mode == "controllable": metas = engine.generate_candidates("clone_voice", folder, character_id=args.character, style=args.style, instruct=args.instruct, **common)
    else: metas = engine.generate_candidates("ultimate_clone", folder, character_id=args.character, style=args.style, **common)
    print("\n".join(m["output"] for m in metas))
if __name__ == "__main__": main()
