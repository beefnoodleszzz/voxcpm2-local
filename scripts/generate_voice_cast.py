#!/usr/bin/env python3
"""Generate or resume the complete audition cast without overwriting takes."""
import argparse
import html
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from src.audio_utils import write_json
from src.engine import VoxCPMEngine
from src.voice_library import ROOT


def complete(wav: Path) -> bool:
    return wav.is_file() and wav.stat().st_size > 1024 and wav.with_suffix(".json").is_file()


def render_html(cast: dict, output_root: Path, rows: list[dict]):
    cards = []
    for row in rows:
        rel = Path(row["output"]).relative_to(output_root)
        cards.append(f'''<article><h2>{html.escape(row["name"])} <small>{html.escape(row["id"])}</small></h2>
<p>{html.escape(row["group"])} · Candidate {row["candidate"]} · Seed {row["seed"]}</p>
<p>{html.escape(row["text"])}</p><audio controls preload="none" src="{html.escape(str(rel))}"></audio></article>''')
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>VoxCPM2 虚拟演员试音</title>
<style>body{{max-width:1100px;margin:40px auto;padding:0 20px;font:16px system-ui;background:#111;color:#eee}}article{{background:#1d1d1f;padding:16px 20px;margin:14px 0;border-radius:12px}}h2{{margin:0}}small{{color:#aaa;font-size:13px}}audio{{width:100%}}</style>
<h1>VoxCPM2 BF16 虚拟演员试音</h1><p>共 {len(cast['voices'])} 类角色，每类 {cast['defaults']['candidates']} 个候选。请记录每个角色选中的 Candidate。</p>{''.join(cards)}</html>'''
    (output_root / "index.html").write_text(page, encoding="utf-8")


def main():
    p = argparse.ArgumentParser(); p.add_argument("--config", type=Path, default=ROOT / "configs/voice_cast.json")
    p.add_argument("--only"); args = p.parse_args()
    cast = json.loads(args.config.read_text(encoding="utf-8")); defaults = cast["defaults"]
    output_root = ROOT / "outputs/raw/voice_cast_v1"; output_root.mkdir(parents=True, exist_ok=True)
    engine = VoxCPMEngine(); rows = []; total = len(cast["voices"]) * defaults["candidates"]; done = 0
    for voice in cast["voices"]:
        if args.only and voice["id"] != args.only: continue
        for index in range(1, defaults["candidates"] + 1):
            seed = 42 + (index - 1) * 3365
            wav = output_root / voice["id"] / f"candidate_{index:02d}.wav"
            if complete(wav):
                meta = json.loads(wav.with_suffix(".json").read_text(encoding="utf-8")); status = "SKIP"
            else:
                meta = engine.design_voice(text=voice["text"], instruct=voice["instruct"], output=wav,
                    seed=seed, inference_timesteps=defaults["inference_timesteps"], cfg_value=defaults["cfg_value"],
                    warmup_patches=defaults["warmup_patches"], max_tokens=defaults["max_tokens"]); status = "DONE"
            done += 1
            row = {**voice, "candidate": index, "seed": seed, "output": meta["output"],
                   "duration": meta["duration"], "generation_seconds": meta["generation_seconds"]}
            rows.append(row); print(f"[{done}/{total}] {status} {voice['id']} candidate_{index:02d} {meta['duration']:.2f}s", flush=True)
            write_json(output_root / "manifest.json", {"config": str(args.config), "rows": rows})
            render_html(cast, output_root, rows)
    print(f"VOICE CAST COMPLETE: {len(rows)} files\nOpen: {output_root / 'index.html'}")

if __name__ == "__main__": main()
