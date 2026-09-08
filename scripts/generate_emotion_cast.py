#!/usr/bin/env python3
"""Generate/resume emotion-matched audition references for core characters."""

import html
import json
from pathlib import Path
import _bootstrap  # noqa: F401
from src.audio_utils import write_json
from src.engine import VoxCPMEngine
from src.voice_library import ROOT


def complete(p):
    return p.is_file() and p.stat().st_size > 1024 and p.with_suffix(".json").is_file()


def render(root, rows):
    cards = []
    for r in rows:
        rel = Path(r["output"]).relative_to(root)
        cards.append(
            f"<article><h2>{html.escape(r['character_id'])} / {r['style']} / Candidate {r['candidate']}</h2><p>{html.escape(r['text'])}</p><audio controls preload='none' src='{html.escape(str(rel))}'></audio></article>"
        )
    page = (
        "<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>多情绪试音</title><style>body{max-width:1100px;margin:40px auto;padding:0 20px;font:16px system-ui;background:#111;color:#eee}article{background:#1d1d1f;padding:16px 20px;margin:14px 0;border-radius:12px}audio{width:100%}</style><h1>核心角色多情绪试音</h1>"
        + "".join(cards)
        + "</html>"
    )
    (root / "index.html").write_text(page, encoding="utf-8")


def main():
    cfg = json.loads((ROOT / "configs/emotion_cast.json").read_text(encoding="utf-8"))
    d = cfg["defaults"]
    root = ROOT / "outputs/raw/emotion_cast_v1"
    root.mkdir(parents=True, exist_ok=True)
    engine = VoxCPMEngine()
    rows = []
    total = len(cfg["characters"]) * len(cfg["styles"]) * d["candidates"]
    done = 0
    for character in cfg["characters"]:
        for style, spec in cfg["styles"].items():
            for i in range(1, d["candidates"] + 1):
                seed = 42 + (i - 1) * 3365
                wav = root / character / style / f"candidate_{i:02d}.wav"
                if complete(wav):
                    meta = json.loads(wav.with_suffix(".json").read_text())
                    status = "SKIP"
                else:
                    meta = engine.clone_voice(
                        character,
                        spec["text"],
                        wav,
                        style="neutral",
                        instruct=spec["instruct"],
                        seed=seed,
                        inference_timesteps=d["inference_timesteps"],
                        cfg_value=d["cfg_value"],
                        warmup_patches=d["warmup_patches"],
                        max_tokens=d["max_tokens"],
                    )
                    status = "DONE"
                done += 1
                row = {
                    "character_id": character,
                    "style": style,
                    "candidate": i,
                    "seed": seed,
                    "text": spec["text"],
                    "output": meta["output"],
                    "duration": meta["duration"],
                    "generation_seconds": meta["generation_seconds"],
                }
                rows.append(row)
                print(
                    f"[{done}/{total}] {status} {character}/{style}/candidate_{i:02d} {meta['duration']:.2f}s",
                    flush=True,
                )
                write_json(root / "manifest.json", {"rows": rows})
                render(root, rows)
    print(f"EMOTION CAST COMPLETE: {len(rows)} files\nOpen: {root / 'index.html'}")


if __name__ == "__main__":
    main()
