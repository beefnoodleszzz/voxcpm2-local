import time
from pathlib import Path
from uuid import uuid4

import gradio as gr

from .engine import VoxCPMEngine
from .voice_library import ROOT

engine = VoxCPMEngine()


def generate(mode, text, character, style, instruct, seed, candidates, steps):
    started = time.perf_counter()
    folder = ROOT / "outputs/raw/webui" / uuid4().hex
    common = dict(text=text, seed=int(seed), candidates=int(candidates),
                  inference_timesteps=int(steps), cfg_value=2.0, warmup_patches=0, max_tokens=2000)
    if mode == "design":
        metas = engine.generate_candidates("design_voice", folder, instruct=instruct, **common)
    elif mode == "controllable":
        metas = engine.generate_candidates("clone_voice", folder, character_id=character,
                                           style=style or None, instruct=instruct or None, **common)
    elif mode == "ultimate":
        metas = engine.generate_candidates("ultimate_clone", folder, character_id=character,
                                           style=style or None, **common)
    else:
        raise gr.Error("Batch 模式请使用 JSON CLI/API，以保留完整 manifest。")
    paths = [m["output"] for m in metas]
    return paths[0], "\n".join(paths) + f"\n总耗时: {time.perf_counter()-started:.2f}s"


with gr.Blocks(title="VoxCPM2 BF16 本地配音") as demo:
    gr.Markdown("# VoxCPM2 BF16 本地高质量配音\n原始音频保存为 48kHz PCM WAV，不覆盖已有输出。")
    mode = gr.Radio(["design", "controllable", "ultimate", "batch"], value="controllable", label="模式")
    text = gr.Textbox(label="台词", lines=4)
    character = gr.Textbox(label="Character ID")
    style = gr.Textbox(label="Style", value="neutral")
    instruct = gr.Textbox(
        label="表演/音色指令",
        value="自然生活化对白，语气克制，停连随语义变化，不要播音腔，不要机械匀速",
    )
    with gr.Row():
        seed = gr.Number(value=42, precision=0, label="Seed")
        candidates = gr.Slider(1, 5, value=3, step=1, label="Candidates")
        steps = gr.Radio([10, 20, 30], value=30, label="Diffusion steps")
    button = gr.Button("生成", variant="primary")
    audio = gr.Audio(label="首个 Candidate")
    result = gr.Textbox(label="输出")
    button.click(generate, [mode, text, character, style, instruct, seed, candidates, steps], [audio, result])


def main(): demo.launch(server_name="127.0.0.1", server_port=7862, show_error=True)

if __name__ == "__main__": main()
