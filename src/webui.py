import time
from pathlib import Path
from uuid import uuid4

import gradio as gr

from .engine import VoxCPMEngine
from .voice_library import ROOT
from .emotions import INSTRUCTIONS
from .generation_policy import rank_candidates
from .review import load_candidate, record_review

engine = VoxCPMEngine()


def generate(
    mode,
    text,
    character,
    style,
    instruct,
    seed,
    candidates,
    steps,
    profile="production",
    normalization="dialogue",
):
    started = time.perf_counter()
    folder = ROOT / "outputs/raw/webui" / uuid4().hex
    common = dict(
        text=text,
        seed=int(seed),
        candidates=int(candidates),
        inference_timesteps=int(steps),
        cfg_value=2.0,
        warmup_patches=0,
        max_tokens=2000,
        profile=None if profile == "custom" else profile,
        normalization_profile=normalization,
    )
    if mode == "design":
        metas = engine.generate_candidates("design_voice", folder, instruct=instruct, **common)
    elif mode == "controllable":
        metas = engine.generate_candidates(
            "clone_voice",
            folder,
            character_id=character,
            style=style or None,
            instruct=instruct or None,
            **common,
        )
    elif mode == "ultimate":
        metas = engine.generate_candidates(
            "ultimate_clone", folder, character_id=character, style=style or None, **common
        )
    else:
        raise gr.Error("Batch 模式请使用 JSON CLI/API，以保留完整 manifest。")
    paths = [m["output"] for m in metas]
    ranking = rank_candidates(metas)
    chosen = ranking["recommended_output"] or paths[0]
    status = f"生成 {len(paths)} 条候选，总耗时 {time.perf_counter() - started:.2f}s。请试听后评分或批准。"
    return (
        gr.update(choices=paths, value=chosen),
        gr.update(choices=paths, value=paths[1] if len(paths) > 1 else paths[0]),
        status,
        ranking,
    )


def voice_details(character, style):
    if not character:
        return "请选择角色", None
    entry = engine.voices.resolve_style(character, style or "neutral")
    info = engine.voices.reference_identity(character, style or "neutral")
    description = entry["voice"].get("description", "")
    return (
        f"{description}\nReference QC: {info['reference_qc']['status']} · {', '.join(info['reference_qc']['warnings'])}",
        str(entry["audio"]),
    )


def inspect_candidate(path):
    if not path:
        return None, {}
    meta = load_candidate(Path(path), ROOT)
    return path, meta


def submit_review(path, decision, reviewer, listened, notes, naturalness, emotion, identity):
    if not path:
        raise gr.Error("请先选择候选音频。")
    try:
        result = record_review(
            Path(path),
            ROOT,
            decision=decision,
            reviewer=reviewer,
            listened=listened,
            notes=notes,
            ratings={"naturalness": naturalness, "emotion": emotion, "identity": identity},
        )
    except (ValueError, OSError) as exc:
        raise gr.Error(str(exc)) from exc
    return f"已记录 {decision}。{result.get('approved_output', result['review_path'])}"


with gr.Blocks(title="VoxCPM2 BF16 本地配音") as demo:
    gr.Markdown("# VoxCPM2 BF16 本地高质量配音\n原始音频保存为 48kHz PCM WAV，不覆盖已有输出。")
    mode = gr.Radio(
        ["design", "controllable", "ultimate", "batch"], value="controllable", label="模式"
    )
    text = gr.Textbox(label="台词", lines=4)
    character = gr.Dropdown(
        choices=[(v.get("name", v["id"]), v["id"]) for v in engine.voices.list()], label="角色"
    )
    style = gr.Textbox(label="Style", value="neutral")
    character_description = gr.Textbox(label="角色描述与参考检查", interactive=False)
    reference_audio = gr.Audio(label="试听 reference", interactive=False, type="filepath")
    emotion = gr.Dropdown(list(INSTRUCTIONS), value="neutral", label="情绪 / 表演方向")
    instruct = gr.Textbox(
        label="表演/音色指令",
        value="自然生活化对白，语气克制，停连随语义变化，不要播音腔，不要机械匀速",
    )
    with gr.Row():
        profile = gr.Dropdown(
            ["production", "fast", "balanced", "quality", "ultimate", "custom"],
            value="production",
            label="生成 Profile",
            info="production 保持 30 steps × 3；fast/balanced/quality 尚待听感 A/B 验证",
        )
        normalization = gr.Dropdown(
            ["none", "dialogue", "narration"], value="dialogue", label="文本处理"
        )
    with gr.Accordion("高级参数（steps / candidates 仅 custom profile 生效）", open=False):
        seed = gr.Number(value=42, precision=0, label="Seed")
        candidates = gr.Slider(1, 5, value=3, step=1, label="Candidates")
        steps = gr.Radio([10, 20, 30], value=30, label="Diffusion steps")
    button = gr.Button("生成", variant="primary")
    result = gr.Textbox(label="生成状态", interactive=False)
    ranking = gr.JSON(label="候选推荐（最终仍需试听）")
    with gr.Row():
        candidate = gr.Dropdown([], label="候选 A / 审核对象", allow_custom_value=True)
        comparison = gr.Dropdown([], label="候选 B", allow_custom_value=True)
    with gr.Row():
        audio = gr.Audio(label="试听 A", interactive=False, type="filepath")
        audio_b = gr.Audio(label="试听 B", interactive=False, type="filepath")
    with gr.Accordion("QC / ASR transcript / Metadata", open=True):
        metadata = gr.JSON(label="候选 A 检查与生成记录")
    existing = gr.Textbox(label="已有 raw WAV 路径（可加载 batch / benchmark 候选）")
    load_existing = gr.Button("加载已有候选")
    reviewer = gr.Textbox(label="审核人")
    with gr.Row():
        naturalness = gr.Slider(1, 5, value=3, step=1, label="自然度")
        emotion_rating = gr.Slider(1, 5, value=3, step=1, label="情绪准确度")
        identity_rating = gr.Slider(1, 5, value=3, step=1, label="角色一致性")
    notes = gr.Textbox(label="试听结论（QC / ASR 未通过或未校准时必填）")
    listened = gr.Checkbox(label="我已完整试听当前候选，并核对台词与角色", value=False)
    decision = gr.Radio(["rate", "approve", "reject"], value="rate", label="审核决定")
    review_button = gr.Button("记录审核 / 批准时保存 Approved Master")
    review_status = gr.Textbox(label="审核结果", interactive=False)
    character.change(voice_details, [character, style], [character_description, reference_audio])
    style.change(voice_details, [character, style], [character_description, reference_audio])
    emotion.change(lambda value: INSTRUCTIONS[value], emotion, instruct)
    button.click(
        generate,
        [mode, text, character, style, instruct, seed, candidates, steps, profile, normalization],
        [candidate, comparison, result, ranking],
        concurrency_limit=1,
        concurrency_id="inference",
    )
    candidate.change(inspect_candidate, candidate, [audio, metadata])
    candidate.change(lambda: False, outputs=listened)
    comparison.change(lambda path: inspect_candidate(path)[0], comparison, audio_b)
    load_existing.click(lambda path: gr.update(choices=[path], value=path), existing, candidate)
    review_button.click(
        submit_review,
        [
            candidate,
            decision,
            reviewer,
            listened,
            notes,
            naturalness,
            emotion_rating,
            identity_rating,
        ],
        review_status,
    )


def main():
    demo.queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1",
        server_port=7862,
        show_error=True,
        allowed_paths=[str(ROOT / "outputs"), str(ROOT / "voices")],
    )


if __name__ == "__main__":
    main()
