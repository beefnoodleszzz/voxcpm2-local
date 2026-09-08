"""Offline SenseVoice child process. Stdout is a single JSON object."""

import contextlib
import json
import os
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from src.voice_library import ROOT


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: asr_transcribe.py AUDIO.wav")
    audio = Path(sys.argv[1]).resolve()
    if not audio.is_file():
        raise FileNotFoundError(audio)
    lock = json.loads((ROOT / "configs/asr.model.lock.json").read_text())
    model_path = ROOT / lock["local_path"]
    for name in (
        "model.safetensors",
        "config.json",
        "am.mvn",
        "chn_jpn_yue_eng_ko_spectok.bpe.model",
    ):
        receipt = model_path / ".cache/huggingface/download" / f"{name}.metadata"
        if (
            not (model_path / name).is_file()
            or not receipt.is_file()
            or receipt.read_text().splitlines()[0] != lock["revision"]
        ):
            raise SystemExit(
                f"Pinned local ASR file missing/mismatched: {name}; run scripts/download_asr.py"
            )
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    with contextlib.redirect_stdout(sys.stderr):
        from mlx_audio.stt.utils import load_model

        model = load_model(model_path, strict=True)
        result = model.generate(str(audio), language="zh", use_itn=False, verbose=False)
    print(
        json.dumps(
            {"text": result.text, "model_repo": lock["repo"], "model_revision": lock["revision"]},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
