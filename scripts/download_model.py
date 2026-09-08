#!/usr/bin/env python3
from pathlib import Path
import json

from huggingface_hub import snapshot_download

import _bootstrap  # noqa: F401
from src.provenance import read_model_lock, model_identity

ROOT = Path(__file__).resolve().parents[1]
REPO = "mlx-community/VoxCPM2-bf16"
TARGET = ROOT / "models/VoxCPM2-bf16"


def main():
    TARGET.mkdir(parents=True, exist_ok=True)
    print(f"Downloading only BF16 model {REPO} -> {TARGET}")
    # Current huggingface_hub resumes local_dir downloads automatically.
    lock = read_model_lock()
    snapshot_download(repo_id=lock["repo"], revision=lock["revision"], local_dir=TARGET)
    required = [TARGET / "config.json", TARGET / "tokenizer.json"]
    weights = list(TARGET.glob("*.safetensors"))
    missing = [str(p.name) for p in required if not p.is_file()]
    if missing or not weights:
        raise RuntimeError(f"Incomplete model. Missing={missing}, safetensors={len(weights)}")
    cfg = json.loads((TARGET / "config.json").read_text())
    quant = cfg.get("quantization") or cfg.get("quantization_config")
    if quant:
        raise RuntimeError(f"Downloaded model unexpectedly declares quantization: {quant}")
    model_identity(TARGET)
    size = sum(p.stat().st_size for p in TARGET.rglob("*") if p.is_file())
    print(f"BF16 model verified: {len(weights)} weight file(s), {size / 1024**3:.2f} GiB")


if __name__ == "__main__":
    main()
