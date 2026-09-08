from __future__ import annotations

import os
import platform
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import psutil

from .audio_utils import TARGET_SAMPLE_RATE, write_json, write_wav
from .voice_library import ROOT, VoiceLibrary


class VoxCPMEngine:
    """Thread-safe, process-singleton BF16 VoxCPM2 engine."""
    _instance = None
    _instance_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, model_path: Path | str | None = None):
        if getattr(self, "_initialized", False):
            return
        self.model_path = Path(model_path or os.getenv("VOXCPM_MODEL_PATH", ROOT / "models/VoxCPM2-bf16")).resolve()
        self.model = None
        self.load_seconds = None
        self.lock = threading.Lock()
        self.voices = VoiceLibrary()
        self._initialized = True

    def load(self):
        if self.model is not None:
            return self.model
        if not self.model_path.is_dir():
            raise FileNotFoundError(f"BF16 model not found: {self.model_path}. Run scripts/download_model.py")
        from mlx_audio.tts import load
        started = time.perf_counter()
        self.model = load(self.model_path, lazy=False, strict=True)
        self.load_seconds = time.perf_counter() - started
        return self.model

    def _generate(self, *, text: str, output: Path, seed: int, inference_timesteps: int = 30,
                  cfg_value: float = 2.0, warmup_patches: int = 0, max_tokens: int = 2000,
                  character_id=None, style=None, instruct=None, ref_audio=None,
                  prompt_audio=None, prompt_text=None, mode="design") -> dict:
        if cfg_value < 2.0:
            raise ValueError("VoxCPM2 requires cfg_value >= 2.0")
        output = output.resolve()
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite raw output: {output}")
        with self.lock:
            model = self.load()
            mx.random.seed(seed)
            rss_before = psutil.Process().memory_info().rss
            started = time.perf_counter()
            result = next(model.generate(
                text=text, instruct=instruct, ref_audio=str(ref_audio) if ref_audio else None,
                prompt_audio=str(prompt_audio) if prompt_audio else None, prompt_text=prompt_text,
                inference_timesteps=inference_timesteps, cfg_value=cfg_value,
                warmup_patches=warmup_patches, max_tokens=max_tokens,
            ))
            elapsed = time.perf_counter() - started
            stats = write_wav(output, result.audio, result.sample_rate)
            rss_after = psutil.Process().memory_info().rss
        meta = {
            "model": "VoxCPM2-bf16", "model_path": str(self.model_path), "model_format": "BF16",
            "mode": mode, "character_id": character_id, "style": style, "text": text,
            "instruct": instruct, "seed": seed, "inference_timesteps": inference_timesteps,
            "cfg_value": cfg_value, "warmup_patches": warmup_patches, "max_tokens": max_tokens,
            "sample_rate": stats["sample_rate"], "generated_at": datetime.now(timezone.utc).isoformat(),
            "duration": stats["duration"], "generation_seconds": elapsed,
            "model_load_seconds": self.load_seconds, "peak": stats["peak"], "rms": stats["rms"],
            "mlx_peak_memory_gb": result.peak_memory_usage,
            "process_rss_before_gb": rss_before / 1e9, "process_rss_after_gb": rss_after / 1e9,
            "output": str(output),
        }
        write_json(output.with_suffix(".json"), meta)
        return meta

    def design_voice(self, text, instruct, output, **quality):
        return self._generate(text=text, instruct=instruct, output=Path(output), mode="design", **quality)

    def clone_voice(self, character_id, text, output, style=None, instruct=None, **quality):
        entry = self.voices.resolve_style(character_id, style)
        return self._generate(text=text, output=Path(output), character_id=character_id,
                              style=entry["style"], instruct=instruct, ref_audio=entry["audio"],
                              mode="controllable", **quality)

    def ultimate_clone(self, character_id, text, output, style=None, **quality):
        entry = self.voices.resolve_style(character_id, style)
        return self._generate(text=text, output=Path(output), character_id=character_id,
                              style=entry["style"], ref_audio=entry["audio"],
                              prompt_audio=entry["audio"], prompt_text=entry["transcript"],
                              mode="ultimate", **quality)

    def generate_candidates(self, method: str, output_dir: Path, candidates=3, seed=42, **kwargs):
        output_dir.mkdir(parents=True, exist_ok=True)
        seeds = [seed + i * 3365 for i in range(candidates)]
        files = []
        for i, candidate_seed in enumerate(seeds, 1):
            output = output_dir / f"candidate_{i:02d}.wav"
            fn = getattr(self, method)
            meta = fn(output=output, seed=candidate_seed, **kwargs)
            files.append(meta)
        return files

    def health(self):
        return {"status": "ok", "backend": "mlx", "model": "VoxCPM2-bf16",
                "model_format": "BF16", "model_available": self.model_path.is_dir(),
                "model_loaded": self.model is not None, "sample_rate": TARGET_SAMPLE_RATE,
                "device": f"Apple Silicon ({platform.machine()})"}
