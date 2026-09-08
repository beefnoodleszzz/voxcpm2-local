from __future__ import annotations

import json
import fcntl
import platform
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil

from .audio_utils import TARGET_SAMPLE_RATE, write_json, write_wav
from .voice_library import ROOT, VoiceLibrary
from .errors import AudioValidationError, GenerationError, OutOfMemory, TranscriptQCError
from .generation_policy import accepted_automatically, plan_attempts, rank_candidates, retry_reason
from .pronunciation import prepare_text
from .provenance import (
    file_sha256,
    fingerprint,
    model_identity,
    model_path as default_model_path,
    runtime_versions,
)
from .qc import audio_qc
from .transcript_qc import TranscriptVerifier
from .schemas import QualityOptions


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
        with self._instance_lock:
            if getattr(self, "_initialized", False):
                return
            self.model_path = Path(model_path).resolve() if model_path else default_model_path()
            self.model = None
            self.load_seconds = None
            self.lock = threading.RLock()
            self.voices = VoiceLibrary()
            self._model_lease = None
            self._initialized = True

    def load(self):
        if self.model is not None:
            return self.model
        if not self.model_path.is_dir():
            raise FileNotFoundError(
                f"BF16 model not found: {self.model_path}. Run scripts/download_model.py"
            )
        from mlx_audio.tts import load

        model_identity(self.model_path)
        if self._model_lease is None:
            lock_path = ROOT / "outputs/.bf16-model.lock"
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            lease = lock_path.open("a")
            try:
                fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                lease.close()
                raise GenerationError(
                    "Another local process owns the BF16 model. Stop its API/WebUI/CLI before loading a second model."
                ) from exc
            self._model_lease = lease
        started = time.perf_counter()
        self.model = load(self.model_path, lazy=False, strict=True)
        self.load_seconds = time.perf_counter() - started
        return self.model

    def _generate(
        self,
        *,
        text: str,
        output: Path,
        seed: int,
        inference_timesteps: int = 30,
        cfg_value: float = 2.0,
        warmup_patches: int = 0,
        max_tokens: int = 2000,
        character_id=None,
        style=None,
        instruct=None,
        ref_audio=None,
        prompt_audio=None,
        prompt_text=None,
        mode="design",
        preparation=None,
        context=None,
        verifier=None,
    ) -> dict:
        QualityOptions(
            seed=seed,
            inference_timesteps=inference_timesteps,
            cfg_value=cfg_value,
            warmup_patches=warmup_patches,
            max_tokens=max_tokens,
        )
        if not text.strip():
            raise ValueError("Text cannot be empty")
        output = output.resolve()
        if output.exists() or output.with_suffix(".json").exists():
            raise FileExistsError(f"Refusing to overwrite raw output: {output}")
        preparation = preparation or prepare_text(text)
        text = preparation["text_spoken"]
        reference = self.voices.reference_identity(character_id, style) if character_id else {}
        identity = model_identity(self.model_path)
        versions = runtime_versions()
        import mlx.core as mx

        with self.lock:
            try:
                model = self.load()
                mx.random.seed(seed)
                rss_before = psutil.Process().memory_info().rss
                started = time.perf_counter()
                result = next(
                    model.generate(
                        text=text,
                        instruct=instruct,
                        ref_audio=str(ref_audio) if ref_audio else None,
                        prompt_audio=str(prompt_audio) if prompt_audio else None,
                        prompt_text=prompt_text,
                        inference_timesteps=inference_timesteps,
                        cfg_value=cfg_value,
                        warmup_patches=warmup_patches,
                        max_tokens=max_tokens,
                    )
                )
                mx.eval(result.audio)
                elapsed = time.perf_counter() - started
                stats = write_wav(output, result.audio, result.sample_rate)
                rss_after = psutil.Process().memory_info().rss
            except (MemoryError, RuntimeError) as exc:
                if isinstance(exc, GenerationError):
                    raise
                is_oom = isinstance(exc, MemoryError) or any(
                    word in str(exc).lower()
                    for word in (
                        "out of memory",
                        "insufficient memory",
                        "allocation",
                        "metal device",
                    )
                )
                error = OutOfMemory if is_oom else GenerationError
                self.log_event(
                    "generation_error",
                    output=str(output),
                    error=str(exc),
                    text_length=len(text),
                    model_format="BF16",
                )
                raise error(
                    f"{exc}; text_length={len(text)}, serial BF16 generation. "
                    "Close memory-heavy apps, shorten/segment text and rerun. No precision fallback."
                ) from exc
            qc = audio_qc(stats, text)
            verifier = verifier or self.transcript_verifier()
            try:
                transcript = verifier.verify(output, preparation["text_expected"])
            except TranscriptQCError as exc:
                transcript = {
                    "status": "ERROR",
                    "pass": None,
                    "cer": None,
                    "error": str(exc),
                    **verifier.identity,
                }
                self.log_event("asr_error", output=str(output), error=str(exc))
        meta = {
            "schema_version": 2,
            "engine": "mlx",
            **identity,
            **versions,
            **reference,
            **preparation,
            "generation_context": context or {},
            "model": "VoxCPM2-bf16",
            "model_path": str(self.model_path),
            "model_format": "BF16",
            "mode": mode,
            "character_id": character_id,
            "style": style,
            "text": text,
            "instruct": instruct,
            "seed": seed,
            "inference_timesteps": inference_timesteps,
            "cfg_value": cfg_value,
            "warmup_patches": warmup_patches,
            "max_tokens": max_tokens,
            "sample_rate": stats["sample_rate"],
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "duration": stats["duration"],
            "generation_seconds": elapsed,
            "model_load_seconds": self.load_seconds,
            "peak": stats["peak"],
            "rms": stats["rms"],
            "mlx_peak_memory_gb": result.peak_memory_usage,
            "process_rss_before_gb": rss_before / 1e9,
            "process_rss_after_gb": rss_after / 1e9,
            "output": str(output),
            "audio_duration": stats["duration"],
            "rtf": elapsed / stats["duration"] if stats["duration"] else None,
            "channels": stats["channels"],
            "subtype": stats["subtype"],
            "audio_sha256": file_sha256(output),
            "qc": qc,
            "audio_stats": stats,
            "transcript_qc": transcript,
            "speaker_similarity": None,
            "human_review_required": True,
        }
        write_json(output.with_suffix(".json"), meta, exclusive=True)
        self.log_event(
            "generated",
            output=str(output),
            character_id=character_id,
            seed=seed,
            steps=inference_timesteps,
            generation_seconds=elapsed,
            qc=qc["status"],
            transcript_qc=transcript["status"],
        )
        return meta

    def transcript_verifier(self):
        return TranscriptVerifier(json.loads((ROOT / "configs/asr.json").read_text()))

    def log_event(self, event, **fields):
        path = ROOT / "outputs/qc/events.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {"event": event, "time": datetime.now(timezone.utc).isoformat(), **fields},
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )

    def design_voice(self, text, instruct, output, **quality):
        return self._generate(
            text=text, instruct=instruct, output=Path(output), mode="design", **quality
        )

    def clone_voice(self, character_id, text, output, style=None, instruct=None, **quality):
        entry = self.voices.resolve_style(character_id, style)
        return self._generate(
            text=text,
            output=Path(output),
            character_id=character_id,
            style=entry["style"],
            instruct=instruct,
            ref_audio=entry["audio"],
            mode="controllable",
            **quality,
        )

    def ultimate_clone(self, character_id, text, output, style=None, **quality):
        entry = self.voices.resolve_style(character_id, style)
        return self._generate(
            text=text,
            output=Path(output),
            character_id=character_id,
            style=entry["style"],
            ref_audio=entry["audio"],
            prompt_audio=entry["audio"],
            prompt_text=entry["transcript"],
            mode="ultimate",
            **quality,
        )

    def generate_candidates(
        self,
        method: str,
        output_dir: Path,
        candidates=3,
        seed=42,
        profile=None,
        normalization_profile="none",
        project_lexicon=None,
        episode_lexicon=None,
        preparation=None,
        resume=False,
        context=None,
        **kwargs,
    ):
        if method not in {"design_voice", "clone_voice", "ultimate_clone"}:
            raise ValueError(f"Unknown generation method: {method}")
        policy, attempts = plan_attempts(
            profile=profile,
            candidates=candidates,
            seed=seed,
            steps=kwargs.pop("inference_timesteps", 30),
        )
        preparation = preparation or prepare_text(
            kwargs["text"], normalization_profile, project_lexicon, episode_lexicon
        )
        context = {**(context or {}), "policy": policy}
        # Include voice/model identities for direct candidate resume as well as batches.
        context["model"] = model_identity(self.model_path)
        if kwargs.get("character_id"):
            context["reference"] = self.voices.reference_identity(
                kwargs["character_id"], kwargs.get("style")
            )
        verifier = self.transcript_verifier()
        output_dir.mkdir(parents=True, exist_ok=True)
        files = []
        for attempt in attempts:
            i, candidate_seed = attempt["candidate"], attempt["seed"]
            output = output_dir / f"candidate_{i:02d}.wav"
            expected = fingerprint(
                {
                    "context": context,
                    "preparation": preparation,
                    "method": method,
                    "kwargs": kwargs,
                    "attempt": attempt,
                    "asr": verifier.identity,
                }
            )
            sidecar = output.with_suffix(".json")
            error_path = output.with_suffix(".error.json")
            if resume and error_path.is_file() and not output.exists():
                previous_error = json.loads(error_path.read_text())
                if previous_error.get("attempt_hash") != expected:
                    raise FileExistsError(f"Failed attempt fingerprint mismatch: {error_path}")
                continue
            if resume and output.is_file() and sidecar.is_file():
                meta = json.loads(sidecar.read_text())
                if meta.get("generation_context", {}).get("attempt_hash") != expected or meta.get(
                    "audio_sha256"
                ) != file_sha256(output):
                    raise FileExistsError(f"Resume fingerprint mismatch: {output}")
            else:
                fn = getattr(self, method)
                try:
                    meta = fn(
                        output=output,
                        seed=candidate_seed,
                        preparation=preparation,
                        inference_timesteps=attempt["inference_timesteps"],
                        verifier=verifier,
                        context={**context, "attempt_hash": expected, "candidate_number": i},
                        **kwargs,
                    )
                except AudioValidationError as exc:
                    if not error_path.exists():
                        write_json(
                            error_path,
                            {"error": str(exc), "attempt": attempt, "attempt_hash": expected},
                            exclusive=True,
                        )
                    self.log_event("candidate_rejected", output=str(output), error=str(exc))
                    continue
            files.append(meta)
            if policy["early_stop"] and accepted_automatically(meta):
                self.log_event("early_stop", output=str(output), candidate=i)
                break
            self.log_event("candidate_review", output=str(output), **retry_reason(meta))
        if not files:
            raise AudioValidationError(
                "All scheduled candidates failed audio validation; see .error.json files"
            )
        write_json(output_dir / "selection.json", rank_candidates(files))
        return files

    def health(self):
        return {
            "status": "ok",
            "backend": "mlx",
            "model": "VoxCPM2-bf16",
            "model_format": "BF16",
            "model_available": self.model_path.is_dir(),
            "model_loaded": self.model is not None,
            "sample_rate": TARGET_SAMPLE_RATE,
            "device": f"Apple Silicon ({platform.machine()})",
        }
