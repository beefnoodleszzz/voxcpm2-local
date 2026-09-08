from pathlib import Path

import numpy as np
import pytest

import src.engine as module
from src.audio_utils import write_json, write_wav
from src.engine import VoxCPMEngine
from src.provenance import file_sha256
from src.transcript_qc import TranscriptVerifier


@pytest.fixture
def engine(monkeypatch):
    engine = object.__new__(VoxCPMEngine)
    engine.model_path = Path("unused")
    monkeypatch.setattr(module, "model_identity", lambda path: {"model_revision": "test"})
    engine.log_event = lambda *args, **kwargs: None
    engine.transcript_verifier = lambda: TranscriptVerifier()
    return engine


def fake_generator(statuses, calls):
    def generate(output, seed, inference_timesteps, context, **kwargs):
        calls.append((seed, inference_timesteps))
        write_wav(output, np.sin(np.arange(4800) * 0.05) * 0.1)
        status = statuses[min(len(calls) - 1, len(statuses) - 1)]
        meta = {
            "output": str(output),
            "audio_sha256": file_sha256(output),
            "generation_context": context,
            "qc": {"status": "PASS"},
            "transcript_qc": {"status": status},
        }
        write_json(output.with_suffix(".json"), meta, exclusive=True)
        return meta

    return generate


def test_adaptive_early_stop_and_resume(engine, tmp_path):
    calls = []
    engine.design_voice = fake_generator(["FAIL", "PASS"], calls)
    first = engine.generate_candidates(
        "design_voice", tmp_path, profile="quality", text="你好", instruct="自然"
    )
    assert len(first) == 2
    assert [steps for seed, steps in calls] == [10, 20]
    again = engine.generate_candidates(
        "design_voice", tmp_path, profile="quality", text="你好", instruct="自然", resume=True
    )
    assert len(again) == 2 and len(calls) == 2


def test_uncalibrated_consumes_bounded_schedule(engine, tmp_path):
    calls = []
    engine.design_voice = fake_generator(["WARN"], calls)
    result = engine.generate_candidates(
        "design_voice", tmp_path, profile="quality", text="你好", instruct="自然"
    )
    assert len(result) == 3
    with pytest.raises(FileExistsError):
        engine.generate_candidates(
            "design_voice",
            tmp_path,
            profile="quality",
            text="不同台词",
            instruct="自然",
            resume=True,
        )
