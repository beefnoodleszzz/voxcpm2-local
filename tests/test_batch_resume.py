from types import SimpleNamespace

import numpy as np
import pytest

import src.batch as batch_module
from src.audio_utils import write_json, write_wav
from src.batch import run_batch
from src.provenance import file_sha256, fingerprint
from src.schemas import BatchRequest


@pytest.fixture
def fake_engine(monkeypatch):
    calls = []

    def preflight(req, engine):
        return {
            "fingerprint": fingerprint(req.model_dump()),
            "identity": {"preparations": {line.id: None for line in req.lines}},
        }

    monkeypatch.setattr(batch_module, "preflight", preflight)

    def generate(method, folder, **kwargs):
        calls.append(folder)
        path = folder / "candidate_01.wav"
        write_wav(path, np.sin(np.arange(4800) * 0.05) * 0.1)
        meta = {"output": str(path), "audio_sha256": file_sha256(path), "qc": {"status": "PASS"}}
        write_json(path.with_suffix(".json"), meta, exclusive=True)
        return [meta]

    return SimpleNamespace(generate_candidates=generate, calls=calls)


def request(text="你好"):
    return BatchRequest(
        project="ep", lines=[{"id": "L1", "mode": "design", "text": text, "instruct": "自然"}]
    )


def test_resume_and_changed_input(tmp_path, fake_engine):
    first = run_batch(request(), fake_engine, tmp_path)
    again = run_batch(request(), fake_engine, tmp_path)
    assert first["manifest"] == again["manifest"]
    assert len(fake_engine.calls) == 1
    changed = run_batch(request("您好"), fake_engine, tmp_path)
    assert changed["revision"] != first["revision"]
    assert len(fake_engine.calls) == 2


def test_corruption_preserves_old_and_creates_revision(tmp_path, fake_engine):
    first = run_batch(request(), fake_engine, tmp_path)
    from pathlib import Path

    path = Path(first["lines"][0]["candidates"][0]["output"])
    path.write_bytes(b"corrupt")
    second = run_batch(request(), fake_engine, tmp_path)
    assert second["revision"] != first["revision"]
    assert path.read_bytes() == b"corrupt"


def test_dry_run_no_output(tmp_path, fake_engine):
    assert run_batch(request(), fake_engine, tmp_path, dry_run=True)["dry_run"]
    assert not list(tmp_path.iterdir())


def test_partial_manifest_resumes_completed_lines(tmp_path, fake_engine):
    req = request()
    req.lines.append(req.lines[0].model_copy(update={"id": "L2"}))
    generate = fake_engine.generate_candidates

    def interrupted(method, folder, **kwargs):
        if folder.name == "L2":
            raise RuntimeError("simulated interruption")
        return generate(method, folder, **kwargs)

    fake_engine.generate_candidates = interrupted
    with pytest.raises(RuntimeError):
        run_batch(req, fake_engine, tmp_path)
    fake_engine.generate_candidates = generate
    result = run_batch(req, fake_engine, tmp_path)
    assert len(result["lines"]) == 2
    assert [p.name for p in fake_engine.calls] == ["L1", "L2"]
