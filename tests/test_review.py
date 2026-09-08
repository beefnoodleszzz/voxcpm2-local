import numpy as np
import pytest

from src.audio_utils import write_json, write_wav
from src.mastering import finalize_audio
from src.provenance import file_sha256
from src.review import record_review


def candidate(tmp_path):
    path = tmp_path / "outputs/raw/ep/run_0001/L1/candidate_01.wav"
    write_wav(path, np.sin(np.arange(48000) * 0.05) * 0.1)
    write_json(
        path.with_suffix(".json"),
        {
            "output": str(path),
            "audio_sha256": file_sha256(path),
            "text": "你好",
            "transcript_qc": {"status": "WARN"},
        },
    )
    return path


def test_human_confirmation_and_warning_notes_required(tmp_path):
    path = candidate(tmp_path)
    with pytest.raises(ValueError):
        record_review(path, tmp_path, decision="approve", reviewer="tester", listened=False)
    with pytest.raises(ValueError):
        record_review(path, tmp_path, decision="approve", reviewer="tester", listened=True)


def test_immutable_promotion_and_delivery(tmp_path):
    from pathlib import Path

    path = candidate(tmp_path)
    original = path.read_bytes()
    a = record_review(
        path,
        tmp_path,
        decision="approve",
        reviewer="tester",
        listened=True,
        notes="Test fixture approval",
    )
    b = record_review(
        path,
        tmp_path,
        decision="approve",
        reviewer="tester",
        listened=True,
        notes="Separate revision",
    )
    assert a["approved_output"] != b["approved_output"]
    delivery = finalize_audio(Path(a["approved_output"]), tmp_path)
    assert Path(delivery["output"]).read_bytes() == original
    assert path.read_bytes() == original
    assert delivery["qc"]["status"] == "PASS"


def test_raw_cannot_be_delivered(tmp_path):
    with pytest.raises(ValueError):
        finalize_audio(candidate(tmp_path), tmp_path)


def test_tampering_prevents_approval(tmp_path):
    path = candidate(tmp_path)
    path.write_bytes(b"changed")
    with pytest.raises(ValueError):
        record_review(
            path, tmp_path, decision="approve", reviewer="tester", listened=True, notes="x"
        )
