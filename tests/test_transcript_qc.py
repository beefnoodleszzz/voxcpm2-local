import sys

import pytest

from src.transcript_qc import TranscriptVerifier, align_transcript, calibration_threshold


def test_edits():
    assert align_transcript("我最后问你一次", "我最后问你一遍")["substitutions"]
    assert align_transcript("我最后问你一次", "我问你一次")["deletions"]
    result = align_transcript("你好", "你你好")
    assert result["insertions"] and result["cer"] == 0.5
    assert align_transcript("今天2026年。", "今天二零二六年")["cer"] == 0


def test_empty_recognized_is_full_deletion():
    assert align_transcript("你好", "")["cer"] == 1


def test_uncalibrated_is_advisory(tmp_path):
    path = tmp_path / "a.wav"
    path.write_bytes(b"audio")
    verifier = TranscriptVerifier(
        {"backend_id": "test", "command": [sys.executable, "-c", 'print(\'{"text":"你好"}\')']}
    )
    result = verifier.verify(path, "你好")
    assert result["status"] == "WARN" and result["pass"] is None
    assert result["cer"] == 0
    assert TranscriptVerifier().verify(path, "你好")["status"] == "UNAVAILABLE"


def test_calibration_requires_evidence():
    with pytest.raises(ValueError):
        calibration_threshold({"threshold": 0.1}, "test")
    data = {
        "threshold": 0.1,
        "backend_id": "test",
        "reviewed_samples": 30,
        "human_reviewed": True,
        "dataset_sha256": "abc",
    }
    assert calibration_threshold(data, "test") == 0.1
    with pytest.raises(ValueError):
        calibration_threshold(data, "other")
