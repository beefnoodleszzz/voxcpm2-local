import numpy as np
import soundfile as sf

from src.audio_utils import write_wav
from src.qc import inspect_qc


def test_silence_fails(tmp_path):
    path = tmp_path / "silent.wav"
    write_wav(path, np.zeros(48000))
    qc = inspect_qc(path)["qc"]
    assert qc["status"] == "FAIL"
    assert "silence_only" in qc["failures"]


def test_reference_length_is_warning(tmp_path):
    path = tmp_path / "ref.wav"
    write_wav(path, np.sin(np.arange(48000) * 0.05) * 0.1)
    qc = inspect_qc(path, reference=True)["qc"]
    assert qc["status"] == "WARN"
    assert "reference_too_short" in qc["warnings"]


def test_nonfinite_float_file(tmp_path):
    path = tmp_path / "nan.wav"
    sf.write(path, np.array([0.0, float("nan"), float("inf")]), 48000, subtype="FLOAT")
    qc = inspect_qc(path)["qc"]
    assert qc["status"] == "FAIL"
    assert "nan" in qc["failures"] and "inf" in qc["failures"]
