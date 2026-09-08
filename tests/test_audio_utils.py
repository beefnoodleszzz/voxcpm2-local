import numpy as np
import pytest
import soundfile as sf

from src.audio_utils import inspect_audio, write_json, write_wav


def test_pcm24_and_immutable(tmp_path):
    path = tmp_path / "test.wav"
    samples = 0.1 * np.sin(np.arange(48000) * 0.1)
    stats = write_wav(path, samples)
    original = path.read_bytes()
    assert stats["subtype"] == "PCM_24"
    assert stats["channels"] == 1
    with pytest.raises(FileExistsError):
        write_wav(path, samples)
    assert path.read_bytes() == original
    assert sf.info(path).samplerate == 48000


@pytest.mark.parametrize(
    "samples",
    [
        np.array([]),
        np.array([float("nan"), 0]),
        np.array([float("inf"), 0]),
        np.array([1.2, -1.2]),
        np.zeros((2, 2)),
    ],
)
def test_reject_before_pcm_conversion(tmp_path, samples):
    with pytest.raises(ValueError):
        write_wav(tmp_path / "bad.wav", samples)
    assert not (tmp_path / "bad.wav").exists()


def test_antiphase_channels_do_not_hide_clipping(tmp_path):
    path = tmp_path / "stereo.wav"
    sf.write(path, np.array([[1.0, -1.0], [0.0, 0.0]]), 48000, subtype="FLOAT")
    assert inspect_audio(path)["clipping"]


def test_json_exclusive_and_nonfinite(tmp_path):
    path = tmp_path / "metadata.json"
    write_json(path, {"ok": True}, exclusive=True)
    with pytest.raises(FileExistsError):
        write_json(path, {}, exclusive=True)
    with pytest.raises(ValueError):
        write_json(path, {"bad": float("nan")})
