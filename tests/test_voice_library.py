import json

import numpy as np
import pytest

from src.audio_utils import write_wav
from src.voice_library import VoiceLibrary


def make_voice(tmp_path):
    base = tmp_path / "v"
    base.mkdir()
    write_wav(base / "ref.wav", np.sin(np.arange(48000) * 0.05) * 0.1)
    (base / "text.txt").write_text("你好")
    (base / "voice.json").write_text(
        json.dumps(
            {"id": "v", "styles": {"neutral": {"reference": "ref.wav", "transcript": "text.txt"}}}
        )
    )
    return VoiceLibrary(tmp_path), base


def test_identity_changes_with_transcript(tmp_path):
    library, base = make_voice(tmp_path)
    before = library.reference_identity("v")
    (base / "text.txt").write_text("您好")
    after = library.reference_identity("v")
    assert before["reference_sha256"] == after["reference_sha256"]
    assert before["reference_transcript_sha256"] != after["reference_transcript_sha256"]


def test_transcript_escape_and_dot_ids(tmp_path):
    library, base = make_voice(tmp_path)
    (tmp_path / "outside.txt").write_text("secret")
    voice = json.loads((base / "voice.json").read_text())
    voice["styles"]["neutral"]["transcript"] = "../outside.txt"
    (base / "voice.json").write_text(json.dumps(voice))
    with pytest.raises(ValueError):
        library.resolve_style("v")
    with pytest.raises(ValueError):
        library.get("..")
