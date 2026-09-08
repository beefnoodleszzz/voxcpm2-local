import json
import subprocess
import sys
from pathlib import Path

from src.emotions import ROUTES


def test_emotion_regression():
    for case in json.loads((Path(__file__).parent / "fixtures/emotion.json").read_text()):
        assert ROUTES[case["emotion"]] == case["style"]


def test_cli_help_is_model_free():
    root = Path(__file__).resolve().parents[1]
    for script in (
        "prepare_episode.py",
        "batch_dub.py",
        "dub.py",
        "review_project.py",
        "finalize_project.py",
    ):
        result = subprocess.run(
            [sys.executable, str(root / "scripts" / script), "--help"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        assert result.returncode == 0, result.stderr
        assert "usage:" in result.stdout
