from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class VoiceLibrary:
    def __init__(self, root: Path | None = None):
        self.root = (root or ROOT / "voices").resolve()

    def list(self) -> list[dict]:
        voices = []
        for path in sorted(self.root.glob("*/voice.json")):
            voices.append(json.loads(path.read_text(encoding="utf-8")))
        return voices

    def get(self, character_id: str) -> dict:
        if not character_id or Path(character_id).name != character_id:
            raise ValueError("Invalid character_id")
        path = self.root / character_id / "voice.json"
        if not path.is_file():
            raise FileNotFoundError(f"Unknown character: {character_id}")
        return json.loads(path.read_text(encoding="utf-8"))

    def resolve_style(self, character_id: str, style: str | None = None) -> dict:
        voice = self.get(character_id)
        selected = style or voice.get("default_style", "neutral")
        entry = voice.get("styles", {}).get(selected)
        if not entry:
            raise FileNotFoundError(f"Character {character_id!r} has no style {selected!r}")
        base = self.root / character_id
        audio = (base / entry["reference"]).resolve()
        transcript_path = (base / entry["transcript"]).resolve()
        if base not in audio.parents or not audio.is_file() or not transcript_path.is_file():
            raise FileNotFoundError("Voice style files are missing or unsafe")
        return {"voice": voice, "style": selected, "audio": audio,
                "transcript": transcript_path.read_text(encoding="utf-8").strip()}

