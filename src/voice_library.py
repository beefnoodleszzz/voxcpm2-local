from __future__ import annotations

import json
from pathlib import Path

from .errors import ReferenceInvalid, VoiceNotFound
from .provenance import file_sha256, fingerprint
from .qc import inspect_qc

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
        if (
            not character_id
            or character_id in {".", ".."}
            or Path(character_id).name != character_id
        ):
            raise ValueError("Invalid character_id")
        path = self.root / character_id / "voice.json"
        if not path.is_file():
            raise VoiceNotFound(f"Unknown character: {character_id}")
        return json.loads(path.read_text(encoding="utf-8"))

    def resolve_style(self, character_id: str, style: str | None = None) -> dict:
        voice = self.get(character_id)
        selected = style or voice.get("default_style", "neutral")
        entry = voice.get("styles", {}).get(selected)
        if not entry:
            raise FileNotFoundError(f"Character {character_id!r} has no style {selected!r}")
        base = (self.root / character_id).resolve()
        audio = (base / entry["reference"]).resolve()
        transcript_path = (base / entry["transcript"]).resolve()
        if (
            not base.is_relative_to(self.root)
            or base not in audio.parents
            or base not in transcript_path.parents
            or not audio.is_file()
            or not transcript_path.is_file()
        ):
            raise ReferenceInvalid("Voice style files are missing or unsafe")
        transcript = transcript_path.read_text(encoding="utf-8").strip()
        if not transcript:
            raise ReferenceInvalid("Voice reference transcript is empty")
        return {"voice": voice, "style": selected, "audio": audio, "transcript": transcript}

    def reference_identity(self, character_id: str, style: str | None = None) -> dict:
        entry = self.resolve_style(character_id, style)
        checksum = file_sha256(entry["audio"])
        recorded = entry["voice"]["styles"][entry["style"]].get("reference_sha256")
        if recorded and recorded != checksum:
            raise ReferenceInvalid("Canonical reference differs from its recorded checksum")
        inspection = inspect_qc(entry["audio"], entry["transcript"], reference=True)
        if inspection["qc"]["status"] == "FAIL":
            raise ReferenceInvalid(f"Reference QC failed: {inspection['qc']['failures']}")
        return {
            "reference_sha256": checksum,
            "reference_transcript": entry["transcript"],
            "reference_transcript_sha256": fingerprint(entry["transcript"]),
            "voice_sha256": fingerprint(entry["voice"]),
            "resolved_style": entry["style"],
            "reference_qc": inspection["qc"],
            "reference_audio": inspection["stats"],
        }
