"""Add measured reference provenance; preserve existing fields and every audio byte."""

import argparse
import json
from datetime import datetime, timezone
from uuid import uuid4

import _bootstrap  # noqa: F401
from src.audio_utils import write_json
from src.provenance import file_sha256
from src.voice_library import ROOT, VoiceLibrary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    library = VoiceLibrary()
    report = {"write": args.write, "voices": [], "audio_modified": False}
    backup = ROOT / "outputs/qc/voice_metadata" / uuid4().hex
    for voice in library.list():
        source_path = library.root / voice["id"] / "voice.json"
        original = json.loads(source_path.read_text())
        for style, entry in voice.get("styles", {}).items():
            measured = library.reference_identity(voice["id"], style)
            previous = entry.get("reference_sha256")
            if previous and previous != measured["reference_sha256"]:
                raise ValueError(
                    f"Recorded canonical reference checksum changed: {voice['id']}/{style}"
                )
            entry.update(
                reference_sha256=measured["reference_sha256"],
                reference_stats={**measured["reference_audio"], "path": entry["reference"]},
                reference_qc=measured["reference_qc"],
                transcript_text=measured["reference_transcript"],
                transcript_sha256=measured["reference_transcript_sha256"],
            )
            entry.setdefault(
                "transcript_verification", "registered_text_not_independently_verified"
            )
        for key, value in {
            "provenance": "unknown",
            "source": "unknown",
            "commercial_use": None,
            "created_at": None,
            "model_repo": None,
            "model_revision": None,
            "mlx_audio_version": None,
            "tags": [],
        }.items():
            voice.setdefault(key, value)
        changed = voice != original
        if changed:
            voice["metadata_inspected_at"] = datetime.now(timezone.utc).isoformat()
        if args.write and changed:
            write_json(backup / voice["id"] / "voice.json", original, exclusive=True)
            write_json(source_path, voice)
        report["voices"].append(
            {
                "id": voice["id"],
                "styles": len(voice["styles"]),
                "changed": changed,
                "metadata_sha256": file_sha256(source_path),
            }
        )
    if args.write:
        write_json(backup / "report.json", report, exclusive=True)
        report["backup"] = str(backup)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
