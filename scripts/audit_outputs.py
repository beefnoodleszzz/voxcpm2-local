#!/usr/bin/env python3
"""Validate raw WAV/JSON pairs and report objective delivery risks."""

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from src.audio_utils import inspect_audio, write_json
from src.qc import audio_qc
from src.provenance import file_sha256, fingerprint
from src.voice_library import ROOT
from uuid import uuid4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    args = parser.parse_args()
    root = Path(args.output_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    errors, warnings, entries = [], [], []
    for wav in sorted(root.rglob("*.wav")):
        meta_path = wav.with_suffix(".json")
        try:
            stats = inspect_audio(wav)
            qc = audio_qc(stats)
            if (
                stats["sample_rate"] != 48000
                or stats["channels"] != 1
                or stats["has_nan"]
                or stats["has_inf"]
                or stats["clipping"]
                or qc["status"] == "FAIL"
            ):
                errors.append(f"{wav}: invalid audio {stats}")
            if stats["peak"] > 0.98:
                warnings.append(f"{wav}: peak {stats['peak']:.3f} is close to full scale")
            if not meta_path.is_file():
                errors.append(f"{wav}: missing metadata JSON")
                meta = {}
            else:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if Path(meta.get("output", "")).resolve() != wav:
                    errors.append(f"{meta_path}: output path does not match WAV")
                if meta.get("audio_sha256") and meta["audio_sha256"] != file_sha256(wav):
                    errors.append(f"{wav}: checksum mismatch")
                if not meta.get("model_revision"):
                    warnings.append(f"{wav}: legacy metadata lacks model revision")
            entries.append(
                {
                    "wav": str(wav),
                    "duration": stats["duration"],
                    "peak": stats["peak"],
                    "rms": stats["rms"],
                    "mode": meta.get("mode"),
                    "text": meta.get("text"),
                    "qc": qc,
                    "transcript_qc": meta.get("transcript_qc", {"status": "UNAVAILABLE"}),
                }
            )
        except (OSError, ValueError, KeyError) as exc:
            errors.append(f"{wav}: {exc}")
    if not entries:
        errors.append(f"no WAV files found under {root}")
    report = {
        "output_dir": str(root),
        "files": len(entries),
        "errors": errors,
        "warnings": warnings,
        "entries": entries,
        "note": "Objective checks do not prove acting quality or transcript correctness.",
    }
    report_path = ROOT / "outputs/qc/audits" / fingerprint(str(root))[:16] / f"{uuid4().hex}.json"
    write_json(report_path, report, exclusive=True)
    print(
        json.dumps(
            {
                "files": len(entries),
                "errors": errors,
                "warning_count": len(warnings),
                "report": str(report_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
