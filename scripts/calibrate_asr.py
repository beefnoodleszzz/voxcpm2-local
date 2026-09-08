"""Build a listening sheet, then calibrate ASR against human-transcribed speech.

Actual human transcript is separate from intended TTS text. This measures ASR's
own errors on correct speech instead of labelling TTS mistakes as recognition noise.
"""

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from src.audio_utils import write_json
from src.provenance import file_sha256, fingerprint
from src.transcript_qc import align_transcript


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--rationale", default="")
    args = parser.parse_args()
    payload = json.loads(args.input.read_text())
    if args.threshold is None:
        rows = [
            {
                "audio": row["output"],
                "audio_sha256": row["audio_sha256"],
                "expected": row["text"],
                "recognized": row["transcript_qc"].get("recognized"),
                "backend_id": row["transcript_qc"].get("backend_id"),
                "human_transcript": None,
                "listened": False,
                "reviewer": None,
                "tts_correct": None,
            }
            for row in payload["rows"]
        ]
        write_json(args.output, {"samples": rows}, exclusive=True)
    else:
        if not 0 <= args.threshold <= 1 or not args.rationale.strip():
            parser.error("Provide a finite 0–1 threshold and evidence-based --rationale")
        samples = payload["samples"]
        checked, seen, backends = [], set(), set()
        for sample in samples:
            if not (
                sample.get("listened")
                and sample.get("reviewer")
                and sample.get("human_transcript")
                and sample.get("tts_correct") is True
                and isinstance(sample.get("recognized"), str)
            ):
                continue
            if file_sha256(Path(sample["audio"])) != sample["audio_sha256"]:
                raise ValueError("Calibration audio has changed")
            if sample["audio_sha256"] in seen:
                continue
            seen.add(sample["audio_sha256"])
            backends.add(sample["backend_id"])
            checked.append(
                align_transcript(sample["human_transcript"], sample["recognized"])["cer"]
            )
        if len(checked) < 30 or len(backends) != 1:
            parser.error(
                "Need >=30 distinct human-reviewed correct-speech samples from one ASR backend"
            )
        result = {
            "backend_id": next(iter(backends)),
            "threshold": args.threshold,
            "reviewed_samples": len(checked),
            "human_reviewed": True,
            "dataset_sha256": fingerprint(payload),
            "rationale": args.rationale,
            "asr_cer_values": sorted(checked),
            "false_retry_rate_on_correct_speech": sum(c > args.threshold for c in checked)
            / len(checked),
        }
        write_json(args.output, result, exclusive=True)
    print(args.output)


if __name__ == "__main__":
    main()
