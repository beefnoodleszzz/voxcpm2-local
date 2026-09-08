"""Optional local embedding QC, isolated from the TTS engine."""

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from src.audio_utils import write_json
from src.speaker_qc import speaker_similarity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("generated", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--config", type=Path, help="Local command adapter returning an embedding JSON"
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text()) if args.config else None
    result = speaker_similarity(args.reference, args.generated, config)
    write_json(args.output, result, exclusive=True)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
