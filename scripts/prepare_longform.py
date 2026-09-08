import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from src.audio_utils import write_json
from src.longform import segment_text
from src.schemas import BatchRequest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--project", required=True)
    parser.add_argument("--character", required=True)
    parser.add_argument("--max-chars", type=int, default=120)
    args = parser.parse_args()
    segments = segment_text(args.input.read_text(), args.max_chars)
    result = {
        "project": args.project,
        "profile": "production",
        "lines": [
            {
                "id": f"segment_{i:04d}",
                "character_id": args.character,
                "text": text,
                "emotion": "叙述",
            }
            for i, text in enumerate(segments, 1)
        ],
    }
    BatchRequest.model_validate(result)
    write_json(args.output, result, exclusive=True)
    print(json.dumps({"output": str(args.output), "segments": len(segments)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
