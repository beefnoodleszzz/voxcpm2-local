import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from src.longform import join_approved
from src.voice_library import ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("segments", type=Path, nargs="+", help="Approved WAVs in narrative order")
    parser.add_argument("--pause-seconds", type=float, default=0)
    args = parser.parse_args()
    print(
        json.dumps(
            join_approved(args.segments, ROOT, pause_seconds=args.pause_seconds),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
