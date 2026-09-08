#!/usr/bin/env python3
import argparse
import json
import _bootstrap  # noqa: F401
from src.schemas import BatchRequest
from pathlib import Path
from src.batch import run_batch
from src.engine import VoxCPMEngine
from src.voice_library import ROOT


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    response = run_batch(
        BatchRequest.model_validate_json(Path(args.input).read_text(encoding="utf-8")),
        VoxCPMEngine(),
        ROOT,
        dry_run=args.dry_run,
    )
    print(json.dumps(response, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
