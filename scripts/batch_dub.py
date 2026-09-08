#!/usr/bin/env python3
import argparse, json
import _bootstrap  # noqa: F401
from src.schemas import BatchRequest
from src.server import batch

def main():
    p = argparse.ArgumentParser(); p.add_argument("input"); args = p.parse_args()
    response = batch(BatchRequest.model_validate_json(open(args.input, encoding="utf-8").read()))
    print(json.dumps(response, ensure_ascii=False, indent=2))
if __name__ == "__main__": main()
