"""List batch candidates or record an explicit listening decision."""

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from src.review import record_review
from src.voice_library import ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("project", nargs="?")
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--decision", choices=["approve", "reject", "rate"])
    parser.add_argument("--reviewer", default="")
    parser.add_argument("--listened", action="store_true")
    parser.add_argument("--notes", default="")
    args = parser.parse_args()
    if args.candidate:
        result = record_review(
            args.candidate,
            ROOT,
            decision=args.decision,
            reviewer=args.reviewer,
            listened=args.listened,
            notes=args.notes,
        )
    else:
        if (
            not args.project
            or Path(args.project).name != args.project
            or args.project in {".", ".."}
        ):
            parser.error("Provide a safe project ID or --candidate with a decision")
        result = []
        for path in sorted((ROOT / "outputs/raw" / args.project).glob("run_*/manifest.json")):
            manifest = json.loads(path.read_text())
            result.append(
                {
                    "manifest": str(path),
                    "status": manifest["status"],
                    "lines": [
                        {"id": line["id"], "selection": line["selection"]}
                        for line in manifest["lines"]
                    ],
                }
            )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
