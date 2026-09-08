"""Fixed text/reference/seed quality A/B with immutable audio and review fields."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import _bootstrap  # noqa: F401
from prepare_episode import INSTRUCTIONS
from src.audio_utils import write_json
from src.engine import VoxCPMEngine
from src.pronunciation import prepare_text
from src.provenance import fingerprint
from src.voice_library import ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--voice", default="male_young_cold_01")
    parser.add_argument("--suite", type=Path, default=ROOT / "benchmark_suite/cases.json")
    parser.add_argument("--steps", type=int, nargs="+", default=[10, 20, 30])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--only", nargs="+")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    suite = json.loads(args.suite.read_text())
    cases = [case for case in suite["cases"] if not args.only or case["id"] in args.only]
    engine = VoxCPMEngine()
    reference = engine.voices.reference_identity(args.voice, "neutral")
    if args.dry_run:
        print(
            json.dumps(
                {"cases": cases, "steps": args.steps, "reference": reference}, ensure_ascii=False
            )
        )
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    root = ROOT / "outputs/raw/benchmark_suite" / stamp
    root.mkdir(parents=True, exist_ok=False)
    report = {
        "suite_sha256": fingerprint(suite),
        "reference": reference,
        "rows": [],
        "human_review_required": True,
        "production_default_changed": False,
    }
    write_json(root / "benchmark.json", report)
    for case in cases:
        preparation = prepare_text(case["text"], "dialogue")
        for steps in args.steps:
            meta = engine.clone_voice(
                args.voice,
                case["text"],
                root / case["id"] / f"steps_{steps}_seed_{args.seed}.wav",
                style="neutral",
                instruct=INSTRUCTIONS[case["emotion"]],
                seed=args.seed,
                inference_timesteps=steps,
                preparation=preparation,
            )
            row = {
                "case": case["id"],
                "text": case["text"],
                "voice": args.voice,
                "style": case["emotion"],
                "seed": args.seed,
                "steps": steps,
                **{
                    k: meta[k]
                    for k in (
                        "generation_seconds",
                        "audio_duration",
                        "rtf",
                        "peak",
                        "rms",
                        "output",
                        "audio_sha256",
                        "mlx_peak_memory_gb",
                        "model_revision",
                        "reference_sha256",
                    )
                },
                "cer": meta["transcript_qc"]["cer"],
                "transcript_qc": meta["transcript_qc"],
                "speaker_similarity": None,
                "qc_pass": meta["qc"]["status"] == "PASS",
                "qc": meta["qc"],
                "human_naturalness": None,
                "human_emotion": None,
                "human_identity": None,
                "human_pronunciation": None,
                "human_prosody": None,
                "human_artifacts": None,
            }
            report["rows"].append(row)
            write_json(root / "benchmark.json", report)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    print(f"Benchmark: {root / 'benchmark.json'}", flush=True)


if __name__ == "__main__":
    main()
