"""Bounded candidate planning. No acoustic metric grants human approval."""

from __future__ import annotations

import json
from pathlib import Path

from .errors import ConfigurationError
from .provenance import fingerprint

PROFILE_PATH = Path(__file__).resolve().parents[1] / "configs/profiles.json"


def load_profile(name: str) -> dict:
    profiles = json.loads(PROFILE_PATH.read_text())["profiles"]
    if name not in profiles:
        raise ConfigurationError(f"Unknown generation profile: {name}")
    value = {"name": name, **profiles[name]}
    steps = value["steps"]
    if not steps or len(steps) > 5 or any(type(s) is not int or not 1 <= s <= 100 for s in steps):
        raise ConfigurationError(f"Invalid steps in profile: {name}")
    return value


def plan_attempts(
    *, profile: str | None, candidates: int, seed: int, steps: int
) -> tuple[dict, list[dict]]:
    if not 1 <= candidates <= 5 or not 0 <= seed <= 2**32 - 1 or not 1 <= steps <= 100:
        raise ConfigurationError("Invalid candidate count, seed or steps")
    policy = (
        load_profile(profile)
        if profile
        else {
            "name": "custom",
            "steps": [steps] * candidates,
            "early_stop": False,
            "experimental": False,
        }
    )
    attempts = [
        {"candidate": i + 1, "seed": (seed + i * 3365) % 2**32, "inference_timesteps": step}
        for i, step in enumerate(policy["steps"])
    ]
    policy["sha256"] = fingerprint(policy)
    return policy, attempts


def accepted_automatically(meta: dict) -> bool:
    # Missing/unconfigured/un-calibrated ASR can never trigger early success.
    return (
        meta.get("qc", {}).get("status") == "PASS"
        and meta.get("transcript_qc", {}).get("status") == "PASS"
    )


def retry_reason(meta: dict) -> dict:
    if meta.get("transcript_qc", {}).get("status") == "FAIL":
        return {
            "code": "transcript_mismatch",
            "action": "review_normalization_or_lexicon",
            "automatic_text_rewrite": False,
        }
    if meta.get("qc", {}).get("status") == "FAIL":
        return {"code": "audio_failure", "action": "next_seed_and_scheduled_steps"}
    return {"code": "review_required", "action": "retain_candidates_for_listening"}


def rank_candidates(metas: list[dict]) -> dict:
    rows = []
    for index, meta in enumerate(metas, 1):
        qc = meta.get("qc", {}).get("status", "UNAVAILABLE")
        transcript = meta.get("transcript_qc", {})
        cer = transcript.get("cer")
        # Transparent ordering rather than fabricated perceptual similarity/naturalness scores.
        key = (
            qc != "FAIL",
            transcript.get("status") != "FAIL",
            qc == "PASS",
            -(cer if cer is not None else 1.0),
        )
        rows.append(
            {
                "candidate": meta.get("generation_context", {}).get("candidate_number", index),
                "output": meta["output"],
                "audio_qc": qc,
                "cer": cer,
                "speaker_score": meta.get("speaker_similarity"),
                "sort_key": list(key),
            }
        )
    eligible = [row for row in rows if row["audio_qc"] in {"PASS", "WARN"}]
    winner = max(eligible, key=lambda row: row["sort_key"]) if eligible else None
    return {
        "recommended_candidate": winner["candidate"] if winner else None,
        "recommended_output": winner["output"] if winner else None,
        "human_review_required": True,
        "candidates": rows,
    }
