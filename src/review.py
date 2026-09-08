"""Explicit human review and immutable promotion of locally generated masters."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .audio_utils import write_json
from .provenance import file_sha256
from .qc import inspect_qc


def load_candidate(path: Path, root: Path) -> dict:
    path = path.resolve()
    raw = (root / "outputs/raw").resolve()
    if not path.is_relative_to(raw) or not path.is_file() or path.suffix != ".wav":
        raise ValueError("Select an existing WAV within outputs/raw")
    meta = json.loads(path.with_suffix(".json").read_text())
    if meta.get("audio_sha256") != file_sha256(path):
        raise ValueError("Candidate audio checksum is missing or changed")
    if Path(meta.get("output", "")).resolve() != path:
        raise ValueError("Candidate metadata output does not match WAV")
    return meta


def record_review(
    path: Path,
    root: Path,
    *,
    decision: str,
    reviewer: str,
    listened: bool,
    notes: str = "",
    ratings: dict | None = None,
) -> dict:
    if decision not in {"approve", "reject", "rate"}:
        raise ValueError("Invalid review decision")
    if not reviewer.strip() or not listened:
        raise ValueError("A named reviewer must confirm listening before recording a review")
    ratings = ratings or {}
    if any(type(v) not in (int, float) or not 1 <= v <= 5 for v in ratings.values()):
        raise ValueError("Ratings must be 1–5")
    path = path.resolve()
    meta = load_candidate(path, root)
    qc = inspect_qc(path, meta.get("text_spoken", meta.get("text", "")))["qc"]
    if decision == "approve" and qc["status"] == "FAIL":
        raise ValueError(f"Cannot approve invalid audio: {qc['failures']}")
    if (
        decision == "approve"
        and (qc["status"] == "WARN" or meta.get("transcript_qc", {}).get("status") != "PASS")
        and not notes.strip()
    ):
        raise ValueError("Document your listening decision when QC/ASR is not PASS")
    review_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid4().hex[:8]
    relative = path.relative_to((root / "outputs/raw").resolve())
    review = {
        "review_id": review_id,
        "decision": decision,
        "reviewer": reviewer.strip(),
        "listened": True,
        "notes": notes,
        "ratings": ratings,
        "raw_output": str(path),
        "audio_sha256": meta["audio_sha256"],
        "metadata_sha256": file_sha256(path.with_suffix(".json")),
        "qc_at_review": qc,
        "source_metadata": meta,
    }
    if decision == "approve":
        folder = root / "outputs/approved" / relative.parent / review_id
        folder.mkdir(parents=True, exist_ok=False)
        approved = folder / path.name
        with path.open("rb") as source, approved.open("xb") as target:
            shutil.copyfileobj(source, target)
        review["approved_output"] = str(approved.resolve())
        write_json(approved.with_suffix(".json"), review, exclusive=True)
    review_path = root / "outputs/qc/reviews" / relative.parent / f"{review_id}.json"
    write_json(review_path, review, exclusive=True)
    return {**review, "review_path": str(review_path)}
