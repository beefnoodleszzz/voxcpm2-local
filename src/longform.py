"""Sentence/clause segmentation and explicit approved-segment assembly."""

from __future__ import annotations

import json
import re
from pathlib import Path
from uuid import uuid4

import numpy as np
import soundfile as sf

from .audio_utils import write_json, write_wav
from .provenance import file_sha256
from .qc import inspect_qc


def segment_text(text: str, max_chars: int = 120) -> list[str]:
    if not 20 <= max_chars <= 500:
        raise ValueError("max_chars must be 20–500")
    if not text.strip():
        raise ValueError("Text cannot be empty")
    sentences = re.findall(
        r"""[^。！？!?\n]+(?:[。！？!?\n]+[”’」』"']*)?|[。！？!?\n]+[”’」』"']*""", text
    )
    segments = []
    for sentence in sentences:
        if len(sentence) <= max_chars:
            segments.append(sentence)
            continue
        clauses = re.findall(r"[^，,；;：:]+[，,；;：:]*|[，,；;：:]+", sentence)
        current = ""
        for clause in clauses:
            if len(clause) > max_chars:
                raise ValueError(
                    "A clause exceeds max_chars without semantic punctuation; add a natural pause"
                )
            if current and len(current) + len(clause) > max_chars:
                segments.append(current)
                current = ""
            current += clause
        if current:
            segments.append(current)
    if "".join(segments) != text:
        raise ValueError("Segmentation failed to preserve the complete source text")
    return segments


def join_approved(paths: list[Path], root: Path, *, pause_seconds: float = 0.0) -> dict:
    if not paths or not 0 <= pause_seconds <= 2:
        raise ValueError("Provide ordered approved segments and 0–2s optional extra pause")
    base = (root / "outputs/approved").resolve()
    arrays, segments = [], []
    for path in paths:
        path = path.resolve()
        if not path.is_relative_to(base):
            raise ValueError("All joined segments must be approved outputs")
        review = json.loads(path.with_suffix(".json").read_text())
        if (
            review.get("decision") != "approve"
            or not review.get("listened")
            or review.get("audio_sha256") != file_sha256(path)
        ):
            raise ValueError("Segment approval/integrity mismatch")
        if inspect_qc(path)["qc"]["status"] == "FAIL":
            raise ValueError("Invalid segment audio")
        data, sr = sf.read(path, dtype="float32")
        if arrays and pause_seconds:
            arrays.append(np.zeros(round(sr * pause_seconds), dtype=np.float32))
        arrays.append(data)
        segments.append(
            {"path": str(path), "sha256": review["audio_sha256"], "review_id": review["review_id"]}
        )
    output = root / "outputs/delivery/longform" / uuid4().hex / "joined.wav"
    write_wav(output, np.concatenate(arrays))
    report = {
        "output": str(output),
        "audio_sha256": file_sha256(output),
        "segments": segments,
        "pause_seconds": pause_seconds,
        "crossfade": False,
        "join_listening_required": True,
        **inspect_qc(output),
    }
    write_json(output.with_suffix(".json"), report, exclusive=True)
    if report["qc"]["status"] == "FAIL":
        raise ValueError(f"Joined audio failed QC; preserved for diagnosis: {output}")
    return report
