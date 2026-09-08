"""Optional independent speaker-embedding adapter; never loads a model by default."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

from .provenance import file_sha256, fingerprint


def cosine_similarity(reference, generated) -> float:
    a, b = np.asarray(reference, dtype=float), np.asarray(generated, dtype=float)
    if (
        a.ndim != 1
        or b.shape != a.shape
        or not a.size
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
    ):
        raise ValueError("Speaker embeddings must be finite matching nonempty vectors")
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if not norm:
        raise ValueError("Zero-norm speaker embedding")
    return float(np.clip(np.dot(a, b) / norm, -1, 1))


def speaker_similarity(reference: Path, generated: Path, config: dict | None = None) -> dict:
    config = config or {}
    if not config.get("command"):
        return {
            "status": "UNAVAILABLE",
            "speaker_similarity": None,
            "reason": "No independently validated local embedding backend configured",
        }
    command = config["command"]
    if not isinstance(command, list) or not all(isinstance(part, str) for part in command):
        raise ValueError("Speaker command must be an argv array")
    embeddings = []
    for path in (reference, generated):
        result = subprocess.run(
            [*command, str(path.resolve())],
            check=True,
            capture_output=True,
            text=True,
            timeout=config.get("timeout_seconds", 120),
        )
        embeddings.append(json.loads(result.stdout)["embedding"])
    return {
        "status": "WARN",
        "speaker_similarity": cosine_similarity(*embeddings),
        "reference_sha256": file_sha256(reference),
        "generated_sha256": file_sha256(generated),
        "backend_id": config.get("backend_id"),
        "config_sha256": fingerprint(config),
        "note": "Similarity is advisory until the embedding backend is calibrated across character emotions",
    }
