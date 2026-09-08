"""Deterministic character alignment and optional offline command-based ASR.

The ASR command receives only a WAV path, never the expected transcript. Its stdout
must be JSON {"text": "..."}. A bounded subprocess releases ASR memory on completion.
"""

from __future__ import annotations

import json
import math
import subprocess
import unicodedata
from pathlib import Path

from .errors import ConfigurationError, TranscriptQCError
from .provenance import ROOT, file_sha256, fingerprint
from .text_normalizer import normalize_text


def comparison_text(text: str) -> str:
    if not text.strip():
        return ""
    text = normalize_text(text, "narration")["normalized"].casefold()
    return "".join(
        c
        for c in unicodedata.normalize("NFKC", text)
        if not c.isspace() and unicodedata.category(c)[0] not in "PS"
    )


def align_transcript(expected: str, recognized: str) -> dict:
    a, b = comparison_text(expected), comparison_text(recognized)
    if not a:
        raise ConfigurationError("Expected transcript has no comparable characters")
    if max(len(a), len(b)) > 5000:
        raise TranscriptQCError("Transcript exceeds 5000-character alignment limit; segment first")
    # Full matrix preserves exact edit positions with deterministic tie breaking.
    matrix = [list(range(len(b) + 1))]
    for i, ca in enumerate(a, 1):
        row = [i]
        for j, cb in enumerate(b, 1):
            row.append(min(matrix[i - 1][j] + 1, row[j - 1] + 1, matrix[i - 1][j - 1] + (ca != cb)))
        matrix.append(row)
    substitutions, deletions, insertions = [], [], []
    i, j = len(a), len(b)
    while i or j:
        if i and j and matrix[i][j] == matrix[i - 1][j - 1] + (a[i - 1] != b[j - 1]):
            if a[i - 1] != b[j - 1]:
                substitutions.append(
                    {
                        "expected_index": i - 1,
                        "recognized_index": j - 1,
                        "expected": a[i - 1],
                        "recognized": b[j - 1],
                    }
                )
            i, j = i - 1, j - 1
        elif i and matrix[i][j] == matrix[i - 1][j] + 1:
            deletions.append({"expected_index": i - 1, "expected": a[i - 1]})
            i -= 1
        else:
            insertions.append(
                {
                    "recognized_index": j - 1,
                    "recognized": b[j - 1],
                    "possible_repetition": j > 1 and b[j - 1] == b[j - 2],
                }
            )
            j -= 1
    return {
        "expected": expected,
        "recognized": recognized,
        "expected_comparison": a,
        "recognized_comparison": b,
        "cer": matrix[-1][-1] / len(a),
        "substitutions": substitutions[::-1],
        "deletions": deletions[::-1],
        "insertions": insertions[::-1],
        "edit_distance": matrix[-1][-1],
    }


def calibration_threshold(calibration: dict | None, backend_id: str) -> float | None:
    if calibration is None:
        return None
    threshold = calibration.get("threshold")
    if (
        calibration.get("backend_id") != backend_id
        or calibration.get("reviewed_samples", 0) < 30
        or not calibration.get("human_reviewed")
        or not calibration.get("dataset_sha256")
        or type(threshold) not in (int, float)
        or not math.isfinite(threshold)
        or not 0 <= threshold <= 1
    ):
        raise ConfigurationError(
            "ASR calibration needs matching backend, >=30 reviewed samples and an evidenced threshold"
        )
    return float(threshold)


class TranscriptVerifier:
    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.backend_id = self.config.get("backend_id", "unconfigured")
        self.threshold = calibration_threshold(self.config.get("calibration"), self.backend_id)
        command = self.config.get("command")
        if command is not None and (
            not isinstance(command, list)
            or not command
            or not all(isinstance(p, str) and p for p in command)
        ):
            raise ConfigurationError(
                "ASR command must be an argv array; shell strings are not accepted"
            )

    @property
    def identity(self) -> dict:
        return {"backend_id": self.backend_id, "config_sha256": fingerprint(self.config)}

    def verify(self, path: Path, expected: str) -> dict:
        command = self.config.get("command")
        if not command:
            return {
                "status": "UNAVAILABLE",
                "pass": None,
                "cer": None,
                "reason": "No local ASR backend configured",
                **self.identity,
            }
        try:
            result = subprocess.run(
                [*command, str(path.resolve())],
                check=True,
                capture_output=True,
                text=True,
                timeout=self.config.get("timeout_seconds", 180),
                cwd=ROOT,
            )
            payload = json.loads(result.stdout)
            recognized = payload["text"]
            if not isinstance(recognized, str):
                raise ValueError("ASR text must be a string")
            alignment = align_transcript(expected, recognized)
        except (OSError, subprocess.SubprocessError, ValueError, KeyError) as exc:
            raise TranscriptQCError(f"Local ASR failed ({self.backend_id}): {exc}") from exc
        passed = alignment["cer"] <= self.threshold if self.threshold is not None else None
        return {
            **alignment,
            **self.identity,
            "threshold": self.threshold,
            "audio_sha256": file_sha256(path),
            "pass": passed,
            "status": "WARN" if passed is None else ("PASS" if passed else "FAIL"),
            "note": "Uncalibrated ASR is advisory; edit differences are not proof of mispronunciation",
        }
