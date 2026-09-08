"""Local model identity and deterministic content fingerprints; no network calls."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from functools import lru_cache

from .errors import ConfigurationError

ROOT = Path(__file__).resolve().parents[1]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(payload) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def runtime_versions() -> dict:
    return {
        "python_version": platform.python_version(),
        "machine": platform.machine(),
        "os": platform.platform(),
        "mlx_version": package_version("mlx"),
        "mlx_audio_version": package_version("mlx-audio"),
    }


def read_model_lock(path: Path | None = None) -> dict:
    lock = json.loads((path or ROOT / "configs/model.lock.json").read_text())
    if (
        lock.get("repo") != "mlx-community/VoxCPM2-bf16"
        or lock.get("format") != "bf16"
        or not re.fullmatch(r"[a-f0-9]{40}", lock.get("revision", ""))
    ):
        raise ConfigurationError("Invalid BF16 model lock")
    return lock


def model_path() -> Path:
    configured = Path(os.getenv("VOXCPM_MODEL_PATH", "models/VoxCPM2-bf16"))
    return (configured if configured.is_absolute() else ROOT / configured).resolve()


def model_identity(path: Path, *, verify: bool = True) -> dict:
    lock = read_model_lock()
    hashes = {}
    if verify:
        config = json.loads((path / "config.json").read_text())
        if config.get("quantization") or config.get("quantization_config"):
            raise ConfigurationError("Production model must remain BF16")
        index = json.loads((path / "model.safetensors.index.json").read_text())
        required = [
            "config.json",
            "tokenizer.json",
            "model.safetensors.index.json",
            *sorted(set(index["weight_map"].values())),
        ]
        for name in required:
            target = (path / name).resolve()
            if not target.is_relative_to(path.resolve()) or not target.is_file():
                raise ConfigurationError(f"Missing or unsafe model file: {name}")
            receipt = path / ".cache/huggingface/download" / f"{name}.metadata"
            receipt_lines = receipt.read_text().splitlines() if receipt.is_file() else []
            if len(receipt_lines) < 2 or receipt_lines[0] != lock["revision"]:
                raise ConfigurationError(
                    f"Model revision not verified for {name}; run download_model.py"
                )
            stat = target.stat()
            hashes[name] = verified_artifact(
                str(target), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, receipt_lines[1]
            )
        if package_version("mlx-audio") != lock["mlx_audio_version"]:
            raise ConfigurationError("Production requires mlx-audio==0.5.1")
    return {
        "model_repo": lock["repo"],
        "model_revision": lock["revision"],
        "model_format": "BF16",
        "model_lock_sha256": fingerprint(lock),
        "model_artifact_hashes": hashes,
    }


@lru_cache(maxsize=64)
def verified_artifact(path: str, size: int, mtime_ns: int, ctime_ns: int, etag: str) -> str:
    """Check HF LFS sha256 or git-blob sha1; cache only while file identity is unchanged."""
    del mtime_ns, ctime_ns
    sha256 = hashlib.sha256()
    blob = hashlib.sha1(f"blob {size}\0".encode())
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha256.update(block)
            blob.update(block)
    observed = sha256.hexdigest() if len(etag) == 64 else blob.hexdigest()
    if observed != etag:
        raise ConfigurationError(f"Model artifact checksum mismatch: {Path(path).name}")
    return sha256.hexdigest()
