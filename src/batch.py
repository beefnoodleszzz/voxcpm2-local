"""Content-addressed, checkpointed local batch revisions."""

from __future__ import annotations

import fcntl
import json
from contextlib import contextmanager
from pathlib import Path

from .audio_utils import write_json
from .generation_policy import load_profile, rank_candidates
from .pronunciation import prepare_text
from .provenance import file_sha256, fingerprint, model_identity, runtime_versions


@contextmanager
def project_lock(folder: Path):
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / ".batch.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("This project already has a running batch") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def preflight(req, engine) -> dict:
    voices = {}
    prepared = {}
    for line in req.lines:
        if line.character_id:
            key = f"{line.character_id}/{line.style or '<default>'}"
            voices[key] = engine.voices.reference_identity(line.character_id, line.style)
        prepared[line.id] = line.preparation or prepare_text(
            line.text, req.normalization_profile, req.project_lexicon, req.episode_lexicon
        )
    identity = {
        "input": req.model_dump(),
        "preparations": prepared,
        "voices": voices,
        "model": model_identity(engine.model_path),
        "runtime": runtime_versions(),
        "profile": load_profile(req.profile) if req.profile else None,
        "asr": engine.transcript_verifier().identity,
        "implementation": {
            p.name: file_sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))
        },
    }
    return {"fingerprint": fingerprint(identity), "identity": identity}


def reusable_revision(folder: Path, expected: str) -> dict | None:
    path = folder / "manifest.json"
    if not path.is_file():
        return None
    try:
        manifest = json.loads(path.read_text())
        if manifest.get("fingerprint") != expected:
            return None
        # Also inspect candidates written just before a crash/checkpoint.
        for wav in folder.rglob("*.wav"):
            sidecar = wav.with_suffix(".json")
            if not sidecar.is_file():
                return None
            meta = json.loads(sidecar.read_text())
            if meta.get("audio_sha256") != file_sha256(wav):
                return None
        for line in manifest.get("lines", []):
            for meta in line["candidates"]:
                wav = Path(meta["output"])
                if not wav.is_relative_to(folder) or not wav.is_file():
                    return None
                if meta.get("metadata_sha256") is not None and meta[
                    "metadata_sha256"
                ] != file_sha256(wav.with_suffix(".json")):
                    return None
        return manifest
    except (ValueError, OSError, KeyError, TypeError):
        return None


def run_batch(req, engine, root: Path, *, dry_run: bool = False) -> dict:
    inspected = preflight(req, engine)
    if dry_run:
        return {"success": True, "dry_run": True, **inspected}
    project_dir = root / "outputs/raw" / req.project
    with project_lock(project_dir):
        revisions = sorted(project_dir.glob("run_[0-9][0-9][0-9][0-9]"))
        run, manifest = None, None
        for candidate in reversed(revisions):
            existing = reusable_revision(candidate, inspected["fingerprint"])
            if existing is not None:
                run, manifest = candidate, existing
                break
        if run is None:
            number = max((int(p.name[4:]) for p in revisions), default=0) + 1
            run = project_dir / f"run_{number:04d}"
            run.mkdir(exist_ok=False)
            manifest = {
                "project": req.project,
                "revision": run.name,
                **inspected,
                "status": "running",
                "lines": [],
            }
            write_json(run / "manifest.json", manifest)
        completed = {line["id"]: line for line in manifest["lines"]}
        for line in req.lines:
            if line.id in completed:
                continue
            common = dict(
                text=line.text,
                seed=line.seed if line.seed is not None else req.seed,
                candidates=req.candidates,
                inference_timesteps=req.inference_timesteps,
                cfg_value=req.cfg_value,
                warmup_patches=req.warmup_patches,
                max_tokens=req.max_tokens,
                profile=req.profile,
                resume=True,
                preparation=inspected["identity"]["preparations"][line.id],
                context={"batch_hash": inspected["fingerprint"], "line_id": line.id},
            )
            if line.mode == "design":
                method, extra = "design_voice", {"instruct": line.instruct}
            else:
                method = "clone_voice" if line.mode == "controllable" else "ultimate_clone"
                extra = {"character_id": line.character_id, "style": line.style}
                if line.mode == "controllable":
                    extra["instruct"] = line.instruct
            metas = engine.generate_candidates(method, run / line.id, **common, **extra)
            for meta in metas:
                sidecar = Path(meta["output"]).with_suffix(".json")
                if sidecar.is_file():
                    meta["metadata_sha256"] = file_sha256(sidecar)
            row = {**line.model_dump(), "candidates": metas, "selection": rank_candidates(metas)}
            manifest["lines"].append(row)
            write_json(run / "manifest.json", manifest)
        manifest["status"] = "generated_awaiting_review"
        write_json(run / "manifest.json", manifest)
        return {
            "success": True,
            "manifest": str(run / "manifest.json"),
            "lines": manifest["lines"],
            "revision": run.name,
            "fingerprint": inspected["fingerprint"],
        }
