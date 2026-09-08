from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException

from .engine import VoxCPMEngine
from .schemas import BatchRequest, CloneRequest, DesignRequest, UltimateRequest
from .voice_library import ROOT

app = FastAPI(title="VoxCPM2 Local BF16 API", version="1.0.0")
engine = VoxCPMEngine()


def quality(req):
    return dict(inference_timesteps=req.inference_timesteps, cfg_value=req.cfg_value,
                warmup_patches=req.warmup_patches, max_tokens=req.max_tokens)


@app.get("/health")
def health(): return engine.health()


@app.get("/voices")
def voices(): return {"voices": engine.voices.list()}


def execute(req, method, **kwargs):
    try:
        folder = ROOT / "outputs/raw/api" / uuid4().hex
        metas = engine.generate_candidates(method, folder, candidates=req.candidates,
                                           seed=req.seed, text=req.text, **quality(req), **kwargs)
        return {"success": True, "files": [m["output"] for m in metas], "metadata": metas}
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v1/tts/design")
def design(req: DesignRequest):
    return execute(req, "design_voice", instruct=req.instruct)


@app.post("/v1/tts/clone")
def clone(req: CloneRequest):
    return execute(req, "clone_voice", character_id=req.character_id, style=req.style,
                   instruct=req.instruct)


@app.post("/v1/tts/ultimate")
def ultimate(req: UltimateRequest):
    return execute(req, "ultimate_clone", character_id=req.character_id, style=req.style)


@app.post("/v1/tts/batch")
def batch(req: BatchRequest):
    try:
        project_dir = ROOT / "outputs/raw" / req.project
        manifest = {"project": req.project, "lines": []}
        for line in req.lines:
            seed = line.seed if line.seed is not None else req.seed
            common = dict(text=line.text, seed=seed, candidates=req.candidates, **quality(req))
            line_dir = project_dir / line.id
            if line.mode == "design":
                if not line.instruct: raise ValueError(f"{line.id}: design needs instruct")
                metas = engine.generate_candidates("design_voice", line_dir, instruct=line.instruct, **common)
            elif line.mode == "controllable":
                metas = engine.generate_candidates("clone_voice", line_dir, character_id=line.character_id,
                                                   style=line.style, instruct=line.instruct, **common)
            else:
                metas = engine.generate_candidates("ultimate_clone", line_dir, character_id=line.character_id,
                                                   style=line.style, **common)
            manifest["lines"].append({"id": line.id, "character_id": line.character_id,
                                      "style": line.style, "text": line.text, "mode": line.mode,
                                      "candidates": metas})
        from .audio_utils import write_json
        write_json(project_dir / "manifest.json", manifest)
        return {"success": True, "manifest": str(project_dir / "manifest.json"), "lines": manifest["lines"]}
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
