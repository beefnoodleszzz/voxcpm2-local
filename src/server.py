from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from .engine import VoxCPMEngine
from .schemas import BatchRequest, CloneRequest, DesignRequest, UltimateRequest
from .voice_library import ROOT
from .batch import run_batch
from .errors import GenerationError
from .generation_policy import rank_candidates

app = FastAPI(title="VoxCPM2 Local BF16 API", version="1.0.0")
engine = VoxCPMEngine()


@app.exception_handler(GenerationError)
def generation_error_handler(request, exc):
    return JSONResponse(
        status_code=503,
        content={"success": False, "error_type": type(exc).__name__, "detail": str(exc)},
    )


def quality(req):
    return dict(
        inference_timesteps=req.inference_timesteps,
        cfg_value=req.cfg_value,
        warmup_patches=req.warmup_patches,
        max_tokens=req.max_tokens,
        profile=req.profile,
        normalization_profile=req.normalization_profile,
        project_lexicon=req.project_lexicon,
        episode_lexicon=req.episode_lexicon,
    )


@app.get("/health")
def health():
    return engine.health()


@app.get("/voices")
def voices():
    return {"voices": engine.voices.list()}


def execute(req, method, **kwargs):
    try:
        folder = ROOT / "outputs/raw/api" / uuid4().hex
        metas = engine.generate_candidates(
            method,
            folder,
            candidates=req.candidates,
            seed=req.seed,
            text=req.text,
            **quality(req),
            **kwargs,
        )
        return {
            "success": True,
            "files": [m["output"] for m in metas],
            "metadata": metas,
            "selection": rank_candidates(metas),
        }
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v1/tts/design")
def design(req: DesignRequest):
    return execute(req, "design_voice", instruct=req.instruct)


@app.post("/v1/tts/clone")
def clone(req: CloneRequest):
    return execute(
        req, "clone_voice", character_id=req.character_id, style=req.style, instruct=req.instruct
    )


@app.post("/v1/tts/ultimate")
def ultimate(req: UltimateRequest):
    return execute(req, "ultimate_clone", character_id=req.character_id, style=req.style)


@app.post("/v1/tts/batch")
def batch(req: BatchRequest):
    try:
        return run_batch(req, engine, ROOT)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
