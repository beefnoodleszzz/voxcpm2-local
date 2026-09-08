# VoxCPM2 Local — Apple Silicon BF16

Production-oriented local Chinese dubbing on MLX-Audio with `mlx-community/VoxCPM2-bf16`. Raw masters are 48kHz, mono, PCM-24 WAV and are never overwritten. The model is loaded once per process.

The workstation adds traceable preparation, bounded generation policies, offline ASR, audio QC,
content-based resume, candidate review and separate approved/delivery masters. The `production`
profile uses an approved **10 → 20 → 30-step adaptive schedule**. Normally it generates one
10-step take and stops when audio QC passes and calibrated ASR CER is at most 0.12. It generates
the second or third take only when a gate fails. The recommended take still needs one human listen.

## Workstation workflow

```bash
uv sync --locked
.venv/bin/python scripts/download_model.py
.venv/bin/python scripts/download_asr.py  # optional, pinned local ASR weights (~936 MB)
.venv/bin/python scripts/check_system.py
.venv/bin/python scripts/prepare_episode.py configs/episode.source.example.json episode.prepared.json
.venv/bin/python scripts/batch_dub.py episode.prepared.json --dry-run
.venv/bin/python scripts/batch_dub.py episode.prepared.json
.venv/bin/python scripts/review_project.py episode_workstation_example
```

Repeat the batch command to resume matching WAV/JSON pairs. Changed inputs, reference, voice
metadata, model, code or configuration create a new `outputs/raw/<project>/run_NNNN/` revision.
Old or corrupt outputs remain intact and are not reused without sufficient provenance.

Use `./scripts/start_webui.sh` for reference listening, emotion direction, candidate A/B,
QC/ASR inspection, ratings and approval. Existing batch/benchmark raw WAVs can also be loaded.
Regeneration creates a new directory. Approval requires a named reviewer, listening confirmation
and notes when QC/ASR remains unresolved. Alternatively, after listening:

```bash
.venv/bin/python scripts/review_project.py --candidate outputs/raw/PROJECT/run_0001/LINE/candidate_01.wav \
  --decision approve --reviewer YOUR_NAME --listened --notes "核对了台词、角色和表演"
.venv/bin/python scripts/finalize_project.py PROJECT
```

Finalization accepts a project ID (exports every approved revision) or an exact approved WAV.
Default delivery is a byte-identical copy. Optional `--trim`, `--peak-dbfs -3` and
`--target-lufs <delivery-spec>` produce a separate processed delivery. Raw and approved stay intact.

See [ARCHITECTURE.md](ARCHITECTURE.md) for data flow, [QUALITY.md](QUALITY.md) for profiles,
ASR calibration and benchmark listening, [VOICE_LIBRARY.md](VOICE_LIBRARY.md) for references,
and [AUDIT_REPORT.md](AUDIT_REPORT.md) for the initial audit.

## Regression and real-model checks

```bash
ruff check src scripts tests
ruff format --check src scripts tests
.venv/bin/python -m pytest -q  # no real model required
.venv/bin/python scripts/benchmark_quality.py --dry-run
.venv/bin/python scripts/benchmark_quality.py  # 10 categories × 10/20/30 steps
.venv/bin/python scripts/benchmark_quality.py --suite benchmark_suite/consistency.json --steps 30
```

Ruff 0.11.11 was used for validation. Keep mlx-audio==0.5.1 and the committed uv lock.
The 30-take A/B was human-approved on 2026-09-08 with no obvious audible difference among step
counts. Measurements and the production decision are recorded in `benchmark_suite/decision.json`.
Only one process may own the BF16 model; stop a loaded API/WebUI before generating through a
separate CLI process. All services bind to loopback.

## Long-form workflow

```bash
.venv/bin/python scripts/prepare_longform.py story.txt story.source.json --project story_001 --character narrator_warm_01
.venv/bin/python scripts/prepare_episode.py story.source.json story.prepared.json
.venv/bin/python scripts/batch_dub.py story.prepared.json
# After approving each segment, pass approved WAVs in narrative order:
.venv/bin/python scripts/join_approved.py APPROVED_SEGMENT_1.wav APPROVED_SEGMENT_2.wav
```

Segmentation uses sentence/clause punctuation; an overlong unpunctuated clause requires a natural
pause. Joining preserves breaths and optionally adds `--pause-seconds`; listen to the assembly.

## Install / download

```bash
uv sync
.venv/bin/python scripts/download_model.py
```

No 8-bit or 4-bit repository is referenced by the downloader or runtime.

## Run

```bash
./scripts/start_api.sh       # http://127.0.0.1:7861/docs
./scripts/start_webui.sh     # http://127.0.0.1:7862
```

CLI examples:

```bash
.venv/bin/python scripts/dub.py --mode design --instruct "年轻中国男性，自然影视对白" --text "从今天开始，我不会再替任何人解释。" --candidates 3
.venv/bin/python scripts/dub.py --character male_lead_01 --style angry --mode ultimate --text "我最后问你一次。" --candidates 3
.venv/bin/python scripts/batch_dub.py episode.json
```

Generate/resume the 24-role audition library (72 BF16 takes):

```bash
.venv/bin/python scripts/generate_voice_cast.py
open outputs/raw/voice_cast_v1/index.html
```

After listening, enter candidate numbers in `configs/voice_cast_selections.json`, then promote them into the canonical library:

```bash
.venv/bin/python scripts/finalize_voice_cast.py
```

Core-character emotion auditions and promotion:

```bash
.venv/bin/python scripts/generate_emotion_cast.py
open outputs/raw/emotion_cast_v1/index.html
# Fill configs/emotion_cast_selections.json, then:
.venv/bin/python scripts/finalize_emotion_cast.py
```

See `VOICE_LIBRARY.md` for the production cast and `configs/episode.production.example.json` for an episode template.

## API

`GET /health`, `GET /voices`, `POST /v1/tts/design`, `/v1/tts/clone`, `/v1/tts/ultimate`, `/v1/tts/batch`. Generated file paths and metadata are returned; audio is not base64-encoded.

## Quality / offline checks

```bash
.venv/bin/python scripts/smoke_test.py
.venv/bin/python scripts/benchmark_quality.py
./scripts/verify_offline.sh
```

MLX-Audio 0.5.1 implements Ultimate mode as combined reference cloning and continuation. The exact `prompt_text` belongs to `prompt_audio`; `ref_text` is only part of the common CLI interface and VoxCPM2 conditions directly on `ref_audio`.

## Natural-dialogue production profile

For ordinary drama dialogue, the CLI and Web UI default to `controllable` mode with the
character's neutral reference and a concise acting instruction. This preserves identity while
avoiding unnecessary continuation of a short reference clip's exact rhythm. Choose `ultimate`
only when deliberately reproducing the reference performance.

Run the production checks and prepare human-readable emotion labels before a batch:

```bash
.venv/bin/python scripts/check_system.py
.venv/bin/python scripts/prepare_episode.py episode.json prepared.json
.venv/bin/python scripts/batch_dub.py prepared.json
.venv/bin/python scripts/audit_outputs.py outputs/raw/<project>
```

`prepare_episode.py --exact-performance` opts into registered emotion references and Ultimate
mode. Canonical references should be clean mono 48kHz speech, at least 5 seconds and preferably
8–15 seconds. Existing raw WAV files and canonical references are never overwritten.

On a 16GB Mac, BF16 may encounter memory pressure at model load or during long/30-step generations. This project deliberately fails visibly and never switches to a quantized model.
