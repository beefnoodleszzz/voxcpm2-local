# VoxCPM2 Local — Apple Silicon BF16

Production-oriented local Chinese dubbing on MLX-Audio with `mlx-community/VoxCPM2-bf16`. Raw masters are 48kHz, mono, PCM-24 WAV and are never overwritten. The model is loaded once per process.

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
