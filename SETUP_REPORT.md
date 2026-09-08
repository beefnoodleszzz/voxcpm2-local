# VoxCPM2 BF16 Setup Report

Generated: 2026-09-07 (Asia/Shanghai)

- Machine: MacBook Pro, Apple M2 Pro (12 cores), arm64
- Unified memory: 16 GB
- macOS: 26.3.1 (25D771280a)
- Disk available before setup: 411 GiB
- Python: 3.11.15 (isolated `.venv`)
- MLX: 0.32.2
- MLX-Audio: 0.5.1
- Model: `mlx-community/VoxCPM2-bf16`
- Local model size: 4,960,144,202 bytes (4.62 GiB)
- Quantization declaration: none
- Output: 48,000 Hz, mono, PCM-24 WAV

## Acceptance results

| Test | Result |
|---|---|
| BF16 30-step smoke | PASS; load 3.07s, generation 23.98s, audio 5.760s |
| Voice Design | PASS |
| Controllable Clone | PASS; 4.320s WAV |
| Ultimate Clone | PASS; combined reference + continuation, 4.000s WAV |
| API `/health` | PASS on `127.0.0.1:7861` |
| API synthesis | PASS; 2.720s WAV |
| Web UI | PASS; HTTP 200 on `127.0.0.1:7862` |
| Offline generation | PASS; printed `OFFLINE TTS VERIFIED` |
| Batch + manifest | PASS; `episode_001/S01_L01.wav` |
| Unit tests | PASS; 2 passed |

## Quality benchmark

| Steps | Generation | Audio duration | Peak | RMS |
|---:|---:|---:|---:|---:|
| 10 | 9.54s | 5.440s | 0.3653 | 0.0471 |
| 20 | 15.36s | 5.440s | 0.4151 | 0.0547 |
| 30 | 22.45s | 5.760s | 0.4408 | 0.0617 |

The 30-step output was stable, so the production default remains 30. Listening selection is subjective; all three files are retained.

## Memory

The smoke generation reported MLX peak memory of 6.13 GB. macOS `/usr/bin/time -l` reported peak memory footprint of 11,697,329,168 bytes. BF16 worked on this 16GB machine without swapping in the observed smoke run. Long inputs and concurrent requests can create greater pressure; the engine serializes generation and never silently changes precision.

## Known warnings

- Transformers 5.16.1 prints a model-type advisory while loading the tokenizer. MLX-Audio's native VoxCPM2 model still loads and all inference modes pass; this is an upstream cosmetic warning.
- FastAPI tests emit upstream Starlette/httpx deprecation warnings; tests pass and runtime is unaffected.
