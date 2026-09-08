# Workstation audit — 2026-09-08

Scope: the Desktop “VoxCPM2 Local 极致优化 Master Prompt”. Reviewed all runtime modules,
scripts, schemas, tests, configuration, startup scripts and production documentation before
implementation. Durable implementation/acceptance status is tracked in Beads `voxcpm2-local-7bg`.

## Baseline evidence

- Native Apple Silicon environment: Python 3.11.15, mlx-audio 0.5.1, local BF16 model.
- `scripts/check_system.py`: passes; 25 voices; 51 references shorter than 5 seconds.
- `python -m pytest -q`: 3 passed, two dependency deprecation warnings.
- No baseline inference or subjective quality comparison has yet been performed.
- Current model download metadata records revision `a7fb80d9056983135a54ca3f71fcc5a637ccddf5`.

## Findings, risk and benefit

| Finding | Risk | Improvement |
| --- | --- | --- |
| Personal absolute path in .env.example; incomplete audio/secret ignores | Local paths/material accidentally shared | Portable defaults and broader local-material ignores |
| Downloader follows default branch | Model changes invisibly | Lock existing revision and validate local provenance |
| Fixed steps/seeds; no acceptance gate | Failed takes appear successful | Bounded policy, explicit QC states, retained candidates |
| Only peak/RMS/silence measured; PCM conversion may mask invalid peaks | Clipping/silence accepted | Validate before encoding; PASS/WARN/FAIL checks |
| No normalization, lexicon or transcript alignment | Number/name errors; no diagnosis | Traceable preparation and local ASR verification |
| Batch aborts at existing WAV; manifest written only at end | Expensive work lost after interruption | Content fingerprints, verified reuse and new revisions |
| No revision/reference checksums/software provenance | Cannot reproduce takes | Complete per-candidate metadata and file hashes |
| Default BatchLine is Ultimate; voice docs contradict natural-dialogue routing | Unintended reference performance continuation | Controllable default, deliberate exact-performance opt-in |
| Voice transcript path containment incomplete; short references widespread | Invalid reference inputs; identity/acting instability | Path validation, reference gates and explicit review warnings |
| UI exposes only first candidate, no approval/delivery workflow | Raw mistaken for final | A/B review, immutable approval and separate delivery |
| Three API tests; benchmark is performance-only | Regressions undetected | Pure logic tests, fixtures, fixed 10/20/30 comparison and listening rubric |

## Priority

P0 (in order): privacy paths; model lock; generation profiles; adaptive generation;
normalization; pronunciation; ASR transcript QC; enhanced audio QC; batch resume; reproducibility.

P1: ranking; human review; approved/delivery; optional speaker verification;
long-form segmentation; expanded tests and regression benchmark.

P2: visual polish and optional future quantization/LoRA experiments, deferred until quality evidence.

## Explicit exclusions and acceptance limits

Keep BF16, mlx-audio 0.5.1, one serialized model and existing generation modes. No cloud,
database, distributed inference, dependency upgrade, reference replacement or raw overwrite.
Do not lower production steps/candidate defaults without measured and listened A/B evidence.
Unconfigured ASR/speaker analysis is unavailable, never a passing measurement. ASR thresholds
need a reviewed calibration set; acoustic heuristics cannot prove emotion or speaker identity.
Human approval and loudness targets require listening. Follow the active no-git-operations hook.

## Implemented and verified outcome

P0 and the code-supported P1 workflow were implemented without changing BF16, mlx-audio 0.5.1,
canonical audio, or the production defaults before benchmark approval. This includes model artifact
locking, profiles and bounded adaptive attempts, conservative text/pronunciation preparation,
pinned offline ASR and exact edit alignment, PASS/WARN/FAIL audio QC, content-addressed batch
revisions/resume, complete generation metadata, candidate recommendation, explicit listening
approval, separate delivery, optional speaker adapter, semantic long-form segmentation, tests,
fixed benchmarks, local event logs and user/architecture/quality/voice documentation.

Final automated evidence: Ruff and formatting pass; 71 tests pass; system/model/reference
integrity check passes with 51 short-reference warnings; episode preparation and batch dry-run
pass. A real 30-step BF16 smoke produced 48kHz mono PCM-24 audio with objective QC PASS and local
ASR exact-match comparison. It is byte-identical to the repository's prior same-input/seed/parameter
baseline; current generation was 25.04s for 5.76s audio, with 6.24GB reported MLX peak memory.

The full real-model A/B generated 30 immutable WAVs (10 text categories × 10/20/30 steps); all
30 pass objective audio QC. Mean advisory CER was 1.65% / 2.70% / 1.47%, respectively. Errors
concentrate in English mixed speech, names and one long sentence. These figures do not establish
naturalness, emotion or identity on their own. The user subsequently passed all 30 takes by ear
without an obvious difference. Production now uses bounded 10→20→30 adaptive generation: audio
QC PASS and CER ≤0.12 stop after the current take; failures escalate. This accepts every approved
10-step sample while conservatively retrying the 20-step mixed-language CER 0.2105 outlier.
Human listening is required once for the recommended master, rather than routine three-way selection.
